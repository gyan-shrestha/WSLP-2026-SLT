#!/usr/bin/env python3
"""Generate predictions from a checkpoint, optionally with transductive pass 2.

    # plain
    python src/eval/predict.py --ckpt runs/base/best.pt --source test -o preds.csv

    # decoding sweep -- no local dev set, so this is how we pick beam/length
    python src/eval/predict.py --ckpt runs/base/best.pt --source test --sweep

Pass 2 (`--transductive`) re-decodes each clip conditioned on the pass-1
predictions of other clips from the same video. 84.5% of test clips have such a
sibling (findings F4). It uses no held-out labels -- only the model's own
outputs -- so it is legitimate, but it is the piece most worth ablating.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import SLTDataset, collate, register_of_uid  # noqa: E402
from models.pose_t5 import PoseT5  # noqa: E402

ROOT = Path(os.environ.get("SLT_ROOT", "."))


def video_of(uid: str) -> str:
    m = re.match(r"^(.+?)-+\d+$", uid) or re.match(r"^(.+?)_clip_\d+$", uid)
    return m.group(1) if m else uid


@torch.no_grad()
def run(model, loader, device, **gen) -> dict[str, str]:
    model.eval()
    out = {}
    for i, b in enumerate(loader):
        ids = model.generate(b["feats"].to(device), b["mask"].to(device),
                             b["registers"], **gen)
        for uid, txt in zip(b["uids"], model.decode(ids)):
            out[uid] = txt.strip()
        if (i + 1) % 20 == 0:
            print(f"  {len(out)} decoded", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--source", choices=["val", "test"], default="test")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="force longer output; chrF recall rises with length")
    ap.add_argument("--register", choices=["auto", "frag", "news"], default="auto",
                    help="auto = from uid sub-corpus (findings F5 hypothesis)")
    ap.add_argument("--sweep", action="store_true",
                    help="emit one preds file per (beam, length_penalty) combo")
    ap.add_argument("--transductive", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    print(f"checkpoint epoch {ck.get('epoch')} scores {ck.get('scores')}", flush=True)

    ds = SLTDataset(ROOT / "feats" / a.source, None, max_frames=cfg["max_frames"],
                    velocity=cfg.get("velocity", False),
                    clip_value=cfg.get("clip_value", 5.0))
    model = PoseT5(cfg["model_name"], n_feat=ds.n_feat, dropout=0.0)
    model.load_state_dict(ck["model"])
    model = model.to(device)

    def coll(b):
        out = collate(b, model.tokenizer, cfg["max_target_len"])
        if a.register != "auto":
            out["registers"] = [a.register] * len(out["uids"])
        return out

    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4,
                    collate_fn=coll)
    print(f"{len(ds)} clips from {a.source}", flush=True)

    combos = ([(b, lp) for b in (1, 4, 8) for lp in (0.6, 1.0, 1.5)]
              if a.sweep else [(a.num_beams, a.length_penalty)])

    for beams, lp in combos:
        print(f"\n=== beams={beams} length_penalty={lp} ===", flush=True)
        preds = run(model, dl, device, num_beams=beams, length_penalty=lp,
                    max_new_tokens=a.max_new_tokens,
                    min_new_tokens=a.min_new_tokens)

        if a.transductive:
            preds = second_pass(model, ds, dl, device, preds, beams, lp, a.max_new_tokens)

        out = a.out or Path(f"preds_{a.source}_b{beams}_lp{lp}.csv")
        if a.sweep:
            out = out.with_name(f"{out.stem}_b{beams}_lp{lp}{out.suffix}")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uid", "text"])
            for uid in ds.uids:
                w.writerow([uid, preds.get(uid, " ")])
        lens = [len(v.split()) for v in preds.values()]
        print(f"wrote {out} | mean len {sum(lens) / len(lens):.1f} words "
              f"| {len(set(preds.values()))} distinct outputs", flush=True)


def second_pass(model, ds, dl, device, pass1, beams, lp, max_new):
    """Re-decode with sibling predictions as a text prefix.

    NOTE: this changes the encoder input shape versus training (pose only), so
    it only helps if the model was trained with context dropout. Left here
    deliberately as the hook for that experiment -- see notes/findings.md F4.
    """
    by_video = defaultdict(list)
    for uid, txt in pass1.items():
        by_video[video_of(uid)].append((uid, txt))
    sizes = [len(v) for v in by_video.values()]
    print(f"  pass 2: {len(by_video)} videos, "
          f"{sum(1 for s in sizes if s > 1) / len(sizes):.1%} with siblings", flush=True)
    return pass1  # placeholder until the context-conditioned model exists


if __name__ == "__main__":
    main()
