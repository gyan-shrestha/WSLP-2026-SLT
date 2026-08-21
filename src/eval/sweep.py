#!/usr/bin/env python3
"""Tune decoding against chrF on the local dev split.

Costs zero Codabench submissions, which is the whole point -- our metric
implementation is validated against the scorer (predicted 0.0317 for constant
"hello", scored 0.03), so local chrF is a faithful measurement of the same
quantity. The domain gap to the real test set is a separate problem and is not
something more uploads would fix.

chrF is a character n-gram F-score, so it is sensitive to output length in a way
BLEU is not. Length penalty is therefore the highest-value knob here and it
costs no training.

    python src/eval/sweep.py --ckpt runs/contrastive/best.pt
"""

from __future__ import annotations

import argparse
import os
import csv
import itertools
import json
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import SLTDataset, collate, load_text  # noqa: E402
from eval.metrics import score  # noqa: E402
from models.pose_t5 import PoseT5  # noqa: E402

ROOT = Path(os.environ.get("SLT_ROOT", "."))


def video_of(uid: str) -> str:
    m = re.match(r"^(.+?)-+\d+$", uid)
    return m.group(1) if m else uid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=600, help="dev clips to score per combo")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--beams", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--length-penalty", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--no-repeat", type=int, nargs="+", default=[3])
    ap.add_argument("--min-new-tokens", type=int, nargs="+", default=[0, 8, 12])
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]

    # Same video-level dev split as training, so this never sees train clips.
    text = load_text(Path(cfg["text_csv"]), drop_junk=True)
    full = SLTDataset(Path(cfg["feat_dir"]), text, max_frames=cfg["max_frames"])
    vids = sorted({video_of(u) for u in full.uids})
    rng = random.Random(cfg["seed"])
    rng.shuffle(vids)
    dev_v = set(vids[:max(1, int(len(vids) * cfg["dev_frac"]))])
    dev_uids = [u for u in full.uids if video_of(u) in dev_v][: a.n]

    ds = SLTDataset(Path(cfg["feat_dir"]), text, cfg["max_frames"], uids=dev_uids)
    model = PoseT5(cfg["model_name"], n_feat=ds.n_feat, dropout=0.0)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4,
                    collate_fn=lambda b: collate(b, model.tokenizer, cfg["max_target_len"]))
    print(f"sweeping on {len(ds)} dev clips (checkpoint epoch {ck.get('epoch')})\n", flush=True)

    # The first sweep peaked at the grid's upper corner (beams=8, lp=2.0), so
    # the optimum was outside it. chrF's recall term rewards length at our
    # precision level, hence min_new_tokens as an explicit length floor.
    grid = list(itertools.product(
        a.beams,
        a.length_penalty,
        a.no_repeat,
        a.min_new_tokens,
    ))
    print(f"{len(grid)} combinations\n", flush=True)

    results = []
    for beams, lp, nrep, minlen in grid:
        hyps, refs = [], []
        with torch.no_grad():
            for b in dl:
                ids = model.generate(b["feats"].to(device), b["mask"].to(device),
                                     b["registers"], num_beams=beams, length_penalty=lp,
                                     no_repeat_ngram_size=nrep, min_new_tokens=minlen,
                                     max_new_tokens=64)
                hyps.extend(model.decode(ids))
                refs.extend(b["texts"])
        sc = score(hyps, refs)
        mean_len = sum(len(h.split()) for h in hyps) / len(hyps)
        distinct = len(set(hyps)) / len(hyps)
        results.append({"beams": beams, "length_penalty": lp, "no_repeat": nrep,
                        "min_new_tokens": minlen, "mean_len": mean_len,
                        "distinct": distinct, **sc.as_dict()})
        print(f"beams={beams} lp={lp} nrep={nrep} minlen={minlen} -> "
              f"chrF {sc.chrf:.4f} BLEU {sc.bleu:.4f} | len {mean_len:.1f} "
              f"| distinct {distinct:.1%}", flush=True)

    # Rank on all three task metrics, not chrF alone.
    #
    # Measured on 3,000 iSign references, every degenerate system scores
    # BLEU exactly 0.0000 -- constant sentence, random real sentence, "hello" --
    # while a constant string scores chrF 0.155 and ROUGE 0.076. So BLEU is the
    # metric that certifies real translation; chrF and ROUGE can be had for free
    # with fluent filler.
    #
    # Ranking by chrF alone selected configs with BLEU 0.0000, i.e.
    # indistinguishable from a constant string on the only unfakeable metric.
    #
    # We score GAIN OVER THE DEGENERATE FLOOR, not over a competitor. Anchoring
    # to another team's number is unstable -- one early submission out of ~20
    # participants says nothing about the final bar. The floor is a property of
    # the data and metric, so it does not move.
    FLOOR = {"chrf": 0.155, "rouge_l": 0.076, "bleu": 0.0}   # best degenerate system
    SCALE = {"chrf": 0.155, "rouge_l": 0.076, "bleu": 0.01}  # per-metric unit

    def composite(r):
        return sum((r[k] - FLOOR[k]) / SCALE[k] for k in FLOOR)

    for r in results:
        r["composite"] = composite(r)

    def show(title, key):
        print(f"\n=== best by {title} ===")
        for r in sorted(results, key=lambda x: -x[key])[:5]:
            flag = "  <-- BLEU 0, degenerate" if r["bleu"] == 0 else ""
            print(f"  chrF {r['chrf']:.4f}  ROUGE {r['rouge_l']:.4f}  BLEU {r['bleu']:.4f}"
                  f"  | beams={r['beams']} lp={r['length_penalty']} "
                  f"nrep={r['no_repeat']} minlen={r['min_new_tokens']} "
                  f"len={r['mean_len']:.1f}{flag}")

    show("COMPOSITE (gain over degenerate floor, all three)", "composite")
    show("chrF alone", "chrf")
    show("BLEU alone", "bleu")

    best = max(results, key=composite)
    print(f"\nRECOMMENDED: beams={best['beams']} lp={best['length_penalty']} "
          f"nrep={best['no_repeat']} minlen={best['min_new_tokens']}")
    print(f"  chrF {best['chrf']:.4f} (floor 0.155) | "
          f"ROUGE {best['rouge_l']:.4f} (floor 0.076) | "
          f"BLEU {best['bleu']:.4f} (floor 0.000)")

    out = a.out or ROOT / "runs" / "sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
