#!/usr/bin/env python3
"""Retrieval-based translation: nearest training sentence by pose similarity.

The contrastive objective trained the pose tower to sit near its matching
sentence in T5's encoder space. We used that only as a training signal, but it
is also a retrieval index: embed every training clip, embed a query clip, take
the nearest neighbour's *human-written* sentence.

Why this can beat generation on this task. At low visual signal a decoder
converges on fluent, generic text -- our submitted system emitted one sentence
for 9% of the test set. A retrieved sentence is real English with natural
length, specific content words, and correct register, which is what chrF and
ROUGE reward. And when a neighbour happens to be roughly right, it yields exact
multi-word matches, which is the only thing unsmoothed BLEU counts.

    python src/eval/retrieve.py --ckpt runs/norm/best.pt --source dev
    python src/eval/retrieve.py --ckpt runs/norm/best.pt --source test -o preds/knn.csv
"""

from __future__ import annotations

import argparse
import os
import csv
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import SLTDataset, collate, load_text  # noqa: E402
from eval.metrics import score  # noqa: E402
from models.pose_t5 import PoseT5  # noqa: E402

ROOT = Path(os.environ.get("SLT_ROOT", "."))


def video_of(uid: str) -> str:
    m = re.match(r"^(.+?)-+\d+$", uid)
    return m.group(1) if m else uid


def _masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


@torch.no_grad()
def embed_poses(model, ds, device, batch_size=64):
    """Mean-pooled encoder states for every clip, L2-normalised."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=6,
                    collate_fn=lambda b: collate(b, model.tokenizer, 48))
    vecs, uids = [], []
    for i, b in enumerate(dl):
        e, a = model._embed(b["feats"].to(device), b["mask"].to(device), b["registers"])
        h = model.t5.encoder(inputs_embeds=e, attention_mask=a).last_hidden_state
        vecs.append(F.normalize(_masked_mean(h, a), dim=-1).cpu())
        uids.extend(b["uids"])
        if (i + 1) % 50 == 0:
            print(f"    embedded {len(uids)}", flush=True)
    return torch.cat(vecs), uids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--source", choices=["dev", "test", "val"], default="dev")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=800, help="dev clips to score")
    ap.add_argument("--topk", type=int, default=1)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]

    text = load_text(Path(cfg["text_csv"]), drop_junk=True)
    full = SLTDataset(Path(cfg["feat_dir"]), text, max_frames=cfg["max_frames"])

    model = PoseT5(cfg["model_name"], n_feat=full.n_feat, dropout=0.0)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    # Split by video exactly as training did, so the index never contains a clip
    # from a query clip's own video.
    vids = sorted({video_of(u) for u in full.uids})
    rng = random.Random(cfg["seed"])
    rng.shuffle(vids)
    dev_v = set(vids[: max(1, int(len(vids) * cfg["dev_frac"]))])
    train_uids = [u for u in full.uids if video_of(u) not in dev_v]

    print(f"building index over {len(train_uids)} training clips", flush=True)
    index_ds = SLTDataset(Path(cfg["feat_dir"]), text, cfg["max_frames"], uids=train_uids)
    bank, bank_uids = embed_poses(model, index_ds, device)
    bank = bank.to(device)
    bank_text = [text[u] for u in bank_uids]
    print(f"index: {bank.shape}", flush=True)

    if a.source == "dev":
        q_uids = [u for u in full.uids if video_of(u) in dev_v][: a.limit]
        q_ds = SLTDataset(Path(cfg["feat_dir"]), text, cfg["max_frames"], uids=q_uids)
        refs = [text[u] for u in q_ds.uids]
    else:
        q_ds = SLTDataset(ROOT / "feats" / a.source, None, max_frames=cfg["max_frames"])
        refs = None

    print(f"embedding {len(q_ds)} queries", flush=True)
    q, q_uid = embed_poses(model, q_ds, device)
    q = q.to(device)

    # Cosine similarity; both sides are already L2-normalised.
    preds = {}
    for s in range(0, len(q), 256):
        sim = q[s:s + 256] @ bank.t()
        top = sim.topk(a.topk, dim=-1).indices
        for r, row in enumerate(top):
            preds[q_uid[s + r]] = bank_text[row[0].item()]

    if refs is not None:
        hyps = [preds[u] for u in q_ds.uids]
        print(f"\nRETRIEVAL (top-1): {score(hyps, refs)}", flush=True)
        for h, r in list(zip(hyps, refs))[:5]:
            print(f"    ref: {r}\n    ret: {h}", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uid", "text"])
            for uid in q_ds.uids:
                w.writerow([uid, preds.get(uid, " ")])
        print(f"wrote {a.out} | {len(set(preds.values()))} distinct "
              f"({len(set(preds.values())) / len(preds):.1%})", flush=True)


if __name__ == "__main__":
    main()
