#!/usr/bin/env python3
"""Masked-pose pretraining on iSign + the task's unlabeled val/test poses.

Why this exists
---------------
Our translation model grounds well in-domain (dev BLEU 0.0119) and not at all
out-of-domain (test BLEU 0.00). iSign and the evaluation set share zero videos,
so the encoder has never seen the test signers, framing, or signing style.

The task ships 10,589 pose clips from the evaluation videos with no labels. We
used them only for inference. Training the encoder on them -- predicting masked
frames from surrounding context -- teaches it to represent the target domain
before any translation objective is applied. No labels are touched, so this is
ordinary unsupervised domain adaptation, not leakage.

    python src/train/pretrain.py --config configs/pretrain.json

Produces `frontend.pt`, loaded by train.py via cfg["init_frontend"].
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import SLTDataset  # noqa: E402
from models.pose_t5 import PoseFrontend  # noqa: E402


class MaskedPoseModel(nn.Module):
    """Frontend + transformer encoder + linear head back to pose space.

    The frontend is the piece we keep; the encoder and head are scaffolding
    that make the pretraining task solvable and are discarded afterwards.
    """

    def __init__(self, n_feat: int, d_model: int = 768, layers: int = 4,
                 heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.frontend = PoseFrontend(n_feat, d_model, dropout, target_norm=None)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, n_feat * 4)   # 4 frames per output step
        self.mask_token = nn.Parameter(torch.randn(n_feat) * 0.02)

    def forward(self, feats, mask, mask_frames):
        x = feats.clone()
        x[mask_frames] = self.mask_token.to(x.dtype)
        h, m = self.frontend(x, mask)
        h = self.encoder(h, src_key_padding_mask=~m.bool())
        return self.head(h)


def span_mask(B, T, device, ratio=0.15, span=8):
    """Mask contiguous spans, not isolated frames.

    Neighbouring pose frames are nearly identical at 25 fps, so masking single
    frames is trivially solved by copying the neighbour and teaches nothing.
    Spans force the model to infer motion over a real interval.
    """
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    n_spans = max(1, int(T * ratio / span))
    for b in range(B):
        for _ in range(n_spans):
            s = random.randint(0, max(0, T - span))
            m[b, s:s + span] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(cfg["feat_root"])
    parts = []
    for src in cfg["sources"]:
        ds = SLTDataset(root / src, None, max_frames=cfg["max_frames"])
        parts.append(ds)
        print(f"  {src}: {len(ds)} clips", flush=True)
    data = ConcatDataset(parts)
    n_feat = parts[0].n_feat
    print(f"total {len(data)} clips | feat dim {n_feat} | device {device}", flush=True)

    def collate(batch):
        lens = [b["feats"].shape[0] for b in batch]
        T = max(lens)
        f = torch.zeros(len(batch), T, n_feat)
        m = torch.zeros(len(batch), T, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["feats"].shape[0]
            f[i, :n] = b["feats"]
            m[i, :n] = 1
        return f, m

    dl = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], pin_memory=True,
                    collate_fn=collate, drop_last=True,
                    persistent_workers=cfg["num_workers"] > 0)

    model = MaskedPoseModel(n_feat, cfg["d_model"], cfg["layers"],
                            cfg["heads"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total = len(dl) * cfg["epochs"]
    warm = int(total * 0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None

    print(f"steps/epoch {len(dl)} | total {total}", flush=True)
    step, t0 = 0, time.time()

    for epoch in range(cfg["epochs"]):
        run, n = 0.0, 0
        for feats, mask in dl:
            feats, mask = feats.to(device, non_blocking=True), mask.to(device)
            mf = span_mask(feats.shape[0], feats.shape[1], device,
                           cfg["mask_ratio"], cfg["mask_span"])

            def compute():
                pred = model(feats, mask, mf)                  # (B, T/4, F*4)
                B, S, _ = pred.shape
                pred = pred.reshape(B, S * 4, n_feat)[:, : feats.shape[1]]
                sel = mf & mask.bool()
                if not sel.any():
                    return None
                # Reconstruct only the masked, non-padded frames.
                return nn.functional.smooth_l1_loss(pred[sel], feats[sel])

            if amp:
                with amp:
                    loss = compute()
            else:
                loss = compute()
            if loss is None:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            n += 1
            step += 1

            if step % cfg["log_every"] == 0:
                print(f"  step {step}/{total} recon {run / n:.5f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / 60:.1f}m", flush=True)
                run, n = 0.0, 0

        torch.save({"frontend": model.frontend.state_dict(), "cfg": cfg,
                    "epoch": epoch, "n_feat": n_feat}, out_dir / "frontend.pt")
        print(f"[epoch {epoch}] saved frontend.pt", flush=True)

    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
