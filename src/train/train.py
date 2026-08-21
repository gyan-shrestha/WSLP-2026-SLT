#!/usr/bin/env python3
"""Train PoseT5 on iSign.

There is no real dev set (findings F3), so we carve one out of iSign by *video*
-- never by clip. A clip-level split would put adjacent clips of the same video
on both sides and report a number inflated by topic memorisation, which is
exactly the trap the organizers avoided when they built the real split.

    python src/train/train.py --config configs/base.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import SLTDataset, collate, load_text  # noqa: E402
from eval.metrics import score  # noqa: E402
from models.pose_t5 import PoseT5  # noqa: E402


def video_of(uid: str) -> str:
    m = re.match(r"^(.+?)-+\d+$", uid)
    return m.group(1) if m else uid


def split_by_clip(uids: list[str], dev_frac: float, seed: int) -> tuple[list, list]:
    """Random clip-level split -- deliberately LEAKY, for measurement only.

    Adjacent clips of one video land on both sides, so the dev set shares
    signer, topic and narrative with training. Used solely to quantify how much
    split methodology inflates reported SLT scores; never for model selection.
    """
    us = sorted(uids)
    rng = random.Random(seed)
    rng.shuffle(us)
    n_dev = max(1, int(len(us) * dev_frac))
    return us[n_dev:], us[:n_dev]


def split_by_video(uids: list[str], dev_frac: float, seed: int) -> tuple[list, list]:
    vids = sorted({video_of(u) for u in uids})
    rng = random.Random(seed)
    rng.shuffle(vids)
    n_dev = max(1, int(len(vids) * dev_frac))
    dev = set(vids[:n_dev])
    return ([u for u in uids if video_of(u) not in dev],
            [u for u in uids if video_of(u) in dev])


def evaluate(model, loader, device, max_batches: int | None = None) -> tuple:
    model.eval()
    hyps, refs, losses = [], [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            feats, mask = b["feats"].to(device), b["mask"].to(device)
            if "labels" in b:
                out = model(feats, mask, b["registers"], labels=b["labels"].to(device))
                losses.append(out.loss.item())
            ids = model.generate(feats, mask, b["registers"])
            hyps.extend(model.decode(ids))
            refs.extend(b["texts"])
    model.train()
    return score(hyps, refs), (sum(losses) / len(losses) if losses else float("nan")), hyps, refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    out_dir = a.out or Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else ''})", flush=True)

    # --- data ------------------------------------------------------------
    text = load_text(Path(cfg["text_csv"]), drop_junk=True)
    full = SLTDataset(Path(cfg["feat_dir"]), text, max_frames=cfg["max_frames"])
    splitter = split_by_clip if cfg.get("split_by_clip") else split_by_video
    tr_uids, dev_uids = splitter(full.uids, cfg["dev_frac"], cfg["seed"])
    print(f"clips: {len(full.uids)} -> train {len(tr_uids)} / dev {len(dev_uids)} "
          f"(split by {'CLIP (leaky)' if cfg.get('split_by_clip') else 'video'})", flush=True)

    vel = cfg.get("velocity", False)
    aug = cfg.get("augment_prob", 0.0)
    clipv = cfg.get("clip_value", 5.0)
    train_ds = SLTDataset(Path(cfg["feat_dir"]), text, cfg["max_frames"], uids=tr_uids,
                          augment_prob=aug, velocity=vel, clip_value=clipv)
    # dev is never augmented -- otherwise the metric measures the noise, not the model
    dev_ds = SLTDataset(Path(cfg["feat_dir"]), text, cfg["max_frames"], uids=dev_uids,
                        augment_prob=0.0, velocity=vel, clip_value=clipv)
    print(f"velocity={vel} (feat dim {train_ds.n_feat}) | augment_prob={aug}", flush=True)

    model = PoseT5(cfg["model_name"], n_feat=train_ds.n_feat, dropout=cfg["dropout"],
                   downsample=cfg.get("downsample", True))
    if cfg.get("init_frontend"):
        # Warm-start from masked-pose pretraining over iSign + the task's
        # unlabeled val/test poses, so the encoder has already seen the
        # evaluation domain before translation training begins.
        ck = torch.load(cfg["init_frontend"], map_location="cpu", weights_only=False)
        sd = dict(ck["frontend"])
        # Keep the freshly-initialised emb_scale. Pretraining has no T5 decoder,
        # so it learns a scale near 1.0, but T5 does not scale its embeddings by
        # sqrt(d_model) and needs ~10.7 here. Loading the pretrained value
        # silently reintroduced the mismatch that once started training at loss
        # 609, and the backbone LR was too low to climb back: the warm-started
        # run ended at emb_scale 0.94 and halved BLEU (0.0142 -> 0.0071).
        keep = float(model.frontend.emb_scale.detach())
        sd.pop("emb_scale", None)
        missing, unexpected = model.frontend.load_state_dict(sd, strict=False)
        print(f"warm-started frontend from {cfg['init_frontend']} "
              f"(missing={len(missing)} unexpected={len(unexpected)}; "
              f"emb_scale kept at {keep:.3f})", flush=True)
    model = model.to(device)
    tok = model.tokenizer

    def mk(ds, shuffle, bs):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=cfg["num_workers"], pin_memory=True,
                          collate_fn=lambda b: collate(b, tok, cfg["max_target_len"]),
                          drop_last=shuffle, persistent_workers=cfg["num_workers"] > 0)

    train_dl = mk(train_ds, True, cfg["batch_size"])
    dev_dl = mk(dev_ds, False, cfg["eval_batch_size"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.1f}M | feat dim {full.n_feat}", flush=True)

    # --- optim -----------------------------------------------------------
    # Two LRs, not one. The pose frontend is randomly initialised and needs a
    # high rate; T5 is pretrained and destabilises above ~5e-5 under AdamW.
    # A single 1e-4 served neither and produced a loss spike (5.8 -> 15.1) as
    # the schedule reached peak.
    frontend = [p for n, p in model.named_parameters()
                if n.startswith("frontend") and p.requires_grad]
    backbone = [p for n, p in model.named_parameters()
                if not n.startswith("frontend") and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": frontend, "lr": cfg["lr_frontend"]},
         {"params": backbone, "lr": cfg["lr"]}],
        weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    print(f"param groups: frontend {sum(p.numel() for p in frontend) / 1e6:.1f}M "
          f"@ {cfg['lr_frontend']:.0e} | backbone "
          f"{sum(p.numel() for p in backbone) / 1e6:.1f}M @ {cfg['lr']:.0e}", flush=True)
    steps_per_epoch = len(train_dl) // cfg["grad_accum"]
    total = steps_per_epoch * cfg["epochs"]
    warmup = int(total * cfg["warmup_frac"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    # bf16 on B200 (sm_100): same range as fp32, so no GradScaler needed.
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None

    print(f"steps/epoch {steps_per_epoch} | total {total} | warmup {warmup}", flush=True)

    best, step, t0 = -1.0, 0, time.time()
    log_path = out_dir / "log.jsonl"

    for epoch in range(cfg["epochs"]):
        run_loss, n = 0.0, 0
        run_ct, n_ct = 0.0, 0
        for i, b in enumerate(train_dl):
            feats, mask = b["feats"].to(device, non_blocking=True), b["mask"].to(device)
            labels = b["labels"].to(device)

            def compute():
                ce = model(feats, mask, b["registers"], labels=labels).loss
                w = cfg.get("contrastive_weight", 0.0)
                if w <= 0:
                    return ce, ce, None
                # Needs >1 example to have negatives.
                if feats.shape[0] < 2:
                    return ce, ce, None
                ct = model.contrastive_loss(feats, mask, b["registers"], labels,
                                            cfg.get("contrastive_temp", 0.07))
                return ce + w * ct, ce, ct

            if amp:
                with amp:
                    loss, ce_loss, ct_loss = compute()
            else:
                loss, ce_loss, ct_loss = compute()

            (loss / cfg["grad_accum"]).backward()
            run_loss += ce_loss.item()
            if ct_loss is not None:
                run_ct += ct_loss.item()
                n_ct += 1
            n += 1

            if (i + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % cfg["log_every"] == 0:
                    msg = {"epoch": epoch, "step": step, "loss": run_loss / n,
                           "lr": sched.get_last_lr()[0], "mins": (time.time() - t0) / 60}
                    if n_ct:
                        msg["ct"] = run_ct / n_ct
                    ct_s = f" ct {msg['ct']:.3f}" if n_ct else ""
                    print(f"  step {step}/{total} loss {msg['loss']:.4f}{ct_s} "
                          f"lr {msg['lr']:.2e} {msg['mins']:.1f}m", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(msg) + "\n")
                    run_loss, n = 0.0, 0
                    run_ct, n_ct = 0.0, 0

        # Checkpoint BEFORE evaluating. An eval crash previously destroyed a
        # whole run's weights; training progress must never depend on metrics.
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                   out_dir / "last.pt")

        try:
            sc, dev_loss, hyps, refs = evaluate(model, dev_dl, device, cfg["eval_batches"])
        except Exception as e:  # noqa: BLE001
            print(f"[epoch {epoch}] EVAL FAILED ({type(e).__name__}: {e}) "
                  f"-- training continues, checkpoint is saved", flush=True)
            continue

        print(f"[epoch {epoch}] dev_loss {dev_loss:.4f} | {sc}", flush=True)
        for h, r in list(zip(hyps, refs))[:3]:
            print(f"    ref: {r}\n    hyp: {h}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "dev_loss": dev_loss, **sc.as_dict()}) + "\n")

        if sc.chrf > best:
            best = sc.chrf
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "scores": sc.as_dict()}, out_dir / "best.pt")
            print(f"    new best chrF {best:.4f}", flush=True)

    print(f"done in {(time.time() - t0) / 60:.1f} min | best dev chrF {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
