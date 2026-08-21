#!/usr/bin/env python3
"""Decode `.pose` archives into packed float16 shards.

Output per shard is two files, not 127k small ones -- Lustre punishes many
small files badly, and the training dataloader wants memory-mapped random
access anyway:

    shard_000.f16     concatenated (frames, 206) float16, all clips
    shard_000.jsonl   one row per clip: uid, offset, n_frames, n_feat

Shards are independent, so this runs as a SLURM array on the CPU burst QOS
(288 cores, 0 GPUs) and leaves our single GPU alone.

    python scripts/preprocess.py --source isign --shard 0 --num-shards 64
    python scripts/preprocess.py --source test  --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.multipart import open_split_zip  # noqa: E402
from data.pose import PoseLayout, decode, resample  # noqa: E402

ROOT = Path(os.environ.get("SLT_ROOT", "."))

SOURCES = {
    "isign": dict(kind="split", path=ROOT / "isign", stem="iSign-poses_v1.1"),
    "val": dict(kind="zip", path=ROOT / "task/Shared_task_MT/Validation-Dataset/val_pose.zip"),
    "test": dict(kind="zip", path=ROOT / "task/Shared_task_MT/Test-dataset/test_pose.zip"),
}


def open_archive(src: str):
    cfg = SOURCES[src]
    if cfg["kind"] == "split":
        return open_split_zip(cfg["path"], cfg["stem"])
    return zipfile.ZipFile(cfg["path"])


def uid_of(member: str) -> str | None:
    """Archive member path -> uid. Skips directories and non-pose entries."""
    p = Path(member)
    return p.stem if p.suffix == ".pose" else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=512)
    ap.add_argument("--scheme", choices=["signspace", "legacy"], default="signspace",
                    help="signspace = arXiv:2507.01532 (3x BLEU-4 cross-dataset)")
    ap.add_argument("--stride", type=int, default=1,
                    help="temporal stride before the frame budget; 2 halves 25fps to 12.5")
    a = ap.parse_args()

    outdir = a.outdir or ROOT / "feats" / a.source
    outdir.mkdir(parents=True, exist_ok=True)

    z = open_archive(a.source)
    members = [m for m in z.namelist() if uid_of(m)]
    members.sort()                                  # deterministic sharding
    mine = members[a.shard::a.num_shards]
    print(f"[{a.source} shard {a.shard}/{a.num_shards}] {len(mine)} of {len(members)} clips",
          flush=True)

    bin_path = outdir / f"shard_{a.shard:03d}.f16"
    idx_path = outdir / f"shard_{a.shard:03d}.jsonl"

    layout: PoseLayout | None = None
    offset, n_ok, n_fail = 0, 0, 0
    t0 = time.time()

    with open(bin_path, "wb") as fbin, open(idx_path, "w") as fidx:
        for i, member in enumerate(mine):
            uid = uid_of(member)
            try:
                feats, layout = decode(z.read(member), layout, a.scheme)
                feats = resample(feats, a.max_frames, a.stride)
                if feats.shape[0] == 0:
                    raise ValueError("zero frames")
            except Exception as e:                   # noqa: BLE001
                # A handful of corrupt clips must not kill a 2-hour shard.
                n_fail += 1
                print(f"  SKIP {uid}: {type(e).__name__}: {e}", flush=True)
                continue

            fbin.write(feats.tobytes())
            fidx.write(json.dumps({
                "uid": uid,
                "offset": offset,
                "frames": int(feats.shape[0]),
                "feat": int(feats.shape[1]),
            }) + "\n")
            offset += feats.nbytes
            n_ok += 1

            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(mine) - i - 1) / rate / 60
                print(f"  {i + 1}/{len(mine)}  {rate:.0f} clips/s  ETA {eta:.1f} min",
                      flush=True)

    print(f"[done] ok={n_ok} failed={n_fail} bytes={offset / 1e9:.2f} GB "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    if layout:
        (outdir / "layout.json").write_text(json.dumps({
            "n_points": layout.n_points,
            "n_feat": layout.n_points * 2,
            "keep": layout.keep,
            "shoulder_l": layout.shoulder_l,
            "shoulder_r": layout.shoulder_r,
        }, indent=2))


if __name__ == "__main__":
    main()
