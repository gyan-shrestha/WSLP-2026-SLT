#!/usr/bin/env python3
"""Encode sampled video frames with a modern vision backbone.

Why bother, given the iSign baseline found video *worse* than pose?
Because that baseline used I3D (2017). SigLIP and DINOv2 are trained on
billions of images and encode handshape, facial expression, and scene context
that 103 MediaPipe keypoints discard entirely. That representation gap is the
largest untried lever available, and it is exactly the kind of thing two years
of progress delivers.

Frames are sampled uniformly across the clip rather than at a fixed rate, so a
2-second and a 16-second clip both yield `--frames` embeddings and the model
sees a consistent temporal budget.

    python scripts/extract_video_feats.py --source test --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/blue/YOUR_ACCOUNT") / os.environ.get("USER", "") / "wslp"

SOURCES = {
    "test": ROOT / "task/Shared_task_MT/Test-dataset/test_mp4.zip",
    "val": ROOT / "task/Shared_task_MT/Validation-Dataset/val_mp4.zip",
}


def open_source(name: str):
    """Archive handle for a source. iSign video ships as a split zip."""
    if name == "isign":
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from data.multipart import open_split_zip
        return open_split_zip(ROOT / "isign_video", "iSign-videos_v1.1")
    return zipfile.ZipFile(SOURCES[name])


def sample_frames(raw: bytes, n_frames: int, size: int = 224) -> np.ndarray | None:
    """Decode an mp4 from memory and return `n_frames` uniformly-spaced RGB frames."""
    import av

    try:
        container = av.open(io.BytesIO(raw))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        container.close()
    except Exception:
        return None

    if not frames:
        return None

    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    picked = [frames[i] for i in idx]

    import cv2
    out = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
                    for f in picked])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES) + ["isign"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", type=Path, default=None)
    a = ap.parse_args()

    outdir = a.outdir or ROOT / "vfeats" / a.source
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).vision_model.to(device).eval()
    if device == "cuda":
        model = model.half()
    dim = model.config.hidden_size
    print(f"{a.model} -> {dim}-d | device {device}", flush=True)

    zf = open_source(a.source)
    members = sorted(m for m in zf.namelist() if m.endswith(".mp4"))
    mine = members[a.shard::a.num_shards]
    print(f"[{a.source} shard {a.shard}/{a.num_shards}] {len(mine)} clips", flush=True)

    bin_path = outdir / f"shard_{a.shard:03d}.f16"
    idx_path = outdir / f"shard_{a.shard:03d}.jsonl"
    offset, ok, fail = 0, 0, 0
    t0 = time.time()

    with open(bin_path, "wb") as fb, open(idx_path, "w") as fi:
        for i, member in enumerate(mine):
            uid = Path(member).stem
            frames = sample_frames(zf.read(member), a.frames)
            if frames is None:
                fail += 1
                continue

            with torch.no_grad():
                px = proc(images=list(frames), return_tensors="pt")["pixel_values"]
                px = px.to(device).half() if device == "cuda" else px
                # Mean-pool patch tokens per frame -> one embedding per frame.
                out = model(pixel_values=px).last_hidden_state.mean(dim=1)
            feats = out.float().cpu().numpy().astype(np.float16)   # (frames, dim)

            fb.write(feats.tobytes())
            fi.write(json.dumps({"uid": uid, "offset": offset,
                                 "frames": int(feats.shape[0]),
                                 "feat": int(feats.shape[1])}) + "\n")
            # Flush periodically: a cancelled job previously lost its entire
            # index to an unflushed buffer while the 1.1 GB of features it
            # described sat on disk, unusable.
            if ok % 200 == 0:
                fb.flush()
                fi.flush()
            offset += feats.nbytes
            ok += 1

            if (i + 1) % 200 == 0:
                r = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(mine)}  {r:.1f} clips/s  "
                      f"ETA {(len(mine) - i - 1) / r / 60:.1f} min", flush=True)

    print(f"[done] ok={ok} failed={fail} bytes={offset / 1e9:.2f} GB "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
