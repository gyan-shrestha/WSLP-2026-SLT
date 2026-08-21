"""Dataset over the packed float16 shards produced by scripts/preprocess.py.

Shards are memory-mapped, so the 6 GB of features are never loaded into RAM and
workers share the page cache. Text comes from `iSign_v1.1.csv`.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SENT_UID = re.compile(r"^.+-\d+$")


def register_of_text(text: str) -> str:
    """Self-supervised register label from the target's own surface form.

    iSign carries both registers (findings F5 correction), so the control token
    can be trained directly from the data instead of from synthetic examples.
    """
    return "news" if text[:1].isupper() and text[-1:] in ".!?" else "frag"


def register_of_uid(uid: str) -> str:
    """Register to request at inference, from the eval sub-corpus.

    HYPOTHESIS, not a measured fact -- see findings F5. The
    generic/generic-routed submission pair tests it.
    """
    return "news" if "_clip_" in uid else "frag"


def is_junk(text: str) -> bool:
    """Single tokens and bare digits/punctuation (findings F8, 2.7% of rows)."""
    t = text.strip()
    return len(t.split()) <= 1 or bool(re.fullmatch(r"[\W\d]+", t))


def load_text(csv_path: Path, drop_junk: bool = True) -> dict[str, str]:
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid, text = r["uid"].strip(), r["text"].strip()
            if not text or not SENT_UID.match(uid):
                continue
            if drop_junk and is_junk(text):
                continue
            out[uid] = text
    return out


class ShardIndex:
    """Memory-mapped view over one source's shards."""

    def __init__(self, feat_dir: Path):
        self.dir = Path(feat_dir)
        self.entries: list[dict] = []
        self.maps: dict[int, np.memmap] = {}

        for idx_path in sorted(self.dir.glob("shard_*.jsonl")):
            shard = int(idx_path.stem.split("_")[1])
            for line in idx_path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    e["shard"] = shard
                    self.entries.append(e)
        if not self.entries:
            raise FileNotFoundError(f"no shards under {self.dir}")

        self.n_feat = self.entries[0]["feat"]
        self.by_uid = {e["uid"]: e for e in self.entries}

    def _map(self, shard: int) -> np.memmap:
        if shard not in self.maps:
            self.maps[shard] = np.memmap(
                self.dir / f"shard_{shard:03d}.f16", dtype=np.float16, mode="r"
            )
        return self.maps[shard]

    def get(self, uid: str) -> np.ndarray:
        e = self.by_uid[uid]
        start = e["offset"] // 2                      # byte offset -> float16 index
        n = e["frames"] * e["feat"]
        return np.asarray(self._map(e["shard"])[start:start + n]).reshape(
            e["frames"], e["feat"]
        )


def augment(feats: np.ndarray, rng: np.random.Generator, n_points: int) -> np.ndarray:
    """Training-time augmentation on normalized keypoints.

    At 124k pairs we are data-limited, so this is cheap regularisation:

    * **horizontal flip** -- signers differ in dominant hand, and a mirrored
      utterance means the same thing. Negating x and swapping the left/right
      hand blocks doubles the effective data.
    * **temporal jitter** -- resample at 0.9-1.1x speed; signing rate varies.
    * **frame dropout** -- blank ~5% of frames, mimicking detection failures
      that occur in the real test data anyway.
    * **spatial noise** -- small jitter, since keypoint estimation is noisy.
    """
    T, F = feats.shape
    xy = feats[:, : n_points * 2].reshape(T, n_points, 2)

    if rng.random() < 0.5:
        xy = xy.copy()
        xy[..., 0] *= -1.0
        # Hands are the last 42 points, left block then right; a mirrored
        # signer's left hand plays the role of the right.
        lh = slice(n_points - 42, n_points - 21)
        rh = slice(n_points - 21, n_points)
        xy[:, lh], xy[:, rh] = xy[:, rh].copy(), xy[:, lh].copy()

    if rng.random() < 0.5:
        xy = xy + rng.normal(0, 0.01, xy.shape).astype(xy.dtype)

    if rng.random() < 0.3:
        keep = rng.random(T) > 0.05
        if keep.any():
            xy = xy[keep]

    if rng.random() < 0.5:
        speed = rng.uniform(0.9, 1.1)
        n = max(4, int(len(xy) / speed))
        idx = np.linspace(0, len(xy) - 1, n).round().astype(int)
        xy = xy[idx]

    return xy.reshape(len(xy), -1)


class SLTDataset(Dataset):
    """(pose, register, text) triples.

    `text_map=None` gives an inference dataset: every uid in the shard index,
    with the register taken from the uid instead of the target.
    """

    def __init__(self, feat_dir: Path, text_map: dict[str, str] | None = None,
                 max_frames: int = 512, uids: list[str] | None = None,
                 augment_prob: float = 0.0, velocity: bool = False,
                 clip_value: float = 5.0):
        self.index = ShardIndex(feat_dir)
        self.text = text_map
        self.max_frames = max_frames
        self.augment_prob = augment_prob
        self.velocity = velocity
        self.clip_value = clip_value

        if uids is not None:
            self.uids = [u for u in uids if u in self.index.by_uid]
        elif text_map is not None:
            # Only clips that have both features and non-junk text.
            self.uids = sorted(set(self.index.by_uid) & set(text_map))
        else:
            self.uids = sorted(self.index.by_uid)

        self.n_points = self.index.n_feat // 2
        # Velocity doubles the feature width; the model must be built to match.
        self.n_feat = self.index.n_feat * (2 if velocity else 1)
        self._rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        uid = self.uids[i]
        feats = np.asarray(self.index.get(uid), dtype=np.float32)

        # Clamp normalization blow-ups. Shoulder-width normalization divides by
        # a per-frame width guarded only at 1e-3, so a poor shoulder detection
        # can scale coordinates by ~500x. Measured: 0.8% of training clips
        # exceed |x|=5, but 1.7% of TEST clips do, with a p99.9 of 13.9 against
        # training's 3.29. The model therefore meets input scales at test time
        # it never saw in training. Sane keypoints live well inside +-5.
        if self.clip_value:
            np.clip(feats, -self.clip_value, self.clip_value, out=feats)

        if self.augment_prob and self._rng.random() < self.augment_prob:
            feats = augment(feats, self._rng, self.n_points)

        feats = feats[: self.max_frames]
        if self.velocity:
            from data.pose import add_velocity
            feats = add_velocity(feats)

        if self.text is not None:
            text = self.text[uid]
            reg = register_of_text(text)
        else:
            text, reg = "", register_of_uid(uid)
        return {"uid": uid, "feats": torch.from_numpy(np.ascontiguousarray(feats)),
                "text": text, "register": reg}


def collate(batch: list[dict], tokenizer=None, max_target_len: int = 48) -> dict:
    """Pad pose sequences to the batch max and tokenize targets."""
    lens = [b["feats"].shape[0] for b in batch]
    T, F = max(lens), batch[0]["feats"].shape[1]

    feats = torch.zeros(len(batch), T, F)
    mask = torch.zeros(len(batch), T, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["feats"].shape[0]
        feats[i, :n] = b["feats"]
        mask[i, :n] = 1

    out = {
        "uids": [b["uid"] for b in batch],
        "feats": feats,
        "mask": mask,
        "registers": [b["register"] for b in batch],
        "texts": [b["text"] for b in batch],
    }

    if tokenizer is not None and any(b["text"] for b in batch):
        enc = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                        max_length=max_target_len, return_tensors="pt")
        labels = enc.input_ids
        # -100 masks pad positions out of the loss.
        labels[labels == tokenizer.pad_token_id] = -100
        out["labels"] = labels
    return out
