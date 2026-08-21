"""Decode and normalize MediaPipe Holistic `.pose` files.

Design notes
------------
**Keypoint subset.** Holistic emits 543 points, but sign is carried by the
hands, the face, and the upper body. Legs and the dense face mesh are noise
that costs memory and invites overfitting, so we keep both hands in full (42),
the upper-body pose joints, and a sparse face subset (lips/eyes/brows).
Roughly 543 -> ~90 points.

**Normalization.** Signers vary in distance from camera and position in frame,
and none of that is linguistic. We centre on the shoulder midpoint and scale by
shoulder width, which makes the representation invariant to both. This is the
single highest-value preprocessing step.

**Precision.** float16. Keypoints are normalized image coordinates in [0,1]
with ~3 decimal digits of real precision; fp16 is lossless in practice here and
halves both disk and dataloader bandwidth.

Net effect: 159 GiB of `.pose` -> ~5 GiB of arrays.
"""

from __future__ import annotations

import numpy as np

# MediaPipe Holistic pose indices worth keeping. Everything from the hips down
# is irrelevant to sign and adds jitter.
UPPER_BODY_POSE = [
    0,                    # nose
    2, 5,                 # eyes (inner)
    7, 8,                 # ears
    9, 10,                # mouth corners
    11, 12,               # shoulders   <- normalization anchors
    13, 14,               # elbows
    15, 16,               # wrists
    17, 18, 19, 20, 21, 22,  # hand stubs (pinky/index/thumb on the pose model)
    23, 24,               # hips (retained: torso scale reference)
]

# Sparse face subset. The full 468-point mesh is overkill; mouth shape, eye
# aperture and brow position carry the grammatical facial expression that
# matters for sign.
FACE_SUBSET = [
    # outer lips
    61, 291, 0, 17, 40, 270, 39, 269, 37, 267,
    # inner lips
    78, 308, 13, 14, 82, 312, 87, 317,
    # eyes
    33, 133, 159, 145, 362, 263, 386, 374,
    # brows
    70, 63, 105, 66, 107, 300, 293, 334, 296, 336,
    # face outline anchors
    10, 152, 234, 454,
]

L_SHOULDER, R_SHOULDER = 11, 12


class PoseLayout:
    """Resolved index map for one archive's component layout.

    Component naming varies between pose-format versions, so we resolve by
    name at runtime rather than hardcoding offsets and silently slicing the
    wrong joints.
    """

    def __init__(self, header):
        self.offsets: dict[str, tuple[int, int]] = {}
        off = 0
        for c in header.components:
            self.offsets[c.name.upper()] = (off, off + len(c.points))
            off += len(c.points)
        self.total = off

        self.keep: list[int] = []
        self.pose_idx: dict[str, int] = {}

        pose = self._find("POSE")
        if pose:
            base, _ = pose
            for i in UPPER_BODY_POSE:
                self.pose_idx[f"pose{i}"] = len(self.keep)
                self.keep.append(base + i)
            self.shoulder_l = self.pose_idx.get(f"pose{L_SHOULDER}")
            self.shoulder_r = self.pose_idx.get(f"pose{R_SHOULDER}")
        else:
            self.shoulder_l = self.shoulder_r = None

        self.n_face_kept = 0
        face = self._find("FACE")
        if face:
            base, end = face
            for i in FACE_SUBSET:
                if base + i < end:
                    self.keep.append(base + i)
                    self.n_face_kept += 1

        for hand in ("LEFT_HAND", "RIGHT_HAND"):
            h = self._find(hand)
            if h:
                base, end = h
                self.keep.extend(range(base, end))

        # Index groups within `keep`, for SignSpace local normalization.
        # Order of `keep`: body pose, then face, then left hand, then right hand.
        n_body = len(self.pose_idx)
        n_face = self.n_face_kept
        self.local_groups = [
            list(range(n_body, n_body + n_face)),                        # face
            list(range(n_body + n_face, n_body + n_face + 21)),           # left hand
            list(range(n_body + n_face + 21, n_body + n_face + 42)),      # right hand
        ]
        self.local_groups = [g for g in self.local_groups if g]
        self.local_groups_flat = {i for g in self.local_groups for i in g}
        self.n_points = len(self.keep)

    def _find(self, prefix: str) -> tuple[int, int] | None:
        """Resolve a component by name prefix.

        Prefix, not substring: archives carry both POSE_LANDMARKS and
        POSE_WORLD_LANDMARKS, and a substring match on "POSE" would pick
        whichever happened to be enumerated first. Exact match wins outright so
        "POSE" can never resolve to the world-landmark block.
        """
        want = f"{prefix}_LANDMARKS"
        if want in self.offsets:
            return self.offsets[want]
        hits = [n for n in self.offsets if n.startswith(prefix)]
        if not hits:
            return None
        if len(hits) > 1:
            raise ValueError(f"ambiguous component prefix {prefix!r}: {hits}")
        return self.offsets[hits[0]]


def normalize(xy: np.ndarray, layout: PoseLayout,
              standardize: bool = True) -> np.ndarray:
    """Centre on the shoulder midpoint and remove clip-level scale.

    xy: (frames, points, 2). Returns the same shape, float32.

    Two changes over naive per-frame shoulder-width scaling, both aimed at
    domain shift rather than at accuracy on any single clip:

    **Robust scale.** Dividing each frame by *its own* shoulder width amplifies
    estimation noise, and a frame with a poor shoulder detection (width ~1e-3)
    scales coordinates by ~500x. We use the clip's median width instead, so one
    bad frame cannot distort the clip.

    **Per-clip standardization.** Even after shoulder scaling, feature
    distributions differ by source: iSign std 1.99 vs task-test std 3.04, with
    p99.9 |x| of 3.29 vs 13.9. The model met input scales at test time it never
    saw in training, and test BLEU was 0.00 against 0.0119 in-domain. Z-scoring
    each clip makes every clip statistically identical regardless of camera,
    framing, or signer, removing domain shift at the input rather than asking
    the encoder to absorb it.
    """
    xy = xy.astype(np.float32, copy=True)
    if layout.shoulder_l is None or layout.shoulder_r is None:
        return xy

    ls, rs = xy[:, layout.shoulder_l], xy[:, layout.shoulder_r]
    centre = (ls + rs) / 2.0                                   # (frames, 2)
    width = np.linalg.norm(ls - rs, axis=-1)                   # (frames,)

    valid = width > 1e-2
    if not valid.any():
        out = xy - centre[:, None, :]
    else:
        # One robust scale for the whole clip, not one per frame.
        scale = float(np.median(width[valid]))
        # Missing centres read as (0,0); hold the last good centre instead.
        centre = np.where(valid[:, None], centre, np.median(centre[valid], axis=0))
        out = (xy - centre[:, None, :]) / max(scale, 1e-2)

    if standardize:
        present = np.abs(out).sum(axis=-1) > 0                 # (frames, points)
        if present.any():
            vals = out[present]
            mu, sigma = vals.mean(axis=0), vals.std(axis=0)
            out = (out - mu) / np.maximum(sigma, 1e-3)
            out[~present] = 0.0                                # keep "absent" as 0
        np.clip(out, -6.0, 6.0, out=out)

    return out


def decode(raw: bytes, layout: PoseLayout | None = None, scheme: str = "signspace"):
    """Decode `.pose` bytes -> (features (T, P*2) float16, layout).

    Confidence-zero points are zeroed rather than left at whatever coordinate
    the estimator hallucinated, so the model can learn "absent" as a value.
    """
    from pose_format import Pose

    p = Pose.read(raw)
    if layout is None:
        layout = PoseLayout(p.header)

    data = np.asarray(p.body.data)          # (T, people, points, dims)
    conf = np.asarray(p.body.confidence)    # (T, people, points)
    if data.ndim == 4:
        data, conf = data[:, 0], conf[:, 0]

    data = data[:, layout.keep, :2]
    conf = conf[:, layout.keep]
    data = np.where(conf[..., None] > 0, data, 0.0)

    if scheme == "signspace":
        out = signspace_normalize(data, layout)
    else:
        out = normalize(data, layout)
    return out.reshape(out.shape[0], -1).astype(np.float16), layout


MISSING = -10.0   # sentinel for undetected keypoints, per arXiv:2507.01532


def _local_norm(xy: np.ndarray, present: np.ndarray, border: float = 0.1) -> np.ndarray:
    """Scale one body part into [-1,1] per frame, preserving aspect ratio.

    Applied independently to each hand and to the face, which is what makes
    handshape invariant to where the hand is and how large it appears. A 10%
    border absorbs pose-estimation jitter at the extremes.
    """
    out = xy.copy()
    for t in range(len(xy)):
        m = present[t]
        if m.sum() < 2:
            out[t] = MISSING
            continue
        pts = xy[t][m]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = (lo + hi) / 2.0
        # One scale for both axes -> aspect ratio preserved.
        half = max((hi - lo).max() / 2.0, 1e-6) * (1.0 + border)
        o = (xy[t] - centre) / half
        o[~m] = MISSING
        out[t] = o
    return out


def signspace_normalize(xy: np.ndarray, layout: "PoseLayout") -> np.ndarray:
    """SignSpace normalization (arXiv:2507.01532, Table 1).

    Measured there as the single largest preprocessing lever for *cross-dataset*
    translation: BLEU-4 0.73 (none) -> 1.13 (frame bbox) -> **2.17** (SignSpace)
    on How2Sign without finetuning. Interpolation and augmentation moved the
    same number by <0.3.

    Two regimes:

    * **Body, globally.** A box centred between the shoulders, sized at 3x the
      shoulder distance, mapped to [-1,1]. This preserves the spatial relation
      between body parts -- where the hands are relative to the torso.
    * **Each hand and the face, locally.** Normalised independently so that
      handshape is invariant to position and apparent size. This is the part we
      were missing: a global transform leaves a distant signer's handshape tiny
      and a close one's huge, so the model must learn every scale separately.

    Undetected keypoints are set to MISSING (-10) rather than 0, so "absent" is
    unambiguous instead of colliding with a legitimate coordinate.
    """
    xy = xy.astype(np.float32, copy=True)
    present = np.abs(xy).sum(axis=-1) > 0            # (T, P)
    out = np.full_like(xy, MISSING)

    # --- body, global ----------------------------------------------------
    body = [i for i in range(layout.n_points) if i not in layout.local_groups_flat]
    if layout.shoulder_l is not None and layout.shoulder_r is not None and body:
        ls, rs = xy[:, layout.shoulder_l], xy[:, layout.shoulder_r]
        centre = (ls + rs) / 2.0
        width = np.linalg.norm(ls - rs, axis=-1)
        valid = width > 1e-2
        if valid.any():
            width = np.where(valid, width, np.median(width[valid]))
            centre = np.where(valid[:, None], centre, np.median(centre[valid], axis=0))
            half = (3.0 * width) / 2.0               # box is 3x shoulder distance
            b = (xy[:, body] - centre[:, None, :]) / half[:, None, None]
            b[~present[:, body]] = MISSING
            out[:, body] = b

    # --- hands and face, local -------------------------------------------
    for group in layout.local_groups:
        if group:
            out[:, group] = _local_norm(xy[:, group], present[:, group])

    np.clip(out, -3.0, 3.0, out=out)
    out[out == np.clip(MISSING, -3.0, 3.0)] = MISSING   # keep the sentinel intact
    return out


def add_velocity(feats: np.ndarray) -> np.ndarray:
    """Append first-order frame differences: (T, F) -> (T, 2F).

    Sign meaning is carried by *movement* -- handshape trajectory, direction,
    speed -- not by static posture. Feeding only per-frame coordinates forces
    the conv stack to rediscover motion from scratch; giving it velocity
    directly is standard practice in gesture and speech modelling and costs
    one subtraction.

    Frame 0's velocity is zero (no predecessor).
    """
    v = np.zeros_like(feats)
    v[1:] = feats[1:] - feats[:-1]
    return np.concatenate([feats, v], axis=-1)


def resample(feats: np.ndarray, max_frames: int = 512, stride: int = 1) -> np.ndarray:
    """Uniformly subsample long clips to a frame budget.

    Uniform subsampling beats truncation: sign meaning is distributed across a
    clip, so cutting the tail drops content, while thinning preserves the whole
    utterance at lower temporal resolution.
    """
    if stride > 1:
        feats = feats[::stride]
    if len(feats) > max_frames:
        idx = np.linspace(0, len(feats) - 1, max_frames).round().astype(int)
        feats = feats[idx]
    return feats
