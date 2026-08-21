"""Parsing and grouping for iSign / WSLP clip uids.

Two uid styles appear in the task data:
    {video_id}_clip_{n}     e.g. hKxZn68hWIc_clip_8
    {video_id}-{n}          e.g. uYAtGAJORZY-305

Both encode a video and an ordinal position of the clip within that video,
which is what lets us treat a video's clips as a document.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

_CLIP = re.compile(r"^(?P<vid>.+?)_clip_(?P<idx>\d+)$")
_DASH = re.compile(r"^(?P<vid>.+?)-(?P<idx>\d+)$")


@dataclass(frozen=True)
class Uid:
    raw: str
    video: str
    index: int
    style: str  # "_clip_" or "-"

    def sibling(self, index: int) -> str:
        """Reconstruct the uid of another clip in the same video."""
        return f"{self.video}{self.style}{index}"


def parse(uid: str) -> Uid | None:
    """Parse a uid, or return None if it matches neither style."""
    for pat, style in ((_CLIP, "_clip_"), (_DASH, "-")):
        m = pat.match(uid)
        if m:
            return Uid(uid, m["vid"], int(m["idx"]), style)
    return None


def parse_all(uids) -> tuple[list[Uid], list[str]]:
    """Parse many uids. Returns (parsed, unparseable_raw_uids)."""
    ok, bad = [], []
    for u in uids:
        p = parse(u)
        (ok if p else bad).append(p if p else u)
    return ok, bad


def group_by_video(uids) -> dict[str, list[Uid]]:
    """Group parsed uids by video id, each list sorted by clip index."""
    out = defaultdict(list)
    for u in uids:
        p = u if isinstance(u, Uid) else parse(u)
        if p:
            out[p.video].append(p)
    for v in out:
        out[v].sort(key=lambda x: x.index)
    return dict(out)


def neighbours(uid: Uid, radius: int = 3, exclude: set[int] | None = None) -> list[str]:
    """Uids of clips within +-radius of this one, nearest first.

    `exclude` drops clip indices we must not condition on (e.g. held-out
    evaluation clips whose text we do not have).
    """
    exclude = exclude or set()
    out = []
    for d in range(1, radius + 1):
        for j in (uid.index - d, uid.index + d):
            if j >= 0 and j not in exclude:
                out.append(uid.sibling(j))
    return out
