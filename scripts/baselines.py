#!/usr/bin/env python3
"""Degenerate baselines: systems that perform no translation whatsoever.

Purpose is twofold.

1. **Measure the metric floor.** The leaderboard sits at chrF 0.14. If a system
   that never looks at the input scores near that, every later number must be
   read against this floor rather than against zero. That result is worth more
   to the workshop than a marginal BLEU delta.

2. **Test register routing cheaply.** F5 in notes/findings.md: eval splits into
   two disjoint sub-corpora, identifiable from the uid alone -- `_clip_` is
   fluent punctuated news prose, `-` is lowercase fragments. `generic` and
   `generic-routed` are identical except the latter emits a register-matched
   string per sub-corpus. The gap between them is the value of routing,
   measured before we build any of it.

    python scripts/baselines.py --isign data/iSign_v1.1.csv --outdir subs/

Each baseline costs one of ~20 daily Codabench submissions.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.uid import parse  # noqa: E402

SENT_UID = re.compile(r"^.+-\d+$")


def register_of(uid: str) -> str:
    """Which eval sub-corpus a uid belongs to. Free, deterministic, at test time."""
    return "news" if "_clip_" in uid else "frag"


def load_train_text(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            r["text"].strip()
            for r in csv.DictReader(f)
            if SENT_UID.match(r["uid"]) and r["text"].strip()
        ]


def load_uids(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["uid"].strip() for r in csv.DictReader(f)]


def build_baselines(uids: list[str], train: list[str], seed: int = 0) -> dict[str, dict]:
    rng = random.Random(seed)
    counts = Counter(train)
    most_common = counts.most_common(1)[0][0]

    # Median-length training sentence: matches the corpus length profile, which
    # is what chrF's F-score balance rewards, without matching any content.
    by_len = sorted(train, key=lambda s: len(s.split()))
    median_len_sent = by_len[len(by_len) // 2]

    # Highest-frequency content words, in frequency order. Isolates the pure
    # unigram-overlap contribution to the score.
    words = Counter(w.lower() for s in train for w in re.findall(r"[a-zA-Z']+", s))
    top_words = " ".join(w for w, _ in words.most_common(12))

    # Register-matched strings. The `frag` form mimics iSign's lowercase,
    # unpunctuated fragments; the `news` form mimics the capitalised, punctuated
    # sentences seen in sample_submission.csv.
    routed = {
        "frag": median_len_sent.lower().rstrip(" .!?"),
        "news": "The video shows a person explaining the details of the news report.",
    }

    return {
        # a literal format check -- proves the scorer accepts our archive
        "hello": {u: "hello" for u in uids},
        "constant-frequent": {u: most_common for u in uids},
        "generic": {u: median_len_sent for u in uids},
        "generic-routed": {u: routed[register_of(u)] for u in uids},
        "token-prior": {u: top_words for u in uids},
        "random-train": {u: rng.choice(train) for u in uids},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isign", type=Path, default=Path("data/iSign_v1.1.csv"))
    ap.add_argument("--uids", type=Path, default=Path("data/task/test.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("subs"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_submission import build

    uids = load_uids(a.uids)
    train = load_train_text(a.isign)
    print(f"{len(uids)} eval uids | {len(train)} training sentences")
    reg = Counter(register_of(u) for u in uids)
    print(f"register split: {dict(reg)}\n")

    for name, preds in build_baselines(uids, train, a.seed).items():
        sample = next(iter(preds.values()))
        print(f"--- {name}\n    e.g. {sample[:90]!r}")
        build(uids, preds, a.outdir / f"{name}.zip")

    print(
        "\nSubmit in this order and record each score in notes/leaderboard.md:\n"
        "  1. hello              -> proves the format is accepted\n"
        "  2. generic            -> the floor\n"
        "  3. generic-routed     -> floor + register routing; the gap is what routing buys\n"
        "  4. random-train       -> does the metric reward fluency without correctness?\n"
        "Hold constant-frequent and token-prior unless a slot is spare."
    )


if __name__ == "__main__":
    main()
