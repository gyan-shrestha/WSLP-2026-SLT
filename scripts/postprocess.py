#!/usr/bin/env python3
"""Surface-form cleanup on predictions. No model, no GPU.

Two measured facts motivate this:

  constant sentence  -> chrF 0.15   (no content, well-formed English)
  our best model     -> chrF 0.13   (real content, malformed output)

The constant string wins on surface form alone. chrF is a character n-gram
F-score, so repeated tokens and wrong casing cost precision directly while
contributing nothing. 18.4% of our clips contain an immediately repeated word
("a little little little"), and 2.1% of all emitted tokens are such repeats.

    python scripts/postprocess.py --preds preds.csv --out cleaned.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def collapse_repeats(text: str, max_run: int = 1) -> str:
    """Collapse immediately repeated words to `max_run` occurrences.

    "a little little little sad" -> "a little sad". Degenerate decoder loops
    emit these; the reference never contains them, so each extra copy is
    character-level precision thrown away.
    """
    out: list[str] = []
    for w in text.split():
        run = 0
        for prev in reversed(out):
            if prev.lower() == w.lower():
                run += 1
            else:
                break
        if run < max_run:
            out.append(w)
    return " ".join(out)


def collapse_phrase_repeats(text: str) -> str:
    """Collapse an immediately repeated 2- or 3-word phrase.

    Beam search under weak input also loops at phrase level ("of the of the").
    """
    for n in (3, 2):
        w = text.split()
        out, i = [], 0
        while i < len(w):
            if i + 2 * n <= len(w) and [x.lower() for x in w[i:i + n]] == \
                                       [x.lower() for x in w[i + n:i + 2 * n]]:
                out.extend(w[i:i + n])
                i += 2 * n
            else:
                out.append(w[i])
                i += 1
        text = " ".join(out)
    return text


def fix_register(text: str, uid: str) -> str:
    """Match the surface register of the uid's sub-corpus.

    Evaluation uids come in two styles and no video contains both. The
    `_clip_` corpus is capitalised, punctuated prose; the `-` corpus is
    lowercase fragments. chrF is character-level, so casing and terminal
    punctuation are worth real points on the 43% of clips in the `_clip_` half.
    """
    t = text.strip()
    if not t:
        return t
    if "_clip_" in uid:
        t = t[0].upper() + t[1:]
        if t[-1] not in ".!?":
            t += "."
    return t


def clean(text: str, uid: str, do_register: bool) -> str:
    t = collapse_phrase_repeats(collapse_repeats(text))
    t = re.sub(r"\s+([,.!?])", r"\1", t)        # no space before punctuation
    t = re.sub(r"\s+", " ", t).strip()
    if do_register:
        t = fix_register(t, uid)
    return t or "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.preds, newline="", encoding="utf-8")))
    changed = 0
    out_rows = []
    for r in rows:
        c = clean(r["text"], r["uid"], not a.no_register)
        changed += c != r["text"]
        out_rows.append({"uid": r["uid"], "text": c})

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["uid", "text"])
        w.writeheader()
        w.writerows(out_rows)

    before = sum(len(r["text"].split()) for r in rows) / len(rows)
    after = sum(len(r["text"].split()) for r in out_rows) / len(out_rows)
    print(f"{len(rows)} rows, {changed} changed ({100 * changed / len(rows):.1f}%)")
    print(f"mean length {before:.2f} -> {after:.2f} words")


if __name__ == "__main__":
    main()
