#!/usr/bin/env python3
"""Build a Codabench submission zip.

The scorer requires, per the phase description:
  * a zip containing ONLY `answer.csv`, at the archive root
  * rows in the exact order of the phase's uid file
so this refuses to emit anything that violates either.

    # from a predictions file (csv with uid,text  or  jsonl with uid/text keys)
    python scripts/make_submission.py --preds runs/foo/preds.csv -o subs/foo.zip

    # constant string, for format checks and floor measurement
    python scripts/make_submission.py --constant "hello" -o subs/hello.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ANSWER_NAME = "answer.csv"


def read_uids(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["uid"].strip() for r in csv.DictReader(f)]


def read_preds(path: Path) -> dict[str, str]:
    """Accept csv (uid,text) or jsonl ({"uid":..., "text":...})."""
    out: dict[str, str] = {}
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[str(d["uid"]).strip()] = str(d.get("text", ""))
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            tcol = "text" if "text" in rdr.fieldnames else rdr.fieldnames[1]
            for r in rdr:
                out[r["uid"].strip()] = (r.get(tcol) or "")
    return out


def sanitize(text: str) -> str:
    """One line, no stray whitespace. Empty predictions become a single space.

    A truly empty field can make a scorer error out or silently drop the row;
    a space keeps the row shape intact and scores ~0, which is what we want.
    """
    t = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    return t if t else " "


def build(uids: list[str], preds: dict[str, str], out: Path, strict: bool = True) -> None:
    missing = [u for u in uids if u not in preds]
    if missing:
        msg = f"{len(missing)} uids have no prediction, e.g. {missing[:5]}"
        if strict:
            sys.exit(f"FATAL: {msg}\n(pass --allow-missing to fill them with a space)")
        print(f"WARN: {msg} -- filling with a space", file=sys.stderr)

    extra = set(preds) - set(uids)
    if extra:
        print(f"NOTE: dropping {len(extra)} predictions not in the uid file", file=sys.stderr)

    buf = io.StringIO()
    # QUOTE_ALL: sentences contain commas and quotes; keep the parser honest.
    w = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
    w.writerow(["uid", "text"])
    for u in uids:
        w.writerow([u, sanitize(preds.get(u, " "))])

    out.parent.mkdir(parents=True, exist_ok=True)
    # ZIP_DEFLATED and a fixed date so identical predictions give identical zips.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(ANSWER_NAME, date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, buf.getvalue())

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert names == [ANSWER_NAME], f"archive must contain only {ANSWER_NAME}, got {names}"
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB, {len(uids)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", type=Path, default=Path("data/task/test.csv"),
                    help="phase uid file; defines row order")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preds", type=Path, help="csv or jsonl of predictions")
    src.add_argument("--constant", type=str, help="emit this string for every uid")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()

    uids = read_uids(a.uids)
    preds = {u: a.constant for u in uids} if a.constant is not None else read_preds(a.preds)
    build(uids, preds, a.out, strict=not a.allow_missing)


if __name__ == "__main__":
    main()
