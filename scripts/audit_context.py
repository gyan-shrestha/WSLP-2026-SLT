#!/usr/bin/env python3
"""Gate check: does the document-context premise actually hold?

The task split holds out ~10% of each video's clips, so every evaluation clip
sits inside a mostly-observed document. That is only useful if iSign actually
ships the *neighbouring* clips with ground-truth English. This script measures
that, and nothing about the architecture is settled until it has run.

    python scripts/audit_context.py --isign data/iSign_v1.1.csv

Reports, for val and test separately:
  * how many eval uids exist in iSign at all (leakage check -- these MUST be
    dropped from training)
  * what fraction of eval clips have >=1 neighbour carrying text
  * how that coverage decays with radius

Exit status is 0 regardless; read the verdict at the bottom.
"""

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.uid import group_by_video, parse, parse_all  # noqa: E402

TEXT_COLS = ("translation", "text", "sentence", "caption", "english")
UID_COLS = ("uid", "id", "video_id", "clip_id", "name")


def sniff(header: list[str], candidates: tuple[str, ...], what: str) -> str:
    low = {c.lower().strip(): c for c in header}
    for c in candidates:
        if c in low:
            return low[c]
    sys.exit(f"could not find a {what} column in: {header}")


def load_isign(path: Path) -> dict[str, str]:
    """uid -> English text, skipping rows with empty text."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        ucol = sniff(rdr.fieldnames, UID_COLS, "uid")
        tcol = sniff(rdr.fieldnames, TEXT_COLS, "text")
        print(f"iSign columns: uid={ucol!r} text={tcol!r}  (all: {rdr.fieldnames})")
        return {
            r[ucol].strip(): r[tcol].strip()
            for r in rdr
            if r.get(ucol, "").strip() and r.get(tcol, "").strip()
        }


def load_uids(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["uid"].strip() for r in csv.DictReader(f)]


def report(name, eval_uids, isign, held_all, radii):
    print(f"\n{'=' * 62}\n{name.upper()}  ({len(eval_uids)} clips)\n{'=' * 62}")

    parsed, bad = parse_all(eval_uids)
    if bad:
        print(f"!! {len(bad)} uids did not parse, e.g. {bad[:3]}")

    # --- leakage: eval uids present in iSign with text -------------------
    leaked = [u.raw for u in parsed if u.raw in isign]
    print(f"eval uids present in iSign with text : {len(leaked)}/{len(parsed)} "
          f"({len(leaked) / len(parsed):.1%})")
    if leaked:
        print("   ^ MUST be removed from training. Sample:")
        for u in leaked[:3]:
            print(f"     {u} -> {isign[u][:70]!r}")

    # --- context coverage -------------------------------------------------
    by_vid_held = {v: {c.index for c in cs} for v, cs in group_by_video(held_all).items()}

    print(f"\n{'radius':>7} {'>=1 ctx':>10} {'mean #ctx':>11} {'median gap':>11}")
    for r in radii:
        have, counts, gaps = 0, [], []
        for u in parsed:
            excl = by_vid_held.get(u.video, set())
            found = []
            for d in range(1, r + 1):
                for j in (u.index - d, u.index + d):
                    if j < 0 or j in excl:
                        continue
                    sib = u.sibling(j)
                    if sib in isign:
                        found.append((d, sib))
            if found:
                have += 1
                gaps.append(min(d for d, _ in found))
            counts.append(len(found))
        print(f"{r:>7} {have / len(parsed):>9.1%} {statistics.mean(counts):>11.2f} "
              f"{(statistics.median(gaps) if gaps else float('nan')):>11.1f}")

    return len(leaked), have / len(parsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isign", type=Path, required=True, help="iSign_v1.1.csv")
    ap.add_argument("--val", type=Path, default="data/task/val.csv")
    ap.add_argument("--test", type=Path, default="data/task/test.csv")
    ap.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    a = ap.parse_args()

    isign = load_isign(a.isign)
    print(f"iSign rows with text: {len(isign)}")
    styles = Counter(p.style for p in (parse(u) for u in isign) if p)
    print(f"iSign uid styles: {dict(styles)}  unparseable: {len(isign) - sum(styles.values())}")

    val, test = load_uids(a.val), load_uids(a.test)
    held_all = val + test

    v_leak, v_cov = report("val", val, isign, held_all, a.radii)
    t_leak, t_cov = report("test", test, isign, held_all, a.radii)

    print(f"\n{'=' * 62}\nVERDICT\n{'=' * 62}")
    print(f"val references recoverable from iSign : {v_leak}/{len(val)}"
          f"  -> {'local eval possible' if v_leak > len(val) * 0.5 else 'NO local dev set'}")
    print(f"test references present in iSign      : {t_leak}/{len(test)}"
          f"  -> {'DROP from training (leakage)' if t_leak else 'clean'}")
    print(f"test clips with context at radius 3   : {t_cov:.1%}")
    if t_cov > 0.8:
        print("\n=> Document context is the centerpiece. Build the context-conditioned model.")
    elif t_cov > 0.3:
        print("\n=> Context is a useful auxiliary signal, not the centerpiece.")
    else:
        print("\n=> Premise FAILS. Fall back to pose-only seq2seq.")


if __name__ == "__main__":
    main()
