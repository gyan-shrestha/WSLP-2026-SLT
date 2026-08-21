"""Local metrics that mirror the Codabench scorer.

The scorer's own install log (captured from a submission) shows it uses:

    rouge==1.0.1              (pltrdy's `rouge`, NOT Google's `rouge_score`)
    sacrebleu==2.4.2          (chrF)
    nltk.translate.bleu_score (BLEU, punkt tokenizer, NO SmoothingFunction)

That last point matters more than it looks. Unsmoothed NLTK BLEU is
**exactly 0** whenever the corpus has no 4-gram match, which is why the
leaderboard reads BLEU 0.01 -- the metric is near-binary at this signal level
and cannot rank systems. chrF is character-level and continuous, so it is the
metric worth optimising, and the one our decoding calibration targets.

Two things remain unverified because the log does not show them: whether the
scorer lowercases before tokenizing, and which ROUGE variant it reports (the
package returns rouge-1/2/l). We report all of them; `chrf` is the number to
trust for ranking.

`nltk` refuses to import from some working directories (its `inisec` guard
misfires under /blue), so callers should run from a neutral cwd. If the import
fails we fall back to a self-contained implementation rather than killing a
training run -- a metric must never take down a job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

_WARNED = False


@dataclass
class Scores:
    bleu: float
    chrf: float
    rouge_l: float
    rouge_1: float = 0.0
    exact: bool = True          # False if computed by the fallback path

    def __str__(self) -> str:
        tag = "" if self.exact else " (approx)"
        return (f"BLEU {self.bleu:.4f} | chrF {self.chrf:.4f} | "
                f"ROUGE-L {self.rouge_l:.4f}{tag}")

    def as_dict(self) -> dict:
        return asdict(self)


# --- fallback, used only if nltk/rouge cannot be imported -----------------

def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def _rouge_l_fallback(hyps: list[str], refs: list[str]) -> float:
    total = 0.0
    for h, r in zip(hyps, refs):
        ht, rt = h.lower().split(), r.lower().split()
        if not ht or not rt:
            continue
        l = _lcs(ht, rt)
        if l:
            p, rec = l / len(ht), l / len(rt)
            total += 2 * p * rec / (p + rec)
    return total / max(1, len(hyps))


# --- scorer-faithful path -------------------------------------------------

def score(hyps: list[str], refs: list[str], lowercase: bool = False) -> Scores:
    if len(hyps) != len(refs):
        raise ValueError(f"length mismatch: {len(hyps)} hyps vs {len(refs)} refs")

    import sacrebleu
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score / 100.0

    try:
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import corpus_bleu
        from rouge import Rouge

        def prep(s: str) -> str:
            return s.lower() if lowercase else s

        # NLTK corpus_bleu: uniform 4-gram weights, brevity penalty, no smoothing.
        bleu = corpus_bleu(
            [[word_tokenize(prep(r))] for r in refs],
            [word_tokenize(prep(h)) for h in hyps],
        )

        # `rouge` raises on an empty hypothesis, so substitute a placeholder
        # token; the submission builder emits " " for missing predictions.
        safe_h = [h if h.strip() else "." for h in hyps]
        safe_r = [r if r.strip() else "." for r in refs]
        rg = Rouge().get_scores(safe_h, safe_r, avg=True)

        return Scores(bleu=bleu, chrf=chrf,
                      rouge_l=rg["rouge-l"]["f"], rouge_1=rg["rouge-1"]["f"])

    except Exception as e:  # noqa: BLE001
        # Never let a metric dependency kill a run -- but say why we degraded,
        # otherwise a silent fallback quietly changes what every ablation means.
        global _WARNED
        if not _WARNED:
            print(f"[metrics] exact scorer unavailable ({type(e).__name__}: {e}); "
                  f"using approximate fallback", flush=True)
            _WARNED = True
        import sacrebleu as sb
        return Scores(bleu=sb.corpus_bleu(hyps, [refs]).score / 100.0,
                      chrf=chrf, rouge_l=_rouge_l_fallback(hyps, refs), exact=False)


def score_by_group(hyps, refs, groups) -> dict[str, Scores]:
    """Scores broken out per key -- used to track the two eval sub-corpora."""
    buckets: dict[str, tuple[list, list]] = {}
    for h, r, g in zip(hyps, refs, groups):
        buckets.setdefault(g, ([], []))
        buckets[g][0].append(h)
        buckets[g][1].append(r)
    return {g: score(h, r) for g, (h, r) in sorted(buckets.items())}
