"""Significance testing for fine-tuned vs zero-shot WER.

Answers: is the WER gap real, or could it plausibly arise by chance from
this particular set of test utterances?

Three complementary tests on the *same paired* test set:
  1. Bootstrap CI on the WER difference (resamples utterances with
     replacement, keeping errors and ref-length together -> respects the
     fact that corpus WER is a ratio of sums, not a mean of rates).
  2. Exact paired permutation test (randomly swaps which system each
     utterance's result is attributed to).
  3. Utterance-level sign test / Wilcoxon-style win-loss count.

Usage:
    python -m s2t.evaluation.significance_test \
        --baseline reports/zeroshot_medium_en.json \
        --system   reports/finetuned_medium_en_lora.json \
        --split    test
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import jiwer
import numpy as np

from s2t.paths import PROJECT_ROOT


def load_samples(path: Path, split: str) -> Dict[str, dict]:
    report = json.loads(Path(path).read_text())
    if split not in report.get("splits", {}):
        raise SystemExit(f"{path}: split '{split}' not found (has {list(report['splits'])})")
    samples = report["splits"][split]["samples"]
    by_id = {s["utt_id"]: s for s in samples}
    if len(by_id) != len(samples):
        raise SystemExit(f"{path}: duplicate utt_id in split '{split}'")
    return by_id


def error_counts(ref: str, hyp: str) -> Tuple[int, int]:
    """Return (edit_distance, n_reference_words) for one utterance.

    Mirrors common.corpus_wer's guard: an empty string becomes "empty" so
    the alignment never sees a zero-length sequence.
    """
    ref = ref if ref.strip() else "empty"
    hyp = hyp if hyp.strip() else "empty"
    out = jiwer.process_words([ref], [hyp])
    errors = out.substitutions + out.deletions + out.insertions
    n_ref = len(ref.split())
    return errors, n_ref


def corpus_wer(errors: np.ndarray, lengths: np.ndarray) -> float:
    return float(errors.sum() / lengths.sum())


def bootstrap(
    e_a: np.ndarray, e_b: np.ndarray, n_ref: np.ndarray, n_boot: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Paired utterance bootstrap. Returns (diffs, rel_reductions) in percent."""
    n = len(n_ref)
    idx = rng.integers(0, n, size=(n_boot, n))
    len_b = n_ref[idx].sum(axis=1)
    wer_a = e_a[idx].sum(axis=1) / len_b
    wer_b = e_b[idx].sum(axis=1) / len_b
    diff = (wer_a - wer_b) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(wer_a > 0, (wer_a - wer_b) / wer_a * 100.0, np.nan)
    return diff, rel


def permutation_test(
    e_a: np.ndarray, e_b: np.ndarray, n_ref: np.ndarray, n_perm: int, rng: np.random.Generator
) -> Tuple[float, float]:
    """Paired permutation test on the corpus-WER difference.

    Under H0 (systems are interchangeable) each utterance's two results
    could equally well have come from either system, so we randomly swap
    them within each pair. Two-sided p, with the +1 correction that keeps
    the p-value valid for a sampled (rather than exhaustive) permutation set.
    """
    observed = abs(corpus_wer(e_a, n_ref) - corpus_wer(e_b, n_ref))
    total_len = n_ref.sum()
    n = len(n_ref)
    count = 0
    batch = 2000
    done = 0
    while done < n_perm:
        k = min(batch, n_perm - done)
        swap = rng.random((k, n)) < 0.5
        perm_a = np.where(swap, e_b, e_a).sum(axis=1)
        perm_b = np.where(swap, e_a, e_b).sum(axis=1)
        stat = np.abs(perm_a - perm_b) / total_len
        count += int((stat >= observed - 1e-12).sum())
        done += k
    return (count + 1) / (n_perm + 1), observed * 100.0


def binom_two_sided(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value (ties excluded), no scipy needed."""
    n = wins + losses
    if n == 0:
        return 1.0
    def pmf(k: int) -> float:
        return math.comb(n, k) * 0.5 ** n
    observed = pmf(wins)
    return min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= observed + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="zero-shot report JSON")
    ap.add_argument("--system", required=True, help="fine-tuned report JSON")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--by-speaker", action="store_true", help="also report per-speaker breakdown")
    ap.add_argument("--json-out", help="write machine-readable results here")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    base = load_samples(PROJECT_ROOT / args.baseline, args.split)
    syst = load_samples(PROJECT_ROOT / args.system, args.split)

    shared = sorted(set(base) & set(syst))
    if not shared:
        raise SystemExit("no overlapping utt_id between the two reports")
    dropped = (set(base) | set(syst)) - set(shared)
    if dropped:
        print(f"WARNING: {len(dropped)} utterance(s) not present in both reports; excluded.")

    ref_mismatch = [u for u in shared if base[u]["ref_norm"] != syst[u]["ref_norm"]]
    if ref_mismatch:
        raise SystemExit(
            f"{len(ref_mismatch)} utterance(s) have different references in the two "
            f"reports (e.g. {ref_mismatch[0]}); the pairing would be invalid."
        )

    e_a, e_b, lengths, speakers = [], [], [], []
    for u in shared:
        ref = base[u]["ref_norm"]
        ea, n_ref = error_counts(ref, base[u]["hyp_norm"])
        eb, _ = error_counts(ref, syst[u]["hyp_norm"])
        e_a.append(ea)
        e_b.append(eb)
        lengths.append(n_ref)
        speakers.append(base[u].get("speaker", "?"))

    e_a = np.array(e_a, dtype=float)
    e_b = np.array(e_b, dtype=float)
    lengths = np.array(lengths, dtype=float)
    speakers = np.array(speakers)

    wer_a = corpus_wer(e_a, lengths)
    wer_b = corpus_wer(e_b, lengths)
    abs_diff = (wer_a - wer_b) * 100.0
    rel_diff = (wer_a - wer_b) / wer_a * 100.0 if wer_a > 0 else float("nan")

    lo_q, hi_q = args.alpha / 2 * 100, (1 - args.alpha / 2) * 100
    diffs, rels = bootstrap(e_a, e_b, lengths, args.n_boot, rng)
    ci_lo, ci_hi = np.percentile(diffs, [lo_q, hi_q])
    rci_lo, rci_hi = np.nanpercentile(rels, [lo_q, hi_q])
    p_boot = float((diffs <= 0).mean())

    p_perm, obs_stat = permutation_test(e_a, e_b, lengths, args.n_perm, rng)

    better = int((e_b < e_a).sum())
    worse = int((e_b > e_a).sum())
    tied = int((e_b == e_a).sum())
    p_sign = binom_two_sided(better, worse)

    conf = int(round((1 - args.alpha) * 100))
    print()
    print("=" * 72)
    print(f"  Fine-tuned vs zero-shot - split '{args.split}', {len(shared)} paired utterances")
    print("=" * 72)
    print(f"  baseline : {args.baseline}")
    print(f"  system   : {args.system}")
    print(f"  reference words: {int(lengths.sum())}")
    print()
    print(f"  Zero-shot  corpus WER : {wer_a * 100:6.2f}%   ({int(e_a.sum())} word errors)")
    print(f"  Fine-tuned corpus WER : {wer_b * 100:6.2f}%   ({int(e_b.sum())} word errors)")
    print(f"  Absolute improvement  : {abs_diff:6.2f} pp")
    print(f"  Relative reduction    : {rel_diff:6.2f}%")
    print()
    print(f"  [1] Paired bootstrap ({args.n_boot:,} resamples)")
    print(f"      {conf}% CI, absolute : [{ci_lo:6.2f}, {ci_hi:6.2f}] pp")
    print(f"      {conf}% CI, relative : [{rci_lo:6.2f}, {rci_hi:6.2f}] %")
    print(f"      P(fine-tuned no better) = {p_boot:.4f}")
    print()
    print(f"  [2] Paired permutation test ({args.n_perm:,} shuffles, two-sided)")
    print(f"      observed |WER gap| = {obs_stat:.2f} pp")
    print(f"      p = {p_perm:.4f}")
    print()
    print(f"  [3] Utterance-level sign test")
    print(f"      fine-tuned better on {better}, worse on {worse}, tied on {tied}")
    print(f"      p = {p_sign:.4g} (two-sided exact binomial, ties excluded)")
    print()
    verdict = (
        "significant - the gap is very unlikely to be sampling noise"
        if p_perm < args.alpha and ci_lo > 0
        else "NOT significant at this alpha - cannot rule out chance"
    )
    print(f"  Verdict (alpha={args.alpha}): {verdict}")
    print("=" * 72)

    if args.by_speaker:
        print()
        print(f"  Per-speaker breakdown (bootstrap {conf}% CI on absolute gap)")
        print(f"  {'speaker':<10} {'n':>4} {'zero-shot':>10} {'fine-tuned':>11} {'gap':>8}  {str(conf)+'% CI':>18}")
        for spk in sorted(set(speakers)):
            m = speakers == spk
            if m.sum() < 2:
                continue
            s_wer_a = corpus_wer(e_a[m], lengths[m]) * 100
            s_wer_b = corpus_wer(e_b[m], lengths[m]) * 100
            s_diffs, _ = bootstrap(e_a[m], e_b[m], lengths[m], args.n_boot, rng)
            s_lo, s_hi = np.percentile(s_diffs, [lo_q, hi_q])
            flag = "" if s_lo > 0 else "   (CI crosses 0)"
            print(
                f"  {spk:<10} {int(m.sum()):>4} {s_wer_a:>9.2f}% {s_wer_b:>10.2f}% "
                f"{s_wer_a - s_wer_b:>7.2f}p  [{s_lo:6.2f},{s_hi:6.2f}]{flag}"
            )

    if args.json_out:
        out = {
            "baseline": args.baseline,
            "system": args.system,
            "split": args.split,
            "n_utterances": len(shared),
            "n_ref_words": int(lengths.sum()),
            "wer_baseline_pct": round(wer_a * 100, 4),
            "wer_system_pct": round(wer_b * 100, 4),
            "absolute_gap_pp": round(abs_diff, 4),
            "relative_reduction_pct": round(rel_diff, 4),
            "bootstrap": {
                "n_resamples": args.n_boot,
                "ci_absolute_pp": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
                "ci_relative_pct": [round(float(rci_lo), 4), round(float(rci_hi), 4)],
                "p_no_improvement": round(p_boot, 6),
            },
            "permutation": {"n_shuffles": args.n_perm, "p_two_sided": round(p_perm, 6)},
            "sign_test": {"better": better, "worse": worse, "tied": tied, "p_two_sided": p_sign},
            "alpha": args.alpha,
            "seed": args.seed,
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
