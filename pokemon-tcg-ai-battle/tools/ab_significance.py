#!/usr/bin/env python3
"""Turn an A/B win count into a real significance verdict.

Every ship/reject decision in this project has been made by eyeballing a
self-play win rate against a hand-calibrated "43-57% is just noise" band
(see STRATEGY_REPORT.md). That band was calibrated from small batches and
is badly miscalibrated for the batch sizes actually used since: under the
null hypothesis (the change does nothing, so each game is a fair coin) the
95% interval is 40-60% at 300 games but only 48.4-51.6% at 3,600. Judging a
3,600-game result against a +-7% band throws away most of the resolution
those 3,600 games bought, and can call a genuine effect "noise".

This computes the two numbers that actually answer "should we ship it":
  - a Wilson score interval for the observed win rate, and
  - an exact two-tailed binomial p-value against p=0.5.

Draws are excluded from the trial count rather than counted as half a win:
this engine produces them very rarely (0 in most batches measured here) and
excluding them keeps the test a clean Bernoulli comparison.

Usage:
    python tools/ab_significance.py WINS LOSSES [--alpha 0.05]
    python tools/ab_significance.py 326 274
    python tools/ab_significance.py --pooled 326,274 294,306 301,299
"""

import argparse
import math


def wilson_interval(wins, n, z=1.96):
    """Wilson score interval -- behaves sensibly near 0/1 and at small n,
    unlike the textbook normal approximation to the binomial."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_binom_pmf(k, n, p0):
    """log P(X=k) for X~Binomial(n, p0), via lgamma. Computed in log space
    because the direct form overflows: math.comb(3600, 1800) is a 1084-digit
    integer, and multiplying it by an underflowing 0.5**1800 raises
    OverflowError rather than yielding the (perfectly ordinary) result."""
    if k < 0 or k > n:
        return -math.inf
    log_choose = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    return log_choose + k * math.log(p0) + (n - k) * math.log(1 - p0)


def binomial_test_two_tailed(wins, n, p0=0.5):
    """Exact two-tailed binomial test: total probability of any outcome at
    least as extreme (in either direction) as the one observed. Exact rather
    than a normal approximation because the interesting results here sit near
    the decision boundary, where the approximation is least trustworthy."""
    if n == 0:
        return 1.0
    log_observed = _log_binom_pmf(wins, n, p0)
    # Sum every outcome whose probability is <= the observed one. Compare in
    # log space with a small additive slack so the mirror-image outcome of a
    # symmetric case isn't dropped by floating-point rounding.
    tol = log_observed + 1e-9
    total = 0.0
    for k in range(n + 1):
        lp = _log_binom_pmf(k, n, p0)
        if lp <= tol:
            total += math.exp(lp)
    return min(1.0, total)


def report(wins, losses, alpha=0.05, label=None):
    n = wins + losses
    rate = wins / n if n else 0.0
    lo, hi = wilson_interval(wins, n)
    pv = binomial_test_two_tailed(wins, n)
    verdict = "SIGNIFICANT" if pv < alpha else "not distinguishable from noise"
    direction = "improvement" if rate > 0.5 else "regression"
    if label:
        print(f"== {label} ==")
    print(f"  {wins}W-{losses}L / {n} games -> {rate:.2%}")
    print(f"  Wilson 95% CI: {lo:.2%} - {hi:.2%}")
    print(f"  exact two-tailed binomial p = {pv:.4f} (alpha={alpha})")
    print(f"  verdict: {verdict}" + (f" ({direction})" if pv < alpha else ""))
    # The minimum effect this batch size could have resolved, so a null
    # result can be read as "too small to detect" rather than "no effect".
    if n:
        resolvable = 1.96 * 0.5 / math.sqrt(n)
        print(f"  smallest effect {n} games can resolve at 95%: +-{resolvable:.2%} "
              f"(i.e. needs to clear {50 + resolvable*100:.1f}%)")
    return pv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wins", nargs="?", type=int)
    ap.add_argument("losses", nargs="?", type=int)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--pooled", nargs="+", metavar="W,L",
                    help="several independent runs as W,L pairs -- reports each "
                         "run and then the pooled total")
    args = ap.parse_args()

    if args.pooled:
        tw = tl = 0
        for i, pair in enumerate(args.pooled, 1):
            w, l = (int(x) for x in pair.split(","))
            report(w, l, args.alpha, label=f"run {i}")
            print()
            tw += w
            tl += l
        report(tw, tl, args.alpha, label=f"pooled ({len(args.pooled)} runs)")
        return 0

    if args.wins is None or args.losses is None:
        ap.error("give WINS and LOSSES, or --pooled W,L W,L ...")
    report(args.wins, args.losses, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
