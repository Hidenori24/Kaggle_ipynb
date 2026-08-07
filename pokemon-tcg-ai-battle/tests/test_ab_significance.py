"""Unit tests for `tools/ab_significance.py`.

This is pure statistics with no engine dependency, but it's worth pinning:
it now decides whether an A/B result ships, and the exact binomial test had
a real overflow bug in its first form (`math.comb(3600, 1800)` is a 1084-digit
integer; multiplying it by an underflowing `0.5**1800` raises OverflowError
rather than returning the ordinary answer). These tests cover the invariants
that catch that class of mistake -- and the batch sizes this project actually
runs, where the bug appeared.
"""

import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ab_significance import (  # noqa: E402
    binomial_test_two_tailed,
    wilson_interval,
)


# --- binomial_test_two_tailed ---------------------------------------------

def test_perfectly_balanced_result_is_maximally_unsurprising():
    assert binomial_test_two_tailed(50, 100) == pytest.approx(1.0)


def test_extreme_result_is_vanishingly_unlikely():
    assert binomial_test_two_tailed(90, 100) < 1e-15


def test_two_tailed_test_is_symmetric():
    # An equal-sized win and loss skew must be equally surprising, or the
    # test would silently favour one direction of change over the other.
    assert binomial_test_two_tailed(326, 600) == pytest.approx(binomial_test_two_tailed(274, 600))
    assert binomial_test_two_tailed(850, 1800) == pytest.approx(binomial_test_two_tailed(950, 1800))


def test_p_value_decreases_as_the_effect_grows():
    assert (binomial_test_two_tailed(310, 600)
            > binomial_test_two_tailed(326, 600)
            > binomial_test_two_tailed(360, 600))


@pytest.mark.parametrize("wins,n", [(1800, 3600), (1863, 3600), (1542, 3000), (2700, 5400)])
def test_no_overflow_at_the_batch_sizes_this_project_uses(wins, n):
    # The direct math.comb form raised OverflowError for exactly these
    # sizes; the log-space implementation must return a real probability.
    p = binomial_test_two_tailed(wins, n)
    assert 0.0 <= p <= 1.0
    assert math.isfinite(p)


def test_matches_the_normal_approximation_closely_at_large_n():
    # Not a tautology -- it's an independent cross-check that the exact
    # log-space sum agrees with the textbook approximation where the
    # approximation is known to be good.
    for wins, n in [(326, 600), (1863, 3600), (850, 1800)]:
        z = (wins / n - 0.5) / (0.5 / math.sqrt(n))
        approx = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        assert binomial_test_two_tailed(wins, n) == pytest.approx(approx, abs=0.005)


def test_zero_games_is_not_a_division_by_zero():
    assert binomial_test_two_tailed(0, 0) == 1.0


# --- wilson_interval ------------------------------------------------------

def test_wilson_interval_brackets_the_observed_rate():
    lo, hi = wilson_interval(326, 600)
    assert lo < 326 / 600 < hi


def test_wilson_interval_narrows_as_games_accumulate():
    # The whole point of the fix this tool represents: more games must buy
    # more resolution. A 50% result at 600 games and at 3,600 games are very
    # different pieces of evidence, which a fixed 43-57% band cannot express.
    lo_small, hi_small = wilson_interval(300, 600)
    lo_big, hi_big = wilson_interval(1800, 3600)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_interval_stays_inside_zero_and_one_at_the_extremes():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 <= hi <= 1.0
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.0 <= lo <= 1.0


def test_wilson_interval_with_no_games_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)
