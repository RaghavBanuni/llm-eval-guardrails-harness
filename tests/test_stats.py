"""The statistics: what each function is claimed to fix, asserted."""

import numpy as np
import pytest

from llmeval.demo import INCONCLUSIVE_CELLS, REGRESSION_CELLS, paired_outcomes_from_cells
from llmeval.stats import (
    mcnemar,
    paired_bootstrap_ci,
    required_samples_for_difference,
    wilson_interval,
)


def test_wilson_does_not_claim_certainty_at_a_perfect_score():
    """The failure mode the normal approximation has and this does not.

    ``p +/- z*sqrt(p(1-p)/n)`` gives a zero-width interval at 40/40, i.e. "we are certain the
    true rate is exactly 100%" from forty prompts. Wilson keeps a lower bound below 1.
    """
    low, high = wilson_interval(40, 40)
    assert high == pytest.approx(1.0)
    assert 0.85 < low < 0.95


def test_wilson_stays_sensible_at_zero_successes():
    low, high = wilson_interval(0, 40)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < high < 0.15


def test_wilson_narrows_as_the_suite_grows():
    widths = [
        high - low
        for n in (25, 100, 400, 1600)
        for low, high in [wilson_interval(int(0.8 * n), n)]
    ]
    assert widths == sorted(widths, reverse=True)


def test_wilson_interval_covers_the_true_rate_at_the_advertised_rate():
    """Coverage is the only property a confidence interval actually promises."""
    rng = np.random.default_rng(5)
    true_rate, n_trials = 0.8, 60
    covered = 0
    for _ in range(2000):
        successes = int(rng.binomial(n_trials, true_rate))
        low, high = wilson_interval(successes, n_trials, confidence=0.95)
        covered += low <= true_rate <= high
    assert 0.93 <= covered / 2000 <= 0.99


def test_wilson_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(10, 5)
    with pytest.raises(ValueError):
        wilson_interval(1, 5, confidence=1.5)


def test_mcnemar_ignores_concordant_pairs():
    """The core insight, as an invariant.

    Prompts both systems passed, and prompts both failed, carry no information about which is
    better. Inflating those cells must leave the p-value untouched -- which is exactly what an
    unpaired two-proportion test would fail to do.
    """
    small = mcnemar(*paired_outcomes_from_cells(10, 8, 2, 10))
    large = mcnemar(*paired_outcomes_from_cells(900, 8, 2, 900))
    assert small.p_value == pytest.approx(large.p_value)
    assert small.n_discordant == large.n_discordant == 10


def test_mcnemar_reports_no_evidence_when_the_systems_never_disagree():
    """Zero discordant pairs is a statement about the suite, not about the systems."""
    result = mcnemar(*paired_outcomes_from_cells(150, 0, 0, 50))
    assert result.n_discordant == 0
    assert result.p_value == 1.0
    assert result.winner is None
    assert "no evidence" in result.explain()


def test_three_point_gap_on_two_hundred_prompts_is_not_significant():
    """The headline claim: 84% vs 81% at n=200 does not survive a paired test."""
    a, b = paired_outcomes_from_cells(**INCONCLUSIVE_CELLS)
    assert sum(a) / len(a) == pytest.approx(0.84)
    assert sum(b) / len(b) == pytest.approx(0.81)
    result = mcnemar(a, b)
    assert not result.significant
    assert result.winner is None
    assert result.p_value > 0.05


def test_a_real_regression_is_detected_at_the_same_suite_size():
    """A test that never rejects anything would be useless in the other direction."""
    result = mcnemar(*paired_outcomes_from_cells(**REGRESSION_CELLS))
    assert result.significant
    assert result.winner == "a"
    assert result.p_value < 0.001


def test_mcnemar_is_symmetric_under_swapping_the_systems():
    a, b = paired_outcomes_from_cells(100, 15, 4, 30)
    forward = mcnemar(a, b)
    reverse = mcnemar(b, a)
    assert forward.p_value == pytest.approx(reverse.p_value)
    assert forward.winner == "a" and reverse.winner == "b"


def test_mcnemar_requires_aligned_inputs():
    """Unequal lengths mean the pairing is wrong, and wrong pairing is silent otherwise."""
    with pytest.raises(ValueError, match="paired by prompt"):
        mcnemar([True, False, True], [True, False])
    with pytest.raises(ValueError):
        mcnemar([], [])


def test_mcnemar_counts_cells_correctly():
    result = mcnemar(*paired_outcomes_from_cells(7, 5, 3, 2))
    assert (result.both_passed, result.only_a_passed, result.only_b_passed, result.both_failed) == (
        7,
        5,
        3,
        2,
    )
    assert result.n_pairs == 17


def test_required_samples_grows_as_the_difference_shrinks():
    needed = [required_samples_for_difference(0.8, d) for d in (0.15, 0.10, 0.05, 0.02)]
    assert needed == sorted(needed)


def test_detecting_a_small_difference_needs_a_suite_almost_nobody_builds():
    """The sobering number, pinned so it cannot quietly drift.

    Two points from an 80% baseline at 80% power needs thousands of prompts per system. Most
    suites are two orders of magnitude smaller than the question asked of them.
    """
    assert required_samples_for_difference(0.80, 0.02) > 1000
    assert required_samples_for_difference(0.80, 0.05) > 150


def test_more_power_requires_more_samples():
    assert required_samples_for_difference(0.8, 0.05, power=0.9) > required_samples_for_difference(
        0.8, 0.05, power=0.8
    )


def test_required_samples_rejects_impossible_targets():
    with pytest.raises(ValueError):
        required_samples_for_difference(0.95, 0.10)  # would exceed a rate of 1.0
    with pytest.raises(ValueError):
        required_samples_for_difference(0.8, 0.0)


def test_paired_bootstrap_interval_brackets_the_observed_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(4.0, 0.5, 300)
    b = a - 0.3 + rng.normal(0, 0.1, 300)
    mean_difference, low, high = paired_bootstrap_ci(a, b, seed=1)
    assert mean_difference == pytest.approx(0.3, abs=0.05)
    assert low < mean_difference < high
    assert low > 0  # the difference is real, and the interval says so


def test_paired_bootstrap_reports_no_difference_when_there_is_none():
    """Two systems that differ only by noise must produce an interval spanning zero."""
    rng = np.random.default_rng(4)
    a = rng.normal(3.0, 0.8, 200)
    b = a + rng.normal(0.0, 0.05, 200)
    mean_difference, low, high = paired_bootstrap_ci(a, b, seed=3)
    assert abs(mean_difference) < 0.02
    assert low < 0 < high


def test_paired_bootstrap_exploits_pairing():
    """Pairing is the whole reason to run both systems on the same prompts.

    With strongly correlated scores the paired interval must be far tighter than one built by
    resampling columns that no longer correspond, which is what discarding the pairing amounts
    to.
    """
    rng = np.random.default_rng(6)
    a = rng.normal(4.0, 1.5, 400)
    b = a - 0.2 + rng.normal(0, 0.05, 400)

    _, paired_low, paired_high = paired_bootstrap_ci(a, b, seed=8)
    shuffled = np.random.default_rng(9).permutation(b)
    _, naive_low, naive_high = paired_bootstrap_ci(a, shuffled, seed=8)

    assert (paired_high - paired_low) < (naive_high - naive_low) / 5


def test_paired_bootstrap_validates_its_inputs():
    with pytest.raises(ValueError, match="paired by prompt"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [2.0], n_resamples=10)


def test_paired_bootstrap_is_reproducible():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.5, 2.5, 2.0, 4.5, 4.0]
    assert paired_bootstrap_ci(a, b, seed=42) == paired_bootstrap_ci(a, b, seed=42)
