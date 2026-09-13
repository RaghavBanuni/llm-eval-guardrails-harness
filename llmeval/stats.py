"""The statistics an eval report needs and usually omits.

"84% versus 81% on 200 prompts" is not a result. It is two numbers with no statement
about whether the difference would survive a different 200 prompts, and on a suite that
size the answer is usually no.

Three tools, each for a specific question.

**How precise is a single pass rate?** :func:`wilson_interval`. The textbook normal
approximation ``p +/- z*sqrt(p(1-p)/n)`` breaks exactly where eval suites live: at 0 or
100% it returns an interval of zero width, claiming perfect certainty from 40 prompts.
The Wilson score interval stays sensible there, which is why it is the default here
rather than a footnote.

**Did model B beat model A?** :func:`mcnemar`. The two models were run on the *same*
prompts, so the samples are paired and an unpaired two-proportion test throws away that
structure. McNemar's insight is that prompts both models got right, and prompts both got
wrong, carry *no information about which is better* -- only the disagreements do. If A
passed 170/200 and B passed 162/200, what matters is not the eight-point gap but how many
prompts flipped in each direction. Twenty flips one way and twelve the other is a very
different result from four and zero, even though both give the same aggregate gap.

**How big a difference could this suite even detect?** :func:`required_samples_for_difference`.
Worth running before building the suite, because the usual answer is uncomfortable: a
five-point difference near an 80% pass rate needs well over a thousand paired prompts at
conventional power. Most eval suites are too small to detect the differences their owners
argue about, and that is better known in advance than discovered in a disagreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


def wilson_interval(
    successes: int,
    n_trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a pass rate.

    Derived by inverting the score test rather than assuming normality of ``p_hat``:

        centre = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z*sqrt( p(1-p)/n + z^2/(4n^2) ) / (1 + z^2/n)

    At ``successes = n_trials`` the upper bound is 1 but the lower bound is strictly less
    than 1, which is the behaviour the normal approximation gets wrong.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if not 0 <= successes <= n_trials:
        raise ValueError("successes must be in [0, n_trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = successes / n_trials
    denominator = 1.0 + z**2 / n_trials
    centre = (p + z**2 / (2 * n_trials)) / denominator
    half_width = (
        z * math.sqrt(p * (1.0 - p) / n_trials + z**2 / (4 * n_trials**2)) / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of a paired comparison between two systems.

    ``only_a_passed`` and ``only_b_passed`` are the discordant counts -- the only cells
    that carry information. ``both_passed`` and ``both_failed`` are reported for context
    but do not enter the test.
    """

    both_passed: int
    only_a_passed: int
    only_b_passed: int
    both_failed: int
    p_value: float
    alpha: float

    @property
    def n_pairs(self) -> int:
        return self.both_passed + self.only_a_passed + self.only_b_passed + self.both_failed

    @property
    def n_discordant(self) -> int:
        return self.only_a_passed + self.only_b_passed

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def winner(self) -> str | None:
        if not self.significant:
            return None
        return "a" if self.only_a_passed > self.only_b_passed else "b"

    def explain(self) -> str:
        if self.n_discordant == 0:
            return (
                "The two systems agreed on every prompt, so this suite contains no "
                "evidence about which is better. That is a statement about the suite, "
                "not about the systems."
            )
        if not self.significant:
            return (
                f"{self.n_discordant} prompts disagreed "
                f"({self.only_a_passed} favouring A, {self.only_b_passed} favouring B), "
                f"p = {self.p_value:.4f}. Not distinguishable at alpha = {self.alpha}. "
                "Shipping either on this evidence is a coin flip dressed as a decision."
            )
        return (
            f"System {self.winner.upper()} wins: of {self.n_discordant} disagreements, "
            f"{max(self.only_a_passed, self.only_b_passed)} favour it, "
            f"p = {self.p_value:.4f} < {self.alpha}."
        )


def mcnemar(
    a_passed: Sequence[bool],
    b_passed: Sequence[bool],
    alpha: float = 0.05,
) -> McNemarResult:
    """Exact McNemar test on paired pass/fail outcomes.

    Under the null hypothesis that the two systems are equally likely to be the one that
    succeeds when they disagree, the discordant pairs are ``Binomial(n_discordant, 0.5)``.
    The exact binomial test is used rather than the chi-square approximation with
    continuity correction, because eval suites routinely produce fewer than 25 discordant
    pairs -- precisely where the approximation is unreliable.

    Inputs must be aligned: element ``i`` of both sequences must refer to the same prompt.
    """
    if len(a_passed) != len(b_passed):
        raise ValueError("a_passed and b_passed must be the same length (paired by prompt)")
    if len(a_passed) == 0:
        raise ValueError("need at least one pair")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    both_passed = only_a = only_b = both_failed = 0
    for a, b in zip(a_passed, b_passed):
        a, b = bool(a), bool(b)
        if a and b:
            both_passed += 1
        elif a and not b:
            only_a += 1
        elif b and not a:
            only_b += 1
        else:
            both_failed += 1

    n_discordant = only_a + only_b
    if n_discordant == 0:
        # No information either way. Reporting p = 1.0 is the honest encoding: the data
        # give no reason to prefer either system.
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(only_a, n_discordant, 0.5).pvalue)

    return McNemarResult(
        both_passed=both_passed,
        only_a_passed=only_a,
        only_b_passed=only_b,
        both_failed=both_failed,
        p_value=p_value,
        alpha=alpha,
    )


def paired_bootstrap_ci(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean difference on continuous scores.

    Returns ``(mean_difference, lower, upper)`` for ``a - b``.

    Resampling is done over *prompts*, keeping each prompt's pair of scores together.
    Resampling the two score vectors independently would destroy the pairing and inflate
    the interval, discarding the variance reduction that running both systems on the same
    prompts was meant to buy.

    Use this for graded scores (a 1-5 rubric, a similarity metric). For pass/fail use
    :func:`mcnemar`, which is exact and needs no resampling.
    """
    a = np.asarray(a_scores, dtype=float)
    b = np.asarray(b_scores, dtype=float)
    if a.size != b.size:
        raise ValueError("a_scores and b_scores must be the same length (paired by prompt)")
    if a.size == 0:
        raise ValueError("need at least one pair")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")

    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    resampled_means = differences[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower = float(np.quantile(resampled_means, tail))
    upper = float(np.quantile(resampled_means, 1.0 - tail))
    return float(differences.mean()), lower, upper


def required_samples_for_difference(
    baseline_rate: float,
    absolute_difference: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Prompts per system needed to detect a pass-rate difference.

    This is the unpaired two-proportion calculation, which is *conservative* for a paired
    suite: pairing removes prompt-difficulty variance, so the same power is reachable with
    fewer prompts. How many fewer depends on the correlation between the systems, which is
    not known in advance -- so the conservative number is the one reported rather than an
    optimistic estimate resting on a guess.

    The purpose is to be run *before* writing the suite. The answers are usually sobering.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if absolute_difference <= 0:
        raise ValueError("absolute_difference must be positive")
    treatment_rate = baseline_rate + absolute_difference
    if not 0.0 < treatment_rate < 1.0:
        raise ValueError("baseline_rate + absolute_difference must stay inside (0, 1)")

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(stats.norm.ppf(power))
    pooled = (baseline_rate + treatment_rate) / 2.0
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta
        * math.sqrt(
            baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2
    return int(math.ceil(numerator / absolute_difference**2))
