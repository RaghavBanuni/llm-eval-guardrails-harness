"""Validating a judge before believing it.

An LLM judge is a classifier. Nobody would ship a classifier without measuring it
against labelled data, but judges get treated as ground truth constantly, because their
output *looks* like an assessment rather than a prediction.

Why raw agreement is not enough
-------------------------------
Suppose 90% of your eval cases are passes, and the judge simply says "pass" every time.
Raw agreement with human labels: 90%. That number would appear in a slide deck. The
judge has learnt nothing and detects no failure whatsoever -- the only cases anybody
cares about.

Cohen's kappa corrects for exactly this by subtracting the agreement expected from the
marginals alone:

    kappa = (p_observed - p_expected) / (1 - p_expected)

The always-pass judge above scores kappa = 0, correctly reporting no skill beyond chance.
That is why kappa is the headline number here and raw agreement is reported beside it as
context rather than as the metric.

On the interpretation bands
---------------------------
The Landis and Koch (1977) labels -- slight, fair, moderate, substantial, almost perfect
-- are convention, not theory. They were proposed for medical diagnosis agreement and
have no special authority over LLM evaluation. They are included because they are what
readers expect, and flagged as arbitrary because pretending otherwise would be the same
mistake as trusting an unvalidated judge.

One caveat kappa cannot fix: it measures agreement with *your* human labels. If those
labels are inconsistent, a high kappa means the judge reproduces the inconsistency. Human
label quality is upstream of everything here, and measuring inter-annotator agreement
first -- with this same function -- is the way to check it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


def confusion_matrix(
    labels_a: Sequence[Hashable],
    labels_b: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> tuple[np.ndarray, list[Hashable]]:
    """Counts of every ``(a, b)`` label pair, plus the label ordering used."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    if len(labels_a) == 0:
        raise ValueError("need at least one labelled item")

    if labels is None:
        ordered = sorted(set(labels_a) | set(labels_b), key=repr)
    else:
        ordered = list(labels)
        unknown = (set(labels_a) | set(labels_b)) - set(ordered)
        if unknown:
            raise ValueError(f"labels missing from the provided ordering: {sorted(map(repr, unknown))}")

    index = {label: position for position, label in enumerate(ordered)}
    matrix = np.zeros((len(ordered), len(ordered)), dtype=int)
    for a, b in zip(labels_a, labels_b):
        matrix[index[a], index[b]] += 1
    return matrix, ordered


def percent_agreement(labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]) -> float:
    """Fraction of items where the two labellers agree.

    Reported for context only. On an imbalanced set it is inflated and misleading; see the
    module docstring.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    if len(labels_a) == 0:
        raise ValueError("need at least one labelled item")
    return float(np.mean([a == b for a, b in zip(labels_a, labels_b)]))


def cohen_kappa(labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]) -> float:
    """Cohen's kappa: agreement corrected for chance.

    Returns 1.0 for perfect agreement, 0.0 for chance-level agreement, and negative
    values for systematic disagreement (worse than guessing, which usually means a label
    convention is inverted somewhere).

    When both labellers put everything in the same single category, expected agreement is
    1.0 and kappa is undefined -- ``0/0``. This returns 1.0 in that case, since the
    labellers did in fact agree on every item; the degenerate marginals are a property of
    the dataset, not a disagreement.
    """
    matrix, _ = confusion_matrix(labels_a, labels_b)
    total = matrix.sum()
    observed = np.trace(matrix) / total

    row_marginals = matrix.sum(axis=1) / total
    column_marginals = matrix.sum(axis=0) / total
    expected = float(np.sum(row_marginals * column_marginals))

    if expected >= 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def interpret_kappa(kappa: float) -> str:
    """Landis and Koch (1977) band. Convention, not theory -- see the module docstring."""
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


@dataclass(frozen=True)
class JudgeValidation:
    """How well a judge reproduces human labels."""

    n_items: int
    kappa: float
    percent_agreement: float
    interpretation: str
    minimum_kappa: float

    @property
    def trustworthy(self) -> bool:
        """Whether the judge cleared the agreement bar set for this use.

        The bar is a parameter, not a constant, because it depends on what the judge
        gates. A judge feeding an exploratory dashboard can be weaker than one blocking a
        release.
        """
        return self.kappa >= self.minimum_kappa

    def explain(self) -> str:
        base = (
            f"kappa = {self.kappa:.3f} ({self.interpretation}) against {self.n_items} "
            f"human labels; raw agreement {self.percent_agreement:.1%}."
        )
        if self.trustworthy:
            return base + (
                f" Clears the required kappa of {self.minimum_kappa}, so judge scores can "
                "stand in for human labels on this task -- for this task only."
            )
        return base + (
            f" Below the required kappa of {self.minimum_kappa}. Judge scores should not "
            "be reported as if they were human labels here. Either revise the rubric "
            "(most disagreement traces to ambiguous criteria rather than model "
            "capability), or fall back to deterministic checks and human review."
        )


def validate_judge(
    judge_labels: Sequence[Hashable],
    human_labels: Sequence[Hashable],
    minimum_kappa: float = 0.6,
) -> JudgeValidation:
    """Measure a judge against human labels. Run this before trusting any judge output."""
    kappa = cohen_kappa(judge_labels, human_labels)
    return JudgeValidation(
        n_items=len(judge_labels),
        kappa=kappa,
        percent_agreement=percent_agreement(judge_labels, human_labels),
        interpretation=interpret_kappa(kappa),
        minimum_kappa=minimum_kappa,
    )
