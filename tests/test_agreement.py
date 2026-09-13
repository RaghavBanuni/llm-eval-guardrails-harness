"""Judge validation: why kappa is the headline number and agreement is not."""

import pytest

from llmeval.agreement import (
    cohen_kappa,
    confusion_matrix,
    interpret_kappa,
    percent_agreement,
    validate_judge,
)
from llmeval.demo import always_pass_judge_labels, useful_judge_labels


def test_always_pass_judge_looks_good_on_agreement_and_scores_zero_kappa():
    """The central argument of the module, as a test.

    A judge that says "pass" to everything agrees with humans on 90% of an imbalanced set and
    detects exactly zero failures. Raw agreement rewards it; kappa does not.
    """
    judge, human = always_pass_judge_labels(n_items=200, true_fail_rate=0.10)
    assert percent_agreement(judge, human) > 0.85
    assert cohen_kappa(judge, human) == pytest.approx(0.0, abs=1e-9)


def test_a_judge_that_tracks_humans_scores_well():
    judge, human = useful_judge_labels(n_items=300, judge_error_rate=0.05)
    kappa = cohen_kappa(judge, human)
    assert kappa > 0.8
    assert interpret_kappa(kappa) in ("substantial", "almost perfect")


def test_perfect_agreement_is_one():
    labels = ["pass", "fail", "pass", "fail", "pass"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)
    assert percent_agreement(labels, labels) == 1.0


def test_systematic_disagreement_is_negative():
    """Worse than chance, which in practice means an inverted label convention."""
    human = ["pass", "fail"] * 20
    judge = ["fail", "pass"] * 20
    assert cohen_kappa(judge, human) < 0


def test_degenerate_single_class_returns_one_rather_than_dividing_by_zero():
    """Expected agreement is 1.0 here, so kappa is 0/0.

    The labellers did agree on every item; the degenerate marginals are a property of the
    dataset, not a disagreement, so 1.0 is the defensible answer and a crash is not.
    """
    labels = ["pass"] * 30
    assert cohen_kappa(labels, labels) == 1.0


def test_kappa_is_symmetric():
    judge, human = useful_judge_labels(n_items=150)
    assert cohen_kappa(judge, human) == pytest.approx(cohen_kappa(human, judge))


def test_kappa_handles_more_than_two_labels():
    a = ["good", "bad", "unsure", "good", "bad", "unsure"] * 5
    b = ["good", "bad", "good", "good", "bad", "unsure"] * 5
    kappa = cohen_kappa(a, b)
    assert 0.5 < kappa < 1.0


def test_confusion_matrix_counts_and_orders_labels():
    matrix, labels = confusion_matrix(["pass", "pass", "fail"], ["pass", "fail", "fail"])
    assert labels == ["fail", "pass"]
    assert matrix.sum() == 3
    assert matrix[labels.index("pass"), labels.index("fail")] == 1


def test_confusion_matrix_rejects_a_label_ordering_that_omits_observed_labels():
    with pytest.raises(ValueError, match="missing from the provided ordering"):
        confusion_matrix(["pass", "skip"], ["pass", "pass"], labels=["pass", "fail"])


def test_agreement_functions_require_aligned_non_empty_input():
    with pytest.raises(ValueError):
        cohen_kappa(["pass"], ["pass", "fail"])
    with pytest.raises(ValueError):
        percent_agreement([], [])


def test_validation_gate_reflects_the_required_kappa():
    """The bar is a parameter because it depends on what the judge gates."""
    judge, human = always_pass_judge_labels(n_items=100)
    weak = validate_judge(judge, human, minimum_kappa=0.6)
    assert not weak.trustworthy
    assert "should not" in weak.explain()

    judge, human = useful_judge_labels(n_items=200, judge_error_rate=0.05)
    strong = validate_judge(judge, human, minimum_kappa=0.6)
    assert strong.trustworthy
    assert "for this task only" in strong.explain()


def test_validation_reports_the_sample_size_it_rests_on():
    judge, human = useful_judge_labels(n_items=64)
    validation = validate_judge(judge, human)
    assert validation.n_items == 64
    assert "64" in validation.explain()


def test_interpretation_bands_are_ordered():
    assert interpret_kappa(-0.2) == "worse than chance"
    assert interpret_kappa(0.1) == "slight"
    assert interpret_kappa(0.3) == "fair"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.7) == "substantial"
    assert interpret_kappa(0.95) == "almost perfect"
