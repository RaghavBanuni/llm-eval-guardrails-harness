"""The runner: error accounting, scoping, and paired comparison."""

import pytest

from llmeval import demo
from llmeval.checks import ContainsCheck, JsonSchemaCheck, MaxLengthCheck, NumericToleranceCheck
from llmeval.cli import build_extraction_checks
from llmeval.runner import SuiteResult, compare_suites, run_suite, scoped
from llmeval.types import CaseResult, CheckOutcome, Prediction, Sample


def _suite_from_flags(name: str, flags: dict[str, bool]) -> SuiteResult:
    """Build a SuiteResult directly, for testing comparison logic without a model."""
    cases = []
    for sample_id, passed in flags.items():
        outcome = CheckOutcome("stub", passed, "", "error")
        cases.append(CaseResult(sample_id, Prediction(sample_id, "out"), (outcome,), ("stub",)))
    return SuiteResult(name=name, cases=tuple(cases))


def test_baseline_and_candidate_pass_counts_are_what_the_fixture_says():
    checks = build_extraction_checks()
    baseline = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="baseline")
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="candidate")
    assert (baseline.n_passed, baseline.n_cases) == (9, 10)
    assert (candidate.n_passed, candidate.n_cases) == (4, 10)


def test_a_model_error_is_a_failure_not_a_skipped_case():
    """The accounting decision that keeps the pass rate honest.

    Dropping errored calls computes the rate over a biased subset -- the prompts that time out
    are the long, complex, adversarial ones.
    """
    checks = build_extraction_checks()
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="candidate")
    timed_out = next(case for case in candidate.cases if case.sample_id == "tk-005")
    assert timed_out.prediction.failed
    assert "TimeoutError" in timed_out.prediction.error
    assert not timed_out.passed
    assert candidate.n_errors == 1
    assert candidate.n_cases == 10  # the case is still counted in the denominator


def test_checks_are_not_run_on_a_case_whose_model_call_failed():
    """Checking an empty string would produce misleading per-check failure counts."""
    checks = build_extraction_checks()
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks)
    timed_out = next(case for case in candidate.cases if case.sample_id == "tk-005")
    assert [outcome.name for outcome in timed_out.outcomes] == ["model_call"]


def test_scoping_keeps_irrelevant_checks_off_a_case():
    """A schema check has no business running on an arithmetic prompt."""
    checks = build_extraction_checks()
    baseline = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES, checks)
    arithmetic = next(case for case in baseline.cases if case.sample_id == "ar-002")
    names = {outcome.name for outcome in arithmetic.outcomes}
    assert "numeric_answer" in names
    assert "ticket_schema" not in names


def test_opposite_refusal_expectations_coexist_in_one_suite():
    """Scoping is what lets the benign and harmful prompts demand opposite behaviour."""
    checks = build_extraction_checks()
    baseline = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES, checks)
    benign = next(case for case in baseline.cases if case.sample_id == "rf-001")
    harmful = next(case for case in baseline.cases if case.sample_id == "rf-002")
    assert "no_false_refusal" in {o.name for o in benign.outcomes}
    assert "refuses_harmful" in {o.name for o in harmful.outcomes}
    assert benign.passed and harmful.passed


def test_a_sample_no_check_applies_to_fails_rather_than_passing_silently():
    """An unevaluated case counted as a pass is the quietest way an eval can lie."""
    samples = [Sample("orphan", "prompt", tags=("unmatched",))]
    result = run_suite(lambda p: "anything", samples, [scoped(ContainsCheck("x"), "json")])
    case = result.cases[0]
    assert not case.passed
    assert case.failed_checks == ("no_applicable_check",)
    assert "unevaluated" in case.outcomes[0].detail


def test_a_failing_warning_does_not_fail_the_case():
    """Advisory signals must be recordable without gating a release."""
    samples = [Sample("s1", "prompt", expected="96")]
    checks = [NumericToleranceCheck(name="numeric"), MaxLengthCheck(5, name="length")]
    result = run_suite(lambda p: "96 minutes exactly", samples, checks)
    case = result.cases[0]
    assert case.passed
    assert case.warnings == ("length",)
    assert "length" in case.failed_checks  # recorded, just not fatal


def test_duplicate_sample_ids_are_rejected():
    """Duplicate ids would silently corrupt the id-keyed pairing in compare_suites."""
    samples = [Sample("same", "a"), Sample("same", "b")]
    with pytest.raises(ValueError, match="duplicate sample id"):
        run_suite(lambda p: "x", samples, [ContainsCheck("x")])


def test_empty_suite_or_empty_checks_are_rejected():
    with pytest.raises(ValueError, match="at least one check"):
        run_suite(lambda p: "x", [Sample("s", "p")], [])
    with pytest.raises(ValueError, match="at least one sample"):
        run_suite(lambda p: "x", [], [ContainsCheck("x")])


def test_scoped_requires_at_least_one_tag():
    with pytest.raises(ValueError, match="at least one tag"):
        scoped(ContainsCheck("x"))


def test_tag_breakdown_surfaces_a_capability_that_collapsed():
    """The aggregate hides this; the per-tag view is where a regression is visible."""
    checks = build_extraction_checks()
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks)
    by_tag = candidate.by_tag()
    assert by_tag["json"] == (1, 5)  # four of five extraction cases broken
    assert by_tag["arithmetic"] == (2, 2)  # arithmetic is fine, and better than baseline


def test_failures_by_check_is_ordered_for_triage():
    checks = build_extraction_checks()
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks)
    failures = candidate.failures_by_check()
    assert failures["ticket_schema"] == 3
    assert failures["no_pii_in_output"] == 1
    assert list(failures.values()) == sorted(failures.values(), reverse=True)


def test_pass_rate_interval_is_wide_on_a_ten_prompt_suite():
    """Ten prompts cannot pin a pass rate, and the interval says so out loud."""
    checks = build_extraction_checks()
    baseline = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES, checks)
    low, high = baseline.pass_rate_interval()
    assert low < baseline.pass_rate < high
    assert high - low > 0.25


def test_cost_and_latency_are_aggregated():
    samples = [Sample(f"s{i}", "prompt text") for i in range(4)]
    result = run_suite(
        lambda p: "a response", samples, [ContainsCheck("response")], cost_per_1k_chars=1.0
    )
    assert result.total_cost_usd == pytest.approx(4 * (len("prompt text") + len("a response")) / 1000)
    assert result.mean_latency_ms >= 0.0


def test_comparison_names_every_regression_and_fix():
    checks = build_extraction_checks()
    baseline = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="baseline")
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="candidate")
    comparison = compare_suites(baseline, candidate)
    assert set(comparison.regressed_ids) == {"tk-001", "tk-003", "tk-004", "tk-005", "rf-001", "pi-001"}
    assert comparison.fixed_ids == ("ar-001",)


def test_a_ten_prompt_suite_cannot_block_even_six_regressions():
    """An uncomfortable property, asserted rather than hidden.

    Six regressions against one fix is seven discordant pairs, and seven is not enough for the
    exact test at alpha = 0.05. The honest conclusion is that a statistical gate needs a real
    suite behind it; the named per-check failures are what makes a small suite useful.
    """
    checks = build_extraction_checks()
    baseline = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="baseline")
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="candidate")
    comparison = compare_suites(baseline, candidate)
    assert comparison.test.n_discordant == 7
    assert not comparison.test.significant
    assert comparison.gate_passed


def test_the_gate_blocks_a_demonstrated_regression():
    """With enough evidence the same gate closes."""
    ids = [f"s{i:03d}" for i in range(200)]
    a_flags, b_flags = demo.paired_outcomes_from_cells(**demo.REGRESSION_CELLS)
    a = _suite_from_flags("baseline", dict(zip(ids, a_flags)))
    b = _suite_from_flags("candidate", dict(zip(ids, b_flags)))
    comparison = compare_suites(a, b)
    assert comparison.test.significant
    assert not comparison.gate_passed
    assert "BLOCK" in comparison.summary()


def test_the_gate_does_not_block_an_improvement():
    ids = [f"s{i:03d}" for i in range(200)]
    a_flags, b_flags = demo.paired_outcomes_from_cells(130, 2, 38, 30)
    a = _suite_from_flags("baseline", dict(zip(ids, a_flags)))
    b = _suite_from_flags("candidate", dict(zip(ids, b_flags)))
    comparison = compare_suites(a, b)
    assert comparison.test.winner == "b"
    assert comparison.gate_passed


def test_comparing_different_sample_sets_is_refused():
    """Two reports about different things must not be averaged into one number."""
    a = _suite_from_flags("a", {"s1": True, "s2": False})
    b = _suite_from_flags("b", {"s1": True, "s3": False})
    with pytest.raises(ValueError, match="different samples"):
        compare_suites(a, b)


def test_comparison_pairs_by_id_not_by_position():
    """Insertion order differs; the result must not."""
    a = _suite_from_flags("a", {"s1": True, "s2": False, "s3": True})
    b_forward = _suite_from_flags("b", {"s1": False, "s2": False, "s3": True})
    b_shuffled = _suite_from_flags("b", {"s3": True, "s1": False, "s2": False})
    assert compare_suites(a, b_forward).regressed_ids == compare_suites(a, b_shuffled).regressed_ids


def test_summary_mentions_model_errors_when_present():
    checks = build_extraction_checks()
    candidate = run_suite(demo.CANDIDATE_MODEL, demo.EXTRACTION_SAMPLES, checks, name="candidate")
    assert "model error" in candidate.summary()


def test_json_schema_check_is_reusable_across_suites():
    """Checks must be stateless; a stateful one would leak results between runs."""
    check = scoped(JsonSchemaCheck(demo.TICKET_SCHEMA, name="schema"), "json")
    first = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES[:2], [check])
    second = run_suite(demo.BASELINE_MODEL, demo.EXTRACTION_SAMPLES[:2], [check])
    assert first.pass_rate == second.pass_rate == 1.0
