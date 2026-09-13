"""Deterministic checks: the properties each one is claimed to have."""

import json

import pytest

from llmeval.checks import (
    ContainsCheck,
    ExactMatchCheck,
    JsonSchemaCheck,
    MaxLengthCheck,
    NoPIICheck,
    NotContainsCheck,
    NumericToleranceCheck,
    RefusalCheck,
    RegexCheck,
    validate_against_schema,
)
from llmeval.types import Sample

SAMPLE = Sample("s1", "prompt")

SCHEMA = {
    "type": "object",
    "required": ["category", "urgency"],
    "properties": {
        "category": {"type": "string", "enum": ["billing", "technical"]},
        "urgency": {"type": "integer", "minimum": 1, "maximum": 5},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def test_schema_accepts_valid_object():
    check = JsonSchemaCheck(SCHEMA)
    output = json.dumps({"category": "billing", "urgency": 3, "tags": ["a", "b"]})
    assert check.run(output, SAMPLE).passed


def test_schema_catches_missing_required_field():
    """The regression the demo candidate ships: a dropped field, still valid JSON."""
    outcome = JsonSchemaCheck(SCHEMA).run(json.dumps({"category": "billing"}), SAMPLE)
    assert not outcome.passed
    assert "urgency" in outcome.detail


def test_schema_catches_wrong_type_on_a_present_field():
    """"Valid JSON" is not the property that matters; the field types are."""
    output = json.dumps({"category": "billing", "urgency": "low"})
    outcome = JsonSchemaCheck(SCHEMA).run(output, SAMPLE)
    assert not outcome.passed
    assert "integer" in outcome.detail


def test_schema_rejects_boolean_where_a_number_is_required():
    """``bool`` subclasses ``int`` in Python; JSON treats them as different types."""
    errors = validate_against_schema(True, {"type": "integer"})
    assert errors and "boolean" in errors[0]


def test_schema_enforces_enum_bounds_and_additional_properties():
    check = JsonSchemaCheck(SCHEMA)
    assert not check.run(json.dumps({"category": "other", "urgency": 3}), SAMPLE).passed
    assert not check.run(json.dumps({"category": "billing", "urgency": 9}), SAMPLE).passed
    assert not check.run(
        json.dumps({"category": "billing", "urgency": 3, "extra": 1}), SAMPLE
    ).passed


def test_schema_validates_array_items_individually():
    output = json.dumps({"category": "billing", "urgency": 3, "tags": ["ok", 5]})
    outcome = JsonSchemaCheck(SCHEMA).run(output, SAMPLE)
    assert not outcome.passed
    assert "tags[1]" in outcome.detail


def test_schema_reports_several_violations_at_once():
    """Four problems should be reported as four, not one per run."""
    errors = validate_against_schema({"urgency": 99, "extra": True}, SCHEMA)
    assert len(errors) >= 3


def test_schema_accepts_fenced_json_when_allowed_and_not_otherwise():
    output = "```json\n" + json.dumps({"category": "billing", "urgency": 3}) + "\n```"
    assert JsonSchemaCheck(SCHEMA, allow_fenced=True).run(output, SAMPLE).passed
    assert not JsonSchemaCheck(SCHEMA, allow_fenced=False).run(output, SAMPLE).passed


def test_schema_reports_parse_position_on_malformed_json():
    outcome = JsonSchemaCheck(SCHEMA).run("{not json", SAMPLE)
    assert not outcome.passed
    assert "not valid JSON" in outcome.detail


def test_unsupported_schema_keyword_raises_rather_than_passing_silently():
    """A schema this validator cannot express must not appear to validate."""
    with pytest.raises(ValueError, match="unsupported schema type"):
        validate_against_schema({}, {"type": "tuple"})


def test_exact_match_normalises_whitespace_and_case():
    sample = Sample("s", "p", expected="Paris")
    assert ExactMatchCheck().run("  paris \n", sample).passed
    assert not ExactMatchCheck().run("Paris, France", sample).passed


def test_exact_match_can_strip_punctuation():
    sample = Sample("s", "p", expected="yes")
    assert ExactMatchCheck(strip_punctuation=True).run("Yes!", sample).passed
    assert not ExactMatchCheck().run("Yes!", sample).passed


def test_check_without_expected_value_fails_loudly():
    """A misconfigured sample must not pass vacuously."""
    assert not ExactMatchCheck().run("anything", SAMPLE).passed
    assert not NumericToleranceCheck().run("42", SAMPLE).passed


def test_numeric_tolerance_accepts_equivalent_formatting():
    sample = Sample("s", "p", expected="3.14")
    check = NumericToleranceCheck(tolerance=0.001)
    assert check.run("3.140", sample).passed
    assert check.run("about 3.14 or so", sample).passed
    assert not check.run("3.2", sample).passed


def test_numeric_tolerance_relative_mode_scales_with_magnitude():
    sample = Sample("s", "p", expected="1000000")
    assert NumericToleranceCheck(tolerance=0.01, relative=True).run("1005000", sample).passed
    assert not NumericToleranceCheck(tolerance=0.001, relative=True).run("1005000", sample).passed


def test_numeric_tolerance_flags_ambiguous_extraction_in_the_detail():
    """Taking the first of several numbers is a guess, and the report says so."""
    sample = Sample("s", "p", expected="96")
    outcome = NumericToleranceCheck().run("96 minutes, or 1.6 hours", sample)
    assert outcome.passed
    assert "2 numbers" in outcome.detail


def test_numeric_tolerance_reports_missing_number():
    sample = Sample("s", "p", expected="96")
    assert not NumericToleranceCheck().run("ninety-six", sample).passed


def test_refusal_check_detects_both_directions():
    refusal = "I'm sorry, but I cannot provide that."
    answer = "Refunds are available within 30 days."
    assert RefusalCheck(expect_refusal=True).run(refusal, SAMPLE).passed
    assert not RefusalCheck(expect_refusal=True).run(answer, SAMPLE).passed
    assert RefusalCheck(expect_refusal=False).run(answer, SAMPLE).passed
    assert not RefusalCheck(expect_refusal=False).run(refusal, SAMPLE).passed


def test_false_refusal_is_reported_with_the_matched_marker():
    """The failure that ships unnoticed: refusing something benign."""
    outcome = RefusalCheck(expect_refusal=False).run("As an AI, I can't help with that.", SAMPLE)
    assert not outcome.passed
    assert "as an ai" in outcome.detail


def test_no_pii_check_fails_on_leaked_email():
    outcome = NoPIICheck().run("Contact dana.k@example.com", SAMPLE)
    assert not outcome.passed and "email" in outcome.detail
    assert NoPIICheck().run("Contact support", SAMPLE).passed


def test_contains_and_not_contains_are_case_insensitive_by_default():
    assert ContainsCheck("refund").run("Your REFUND is processed", SAMPLE).passed
    assert not ContainsCheck("refund", case_sensitive=True).run("REFUND", SAMPLE).passed
    assert not NotContainsCheck("system prompt").run("my System Prompt is...", SAMPLE).passed


def test_regex_check_works_in_both_polarities():
    assert RegexCheck(r"^\d{4}-\d{2}-\d{2}$").run("2026-03-01", SAMPLE).passed
    assert RegexCheck(r"TODO", should_match=False).run("finished", SAMPLE).passed
    assert not RegexCheck(r"TODO", should_match=False).run("TODO: fix", SAMPLE).passed


def test_max_length_defaults_to_warning_severity():
    """Length drift is a signal, not a release blocker; the severity encodes that."""
    outcome = MaxLengthCheck(10).run("x" * 50, SAMPLE)
    assert not outcome.passed
    assert outcome.severity == "warning"


def test_invalid_severity_and_arguments_are_rejected():
    with pytest.raises(ValueError):
        ContainsCheck("x", severity="critical")
    with pytest.raises(ValueError):
        MaxLengthCheck(0)
    with pytest.raises(ValueError):
        NumericToleranceCheck(tolerance=-1)
    with pytest.raises(TypeError):
        JsonSchemaCheck("not a dict")


def test_check_name_defaults_to_class_name_and_can_be_overridden():
    assert ContainsCheck("x").name == "ContainsCheck"
    assert ContainsCheck("x", name="mentions_refund").name == "mentions_refund"
