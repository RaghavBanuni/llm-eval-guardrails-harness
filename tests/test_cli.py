"""The demonstrations must actually demonstrate what they claim.

These are not smoke tests. Each one asserts the specific finding the corresponding section of
the README rests on, so a change that quietly breaks an argument breaks a test.
"""

import pytest

from llmeval.cli import main


def run(command: str, capsys) -> str:
    assert main([command]) == 0
    return capsys.readouterr().out


def test_checks_demo_names_the_regressions_and_shows_the_weak_gate(capsys):
    output = run("checks", capsys)
    assert "9/10" in output and "4/10" in output
    assert "ticket_schema" in output
    assert "regressed (6)" in output
    assert "fixed (1)" in output
    # Named, actionable failures...
    assert "missing required property 'urgency'" in output
    assert "expected integer, got str" in output
    assert "TimeoutError" in output
    assert "unexpected refusal" in output
    # ...and an honest statement that the statistical gate cannot carry a ten-prompt suite.
    assert "gate: PASS" in output
    assert "not enough" in output


def test_compare_demo_shows_the_non_result_then_a_real_one(capsys):
    output = run("compare", capsys)
    assert "84.0%" in output and "81.0%" in output
    assert "Not distinguishable" in output
    assert "14 disagreements" in output
    assert "System A wins" in output  # the second, genuine regression


def test_power_demo_reports_four_figure_sample_sizes(capsys):
    output = run("power", capsys)
    assert "80% power" in output
    assert "prompts needed" in output
    # A two-point difference needs a suite with a thousands separator in it.
    assert any("," in line for line in output.splitlines() if "2.0 points" in line)


def test_judge_demo_shows_bias_detection_and_a_refused_score(capsys):
    output = run("judge", capsys)
    assert "position bias detected: True" in output
    assert "consistent: True" in output
    assert "parse_failed: True" in output
    assert "parsed value  : None" in output


def test_validate_demo_contrasts_agreement_with_kappa(capsys):
    output = run("validate", capsys)
    assert "raw agreement" in output
    assert "Cohen's kappa : 0.000" in output
    assert "trustworthy   : False" in output
    assert "trustworthy   : True" in output


def test_guardrails_demo_shows_redaction_blocking_and_fail_closed(capsys):
    output = run("guardrails", capsys)
    assert "[REDACTED:email]" in output
    assert "blocked by secret_leak" in output
    assert "injection_heuristic" in output
    assert "fail_closed=True   allowed=False" in output
    assert "fail_closed=False  allowed=True" in output


def test_unknown_command_exits_nonzero(capsys):
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_no_command_is_an_error_rather_than_a_silent_no_op(capsys):
    with pytest.raises(SystemExit):
        main([])
