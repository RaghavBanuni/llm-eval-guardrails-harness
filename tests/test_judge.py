"""Judge behaviour: position bias, length bias, and refusing to invent a score."""

import pytest

from llmeval import demo
from llmeval.judge import PairwiseJudge, RubricJudge, length_bias_correlation

QUESTION = "A plan costs 40 dollars for 3 seats. What do 7 seats cost?"
WRONG = "Seven seats cost 90 dollars."
RIGHT = "Per seat that is 13.33 dollars, so seven seats cost 93.33 dollars."


def test_first_slot_bias_is_reported_as_a_tie_not_a_win():
    """A judge with no view on content must not produce a winner.

    Run once, this judge confidently backs whichever answer happens to be first -- here the
    wrong one. Running both orders turns that into the honest answer: no preference.
    """
    verdict = PairwiseJudge(demo.position_biased_judge).compare(QUESTION, WRONG, RIGHT)
    assert verdict.winner == "tie"
    assert verdict.position_bias_detected
    assert not verdict.consistent


def test_second_slot_bias_is_also_caught():
    """The mirror case, which a buggy swap implementation would silently pass."""
    verdict = PairwiseJudge(lambda prompt: "VERDICT: B").compare(QUESTION, WRONG, RIGHT)
    assert verdict.winner == "tie"
    assert verdict.position_bias_detected


def test_a_content_based_judge_produces_a_consistent_winner():
    verdict = PairwiseJudge(demo.content_judge).compare(QUESTION, WRONG, RIGHT)
    assert verdict.winner == "b"
    assert verdict.consistent
    assert verdict.first_order_winner == verdict.second_order_winner == "b"


def test_swap_relabelling_is_correct_in_both_directions():
    """The standard bug in position-swap code is forgetting to undo the swap.

    Putting the right answer in slot A must give winner "a" -- if the relabelling were
    dropped, or applied to the wrong call, this would come back as "b" or as a tie.
    """
    judge = PairwiseJudge(demo.content_judge)
    assert judge.compare(QUESTION, RIGHT, WRONG).winner == "a"
    assert judge.compare(QUESTION, WRONG, RIGHT).winner == "b"


def test_genuine_tie_is_reported_without_flagging_bias():
    """Agreeing on a tie is consistency, not position bias."""
    verdict = PairwiseJudge(demo.content_judge).compare(QUESTION, WRONG, WRONG)
    assert verdict.winner == "tie"
    assert not verdict.position_bias_detected


def test_a_preference_that_softens_to_a_tie_is_inconsistent_but_not_position_bias():
    """One order prefers A, the other abstains: no winner, and not a slot preference either.

    The distinction matters because the two problems have different fixes -- a slot preference
    needs a different judge, an unstable one needs a clearer rubric.
    """
    responses = iter(["VERDICT: A", "VERDICT: TIE"])
    verdict = PairwiseJudge(lambda prompt: next(responses)).compare(QUESTION, WRONG, RIGHT)
    assert verdict.winner == "tie"
    assert not verdict.position_bias_detected


def test_pairwise_judge_always_calls_the_model_twice():
    calls = []

    def counting_judge(prompt: str) -> str:
        calls.append(prompt)
        return "VERDICT: A"

    PairwiseJudge(counting_judge).compare(QUESTION, WRONG, RIGHT)
    assert len(calls) == 2
    # The two prompts must differ, or the swap never happened.
    assert calls[0] != calls[1]


def test_unreadable_verdict_becomes_a_tie_rather_than_a_coin_flip():
    verdict = PairwiseJudge(lambda prompt: "I think both are fine").compare(QUESTION, WRONG, RIGHT)
    assert verdict.winner == "tie"


def test_rubric_parse_failure_yields_none_not_a_midpoint():
    """The claim in the module docstring, pinned.

    Substituting 3-out-of-5 for "the judge did not answer" would move every aggregate towards
    the middle and leave no trace of why.
    """
    score = RubricJudge(demo.unparseable_judge, rubric="Correct?").score("q", "a")
    assert score.value is None
    assert score.parse_failed
    assert score.raw_response  # the raw reply is kept for triage


def test_out_of_range_score_is_a_parse_failure_not_a_clamp():
    """A judge answering 8 on a 1-5 scale has misunderstood; clamping to 5 would hide that."""
    score = RubricJudge(lambda p: "SCORE: 8", rubric="Correct?", scale=(1, 5)).score("q", "a")
    assert score.value is None and score.parse_failed


def test_rubric_accepts_the_requested_format_and_a_preamble():
    judge = RubricJudge(lambda p: "SCORE: 4", rubric="Correct?")
    assert judge.score("q", "a").value == 4.0

    lenient = RubricJudge(lambda p: "I would give this a 2 out of 5.", rubric="Correct?")
    assert lenient.score("q", "a").value == 2.0


def test_rubric_prompt_contains_the_rubric_question_and_answer():
    judge = RubricJudge(lambda p: "SCORE: 3", rubric="Is it factually correct?")
    prompt = judge.build_prompt("How many seats?", "Seven.")
    assert "Is it factually correct?" in prompt
    assert "How many seats?" in prompt
    assert "Seven." in prompt
    assert "SCORE:" in prompt


def test_rubric_rejects_an_inverted_scale():
    with pytest.raises(ValueError, match="low < high"):
        RubricJudge(lambda p: "SCORE: 1", rubric="r", scale=(5, 1))


def test_score_many_requires_aligned_inputs():
    judge = RubricJudge(lambda p: "SCORE: 3", rubric="r")
    assert len(judge.score_many(["q1", "q2"], ["a1", "a2"])) == 2
    with pytest.raises(ValueError):
        judge.score_many(["q1"], ["a1", "a2"])


def test_length_bias_is_detected_when_the_judge_rewards_verbosity():
    answers = ["96.", "The total is 96 minutes.", "a" * 120, "b" * 250]
    judge = RubricJudge(demo.verbose_preferring_judge, rubric="Correct?")
    scores = [judge.score("q", answer).value for answer in answers]
    assert length_bias_correlation(scores, answers) > 0.9


def test_length_bias_is_zero_for_a_judge_that_ignores_length():
    answers = ["96.", "The total is 96 minutes.", "a" * 120, "b" * 250]
    scores = [4.0, 4.0, 4.0, 4.0]
    assert length_bias_correlation(scores, answers) == 0.0


def test_length_bias_correlation_validates_input():
    with pytest.raises(ValueError):
        length_bias_correlation([1.0, 2.0], ["a"])
    with pytest.raises(ValueError, match="at least 3"):
        length_bias_correlation([1.0, 2.0], ["a", "bb"])
