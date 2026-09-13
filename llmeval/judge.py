"""LLM-as-judge, with the known biases handled rather than ignored.

Judges have documented, reproducible biases (Zheng et al., 2023). Three matter enough to
build around:

**Position bias.** Present two answers as A and B, and the judge favours one slot
regardless of content -- often strongly. This is the easiest to fix and the most commonly
left unfixed: run the comparison twice with the answers swapped, and only count a win if
both orderings agree. Disagreement between the two orderings is not noise to average
away; it is the judge telling you it has no real preference, so it is recorded as a tie
and flagged. :class:`PairwiseJudge` does this by default and there is no way to switch it
off, because a single-order comparison is not worth reporting.

**Verbosity bias.** Judges reward length independently of quality. This cannot be fixed
by prompting, but it can be *measured*: :func:`length_bias_correlation` correlates scores
with answer length. A high correlation on a task where length should not matter means the
scores are partly measuring word count.

**Self-preference bias.** Models rate their own output higher. The only real mitigation
is to avoid judging a model with itself, which is a deployment choice rather than
something code can enforce -- so it is documented here rather than silently unhandled.

Parsing failures are surfaced, never defaulted
----------------------------------------------
When a judge returns something unparseable, :class:`RubricJudge` records ``value=None``
and ``parse_failed=True``. It does not fall back to a midpoint score. Silently
substituting 3 out of 5 for "the judge did not answer" contaminates the aggregate with
fabricated data, and it does so invisibly -- the mean shifts towards the midpoint and
nothing in the report says why.

A judge here is any callable from prompt string to response string, so this module makes
no network calls and is fully testable with a stub.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

JudgeModel = Callable[[str], str]

RUBRIC_TEMPLATE = """You are grading a response against a rubric.

Rubric:
{rubric}

Question:
{question}

Response:
{answer}

Grade the response on a scale from {low} to {high}, where {low} is worst and {high} is
best. Reply with a single line in exactly this format, and nothing else:
SCORE: <number>"""

PAIRWISE_TEMPLATE = """You are comparing two responses to the same question.

{criteria}

Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

Which response is better? Reply with a single line in exactly this format, and nothing
else:
VERDICT: A
or
VERDICT: B
or
VERDICT: TIE"""


@dataclass(frozen=True)
class RubricScore:
    """One graded response.

    ``value`` is ``None`` when the judge's reply could not be parsed. Callers must handle
    that case; the type makes it impossible to ignore by accident.
    """

    value: float | None
    raw_response: str
    parse_failed: bool


class RubricJudge:
    """Grade single responses on a numeric scale."""

    SCORE_LINE = re.compile(r"SCORE\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
    ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(
        self,
        model: JudgeModel,
        rubric: str,
        scale: tuple[float, float] = (1.0, 5.0),
        template: str = RUBRIC_TEMPLATE,
    ):
        low, high = scale
        if low >= high:
            raise ValueError("scale must be (low, high) with low < high")
        self.model = model
        self.rubric = rubric
        self.scale = scale
        self.template = template

    def build_prompt(self, question: str, answer: str) -> str:
        return self.template.format(
            rubric=self.rubric,
            question=question,
            answer=answer,
            low=self.scale[0],
            high=self.scale[1],
        )

    def parse(self, response: str) -> RubricScore:
        """Extract a score, preferring the requested format.

        The fallback to "first number in range" is a deliberate leniency for models that
        add a preamble. A number outside the scale is treated as a parse failure rather
        than clamped -- a judge answering 8 on a 1-5 scale has misunderstood the task, and
        clamping to 5 would hide that.
        """
        low, high = self.scale
        match = self.SCORE_LINE.search(response)
        if match is not None:
            value = float(match.group(1))
            if low <= value <= high:
                return RubricScore(value, response, False)
            return RubricScore(None, response, True)

        for candidate in self.ANY_NUMBER.findall(response):
            value = float(candidate)
            if low <= value <= high:
                return RubricScore(value, response, False)
        return RubricScore(None, response, True)

    def score(self, question: str, answer: str) -> RubricScore:
        return self.parse(self.model(self.build_prompt(question, answer)))

    def score_many(self, questions: Sequence[str], answers: Sequence[str]) -> list[RubricScore]:
        if len(questions) != len(answers):
            raise ValueError("questions and answers must be the same length")
        return [self.score(q, a) for q, a in zip(questions, answers)]


@dataclass(frozen=True)
class PairwiseVerdict:
    """The result of a position-debiased pairwise comparison.

    ``winner`` is ``"a"``, ``"b"`` or ``"tie"``. It is ``"tie"`` whenever the two
    orderings disagreed, which is the honest reading: the judge's preference did not
    survive swapping the labels.
    """

    winner: str
    first_order_winner: str
    second_order_winner: str
    position_bias_detected: bool
    raw_responses: tuple[str, str]

    @property
    def consistent(self) -> bool:
        return not self.position_bias_detected


class PairwiseJudge:
    """Compare two responses, always in both orders."""

    VERDICT_LINE = re.compile(r"VERDICT\s*:\s*(A|B|TIE)", re.IGNORECASE)

    def __init__(
        self,
        model: JudgeModel,
        criteria: str = "Judge on correctness first, then clarity. Ignore length.",
        template: str = PAIRWISE_TEMPLATE,
    ):
        self.model = model
        self.criteria = criteria
        self.template = template

    def build_prompt(self, question: str, answer_a: str, answer_b: str) -> str:
        return self.template.format(
            criteria=self.criteria,
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
        )

    def parse(self, response: str) -> str:
        """Return ``"a"``, ``"b"`` or ``"tie"``; unparseable replies become ties.

        A tie is the right default for an unreadable verdict: it asserts no preference,
        whereas defaulting to either side would invent one.
        """
        match = self.VERDICT_LINE.search(response)
        if match is None:
            return "tie"
        verdict = match.group(1).upper()
        return {"A": "a", "B": "b", "TIE": "tie"}[verdict]

    def compare(self, question: str, answer_a: str, answer_b: str) -> PairwiseVerdict:
        """Run the comparison twice, swapped, and require both orderings to agree.

        In the second call the answers are physically swapped, so a verdict of ``"a"``
        there refers to ``answer_b``. That relabelling is done here; getting it wrong is
        the standard bug in position-swap implementations and would silently invert every
        result.
        """
        first_response = self.model(self.build_prompt(question, answer_a, answer_b))
        first = self.parse(first_response)

        second_response = self.model(self.build_prompt(question, answer_b, answer_a))
        raw_second = self.parse(second_response)
        # Undo the swap: "a" in the swapped prompt means answer_b won.
        second = {"a": "b", "b": "a", "tie": "tie"}[raw_second]

        if first == second:
            winner = first
            bias = False
        else:
            winner = "tie"
            # Both orderings naming the same *slot* is the signature of position bias, as
            # opposed to ordinary inconsistency between a preference and a tie.
            bias = first != "tie" and second != "tie"

        return PairwiseVerdict(
            winner=winner,
            first_order_winner=first,
            second_order_winner=second,
            position_bias_detected=bias,
            raw_responses=(first_response, second_response),
        )


def length_bias_correlation(scores: Sequence[float], answers: Sequence[str]) -> float:
    """Spearman correlation between judge scores and answer length.

    Rank correlation rather than Pearson, because the relationship need not be linear and
    rubric scores are ordinal.

    Interpretation needs care. On a task where thoroughness genuinely is quality, a
    positive correlation is expected and fine. It is evidence of verbosity bias only where
    length *should* be irrelevant -- extraction, classification, arithmetic. The
    correlation is the measurement; whether it is a problem is a judgement about the task.
    """
    if len(scores) != len(answers):
        raise ValueError("scores and answers must be the same length")
    if len(scores) < 3:
        raise ValueError("need at least 3 items for a rank correlation")
    lengths = [len(answer) for answer in answers]
    if len(set(lengths)) == 1 or len(set(scores)) == 1:
        return 0.0
    return float(stats.spearmanr(np.asarray(scores, dtype=float), lengths).statistic)
