"""Core data types.

Deliberately plain: a sample, a prediction, the outcome of one check, and the result of
checking one prediction. The reason to write these down rather than pass dictionaries
around is that an eval harness accumulates fields over time -- latency, cost, token counts,
retry counts -- and a dataclass makes it obvious when one is missing rather than silently
producing ``None`` halfway through an aggregation.

:class:`CheckOutcome` lives here rather than in :mod:`llmeval.checks` so that
:class:`CaseResult` can reference it without a circular import. ``checks`` re-exports it,
which is where callers naturally look for it.

One design decision worth naming: ``Prediction.error`` exists. A model call that raises is
a *result*, not an absence of one, and an eval that silently drops failed calls reports a
pass rate computed over a biased subset -- exactly the prompts the model found hardest. So
errors are recorded and counted as failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class Sample:
    """One evaluation case.

    ``expected`` is optional because many checks are reference-free -- a schema check or a
    PII check needs no gold answer. ``tags`` allows slicing the suite by capability, which
    is where regressions actually show up; an aggregate pass rate hides a catastrophic
    failure on 5% of traffic.
    """

    id: str
    prompt: str
    expected: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """What the model returned, plus what it cost."""

    sample_id: str
    output: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class CheckOutcome:
    """The result of one check against one output.

    ``severity`` distinguishes a release-blocking failure from an advisory signal. See
    :mod:`llmeval.checks` for why that distinction is load-bearing.
    """

    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")


@dataclass(frozen=True)
class CaseResult:
    """The outcome of running every check against one prediction.

    ``tags`` is copied from the sample rather than looked up later, so a result object is
    self-contained and a suite can be aggregated without carrying the sample list alongside
    it.
    """

    sample_id: str
    prediction: Prediction
    outcomes: tuple[CheckOutcome, ...]
    tags: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """A case passes only if every error-severity check passes.

        Warnings are recorded and reported but do not fail the case, so a suite can carry
        advisory signals -- length drift, latency -- without them gating a deploy.
        """
        if self.prediction.failed:
            return False
        return all(o.passed for o in self.outcomes if o.severity == "error")

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.outcomes if not o.passed)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.outcomes if not o.passed and o.severity == "warning")
