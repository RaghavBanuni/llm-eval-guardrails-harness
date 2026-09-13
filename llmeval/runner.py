"""Running a suite and comparing two runs.

Three decisions here are worth stating, because each is a place harnesses commonly go
wrong.

**Model errors are results.** If a model call raises, the prediction is recorded with an
``error`` and the case counts as a failure. The tempting alternative -- skip it and compute
the pass rate over what remains -- produces a number computed on a biased subset, since the
prompts that error are disproportionately the long, complex or adversarial ones. A pass
rate that silently excludes the hardest 5% of the suite is worse than no pass rate.

**Scoping is the suite's job, not the check's.** A check is a pure predicate on
``(output, sample)``; *which* samples it should run against is a property of how the suite
is assembled. So :func:`scoped` wraps a check with a tag filter rather than every check
class growing a tag parameter. A sample that ends up with no applicable check is reported
as a failure, not a pass -- an unevaluated case counted as a pass inflates the pass rate
with cases nobody checked, which is the quietest way an eval can lie.

**Comparison is by sample id, not by position.** :func:`compare_suites` matches cases on id
and refuses to compare runs over different sample sets. Zipping two result lists
positionally is a bug that produces a plausible-looking number from misaligned pairs, and
nothing downstream can detect it.

Every aggregate comes with a Wilson interval, because a pass rate without one invites
exactly the over-reading the interval is there to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Union

from .checks import Check
from .stats import McNemarResult, mcnemar, wilson_interval
from .types import CaseResult, CheckOutcome, Prediction, Sample

ModelFn = Callable[[str], str]


@dataclass(frozen=True)
class ScopedCheck:
    """A check restricted to samples carrying at least one of ``tags``."""

    check: Check
    tags: frozenset[str]

    def applies(self, sample: Sample) -> bool:
        return bool(self.tags & set(sample.tags))


CheckSpec = Union[Check, ScopedCheck]


def scoped(check: Check, *tags: str) -> ScopedCheck:
    """Restrict ``check`` to samples with any of ``tags``.

    ``scoped(JsonSchemaCheck(schema), "json")`` reads better at the call site than a
    ``applies_to=`` argument threaded through every check constructor, and it keeps the
    check classes ignorant of suite composition.
    """
    if not tags:
        raise ValueError("scoped() needs at least one tag")
    return ScopedCheck(check, frozenset(tags))


def _applies(spec: CheckSpec, sample: Sample) -> bool:
    return spec.applies(sample) if isinstance(spec, ScopedCheck) else True


def _unwrap(spec: CheckSpec) -> Check:
    return spec.check if isinstance(spec, ScopedCheck) else spec


@dataclass(frozen=True)
class SuiteResult:
    """Aggregated outcome of one suite run."""

    name: str
    cases: tuple[CaseResult, ...]

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_cases if self.n_cases else 0.0

    @property
    def n_errors(self) -> int:
        return sum(1 for case in self.cases if case.prediction.failed)

    @property
    def total_cost_usd(self) -> float:
        return sum(case.prediction.cost_usd for case in self.cases)

    @property
    def mean_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        return sum(case.prediction.latency_ms for case in self.cases) / self.n_cases

    def pass_rate_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        if not self.n_cases:
            return (0.0, 0.0)
        return wilson_interval(self.n_passed, self.n_cases, confidence)

    def pass_map(self) -> dict[str, bool]:
        """``{sample_id: passed}``, the input :func:`compare_suites` pairs on."""
        return {case.sample_id: case.passed for case in self.cases}

    def by_tag(self) -> dict[str, tuple[int, int]]:
        """``{tag: (passed, total)}``.

        The reason to slice by tag: an aggregate pass rate can stay flat while one
        capability collapses. A change that breaks every extraction prompt but fixes a few
        arithmetic prompts barely moves the total and is obvious per tag.
        """
        counts: dict[str, list[int]] = {}
        for case in self.cases:
            for tag in case.tags:
                bucket = counts.setdefault(tag, [0, 0])
                bucket[1] += 1
                if case.passed:
                    bucket[0] += 1
        return {tag: (passed, total) for tag, (passed, total) in sorted(counts.items())}

    def failures_by_check(self) -> dict[str, int]:
        """How many cases each check failed -- the triage view."""
        counts: dict[str, int] = {}
        for case in self.cases:
            for name in case.failed_checks:
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def summary(self) -> str:
        low, high = self.pass_rate_interval()
        line = (
            f"{self.name}: {self.n_passed}/{self.n_cases} passed "
            f"({self.pass_rate:.1%}, 95% CI {low:.1%}-{high:.1%})"
        )
        if self.n_errors:
            line += f", {self.n_errors} model error(s) counted as failures"
        return line


def run_suite(
    model: ModelFn,
    samples: Sequence[Sample],
    checks: Iterable[CheckSpec],
    name: str = "suite",
    cost_per_1k_chars: float = 0.0,
) -> SuiteResult:
    """Run every sample through ``model`` and apply every applicable check.

    ``cost_per_1k_chars`` is a stand-in for real token accounting, which depends on the
    provider's tokeniser. It is here so cost aggregation has something to aggregate; wire
    real usage numbers in through :class:`~llmeval.types.Prediction` for anything you intend
    to bill against.
    """
    specs = list(checks)
    if not specs:
        raise ValueError("a suite needs at least one check")
    if not samples:
        raise ValueError("a suite needs at least one sample")

    seen: set[str] = set()
    cases: list[CaseResult] = []
    for sample in samples:
        if sample.id in seen:
            raise ValueError(f"duplicate sample id {sample.id!r}")
        seen.add(sample.id)

        started = time.perf_counter()
        try:
            output = model(sample.prompt)
            error = None
        except Exception as exc:  # noqa: BLE001 - an error is a result, see module docstring
            output = ""
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000

        prediction = Prediction(
            sample_id=sample.id,
            output=output,
            latency_ms=elapsed_ms,
            cost_usd=cost_per_1k_chars * (len(sample.prompt) + len(output)) / 1000.0,
            error=error,
        )

        if error is not None:
            outcomes: tuple[CheckOutcome, ...] = (CheckOutcome("model_call", False, error, "error"),)
        else:
            applicable = [_unwrap(spec) for spec in specs if _applies(spec, sample)]
            if applicable:
                outcomes = tuple(check.run(output, sample) for check in applicable)
            else:
                outcomes = (
                    CheckOutcome(
                        "no_applicable_check",
                        False,
                        f"no check is scoped to tags {list(sample.tags)}; sample is unevaluated",
                        "error",
                    ),
                )

        cases.append(CaseResult(sample.id, prediction, outcomes, sample.tags))

    return SuiteResult(name=name, cases=tuple(cases))


@dataclass(frozen=True)
class ComparisonResult:
    """A paired comparison of two suite runs."""

    a_name: str
    b_name: str
    a_pass_rate: float
    b_pass_rate: float
    test: McNemarResult
    regressed_ids: tuple[str, ...]
    fixed_ids: tuple[str, ...]

    @property
    def gate_passed(self) -> bool:
        """Whether B is safe to ship over A on statistical evidence alone.

        The rule is asymmetric on purpose. B is blocked only when the test finds a
        *significant* win for A -- evidence of real regression. An inconclusive result does
        not block, because on a suite of realistic size most differences are inconclusive and
        a gate demanding significance to ship would never open.

        The limitation is worth being blunt about: on a small suite this gate is weak, and a
        candidate that broke several cases can still clear it. That is not a flaw in the
        test -- it is a small suite failing to carry the evidence. The per-check failure
        lists are the actionable output at that size; the statistical gate becomes
        meaningful when the suite is large enough to make it so.
        """
        return self.test.winner != "a"

    def summary(self) -> str:
        lines = [
            f"{self.a_name}: {self.a_pass_rate:.1%}    {self.b_name}: {self.b_pass_rate:.1%}",
            self.test.explain(),
        ]
        if self.regressed_ids:
            lines.append(
                f"regressed ({len(self.regressed_ids)}): {', '.join(self.regressed_ids[:8])}"
            )
        if self.fixed_ids:
            lines.append(f"fixed ({len(self.fixed_ids)}): {', '.join(self.fixed_ids[:8])}")
        lines.append("gate: PASS" if self.gate_passed else "gate: BLOCK (significant regression)")
        return "\n".join(lines)


def compare_suites(a: SuiteResult, b: SuiteResult, alpha: float = 0.05) -> ComparisonResult:
    """Compare two runs over the same samples, paired by sample id.

    Raises if the two runs cover different sample sets. Comparing partially overlapping
    suites is not a statistics problem to be worked around; it is a report about two
    different things.
    """
    a_map = a.pass_map()
    b_map = b.pass_map()
    if set(a_map) != set(b_map):
        only_a = sorted(set(a_map) - set(b_map))[:5]
        only_b = sorted(set(b_map) - set(a_map))[:5]
        raise ValueError(
            "suites cover different samples and cannot be paired; "
            f"only in {a.name}: {only_a}, only in {b.name}: {only_b}"
        )

    ids = sorted(a_map)
    a_passed = [a_map[sample_id] for sample_id in ids]
    b_passed = [b_map[sample_id] for sample_id in ids]

    return ComparisonResult(
        a_name=a.name,
        b_name=b.name,
        a_pass_rate=a.pass_rate,
        b_pass_rate=b.pass_rate,
        test=mcnemar(a_passed, b_passed, alpha=alpha),
        regressed_ids=tuple(i for i in ids if a_map[i] and not b_map[i]),
        fixed_ids=tuple(i for i in ids if b_map[i] and not a_map[i]),
    )
