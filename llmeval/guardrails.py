"""Runtime guardrails, which fail in the opposite direction from evaluation.

The distinction this module exists to enforce
---------------------------------------------
An **eval** that crashes should be loud: raise, fail the run, block the release. Silence
there means shipping untested code.

A **guardrail** that crashes should *block the request*. It sits in the live path, and if
the PII detector throws on a malformed input, the two available behaviours are "block this
one request" and "disable the safety control for this request". Only the first is
defensible, and only the second is what a bare ``try/except: pass`` produces.

So :class:`GuardrailPipeline` is fail-closed by default. A rule that raises is recorded as
triggered, and the request is blocked. ``fail_closed=False`` exists because some
non-safety rules genuinely are advisory, but it must be chosen explicitly and per
pipeline.

Ordering
--------
Rules run in registration order and ``block`` short-circuits. Put cheap, high-confidence
rules first: there is no reason to run an expensive classifier on input a length check has
already rejected. Redaction rules transform the text and let it continue, so a redaction
rule placed after a rule that inspects the same content will see unredacted text -- the
order is a real decision, not a formality.

What pattern matching can and cannot do
---------------------------------------
:class:`InjectionHeuristicRule` catches the crude, high-volume attempts ("ignore previous
instructions"). It does not solve prompt injection, and nothing at this layer does.
Injection is a consequence of instructions and data sharing one channel; a paraphrase, a
translation, a base64 blob or an instruction embedded in a retrieved document defeats any
keyword list. Treat it as noise reduction on top of architectural controls -- least
privilege on tools, no untrusted content in privileged context, confirmation on
consequential actions -- not as a substitute for them. Marketing a keyword filter as
injection protection is how systems end up with a false sense of safety.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .pii import detect_pii, redact_pii

ACTIONS = ("block", "redact", "flag")
STAGES = ("input", "output")


@dataclass(frozen=True)
class RuleOutcome:
    """What one rule concluded.

    ``text`` is the possibly-transformed text to pass onward. Rules that do not transform
    return the text unchanged rather than ``None``, so the pipeline never has to special-case
    the absence of a value.
    """

    triggered: bool
    detail: str = ""
    text: str | None = None


class GuardrailRule(ABC):
    """One rule. Subclasses implement :meth:`evaluate`.

    ``action`` decides what a trigger *means*: ``block`` rejects the request, ``redact``
    modifies the text and continues, ``flag`` records and continues untouched.
    """

    def __init__(self, name: str, action: str = "block", stage: str = "input"):
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        self.name = name
        self.action = action
        self.stage = stage

    @abstractmethod
    def evaluate(self, text: str) -> RuleOutcome:
        ...


class DenyPatternRule(GuardrailRule):
    """Trigger when a regex matches."""

    def __init__(self, name: str, pattern: str, flags: int = re.IGNORECASE, **kwargs):
        super().__init__(name, **kwargs)
        self.pattern = re.compile(pattern, flags)

    def evaluate(self, text):
        match = self.pattern.search(text)
        if match is None:
            return RuleOutcome(False, text=text)
        return RuleOutcome(True, f"matched {self.pattern.pattern!r} at {match.start()}", text)


class MaxLengthRule(GuardrailRule):
    """Trigger when input exceeds a character budget.

    Cheap, so it belongs first. Oversized input is both a cost problem and the carrier for
    a class of context-stuffing attacks.
    """

    def __init__(self, name: str, max_chars: int, **kwargs):
        super().__init__(name, **kwargs)
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    def evaluate(self, text):
        if len(text) <= self.max_chars:
            return RuleOutcome(False, text=text)
        return RuleOutcome(True, f"{len(text)} chars exceeds limit {self.max_chars}", text)


class PIIRedactionRule(GuardrailRule):
    """Strip detected PII, letting the redacted text through.

    Defaults to ``action="redact"``: on an outbound path, removing a leaked email is
    usually better for the user than refusing to answer. On an inbound path into a system
    that must never store personal data, ``action="block"`` is the right choice instead.
    """

    def __init__(self, name: str = "pii", kinds: tuple[str, ...] | None = None, **kwargs):
        kwargs.setdefault("action", "redact")
        kwargs.setdefault("stage", "output")
        super().__init__(name, **kwargs)
        self.kinds = kinds

    def evaluate(self, text):
        redacted, matches = redact_pii(text, self.kinds)
        if not matches:
            return RuleOutcome(False, text=text)
        kinds = sorted({match.kind for match in matches})
        return RuleOutcome(True, f"redacted {len(matches)} span(s) of kinds {kinds}", redacted)


class SecretLeakRule(GuardrailRule):
    """Block output containing anything shaped like a credential.

    Separate from PII redaction and deliberately set to ``block``: a redacted API key is
    still a signal that something upstream is reading secrets into the model's context, and
    quietly masking it hides a real defect.
    """

    def __init__(self, name: str = "secret_leak", **kwargs):
        kwargs.setdefault("action", "block")
        kwargs.setdefault("stage", "output")
        super().__init__(name, **kwargs)

    def evaluate(self, text):
        matches = [match for match in detect_pii(text, ("api_key",))]
        if not matches:
            return RuleOutcome(False, text=text)
        return RuleOutcome(True, f"{len(matches)} credential-shaped token(s) in output", text)


INJECTION_MARKERS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
    r"forget\s+(?:everything|all)\s+(?:above|before)",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt",
    r"print\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
    r"repeat\s+(?:the\s+)?(?:text|words)\s+above",
    r"developer\s+mode",
    r"\bDAN\b",
)


class InjectionHeuristicRule(GuardrailRule):
    """Flag crude prompt-injection attempts.

    Defaults to ``action="flag"``, not ``block``, and that default is the point. These
    patterns have real false-positive rates -- a user legitimately asking "what are your
    instructions?" or a document *about* prompt injection will match -- and blocking on a
    heuristic this shallow trades a speculative attack for a certain outage.

    Read the module docstring before relying on this for anything.
    """

    def __init__(self, name: str = "injection_heuristic", markers=INJECTION_MARKERS, **kwargs):
        kwargs.setdefault("action", "flag")
        super().__init__(name, **kwargs)
        self.patterns = [re.compile(marker, re.IGNORECASE) for marker in markers]

    def evaluate(self, text):
        hits = [pattern.pattern for pattern in self.patterns if pattern.search(text)]
        if not hits:
            return RuleOutcome(False, text=text)
        return RuleOutcome(True, f"{len(hits)} injection marker(s): {hits[:3]}", text)


@dataclass
class GuardrailDecision:
    """The pipeline's verdict for one text."""

    allowed: bool
    text: str
    triggered: list[tuple[str, str, str]] = field(default_factory=list)
    blocked_by: str | None = None
    rule_errors: list[tuple[str, str]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def modified(self) -> bool:
        return any(action == "redact" for _, action, _ in self.triggered)

    @property
    def flags(self) -> list[str]:
        return [name for name, action, _ in self.triggered if action == "flag"]

    def explain(self) -> str:
        if not self.allowed:
            return f"blocked by {self.blocked_by}"
        parts = []
        if self.modified:
            parts.append("text modified")
        if self.flags:
            parts.append(f"flagged: {', '.join(self.flags)}")
        return "; ".join(parts) if parts else "clean"


class GuardrailPipeline:
    """Ordered rules with fail-closed error handling."""

    def __init__(self, rules: list[GuardrailRule], fail_closed: bool = True):
        if not rules:
            raise ValueError("a pipeline needs at least one rule")
        self.rules = rules
        self.fail_closed = fail_closed

    def rules_for(self, stage: str) -> list[GuardrailRule]:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        return [rule for rule in self.rules if rule.stage == stage]

    def run(self, text: str, stage: str = "input") -> GuardrailDecision:
        """Apply every rule for ``stage``, short-circuiting on the first block.

        A rule that raises is caught, recorded in ``rule_errors``, and -- under
        ``fail_closed`` -- blocks the request. The exception is deliberately not
        propagated: a crash in the safety layer should degrade to a refusal, not to a
        500 that some upstream retry loop turns into an unguarded call.
        """
        started = time.perf_counter()
        decision = GuardrailDecision(allowed=True, text=text)
        current = text

        for rule in self.rules_for(stage):
            try:
                outcome = rule.evaluate(current)
            except Exception as error:  # noqa: BLE001 - fail-closed is the whole point
                decision.rule_errors.append((rule.name, f"{type(error).__name__}: {error}"))
                if self.fail_closed:
                    decision.allowed = False
                    decision.blocked_by = f"{rule.name} (rule error, failed closed)"
                    decision.text = current
                    decision.elapsed_ms = (time.perf_counter() - started) * 1000
                    return decision
                continue

            if outcome.text is not None:
                current = outcome.text
            if not outcome.triggered:
                continue

            decision.triggered.append((rule.name, rule.action, outcome.detail))
            if rule.action == "block":
                decision.allowed = False
                decision.blocked_by = rule.name
                decision.text = current
                decision.elapsed_ms = (time.perf_counter() - started) * 1000
                return decision

        decision.text = current
        decision.elapsed_ms = (time.perf_counter() - started) * 1000
        return decision
