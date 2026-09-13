"""PII detection, with an emphasis on not crying wolf.

A PII detector that fires on every 16-digit number is worse than useless: it trains reviewers
to ignore it, and once alerts are ignored the real leak passes through with everything else.
So the detectors here are built to be *specific* as well as sensitive.

The clearest example is card numbers. A naive detector matches any 16 digits, which hits
order ids, tracking numbers, concatenated timestamps and hashes. Real card numbers carry a
check digit under the Luhn algorithm (ISO/IEC 7812), so validating it rejects roughly 90% of
random digit strings while keeping every genuine card. That is a ten-fold reduction in false
positives for about fifteen lines of arithmetic.

Two boundary decisions deserve a note, because this is where such code usually breaks
quietly.

*Trailing punctuation.* A card number at the end of a sentence is followed by a full stop, and
a naive ``(?![\w.])`` guard rejects it -- so the detector silently misses exactly the phrasing
a model is most likely to produce ("your card ending 4111 1111 1111 1111."). The guards here
exclude a *decimal point followed by a digit* rather than any dot, and there is a regression
test for the full-stop case.

*Phone length.* The phone pattern is capped at 15 characters, matching E.164's 15-digit
maximum. Without that cap a 16-digit order reference matches as a phone number, so the
Luhn-based specificity won on cards and was immediately given back on phones.

What this module is not
-----------------------
It is regex plus arithmetic. It will miss PII written in words ("my card ends in four two one
one"), PII split across turns, names, addresses, and anything deliberately obfuscated. The
credential detector in particular trades precision for recall -- a hyphenated identifier
containing a digit can trip it -- because a missed key costs more than a reviewed false
positive. Detection reduces the rate of accidental leaks; it does not make a system safe to
hand untrusted data. Where that distinction matters, on a regulated surface, this belongs
behind a real DLP system rather than in front of one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Optional +, then 7-15 characters of digits and separators. The 15-character ceiling is the
# E.164 maximum and is what keeps long numeric identifiers out; the floor keeps years and
# small integers out.
PHONE_PATTERN = re.compile(r"(?<![\d.])\+?\d[\d\s().-]{5,13}\d(?!\.?\d)")

# 13-19 digit sequences, separators allowed. Candidates only -- Luhn decides.
CARD_CANDIDATE_PATTERN = re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?!\.?\d)")

# US SSN, excluding the ranges the SSA never issues. That exclusion removes a large class of
# false positives from formatted numbers that merely look like SSNs.
SSN_PATTERN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

# Credential shapes. The lookahead requires at least one digit in the tail, which is what
# stops ordinary hyphenated prose ("api-endpoint-configuration") from matching. Key leakage is
# a distinct failure from personal data and gets its own label so it can be handled
# differently -- see :class:`llmeval.guardrails.SecretLeakRule`.
API_KEY_PATTERN = re.compile(
    r"\b(?:sk|pk|api[-_]?key|api|key|token|secret)[-_]"
    r"(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9][A-Za-z0-9_-]{15,}\b",
    re.IGNORECASE,
)


def luhn_valid(digits: str) -> bool:
    """Luhn (mod 10) checksum, per ISO/IEC 7812.

    Walking right to left, double every second digit and subtract 9 when the result exceeds 9;
    a valid number's digits sum to a multiple of 10. Separators are ignored, and anything
    outside 13-19 digits is rejected since no card scheme uses a length outside that range.
    """
    stripped = re.sub(r"[ -]", "", digits)
    if not stripped.isdigit() or not 13 <= len(stripped) <= 19:
        return False
    total = 0
    for index, character in enumerate(reversed(stripped)):
        value = int(character)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


@dataclass(frozen=True)
class PIIMatch:
    """One detected span."""

    kind: str
    value: str
    start: int
    end: int


# Order matters: on an equal span the earlier detector wins the overlap resolution below, so
# the more specific kinds are listed before the more permissive ones.
_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", EMAIL_PATTERN),
    ("ssn", SSN_PATTERN),
    ("api_key", API_KEY_PATTERN),
    ("ipv4", IPV4_PATTERN),
    ("credit_card", CARD_CANDIDATE_PATTERN),
    ("phone", PHONE_PATTERN),
)

PII_KINDS = tuple(kind for kind, _ in _DETECTORS)


def detect_pii(text: str, kinds: tuple[str, ...] | None = None) -> list[PIIMatch]:
    """Find PII spans, ordered by position.

    Card candidates are filtered through :func:`luhn_valid`. Overlapping matches are resolved
    by preferring the earlier span, then the longer one, then the more specific detector --
    which is what stops one number being reported under two labels.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if kinds is not None:
        unknown = set(kinds) - set(PII_KINDS)
        if unknown:
            raise ValueError(
                f"unknown PII kinds {sorted(unknown)}; known kinds are {list(PII_KINDS)}"
            )

    candidates: list[tuple[int, int, int, PIIMatch]] = []
    for priority, (kind, pattern) in enumerate(_DETECTORS):
        if kinds is not None and kind not in kinds:
            continue
        for match in pattern.finditer(text):
            value = match.group(0)
            if kind == "credit_card" and not luhn_valid(value):
                continue
            candidates.append(
                (
                    match.start(),
                    -(match.end() - match.start()),
                    priority,
                    PIIMatch(kind, value, match.start(), match.end()),
                )
            )

    candidates.sort(key=lambda item: item[:3])
    selected: list[PIIMatch] = []
    last_end = -1
    for _, _, _, candidate in candidates:
        if candidate.start >= last_end:
            selected.append(candidate)
            last_end = candidate.end
    return selected


def redact_pii(
    text: str,
    kinds: tuple[str, ...] | None = None,
    template: str = "[REDACTED:{kind}]",
) -> tuple[str, list[PIIMatch]]:
    """Replace detected spans, returning the redacted text and what was removed.

    Replacement runs back to front so earlier offsets stay valid. The kind is kept in the
    placeholder because a downstream reader usually needs to know *what* was removed -- a
    redacted email and a redacted card number mean very different things for triage.
    """
    matches = detect_pii(text, kinds)
    redacted = text
    for match in reversed(matches):
        redacted = redacted[: match.start] + template.format(kind=match.kind) + redacted[match.end :]
    return redacted, matches
