"""PII detection: the specificity claims, and the boundary bug that used to hide cards."""

import pytest

from llmeval.pii import PII_KINDS, detect_pii, luhn_valid, redact_pii

# Well-known test numbers published by the card schemes for exactly this purpose.
VALID_CARDS = (
    "4111111111111111",  # Visa
    "5500005555555559",  # Mastercard
    "378282246310005",   # Amex, 15 digits
    "6011111111111117",  # Discover
)


@pytest.mark.parametrize("number", VALID_CARDS)
def test_luhn_accepts_scheme_test_numbers(number):
    assert luhn_valid(number)


@pytest.mark.parametrize("number", VALID_CARDS)
def test_luhn_rejects_single_digit_corruption(number):
    """The check digit exists to catch exactly this, so it must."""
    corrupted = number[:-1] + str((int(number[-1]) + 1) % 10)
    assert not luhn_valid(corrupted)


def test_luhn_ignores_separators():
    assert luhn_valid("4111 1111 1111 1111")
    assert luhn_valid("4111-1111-1111-1111")


def test_luhn_rejects_wrong_lengths_and_non_digits():
    assert not luhn_valid("411111111111")  # 12 digits, no scheme uses it
    assert not luhn_valid("41111111111111111111")  # 20 digits
    assert not luhn_valid("4111abcd11111111")
    assert not luhn_valid("")


def test_luhn_rejects_most_random_digit_strings():
    """The specificity claim, stated as a property.

    Roughly one in ten random 16-digit strings satisfies Luhn by chance, so a large sample
    must land near 10% -- decisively below the 100% a length-only detector would flag. The
    bound is loose because the point is the order of magnitude, not a precise rate.
    """
    import numpy as np

    rng = np.random.default_rng(17)
    strings = ["".join(str(digit) for digit in rng.integers(0, 10, 16)) for _ in range(4000)]
    rate = sum(luhn_valid(candidate) for candidate in strings) / len(strings)
    assert 0.05 < rate < 0.15


def test_order_id_is_not_reported_as_a_card():
    """A 16-digit identifier that fails Luhn must not be flagged."""
    text = "Your order reference is 1234567890123456, keep it for your records."
    assert not luhn_valid("1234567890123456")
    assert [match.kind for match in detect_pii(text)] == []


def test_card_at_end_of_sentence_is_detected():
    """Regression test for the trailing-full-stop boundary bug.

    A guard of ``(?![\\w.])`` makes the detector miss a card followed by a full stop, which is
    the most natural way for a model to write one. This asserts the fix.
    """
    text = "Confirmed for card 4111 1111 1111 1111."
    kinds = [match.kind for match in detect_pii(text)]
    assert "credit_card" in kinds


def test_card_is_not_also_reported_as_a_phone_number():
    """Overlap resolution: one span, one label, and the more specific label wins."""
    matches = detect_pii("card 4111 1111 1111 1111 on file")
    assert [match.kind for match in matches] == ["credit_card"]


def test_email_and_card_both_found_in_order():
    matches = detect_pii("Ticket for dana.k@example.com, card 4111 1111 1111 1111.")
    assert [match.kind for match in matches] == ["email", "credit_card"]
    assert matches[0].start < matches[1].start


def test_ssn_excludes_never_issued_ranges():
    assert [m.kind for m in detect_pii("SSN 123-45-6789")] == ["ssn"]
    for impossible in ("000-45-6789", "666-45-6789", "900-45-6789", "123-00-6789", "123-45-0000"):
        assert "ssn" not in [m.kind for m in detect_pii(f"SSN {impossible}")]


def test_api_key_detected_with_underscore_prefix():
    """``api_key_...`` is the common shape and must not be defeated by the word boundary."""
    matches = detect_pii("Use api_key_9f8e7d6c5b4a3210ff to authenticate.")
    assert [match.kind for match in matches] == ["api_key"]


def test_api_key_detector_ignores_hyphenated_prose():
    """The digit requirement in the tail is what buys this."""
    assert detect_pii("See the api-endpoint-configuration-guide for details.") == []


def test_ipv4_rejects_out_of_range_octets():
    assert [m.kind for m in detect_pii("host 192.168.1.24")] == ["ipv4"]
    assert "ipv4" not in [m.kind for m in detect_pii("version 999.168.1.24")]


def test_redaction_preserves_surrounding_text_and_labels_kind():
    redacted, matches = redact_pii("Mail dana.k@example.com now")
    assert redacted == "Mail [REDACTED:email] now"
    assert len(matches) == 1


def test_redaction_handles_multiple_spans_without_offset_drift():
    """Back-to-front replacement is what keeps the later offsets valid."""
    text = "a@b.com and c@d.com and e@f.com"
    redacted, matches = redact_pii(text)
    assert len(matches) == 3
    assert redacted == "[REDACTED:email] and [REDACTED:email] and [REDACTED:email]"


def test_redaction_is_idempotent():
    once, _ = redact_pii("mail dana.k@example.com")
    twice, matches = redact_pii(once)
    assert twice == once and matches == []


def test_clean_text_is_untouched():
    text = "Your invoice for March is attached."
    redacted, matches = redact_pii(text)
    assert redacted == text and matches == []


def test_kind_filter_restricts_detectors():
    text = "dana.k@example.com and card 4111 1111 1111 1111."
    assert [m.kind for m in detect_pii(text, kinds=("email",))] == ["email"]
    assert [m.kind for m in detect_pii(text, kinds=("credit_card",))] == ["credit_card"]


def test_unknown_kind_is_rejected_rather_than_silently_ignored():
    """A typo in a kind name must not quietly disable a detector."""
    with pytest.raises(ValueError, match="unknown PII kinds"):
        detect_pii("anything", kinds=("emails",))


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        detect_pii(None)


def test_detector_kinds_are_exposed():
    assert "email" in PII_KINDS and "credit_card" in PII_KINDS
