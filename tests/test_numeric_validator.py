"""Regression guards for numeric-claim extraction (pure regex, no API cost).

This is the first layer of the hallucination guardrail: before an answer can be
checked against the source docs (Week 3), every numeric claim in it must be
reliably extracted. These tests pin which shapes are caught today and document
the known gaps (words like "five percent") as explicit TODOs, not surprises.
"""
import pytest

from src.evaluation.numeric_validator import extract_numeric_claims


@pytest.mark.parametrize(
    "text,expected_substr",
    [
        ("Earn 5% cashback online.", "5%"),
        ("Annual fee is $395 this year.", "$395"),
        ("Capped at ₹1,000 per month.", "₹1,000"),
        ("Get 10X reward points on travel.", "10X"),
        ("Earn 5 Reward Points per spend.", "5 Reward Points"),
    ],
)
def test_extracts_each_claim_shape(text, expected_substr):
    claims = extract_numeric_claims(text)
    assert any(expected_substr.lower() in c.lower() for c in claims), (
        f"expected to extract {expected_substr!r} from {text!r}, got {claims}"
    )


def test_extracts_multiple_claims_from_one_answer():
    text = (
        "HDFC Millennia gives 5% cashback online and 1% otherwise, "
        "capped at ₹1,000. SBI SimplyCLICK earns 10X Reward Points."
    )
    claims = extract_numeric_claims(text)
    assert len(claims) >= 4


def test_no_claims_in_plain_prose():
    text = "This card is great for everyday spending and offers solid rewards."
    assert extract_numeric_claims(text) == []


def test_case_insensitive_multiplier():
    # both "10x" and "10X" are caught (matching is case-insensitive); the
    # returned text preserves the original casing, so compare normalized.
    lower = extract_numeric_claims("earn 10x rewards")
    upper = extract_numeric_claims("earn 10X rewards")
    assert [c.lower() for c in lower] == [c.lower() for c in upper] == ["10x"]


def test_input_is_length_capped():
    # ReDoS / resource-exhaustion guard: large input must not hang.
    big = "no numbers here " * 100_000
    assert extract_numeric_claims(big) == []


def test_known_gap_word_form_numbers_are_missed():
    # Documents a deliberate Week-3 gap: word-form numbers ("five percent")
    # are NOT extracted by the regex layer. If this ever starts passing,
    # the validator improved and this test should be updated.
    assert extract_numeric_claims("earn five percent cashback") == []
