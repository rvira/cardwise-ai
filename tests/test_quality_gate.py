"""Regression guards for the data-quality gate (no API/embedding cost).

The gate's whole job is to *fail closed*: a PDF that mentions rewards in a
disclaimer but carries no concrete earning rate must NOT be embedded. These
tests pin that behaviour so a future loader change can't silently weaken it.
"""
import pytest

from src.ingestion.loader import CardDocument, find_rates, quality_check


def _doc(raw_text: str) -> CardDocument:
    return CardDocument(
        card_name="Test Card",
        card_id="test_card",
        issuer="Test Bank",
        card_type="cashback",
        annual_fee=499,
        raw_text=raw_text,
        source_path="data/raw/test.pdf",
    )


# --- find_rates: detects a concrete number+unit, not just the word "reward" ---

@pytest.mark.parametrize(
    "text",
    [
        "Earn 5% cashback on online spends.",
        "Get 10X Reward Points on partner merchants.",
        "Welcome benefit worth ₹500.",
        "1.5 % unlimited cashback.",
    ],
)
def test_find_rates_detects_concrete_rate(text):
    assert find_rates(text), f"expected a concrete rate in: {text!r}"


def test_find_rates_rejects_rateless_disclaimer():
    # mentions rewards/cashback as words but states no earning rate at all
    text = "This card offers rewards and cashback benefits. Terms and conditions apply."
    assert find_rates(text) == []


def test_find_rates_input_is_length_capped():
    # huge input must not hang or error (ReDoS guard: patterns are linear + capped)
    big = "no rate here " * 100_000
    assert find_rates(big) == []


# --- quality_check: the pass/fail contract -----------------------------------

def test_quality_check_passes_with_chars_and_rate():
    doc = _doc("Axis ACE gives 2% cashback. " + ("filler text " * 200))
    result = quality_check(doc)
    assert result["passed"] is True
    assert result["concrete_rates"]


def test_quality_check_fails_when_too_short():
    doc = _doc("5% cashback")  # has a rate but well under the 1000-char floor
    assert quality_check(doc)["passed"] is False


def test_quality_check_fails_without_concrete_rate():
    doc = _doc("This card mentions rewards and cashback. " + ("padding " * 300))
    assert quality_check(doc)["passed"] is False


def test_strict_mode_fails_closed():
    # the production-critical assertion: strict mode must RAISE, not return,
    # when the gate fails — so bad data can never flow into the vector store.
    doc = _doc("Just a generic card description. " + ("padding " * 300))
    with pytest.raises(ValueError):
        quality_check(doc, strict=True)


def test_strict_mode_passes_silently_on_good_doc():
    doc = _doc("HDFC Millennia: 5% cashback online. " + ("filler " * 200))
    result = quality_check(doc, strict=True)
    assert result["passed"] is True
