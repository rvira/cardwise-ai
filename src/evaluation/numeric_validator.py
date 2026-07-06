# src/evaluation/numeric_validator.py
"""Numeric guardrail for CardWise: claim extraction (answer-time) plus the
eval-time numeric-hit check used by the gold-set evaluation."""

import re

# Patterns are deliberately simple and linear (no nested quantifiers); input is
# length-capped before matching to avoid catastrophic backtracking (ReDoS).
_PATTERNS = [
    r"\d+\s?x\b",  # "10x", "5 X" reward multipliers
    r"\$\d+(?:\.\d{2})?",  # "$395", "$4.50"
    r"₹\s?\d[\d,]*",  # "₹500", "₹1,000"  (Indian cards)
    r"\d+(?:\.\d+)?\s?%",  # "5%", "1.5 %"
    r"\d+\s+reward points",  # "5 Reward Points"
]


def extract_numeric_claims(text: str) -> list:
    """Return every numeric claim found in the answer text. Extraction only —
    validation against the source documents lands in Week 3."""
    capped = text[:200_000]
    claims = []
    for pattern in _PATTERNS:
        claims.extend(re.findall(pattern, capped, flags=re.IGNORECASE))
    return claims


def _norm(text: str) -> str:
    """Lower-case and strip currency symbols, but KEEP commas and % so that
    '10,000' stays distinct from '1,000' and '4%' matches exactly."""
    return " ".join(
        text.lower().replace("rs.", "").replace("₹", "").replace("$", "").split()
    )


def numeric_hit(contexts: list, expected_values: list) -> bool | None:
    """Deterministic 'zero fabricated fees' guard for the gold-set eval.

    Returns True iff every expected numeric token (e.g. '4%', '250', '10,000')
    appears somewhere in the retrieved context — i.e. the number the answer needs
    was actually retrieved. Returns None for non-numeric questions (no expected
    values), so they are excluded from numeric-exact accuracy rather than counted
    as failures.
    """
    if not expected_values:
        return None
    blob = _norm(" ".join(contexts))
    return all(_norm(v) in blob for v in expected_values)


if __name__ == "__main__":
    # Quick self-check (no API). Run: python -m src.evaluation.numeric_validator
    sample = (
        "HDFC Millennia gives 5% cashback on online spends and 1% otherwise, "
        "capped at ₹1,000. SBI SimplyCLICK earns 10X Reward Points online. "
        "Example: ₹2,000 x 5% = ₹100 cashback."
    )
    print("Extracted claims:", extract_numeric_claims(sample))
