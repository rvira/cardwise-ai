import re
import fitz
from dataclasses import dataclass

RATE_PATTERNS = [
    r"\d{1,3}\s?%",  # "5%", "1.5 %"
    r"\d{1,2}\s?x\s+reward",  # "10X Reward", "5x reward"
    r"\d{1,3}\s+reward points",  # "5 Reward Points"
    r"₹\s?\d",  # "₹500"
]


def find_rates(text: str) -> list:
    """Return which rate patterns actually appear — proves a concrete earning
    rate exists, not merely the word 'cashback' in a disclaimer."""
    low = text[:200_000].lower()  # cap input length as a guard
    return [p for p in RATE_PATTERNS if re.search(p, low)]


@dataclass
class CardDocument:
    card_name: str
    issuer: str
    card_type: str  # 'travel' | 'cashback' | 'dining'
    annual_fee: int
    raw_text: str
    source_path: str


def load_card_pdf(pdf_path, card_name, issuer, card_type, annual_fee):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text", sort=True)  # sort=True preserves reading order
        pages.append(f"[PAGE {i+1}]\n{text}")
    return CardDocument(
        card_name=card_name,
        issuer=issuer,
        card_type=card_type,
        annual_fee=annual_fee,
        raw_text="\n\n".join(pages),
        source_path=pdf_path,
    )


CARDS = [
    {
        "pdf": "data/raw/sbi_simplyclick.pdf",
        "card_name": "SBI SimplyCLICK",
        "issuer": "SBI Card",
        "card_type": "online-shopping",
        "annual_fee": 499,
    },
    {
        "pdf": "data/raw/axis_ace.pdf",
        "card_name": "Axis ACE",
        "issuer": "Axis Bank",
        "card_type": "cashback",
        "annual_fee": 499,
    },
    {
        "pdf": "data/raw/hdfc_millennia.pdf",
        "card_name": "HDFC Millennia",
        "issuer": "HDFC Bank",
        "card_type": "cashback",
        "annual_fee": 1000,
    },
]


def load_card(card: dict) -> CardDocument:
    """Load a CardDocument from a CARDS catalog entry (a dict carrying the PDF
    path + metadata). The one place the catalog dict is unpacked into the
    positional loader call, so callers don't each repeat the key mapping."""
    return load_card_pdf(
        card["pdf"],
        card["card_name"],
        card["issuer"],
        card["card_type"],
        card["annual_fee"],
    )


def get_card(card_name: str) -> dict:
    """Look up a single catalog entry by card name (for scripts that operate on
    one card, e.g. the chunker benchmark)."""
    for card in CARDS:
        if card["card_name"] == card_name:
            return card
    raise KeyError(f"No card named {card_name!r} in CARDS catalog")


def quality_check(doc, strict: bool = False) -> dict:
    t = doc.raw_text
    low = t.lower()  # lowercase so 'X points' / 'Miles' also match
    concrete_rates = find_rates(t)
    result = {
        "total_chars": len(t),
        "reward_mentions": (
            # Indian-card phrasing (HDFC/Axis/SBI): "cashback", "Reward Points", "5X/10X Reward"
            low.count("reward point")
            + low.count("cashback")
            + low.count("cash back")
            + low.count("x reward")
            # US-card phrasing kept as fallback
            + low.count("x points")
            + low.count("miles per")
            + low.count("per $1")
        ),
        "concrete_rates": concrete_rates,  # Rung 2: actual number+unit patterns found
        "annual_fee_mentions": low.count("annual fee"),
        "fx_fee_mentions": low.count("foreign transaction"),
        "has_exclusions": "exclud" in low,
    }
    result["passed"] = result["total_chars"] > 1000 and bool(concrete_rates)
    print(result)

    if strict and not result["passed"]:
        raise ValueError(
            f"{doc.card_name}: quality gate failed "
            f"(chars={result['total_chars']}, concrete_rates={concrete_rates}). "
            f"Fix the source PDF/loader before ingesting."
        )
    return result
