import fitz  # PyMuPDF
from dataclasses import dataclass


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


def quality_check(doc) -> dict:
    t = doc.raw_text
    low = t.lower()  # lowercase so 'X points' / 'Miles' also match
    result = {
        "total_chars": len(t),
        "reward_mentions": (
            low.count("x points")
            + low.count("% cash back")
            + low.count("miles per")
            + low.count("points per")
            + low.count("per $1")
        ),
        "annual_fee_mentions": low.count("annual fee"),
        "fx_fee_mentions": low.count("foreign transaction"),
        "has_exclusions": "exclud" in low,
    }
    print(result)
    return result
