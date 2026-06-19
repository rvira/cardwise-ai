# scripts/ingest_all.py
from dotenv import load_dotenv

# Load GOOGLE_API_KEY (and friends) from .env before any Gemini client is built.
# The key is read from the environment — never hardcoded.
load_dotenv()

from src.ingestion.loader import load_card_pdf, quality_check
from src.ingestion.chunker import chunk_document
from src.vectorstore.embedder import build_vectorstore

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


def ingest():
    for card in CARDS:
        doc = load_card_pdf(
            card["pdf"],
            card["card_name"],
            card["issuer"],
            card["card_type"],
            card["annual_fee"],
        )
        qc = quality_check(doc)
        print(f"\n{card['card_name']}: {qc}")
        # recursive_800 — validated in Week 1; cheap, no embedding quota spent on chunking
        chunks = chunk_document(doc.raw_text)
        build_vectorstore(chunks, card)

    print(f"\nAll {len(CARDS)} cards ingested successfully.")


if __name__ == "__main__":
    ingest()
