# scripts/ingest_all.py
import sys
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.loader import CARDS, load_card, quality_check
from src.ingestion.chunker import chunk_document
from src.vectorstore.embedder import build_vectorstore


def ingest(check_only: bool = False):
    """Load + gate every card. When check_only=True, run ONLY the strict
    quality_check and skip chunking/embedding — so no Gemini embedding quota
    is spent. Used to validate PDFs before committing to a full ingest."""
    for card in CARDS:
        doc = load_card(card)
        # Rung 3: strict gate — raises if a PDF lacks a concrete reward rate
        quality_check(doc, strict=True)
        if check_only:
            continue
        # recursive_800 — validated in Week 1; cheap, no embedding quota spent on chunking
        chunks = chunk_document(doc.raw_text)
        build_vectorstore(chunks, card)

    if check_only:
        print(
            f"\nQuality gate PASSED for all {len(CARDS)} cards (no embedding performed)."
        )
    else:
        print(f"\nAll {len(CARDS)} cards ingested successfully.")


if __name__ == "__main__":
    ingest(check_only="--check-only" in sys.argv)
