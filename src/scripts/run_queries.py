# scripts/run_queries.py
"""Run sample RAG queries against the persisted Chroma store.

Prerequisite: build the store first with `python -m src.scripts.ingest_all`.
Run from the project root:  python -m src.scripts.run_queries
"""

from dotenv import load_dotenv

# Load GOOGLE_API_KEY from .env before any Gemini client is built — never hardcoded.
load_dotenv()

from src.rag.chain import build_card_rag_chain
from src.vectorstore.embedder import load_vectorstore

# Questions target the cards actually ingested (the Indian set in ingest_all.py).
QUESTIONS = [
    "What reward rate does the SBI SimplyCLICK card offer on online spends?",
    "What is the cashback rate for the Axis ACE card, and are there any exclusions?",
    "What is the annual fee for the HDFC Millennia card?",
    "Does the Axis ACE card charge foreign transaction fees?",
    "What is the reward rate on the HDFC Millennia card for online shopping?",
]


def main():
    vectorstore = load_vectorstore()
    chain = build_card_rag_chain(vectorstore)
    for q in QUESTIONS:
        print(f"\n{'=' * 80}\nQ: {q}\n{'-' * 80}")
        print(chain.invoke(q))


if __name__ == "__main__":
    main()
