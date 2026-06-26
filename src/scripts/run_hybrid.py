# src/scripts/run_hybrid.py
"""Verify hybrid + stratified retrieval are live.

  - stratified_retrieve should return chunks from ALL cards on a
    comparison query (fixes the cross-card bias).
  - the RAG chain answers using the hybrid BM25+semantic retriever,
    which handles exact-value queries (e.g. a specific fee) better than pure
    semantic search.

Run from project root:  python -m src.scripts.run_hybrid
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dotenv import load_dotenv

load_dotenv()

from src.vectorstore.embedder import load_vectorstore
from src.retrieval.retriever import (
    build_hybrid_retriever,
    stratified_retrieve,
    as_runnable,
)
from src.rag.chain import build_card_rag_chain

CARD_NAMES = ["SBI SimplyCLICK", "Axis ACE", "HDFC Millennia"]


def main():
    vs = load_vectorstore()
    hybrid = build_hybrid_retriever(vs, alpha=0.5)

    # --- AI-2.4: stratified retrieval must surface every card ---
    comparison_q = "Which card gives the best rewards on online spends?"
    docs = stratified_retrieve(hybrid, comparison_q, CARD_NAMES, k_per_card=2)
    represented = sorted({d.metadata.get("card_name") for d in docs})
    print(f"\n[stratified] cards represented for a comparison query: {represented}")
    print(f"[stratified] all 3 present? {set(represented) == set(CARD_NAMES)}")

    # --- AI-2.3: hybrid retriever wired into the chain ---
    chain = build_card_rag_chain(vs, retriever=as_runnable(hybrid, k=6))
    exact_q = "What is the exact annual fee for HDFC Millennia?"
    print(f"\n[hybrid chain] Q: {exact_q}\n{'-' * 60}")
    print(chain.invoke(exact_q))


if __name__ == "__main__":
    main()
