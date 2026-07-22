"""Tool #1 — the agent's factual-lookup tool: a thin wrapper over the existing
hybrid (BM25 + semantic) retriever.

The docstring on ``card_search`` below IS the routing prompt — the LLM decides
whether to call this tool by reading it, so it states exactly what this tool is for
and (just as importantly) what it is not. A vague docstring is a routing bug.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.rag.chain import format_with_citations
from src.retrieval.retriever import build_hybrid_retriever
from src.vectorstore.embedder import load_vectorstore

# k=10 RRF is the config validated in the gold-set eval (context recall 0.883).
_RETRIEVAL_K = 10
# The query originates from the user via the model — cap it like app.py does.
_MAX_QUERY_LEN = 500

_retriever = None


def _get_retriever():
    """Lazily build and cache the hybrid retriever. Building it loads every chunk
    and constructs the BM25 index, so do it once per process — and lazily, so
    importing this module needs no API key or vector store (keeps tests fast)."""
    global _retriever
    if _retriever is None:
        _retriever = build_hybrid_retriever(load_vectorstore())
    return _retriever


@tool
def card_search(query: str) -> str:
    """Look up a single FACTUAL detail about a card from its official document: a
    reward/cashback rate, a cap, the annual fee, an exclusion, or eligibility terms.

    Use this only to retrieve a stated fact (e.g. "what is the cashback rate on Axis
    ACE?", "does SBI SimplyCLICK exclude rent?"). Do NOT use it to compute or compare
    how much cards earn for a spending pattern — for "which card earns me more?" or
    "how much will I get?", use rewards_value instead.
    """
    query = (query or "").strip()[:_MAX_QUERY_LEN]
    if not query:
        return "No query provided."
    docs = _get_retriever().retrieve_rrf(query, k=_RETRIEVAL_K)
    if not docs:
        return "No matching card information found."
    return format_with_citations(docs)
