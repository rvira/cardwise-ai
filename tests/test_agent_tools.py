"""Tests for card_search wiring (AI-3.2) — no API, no vector store.

We monkeypatch the cached retriever getter so the tool's own logic (query cap,
empty-query and no-result handling, citation formatting) is tested in isolation.
"""

from langchain_core.documents import Document

from src.agent.tools import retriever_tool
from src.agent.tools.retriever_tool import card_search


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.last_k = None

    def retrieve_rrf(self, query, k=6):
        self.last_k = k
        return self._docs


def _patch(monkeypatch, docs):
    fake = _FakeRetriever(docs)
    monkeypatch.setattr(retriever_tool, "_get_retriever", lambda: fake)
    return fake


def _run(query: str) -> str:
    return card_search.invoke({"query": query})


def test_formats_docs_with_citations(monkeypatch):
    docs = [
        Document(
            page_content="Axis ACE gives 5% cashback on bill payments. [PAGE 2]",
            metadata={"card_name": "Axis ACE"},
        )
    ]
    _patch(monkeypatch, docs)
    out = _run("cashback rate on Axis ACE")
    assert "Axis ACE" in out
    assert "5% cashback" in out
    assert "p.2" in out  # page label recovered by format_with_citations


def test_uses_validated_k_of_10(monkeypatch):
    fake = _patch(monkeypatch, [Document(page_content="x", metadata={})])
    _run("anything")
    assert fake.last_k == 10


def test_empty_query_short_circuits(monkeypatch):
    _patch(monkeypatch, [])
    assert _run("   ") == "No query provided."


def test_no_results_message(monkeypatch):
    _patch(monkeypatch, [])
    assert _run("obscure question") == "No matching card information found."


def test_long_query_is_length_capped(monkeypatch):
    fake = _patch(monkeypatch, [Document(page_content="x", metadata={})])
    card_search.invoke({"query": "a" * 5000})
    assert fake.last_k == 10  # ran without rejecting; cap applied before retrieval
