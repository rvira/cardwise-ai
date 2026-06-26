"""Regression guards for hybrid + stratified retrieval (no API cost).

We stub the vector store so BM25's contribution can be tested in isolation:
the whole point of going hybrid was that pure semantic search ranks "$395" and
"$350" similarly, while exact-token (BM25) matching does not. These tests pin
that the merge actually surfaces the exact-value chunk, and that stratified
retrieval skips a card with zero chunks instead of erroring.
"""

from langchain_core.documents import Document

from src.retrieval.retriever import HybridRetriever, stratified_retrieve, tokenize


class FakeVectorStore:
    """Minimal stand-in for Chroma: returns every doc with an identical, flat
    semantic score. That isolates the test to BM25's exact-match contribution —
    if ranking still works, it's BM25 doing the work, not the (flat) semantic leg.
    """

    def __init__(self, docs):
        self._docs = docs

    def similarity_search_with_relevance_scores(self, query, k):
        return [(d, 0.5) for d in self._docs[:k]]


def _chunk(text, card_name="Axis ACE"):
    return Document(page_content=text, metadata={"card_name": card_name})


def test_tokenize_lowercases_and_splits():
    assert tokenize("Annual Fee $395!") == ["annual", "fee", "395"]


def test_hybrid_surfaces_exact_value_match():
    # A realistic, varied corpus (not three near-identical lines): the
    # distinguishing token "395" then has positive IDF, so BM25 can do its job.
    chunks = [
        _chunk("The annual fee is $395 per year."),
        _chunk("The annual fee is $350 per year."),
        _chunk("The annual fee is $450 per year."),
        _chunk("Earn 3x points on dining and travel."),
        _chunk("Airport lounge access is included worldwide."),
        _chunk("No foreign transaction charges apply abroad."),
    ]
    vs = FakeVectorStore(chunks)
    # alpha=0.3 → semantic leg is flat (no tie-break), BM25 leg decides ranking
    retriever = HybridRetriever(vs, chunks, alpha=0.3)

    results = retriever.retrieve("what is the $395 annual fee", k=3)

    assert (
        "$395" in results[0].page_content
    ), "exact-value query should rank the $395 chunk first via BM25"


def test_hybrid_respects_k():
    chunks = [_chunk(f"reward rate number {i} percent cashback") for i in range(10)]
    vs = FakeVectorStore(chunks)
    retriever = HybridRetriever(vs, chunks, alpha=0.5)
    assert len(retriever.retrieve("reward rate", k=4)) <= 4


def test_stratified_skips_card_with_zero_chunks():
    # only Axis chunks exist; asking for a card with no chunks must not raise.
    chunks = [_chunk("2% cashback", card_name="Axis ACE")]
    vs = FakeVectorStore(chunks)
    retriever = HybridRetriever(vs, chunks, alpha=0.5)

    # the missing card alone -> empty result, no exception (the `continue` guard)
    assert stratified_retrieve(retriever, "rewards", ["Nonexistent Card"]) == []

    # a present card still returns chunks
    present = stratified_retrieve(retriever, "rewards", ["Axis ACE"], k_per_card=1)
    assert present and all(d.metadata["card_name"] == "Axis ACE" for d in present)
