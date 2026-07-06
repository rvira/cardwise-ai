"""Regression guards for hybrid + stratified retrieval (no API cost).

We stub the vector store so BM25's contribution can be tested in isolation:
the whole point of going hybrid was that pure semantic search ranks "$395" and
"$350" similarly, while exact-token (BM25) matching does not. These tests pin
that the merge actually surfaces the exact-value chunk, and that stratified
retrieval skips a card with zero chunks instead of erroring.
"""

from langchain_core.documents import Document

from src.retrieval.retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    stratified_retrieve,
    tokenize,
)


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


# --- Reciprocal Rank Fusion (AI-2.2) ------------------------------------------
# RRF fuses ranked lists by *position only* (no score normalization). These pin
# the three properties that make it the right merge for CardWise and the tunable
# knob (rrf_k) the checklist asks us to sweep.


def _doc(text, chunk_id):
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


def test_rrf_favors_doc_ranked_high_in_both_lists():
    # `shared` is rank 0 in BOTH lists -> 2/(rrf_k+1); any doc appearing in a
    # single list scores at most 1/(rrf_k+1), so `shared` must come first.
    shared = _doc("shared top hit", "shared")
    list_a = [shared, _doc("a1", "a1"), _doc("a2", "a2")]
    list_b = [shared, _doc("b1", "b1"), _doc("b2", "b2")]

    fused = reciprocal_rank_fusion([list_a, list_b], rrf_k=60, top_n=5)

    assert fused[0].metadata["chunk_id"] == "shared"


def test_rrf_dedups_by_chunk_id_not_object_identity():
    # Two DISTINCT Document objects sharing a chunk_id are the same chunk; RRF
    # must collapse them into one fused (and boosted) entry, not double-count.
    a_view = _doc("annual fee v1", "c1")
    b_view = _doc("annual fee v2", "c1")  # same id, different object/text
    list_a = [a_view, _doc("other-a", "a")]
    list_b = [b_view, _doc("other-b", "b")]

    fused = reciprocal_rank_fusion([list_a, list_b], rrf_k=60, top_n=5)

    c1_hits = [d for d in fused if d.metadata["chunk_id"] == "c1"]
    assert len(c1_hits) == 1, "same chunk_id must appear exactly once"
    assert fused[0].metadata["chunk_id"] == "c1", "appearing in both lists boosts it first"


def test_rrf_respects_top_n():
    lists = [[_doc(f"d{i}", f"d{i}") for i in range(10)]]
    assert len(reciprocal_rank_fusion(lists, top_n=3)) == 3


def test_rrf_k_constant_changes_ranking():
    # Small rrf_k rewards a strong single-list top rank; large rrf_k flattens
    # ranks so a doc present in BOTH lists (at middling ranks) overtakes it.
    #   top_a:  rank 0 in A, rank 5 in B
    #   midboth: rank 2 in A, rank 1 in B
    top_a = _doc("top of A", "top_a")
    midboth = _doc("middle of both", "midboth")
    list_a = [top_a, _doc("fa1", "fa1"), midboth]
    list_b = [_doc("fb0", "fb0"), midboth, _doc("fb2", "fb2"),
              _doc("fb3", "fb3"), _doc("fb4", "fb4"), top_a]

    small = reciprocal_rank_fusion([list_a, list_b], rrf_k=1, top_n=2)
    large = reciprocal_rank_fusion([list_a, list_b], rrf_k=100, top_n=2)

    assert small[0].metadata["chunk_id"] == "top_a"
    assert large[0].metadata["chunk_id"] == "midboth"


def test_retrieve_rrf_surfaces_exact_value_match():
    # Same exact-value scenario as the weighted test, but via the RRF path: the
    # "$395" chunk is rank 0 on the BM25 leg, so RRF must rank it first.
    chunks = [
        _chunk("The annual fee is $395 per year."),
        _chunk("The annual fee is $350 per year."),
        _chunk("The annual fee is $450 per year."),
        _chunk("Earn 3x points on dining and travel."),
        _chunk("Airport lounge access is included worldwide."),
        _chunk("No foreign transaction charges apply abroad."),
    ]
    vs = FakeVectorStore(chunks)
    retriever = HybridRetriever(vs, chunks)

    results = retriever.retrieve_rrf("what is the $395 annual fee", k=3)

    assert "$395" in results[0].page_content


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
