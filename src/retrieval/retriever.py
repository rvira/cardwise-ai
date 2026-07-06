# src/retrieval/retriever.py
from rank_bm25 import BM25Okapi
from typing import List
import re

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self, vectorstore, all_chunks, alpha: float = 0.5):
        self.vectorstore = vectorstore
        self.all_chunks = all_chunks
        self.alpha = alpha
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in all_chunks])

    def _candidates(self, query: str, fetch_k: int):
        """Shared candidate generation for both fusion strategies.
        Returns (dense, sparse) as ranked lists of (doc, score)."""
        dense = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k
        )
        bm25_scores = self.bm25.get_scores(tokenize(query))
        sparse = [
            (self.all_chunks[idx], score)
            for idx, score in sorted(enumerate(bm25_scores), key=lambda x: -x[1])[
                :fetch_k
            ]
        ]
        return dense, sparse

    def retrieve(self, query: str, k: int = 6):
        dense, sparse = self._candidates(query, fetch_k=k * 2)
        combined = {}
        for doc, score in dense:
            key = doc.metadata.get("chunk_id") or doc.page_content[:100]
            combined[key] = {"doc": doc, "score": self.alpha * score}
        max_bm25 = max((s for _, s in sparse), default=0.0)
        for doc, score in sparse:
            key = doc.metadata.get("chunk_id") or doc.page_content[:100]
            norm = score / (max_bm25 + 1e-6)
            if key in combined:
                combined[key]["score"] += (1 - self.alpha) * norm
            else:
                combined[key] = {"doc": doc, "score": (1 - self.alpha) * norm}
        return [
            v["doc"] for v in sorted(combined.values(), key=lambda x: -x["score"])[:k]
        ]

    def retrieve_rrf(self, query: str, k: int = 6, rrf_k: int = 60):
        dense, sparse = self._candidates(query, fetch_k=k * 2)
        return reciprocal_rank_fusion(
            [[doc for doc, _ in dense], [doc for doc, _ in sparse]],
            rrf_k=rrf_k,
            top_n=k,
        )


def reciprocal_rank_fusion(result_lists, rrf_k: int = 60, top_n: int = 6):
    """Fuse ranked [Document] lists by rank only (no score normalization).
    rrf_k = RRF smoothing constant; larger = flatter rank weighting."""
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            did = doc.metadata.get("chunk_id") or doc.page_content[:80]
            docs[did] = doc
            scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[did] for did in ranked[:top_n]]


def stratified_retrieve(
    hybrid_retriever, query, card_names, k_per_card=3, fusion="rrf", rrf_k=60
):
    all_results = []
    for card in card_names:
        card_chunks = [
            c
            for c in hybrid_retriever.all_chunks
            if c.metadata.get("card_name") == card
        ]
        if not card_chunks:
            continue
        card_retriever = HybridRetriever(
            hybrid_retriever.vectorstore, card_chunks, alpha=hybrid_retriever.alpha
        )
        if fusion == "rrf":
            all_results.extend(
                card_retriever.retrieve_rrf(query, k=k_per_card, rrf_k=rrf_k)
            )
        else:
            all_results.extend(card_retriever.retrieve(query, k=k_per_card))
    return all_results


DEFAULT_SEARCH_KWARGS = {"k": 6, "fetch_k": 20, "lambda_mult": 0.7}


def build_default_retriever(vectorstore, **overrides):
    """MMR retriever over the store — the default used when no custom retriever
    (e.g. the hybrid one) is supplied. Pass kwargs to override individual
    search params (e.g. build_default_retriever(vs, k=10))."""

    search_kwargs = {**DEFAULT_SEARCH_KWARGS, **overrides}
    return vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def load_all_chunks(vectorstore) -> List[Document]:
    """Pull every stored chunk back out of Chroma as LangChain Documents.
    BM25 needs the raw text, which already lives in the vector store after ingest."""

    raw = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=d, metadata=m or {})
        for d, m in zip(raw["documents"], raw["metadatas"])
    ]


def build_hybrid_retriever(vectorstore, alpha: float = 0.5) -> "HybridRetriever":
    """Build a HybridRetriever from a persisted store (loads all chunks for BM25)."""
    return HybridRetriever(vectorstore, load_all_chunks(vectorstore), alpha=alpha)


def as_runnable(retriever, k: int = 6, fusion: str = "rrf", rrf_k: int = 60):
    if fusion == "rrf":
        return RunnableLambda(lambda q: retriever.retrieve_rrf(q, k=k, rrf_k=rrf_k))
    return RunnableLambda(lambda q: retriever.retrieve(q, k=k))
