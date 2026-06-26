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

    def retrieve(self, query: str, k: int = 6):
        semantic = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k * 2
        )
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: -x[1])[: k * 2]

        combined = {}
        for doc, score in semantic:
            key = doc.page_content[:100]
            combined[key] = {"doc": doc, "score": self.alpha * score}
        for idx, score in bm25_ranked:
            doc = self.all_chunks[idx]
            key = doc.page_content[:100]
            norm = score / (max(bm25_scores) + 1e-6)
            if key in combined:
                combined[key]["score"] += (1 - self.alpha) * norm
            else:
                combined[key] = {"doc": doc, "score": (1 - self.alpha) * norm}
        return [
            v["doc"] for v in sorted(combined.values(), key=lambda x: -x["score"])[:k]
        ]


def stratified_retrieve(
    hybrid_retriever, query: str, card_names: List[str], k_per_card: int = 3
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


def as_runnable(retriever, k: int = 6) -> RunnableLambda:
    """Wrap a HybridRetriever's .retrieve so it drops into build_card_rag_chain
    exactly like a normal LangChain retriever (the chain pipes a query string in)."""
    return RunnableLambda(lambda q: retriever.retrieve(q, k=k))
