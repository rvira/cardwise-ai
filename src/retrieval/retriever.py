# src/retrieval/retriever.py
from rank_bm25 import BM25Okapi
from typing import List
import re


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
