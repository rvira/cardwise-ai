import time

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class RetryingEmbeddings(Embeddings):
    """Wrap an embeddings backend with exponential backoff on rate-limit (HTTP 429
    / RESOURCE_EXHAUSTED) responses. SemanticChunker embeds every sentence in a
    burst, which can trip the Gemini per-minute quota; backing off lets the run
    recover instead of crashing."""

    def __init__(self, inner: Embeddings, max_retries: int = 5, base_delay: float = 2.0):
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay

    def _with_retry(self, fn, *args):
        delay = self._base_delay
        for attempt in range(self._max_retries):
            try:
                return fn(*args)
            except Exception as ex:  # noqa: BLE001 — re-raised below if not rate-limit
                msg = str(ex)
                is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg
                if not is_rate_limit or attempt == self._max_retries - 1:
                    raise
                print(f"  rate-limited, backing off {delay:.0f}s (attempt {attempt + 1})")
                time.sleep(delay)
                delay *= 2

    def embed_documents(self, texts):
        return self._with_retry(self._inner.embed_documents, texts)

    def embed_query(self, text):
        return self._with_retry(self._inner.embed_query, text)


def benchmark_chunking(raw_text: str) -> dict:
    results = {}

    # Strategy 1: Fixed-size (baseline — expect poor table handling)
    fixed = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    results["fixed_512"] = fixed.create_documents([raw_text])

    # Strategy 2: Recursive with overlap (better for T&C prose)
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=160, separators=["\n\n", "\n", ". ", " ", ""]
    )
    results["recursive_800"] = recursive.create_documents([raw_text])

    # Strategy 3: Semantic (best coherence for reward tables)
    # GOOGLE_API_KEY is read from the environment (.env) by the client — never hardcode it
    embeddings = RetryingEmbeddings(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )
    semantic = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    results["semantic"] = semantic.create_documents([raw_text])

    for name, chunks in results.items():
        reward_chunks = [
            c
            for c in chunks
            if "x points" in c.page_content
            or "% cash back" in c.page_content
            or "per $1" in c.page_content
        ]
        print(
            f"{name}: {len(chunks)} chunks | {len(reward_chunks)} contain reward data"
        )
        if reward_chunks:
            print(f"  Sample: {reward_chunks[0].page_content[:300]}\n")

    return results
