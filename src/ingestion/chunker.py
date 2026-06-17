from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings


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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
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
