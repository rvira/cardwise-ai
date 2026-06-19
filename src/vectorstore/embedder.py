from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDoc
from typing import List


def build_vectorstore(
    chunks: List[LCDoc], card_meta: dict, persist_dir="chroma_db"
) -> Chroma:
    # Inject structured metadata — this enables filter-by-card in Week 2
    for chunk in chunks:
        chunk.metadata.update(
            {
                "card_name": card_meta["card_name"],  # 'Chase Sapphire Reserve'
                "issuer": card_meta["issuer"],  # 'Chase'
                "card_type": card_meta["card_type"],  # 'travel'
                "annual_fee": card_meta["annual_fee"],  # 550
                # optional — empty string (not None) so Chroma accepts the metadata
                "signup_bonus": card_meta.get("signup_bonus", ""),
            }
        )
    # Must be the SAME embedding model used in chunking, and at query time — or distances are meaningless
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="credit_cards",
    )
    print(f"Stored {len(chunks)} chunks | {card_meta['card_name']}")
    return vs
