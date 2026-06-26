import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompts import CARD_ADVISOR_PROMPT
from src.retrieval.retriever import build_default_retriever

_PAGE_RE = re.compile(r"\[PAGE (\d+)\]")


def _page_label(doc) -> str:
    """Recover the page number from the `[PAGE n]` marker the loader embeds in
    the text. Returns ' | p.N' when found, else '' (no noisy 'p.?')."""
    m = _PAGE_RE.search(doc.page_content)
    return f" | p.{m.group(1)}" if m else ""


def build_card_rag_chain(vectorstore, retriever=None):
    # Default retriever = MMR over the store. Pass a custom Runnable (e.g. the
    # hybrid BM25+semantic retriever via retriever.as_runnable) to override.
    if retriever is None:
        retriever = build_default_retriever(vectorstore)
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=CARD_ADVISOR_PROMPT
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.0
    )  # temp=0: non-negotiable

    def format_with_citations(docs):
        return "\n\n".join(
            f"[Source {i+1} | {d.metadata.get('card_name','?')}{_page_label(d)}]\n{d.page_content}"
            for i, d in enumerate(docs)
        )

    return (
        {
            "context": retriever | format_with_citations,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
