from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompts import CARD_ADVISOR_PROMPT


def build_card_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=CARD_ADVISOR_PROMPT
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.0
    )  # temp=0: non-negotiable

    def format_with_citations(docs):
        return "\n\n".join(
            f"[Source {i+1} | {d.metadata.get('card_name','?')} | p.{d.metadata.get('page_num','?')}]\n{d.page_content}"
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
