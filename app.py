"""CardWise — interactive RAG demo frontend (Streamlit).

Run from the project root:
    streamlit run app.py

Prerequisite: build the vector store first with
    python -m src.scripts.ingest_all
"""
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from .env before any Gemini client is built — never hardcoded.
load_dotenv()

import streamlit as st

from src.rag.chain import build_card_rag_chain
from src.vectorstore.embedder import load_vectorstore

# Keep questions short enough that an unbounded paste can't be sent to the model.
MAX_QUESTION_LEN = 500

EXAMPLE_QUESTIONS = [
    "What reward rate does SBI SimplyCLICK offer on online spends?",
    "What is the cashback rate for Axis ACE, and any exclusions?",
    "What is the annual fee for HDFC Millennia?",
]


@st.cache_resource(show_spinner="Loading the card knowledge base…")
def get_chain():
    """Build the RAG chain once and reuse it across reruns/sessions."""
    return build_card_rag_chain(load_vectorstore())


def answer(question: str):
    """Stream the model's answer for a question, with friendly error handling."""
    chain = get_chain()
    try:
        # chain.stream yields string chunks (StrOutputParser at the tail).
        yield from chain.stream(question)
    except Exception as ex:  # noqa: BLE001 — surface a safe message, log the rest
        msg = str(ex)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            yield "⚠️ The service is busy (rate limit reached). Please try again in a minute."
        else:
            # Generic message to the UI; full error stays in the server logs.
            print(f"[run_error] {type(ex).__name__}: {ex}")
            yield "⚠️ Something went wrong answering that. Please try again."


def render_streaming_answer(question: str) -> str:
    """ChatGPT/Gemini-style: show a 'Thinking…' loader until the first token
    arrives (retrieval + LLM latency happen here), then stream the rest into a
    single placeholder with a typing cursor."""
    gen = answer(question)
    with st.spinner("Thinking…"):
        first = next(gen, "")  # block under the loader until the first chunk
    placeholder = st.empty()
    full = first
    placeholder.markdown(full + " ▌")
    for chunk in gen:
        full += chunk
        placeholder.markdown(full + " ▌")
    placeholder.markdown(full)  # drop the cursor when done
    return full


st.set_page_config(page_title="CardWise — Credit Card Advisor", page_icon="💳")
st.title("💳 CardWise")
st.caption("Ask about the cards in the knowledge base. Answers are grounded in the official card documents, with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay prior conversation.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Example-question buttons (only before the first message, to keep the UI clean).
clicked = None
if not st.session_state.messages:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            clicked = q

prompt = st.chat_input("Ask about a credit card…") or clicked

if prompt:
    prompt = prompt.strip()[:MAX_QUESTION_LEN]  # cap length; reject oversized input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        reply = render_streaming_answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})
