"""CardWise — interactive RAG demo frontend (Streamlit).

Run from the project root:
    streamlit run app.py

Prerequisite: build the vector store first with
    python -m src.scripts.ingest_all
"""

from dotenv import load_dotenv

# Load GOOGLE_API_KEY from .env before any Gemini client is built — never hardcoded.
load_dotenv()

import json
import os
from string import Template

import streamlit as st

# On Streamlit Community Cloud there is no .env file — the API key is provided
# through the Cloud "Secrets" manager and exposed via st.secrets. Bridge it into
# the env var the Gemini client reads. The value comes from the secret manager;
# it is never hardcoded here. .env still wins for local runs.
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass  # Surfaced as a friendly error after set_page_config below.

from src.rag.chain import build_answer_chain, format_with_citations
from src.rag.errors import friendly_error
from src.retrieval.retriever import (
    build_default_retriever,
    build_hybrid_retriever,
    load_all_chunks,
    stratified_retrieve,
)
from src.vectorstore.embedder import load_vectorstore

# Avatars shown beside each chat message.
AVATARS = {"user": "🧑", "assistant": "💳"}

# Keep questions short enough that an unbounded paste can't be sent to the model.
MAX_QUESTION_LEN = 500

EXAMPLE_QUESTIONS = [
    "What reward rate does SBI SimplyCLICK offer on online spends?",
    "What is the cashback rate for Axis ACE, and any exclusions?",
    "What is the annual fee for HDFC Millennia?",
]

# Retrieval methods selectable from the sidebar (label → internal mode).
RETRIEVAL_MODES = {
    "MMR (semantic + diversity)": "mmr",
    "Hybrid (BM25 + semantic)": "hybrid",
    "Agent (tools: search + rewards calc)": "agent",
}


@st.cache_resource(show_spinner="Loading the card knowledge base…")
def get_vectorstore():
    return load_vectorstore()


@st.cache_resource(show_spinner="Building the hybrid index…")
def get_hybrid_retriever():
    """Hybrid BM25+semantic retriever — expensive (loads every chunk and builds
    the BM25 index), so it's cached. The MMR retriever just wraps the store and
    is cheap to rebuild per request, which it must be: it varies with the
    active card filter."""
    return build_hybrid_retriever(get_vectorstore())


@st.cache_resource
def get_answer_chain():
    return build_answer_chain()


@st.cache_data(show_spinner=False)
def get_card_names():
    """Distinct card names present in the store. Drives the card filter so it
    always offers exactly the cards that were actually ingested."""
    names = {
        c.metadata.get("card_name")
        for c in load_all_chunks(get_vectorstore())
        if c.metadata.get("card_name")
    }
    return sorted(names)


def retrieve_docs(question: str, mode: str, cards):
    """Retrieve source docs for a question, optionally restricted to `cards`
    (a list of card names; empty/None = search all). Returned docs feed both the
    answer context and the citation panel."""
    if mode == "hybrid":
        hybrid = get_hybrid_retriever()
        if cards:
            return stratified_retrieve(hybrid, question, cards)
        # RRF at k=10 — the config validated in the gold-set eval (recall 0.883).
        return hybrid.retrieve_rrf(question, k=10)
    # MMR over the store, with an optional Chroma metadata filter on card_name.
    overrides = {"filter": {"card_name": {"$in": cards}}} if cards else {}
    return build_default_retriever(get_vectorstore(), **overrides).invoke(question)


def render_streaming_answer(question: str, mode: str, cards) -> str:
    """ChatGPT/Gemini-style: show a 'Thinking…' loader during retrieval + first
    token, then stream the rest into a placeholder with a typing cursor. The
    answer itself carries inline [Source …] citations from the prompt. Any
    backend error (retrieval or generation) is shown as a friendly message
    rather than a raw traceback."""
    placeholder = st.empty()
    try:
        with st.spinner("Searching the card documents…"):
            docs = retrieve_docs(question, mode, cards)
        chain = get_answer_chain()
        context = format_with_citations(docs)
        with st.spinner("Thinking…"):
            stream = chain.stream({"context": context, "question": question})
            full = next(stream, "")
        placeholder.markdown(full + " ▌")
        for chunk in stream:
            full += chunk
            placeholder.markdown(full + " ▌")
        placeholder.markdown(full)
        return full
    except Exception as ex:
        # Log full details server-side; show the client only a generic message.
        print(f"[run_error] {type(ex).__name__}: {ex}")
        message = friendly_error(ex)
        placeholder.markdown(message)
        return message


def build_trace(messages, text_of):
    """Flatten the agent's message list into a small, JSON-serialisable trace so it
    can be rendered now and re-rendered on chat replay (stored in session_state).
    `text_of` extracts plain text from a message whose content may be a string or a
    list of content blocks (Gemini 2.5)."""
    steps = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None) or []
        steps.append(
            {
                "kind": type(m).__name__,
                "content": text_of(m),
                "tool_calls": [
                    {"name": c["name"], "args": c["args"]} for c in tool_calls
                ],
            }
        )
    return steps


def render_trace(trace):
    """Show the agent's tool-call trace in a collapsed expander (a ChatGPT/Claude-
    style 'thinking' disclosure). Tool args/results are rendered with st.code and
    the answer with plain st.markdown (unsafe_allow_html stays off) — so untrusted
    tool/model output is shown as text, never executed as HTML."""
    if not trace:
        return
    n_calls = sum(len(step["tool_calls"]) for step in trace)
    label = f"🛠️ Tool-call trace — {n_calls} tool call(s) over {len(trace)} steps"
    with st.expander(label, expanded=False):
        for step in trace:
            kind = step["kind"]
            if kind == "HumanMessage":
                st.markdown("**🧑 Question**")
                st.markdown(step["content"])
            elif step["tool_calls"]:
                for call in step["tool_calls"]:
                    st.markdown(f"**🤖 decides to call** `{call['name']}`")
                    st.code(json.dumps(call["args"], indent=2), language="json")
            elif kind == "ToolMessage":
                st.markdown("**🔧 tool result**")
                st.code((step["content"] or "(empty)")[:2000], language="text")
            elif kind == "AIMessage":
                st.markdown("**🤖 final answer**")
                st.markdown(step["content"])


def render_agent_answer(question: str):
    """Agent mode: run the LangGraph agent (an LLM that picks between doc search and
    a deterministic rewards calculator), show its tool-call trace in a collapsible
    panel, then the answer. Returns (answer_text, trace) so both persist in history.
    The agent searches all cards, so the sidebar card filter does not apply here."""
    try:
        from src.agent.graph import ask, message_text
    except ImportError:
        msg = (
            "⚠️ The agent isn't available — install its dependency with "
            "`pip install langgraph` and restart the app."
        )
        st.markdown(msg)
        return msg, None
    try:
        with st.spinner("Agent is working (choosing tools, searching, calculating)…"):
            result = ask(question)
        messages = result["messages"]
        answer = message_text(messages[-1])
        trace = build_trace(messages, message_text)
    except Exception as ex:
        # Log full details server-side; show the client only a generic message.
        print(f"[agent_error] {type(ex).__name__}: {ex}")
        answer, trace = friendly_error(ex), None
    render_trace(trace)
    st.markdown(answer)
    return answer, trace


st.set_page_config(
    page_title="CardWise — Credit Card Advisor",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Fail clearly (not with a stack trace) if the key was never provided — or is
# still the example placeholder from secrets.toml.example.
_key = os.environ.get("GOOGLE_API_KEY", "").strip()
if not _key or _key == "your-gemini-api-key-here":
    st.error(
        "🔑 GOOGLE_API_KEY is not configured. Add a real key under the app's "
        "**Settings → Secrets** (Streamlit Community Cloud) or in a local `.env`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Styling. Two fixed palettes (light / dark) selected by a boolean toggle. The
# template below interpolates ONLY these author-defined constants — never user
# input — so unsafe_allow_html carries no injection risk.
# ---------------------------------------------------------------------------
_STYLE = Template(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 820px; }
      .stApp { background: $bg; color: $text; }

      /* App chrome that has its own background and won't inherit .stApp:
         the top toolbar and the full-width fixed bottom chat-input bar
         (its outer wrappers stay white otherwise — the corners by the input). */
      [data-testid="stHeader"],
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="stBottom"],
      [data-testid="stBottom"] > div,
      [data-testid="stBottomBlockContainer"] { background: $bg !important; }

      [data-testid="stHeader"] svg { color: $text; fill: $text; }

      /* Hero header with a soft gradient (shared by both themes). */
      .cw-hero {
        background: linear-gradient(135deg, #6C5CE7 0%, #8E7BFF 50%, #A29BFE 100%);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        color: #FFFFFF;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.30);
        margin-bottom: 1.4rem;
      }
      .cw-hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; }
      .cw-hero p  { margin: 0.4rem 0 0; opacity: 0.92; font-size: 0.98rem; line-height: 1.4; }

      .cw-eyebrow {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #8E7BFF; margin: 0.2rem 0 0.6rem;
      }

      /* Example-question buttons / sidebar buttons styled as soft cards. */
      div[data-testid="stButton"] > button {
        border-radius: 14px;
        border: 1px solid $border;
        background: $chip;
        color: $text;
        text-align: left;
        padding: 0.8rem 0.9rem;
        font-size: 0.88rem;
        line-height: 1.35;
        height: 100%;
        transition: all 0.15s ease;
      }
      div[data-testid="stButton"] > button:hover {
        border-color: #6C5CE7;
        box-shadow: 0 6px 18px rgba(108, 92, 231, 0.22);
        transform: translateY(-2px);
        color: #8E7BFF;
      }

      div[data-testid="stChatMessage"] {
        background: $chip;
        border-radius: 14px;
        padding: 0.6rem 0.8rem;
      }
      /* Message text: Streamlit colors markdown with the (light) theme body
         color, which is unreadable on the dark bubble — follow the palette. */
      div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
      div[data-testid="stChatMessage"] :is(p, li, span, strong, em) {
        color: $text !important;
      }

      /* Sidebar surface + force its text to follow the palette (Streamlit's
         theme sets its own text colors that otherwise stay dark-on-dark). */
      section[data-testid="stSidebar"] { background: $panel; }

      /* Sidebar collapse (open) + expand (collapsed) controls. Streamlit keeps
         the collapse arrow visibility:hidden until you hover the sidebar and
         fades its icon — which disappears entirely in dark mode. Force it
         always visible, brighten the icon, and give it a ChatGPT-style hover. */
      [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
      }
      [data-testid="stSidebarCollapseButton"] *,
      [data-testid="stExpandSidebarButton"] * {
        color: $text !important;
        fill: $text !important;
      }
      [data-testid="stSidebarCollapseButton"] button:hover,
      [data-testid="stExpandSidebarButton"]:hover,
      [data-testid="stExpandSidebarButton"] button:hover {
        background: $chip !important;
        border-radius: 10px;
      }
      section[data-testid="stSidebar"] :is(h1, h2, h3, h4, p, label, span, li, small),
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: $text !important; }

      /* Multiselect control (the "Choose options" box). */
      section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: $chip !important;
        border-color: $border !important;
      }
      section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: $text !important; }

      /* ChatGPT-style search bar: pill-rounded, soft border, subtle lift,
         and a focus glow in the brand purple. */
      [data-testid="stChatInput"] {
        background: $chip;
        border: 1px solid $border;
        border-radius: 26px;
        padding: 0.15rem 0.4rem;
        box-shadow: 0 2px 14px rgba(0, 0, 0, 0.18);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
      }
      [data-testid="stChatInput"]:focus-within {
        border-color: #6C5CE7;
        box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.28);
      }
      /* The BaseWeb textarea wrappers carry their own (light) background — make
         them transparent so the rounded $chip frame shows through, with no
         white box nested inside. */
      [data-testid="stChatInput"] > div,
      [data-testid="stChatInput"] div[data-baseweb="textarea"],
      [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
      }
      [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: $text;
      }
      [data-testid="stChatInput"] textarea::placeholder { color: $muted; }

      /* Solid circular send button. */
      [data-testid="stChatInputSubmitButton"] {
        background: #6C5CE7;
        color: #FFFFFF;
        border-radius: 50%;
        transition: background 0.15s ease, transform 0.15s ease;
      }
      [data-testid="stChatInputSubmitButton"]:hover {
        background: #8E7BFF;
        transform: scale(1.06);
      }
      [data-testid="stChatInputSubmitButton"]:disabled {
        background: $border;
        color: $muted;
      }

      /* Chat avatars. We pass emoji avatars, so Streamlit renders them under
         the *Custom* test-id (default white background → fix that for dark). */
      [data-testid="stChatMessageAvatarCustom"],
      [data-testid="stChatMessageAvatarUser"],
      [data-testid="stChatMessageAvatarAssistant"] {
        background: $chip;
        border: 1px solid $border;
      }

      /* Tool-call trace expander: drop the default border (and its box-shadow
         edge) so the collapsible blends into the chat bubble. */
      [data-testid="stExpander"],
      [data-testid="stExpander"] details {
        border: none !important;
        box-shadow: none !important;
      }
    </style>
    """
)

# Palette is chosen by a boolean — these are the only values interpolated.
# Dark = matte charcoal greys (ChatGPT-ish), deliberately not pure black.
DARK = {
    "bg": "#212121",
    "panel": "#171717",
    "text": "#ECECEC",
    "border": "#3A3A3A",
    "chip": "#2B2B2B",
    "muted": "#9B9B9B",
}
LIGHT = {
    "bg": "#FFFFFF",
    "panel": "#F4F2FF",
    "text": "#1E1B2E",
    "border": "#E4E0FB",
    "chip": "#F8F7FF",
    "muted": "#6B6680",
}

# Sidebar controls. Read the dark-mode toggle first so styling reflects it.
with st.sidebar:
    dark_mode = st.toggle("🌙 Dark mode", value=False)

    st.markdown("#### ⚙️ Retrieval")
    mode_label = st.radio(
        "Method",
        list(RETRIEVAL_MODES),
        index=0,
        help=(
            "MMR: semantic search with diversity (default). "
            "Hybrid: BM25 keyword + semantic — better for exact values like fees/rates. "
            "Agent: an LLM that chooses tools (doc search + a rewards calculator) and "
            "shows its tool-call trace; searches all cards (ignores the card filter)."
        ),
    )
    retrieval_mode = RETRIEVAL_MODES[mode_label]

    st.markdown("### 💳 Filter by card")
    selected_cards = st.multiselect(
        "Limit answers to these cards",
        options=get_card_names(),
        default=[],
        help="Leave empty to search every card in the knowledge base.",
    )

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Answers are grounded in official card documents, with citations.")

st.markdown(_STYLE.substitute(DARK if dark_mode else LIGHT), unsafe_allow_html=True)

st.markdown(
    """
    <div class="cw-hero">
      <h1>💳 CardWise</h1>
      <p>Your credit-card advisor. Ask about rewards, fees, and benefits —
      every answer is grounded in the official card documents, with citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay prior conversation.
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        if m.get("trace"):
            render_trace(m["trace"])
        st.markdown(m["content"])

# Example-question cards (only before the first message, to keep the UI clean).
clicked = None
if not st.session_state.messages:
    st.markdown('<p class="cw-eyebrow">✨ Try an example</p>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            clicked = q

prompt = st.chat_input("Ask about a credit card…") or clicked

if prompt:
    prompt = prompt.strip()[:MAX_QUESTION_LEN]  # cap length; reject oversized input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        if retrieval_mode == "agent":
            reply, trace = render_agent_answer(prompt)
        else:
            reply = render_streaming_answer(prompt, retrieval_mode, selected_cards)
            trace = None
    message = {"role": "assistant", "content": reply}
    if trace:
        message["trace"] = trace
    st.session_state.messages.append(message)
