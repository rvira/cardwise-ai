"""AI-3 — the agentic layer: a ReAct loop (LLM <-> tools) whose two safety rails
live in the graph *topology*, not the prompt:
  * a step cap (MAX_STEPS) on the conditional edge — the loop can't be talked out
    of stopping, because stopping is a property of the edge; and
  * a rag_fallback node — if the loop is cut off, the user still gets a plain-RAG
    answer instead of a truncated agent or a stack trace.
The LLM is built lazily so importing the graph needs no API key (keeps tests API-free).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from src.agent.tools.retriever_tool import card_search
from src.agent.tools.rewards_calc import rewards_value

MAX_STEPS = 6  # hard cap on agent turns — guarantees termination
TOOLS = [card_search, rewards_value]

SYSTEM = SystemMessage(
    content=(
        "You are CardWise, an advisor for three Indian credit cards (SBI SimplyCLICK, "
        "Axis ACE, HDFC Millennia).\n\n"
        "TOOLS — choose by intent:\n"
        "- rewards_value: use for ANY question about how much a card earns/saves, or "
        "which card is best/better for a spending pattern. It is the ONLY way to "
        "compute or compare reward amounts. NEVER do the arithmetic yourself in your "
        "reply — always call this tool and report its numbers.\n"
        "- card_search: use ONLY to look up a stated fact (a rate, cap, fee, "
        "exclusion, eligibility). Do NOT use it to compare cards by value.\n\n"
        "SPEND PROFILE for rewards_value: amounts are MONTHLY, in rupees. Map "
        "merchants to a category first: food delivery / restaurants (Swiggy, Zomato) "
        "-> dining; online shopping (Amazon, Flipkart) -> online; utilities, bill "
        "payments and recharges -> bills; groceries -> grocery; flights/hotels -> "
        "travel; anything else -> other.\n\n"
        "ANSWERING:\n"
        "- Open with a one-line statement of your assumptions (e.g. 'Assuming "
        "Rs.10,000/month on Swiggy, treated as dining').\n"
        "- Recommend the winner by NET ANNUAL value after the annual fee (the "
        "'net_after_fee' field), state each card's fee, and rank all three cards.\n"
        "- ALWAYS surface the caveats in the result's 'summary.notes'. In particular: "
        "if 'all_net_negative' is true, tell the user plainly that no card's rewards "
        "cover its fee for this spend — do NOT call a net-negative card 'best suited'; "
        "if 'no_rewards_difference' is true, say the cards don't differ on rewards "
        "here (it's only a fee comparison); present tied cards as a tie.\n"
        "- Be concise. Do NOT paste long exclusion lists unless asked; cite the card "
        "briefly when you state a retrieved fact.\n"
        "- Answer only from tool results; if the documents don't cover something, say "
        "so plainly — never invent a number."
    )
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


_llm = None


def _get_llm():
    """Lazily build the tool-bound chat model (temp=0, matching the RAG chain)."""
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.0
        ).bind_tools(TOOLS)
    return _llm


def agent(state: AgentState) -> dict:
    """Reasoning node: model either answers or requests a tool call; bump the step."""
    reply = _get_llm().invoke([SYSTEM] + state["messages"])
    return {"messages": [reply], "steps": state.get("steps", 0) + 1}


def route(state: AgentState) -> str:
    """Conditional edge = the safety policy. Step cap wins first (model can't
    override it), then tools if requested, else end."""
    if state.get("steps", 0) >= MAX_STEPS:
        return "fallback"
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def message_text(message) -> str:
    """Plain display text for a message whose ``.content`` may be a string OR a list
    of content blocks. Gemini 2.5 returns text blocks that also carry an opaque
    thought ``signature`` in ``extras`` — we keep only the human-readable text and
    drop the signature (it is reasoning metadata, not for display)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "".join(parts).strip()
    return str(content)


def _first_user_question(messages) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            return m.content
    return getattr(messages[0], "content", "")


def rag_fallback(state: AgentState) -> dict:
    """Plain RAG — retrieve + answer once, no loop. Reuses the existing hybrid
    retriever and citation-enforcing answer chain. Never leaks a trace."""
    from src.agent.tools.retriever_tool import _RETRIEVAL_K, _get_retriever
    from src.rag.chain import build_answer_chain, format_with_citations

    question = _first_user_question(state["messages"])
    try:
        docs = _get_retriever().retrieve_rrf(question, k=_RETRIEVAL_K)
        answer = build_answer_chain().invoke(
            {"context": format_with_citations(docs), "question": question}
        )
    except Exception as err:  # generic client-safe message, log server-side
        from src.rag.errors import friendly_error

        print(f"[agent_fallback_error] {type(err).__name__}: {err}")
        answer = friendly_error(err)
    return {"messages": [AIMessage(content=answer)]}


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_node("fallback", rag_fallback)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "fallback": "fallback", "end": END}
    )
    graph.add_edge("tools", "agent")  # the ReAct loop
    graph.add_edge("fallback", END)
    return graph.compile()


app = build_agent_graph()  # compiled at import; cheap + API-free (LLM is lazy)


def ask(question: str) -> dict:
    """Run the agent on one question; result['messages'] IS the tool-call trace."""
    return app.invoke({"messages": [HumanMessage(content=question)], "steps": 0})
