"""Tests for the agent graph: routing, the step cap, and error recovery (AI-3.4,
AI-3.5). No API — the chat model is replaced with a scripted fake, and the
retriever/answer-chain are monkeypatched, so the *topology* (loop, cap, fallback)
is what's under test.
"""

import pytest

pytest.importorskip("langgraph")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.agent import graph as G  # noqa: E402
from src.agent.tools import retriever_tool  # noqa: E402


# --- scripted fake chat model ------------------------------------------------


class _FakeLLM:
    """Returns queued AIMessages in order; repeats the last one thereafter."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, _messages):
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return reply


def _tool_call_msg(call_id, name="card_search", args=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {"query": "fee"}, "id": call_id}],
    )


class _FakeRetriever:
    def __init__(self, docs=None, raises=False):
        self._docs = docs or [
            Document(
                page_content="Annual fee is Rs. 499. [PAGE 1]",
                metadata={"card_name": "Axis ACE"},
            )
        ]
        self._raises = raises

    def retrieve_rrf(self, query, k=6):
        if self._raises:
            raise RuntimeError("boom: retriever exploded")
        return self._docs


# --- route(): the safety policy lives here -----------------------------------


def test_route_step_cap_forces_fallback():
    state = {"messages": [AIMessage(content="anything")], "steps": G.MAX_STEPS}
    assert G.route(state) == "fallback"


def test_route_to_tools_when_model_requests_a_tool():
    state = {"messages": [_tool_call_msg("c1")], "steps": 1}
    assert G.route(state) == "tools"


def test_route_ends_on_plain_answer():
    state = {"messages": [AIMessage(content="here is the answer")], "steps": 1}
    assert G.route(state) == "end"


def test_step_cap_wins_even_with_a_pending_tool_call():
    # A tool call at the cap must still divert to fallback, not loop again.
    state = {"messages": [_tool_call_msg("c1")], "steps": G.MAX_STEPS}
    assert G.route(state) == "fallback"


# --- full loop: factual query -> tool -> answer ------------------------------


def test_factual_query_calls_tool_then_answers(monkeypatch):
    # One fake instance so the scripted sequence advances (a fresh instance per
    # call would replay reply #1 forever and loop to the cap).
    fake = _FakeLLM(
        [
            _tool_call_msg("c1"),
            AIMessage(content="The annual fee for Axis ACE is Rs. 499."),
        ]
    )
    monkeypatch.setattr(G, "_get_llm", lambda: fake)
    monkeypatch.setattr(retriever_tool, "_get_retriever", lambda: _FakeRetriever())

    result = G.app.invoke(
        {
            "messages": [HumanMessage(content="What is the Axis ACE annual fee?")],
            "steps": 0,
        }
    )
    kinds = [type(m).__name__ for m in result["messages"]]
    assert "ToolMessage" in kinds  # the tool actually executed
    assert "Rs. 499" in result["messages"][-1].content


# --- error recovery: tool raises -> still get an answer (AI-3.5) -------------


def test_tool_failure_recovers_via_tool_message(monkeypatch):
    # card_search raises; ToolNode(handle_tool_errors=True) turns it into a
    # ToolMessage, and the model then produces a final answer instead of crashing.
    fake = _FakeLLM(
        [
            _tool_call_msg("c1"),
            AIMessage(
                content="Sorry, I couldn't look that up, but the fee is on page 1."
            ),
        ]
    )
    monkeypatch.setattr(G, "_get_llm", lambda: fake)
    monkeypatch.setattr(
        retriever_tool, "_get_retriever", lambda: _FakeRetriever(raises=True)
    )

    result = G.app.invoke({"messages": [HumanMessage(content="fee?")], "steps": 0})
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs  # the failure surfaced as a ToolMessage
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content  # non-empty answer returned


# --- step cap integration: runaway loop -> fallback answer -------------------


def test_runaway_loop_hits_fallback_and_still_answers(monkeypatch):
    # A model that ALWAYS asks for a tool would loop forever without the cap.
    class _AlwaysToolLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return _tool_call_msg(f"c{self.calls}")

    monkeypatch.setattr(G, "_get_llm", _AlwaysToolLLM)
    monkeypatch.setattr(retriever_tool, "_get_retriever", lambda: _FakeRetriever())

    # Fallback uses the plain-RAG answer chain — stub it so no API is hit.
    import src.rag.chain as chain_mod

    class _FakeChain:
        def invoke(self, _inputs):
            return "Fallback answer: the annual fee is Rs. 499."

    monkeypatch.setattr(chain_mod, "build_answer_chain", lambda: _FakeChain())

    result = G.app.invoke({"messages": [HumanMessage(content="fee?")], "steps": 0})
    # The graph terminated (cap enforced) and produced the fallback answer.
    assert result["steps"] >= G.MAX_STEPS
    assert result["messages"][-1].content.startswith("Fallback answer")
