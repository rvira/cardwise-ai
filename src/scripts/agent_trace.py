# scripts/agent_trace.py
"""Print a CardWise agent tool-call trace for the README (AI-3.5 DoD).

Runs the agent on one factual and one computational question and prints each
message in the final state as a markdown list — the message list IS the trace.
Spends Gemini API quota, so run it yourself; it is not wired into CI.

Prerequisite: build the store first with `python -m src.scripts.ingest_all`.
Run from the project root:  python -m src.scripts.agent_trace
"""

from dotenv import load_dotenv

# Load GOOGLE_API_KEY from .env before any Gemini client is built — never hardcoded.
load_dotenv()

from src.agent.graph import ask, message_text

QUESTIONS = [
    # Factual -> should route to card_search.
    "What is the annual fee for the HDFC Millennia card?",
    # Computational -> should route to rewards_value.
    "I spend 8000/mo on dining and 5000/mo online — which card nets me more?",
]


def _describe(message) -> str:
    """One-line, README-friendly summary of a message: tool calls if present,
    else the (truncated, single-line) text content."""
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return "calls " + ", ".join(f"{c['name']}({c['args']})" for c in tool_calls)
    content = " ".join(message_text(message).split())
    return content[:200] + ("…" if len(content) > 200 else "")


def main() -> None:
    for question in QUESTIONS:
        print(f"\n### Q: {question}\n")
        result = ask(question)
        for message in result["messages"]:
            print(f"- **{type(message).__name__}** — {_describe(message)}")
        print(f"\n_(steps used: {result['steps']})_")


if __name__ == "__main__":
    main()
