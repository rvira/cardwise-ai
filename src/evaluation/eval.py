import sys as _sys
import types as _types

from dotenv import load_dotenv

load_dotenv()


if "langchain_community.chat_models.vertexai" not in _sys.modules:
    try:  # use the real class if the Vertex package happens to be installed
        from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    except Exception:  # otherwise a stub is enough — it is never instantiated here

        class _ChatVertexAI:  # placeholder so `from ... import ChatVertexAI` works
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "ChatVertexAI is not available; this project uses a Gemini judge."
                )

    _vertex_stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    _vertex_stub.ChatVertexAI = _ChatVertexAI
    _sys.modules["langchain_community.chat_models.vertexai"] = _vertex_stub
# ---------------------------------------------------------------------------

import os
from statistics import mean

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from src.rag.chain import build_card_rag_chain
from src.retrieval.retriever import build_default_retriever
from src.vectorstore.embedder import load_vectorstore

# ragas 0.4.x metrics live in `ragas.metrics.collections` and are scored
# per-sample via .score(); their underlying path is async-only, so the judge
# LLM must be backed by an *async* client. We reach Gemini through its
# OpenAI-compatible endpoint, which gives a real AsyncOpenAI client (ragas wraps
# it as an AsyncInstructor) — no langchain wrappers, no litellm needed. The same
# client serves both the judge and the embeddings used by answer_relevancy.
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

_api_key = os.environ.get("GOOGLE_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Add it to .env (loaded above) before running the eval."
    )

_gemini_client = AsyncOpenAI(api_key=_api_key, base_url=GEMINI_OPENAI_BASE)

JUDGE_MAX_TOKENS = 8192

# Gemini judge (temp 0 for deterministic scoring) and Gemini embeddings.
RAGAS_JUDGE = llm_factory(
    "gemini-2.5-flash",
    provider="openai",
    client=_gemini_client,
    temperature=0.0,
    max_tokens=JUDGE_MAX_TOKENS,
)
RAGAS_EMB = OpenAIEmbeddings(client=_gemini_client, model="gemini-embedding-001")

WEEK1_EVAL_SET = [
    # Axis ACE — Cashback proposition T&C
    {
        "question": "What cashback rate does the Axis ACE card give on Swiggy, Zomato and Ola, and any exclusions?",
        "ground_truth": "4% cashback on Swiggy, Zomato and Ola; not eligible on fuel, EMI, wallet loads, rent, gold/jewellery, insurance, education and similar excluded spends",
    },
    {
        "question": "What cashback does the Axis ACE card give on utility bill payments and recharges through Google Pay?",
        "ground_truth": "5% cashback on bill payments and recharges via Google Pay, capped at Rs. 500 per statement",
    },
    # HDFC Millennia — Cashback proposition T&C
    {
        "question": "What cashback rate does the HDFC Millennia card offer on its 10 partner online merchants?",
        "ground_truth": "5% CashBack on the 10 online merchants (Amazon, Flipkart, Swiggy, Zomato, Uber, Myntra, etc.), up to 1,000 CashPoints per cycle",
    },
    {
        "question": "What cashback does the HDFC Millennia card give on other (non-partner) spends?",
        "ground_truth": "1% CashBack on other spends (excluding fuel, EMI, wallet, rent and government transactions)",
    },
    # SBI SimplyCLICK — Reward Points T&C
    {
        "question": "What reward rate does the SBI SimplyCLICK card offer on online spends?",
        "ground_truth": "5X Reward Points on online spends (capped at 10,000 points per calendar month), and 10X Reward Points on exclusive partner brands",
    },
]

DEFAULT_EVAL_LIMIT = len(WEEK1_EVAL_SET)


def run_baseline_eval(chain, retriever, limit: int = DEFAULT_EVAL_LIMIT) -> dict:
    """Score the RAG pipeline on WEEK1_EVAL_SET with ragas 0.4.x collections
    metrics and return the mean of each metric across the evaluated questions.

    Only the first ``limit`` questions are evaluated (defaults to all of them);
    pass a smaller ``limit`` for a quick partial run.

    Each metric is scored per-sample; if a metric fails for a sample it is
    skipped from that metric's mean rather than aborting the whole run.
    """
    faithfulness = Faithfulness(llm=RAGAS_JUDGE)
    answer_relevancy = AnswerRelevancy(llm=RAGAS_JUDGE, embeddings=RAGAS_EMB)
    context_precision = ContextPrecision(llm=RAGAS_JUDGE)
    context_recall = ContextRecall(llm=RAGAS_JUDGE)

    # (metric name, callable producing a MetricResult for one sample)
    scorers = {
        "faithfulness": lambda s: faithfulness.score(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
        ),
        "answer_relevancy": lambda s: answer_relevancy.score(
            user_input=s["question"], response=s["answer"]
        ),
        "context_precision": lambda s: context_precision.score(
            user_input=s["question"],
            reference=s["ground_truth"],
            retrieved_contexts=s["contexts"],
        ),
        "context_recall": lambda s: context_recall.score(
            user_input=s["question"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        ),
    }

    # 1) Run the pipeline once per question to capture answer + retrieved context.
    #    Each step makes live API calls and can take a while, so print progress as
    #    we go — a silent terminal looks frozen and invites the user to kill it.
    eval_set = WEEK1_EVAL_SET[: max(1, limit)]
    n = len(eval_set)
    print(
        f"Evaluating {n}/{len(WEEK1_EVAL_SET)} questions (limit={limit}).", flush=True
    )
    print(
        f"[1/2] Generating answers + retrieving context for {n} question(s)…",
        flush=True,
    )
    samples = []
    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        print(f"  • Q{i}/{n}: {q[:60]}…", flush=True)
        samples.append(
            {
                "question": q,
                "answer": chain.invoke(q),
                "contexts": [d.page_content for d in retriever.invoke(q)],
                "ground_truth": item["ground_truth"],
            }
        )

    # 2) Score every (sample, metric); average the values that succeeded. Each
    #    metric is an LLM-judge call, so print each result as it lands.
    print(f"[2/2] Scoring {len(scorers)} metrics per question (LLM judge)…", flush=True)
    per_metric_values = {name: [] for name in scorers}
    for i, sample in enumerate(samples, 1):
        print(f"  • Q{i}/{n}:", flush=True)
        for name, scorer in scorers.items():
            print(f"      {name:<18}…", end="", flush=True)
            try:
                value = scorer(sample).value
                per_metric_values[name].append(value)
                print(f" {value:.3f}", flush=True)
            except Exception as ex:  # noqa: BLE001 — keep going on a single failure
                print(f" skip ({type(ex).__name__})", flush=True)

    scores = {
        name: (mean(vals) if vals else None) for name, vals in per_metric_values.items()
    }

    print("=== EVAL SCORES ===")
    for name, value in scores.items():
        n = len(per_metric_values[name])
        printed = f"{value:.3f}" if value is not None else "n/a (all samples failed)"
        print(f"  {name:<18}: {printed}  (n={n}/{len(samples)})")
    return scores


if __name__ == "__main__":
    # Runnable entrypoint. Pick the retrieval method to evaluate (default MMR):
    #   python -m src.evaluation.eval           # MMR
    #   python -m src.evaluation.eval hybrid    # BM25 + semantic hybrid
    import sys

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "mmr"
    print(f"Retrieval mode: {mode}", flush=True)

    _vs = load_vectorstore()
    if mode == "hybrid":
        from src.retrieval.retriever import as_runnable, build_hybrid_retriever

        _retriever = as_runnable(build_hybrid_retriever(_vs))
    else:
        _retriever = build_default_retriever(_vs)
    # The chain and the eval share one retriever instance, so the contexts scored
    # here are exactly what the chain sees (no config to keep in sync).
    _chain = build_card_rag_chain(_vs, retriever=_retriever)
    run_baseline_eval(_chain, _retriever)
