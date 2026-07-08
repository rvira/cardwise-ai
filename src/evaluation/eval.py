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

import json
import os
from pathlib import Path
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

from src.evaluation.numeric_validator import numeric_hit
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


def load_gold_set() -> list:
    """Load the 20-query gold set that lives next to this module."""
    return json.loads(Path(__file__).with_name("gold_set.json").read_text())


def gold_eval_set() -> list:
    """Adapt the gold set into the SAME {question, ground_truth} shape as
    WEEK1_EVAL_SET, so run_baseline_eval scores it with identical functions.
    Carries expected_numeric through for the numeric-exact guard."""
    return [
        {
            "question": g["query"],
            "ground_truth": g["reference"],
            "expected_numeric": g.get("expected_numeric"),
        }
        for g in load_gold_set()
    ]


def run_baseline_eval(chain, retriever, eval_set=None, limit: int = None) -> dict:
    """Score the RAG pipeline with ragas 0.4.x collections metrics and return the
    mean of each metric across the evaluated questions.

    ``eval_set`` defaults to WEEK1_EVAL_SET; pass ``gold_eval_set()`` to score the
    20-query gold set with the exact same functions. Only the first ``limit``
    questions are evaluated (defaults to all of them).

    Each metric is scored per-sample; if a metric fails for a sample it is
    skipped from that metric's mean rather than aborting the whole run. Items that
    carry ``expected_numeric`` also contribute to a deterministic numeric-exact
    accuracy via numeric_hit().
    """
    eval_set = eval_set if eval_set is not None else WEEK1_EVAL_SET
    if limit is None:
        limit = len(eval_set)
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
    items = eval_set[: max(1, limit)]
    n = len(items)
    print(f"Evaluating {n}/{len(eval_set)} questions (limit={limit}).", flush=True)
    print(
        f"[1/2] Generating answers + retrieving context for {n} question(s)…",
        flush=True,
    )
    samples = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        print(f"  • Q{i}/{n}: {q[:60]}…", flush=True)
        samples.append(
            {
                "question": q,
                "answer": chain.invoke(q),
                "contexts": [d.page_content for d in retriever.invoke(q)],
                "ground_truth": item["ground_truth"],
                "expected_numeric": item.get("expected_numeric"),
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

    # Deterministic numeric-exact guard: only items carrying expected_numeric
    # contribute (numeric_hit returns None otherwise, and is filtered out).
    num_flags = [
        hit
        for s in samples
        if (hit := numeric_hit(s["contexts"], s.get("expected_numeric"))) is not None
    ]
    scores["numeric_exact_acc"] = mean(num_flags) if num_flags else None

    print("=== EVAL SCORES ===")
    for name, vals in per_metric_values.items():
        value = scores[name]
        printed = f"{value:.3f}" if value is not None else "n/a (all samples failed)"
        print(f"  {name:<18}: {printed}  (n={len(vals)}/{len(samples)})")
    na = scores["numeric_exact_acc"]
    na_printed = f"{na:.3f}" if na is not None else "n/a (no numeric items)"
    print(
        f"  {'numeric_exact_acc':<18}: {na_printed}  (n={len(num_flags)}/{len(samples)})"
    )
    return scores


if __name__ == "__main__":
    # Runnable entrypoint. Pick the retrieval method, dataset, and (for hybrid)
    # whether to stratify across all cards:
    #   python -m src.evaluation.eval                        # MMR,    week1 (5 Q)
    #   python -m src.evaluation.eval hybrid                 # hybrid, week1 (5 Q)
    #   python -m src.evaluation.eval hybrid gold            # hybrid flat, gold (20 Q)
    #   python -m src.evaluation.eval hybrid gold strat      # hybrid stratified, gold (20 Q)
    #   python -m src.evaluation.eval dense gold             # dense-only baseline, gold (20 Q)
    #   python -m src.evaluation.eval hybrid gold k=10       # override chunk budget (default 6)
    import sys

    args = [a.lower() for a in sys.argv[1:]]
    if "hybrid" in args:
        mode = "hybrid"
    elif "dense" in args:
        mode = "dense"
    else:
        mode = "mmr"
    use_gold = "gold" in args
    use_strat = "strat" in args
    # Optional total-chunk budget: `k=10`. Defaults to 6.
    k = next((int(a.split("=", 1)[1]) for a in args if a.startswith("k=")), 6)
    if mode == "hybrid" and use_strat:
        variant = "hybrid-stratified"
    elif mode == "dense":
        variant = "dense-only"
    else:
        variant = mode
    print(
        f"Retrieval: {variant} | k={k} | dataset: {'gold (20)' if use_gold else 'week1 (5)'}",
        flush=True,
    )

    _vs = load_vectorstore()
    if mode == "hybrid":
        from src.retrieval.retriever import (
            as_runnable,
            as_stratified_runnable,
            build_hybrid_retriever,
        )

        _hybrid = build_hybrid_retriever(_vs)
        if use_strat:
            from src.ingestion.loader import CARDS

            _card_names = [c["card_name"] for c in CARDS]
            # Split the total budget across cards (>=1 chunk per card).
            _per_card = max(1, k // len(_card_names))
            _retriever = as_stratified_runnable(
                _hybrid, _card_names, k_per_card=_per_card
            )
        else:
            _retriever = as_runnable(_hybrid, k=k)
    elif mode == "dense":
        from src.retrieval.retriever import build_dense_retriever

        _retriever = build_dense_retriever(_vs, k=k)
    else:
        _retriever = build_default_retriever(_vs, k=k)
    # The chain and the eval share one retriever instance, so the contexts scored
    # here are exactly what the chain sees (no config to keep in sync).
    _chain = build_card_rag_chain(_vs, retriever=_retriever)
    run_baseline_eval(
        _chain, _retriever, eval_set=gold_eval_set() if use_gold else None
    )
