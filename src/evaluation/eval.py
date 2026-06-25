import sys as _sys
import types as _types

from dotenv import load_dotenv

# Load GOOGLE_API_KEY (and friends) from .env before any Gemini client is built.
# The key is read from the environment — never hardcoded.
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
import re
import time
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

# gemini-2.5-flash is a "thinking" model: via the OpenAI-compatible endpoint its
# reasoning tokens count against max_tokens, so a small default truncates the
# structured JSON (faithfulness/context_precision emit lists of statements and
# verdicts) and raises IncompleteOutputException. Give the judge generous output
# room so multi-item metric outputs complete.
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

# Free-tier Gemini allows only ~20 generate_content requests/day for
# gemini-2.5-flash. Each evaluated question costs roughly:
#   1 (RAG answer) + 2 (faithfulness) + 1 (answer_relevancy)
#   + up to k (context_precision, one verdict per retrieved context)
#   + 1 (context_recall)  ≈ up to ~11 calls.
# So one question fits a single free key with margin; two can exceed the daily
# cap. We therefore evaluate only the first DEFAULT_EVAL_LIMIT questions by
# default. Raise it (or pass limit=len(WEEK1_EVAL_SET)) once billing is enabled.
DEFAULT_EVAL_LIMIT = 5


def _score_with_retry(scorer, sample, *, retries: int = 4, default_wait: float = 60.0):
    """Call a metric scorer, retrying on rate-limit (429) errors.

    Free-tier Gemini caps generate_content at ~5 requests/minute, and metrics
    that make several internal calls can trip it. On a 429 we wait the delay the
    API suggests (or ``default_wait``) and retry, so a run can ride out the
    per-minute window instead of failing the metric outright. Non-rate-limit
    errors are re-raised immediately.
    """
    for attempt in range(1, retries + 1):
        try:
            return scorer(sample)
        except Exception as ex:  # noqa: BLE001 — inspect message to decide retry
            msg = str(ex)
            if (
                "429" not in msg and "RESOURCE_EXHAUSTED" not in msg
            ) or attempt == retries:
                raise
            m = re.search(r"retry in ([\d.]+)s", msg) or re.search(
                r"retryDelay['\":\s]+([\d.]+)s", msg
            )
            wait = (float(m.group(1)) + 1) if m else default_wait
            print(
                f"    rate-limited; waiting {wait:.0f}s then retrying ({attempt}/{retries - 1})…"
            )
            time.sleep(wait)


def run_baseline_eval(chain, retriever, limit: int = DEFAULT_EVAL_LIMIT) -> dict:
    """Score the RAG pipeline on WEEK1_EVAL_SET with ragas 0.4.x collections
    metrics and return the mean of each metric across the evaluated questions.

    Only the first ``limit`` questions are evaluated so a full run stays within
    the free-tier daily request cap (see DEFAULT_EVAL_LIMIT). Pass a larger
    ``limit`` (e.g. ``len(WEEK1_EVAL_SET)``) when you have a billed key.

    Each metric is scored per-sample; a failed sample (e.g. a transient API
    error or rate limit) is recorded as None and skipped from that metric's mean
    rather than aborting the whole run.
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
    eval_set = WEEK1_EVAL_SET[: max(1, limit)]
    print(
        f"Evaluating {len(eval_set)}/{len(WEEK1_EVAL_SET)} questions "
        f"(limit={limit}) to stay within the free-tier daily quota."
    )
    samples = []
    for item in eval_set:
        q = item["question"]
        samples.append(
            {
                "question": q,
                "answer": chain.invoke(q),
                "contexts": [d.page_content for d in retriever.invoke(q)],
                "ground_truth": item["ground_truth"],
            }
        )

    # 2) Score every (sample, metric); average the values that succeeded.
    per_metric_values = {name: [] for name in scorers}
    for i, sample in enumerate(samples, 1):
        for name, scorer in scorers.items():
            try:
                per_metric_values[name].append(_score_with_retry(scorer, sample).value)
            except Exception as ex:  # noqa: BLE001 — keep going on a single failure
                print(f"  [skip] {name} on Q{i}: {type(ex).__name__}: {ex}")

    scores = {
        name: (mean(vals) if vals else None) for name, vals in per_metric_values.items()
    }

    print("=== WEEK 1 BASELINE SCORES ===")
    for name, value in scores.items():
        n = len(per_metric_values[name])
        printed = f"{value:.3f}" if value is not None else "n/a (all samples failed)"
        print(f"  {name:<18}: {printed}  (n={n}/{len(samples)})")
    return scores


if __name__ == "__main__":
    # Runnable entrypoint: build the same chain + retriever the app uses and
    # score the baseline. Run from the project root:  python -m src.evaluation.eval
    from src.rag.chain import build_card_rag_chain
    from src.vectorstore.embedder import load_vectorstore

    _vs = load_vectorstore()
    _chain = build_card_rag_chain(_vs)
    # Mirror the retriever config used inside build_card_rag_chain so the
    # evaluated contexts match what the chain actually sees.
    _retriever = _vs.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )
    run_baseline_eval(_chain, _retriever)
