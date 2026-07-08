# 💳 CardWise

> A retrieval-augmented generation (RAG) assistant that answers questions about credit cards
> **grounded in the official card documents** — with mandatory source citations and a
> "don't answer if the docs don't say so" guardrail against hallucination.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
  <img alt="Gemini" src="https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?logo=google&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-1.x-1C3C3C">
  <img alt="Chroma" src="https://img.shields.io/badge/Chroma-vector%20store-FF6F61">
  <img alt="RAGAS" src="https://img.shields.io/badge/RAGAS-evaluation-6E56CF">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-frontend-FF4B4B?logo=streamlit&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://cardwise-ai.streamlit.app/"><img alt="Live demo" src="https://img.shields.io/badge/Live%20demo-cardwise--ai.streamlit.app-FF4B4B?logo=streamlit&logoColor=white"></a>
</p>

Ask things like *"What is the cashback rate for Axis ACE, and any exclusions?"* and get an
answer drawn only from the ingested PDFs, with the card name cited.

---

## Contents

- [Features](#features)
- [Demo](#demo)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Cards currently ingested](#cards-currently-ingested)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Guardrails](#guardrails)
- [Design decisions](#design-decisions)
- [Limitations & roadmap](#limitations--roadmap)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Hybrid retrieval** — dense (semantic) + BM25 (keyword) fused with **Reciprocal Rank
  Fusion**, so exact-value queries ("what's the fee cap?") and conceptual ones both work.
- **Grounded, cited answers** — every response is drawn only from the ingested PDFs and
  cites the card it came from; the model declines when the documents are silent.
- **Hallucination guardrails in depth** — an input quality gate, a citation-enforcing
  prompt, and a numeric-claim validator (see [Guardrails](#guardrails)).
- **Measured, not assumed** — a 20-query gold set scored with **RAGAS** plus a
  deterministic numeric-exact guard; retrieval config chosen from the numbers
  (see [Evaluation](#evaluation)).
- **Interactive demo** — a Streamlit chat UI with live retrieval-mode switching and a
  per-card filter.
- **CI-friendly tests** — fast, API-free unit tests that run on every push.

**Results at a glance** (20-query gold set, hybrid @ k=10): context recall **0.883**,
numeric-exact retrieval **1.000**, 44 passing unit tests.

---

## Demo

**▶ Try it live: [cardwise-ai.streamlit.app](https://cardwise-ai.streamlit.app/)**

<p align="center">
  <img src="docs/demo.gif" alt="CardWise answering a credit-card question with cited sources" width="100%">
</p>

## How it works

<p align="center">
  <img src="docs/architecture.svg" alt="CardWise RAG architecture" width="100%">
</p>

1. **Ingest** — extract text from card PDFs, split into overlapping chunks, embed with
   `gemini-embedding-001`, and store in a persistent **Chroma** collection with per-card metadata.
2. **Retrieve** — fetch relevant chunks via **MMR** (semantic + diversity) or **Hybrid**
   (BM25 + semantic, fused with Reciprocal Rank Fusion); switch in the sidebar. Hybrid is the
   benchmarked, recommended mode for exact-value and comparison queries.
3. **Generate** — pass the chunks to `gemini-2.5-flash` through a citation-enforcing prompt
   and stream the answer.
4. **Evaluate** — score with **RAGAS** + a deterministic numeric-exact guard. See
   [Evaluation](#evaluation).

## Tech stack

- **LangChain** (core, text-splitters, community, experimental) — pipeline orchestration
- **Google Gemini** — `gemini-embedding-001` (embeddings) + `gemini-2.5-flash` (generation & eval judge)
- **Chroma** — local persistent vector store
- **PyMuPDF** — PDF text extraction
- **RAGAS** — RAG quality evaluation
- **rank-bm25** — keyword scoring for the hybrid retriever's sparse leg
- **Streamlit** — interactive demo frontend

## Project structure

```
cardwise-ai/
├── app.py                       # Streamlit chat UI (demo frontend)
├── data/raw/                    # source card PDFs
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # PDF → text + quality_check
│   │   └── chunker.py           # chunk_document() + benchmark_chunking()
│   ├── vectorstore/
│   │   └── embedder.py          # build_vectorstore() / load_vectorstore()
│   ├── rag/
│   │   ├── chain.py             # build_card_rag_chain() / build_answer_chain()
│   │   └── prompts.py           # citation-enforcing advisor prompt
│   ├── retrieval/
│   │   └── retriever.py         # hybrid BM25 + semantic retriever, RRF fusion
│   ├── evaluation/
│   │   ├── eval.py              # RAGAS eval harness (Gemini judge) + gold-set runner
│   │   ├── gold_set.json        # 20-query evaluation gold set
│   │   └── numeric_validator.py # numeric-claim extraction + numeric-hit guard
│   └── scripts/
│       ├── ingest_all.py        # build the vector store from all cards
│       ├── run_queries.py       # run sample queries from the CLI
│       └── run_hybrid.py        # try the hybrid / stratified retrieval
└── requirements.txt
```

## Cards currently ingested

| Card | Issuer | Type | Annual fee |
|---|---|---|---|
| SBI SimplyCLICK | SBI Card | online-shopping | ₹499 |
| Axis ACE | Axis Bank | cashback | ₹499 |
| HDFC Millennia | HDFC Bank | cashback | ₹1000 |

## Setup

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Add your Gemini API key (never commit this file)
echo "GOOGLE_API_KEY=your_key_here" > .env
```

`.env` is gitignored — the key is read from the environment at runtime and is never hardcoded
or sent to the browser.

## Configuration

Sensible defaults are baked in; the knobs worth knowing:

| Setting | Default | Where |
|---|---|---|
| `GOOGLE_API_KEY` | — (required) | `.env` / environment |
| Embedding model | `gemini-embedding-001` | `EMBEDDING_MODEL` in [embedder.py](src/vectorstore/embedder.py) |
| Generation & judge model | `gemini-2.5-flash` | [chain.py](src/rag/chain.py) / [eval.py](src/evaluation/eval.py) |
| Retrieval budget `k` | `10` | `as_runnable` in [retriever.py](src/retrieval/retriever.py) |
| RRF constant `rrf_k` | `60` | `reciprocal_rank_fusion` in [retriever.py](src/retrieval/retriever.py) |
| Chroma persist dir | `chroma_db/` | `persist_dir` in [embedder.py](src/vectorstore/embedder.py) |

> **Note:** the store must be built and queried with the *same* embedding model — changing
> `EMBEDDING_MODEL` requires re-running the ingest.

## Usage

```bash
# Build the vector store (run once, or after changing the card set)
PYTHONPATH=. .venv/bin/python -m src.scripts.ingest_all

# Launch the interactive frontend  →  http://localhost:8501
PYTHONPATH=. .venv/bin/streamlit run app.py

# Run sample queries from the terminal
PYTHONPATH=. .venv/bin/python -m src.scripts.run_queries

# Try the experimental hybrid + stratified retrieval
PYTHONPATH=. .venv/bin/python -m src.scripts.run_hybrid
```

## Evaluation

The RAG pipeline is scored with **RAGAS** using `gemini-2.5-flash` as the judge and
`gemini-embedding-001` for the embedding-based metric. Each question is scored across four
metrics, all ranging **0 → 1** (higher is better):

| Metric | What it measures | Inputs used |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? (anti-hallucination) | answer + contexts |
| **Answer relevancy** | Does the answer actually address the question asked? | question + answer |
| **Context precision** | Of the chunks retrieved, how many are actually relevant? (retrieval signal-to-noise) | question + contexts + reference |
| **Context recall** | Did retrieval surface all the context needed to cover the reference answer? | question + contexts + reference |

```bash
# Run the baseline eval (scores all questions; pass a smaller limit for a quick run)
PYTHONPATH=. .venv/bin/python -m src.evaluation.eval             # MMR,    week1 (5 Q)
PYTHONPATH=. .venv/bin/python -m src.evaluation.eval hybrid      # hybrid, week1 (5 Q)
PYTHONPATH=. .venv/bin/python -m src.evaluation.eval hybrid gold # hybrid, 20-query gold set
```

### Gold set (20 queries)

Retrieval is measured against a fixed **20-query gold set**
([gold_set.json](src/evaluation/gold_set.json)) — a test matrix of **three cards × three
failure modes**, scored by the *same* `run_baseline_eval`:

| Type | Count | What it stress-tests | Retrieval leg exercised |
|---|---|---|---|
| **numeric** | 7 | Exact values (`4%`, `Rs. 500`, `10,000`) | BM25 / keyword — where pure-dense fails |
| **comparison** | 7 | "Which card for X" across cards | Card stratification — surface *all* relevant cards |
| **conceptual** | 6 | How / eligibility / redemption | Dense / semantic — meaning, not tokens |

- **Numeric is over-weighted** — pure-semantic can't tell "5%" from "1%", so those queries
  are where hybrid must prove itself.
- **Every answer is grounded in the PDF text** — annual-fee/FX queries were excluded
  (those live in metadata, not chunks), so the gold answer measures the retriever, not the corpus.
- **`expected_card_ids` + `expected_numeric`** drive deterministic, no-LLM checks
  (ID-precision and the `numeric_hit` "zero fabricated fees" guard).
- **Comparisons span 2–3 cards** (`c7` all three), doubling as stratification checks.

### Before / after: dense-only vs hybrid (gold set, k=10)

Same 20 queries, same retrieval budget (k=10) — only the fusion differs. RAGAS judge,
higher is better:

| Retriever | Ctx Precision | Ctx Recall | Numeric-exact |
|---|---|---|---|
| Dense only (semantic) | 0.630 | 0.829 | 1.000 |
| **Hybrid (RRF)** | 0.619 | **0.883** | 1.000 |

**Takeaway:** hybrid lifts context recall **0.829 → 0.883** (clears the ≥ 0.85 target) at
equal precision. Numeric-exact is 1.000 for *both* — at k=10 a wide net over three cards
always contains the number, so the metric saturates; BM25's edge is a *ranking* effect
visible at smaller k. k=10 is the smallest budget that clears recall (up from 0.792 at k=6);
stratification was tried and rejected — it starved single-card queries.

### Week-1 smoke scores (5 questions)

An earlier, smaller sanity check across retrieval methods:

| Metric | MMR | Hybrid |
|---|---|---|
| Faithfulness | 0.903 | 0.897 |
| Answer relevancy | 0.940 | 0.958 |
| Context recall | 1.000 | 1.000 |
| Context precision | 0.618 | 0.783 |

**Overall:** both methods perform comparably — full context recall and strong, well-grounded answers — with hybrid edging out MMR (keyword matching locks onto the exact terms asked).

## Guardrails

The domain is financial — a wrong fee is a liability — so hallucination is defended at three
points, not one:

1. **Input gate** — `quality_check()` ([loader.py](src/ingestion/loader.py)) requires a
   concrete earning rate (regex: `5%`, `10X`, `₹500`) before a PDF is embedded; in strict mode
   it **fails closed**. Runs with zero API cost:

   ```bash
   PYTHONPATH=. .venv/bin/python -m src.scripts.ingest_all --check-only
   ```

2. **Generation** — a citation-enforcing prompt ([prompts.py](src/rag/prompts.py)) makes the
   model answer only from context, cite the card, and decline when the docs are silent.

3. **Output** — in `numeric_validator.py`, `extract_numeric_claims()` pulls numeric claims and
   `numeric_hit()` deterministically checks the needed figure was retrieved (it powers the
   eval's numeric-exact metric; wiring it into the live answer path is next).

> Regex layers are linear and length-capped to avoid catastrophic backtracking (ReDoS) on
> untrusted PDF/LLM text.

---

## Design decisions

Deliberate choices, and the failure mode each avoids:

- **Hybrid, not pure-semantic.** Embedding search ranks "$395" and "$350" as near-equal,
  silently returning wrong amounts on exact-value queries; keyword scoring fixes it —
  *measured* (recall 0.829 → 0.883 vs dense-only, numeric-exact 1.000), not assumed.
- **Card-stratified retrieval.** `stratified_retrieve()` pulls top chunks *per card* so
  "which card for X?" isn't biased toward the longest PDF.
- **One embedding model as source of truth.** `EMBEDDING_MODEL` is defined once; build and
  query must match, or vector distances are meaningless with no error to warn you.
- **Indian-card corpus.** US card PDFs omit reward rates; the Indian T&Cs contain them — so
  the quality gate and the assistant have concrete facts to work with.
- **RRF, not weighted-sum.** BM25 and cosine scores aren't on comparable scales, so blending
  needs unstable normalization. RRF fuses on **rank only** (`score = Σ 1/(rrf_k + rank)`) —
  no normalization, one knob (`rrf_k`). Weighted-sum stays behind `fusion="weighted"`.

### Vector store: Chroma, for now

**Chroma** fits this stage: zero-ops, local, <100k chunks — retrieval *quality* is the
bottleneck, not indexing. **Migrate to Pinecone/Weaviate** at ~500k+ chunks, multi-tenant
needs, or managed HNSW for uptime — an *operational* win, not better recall. Low-risk:
retrieval sits behind the single `HybridRetriever` interface.


---

## Limitations & roadmap

Known limits, stated honestly:

- **Demo-scale corpus.** Three cards are ingested. The architecture scales, but retrieval
  numbers are reported on a small corpus — e.g. numeric-exact retrieval *saturates* at high
  `k` because a wide net over three cards always contains the answer.
- **Hybrid is benchmarked but not the UI default.** The sidebar still defaults to MMR;
  making the benchmarked hybrid retriever the default is a one-line change under evaluation.
- **Numeric guard is eval-time only.** `numeric_hit()` runs in the evaluation harness;
  cross-checking numeric claims on the *live* answer path is not wired yet.
- **`rrf_k` not swept.** The RRF constant is fixed at 60; a sweep (10 / 60 / 100) against the
  gold set could squeeze out more recall.
- **LLM-judge variance.** RAGAS metrics occasionally return `IncompleteOutputException`
  (judge truncation), so a metric may score on n < 20 for a given run.

---

## Tests

Fast, **API-free** unit tests run in CI with no key or quota. They pin what regresses
silently: the quality gate's fail-closed contract, numeric extraction/guard, and hybrid
(RRF) ranking.

```bash
# Install dev dependencies (runtime + pytest) and run the suite
.venv/bin/pip install -r requirements-dev.txt
PYTHONPATH=. .venv/bin/pytest -q
```

CI runs this suite on every push and pull request to `main` (see the **CI** badge above),
and a `ruff` lint + format check gates merges to `main`.

---

## Contributing

1. Fork and create a feature branch off `main`.
2. Install dev dependencies: `.venv/bin/pip install -r requirements-dev.txt`.
3. Before opening a PR, run the checks that CI runs:
   ```bash
   PYTHONPATH=. .venv/bin/pytest -q
   ruff check . && ruff format --check .
   ```
4. Keep new logic covered by an API-free unit test where possible.

---

## License

Released under the [MIT License](LICENSE).

