# 💳 CardWise

A retrieval-augmented generation (RAG) assistant that answers questions about credit cards
**grounded in the official card documents** — with mandatory source citations and a
"don't answer if the docs don't say so" guardrail against hallucination.

Ask things like *"What is the cashback rate for Axis ACE, and any exclusions?"* and get an
answer drawn only from the ingested PDFs, with the card name cited.

---

## Demo

<p align="center">
  <img src="docs/demo.gif" alt="CardWise answering a credit-card question with cited sources" width="100%">
</p>

## How it works

<p align="center">
  <img src="docs/architecture.svg" alt="CardWise RAG architecture" width="100%">
</p>

1. **Ingest** — extract text from card PDFs, split into overlapping chunks, embed with
   `gemini-embedding-001`, and store in a persistent **Chroma** collection with per-card metadata.
2. **Query** — retrieve the most relevant chunks (MMR for diversity), feed them to
   `gemini-2.5-flash` through a strict, citation-enforcing prompt, and stream the answer.

## Tech stack

- **LangChain** (core, text-splitters, community, experimental) — pipeline orchestration
- **Google Gemini** — `gemini-embedding-001` (embeddings) + `gemini-2.5-flash` (generation)
- **Chroma** — local persistent vector store
- **PyMuPDF** — PDF text extraction
- **Streamlit** — interactive demo frontend

## Project structure

```
cardwise-ai/
├── app.py                      # Streamlit chat UI (demo frontend)
├── data/raw/                   # source card PDFs
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # PDF → text + quality_check
│   │   └── chunker.py          # chunk_document() + benchmark_chunking()
│   ├── vectorstore/
│   │   └── embedder.py         # build_vectorstore() / load_vectorstore()
│   ├── rag/
│   │   ├── chain.py            # build_card_rag_chain()
│   │   └── prompts.py          # citation-enforcing advisor prompt
│   └── scripts/
│       ├── ingest_all.py       # build the vector store from all cards
│       └── run_queries.py      # run sample queries from the CLI
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

## Usage

```bash
# Build the vector store (run once, or after changing the card set)
PYTHONPATH=. .venv/bin/python -m src.scripts.ingest_all

# Launch the interactive frontend  →  http://localhost:8501
PYTHONPATH=. .venv/bin/streamlit run app.py

# Or run sample queries from the terminal
PYTHONPATH=. .venv/bin/python -m src.scripts.run_queries
```