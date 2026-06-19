import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from .env into the environment

from src.ingestion.loader import load_card_pdf
from src.ingestion.chunker import benchmark_chunking

# Pick the richest Indian PDF for the benchmark
doc = load_card_pdf(
    "data/raw/hdfc_millennia.pdf", "HDFC Millennia", "HDFC Bank", "cashback", 1000
)
print(f"Loaded {len(doc.raw_text)} chars\n")
benchmark_chunking(doc.raw_text)
