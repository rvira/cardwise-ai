import os
import sys
from src.ingestion.loader import load_card_pdf, quality_check


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from .env into the environment

from src.ingestion.loader import load_card_pdf
from src.ingestion.chunker import benchmark_chunking

CARDS = [
    {
        "pdf": "data/raw/sbi_simplyclick.pdf",
        "card_name": "SBI SimplyCLICK",
        "issuer": "SBI Card",
        "card_type": "online-shopping",
        "annual_fee": 499,
    },
    {
        "pdf": "data/raw/axis_ace.pdf",
        "card_name": "Axis ACE",
        "issuer": "Axis Bank",
        "card_type": "cashback",
        "annual_fee": 499,
    },
    {
        "pdf": "data/raw/hdfc_millennia.pdf",
        "card_name": "HDFC Millennia",
        "issuer": "HDFC Bank",
        "card_type": "cashback",
        "annual_fee": 1000,
    },
]

for c in CARDS:
    doc = load_card_pdf(
        c["pdf"], c["card_name"], c["issuer"], c["card_type"], c["annual_fee"]
    )
    print(f"\n=== {c['card_name']} ===")
    quality_check(doc)
    benchmark_chunking(doc.raw_text)
