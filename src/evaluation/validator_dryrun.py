# src/evaluation/validator_dryrun.py
"""
Generates real answers from the RAG chain, runs extract_numeric_claims() on
each, and auto-flags where the extractor is WEAK

task list (not a vague "improve later"):
  - word-form numbers ("four times", "ten")           → regex can't see these
  - decimal multipliers ("1.5x")                        → captured as a fragment
  - fragment false-positives ("000 x" from "₹2,000 x")  → mid-number match
"""
import os
import re
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dotenv import load_dotenv

load_dotenv()

from src.vectorstore.embedder import load_vectorstore
from src.rag.chain import build_card_rag_chain
from src.evaluation.numeric_validator import extract_numeric_claims

QUESTIONS = [
    "What cashback does HDFC Millennia earn on online spends, and on other spends?",
    "How many reward points does SBI SimplyCLICK earn on online spends?",
    "What cashback does Axis ACE give on Google Pay bill payments, and the monthly cap?",
    "What is the annual fee for HDFC Millennia?",
    "If I spend ₹2,000 online on HDFC Millennia, how much cashback do I earn? Show the calculation.",
]

# linear, length-capped probes for things a perfect extractor SHOULD catch
WORD_NUM = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twice|thrice|[a-z]+\s+times)\b",
    re.IGNORECASE,
)
DECIMAL_X = re.compile(r"\d+\.\d+\s?x\b", re.IGNORECASE)


def flags_for(answer: str, claims: list) -> dict:
    capped = answer[:200_000]
    joined = " ".join(claims)
    # word-form numbers the regex set cannot represent
    words = sorted({w.strip() for w in WORD_NUM.findall(capped)})
    # decimal multipliers not fully captured (extractor yields just the trailing digit)
    decimals = sorted({d for d in DECIMAL_X.findall(capped) if d not in joined})
    # fragment false-positives: an extracted claim sitting mid-number (prev char digit/comma)
    fragments = []
    for c in claims:
        for m in re.finditer(re.escape(c), capped):
            prev = capped[m.start() - 1] if m.start() else ""
            if prev.isdigit() or prev == ",":
                fragments.append(c)
                break
    return {
        "word_numbers": words,
        "decimal_x": decimals,
        "fragment_fp": sorted(set(fragments)),
    }


def main():
    chain = build_card_rag_chain(load_vectorstore())
    agg = {"word_numbers": set(), "decimal_x": set(), "fragment_fp": set()}

    for i, q in enumerate(QUESTIONS, 1):
        answer = chain.invoke(q)
        claims = extract_numeric_claims(answer)
        f = flags_for(answer, claims)
        for k in agg:
            agg[k].update(f[k])

        print("\n" + "=" * 80)
        print(f"Q{i}: {q}")
        print("-" * 80)
        print(answer[:600])
        print(f"\n  extracted claims : {claims}")
        print(f"  ⚠ word-numbers   : {f['word_numbers'] or '—'}")
        print(f"  ⚠ decimal x      : {f['decimal_x'] or '—'}")
        print(f"  ⚠ fragment FPs   : {f['fragment_fp'] or '—'}")

    print("\n" + "#" * 80)
    print("MISSED / WEAK PATTERNS (Week 3 task list):")
    print(f"  word-form numbers not extracted : {sorted(agg['word_numbers']) or '—'}")
    print(f"  decimal multipliers mis-captured: {sorted(agg['decimal_x']) or '—'}")
    print(f"  fragment false-positives        : {sorted(agg['fragment_fp']) or '—'}")
    print("#" * 80)


if __name__ == "__main__":
    main()
