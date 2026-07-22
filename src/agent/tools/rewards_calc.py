"""Tool #2 — the deterministic rewards calculator (a pure function, no LLM inside).

Why a pure function and not the model: LLMs are unreliable at arithmetic, and in a
financial assistant a wrong number is a liability, not a UX bug. So the model
decides *when* to compute (by choosing this tool from its docstring); this function
decides *what* the numbers are — deterministically, and unit-tested with zero API
cost. That determinism is the whole reason an agent beats plain RAG here.

The single input (`spend_profile_json`) is produced by the LLM, so it is untrusted:
it is length-capped, parsed as JSON only (never eval'd), and validated against an
allow-list of categories with per-value type/finite/range checks. Anything malformed
is returned as a model-readable ``{"error": ...}`` rather than raised, so the agent
can re-ask the user instead of crashing.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

# Allow-list of spend categories. Any key outside this set is rejected — the LLM
# does not get to invent categories the rate tables don't cover.
SPEND_CATEGORIES = ("dining", "grocery", "online", "bills", "travel", "other")

# Illustrative cashback-equivalent rates for the three catalogued cards. These are
# hardcoded demo values (not lifted verbatim from each card's T&C) — the *mechanism*
# and its determinism are the point, not authoritative bps; move to data/ when the
# numbers need to be canonical. `annual_fee` mirrors src/ingestion/loader.py CARDS.
CARD_RATES = {
    "sbi_simplyclick": {
        "name": "SBI SimplyCLICK",
        "rates": {
            "online": 0.025,
            "dining": 0.025,
            "grocery": 0.01,
            "bills": 0.01,
            "travel": 0.01,
            "other": 0.01,
        },
        "annual_fee": 499,
        "monthly_cap": None,
    },
    "axis_ace": {
        "name": "Axis ACE",
        "rates": {
            "bills": 0.05,
            "dining": 0.04,
            "online": 0.015,
            "grocery": 0.015,
            "travel": 0.01,
            "other": 0.015,
        },
        "annual_fee": 499,
        # Axis ACE caps its 4%/5% accelerated categories (Swiggy/Zomato/Ola +
        # utility/recharge bill payments) at Rs.500/month combined. Modeled here as
        # an overall monthly reward cap — a simplification, but it keeps the tool in
        # agreement with the source documents for accelerated-spend queries.
        "monthly_cap": 500,
    },
    "hdfc_millennia": {
        "name": "HDFC Millennia",
        "rates": {
            "online": 0.05,
            "dining": 0.05,
            "grocery": 0.01,
            "bills": 0.01,
            "travel": 0.01,
            "other": 0.01,
        },
        "annual_fee": 1000,
        "monthly_cap": 1000,  # HDFC Millennia caps cashback at ₹1000/month
    },
}

_MAX_INPUT_LEN = 2000  # cap the untrusted JSON string before parsing (DoS guard)
_MAX_MONTHLY_SPEND = 100_000_000  # ₹10cr/mo — reject absurd values, keep math finite


def _parse_spend(spend_profile_json: str) -> dict:
    """Validate the LLM-supplied JSON into a clean ``{category: monthly_amount}``
    dict: allow-list categories, require finite non-negative numbers, reject
    everything else. Raises ``ValueError`` with a client-safe message on any
    problem (no stack trace, no provider internals)."""
    if not isinstance(spend_profile_json, str):
        raise ValueError("spend profile must be a JSON string")
    if len(spend_profile_json) > _MAX_INPUT_LEN:
        raise ValueError("spend profile is too long")
    try:
        raw = json.loads(spend_profile_json)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("spend profile must be valid JSON")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            'spend profile must be a non-empty object, e.g. {"dining": 8000}'
        )

    spend: dict[str, float] = {}
    for category, amount in raw.items():
        if category not in SPEND_CATEGORIES:
            raise ValueError(
                f"unknown category {category!r}; allowed: {', '.join(SPEND_CATEGORIES)}"
            )
        # bool is an int subclass — exclude it so True/False can't pose as amounts.
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError(f"amount for {category!r} must be a number")
        # json.loads accepts NaN/Infinity by default — reject non-finite values.
        if amount != amount or amount in (float("inf"), float("-inf")):
            raise ValueError(f"amount for {category!r} must be finite")
        if amount < 0 or amount > _MAX_MONTHLY_SPEND:
            raise ValueError(f"amount for {category!r} is out of range")
        spend[category] = float(amount)
    return spend


def _annual_value(spend: dict, card: dict) -> dict:
    """Compute one card's annual reward value for a monthly spend profile, applying
    the card's monthly cashback cap (if any) before annualising."""
    rates = card["rates"]
    monthly_reward = sum(amount * rates[cat] for cat, amount in spend.items())
    cap = card["monthly_cap"]
    if cap is not None:
        monthly_reward = min(monthly_reward, float(cap))
    gross = round(monthly_reward * 12, 2)
    return {
        "card_name": card["name"],
        "gross_rewards": gross,
        "annual_fee": card["annual_fee"],
        "net_after_fee": round(gross - card["annual_fee"], 2),
    }


def _summarize(results: list) -> dict:
    """Deterministic advice flags computed from the ranked results, so the caller
    (the LLM) can surface honest caveats instead of mis-selling a result. `results`
    must already be sorted best-first by net_after_fee.

    Catches the cases where the raw ranking is technically correct but misleading:
      * every card is net-negative (rewards never cover the fee for this spend);
      * all cards earn the same gross (no rewards difference — it's a fee compare);
      * several cards tie for the best net value.
    """
    best_net = results[0]["net_after_fee"]
    best_cards = [r["card_name"] for r in results if r["net_after_fee"] == best_net]
    all_net_negative = all(r["net_after_fee"] < 0 for r in results)
    no_rewards_difference = len({r["gross_rewards"] for r in results}) == 1

    notes = []
    if all_net_negative:
        notes.append(
            "Every card's net value is negative — for this spend none earns back its "
            "annual fee, so no card here is worth holding for these rewards alone."
        )
    if no_rewards_difference:
        notes.append(
            "All cards earn the same gross rewards for this spend, so they do not "
            "differ on rewards — the only difference is the annual fee."
        )
    if len(best_cards) > 1:
        notes.append(
            "Cards tied for the best net value: " + ", ".join(best_cards) + "."
        )

    return {
        "best_net_after_fee": best_net,
        "best_cards": best_cards,
        "all_net_negative": all_net_negative,
        "no_rewards_difference": no_rewards_difference,
        "notes": notes,
    }


@tool
def rewards_value(spend_profile_json: str) -> str:
    """Compute and RANK the three cards by estimated ANNUAL rewards value, net of
    the annual fee, for a user's MONTHLY spend profile.

    Use this for ANY question about how much a card earns/saves, or which card is
    best or better for a spending pattern — it is the only correct way to compute or
    compare reward amounts. Do NOT use card_search for that.

    Input: a JSON object mapping spend categories to MONTHLY rupee amounts. Valid
    categories: dining, grocery, online, bills, travel, other. Map merchants to a
    category first — e.g. Swiggy/Zomato -> dining, Amazon/Flipkart -> online,
    utility/recharge/bill payments -> bills, flights/hotels -> travel.
    Example: {"dining": 8000, "online": 5000, "bills": 3000, "other": 10000}
    """
    try:
        spend = _parse_spend(spend_profile_json)
    except ValueError as err:
        # Model-readable error (not an exception) so the agent can re-ask the user.
        return json.dumps({"error": str(err)})

    results = [_annual_value(spend, card) for card in CARD_RATES.values()]
    results.sort(key=lambda r: r["net_after_fee"], reverse=True)
    return json.dumps(
        {
            "currency": "INR",
            "period": "annual",
            "ranked_by": "net_after_fee",
            "results": results,
            "summary": _summarize(results),
        },
        indent=2,
    )
