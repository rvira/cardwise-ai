"""Tests for the deterministic rewards calculator (AI-3.3).

Pure function, no API and no LLM — this is exactly why the tool is a plain function
and not the model: its output is reproducible and its input validation is provable.
"""

import json

from src.agent.tools.rewards_calc import (
    CARD_RATES,
    _annual_value,
    _parse_spend,
    rewards_value,
)


def _call(profile: dict) -> dict:
    """Invoke the @tool with a dict of args (StructuredTool interface)."""
    return json.loads(rewards_value.invoke({"spend_profile_json": json.dumps(profile)}))


def _raw(raw_json: str) -> dict:
    """Invoke with a raw (possibly malformed) JSON string."""
    return json.loads(rewards_value.invoke({"spend_profile_json": raw_json}))


# --- happy path --------------------------------------------------------------


def test_returns_all_cards_ranked_by_net_after_fee():
    result = _call({"dining": 8000, "travel": 3000})
    assert result["currency"] == "INR"
    cards = [r["card_name"] for r in result["results"]]
    assert set(cards) == {c["name"] for c in CARD_RATES.values()}
    nets = [r["net_after_fee"] for r in result["results"]]
    assert nets == sorted(nets, reverse=True)  # ranked best-first


def test_net_is_gross_minus_annual_fee():
    result = _call({"other": 10000})
    for row in result["results"]:
        assert row["net_after_fee"] == round(
            row["gross_rewards"] - row["annual_fee"], 2
        )


def test_arithmetic_is_exact_and_deterministic():
    # Axis ACE: bills 0.05 -> 5000 * 0.05 * 12 = 3000 gross, - 499 fee = 2501 net.
    a = _call({"bills": 5000})
    b = _call({"bills": 5000})
    assert a == b  # deterministic
    axis = next(r for r in a["results"] if r["card_name"] == "Axis ACE")
    assert axis["gross_rewards"] == 3000.0
    assert axis["net_after_fee"] == 2501.0


def test_axis_monthly_cap_is_applied():
    # Axis dining 0.04: at Rs.20,000/mo that's Rs.800/mo, above the Rs.500 cap.
    # Capped gross = 500 * 12 = 6000; net = 6000 - 499 fee = 5501.
    result = _call({"dining": 20000})
    axis = next(r for r in result["results"] if r["card_name"] == "Axis ACE")
    assert axis["gross_rewards"] == 6000.0
    assert axis["net_after_fee"] == 5501.0


def test_hdfc_monthly_cap_is_applied():
    # Huge online spend: uncapped HDFC 5% would dwarf the Rs.1000/mo cap.
    # Capped gross = 1000 * 12 = 12000; net = 12000 - 1000 fee = 11000.
    result = _call({"online": 10_000_000})
    hdfc = next(r for r in result["results"] if r["card_name"] == "HDFC Millennia")
    assert hdfc["gross_rewards"] == 12000.0
    assert hdfc["net_after_fee"] == 11000.0


# --- advice-quality summary flags --------------------------------------------


def test_summary_flags_net_negative_tie_and_fee_only_comparison():
    # The screenshot case: Rs.100/day -> Rs.3,000/mo travel. All cards earn 1% on
    # travel, so gross is identical (Rs.360) and every net is negative after fees.
    summary = _call({"travel": 3000})["summary"]
    assert summary["all_net_negative"] is True  # don't sell a money-loser as "best"
    assert summary["no_rewards_difference"] is True  # it's a fee comparison only
    assert set(summary["best_cards"]) == {"SBI SimplyCLICK", "Axis ACE"}  # tie
    assert summary["notes"]  # human-readable caveats present


def test_summary_positive_case_has_single_winner_no_warnings():
    # HDFC online 5% at Rs.20k/mo hits its Rs.1000 cap -> clearly positive & best.
    summary = _call({"online": 20000})["summary"]
    assert summary["all_net_negative"] is False
    assert summary["no_rewards_difference"] is False
    assert summary["best_cards"] == ["HDFC Millennia"]


# --- untrusted-input validation (fail closed) --------------------------------


def test_malformed_json_is_rejected():
    assert "error" in _raw("{not json")


def test_unknown_category_is_rejected():
    assert "error" in _call({"crypto": 5000})


def test_non_numeric_amount_is_rejected():
    assert "error" in _call({"dining": "lots"})


def test_boolean_amount_is_rejected():
    # bool is an int subclass — must not sneak through as a number.
    assert "error" in _call({"dining": True})


def test_negative_amount_is_rejected():
    assert "error" in _call({"dining": -100})


def test_non_finite_amount_is_rejected():
    # json.loads accepts NaN/Infinity by default; the guard must reject them.
    assert "error" in _raw('{"dining": NaN}')


def test_empty_object_is_rejected():
    assert "error" in _call({})


def test_oversized_input_is_rejected():
    assert "error" in _raw(json.dumps({"other": 1}) + " " * 5000)


# --- helper unit tests -------------------------------------------------------


def test_parse_spend_coerces_ints_to_float():
    parsed = _parse_spend('{"dining": 100}')
    assert parsed == {"dining": 100.0}
    assert isinstance(parsed["dining"], float)


def test_annual_value_uncapped_card():
    card = CARD_RATES["sbi_simplyclick"]
    # online 0.025 * 1000 * 12 = 300 gross; - 499 fee = -199 net.
    row = _annual_value({"online": 1000.0}, card)
    assert row["gross_rewards"] == 300.0
    assert row["net_after_fee"] == -199.0
