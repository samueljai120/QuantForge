"""Widened money-conservation net (#2): extend self-detection beyond futures to spot/alt
accounting, the starting-balance-inflation bug class, the carry sleeve, and NaN/inf.
"""
from qf_safety.invariants import (
    check_btc_qty_nonnegative, check_alt_positions_sane, check_starting_balance_integrity,
    check_carry_sleeve_reconciles, check_finite_components, run_checks,
)


def test_btc_qty_negative_flagged():
    assert check_btc_qty_nonnegative({"btc_qty": -0.5})[0].severity == "critical"
    assert check_btc_qty_nonnegative({"btc_qty": 0.05}) == []


def test_alt_positions_negative_qty_flagged():
    bad = {"alt_positions": {"SOL": {"qty": -1.0, "avg_cost": 100.0}}}
    vs = check_alt_positions_sane(bad)
    assert any(v.name == "alt_qty_negative" for v in vs)
    assert check_alt_positions_sane({"alt_positions": {"SOL": {"qty": 2.0, "avg_cost": 100.0}}}) == []
    assert check_alt_positions_sane({"alt_positions": {}}) == []


def test_starting_balance_inflation_warns():
    assert check_starting_balance_integrity({"starting_balance": 7848.0})[0].name == "starting_balance_inflated"
    assert check_starting_balance_integrity({"starting_balance": 5000.0}) == []
    assert check_starting_balance_integrity({"starting_balance": 0.0})[0].severity == "critical"


def test_carry_sleeve_reconciliation():
    ok = {"realized_pnl": 12.0, "history": [{"pnl_usd": 7.0}, {"pnl_usd": 5.0}]}
    assert check_carry_sleeve_reconciles(ok) == []
    bad = {"realized_pnl": 50.0, "history": [{"pnl_usd": 1.0}]}
    assert check_carry_sleeve_reconciles(bad)[0].name == "carry_pnl_unreconciled"
    assert check_carry_sleeve_reconciles(None) == []   # no sleeve -> silent
    assert check_carry_sleeve_reconciles({}) == []


def test_finite_components():
    assert check_finite_components({"cash": float("nan"), "btc_qty": 0.0})[0].severity == "critical"
    assert check_finite_components({"cash": float("inf"), "btc_qty": 0.0})[0].severity == "critical"
    assert check_finite_components({"cash": 5000.0, "btc_qty": 0.05}) == []


def test_run_checks_includes_widened_and_carry():
    # healthy core but a carry sleeve mismatch -> surfaced via run_checks
    port = {"cash": 4000.0, "btc_qty": 0.05, "futures_pnl": 0.0, "futures_position": None,
            "starting_balance": 5000.0, "alt_positions": {}}
    carry = {"realized_pnl": 99.0, "history": [{"pnl_usd": 1.0}]}
    names = {v.name for v in run_checks(port, [], None, cur_price=64000.0, carry_state=carry)}
    assert "carry_pnl_unreconciled" in names
    # fully healthy -> clean
    assert run_checks(port, [], None, cur_price=64000.0, carry_state={"realized_pnl": 0.0, "history": []}) == []
