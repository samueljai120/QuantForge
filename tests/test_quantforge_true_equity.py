#!/usr/bin/env python3
"""Tests for the shared true-equity computation in quantforge_agent.

Guards against the two 2026-06-12 equity bugs:
  1. Margin transfers (futures/prehedge open) misread as losses because
     the equity formula only counted cash + spot BTC.
  2. Futures unrealized PnL double-counting leverage
     (notional already equals margin * leverage).

Run: python3 test_quantforge_true_equity.py
"""
import sys

import quantforge_agent as qa

PRICE = 60_000.0
FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILURES.append(name)
        print(f"  FAIL  {name}: {e}")


def make_port(cash=5_000.0, btc_qty=0.0):
    return {
        "cash": cash,
        "btc_qty": btc_qty,
        "alt_positions": {},
        "futures_position": {"direction": None, "margin": 0, "notional": 0,
                             "entry_price": 0, "opened_at": None},
        "prehedge": {"open": False, "direction": "SHORT", "margin": 0.0,
                     "notional": 0.0, "entry_price": 0.0, "leverage": 1,
                     "opened_at": None, "pnl_realized": 0.0},
        "starting_balance": 5_000.0,
        "peak_equity": 5_000.0,
    }


def open_futures(port, price, margin, leverage, direction="LONG"):
    """Simulate the FUTURES_OPEN cash->margin transfer the agent performs."""
    port["cash"] -= margin
    port["futures_position"] = {
        "direction": direction,
        "margin": margin,
        "notional": margin * leverage,
        "entry_price": price,
        "leverage": leverage,
        "opened_at": "2026-06-12T00:00:00+00:00",
    }


def test_futures_open_is_equity_neutral():
    """Opening a futures position moves cash to margin — equity must not change.
    This is the bug that recorded a -$547 phantom loss in regime_perf."""
    port = make_port()
    eq_before = qa._true_equity(port, PRICE)
    open_futures(port, PRICE, margin=500.0, leverage=2)
    eq_after = qa._true_equity(port, PRICE)
    assert abs(eq_after - eq_before) < 1e-6, (
        f"margin transfer changed equity: {eq_before:.2f} -> {eq_after:.2f}")


def test_futures_unrealized_no_leverage_double_count():
    """At 2x leverage with $500 margin, notional is $1000. A +1% move on a
    LONG is +$10 — not +$20 (notional already embeds leverage)."""
    port = make_port()
    eq_base = qa._true_equity(port, PRICE)
    open_futures(port, PRICE, margin=500.0, leverage=2, direction="LONG")
    eq_after_move = qa._true_equity(port, PRICE * 1.01)
    gain = eq_after_move - eq_base
    assert abs(gain - 10.0) < 1e-6, f"expected +$10 unrealized, got {gain:+.2f}"


def test_futures_short_unrealized_sign():
    """SHORT profits when price drops."""
    port = make_port()
    eq_base = qa._true_equity(port, PRICE)
    open_futures(port, PRICE, margin=500.0, leverage=2, direction="SHORT")
    eq_after_drop = qa._true_equity(port, PRICE * 0.99)
    gain = eq_after_drop - eq_base
    assert abs(gain - 10.0) < 1e-6, f"expected +$10 on short into drop, got {gain:+.2f}"


def test_futures_leverage_key_missing_falls_back_to_notional():
    """Legacy positions lack the 'leverage' key. Unrealized PnL must still be
    pct * notional with no extra multiplier from the FUTURES_LEVERAGE fallback."""
    port = make_port()
    eq_base = qa._true_equity(port, PRICE)
    open_futures(port, PRICE, margin=500.0, leverage=2, direction="LONG")
    del port["futures_position"]["leverage"]
    eq_after_move = qa._true_equity(port, PRICE * 1.01)
    gain = eq_after_move - eq_base
    assert abs(gain - 10.0) < 1e-6, (
        f"legacy position without leverage key mispriced: got {gain:+.2f}, want +10.00")


def test_prehedge_open_is_equity_neutral():
    """Opening a prehedge moves cash to prehedge margin — equity must not change."""
    port = make_port()
    eq_before = qa._true_equity(port, PRICE)
    margin = 50.0
    port["cash"] -= margin
    port["prehedge"] = {"open": True, "direction": "SHORT", "margin": margin,
                        "notional": margin * 1, "entry_price": PRICE, "leverage": 1,
                        "opened_at": "2026-06-12T00:00:00+00:00", "pnl_realized": 0.0}
    eq_after = qa._true_equity(port, PRICE)
    assert abs(eq_after - eq_before) < 1e-6, (
        f"prehedge margin transfer changed equity: {eq_before:.2f} -> {eq_after:.2f}")


def test_prehedge_short_unrealized():
    """Open SHORT prehedge profits as price drops: 1% drop on $50 notional = +$0.50."""
    port = make_port()
    eq_base = qa._true_equity(port, PRICE)
    port["cash"] -= 50.0
    port["prehedge"] = {"open": True, "direction": "SHORT", "margin": 50.0,
                        "notional": 50.0, "entry_price": PRICE, "leverage": 1,
                        "opened_at": "2026-06-12T00:00:00+00:00", "pnl_realized": 0.0}
    eq_after = qa._true_equity(port, PRICE * 0.99)
    gain = eq_after - eq_base
    assert abs(gain - 0.5) < 1e-6, f"expected +$0.50, got {gain:+.2f}"


def test_unrealized_matches_realized_close_math():
    """Mark-to-market PnL must equal what the close path would realize:
    close pays notional * (price/entry - 1) for LONG."""
    port = make_port()
    open_futures(port, PRICE, margin=500.0, leverage=3, direction="LONG")
    exit_price = PRICE * 1.027
    fp = port["futures_position"]
    realized = fp["notional"] * (exit_price / fp["entry_price"] - 1.0)
    eq_open = qa._true_equity(port, exit_price)
    # simulate the close exactly as the agent does
    port["cash"] += fp["margin"] + realized
    port["futures_position"] = {"direction": None, "margin": 0, "notional": 0,
                                "entry_price": 0, "opened_at": None}
    eq_closed = qa._true_equity(port, exit_price)
    assert abs(eq_open - eq_closed) < 1e-6, (
        f"mark-to-market {eq_open:.2f} != post-close {eq_closed:.2f} "
        f"(unrealized formula disagrees with realized close math)")


def test_call_sites_use_shared_function():
    """Source-level drift guard: panic halt, halt auto-recovery, attribution and
    the cycle equity block must all go through _true_equity; the leverage
    double-count expression must be gone; futures open must store leverage."""
    import inspect
    src = inspect.getsource(qa)
    assert "cur_equity_attr = _true_equity(" in src, (
        "per-regime attribution no longer uses _true_equity")
    assert "pct_change * notional * leverage" not in src, (
        "leverage double-count expression still present")
    assert "pct_change * notional * lev" not in src.replace(
        "pct_change * notional * leverage", ""), (
        "leverage double-count expression still present (lev variant)")
    assert '"leverage": active_leverage' in src, (
        "futures open does not store leverage in the position dict")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} true-equity tests against quantforge_agent...")
    for name, fn in tests:
        check(name, fn)
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("\nAll tests passed.")
    sys.exit(0)
