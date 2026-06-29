"""Money-conservation invariants — the self-DETECTION layer that would have caught the
margin-orphan bug in ONE cycle (4 FUTURES_OPEN / 0 FUTURES_CLOSE; $400 vanished on a flat
price). Each check fires on its bug signature and stays silent on a healthy book.
"""
from qf_safety.invariants import (
    Violation, equity, check_futures_parity, check_futures_pnl_ledger,
    check_cash_nonnegative, check_equity_conservation, run_checks,
)


def _names(vs):
    return {v.name for v in vs}


# --- futures open/close parity (the exact orphan signature) ---
def test_parity_flags_orphaned_margin():
    trades = [{"type": "FUTURES_OPEN"}] * 4 + [{"type": "FUTURES_CLOSE"}] * 0
    vs = check_futures_parity(trades, position_open=False)
    assert vs and vs[0].name == "futures_open_close_parity" and vs[0].severity == "critical"


def test_parity_ok_when_closes_match():
    trades = [{"type": "FUTURES_OPEN"}] * 4 + [{"type": "FUTURES_CLOSE"}] * 3
    assert check_futures_parity(trades, position_open=True) == []   # 4 == 3 + 1 open


def test_parity_ok_single_open_position():
    assert check_futures_parity([{"type": "FUTURES_OPEN"}], position_open=True) == []


# --- futures_pnl vs ledger (catches ledger-blind closes) ---
def test_futures_pnl_ledger_mismatch_flagged():
    port = {"futures_pnl": -100.0}
    trades = [{"type": "FUTURES_CLOSE", "pnl_usd": 0.0}]
    vs = check_futures_pnl_ledger(port, trades)
    assert vs and vs[0].name == "futures_pnl_ledger_mismatch"


def test_futures_pnl_ledger_ok():
    port = {"futures_pnl": 12.0}
    trades = [{"type": "FUTURES_CLOSE", "pnl_usd": 7.0}, {"type": "FUTURES_CLOSE", "pnl_usd": 5.0}]
    assert check_futures_pnl_ledger(port, trades) == []


# --- cash non-negative ---
def test_cash_negative_flagged():
    assert check_cash_nonnegative({"cash": -3.0})[0].severity == "critical"
    assert check_cash_nonnegative({"cash": 10.0}) == []


# --- cross-cycle equity conservation (the general money-bug net) ---
def test_equity_conservation_flags_unexplained_drop():
    prev = {"equity": 5000.0, "price": 100.0, "btc_qty": 10.0, "last_trade_ts": "t0"}
    # flat price, but cash dropped 400 (orphaned margin), no position
    port = {"cash": 3600.0, "btc_qty": 10.0, "futures_position": None}
    vs = check_equity_conservation(prev, port, trades=[{"ts": "t1", "pnl_usd": 9.0, "fee": 0.0}], cur_price=100.0)
    assert vs and vs[0].name == "equity_unreconciled" and vs[0].severity == "critical"


def test_equity_conservation_ok_on_price_move():
    prev = {"equity": 5000.0, "price": 100.0, "btc_qty": 10.0, "last_trade_ts": "t0"}
    port = {"cash": 4000.0, "btc_qty": 10.0, "futures_position": None}   # 4000 + 10*110 = 5100
    assert check_equity_conservation(prev, port, trades=[], cur_price=110.0) == []  # +100 explained by MTM


def test_equity_conservation_no_prev_is_silent():
    assert check_equity_conservation(None, {"cash": 1.0}, trades=[], cur_price=100.0) == []


# --- equity helper ---
def test_equity_includes_all_lanes():
    port = {"cash": 1000.0, "btc_qty": 0.05,
            "futures_position": {"direction": "SHORT", "margin": 200.0, "notional": 1000.0, "entry_price": 100.0}}
    # cash 1000 + btc 0.05*90=4.5 + margin 200 + short upnl 1000*(1-90/100)=+100
    assert equity(port, 90.0) == 1000.0 + 4.5 + 200.0 + 100.0


# --- healthy full run is clean ---
def test_run_checks_clean_on_healthy_book():
    port = {"cash": 4000.0, "btc_qty": 0.05, "futures_pnl": 0.0, "futures_position": None}
    assert run_checks(port, trades=[], prev_snapshot=None, cur_price=64000.0) == []
