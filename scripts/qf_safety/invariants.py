"""Money-conservation invariants — the self-DETECTION layer.

A system can only self-heal what it can DETECT. The margin-orphan bug (2026-06-22)
destroyed ~$400 every regime flip SILENTLY: no crash, no error, and the ledger even
showed +$18 "realized" — because nothing ever asked *"does the money conserve?"*. These
pure checks answer that question and FLAG any unexplained loss / orphaned margin / ledger
mismatch, so the self-heal loop and the daily report catch it in one cycle instead of a
human hunting for it weeks later.

Core law: in a paper book, trades are equity-NEUTRAL (a sell converts BTC->cash at market;
an open moves cash->margin), so equity can only legitimately change from price MTM on held
positions minus fees. Anything else is a leak/orphan/double-count.

Pure + dependency-free: each check takes plain dicts/lists and returns a list of
Violation. A separate runner (quantforge_invariants.py) loads the live files, persists a
snapshot for cross-cycle reconciliation, and escalates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Violation:
    name: str
    severity: str   # "critical" | "warning"
    detail: str
    hint: str = ""   # diagnosis pointer for the self-heal loop


def equity(port: dict, price: float) -> float:
    """True total equity = cash + spot BTC + futures margin + futures unrealized PnL.
    (The same money the portfolio actually owns; futures margin is paper money still held.)"""
    price = float(price or 0)
    cash = float(port.get("cash", 0) or 0)
    btc = float(port.get("btc_qty", 0) or 0) * price
    fp = port.get("futures_position") or {}
    fmargin = float(fp.get("margin", 0) or 0)
    fupnl = 0.0
    if fp.get("direction") and float(fp.get("notional", 0) or 0) > 0:
        entry = float(fp.get("entry_price", price) or price)
        if entry > 0 and price > 0:
            if fp["direction"] == "LONG":
                fupnl = float(fp["notional"]) * (price / entry - 1.0)
            else:
                fupnl = float(fp["notional"]) * (1.0 - price / entry)
    return cash + btc + fmargin + fupnl


def check_futures_parity(trades: list, *, position_open: bool) -> List[Violation]:
    """Every FUTURES_OPEN must have a matching FUTURES_CLOSE unless a position is open now:
    n_open == n_close + (1 if open else 0). A surplus = position(s) opened but never closed
    => orphaned margin. This is the exact 2026-06-22 bug signature (4 opens / 0 closes / flat)."""
    n_open = sum(1 for t in trades if t.get("type") == "FUTURES_OPEN")
    n_close = sum(1 for t in trades if t.get("type") == "FUTURES_CLOSE")
    net = n_open - n_close
    cur = 1 if position_open else 0
    if net != cur:
        orphaned = net - cur
        return [Violation(
            "futures_open_close_parity", "critical",
            f"{n_open} FUTURES_OPEN vs {n_close} FUTURES_CLOSE with {cur} open now -> "
            f"{orphaned} position(s) opened but never closed (orphaned margin / un-ledgered close)",
            "quantforge_agent._execute_futures: every close (incl. direction flip) must credit "
            "margin+pnl back to cash AND append a FUTURES_CLOSE row",
        )]
    return []


def check_futures_pnl_ledger(port: dict, trades: list, *, tol: float = 1.0) -> List[Violation]:
    """portfolio.futures_pnl must ~equal the sum of FUTURES_CLOSE pnl_usd in the ledger.
    A mismatch means a close credited cash/pnl but never wrote a FUTURES_CLOSE (audit
    blindness — the gates/report then read an incomplete ledger)."""
    ledger = sum(float(t.get("pnl_usd", 0) or 0) for t in trades if t.get("type") == "FUTURES_CLOSE")
    book = float(port.get("futures_pnl", 0) or 0)
    if abs(book - ledger) > tol:
        return [Violation(
            "futures_pnl_ledger_mismatch", "warning",
            f"portfolio futures_pnl ${book:+.2f} != ledger FUTURES_CLOSE sum ${ledger:+.2f} "
            f"(gap ${book - ledger:+.2f})",
            "a futures close path credits futures_pnl but does not append_trade(FUTURES_CLOSE)",
        )]
    return []


def check_cash_nonnegative(port: dict) -> List[Violation]:
    cash = float(port.get("cash", 0) or 0)
    if cash < -1e-6:
        return [Violation("cash_negative", "critical", f"cash is ${cash:+.2f} (< 0) — over-deployed",
                          "position sizing deployed more than available cash")]
    return []


def check_equity_conservation(prev: Optional[dict], port: dict, trades: list, *,
                              cur_price: float, abs_tol: float = 30.0, pct_tol: float = 0.015) -> List[Violation]:
    """Cross-cycle conservation: equity may only change by price MTM on held positions
    minus fees (trades are equity-neutral). Anything beyond that is UNEXPLAINED = a
    leak/orphan/double-count. This is the general money-bug net that catches NOVEL bugs,
    not just the futures one. Silent on the first run (no prior snapshot)."""
    if not prev:
        return []
    prev_price = float(prev.get("price", 0) or 0)
    if prev_price <= 0 or not cur_price:
        return []
    prev_eq = float(prev.get("equity", 0) or 0)
    prev_btc = float(prev.get("btc_qty", 0) or 0)
    cur_eq = equity(port, cur_price)
    # Legit delta = spot price MTM on the BTC held at prev - fees since prev. (Futures MTM
    # between snapshots is folded into the tolerance; the parity / pnl-ledger checks cover
    # futures bugs directly. Approximation is exact near flat price — where orphans live.)
    last_ts = str(prev.get("last_trade_ts", ""))
    fees_since = sum(float(t.get("fee", 0) or 0) for t in trades if str(t.get("ts", "")) > last_ts)
    spot_mtm = prev_btc * (cur_price - prev_price)
    expected = spot_mtm - fees_since
    unexplained = (cur_eq - prev_eq) - expected
    thr = max(abs_tol, pct_tol * max(prev_eq, 1.0))
    if abs(unexplained) > thr:
        sev = "critical" if abs(unexplained) > 3 * thr else "warning"
        return [Violation(
            "equity_unreconciled", sev,
            f"equity moved ${cur_eq - prev_eq:+.2f} but only ${expected:+.2f} is explained by "
            f"price MTM/fees -> ${unexplained:+.2f} UNEXPLAINED (threshold ${thr:.0f})",
            "money changed without a trade/price reason — orphaned margin, double-count, or an "
            "untracked write to the portfolio",
        )]
    return []


def check_btc_qty_nonnegative(port: dict) -> List[Violation]:
    """Spot accounting: you can't hold negative BTC (a sell-more-than-held leak)."""
    q = float(port.get("btc_qty", 0) or 0)
    if q < -1e-9:
        return [Violation("btc_qty_negative", "critical", f"btc_qty {q} < 0 — sold more BTC than held",
                          "spot sell path is not bounded by held quantity")]
    return []


def check_alt_positions_sane(port: dict) -> List[Violation]:
    """Alt-lane accounting: every alt position must have qty >= 0 and a non-negative cost basis."""
    out: List[Violation] = []
    for sym, p in (port.get("alt_positions") or {}).items():
        if not isinstance(p, dict):
            continue
        qty = float(p.get("qty", 0) or 0)
        cost = float(p.get("avg_cost", p.get("avg_price", 0)) or 0)
        if qty < -1e-9:
            out.append(Violation("alt_qty_negative", "critical", f"{sym} alt qty {qty} < 0",
                                 "alt sell path is not bounded by held quantity"))
        if cost < 0:
            out.append(Violation("alt_cost_negative", "warning", f"{sym} avg_cost {cost} < 0",
                                 "alt cost-basis accounting"))
    return out


def check_starting_balance_integrity(port: dict, *, base: float = 5000.0) -> List[Violation]:
    """The baseline used for PnL% must not silently inflate (the v30 topup bug hid losses by
    raising starting_balance from $5k to ~$7.8k). Flag gross inflation and non-positive."""
    _raw = port.get("starting_balance", base)
    sb = float(_raw) if _raw is not None else base   # don't coerce 0.0 -> base (it must flag nonpositive)
    if sb <= 0:
        return [Violation("starting_balance_nonpositive", "critical", f"starting_balance ${sb}",
                          "PnL% denominator is zero/negative")]
    if sb > base * 1.5:
        return [Violation("starting_balance_inflated", "warning",
                          f"starting_balance ${sb:.0f} >> baseline ${base:.0f} — inflated baseline hides losses",
                          "a topup raised starting_balance (v30 class); PnL% understates the real loss")]
    return []


def check_carry_sleeve_reconciles(carry_state: Optional[dict], *, tol: float = 1.0) -> List[Violation]:
    """Funding-carry sleeve: realized_pnl must ~equal the sum of closed-episode pnl in history."""
    if not carry_state:
        return []
    realized = float(carry_state.get("realized_pnl", 0) or 0)
    hist = sum(float(h.get("pnl_usd", h.get("pnl", 0)) or 0) for h in (carry_state.get("history") or []))
    if abs(realized - hist) > tol:
        return [Violation("carry_pnl_unreconciled", "warning",
                          f"carry realized_pnl ${realized:.2f} != closed-episode sum ${hist:.2f} "
                          f"(gap ${realized - hist:.2f})",
                          "carry sleeve credits realized_pnl without a matching history episode (or vice versa)")]
    return []


def check_finite_components(port: dict) -> List[Violation]:
    """No NaN/inf in the money fields — a non-finite value silently corrupts equity + every gate."""
    for k in ("cash", "btc_qty", "futures_pnl"):
        v = port.get(k)
        if v is None:
            continue
        try:
            if not math.isfinite(float(v)):
                return [Violation("nonfinite_component", "critical", f"{k}={v!r} is not finite (NaN/inf)",
                                  "a NaN/inf entered the portfolio — corrupts equity + all downstream math")]
        except (TypeError, ValueError):
            return [Violation("nonfinite_component", "critical", f"{k}={v!r} is not numeric", "")]
    return []


def run_checks(port: dict, trades: list, prev_snapshot: Optional[dict], *, cur_price: float,
               carry_state: Optional[dict] = None) -> List[Violation]:
    """Run every invariant. Returns the combined list of violations (empty == healthy)."""
    fp = port.get("futures_position") or {}
    position_open = bool(fp.get("direction") and float(fp.get("notional", 0) or 0) > 0)
    out: List[Violation] = []
    out += check_finite_components(port)
    out += check_cash_nonnegative(port)
    out += check_btc_qty_nonnegative(port)
    out += check_alt_positions_sane(port)
    out += check_starting_balance_integrity(port)
    out += check_futures_parity(trades, position_open=position_open)
    out += check_futures_pnl_ledger(port, trades)
    out += check_carry_sleeve_reconciles(carry_state)
    out += check_equity_conservation(prev_snapshot, port, trades, cur_price=cur_price)
    return out
