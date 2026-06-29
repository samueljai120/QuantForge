#!/usr/bin/env python3
"""Shared QuantForge equity helpers.

These helpers intentionally mirror the live agent's ledger semantics so
reporting, gating, and self-heal surfaces do not drift onto their own
incompatible balance formulas.
"""

from __future__ import annotations


def _num(value) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def compute_spot_equity(port: dict, price: float) -> float:
    """Cash plus BTC mark-to-market only."""

    px = _num(price)
    return _num(port.get("cash")) + _num(port.get("btc_qty")) * px


def compute_true_equity(port: dict, price: float) -> float:
    """Net liquidation value for the QuantForge agent ledger.

    Matches the v29 live-agent rules:
      - includes parked margin as account equity
      - includes direction-signed unrealized PnL
      - does not multiply leverage twice
      - deliberately excludes external ledgers such as funding-arb sleeves
    """

    px = _num(price)
    btc_val = _num(port.get("btc_qty")) * px
    alt_val = sum(
        _num(pos.get("qty")) * px
        for pos in (port.get("alt_positions") or {}).values()
        if isinstance(pos, dict)
    )

    futures = port.get("futures_position") or {}
    futures_margin = _num(futures.get("margin"))
    futures_upnl = _directional_upnl(futures, px)

    prehedge = port.get("prehedge") or {}
    prehedge_margin = _num(prehedge.get("margin")) if prehedge.get("open") else 0.0
    prehedge_upnl = _directional_upnl(prehedge, px) if prehedge.get("open") else 0.0

    liq_dip = port.get("liq_dip_position") or {}
    liq_dip_margin = _num(liq_dip.get("margin"))
    liq_dip_upnl = _directional_upnl(liq_dip, px)

    return (
        _num(port.get("cash"))
        + btc_val
        + alt_val
        + futures_margin
        + futures_upnl
        + prehedge_margin
        + prehedge_upnl
        + liq_dip_margin
        + liq_dip_upnl
    )


def compute_drawdown_from_peak(port: dict, price: float, *, peak_key: str = "peak_equity") -> float:
    """Return drawdown ratio from the stored peak equity."""

    peak = _num(port.get(peak_key))
    if peak <= 0:
        return 0.0
    equity = compute_true_equity(port, price)
    return max(0.0, (peak - equity) / peak)


def _directional_upnl(position: dict, price: float) -> float:
    direction = position.get("direction")
    entry = _num(position.get("entry_price"))
    if not direction or entry <= 0:
        return 0.0
    notional = _num(position.get("notional", position.get("margin")))
    pct_change = (price - entry) / entry
    if str(direction).upper() == "SHORT":
        pct_change = -pct_change
    return pct_change * notional
