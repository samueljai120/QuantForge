#!/usr/bin/env python3
"""QuantForge top-of-book snapshot collector.

Collects a lightweight majors-first spot order book snapshot from KuCoin and
persists it as a parquet artifact plus a compact spread/depth summary report.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_data import _kucoin_get
from quantforge_rebuild_blueprint import build_rebuild_program

BASE_DIR = os.path.join(cfg.data, "quantforge")
BOOK_DIR = os.path.join(BASE_DIR, "book")
PARQUET_FILE = os.path.join(BOOK_DIR, "book_snapshot_latest.parquet")
HISTORY_FILE = os.path.join(BOOK_DIR, "book_snapshot_history.jsonl")
SUMMARY_FILE = os.path.join(BOOK_DIR, "spread-depth-report.json")


def ensure_dirs() -> None:
    os.makedirs(BOOK_DIR, exist_ok=True)


def _float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _parse_levels(levels) -> list[tuple[float, float]]:
    rows = []
    for level in levels or []:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price = _float(level[0], 0.0)
        size = _float(level[1], 0.0)
        if price > 0 and size >= 0:
            rows.append((price, size))
    return rows


def _sum_size(levels: list[tuple[float, float]], depth: int) -> float:
    return sum(size for _, size in levels[:depth])


def _sum_notional(levels: list[tuple[float, float]], depth: int) -> float:
    return sum(price * size for price, size in levels[:depth])


def _impact_bps_for_notional(levels: list[tuple[float, float]], target_usd: float, mid_price: float) -> float:
    if not levels or target_usd <= 0 or mid_price <= 0:
        return 0.0
    remaining = float(target_usd)
    total_cost = 0.0
    total_units = 0.0
    for price, size in levels:
        level_notional = price * size
        if level_notional <= 0:
            continue
        take_notional = min(level_notional, remaining)
        take_units = take_notional / price
        total_cost += take_units * price
        total_units += take_units
        remaining -= take_notional
        if remaining <= 1e-9:
            break
    if total_units <= 0:
        return 0.0
    avg_fill_price = total_cost / total_units
    return max(0.0, ((avg_fill_price - mid_price) / mid_price) * 10000.0)


def _extract_row(symbol: str, allowed_symbols: set[str], now_ts: int, now_iso: str) -> dict | None:
    if symbol not in allowed_symbols:
        return None

    try:
        level1 = _kucoin_get("/api/v1/market/orderbook/level1", {"symbol": symbol})
        level2 = _kucoin_get("/api/v1/market/orderbook/level2_20", {"symbol": symbol})
    except Exception:
        return None

    bids = _parse_levels((level2 or {}).get("bids"))
    asks = _parse_levels((level2 or {}).get("asks"))
    best_bid = _float((level1 or {}).get("bestBid"), bids[0][0] if bids else 0.0)
    best_ask = _float((level1 or {}).get("bestAsk"), asks[0][0] if asks else 0.0)
    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return None

    bid_size_1 = _float((level1 or {}).get("bestBidSize"), bids[0][1] if bids else 0.0)
    ask_size_1 = _float((level1 or {}).get("bestAskSize"), asks[0][1] if asks else 0.0)
    bid_size_5 = _sum_size(bids, 5)
    ask_size_5 = _sum_size(asks, 5)
    bid_notional_5 = _sum_notional(bids, 5)
    ask_notional_5 = _sum_notional(asks, 5)
    mid_price = (best_bid + best_ask) / 2.0
    quoted_spread_bps = ((best_ask - best_bid) / mid_price) * 10000.0 if mid_price > 0 else 0.0
    imbalance = 0.0
    if (bid_size_5 + ask_size_5) > 0:
        imbalance = (bid_size_5 - ask_size_5) / (bid_size_5 + ask_size_5)

    book_time_raw = _float((level1 or {}).get("time"), 0.0)
    book_time_ts = int(book_time_raw // 1000) if book_time_raw > 10_000_000_000 else int(book_time_raw or now_ts)
    book_age_seconds = max(0.0, float(now_ts - book_time_ts))

    return {
        "timestamp": now_ts,
        "timestamp_iso": now_iso,
        "symbol": symbol,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "bid_size_1": bid_size_1,
        "ask_size_1": ask_size_1,
        "bid_size_5": bid_size_5,
        "ask_size_5": ask_size_5,
        "bid_notional_5": bid_notional_5,
        "ask_notional_5": ask_notional_5,
        "quoted_spread_bps": quoted_spread_bps,
        "effective_spread_proxy_bps": quoted_spread_bps,
        "top5_depth_imbalance": imbalance,
        "book_depth_proxy": math.log1p(max(bid_notional_5, 0.0) + max(ask_notional_5, 0.0)),
        "expected_impact_10k_bps": _impact_bps_for_notional(asks, 10_000.0, mid_price),
        "expected_impact_25k_bps": _impact_bps_for_notional(asks, 25_000.0, mid_price),
        "book_staleness_seconds": book_age_seconds,
        "book_staleness_flag": 1 if book_age_seconds > 30 else 0,
        "source": "kucoin_spot_orderbook",
    }


def _format_summary_row(row: dict) -> dict:
    return {
        "symbol": row["symbol"],
        "quoted_spread_bps": round(float(row.get("quoted_spread_bps", 0.0) or 0.0), 4),
        "top5_depth_notional_usd": round(float(row.get("bid_notional_5", 0.0) or 0.0) + float(row.get("ask_notional_5", 0.0) or 0.0), 2),
        "top5_depth_imbalance": round(float(row.get("top5_depth_imbalance", 0.0) or 0.0), 4),
        "expected_impact_10k_bps": round(float(row.get("expected_impact_10k_bps", 0.0) or 0.0), 4),
        "book_staleness_seconds": round(float(row.get("book_staleness_seconds", 0.0) or 0.0), 2),
    }


def _build_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "missing",
            "row_count": 0,
            "source": "kucoin_spot_orderbook",
            "top_widest_symbols": [],
            "top_thinnest_symbols": [],
        }

    widest = sorted(rows, key=lambda row: float(row.get("quoted_spread_bps", 0.0) or 0.0), reverse=True)
    thinnest = sorted(rows, key=lambda row: float(row.get("bid_notional_5", 0.0) or 0.0) + float(row.get("ask_notional_5", 0.0) or 0.0))
    avg_spread = sum(float(row.get("quoted_spread_bps", 0.0) or 0.0) for row in rows) / len(rows)
    avg_impact = sum(float(row.get("expected_impact_10k_bps", 0.0) or 0.0) for row in rows) / len(rows)
    stale_count = sum(1 for row in rows if int(row.get("book_staleness_flag", 0) or 0) > 0)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "row_count": len(rows),
        "source": "kucoin_spot_orderbook",
        "avg_quoted_spread_bps": round(avg_spread, 4),
        "avg_expected_impact_10k_bps": round(avg_impact, 4),
        "stale_symbol_count": stale_count,
        "top_widest_symbols": [_format_summary_row(row) for row in widest[:5]],
        "top_thinnest_symbols": [_format_summary_row(row) for row in thinnest[:5]],
    }


def main() -> None:
    ensure_dirs()
    program = build_rebuild_program()
    majors = program.get("target_universe", {}).get("primary") or []
    allowed_symbols = {f"{str(sym)}-USDT" for sym in majors}
    now_ts = int(time.time())
    now_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()

    rows = []
    for symbol in sorted(allowed_symbols):
        row = _extract_row(symbol, allowed_symbols, now_ts, now_iso)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["symbol", "timestamp"], inplace=True)
        df.to_parquet(PARQUET_FILE, index=False)
        with open(HISTORY_FILE, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    payload = _build_summary(rows)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("QuantForge top-of-book collector")
    print(f"Status: {payload['status']}")
    print(f"Rows:   {len(rows)}")
    print(f"Saved:  {PARQUET_FILE}")


if __name__ == "__main__":
    main()
