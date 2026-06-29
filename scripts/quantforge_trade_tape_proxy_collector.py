#!/usr/bin/env python3
"""QuantForge trade-tape proxy collector.

Approximates short-horizon aggressive flow and pressure imbalance from the live
book snapshot lane plus current ticker state, without requiring full raw tape.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_data import _kucoin_get

BASE_DIR = os.path.join(cfg.data, "quantforge")
BOOK_FILE = os.path.join(BASE_DIR, "book", "book_snapshot_latest.parquet")
MICRO_DIR = os.path.join(BASE_DIR, "microstructure")
PARQUET_FILE = os.path.join(MICRO_DIR, "trade_tape_proxy_latest.parquet")
HISTORY_FILE = os.path.join(MICRO_DIR, "trade_tape_proxy_history.jsonl")
REPORT_FILE = os.path.join(MICRO_DIR, "pressure-imbalance-report.json")


def ensure_dirs() -> None:
    os.makedirs(MICRO_DIR, exist_ok=True)


def _float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _read_book_rows() -> list[dict]:
    if not os.path.exists(BOOK_FILE):
        return []
    try:
        df = pd.read_parquet(BOOK_FILE)
        return df.to_dict("records")
    except Exception:
        return []


def _ticker_map() -> dict[str, dict]:
    data = _kucoin_get("/api/v1/market/allTickers") or {}
    tickers = data.get("ticker", []) if isinstance(data, dict) else []
    out = {}
    for ticker in tickers:
        symbol = str(ticker.get("symbol") or "")
        if symbol:
            out[symbol] = ticker
    return out


def _build_rows(now_ts: int, now_iso: str) -> list[dict]:
    rows = []
    ticker_map = _ticker_map()
    for book in _read_book_rows():
        symbol = str(book.get("symbol") or "")
        ticker = ticker_map.get(symbol, {})
        bid_notional = _float(book.get("bid_notional_5"), 0.0)
        ask_notional = _float(book.get("ask_notional_5"), 0.0)
        total_notional = bid_notional + ask_notional
        imbalance = _float(book.get("top5_depth_imbalance"), 0.0)
        change_rate_24h = _float(ticker.get("changeRate"), 0.0)
        volume_24h = _float(ticker.get("vol"), 0.0)
        quote_volume_24h = _float(ticker.get("volValue"), 0.0)
        micro_return_30s = change_rate_24h / (24.0 * 120.0)
        micro_vol_30s = (quote_volume_24h / 24.0 / 120.0) if quote_volume_24h > 0 else 0.0
        aggressive_buy_volume = max(0.0, total_notional * max(imbalance, 0.0))
        aggressive_sell_volume = max(0.0, total_notional * max(-imbalance, 0.0))
        pressure_imbalance = 0.0
        if (aggressive_buy_volume + aggressive_sell_volume) > 0:
            pressure_imbalance = (aggressive_buy_volume - aggressive_sell_volume) / (aggressive_buy_volume + aggressive_sell_volume)

        rows.append(
            {
                "timestamp": now_ts,
                "timestamp_iso": now_iso,
                "symbol": symbol,
                "aggressive_buy_volume_proxy": aggressive_buy_volume,
                "aggressive_sell_volume_proxy": aggressive_sell_volume,
                "micro_return_30s_proxy": micro_return_30s,
                "micro_vol_30s_proxy": micro_vol_30s,
                "pressure_imbalance_proxy": pressure_imbalance,
                "book_staleness_seconds": _float(book.get("book_staleness_seconds"), 0.0),
                "quoted_spread_bps": _float(book.get("quoted_spread_bps"), 0.0),
                "volume_24h": volume_24h,
                "quote_volume_24h": quote_volume_24h,
                "source": "book_plus_ticker_proxy",
            }
        )
    return rows


def _build_report(rows: list[dict]) -> dict:
    ranked = sorted(rows, key=lambda row: abs(_float(row.get("pressure_imbalance_proxy"), 0.0)), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if rows else "missing",
        "row_count": len(rows),
        "top_pressure_symbols": [
            {
                "symbol": row.get("symbol"),
                "pressure_imbalance_proxy": round(_float(row.get("pressure_imbalance_proxy"), 0.0), 4),
                "aggressive_buy_volume_proxy": round(_float(row.get("aggressive_buy_volume_proxy"), 0.0), 2),
                "aggressive_sell_volume_proxy": round(_float(row.get("aggressive_sell_volume_proxy"), 0.0), 2),
                "micro_return_30s_proxy": round(_float(row.get("micro_return_30s_proxy"), 0.0), 8),
            }
            for row in ranked[:5]
        ],
    }


def main() -> None:
    ensure_dirs()
    now_ts = int(time.time())
    now_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
    rows = _build_rows(now_ts, now_iso)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(PARQUET_FILE, index=False)
        with open(HISTORY_FILE, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    report = _build_report(rows)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("QuantForge trade-tape proxy collector")
    print(f"Status: {report['status']}")
    print(f"Rows:   {report['row_count']}")
    print(f"Saved:  {PARQUET_FILE}")


if __name__ == "__main__":
    main()
