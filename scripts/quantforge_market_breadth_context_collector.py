#!/usr/bin/env python3
"""QuantForge market breadth context collector.

Builds a lightweight majors/alts breadth snapshot from the current liquid
universe and the existing 1h historical candle store.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_data import HISTORICAL_DIR, _kucoin_get
from quantforge_rebuild_blueprint import build_rebuild_program

BASE_DIR = os.path.join(cfg.data, "quantforge")
BREADTH_DIR = os.path.join(BASE_DIR, "breadth")
PARQUET_FILE = os.path.join(BREADTH_DIR, "breadth_context_latest.parquet")
HISTORY_FILE = os.path.join(BREADTH_DIR, "breadth_context_history.jsonl")
REPORT_FILE = os.path.join(BREADTH_DIR, "breadth-report.json")


def ensure_dirs() -> None:
    os.makedirs(BREADTH_DIR, exist_ok=True)


def _float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _read_recent_return(symbol: str) -> float | None:
    path = os.path.join(HISTORICAL_DIR, f"{symbol}_1h.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            return None
        prev_close = _float(rows[-2].get("close"), 0.0)
        close = _float(rows[-1].get("close"), 0.0)
        if prev_close <= 0 or close <= 0:
            return None
        return (close / prev_close) - 1.0
    except Exception:
        return None


def _fallback_return_1h(ticker: dict) -> float:
    # Use a conservative 24h-to-1h scaling fallback when local 1h candles are absent.
    return _float(ticker.get("changeRate"), 0.0) / 24.0


def _weighted_fraction_positive(rows: list[dict]) -> float:
    total = sum(max(_float(row.get("quote_volume"), 0.0), 0.0) for row in rows)
    if total <= 0:
        return 0.0
    positive = sum(max(_float(row.get("quote_volume"), 0.0), 0.0) for row in rows if _float(row.get("return_1h"), 0.0) > 0)
    return positive / total


def _volume_share(rows: list[dict], symbols: set[str]) -> float:
    total = sum(max(_float(row.get("quote_volume"), 0.0), 0.0) for row in rows)
    if total <= 0:
        return 0.0
    selected = sum(max(_float(row.get("quote_volume"), 0.0), 0.0) for row in rows if str(row.get("symbol") or "") in symbols)
    return selected / total


def _build_universe_snapshot() -> list[dict]:
    ticker_data = _kucoin_get("/api/v1/market/allTickers") or {}
    tickers = ticker_data.get("ticker", []) if isinstance(ticker_data, dict) else []
    rows = []
    for ticker in tickers:
        symbol = str(ticker.get("symbol") or "")
        if not symbol.endswith("-USDT"):
            continue
        quote_volume = _float(ticker.get("volValue"), 0.0)
        if quote_volume < 1_000_000:
            continue
        ret_1h = _read_recent_return(symbol)
        if ret_1h is None:
            ret_1h = _fallback_return_1h(ticker)
        rows.append(
            {
                "symbol": symbol,
                "quote_volume": quote_volume,
                "last_price": _float(ticker.get("last"), 0.0),
                "change_rate_24h": _float(ticker.get("changeRate"), 0.0),
                "return_1h": ret_1h,
            }
        )
    return rows


def _build_payload(rows: list[dict], primary_symbols: set[str], now_ts: int, now_iso: str) -> dict:
    majors = [row for row in rows if row["symbol"] in primary_symbols]
    alts = [row for row in rows if row["symbol"] not in primary_symbols]
    btc_row = next((row for row in rows if row["symbol"] == "BTC-USDT"), {})
    eth_row = next((row for row in rows if row["symbol"] == "ETH-USDT"), {})

    majors_breadth = (sum(1 for row in majors if _float(row.get("return_1h"), 0.0) > 0) / len(majors)) if majors else 0.0
    alts_breadth = (sum(1 for row in alts if _float(row.get("return_1h"), 0.0) > 0) / len(alts)) if alts else 0.0
    market_volume_breadth = _weighted_fraction_positive(rows)
    majors_volume_breadth = _weighted_fraction_positive(majors)
    alts_volume_breadth = _weighted_fraction_positive(alts)
    majors_volume_share = _volume_share(rows, primary_symbols)
    stablecoin_dominance_proxy = max(0.0, 1.0 - market_volume_breadth)

    return {
        "timestamp": now_ts,
        "timestamp_iso": now_iso,
        "btc_return_1h": _float(btc_row.get("return_1h"), 0.0),
        "eth_return_1h": _float(eth_row.get("return_1h"), 0.0),
        "majors_breadth": majors_breadth,
        "alts_breadth": alts_breadth,
        "stablecoin_dominance_proxy": stablecoin_dominance_proxy,
        "market_volume_breadth": market_volume_breadth,
        "majors_volume_breadth": majors_volume_breadth,
        "alts_volume_breadth": alts_volume_breadth,
        "majors_volume_share": majors_volume_share,
        "universe_size": len(rows),
        "majors_count": len(majors),
        "alts_count": len(alts),
        "source": "kucoin_spot_tickers_plus_historical_1h",
    }


def _build_report(payload: dict) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if payload.get("universe_size", 0) else "missing",
        "source": payload.get("source"),
        "universe_size": payload.get("universe_size", 0),
        "majors_count": payload.get("majors_count", 0),
        "alts_count": payload.get("alts_count", 0),
        "btc_return_1h": round(_float(payload.get("btc_return_1h"), 0.0), 6),
        "eth_return_1h": round(_float(payload.get("eth_return_1h"), 0.0), 6),
        "majors_breadth": round(_float(payload.get("majors_breadth"), 0.0), 4),
        "alts_breadth": round(_float(payload.get("alts_breadth"), 0.0), 4),
        "market_volume_breadth": round(_float(payload.get("market_volume_breadth"), 0.0), 4),
        "stablecoin_dominance_proxy": round(_float(payload.get("stablecoin_dominance_proxy"), 0.0), 4),
    }


def main() -> None:
    ensure_dirs()
    program = build_rebuild_program()
    primary_symbols = {f"{str(sym)}-USDT" for sym in (program.get("target_universe", {}).get("primary") or [])}
    now_ts = int(time.time())
    now_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()

    rows = _build_universe_snapshot()
    payload = _build_payload(rows, primary_symbols, now_ts, now_iso)
    df = pd.DataFrame([payload])
    if not df.empty:
        df.to_parquet(PARQUET_FILE, index=False)
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(payload) + "\n")

    report = _build_report(payload)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("QuantForge market breadth collector")
    print(f"Status: {report['status']}")
    print(f"Universe:{report['universe_size']}")
    print(f"Saved:  {PARQUET_FILE}")


if __name__ == "__main__":
    main()
