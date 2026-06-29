#!/usr/bin/env python3
"""QuantForge derivatives-state collector.

Collects a lightweight majors-first futures state snapshot from KuCoin Futures
and persists it as a parquet artifact plus a compact crowding report.
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
from quantforge_data import _kucoin_futures_get
from quantforge_rebuild_blueprint import build_rebuild_program

BASE_DIR = os.path.join(cfg.data, "quantforge")
DERIVATIVES_DIR = os.path.join(BASE_DIR, "derivatives")
PARQUET_FILE = os.path.join(DERIVATIVES_DIR, "derivatives_state_latest.parquet")
HISTORY_FILE = os.path.join(DERIVATIVES_DIR, "derivatives_state_history.jsonl")
CROWDING_REPORT_FILE = os.path.join(DERIVATIVES_DIR, "crowding-report.json")


def ensure_dirs() -> None:
    os.makedirs(DERIVATIVES_DIR, exist_ok=True)


def _float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_spot_symbol(contract: dict) -> str:
    base = str(contract.get("baseCurrency") or contract.get("symbol") or "")
    if base.endswith("USDTM"):
        base = base.replace("USDTM", "")
    if base == "XBT":
        base = "BTC"
    return f"{base}-USDT"


def _extract_row(contract: dict, allowed_symbols: set[str], now_ts: int, now_iso: str) -> dict | None:
    if str(contract.get("status") or "").lower() != "open":
        return None
    futures_symbol = str(contract.get("symbol") or "")
    if not futures_symbol.endswith("USDTM"):
        return None
    symbol = _to_spot_symbol(contract)
    if symbol not in allowed_symbols:
        return None

    mark_price = _float(contract.get("markPrice"))
    index_price = _float(contract.get("indexPrice"))
    last_price = _float(contract.get("lastTradePrice"), mark_price)
    funding_rate = _float(
        contract.get("fundingFeeRate")
        or contract.get("fundingRate")
        or contract.get("predictedFundingFeeRate")
    )
    open_interest = _float(contract.get("openInterest"))
    long_short_ratio = _float(contract.get("longShortRatio"))
    turnover_24h = _float(contract.get("turnoverOf24h"))
    volume_24h = _float(contract.get("volumeOf24h"))
    buy_limit = _float(contract.get("buyLimit"))
    sell_limit = _float(contract.get("sellLimit"))
    max_leverage = _float(contract.get("maxLeverage"))

    basis_bps = 0.0
    if index_price > 0 and mark_price > 0:
        basis_bps = ((mark_price - index_price) / index_price) * 10000.0

    liquidation_long_usd = max(0.0, sell_limit) * max(mark_price, last_price)
    liquidation_short_usd = max(0.0, buy_limit) * max(mark_price, last_price)
    crowding_score = abs(funding_rate) * 10000.0 + abs(basis_bps) + (open_interest / 1_000_000.0)

    return {
        "timestamp": now_ts,
        "timestamp_iso": now_iso,
        "symbol": symbol,
        "futures_symbol": futures_symbol,
        "mark_price": mark_price,
        "index_price": index_price,
        "last_trade_price": last_price,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "basis_bps": basis_bps,
        "long_short_ratio": long_short_ratio,
        "liquidation_long_usd": liquidation_long_usd,
        "liquidation_short_usd": liquidation_short_usd,
        "turnover_24h": turnover_24h,
        "volume_24h": volume_24h,
        "buy_limit": buy_limit,
        "sell_limit": sell_limit,
        "max_leverage": max_leverage,
        "crowding_score": crowding_score,
        "source": "kucoin_futures_contracts_active",
    }


def _build_crowding_report(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: float(row.get("crowding_score", 0.0) or 0.0), reverse=True)
    top = []
    for row in ordered[:5]:
        top.append(
            {
                "symbol": row["symbol"],
                "crowding_score": round(float(row.get("crowding_score", 0.0) or 0.0), 4),
                "funding_rate": round(float(row.get("funding_rate", 0.0) or 0.0), 8),
                "basis_bps": round(float(row.get("basis_bps", 0.0) or 0.0), 4),
                "open_interest": round(float(row.get("open_interest", 0.0) or 0.0), 4),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if rows else "missing",
        "row_count": len(rows),
        "source": "kucoin_futures_contracts_active",
        "top_crowded_symbols": top,
    }


def main() -> None:
    ensure_dirs()
    program = build_rebuild_program()
    majors = program.get("target_universe", {}).get("primary") or []
    allowed_symbols = {f"{str(sym)}-USDT" for sym in majors}
    now_ts = int(time.time())
    now_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()

    try:
        contracts = _kucoin_futures_get("/api/v1/contracts/active") or []
    except Exception as exc:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "fetch_failed",
            "error": str(exc),
            "row_count": 0,
            "source": "kucoin_futures_contracts_active",
        }
        with open(CROWDING_REPORT_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        print("QuantForge derivatives collector")
        print(f"Status: {payload['status']}")
        print(f"Saved:  {CROWDING_REPORT_FILE}")
        return

    rows = []
    for contract in contracts:
        row = _extract_row(contract, allowed_symbols, now_ts, now_iso)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["symbol", "timestamp"], inplace=True)
        df.to_parquet(PARQUET_FILE, index=False)
        with open(HISTORY_FILE, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    crowding = _build_crowding_report(rows)
    with open(CROWDING_REPORT_FILE, "w") as f:
        json.dump(crowding, f, indent=2)

    print("QuantForge derivatives collector")
    print(f"Status: {'ready' if rows else 'missing'}")
    print(f"Rows:   {len(rows)}")
    print(f"Saved:  {PARQUET_FILE}")


if __name__ == "__main__":
    main()
