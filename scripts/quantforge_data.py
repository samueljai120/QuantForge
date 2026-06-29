#!/usr/bin/env python3
"""QuantForge — Historical Data Fetcher

Fetches multi-pair, multi-timeframe OHLCV data from KuCoin public API.
Stores as CSV in ~/quantforge/data/quantforge/historical/<PAIR>_<TF>.csv

Usage:
    python3 quantforge_data.py fetch          # Fetch all pairs + timeframes
    python3 quantforge_data.py fetch BTC-USDT # Fetch single pair
    python3 quantforge_data.py status         # Show what data we have
"""

import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HISTORICAL_DIR = os.path.join(cfg.data, "quantforge", "historical")
os.makedirs(HISTORICAL_DIR, exist_ok=True)

KUCOIN_BASE = "https://api.kucoin.com"
KUCOIN_FUTURES_BASE = "https://api-futures.kucoin.com"

STABLECOINS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD",
    "FRAX", "LUSD", "SUSD", "FDUSD", "PYUSD", "UST",
}
DEFAULT_UNIVERSE_TOP_N = 100
MIN_UNIVERSE_VOLUME_USDT = 1_000_000

# Timeframes to fetch: (kucoin_type, seconds_per_candle, label)
TIMEFRAMES = [
    ("1hour",  3600,   "1h"),
    ("4hour",  14400,  "4h"),
    ("1day",   86400,  "1d"),
]
FUTURES_GRANULARITY = {
    "1hour": 60,
    "4hour": 240,
    "1day": 1440,
}

# How far back to fetch (seconds)
LOOKBACK_SECONDS = {
    "1h": 4 * 365 * 24 * 3600,    # 4 years hourly — covers 2022 crash, bear, recovery, 2 bull runs
    "4h": 3 * 365 * 24 * 3600,    # 3 years 4h
    "1d": 5 * 365 * 24 * 3600,    # 5 years daily
}

MAX_CANDLES_PER_REQUEST = 1500
MAX_FUTURES_CANDLES_PER_REQUEST = 200
RATE_LIMIT_SLEEP = 0.35  # KuCoin public API: ~3 req/sec safe

# CSV columns: timestamp_unix, open, high, low, close, volume, turnover
CSV_COLS = ["ts", "open", "high", "low", "close", "volume", "turnover"]
SPOT_UNSUPPORTED_SYMBOLS = set()


# ---------------------------------------------------------------------------
# KuCoin helpers
# ---------------------------------------------------------------------------

def _kucoin_get(path, params=None, retries=4):
    url = KUCOIN_BASE + path
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            body = r.json()
            if body.get("code") == "200000":
                return body["data"]
            raise ValueError(f"API error: {body.get('msg', body)}")
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} after {wait}s: {e}")
            time.sleep(wait)
    return None


def _kucoin_futures_get(path, params=None, retries=4):
    url = KUCOIN_FUTURES_BASE + path
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            body = r.json()
            if body.get("code") == "200000":
                return body["data"]
            raise ValueError(f"Futures API error: {body.get('msg', body)}")
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    futures retry {attempt+1}/{retries} after {wait}s: {e}")
            time.sleep(wait)
    return None


def discover_pairs(top_n=DEFAULT_UNIVERSE_TOP_N, min_volume_usdt=MIN_UNIVERSE_VOLUME_USDT):
    """Discover a broad liquid USDT universe from spot + futures leaders."""
    data = _kucoin_get("/api/v1/market/allTickers")
    tickers = data.get("ticker", []) if isinstance(data, dict) else []
    pair_scores = {}
    for t in tickers:
        symbol = t.get("symbol", "")
        if not symbol.endswith("-USDT"):
            continue
        base = symbol[:-5]
        if base in STABLECOINS:
            continue
        try:
            vol_usdt = float(t.get("volValue", 0) or 0)
            last = float(t.get("last", 0) or 0)
        except (TypeError, ValueError):
            continue
        if vol_usdt < min_volume_usdt or last <= 0:
            continue
        pair_scores[symbol] = max(pair_scores.get(symbol, 0.0), vol_usdt)

    try:
        futures = _kucoin_futures_get("/api/v1/contracts/active") or []
    except Exception as e:
        print(f"[WARN] futures universe discovery failed: {e}")
        futures = []

    for c in futures:
        fsym = c.get("symbol", "")
        if not fsym.endswith("USDTM") or c.get("status") != "Open":
            continue
        base = c.get("baseCurrency", fsym.replace("USDTM", ""))
        if base == "XBT":
            base = "BTC"
        if base[:1].isdigit() or len(base) > 8 or base in STABLECOINS:
            continue
        symbol = f"{base}-USDT"
        try:
            vol_usdt = float(c.get("turnoverOf24h", 0) or 0)
            price = float(c.get("lastTradePrice", c.get("markPrice", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if vol_usdt < min_volume_usdt or price <= 0:
            continue
        pair_scores[symbol] = max(pair_scores.get(symbol, 0.0), vol_usdt)

    pairs = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in pairs[:top_n]]


def to_futures_symbol(symbol):
    """Convert spot-style BTC-USDT to KuCoin futures symbol XBTUSDTM."""
    if symbol == "BTC-USDT":
        return "XBTUSDTM"
    return symbol.replace("-USDT", "USDTM")


def fetch_candles_range(symbol, kline_type, start_ts, end_ts):
    """Fetch candles for a time range. KuCoin returns up to 1500 per call."""
    if symbol in SPOT_UNSUPPORTED_SYMBOLS:
        raise ValueError("Unsupported trading pair.")
    data = _kucoin_get("/api/v1/market/candles", {
        "symbol": symbol,
        "type": kline_type,
        "startAt": int(start_ts),
        "endAt": int(end_ts),
    })
    if not data:
        return []
    # KuCoin returns newest-first — reverse to oldest-first
    data.reverse()
    # Each candle: [time, open, close, high, low, volume, turnover]
    # Note: KuCoin order is [ts, open, CLOSE, HIGH, LOW, volume, turnover]
    # Reorder to standard OHLCV: [ts, open, high, low, close, volume, turnover]
    result = []
    for c in data:
        result.append({
            "ts":       int(c[0]),
            "open":     float(c[1]),
            "high":     float(c[3]),
            "low":      float(c[4]),
            "close":    float(c[2]),
            "volume":   float(c[5]),
            "turnover": float(c[6]),
        })
    return result


def fetch_futures_candles_range(symbol, kline_type, start_ts, end_ts):
    """Fetch futures candles for symbols not available on spot."""
    granularity = FUTURES_GRANULARITY[kline_type]
    fsym = to_futures_symbol(symbol)
    data = _kucoin_futures_get("/api/v1/kline/query", {
        "symbol": fsym,
        "granularity": granularity,
        "from": int(start_ts * 1000),
        "to": int(end_ts * 1000),
    })
    if not data:
        return []
    data = sorted(data, key=lambda row: int(row[0]))
    result = []
    for c in data:
        result.append({
            "ts": int(c[0]) // 1000,
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "turnover": float(c[5]),
        })
    return result


def fetch_full_history(symbol, kline_type, tf_label, seconds_per_candle, lookback_secs):
    """Fetch full history in paginated chunks."""
    end_ts = int(time.time())
    start_ts = end_ts - lookback_secs
    chunk_span = MAX_CANDLES_PER_REQUEST * seconds_per_candle

    all_candles = []
    chunk_start = start_ts
    request_count = 0

    print(f"  Fetching {symbol} {tf_label}...")

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + chunk_span, end_ts)

        try:
            candles = fetch_candles_range(symbol, kline_type, chunk_start, chunk_end)
        except Exception as e:
            if "Unsupported trading pair" in str(e):
                SPOT_UNSUPPORTED_SYMBOLS.add(symbol)
                chunk_end = min(chunk_start + MAX_FUTURES_CANDLES_PER_REQUEST * seconds_per_candle, end_ts)
                try:
                    candles = fetch_futures_candles_range(symbol, kline_type, chunk_start, chunk_end)
                except Exception as futures_e:
                    print(f"    ERROR fetching futures chunk: {futures_e}")
                    break
            else:
                print(f"    ERROR fetching chunk: {e}")
                break

        if not candles:
            # Newer listings often have empty early windows. Keep moving
            # forward until we reach the pair's actual listing period.
            chunk_start = chunk_end + seconds_per_candle
            continue

        all_candles.extend(candles)
        request_count += 1

        # Advance past the last fetched candle
        last_ts = candles[-1]["ts"]
        chunk_start = last_ts + seconds_per_candle

        time.sleep(RATE_LIMIT_SLEEP)

        if request_count % 10 == 0:
            print(f"    {request_count} requests, {len(all_candles)} candles so far...")

    # Deduplicate by timestamp (in case of overlaps)
    seen = {}
    for c in all_candles:
        seen[c["ts"]] = c
    all_candles = sorted(seen.values(), key=lambda x: x["ts"])

    print(f"    Done: {len(all_candles)} candles ({request_count} requests)")
    return all_candles


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def csv_path(symbol, tf_label):
    safe = symbol.replace("-", "_")
    return os.path.join(HISTORICAL_DIR, f"{safe}_{tf_label}.csv")


def load_existing(symbol, tf_label):
    """Load existing CSV, return dict of ts -> candle."""
    path = csv_path(symbol, tf_label)
    if not os.path.exists(path):
        return {}
    existing = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[int(row["ts"])] = {k: float(v) if k != "ts" else int(v) for k, v in row.items()}
    return existing


def save_candles(symbol, tf_label, candles):
    """Save/update CSV file."""
    path = csv_path(symbol, tf_label)
    # Merge with existing
    existing = load_existing(symbol, tf_label)
    for c in candles:
        existing[c["ts"]] = c
    # Sort and write
    rows = sorted(existing.values(), key=lambda x: x["ts"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def get_status():
    """Show what data files exist and how many candles each has."""
    files = sorted(os.listdir(HISTORICAL_DIR)) if os.path.exists(HISTORICAL_DIR) else []
    if not files:
        print("  No historical data yet.")
        return
    for fname in files:
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(HISTORICAL_DIR, fname)
        with open(path) as f:
            rows = sum(1 for _ in f) - 1  # minus header
        # Get date range
        with open(path) as f:
            reader = csv.DictReader(f)
            rows_data = list(reader)
        if rows_data:
            first = datetime.utcfromtimestamp(int(rows_data[0]["ts"])).strftime("%Y-%m-%d")
            last = datetime.utcfromtimestamp(int(rows_data[-1]["ts"])).strftime("%Y-%m-%d")
            size_kb = os.path.getsize(path) // 1024
            print(f"  {fname:<30} {rows:>6} candles  {first} → {last}  ({size_kb}KB)")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch(pairs=None):
    if pairs is None:
        pairs = discover_pairs()

    print(f"Fetching historical data for {len(pairs)} pairs × {len(TIMEFRAMES)} timeframes")
    print(f"Output: {HISTORICAL_DIR}")
    print()

    total_candles = 0
    for symbol in pairs:
        for kline_type, secs, tf_label in TIMEFRAMES:
            lookback = LOOKBACK_SECONDS[tf_label]
            try:
                candles = fetch_full_history(symbol, kline_type, tf_label, secs, lookback)
                if candles:
                    count = save_candles(symbol, tf_label, candles)
                    total_candles += count
            except Exception as e:
                print(f"  ERROR {symbol} {tf_label}: {e}")
        print()

    print(f"Total candles stored: {total_candles:,}")


def cmd_status():
    print("Historical data status:")
    print()
    get_status()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "fetch":
        pairs = sys.argv[2:] if len(sys.argv) > 2 else None
        cmd_fetch(pairs)
    elif cmd == "status":
        cmd_status()
    else:
        print(__doc__)
        sys.exit(1)
