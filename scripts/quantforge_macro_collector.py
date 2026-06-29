#!/usr/bin/env python3
"""QuantForge Macro Collector — free TradFi macro series (2026-06-12).

Borrows the *idea* from FinceptTerminal's free-source list (FRED/Yahoo/etc.) —
none of its code — and pulls the TradFi macro-risk series the QuantForge
feature pipeline lacks. The existing breadth collector already covers
crypto-internal macro (BTC/ETH returns, dominance, breadth); this fills the
GAP: cross-asset risk appetite (equities, vol, dollar, rates, gold, oil).

Source: Yahoo Finance keyless chart API (the endpoint yfinance wraps), via
stdlib urllib — the quant-ops venv has no requests/yfinance. Daily bars,
~5y history. Stooq was rejected (serves HTML to datacenter IPs).

Series (curated for relevance + daily-or-better frequency + keyless):
  VIX   — equity vol / fear gauge       (risk-off spikes)
  SPX   — S&P 500                        (risk-on proxy)
  DXY   — US dollar index                (inverse risk / crypto headwind)
  US10Y — 10y treasury yield (^TNX)      (rates / liquidity)
  GOLD  — gold futures                   (debasement / safe haven)
  OIL   — WTI crude                      (inflation / growth)

POINT-IN-TIME SAFETY (the whole reason this file has a test):
  A daily close for date D is only KNOWN after D ends. So each series row
  carries `usable_from_ts` = 00:00 UTC of (D + 1 day). `merge_macro_features`
  joins crypto hourly bars to the most recent macro row whose usable_from_ts
  <= the crypto bar's ts (merge_asof backward). No bar ever sees a close from
  its own day or the future.

Usage:
  quantforge_macro_collector.py backfill   # 5y history -> macro_history.parquet
  quantforge_macro_collector.py update     # refresh recent bars (idempotent)
  quantforge_macro_collector.py status
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
MACRO_DIR = os.path.join(DATA_DIR, "macro")
MACRO_PATH = os.path.join(MACRO_DIR, "macro_history.parquet")

# name -> Yahoo symbol (URL-encoded ^ as %5E by the fetcher)
SERIES = {
    "vix": "^VIX",
    "spx": "^GSPC",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "gold": "GC=F",
    "oil": "CL=F",
}
DAY = 86400


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def _fetch_series(symbol, rng):
    enc = urllib.parse.quote(symbol, safe="")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
           f"?interval=1d&range={rng}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=25).read().decode())
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    close = res["indicators"]["quote"][0]["close"]
    rows = [(int(t), float(c)) for t, c in zip(ts, close) if c is not None]
    return rows


def collect(rng):
    """Fetch all series, return a daily DataFrame keyed by UTC date."""
    frames = {}
    for name, sym in SERIES.items():
        for attempt in range(3):
            try:
                rows = _fetch_series(sym, rng)
                s = pd.Series({_floor_utc_date(t): c for t, c in rows}, name=name)
                frames[name] = s
                log(f"  {name:6s} {sym:10s} {len(rows)} bars")
                break
            except Exception as e:
                if attempt == 2:
                    log(f"  {name:6s} {sym:10s} FAILED after 3 tries: {str(e)[:80]}")
                else:
                    time.sleep(2)
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date_ts"          # 00:00 UTC of the bar's date
    df = df.ffill()                     # carry last close across TradFi gaps
    df = df.reset_index()
    # close for date D is usable only from D+1 (no intraday lookahead)
    df["usable_from_ts"] = df["date_ts"].astype("int64") + DAY
    return df


def _floor_utc_date(unix_ts):
    """Floor a unix ts to 00:00 UTC of its date."""
    return (int(unix_ts) // DAY) * DAY


def save(df):
    os.makedirs(MACRO_DIR, exist_ok=True)
    df.to_parquet(MACRO_PATH, index=False)


def load():
    if os.path.exists(MACRO_PATH):
        return pd.read_parquet(MACRO_PATH)
    return pd.DataFrame()


def merge_macro_features(df, macro_path=MACRO_PATH):
    """Point-in-time left-join macro features onto an hourly feature frame.

    df must have an integer unix 'ts' column. Returns df with added columns:
      macro_<name>, macro_<name>_chg1d, macro_<name>_chg5d   (per series)
    Every macro value attached to a row at time t was usable_from <= t, so
    there is no same-day or future leakage. Rows before any macro data get NaN
    (XGB/LGB route NaN natively).
    """
    macro = (pd.read_parquet(macro_path) if isinstance(macro_path, str)
             else macro_path)
    if macro.empty:
        return df, []
    macro = macro.sort_values("usable_from_ts").reset_index(drop=True)

    names = [c for c in SERIES if c in macro.columns]
    # derived change features computed on the daily series BEFORE the join
    for n in names:
        macro[f"{n}_chg1d"] = macro[n].pct_change(1)
        macro[f"{n}_chg5d"] = macro[n].pct_change(5)

    feat_cols = []
    for n in names:
        feat_cols += [n, f"{n}_chg1d", f"{n}_chg5d"]
    macro_join = macro[["usable_from_ts"] + feat_cols].copy()
    macro_join.columns = ["usable_from_ts"] + [f"macro_{c}" for c in feat_cols]

    left = df.sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(
        left, macro_join,
        left_on="ts", right_on="usable_from_ts",
        direction="backward",
    )
    merged = merged.drop(columns=["usable_from_ts"])
    added = [f"macro_{c}" for c in feat_cols]
    return merged, added


def cmd_status():
    df = load()
    if df.empty:
        print("No macro data yet. Run: quantforge_macro_collector.py backfill")
        return
    cols = [c for c in SERIES if c in df.columns]
    last = df.iloc[-1]
    first_d = datetime.fromtimestamp(int(df["date_ts"].min()), timezone.utc).date()
    last_d = datetime.fromtimestamp(int(df["date_ts"].max()), timezone.utc).date()
    print(f"Macro history: {len(df)} daily bars, {first_d} -> {last_d}")
    print(f"Series: {cols}")
    print("Latest closes: " + ", ".join(f"{c}={last[c]:.2f}" for c in cols))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    if cmd == "status":
        cmd_status()
        return
    rng = "5y" if cmd == "backfill" else "3mo"
    log(f"macro collect ({cmd}, range={rng})...")
    fresh = collect(rng)
    if fresh.empty:
        log(" no data fetched — keeping existing parquet")
        sys.exit(1)
    if cmd == "update":
        old = load()
        if not old.empty:
            combined = pd.concat([old, fresh], ignore_index=True)
            combined = (combined.sort_values("date_ts")
                                .drop_duplicates("date_ts", keep="last")
                                .reset_index(drop=True))
            fresh = combined
    save(fresh)
    log(f"saved {len(fresh)} bars -> {MACRO_PATH}")
    cmd_status()


if __name__ == "__main__":
    main()
