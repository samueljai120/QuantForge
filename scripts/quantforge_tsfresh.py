#!/usr/bin/env python3
"""QuantForge research-stage tsfresh feature builder.

This module is intentionally separate from live execution. It can be used to
derive richer historical features for research/training experiments without
changing the paper-trading path.

Usage:
    python3 quantforge_tsfresh.py status
    python3 quantforge_tsfresh.py build
    python3 quantforge_tsfresh.py build BTC-USDT ETH-USDT
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_features import load_ohlcv, discover_symbols_from_history

RESEARCH_DIR = os.path.join(cfg.data, "quantforge", "research")
TSFRESH_DIR = os.path.join(RESEARCH_DIR, "tsfresh")
TSFRESH_FILE = os.path.join(TSFRESH_DIR, "tsfresh_features.parquet")
TSFRESH_META_FILE = os.path.join(TSFRESH_DIR, "tsfresh_meta.json")
os.makedirs(TSFRESH_DIR, exist_ok=True)


def _require_production_runtime(script_name):
    if hasattr(cfg, "assert_production_runtime"):
        cfg.assert_production_runtime(script_name)
        return
    if hasattr(cfg, "require_production_runtime"):
        cfg.require_production_runtime(script_name)


_require_production_runtime("quantforge_tsfresh.py")


def tsfresh_available():
    try:
        import tsfresh  # noqa: F401
        return True
    except Exception:
        return False


def _build_long_frame(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLCV candles to a tsfresh long-format frame."""
    frame = df.copy().sort_values("ts").reset_index(drop=True)
    frame["range"] = frame["high"] - frame["low"]
    frame["body"] = (frame["close"] - frame["open"]).abs()
    frame["ret_1h"] = frame["close"].pct_change(1)
    frame["ret_4h"] = frame["close"].pct_change(4)
    frame["turnover_ratio"] = frame["turnover"] / frame["turnover"].rolling(20).mean().replace(0, np.nan)
    frame["vol_ratio"] = frame["volume"] / frame["volume"].rolling(20).mean().replace(0, np.nan)

    rows = []
    kinds = [
        "close",
        "volume",
        "turnover",
        "range",
        "body",
        "ret_1h",
        "ret_4h",
        "turnover_ratio",
        "vol_ratio",
    ]
    for kind in kinds:
        series = frame[["ts", kind]].dropna()
        if series.empty:
            continue
        tmp = series.copy()
        tmp["symbol"] = symbol
        tmp["kind"] = kind
        tmp["value"] = tmp[kind].astype(float)
        rows.append(tmp[["symbol", "ts", "kind", "value"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "kind", "value"])
    return pd.concat(rows, ignore_index=True)


def build_tsfresh_features(symbols=None):
    """Build research-stage tsfresh features from historical candles."""
    if symbols is None:
        symbols = discover_symbols_from_history()
    if not symbols:
        return pd.DataFrame(), {"status": "no_symbols"}

    long_frames = []
    missing = []
    for symbol in symbols:
        df = load_ohlcv(symbol, "1h")
        if df is None or len(df) < 200:
            missing.append(symbol)
            continue
        long_frame = _build_long_frame(symbol, df)
        if not long_frame.empty:
            long_frames.append(long_frame)

    if not long_frames:
        return pd.DataFrame(), {
            "status": "no_data",
            "symbols": symbols,
            "missing": missing,
        }

    long_df = pd.concat(long_frames, ignore_index=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "missing": missing,
        "tsfresh_available": tsfresh_available(),
        "rows_long": int(len(long_df)),
    }

    if not tsfresh_available():
        # Research scaffold only: provide a safe fallback summary so the path exists
        summary = (
            long_df.groupby("symbol")
            .agg(
                ts_count=("ts", "count"),
                value_mean=("value", "mean"),
                value_std=("value", "std"),
                value_min=("value", "min"),
                value_max=("value", "max"),
            )
            .reset_index()
        )
        meta["status"] = "tsfresh_not_installed"
        return summary, meta

    from tsfresh.feature_extraction import extract_features, MinimalFCParameters

    features = extract_features(
        long_df,
        column_id="symbol",
        column_sort="ts",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0,
        disable_progressbar=True,
    )
    features = features.reset_index().rename(columns={"index": "symbol"})
    meta["status"] = "tsfresh_built"
    meta["feature_columns"] = [c for c in features.columns if c != "symbol"]
    return features, meta


def cmd_build(symbols=None):
    features, meta = build_tsfresh_features(symbols)
    with open(TSFRESH_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    if features is None or features.empty:
        print("No tsfresh research features built.")
        print(f"Metadata saved to {TSFRESH_META_FILE}")
        return
    features.to_parquet(TSFRESH_FILE, index=False)
    print(f"Saved tsfresh research features to {TSFRESH_FILE}")
    print(f"Rows: {len(features):,}  Cols: {len(features.columns)}")
    print(f"Metadata: {TSFRESH_META_FILE}")


def cmd_status():
    if os.path.exists(TSFRESH_META_FILE):
        with open(TSFRESH_META_FILE) as f:
            meta = json.load(f)
        print(json.dumps(meta, indent=2))
    else:
        print("No tsfresh research metadata yet. Run: python3 quantforge_tsfresh.py build")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "build":
        cmd_build(sys.argv[2:] if len(sys.argv) > 2 else None)
    elif cmd == "status":
        cmd_status()
    else:
        print(__doc__)
        sys.exit(1)
