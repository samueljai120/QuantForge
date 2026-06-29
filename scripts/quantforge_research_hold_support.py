#!/usr/bin/env python3
"""Shared research-hold support summaries for bounded top-alt expansion."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
FEATURES_FILE = os.path.join(BASE_DIR, "features", "features_all.parquet")

MAJOR_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BCH-USDT", "TRX-USDT"}
TOP_ALT_EXPANSION_SYMBOLS = MAJOR_SYMBOLS | {
    "ADA-USDT",
    "DOGE-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "LTC-USDT",
    "XMR-USDT",
    "TAO-USDT",
}
ALLOWED_LONG_SETUPS = ("trend_long", "breakout_long")
LONG_SETUP_DETAILS = {
    "trend_long": {"score_col": "setup_trend_long_score", "min_score": 0.67},
    "breakout_long": {"score_col": "setup_breakout_long_score", "min_score": 0.68},
}
MIN_STRONG_SYMBOL_POSITIVES = 25


def _safe_rate(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def summarize_top_alt_research_hold_support(features_file: str | None = None) -> dict:
    """Summarize whether research-hold support favors a bounded top-alt retry."""

    features_path = features_file or FEATURES_FILE
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_file": features_path,
        "allowed_long_setups": list(ALLOWED_LONG_SETUPS),
        "thresholds": {
            "strong_symbol_min_long_positive_total": MIN_STRONG_SYMBOL_POSITIVES,
        },
        "expansion_supported": False,
    }
    if not os.path.exists(features_path):
        summary["status"] = "missing_features"
        return summary

    try:
        import pandas as pd
        from quantforge_target_profiles import apply_research_hold_target_profile
    except Exception as exc:
        summary["status"] = "dependency_error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    try:
        df = pd.read_parquet(features_path)
    except Exception as exc:
        summary["status"] = "read_error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    if "symbol" not in df.columns:
        summary["status"] = "missing_symbol_column"
        return summary

    try:
        rebuilt_df, profile = apply_research_hold_target_profile(df, horizon=4)
    except Exception as exc:
        summary["status"] = "profile_error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    long_target_col = str(profile.get("long_target_col") or "")
    setup_target_cols = profile.get("setup_target_cols") or {}
    if long_target_col not in rebuilt_df.columns:
        summary["status"] = "missing_long_target"
        summary["long_target_col"] = long_target_col
        return summary

    rows = []
    for symbol in sorted(TOP_ALT_EXPANSION_SYMBOLS):
        sym_df = rebuilt_df.loc[rebuilt_df["symbol"] == symbol]
        if sym_df.empty:
            continue
        row = {
            "symbol": symbol,
            "bucket": "major" if symbol in MAJOR_SYMBOLS else "top_alt",
            "row_count": int(len(sym_df)),
            "long_positive_total": int(sym_df[long_target_col].fillna(0.0).sum()),
        }
        row["long_positive_rate"] = round(_safe_rate(row["long_positive_total"], row["row_count"]), 6)
        for setup_name, setup_meta in LONG_SETUP_DETAILS.items():
            target_col = str(setup_target_cols.get(setup_name) or "")
            score_col = setup_meta["score_col"]
            min_score = float(setup_meta["min_score"])
            row[f"{setup_name}_positive"] = (
                int(sym_df[target_col].fillna(0.0).sum()) if target_col in sym_df.columns else 0
            )
            row[f"{setup_name}_support_rows"] = (
                int(sym_df[score_col].fillna(0.0).ge(min_score).sum()) if score_col in sym_df.columns else 0
            )
        rows.append(row)

    rows.sort(key=lambda item: (-int(item["long_positive_total"]), -float(item["long_positive_rate"]), item["symbol"]))
    major_rows = [row for row in rows if row["bucket"] == "major"]
    non_major_rows = [row for row in rows if row["bucket"] != "major"]
    strong_non_major_rows = [row for row in non_major_rows if int(row["long_positive_total"]) >= MIN_STRONG_SYMBOL_POSITIVES]
    major_total = sum(int(row["long_positive_total"]) for row in major_rows)
    non_major_total = sum(int(row["long_positive_total"]) for row in non_major_rows)
    best_major = major_rows[0] if major_rows else None
    best_non_major = non_major_rows[0] if non_major_rows else None
    best_major_rate = float(best_major.get("long_positive_rate", 0.0)) if best_major else 0.0
    best_non_major_rate = float(best_non_major.get("long_positive_rate", 0.0)) if best_non_major else 0.0

    expansion_supported = bool(
        major_rows
        and non_major_rows
        and non_major_total > major_total
        and len(strong_non_major_rows) >= 2
        and best_non_major_rate >= best_major_rate
    )

    summary.update(
        {
            "status": "ready",
            "profile": str(profile.get("profile") or ""),
            "long_target_col": long_target_col,
            "major_summary": {
                "symbol_count": len(major_rows),
                "long_positive_total": int(major_total),
                "best_symbol": best_major,
            },
            "non_major_summary": {
                "symbol_count": len(non_major_rows),
                "long_positive_total": int(non_major_total),
                "best_symbol": best_non_major,
                "strong_symbols": strong_non_major_rows[:5],
            },
            "top_symbols": rows[:6],
            "top_non_major_symbols": non_major_rows[:5],
            "expansion_supported": expansion_supported,
        }
    )
    if expansion_supported:
        summary["why"] = (
            "Top-liquidity non-major research-hold long support now beats the current major set "
            f"({non_major_total} vs {major_total}) with {len(strong_non_major_rows)} strong symbols."
        )
    else:
        summary["why"] = "Top-alt research-hold support is not yet stronger than the current major-only survivors."
    return summary
