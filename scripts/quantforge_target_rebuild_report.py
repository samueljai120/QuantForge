#!/usr/bin/env python3
"""QuantForge - labels/targets rebuild report.

Builds a concrete artifact for the research-hold labels/targets lane so the research core
can decide whether the rebuild has enough viable support before another paper
trial is allowed.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_target_profiles import apply_research_hold_target_profile

BASE_DIR = os.path.join(cfg.data, "quantforge")
FEATURES_FILE = os.path.join(BASE_DIR, "features", "features_all.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "target-rebuild-report.json")
PROFILE_HELPER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantforge_target_profiles.py")
REQUIRED_COLUMNS = [
    "target_4h",
    "target_4h_short",
    "target_4h_trend_long",
    "target_4h_breakout_long",
    "target_4h_rebound_long",
    "target_4h_trend_short",
    "target_4h_exhaustion_short",
    "fwd_ret_4h",
    "setup_trend_long_score",
    "setup_breakout_long_score",
    "setup_rebound_long_score",
    "setup_trend_short_score",
    "setup_exhaustion_short_score",
    "fakeout_risk",
    "squeeze_risk",
    "turnover_z_48h",
    "adx",
    "upper_wick_pct",
    "fomo_score",
    "bb_pct_b",
]


def _read_existing_report() -> dict:
    try:
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_report_fresh() -> bool:
    if not os.path.exists(OUTPUT_FILE) or not os.path.exists(FEATURES_FILE):
        return False
    report_mtime = os.path.getmtime(OUTPUT_FILE)
    if report_mtime < os.path.getmtime(FEATURES_FILE):
        return False
    if os.path.exists(PROFILE_HELPER_FILE) and report_mtime < os.path.getmtime(PROFILE_HELPER_FILE):
        return False
    return True


def _safe_rate(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _slice_stats(df: pd.DataFrame, target_col: str, setup_col: str, threshold: float) -> dict:
    if setup_col not in df.columns:
        return {
            "setup_col": setup_col,
            "available": False,
            "threshold": threshold,
            "support_rows": 0,
            "positive_rows": 0,
            "positive_rate_within_support": 0.0,
        }
    support_mask = df[setup_col].fillna(0.0) >= threshold
    support_rows = int(support_mask.sum())
    positive_rows = int(df.loc[support_mask, target_col].fillna(0.0).sum())
    return {
        "setup_col": setup_col,
        "available": True,
        "threshold": threshold,
        "support_rows": support_rows,
        "positive_rows": positive_rows,
        "positive_rate_within_support": round(_safe_rate(positive_rows, support_rows), 6),
    }


def build_report() -> dict:
    if not os.path.exists(FEATURES_FILE):
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "missing_features",
            "features_file": FEATURES_FILE,
        }

    df = pd.read_parquet(FEATURES_FILE)
    available_columns = set(df.columns)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in available_columns]
    rebuilt_df, profile = apply_research_hold_target_profile(df, horizon=4)
    long_target = profile["long_target_col"]
    short_target = profile["short_target_col"]
    long_threshold = float(profile["rules"].get("long_min_setup_score", 0.62))
    short_threshold = float(profile["rules"].get("short_min_setup_score", 0.60))
    long_total = int(profile["support_counts"]["long_total"])
    long_positive = int(profile["support_counts"]["long_positive"])
    short_positive = int(profile["support_counts"]["short_positive"])

    setup_breakdown = {
        "long": [
            _slice_stats(rebuilt_df, long_target, "setup_trend_long_score", long_threshold),
            _slice_stats(rebuilt_df, long_target, "setup_breakout_long_score", long_threshold),
            _slice_stats(rebuilt_df, long_target, "setup_rebound_long_score", long_threshold),
        ],
        "short": [
            _slice_stats(rebuilt_df, short_target, "setup_trend_short_score", short_threshold),
            _slice_stats(rebuilt_df, short_target, "setup_exhaustion_short_score", short_threshold),
        ],
    }

    readiness = profile.get("viability") or {}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": readiness.get("overall_status", "unknown"),
        "features_file": FEATURES_FILE,
        "row_count": int(len(rebuilt_df)),
        "selected_columns": [col for col in REQUIRED_COLUMNS if col in available_columns],
        "missing_columns": missing_columns,
        "profile": profile,
        "summary": {
            "base_long_positive_rate": profile["class_balance"]["base_long_positive_rate"],
            "rebuild_long_positive_rate": profile["class_balance"]["research_long_positive_rate"],
            "base_short_positive_rate": profile["class_balance"]["base_short_positive_rate"],
            "rebuild_short_positive_rate": profile["class_balance"]["research_short_positive_rate"],
            "long_positive_rows": long_positive,
            "short_positive_rows": short_positive,
            "total_rows": long_total,
        },
        "support_counts": profile.get("support_counts") or {},
        "setup_target_summary": profile.get("setup_target_summary") or {},
        "setup_breakdown": setup_breakdown,
        "gates": {
            "long_ready": bool(readiness.get("long_ready")),
            "short_ready": bool(readiness.get("short_ready")),
            "overall_ready": readiness.get("overall_status") == "ready",
            "notes": [
                "Targets are intentionally post-cost and execution-haircut aware.",
                "Support counts should be evaluated before another bounded paper trial is queued.",
            ],
        },
    }
    return report


def main():
    if _is_report_fresh():
        payload = _read_existing_report()
        print("QuantForge target rebuild report")
        print(f"Status: {payload.get('status', 'cached')}")
        print(f"Saved:  {OUTPUT_FILE}")
        return
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge target rebuild report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
