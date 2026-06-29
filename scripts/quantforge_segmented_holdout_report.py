#!/usr/bin/env python3
"""QuantForge - segmented holdout scoring report."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_ml import (
    ROUND_TRIP_COST,
    apply_training_target_profile,
    load_model,
    load_redesign_context,
)
from quantforge_signal_ranking import signal_rank_value

BASE_DIR = os.path.join(cfg.data, "quantforge")
FEATURES_FILE = os.path.join(BASE_DIR, "features", "features_all.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "segmented-holdout-report.json")
TOP_K = 8
MAJOR_SYMBOLS = {"BTC", "ETH", "SOL", "XRP", "BCH", "TRX"}
EXECUTION_LIMITS = {"long": 3, "short": 2}

LONG_SETUP_COLUMNS = [
    ("trend_long", "setup_trend_long_score"),
    ("breakout_long", "setup_breakout_long_score"),
    ("rebound_long", "setup_rebound_long_score"),
]
SHORT_SETUP_COLUMNS = [
    ("trend_short", "setup_trend_short_score"),
    ("exhaustion_short", "setup_exhaustion_short_score"),
]


def _analysis_gate_bypass(meta: dict | None) -> bool:
    meta = meta or {}
    return not bool(meta.get("gate_pass", False))


def _safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _symbol_tier(symbol: str) -> str:
    token = str(symbol or "").split("-")[0].upper()
    return "major" if token in MAJOR_SYMBOLS else "alt"


def _regime_bucket(row: pd.Series) -> str:
    adx = float(row.get("adx") or 0.0)
    turnover = float(row.get("turnover_z_48h") or 0.0)
    fakeout = float(row.get("fakeout_risk") or 0.0)
    if fakeout >= 0.65:
        return "fragile"
    if adx >= 25 and turnover >= 0.0:
        return "trend"
    if adx < 18:
        return "chop"
    return "mixed"


def _setup_tag(row: pd.Series, side: str) -> str:
    columns = LONG_SETUP_COLUMNS if side == "long" else SHORT_SETUP_COLUMNS
    best_name = "generic_long" if side == "long" else "generic_short"
    best_score = -1.0
    for name, col in columns:
        score = float(row.get(col) or 0.0)
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def _build_holdout(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    holdout_rows = []
    for _, grp in df.groupby("symbol"):
        split = int(len(grp) * 0.80)
        holdout_rows.append(grp.iloc[split:])
    return pd.concat(holdout_rows).sort_values("ts").reset_index(drop=True)


def _execution_limit_for_side(side: str) -> int:
    return int(EXECUTION_LIMITS.get(str(side or "").lower(), 1))


def _score_side(df_holdout: pd.DataFrame, side: str) -> list[dict]:
    short = side == "short"
    xgb_model, lgb_model, feature_cols, meta = load_model(short=short)
    if xgb_model is None or meta is None:
        return []

    threshold = float(meta.get("optimal_threshold", 0.60) or 0.60)
    gate_bypassed_for_analysis = _analysis_gate_bypass(meta)
    training_profile = meta.get("training_profile") or {}
    target_profile = training_profile.get("target_profile") or {}
    target_col = str(training_profile.get("target_col") or ("target_4h_short" if short else "target_4h"))
    fwd_ret_col = str(target_profile.get("fwd_ret_col") or "fwd_ret_4h")
    if target_col not in df_holdout.columns or fwd_ret_col not in df_holdout.columns:
        return []
    required = [col for col in feature_cols if col in df_holdout.columns]
    if len(required) != len(feature_cols):
        return []

    clean = df_holdout.dropna(subset=required + [target_col, fwd_ret_col]).copy()
    if clean.empty:
        return []

    X = clean[required].fillna(0.0).values
    probs = (xgb_model.predict_proba(X)[:, 1] + lgb_model.predict_proba(X)[:, 1]) / 2
    clean["score_prob"] = probs
    clean["entry_threshold"] = threshold
    clean["execution_score"] = clean["score_prob"].astype(float)
    clean["edge_rank"] = clean.apply(
        lambda row: signal_rank_value(
            {
                "execution_score": row["execution_score"],
                "entry_threshold": row["entry_threshold"],
            }
        ),
        axis=1,
    )
    candidates = clean[clean["score_prob"] >= threshold].copy()
    if candidates.empty:
        return []
    rows = (
        candidates.sort_values(["ts", "edge_rank", "score_prob"], ascending=[True, False, False])
        .groupby("ts")
        .head(_execution_limit_for_side(side))
        .copy()
    )
    if rows.empty:
        return []
    rows["direction"] = side
    rows["setup_tag"] = rows.apply(lambda r: _setup_tag(r, side), axis=1)
    rows["regime_bucket"] = rows.apply(_regime_bucket, axis=1)
    rows["symbol_tier"] = rows["symbol"].map(_symbol_tier)
    rows["gate_pass"] = bool(meta.get("gate_pass", False))
    rows["gate_bypassed_for_analysis"] = gate_bypassed_for_analysis
    rows["execution_limit_per_ts"] = _execution_limit_for_side(side)
    rows["candidate_count_above_threshold"] = int(len(candidates))
    pnl = -rows[fwd_ret_col].astype(float) if short else rows[fwd_ret_col].astype(float)
    rows["net_edge_bps"] = (pnl - ROUND_TRIP_COST) * 10000.0
    rows["win"] = (rows["net_edge_bps"] > 0).astype(int)
    rows["target_hit"] = rows[target_col].astype(int)
    return rows.to_dict(orient="records")


def _summarize(rows: list[dict], dimension: str) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for key, grp in df.groupby(dimension):
        trades = int(len(grp))
        avg_edge = float(grp["net_edge_bps"].mean())
        out.append({
            "dimension": dimension,
            "segment": str(key),
            "trades": trades,
            "win_rate": round(_safe_rate(grp["win"].sum(), trades), 4),
            "target_hit_rate": round(_safe_rate(grp["target_hit"].sum(), trades), 4),
            "avg_net_edge_bps": round(avg_edge, 2),
            "avg_prob": round(float(grp["score_prob"].mean()), 4),
            "status": "failing" if avg_edge < 0 else "passing",
        })
    out.sort(key=lambda row: (row["avg_net_edge_bps"], -row["trades"]))
    return out


def build_report() -> dict:
    if not os.path.exists(FEATURES_FILE):
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "missing_features",
            "features_file": FEATURES_FILE,
        }
    redesign_context = load_redesign_context()
    df = pd.read_parquet(FEATURES_FILE)
    df, target_profile = apply_training_target_profile(df, redesign_context)
    holdout = _build_holdout(df)
    long_rows = _score_side(holdout, "long")
    short_rows = _score_side(holdout, "short")
    all_rows = long_rows + short_rows

    by_setup = _summarize(all_rows, "setup_tag")
    by_regime = _summarize(all_rows, "regime_bucket")
    by_symbol_tier = _summarize(all_rows, "symbol_tier")
    by_direction = _summarize(all_rows, "direction")
    combined = by_setup + by_regime + by_symbol_tier + by_direction
    failing_only = [row for row in combined if str(row.get("status")) == "failing"]
    weakest = sorted(combined, key=lambda row: (row["avg_net_edge_bps"], -row["trades"]))[:TOP_K]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if all_rows else "insufficient_holdout_trades",
        "candidate_type": redesign_context.get("candidate_type"),
        "row_count": int(len(holdout)),
        "trade_count": int(len(all_rows)),
        "target_profile": target_profile.get("profile"),
        "analysis_mode": "executed_subset_ranked",
        "analysis_gate_bypassed": bool(any(bool(row.get("gate_bypassed_for_analysis")) for row in all_rows)),
        "summary": {
            "long_trade_count": int(len(long_rows)),
            "short_trade_count": int(len(short_rows)),
            "net_edge_bps_mean": round(float(pd.DataFrame(all_rows)["net_edge_bps"].mean()), 2) if all_rows else 0.0,
        },
        "execution_limits": EXECUTION_LIMITS,
        "by_setup": by_setup,
        "by_regime": by_regime,
        "by_symbol_tier": by_symbol_tier,
        "by_direction": by_direction,
        "failing_segments": sorted(failing_only, key=lambda row: (row["avg_net_edge_bps"], -row["trades"]))[:TOP_K],
        "weakest_segments": weakest,
        "strongest_segments": sorted(combined, key=lambda row: (-row["avg_net_edge_bps"], -row["trades"]))[:TOP_K],
    }


def main() -> None:
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge segmented holdout report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
