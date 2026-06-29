#!/usr/bin/env python3
"""QuantForge - data/features rebuild gap report.

Produces a lightweight artifact describing which rebuild-program feature
families are already represented in the current feature store and which are
still missing. This is intentionally cheaper than a full retrain/report pass.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_rebuild_blueprint import build_rebuild_program

BASE_DIR = os.path.join(cfg.data, "quantforge")
FEATURES_FILE = os.path.join(BASE_DIR, "features", "features_all.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "feature-gap-report.json")

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


FAMILY_HINTS = {
    "execution_quality": ["spread", "depth", "impact", "book", "staleness"],
    "crowding_and_stress": ["funding", "open_interest", "basis", "liquidation", "long_short"],
    "cross_sectional_relative_strength": ["btc_beta", "eth_beta", "market_breadth", "ret_24h", "vol_ratio"],
    "regime_and_state": ["entropy", "breadth", "volatility", "atr", "adx", "stress", "regime"],
    "setup_specific_context": ["setup_", "fakeout", "squeeze", "fomo", "upper_wick", "bb_pct_b"],
    "portfolio_and_correlation": ["correlation", "factor", "concentration", "drawdown", "beta"],
}


def _read_columns() -> list[str]:
    if not os.path.exists(FEATURES_FILE):
        return []
    if pq is not None:
        try:
            return list(pq.ParquetFile(FEATURES_FILE).schema.names)
        except Exception:
            pass
    try:
        import pandas as pd

        return list(pd.read_parquet(FEATURES_FILE).columns)
    except Exception:
        return []


def _score_family(columns: list[str], hints: list[str]) -> dict:
    matches = []
    lower_cols = [c.lower() for c in columns]
    for col, lower in zip(columns, lower_cols):
        if any(h in lower for h in hints):
            matches.append(col)
    status = "present" if matches else "missing"
    return {"status": status, "matched_columns": matches[:20], "match_count": len(matches)}


def build_report() -> dict:
    columns = _read_columns()
    program = build_rebuild_program()
    families = []
    present = 0
    for family in program["feature_families"]:
        name = family["name"]
        scored = _score_family(columns, FAMILY_HINTS.get(name, []))
        if scored["status"] == "present":
            present += 1
        families.append(
            {
                "name": name,
                "goal": family["goal"],
                "status": scored["status"],
                "match_count": scored["match_count"],
                "matched_columns": scored["matched_columns"],
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if columns else "missing_features",
        "features_file": FEATURES_FILE,
        "column_count": len(columns),
        "families_present": present,
        "families_total": len(program["feature_families"]),
        "families": families,
    }


def main():
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge feature gap report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
