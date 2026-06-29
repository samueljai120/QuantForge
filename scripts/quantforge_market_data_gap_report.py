#!/usr/bin/env python3
"""QuantForge market-data gap report.

Summarizes how much of the deeper rebuild data contract is actually represented
in the current feature store and on-disk source artifacts.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_rebuild_blueprint import data_source_specs

BASE_DIR = os.path.join(cfg.data, "quantforge")
FEATURES_FILE = os.path.join(BASE_DIR, "features", "features_all.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "market-data-gap-report.json")

FILE_PATTERNS = {
    "venue_ohlcv_plus": [
        os.path.join(BASE_DIR, "features", "*.parquet"),
        os.path.join(BASE_DIR, "raw", "*.parquet"),
    ],
    "top_of_book_snapshots": [
        os.path.join(BASE_DIR, "book", "*.jsonl"),
        os.path.join(BASE_DIR, "book", "*.parquet"),
        os.path.join(BASE_DIR, "market-data", "book*", "*.parquet"),
    ],
    "derivatives_state": [
        os.path.join(BASE_DIR, "derivatives", "*.parquet"),
        os.path.join(BASE_DIR, "market-data", "derivatives*", "*.parquet"),
    ],
    "market_breadth_context": [
        os.path.join(BASE_DIR, "breadth", "*.parquet"),
        os.path.join(BASE_DIR, "regime", "*.parquet"),
    ],
    "trade_tape_or_proxy": [
        os.path.join(BASE_DIR, "microstructure", "*.parquet"),
        os.path.join(BASE_DIR, "trade-tape", "*.jsonl"),
    ],
    "calendar_and_event_flags": [
        os.path.join(BASE_DIR, "events", "*.parquet"),
        os.path.join(BASE_DIR, "calendar", "*.parquet"),
    ],
}

FIELD_HINTS = {
    "venue_ohlcv_plus": ["ret_", "volume", "turnover", "trade_count"],
    "top_of_book_snapshots": ["spread", "depth", "impact", "book", "staleness"],
    "derivatives_state": ["funding", "open_interest", "basis", "liquidation", "long_short"],
    "market_breadth_context": ["breadth", "stablecoin", "btc_", "eth_"],
    "trade_tape_or_proxy": ["taker", "micro_", "pressure", "imbalance"],
    "calendar_and_event_flags": ["event_", "hours_to_event", "macro", "incident"],
}


def _safe_columns() -> list[str]:
    if not os.path.exists(FEATURES_FILE):
        return []
    try:
        parquet = pq.ParquetFile(FEATURES_FILE)
        return [str(c) for c in parquet.schema.names]
    except Exception:
        return []


def _matching_files(source_name: str) -> list[str]:
    matches: list[str] = []
    for pattern in FILE_PATTERNS.get(source_name, []):
        matches.extend(glob.glob(pattern))
    return sorted({m for m in matches if os.path.exists(m)})


def build_report() -> dict:
    columns = _safe_columns()
    sources = []
    required_total = 0
    required_ready = 0

    for spec in data_source_specs():
        name = str(spec.get("name") or "")
        priority = str(spec.get("priority") or "stretch")
        if priority == "required":
            required_total += 1

        file_matches = _matching_files(name)
        hints = FIELD_HINTS.get(name, [])
        matched_columns = sorted(
            [col for col in columns if any(hint in col for hint in hints)]
        )
        if file_matches and matched_columns:
            status = "present"
        elif matched_columns:
            status = "proxy_only"
        elif file_matches:
            status = "raw_only"
        else:
            status = "missing"

        if priority == "required" and status in {"present", "proxy_only"}:
            required_ready += 1

        sources.append(
            {
                "name": name,
                "priority": priority,
                "status": status,
                "minimum_fields": spec.get("minimum_fields") or [],
                "matched_columns": matched_columns[:20],
                "matched_file_count": len(file_matches),
                "matched_files": file_matches[:10],
                "purpose": spec.get("purpose"),
            }
        )

    overall_status = "ok" if required_total and required_ready == required_total else "gaps_open"
    next_steps = []
    for row in sources:
        if row["status"] == "missing" and row["priority"] == "required":
            next_steps.append(f"collect_{row['name']}")
        elif row["status"] == "proxy_only" and row["priority"] == "required":
            next_steps.append(f"upgrade_{row['name']}_from_proxy")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "features_file": FEATURES_FILE,
        "column_count": len(columns),
        "required_sources_ready": required_ready,
        "required_sources_total": required_total,
        "sources": sources,
        "next_steps": next_steps,
    }


def main() -> None:
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge market data gap report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
