#!/usr/bin/env python3
"""QuantForge data-source research report.

Turns the market-data contract and current gap report into an actionable,
prioritized build packet for the next QuantForge data lane.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_rebuild_blueprint import data_source_specs

BASE_DIR = os.path.join(cfg.data, "quantforge")
MARKET_DATA_GAP_FILE = os.path.join(BASE_DIR, "market-data-gap-report.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data-source-research-report.json")

LEVERAGE_SCORES = {
    "top_of_book_snapshots": 10,
    "derivatives_state": 9,
    "market_breadth_context": 8,
    "venue_ohlcv_plus": 7,
    "trade_tape_or_proxy": 6,
    "calendar_and_event_flags": 5,
}

EFFORT_SCORES = {
    "top_of_book_snapshots": 8,
    "derivatives_state": 6,
    "market_breadth_context": 5,
    "venue_ohlcv_plus": 3,
    "trade_tape_or_proxy": 7,
    "calendar_and_event_flags": 4,
}

SOURCE_BUILD_NOTES = {
    "venue_ohlcv_plus": {
        "collector": "normalize existing venue candles into one majors-first parquet contract",
        "first_artifact": "venue-quality report",
    },
    "top_of_book_snapshots": {
        "collector": "sample best bid/ask and top-5 depth on a fixed cadence for majors",
        "first_artifact": "spread-depth parquet",
    },
    "derivatives_state": {
        "collector": "capture funding, OI, basis, long-short skew, and liquidation pressure",
        "first_artifact": "crowding report",
    },
    "market_breadth_context": {
        "collector": "build BTC/ETH and majors/alts breadth context at the same bar cadence",
        "first_artifact": "breadth parquet",
    },
    "trade_tape_or_proxy": {
        "collector": "start with aggressive flow proxy from short-horizon quote/volume changes before raw tape",
        "first_artifact": "pressure imbalance report",
    },
    "calendar_and_event_flags": {
        "collector": "tag macro, exchange, listing, and incident events with hours-to-event overlap",
        "first_artifact": "event overlap report",
    },
}


def read_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_report() -> dict:
    gap = read_json(MARKET_DATA_GAP_FILE)
    gap_rows = {str(row.get("name") or ""): row for row in (gap.get("sources") or [])}
    required_ready = int(gap.get("required_sources_ready") or 0)
    required_total = int(gap.get("required_sources_total") or 0)

    prioritized = []
    for spec in data_source_specs():
        name = str(spec.get("name") or "")
        row = gap_rows.get(name, {})
        status = str(row.get("status") or "missing")
        priority = str(spec.get("priority") or "stretch")
        leverage = LEVERAGE_SCORES.get(name, 5)
        effort = EFFORT_SCORES.get(name, 5)
        urgency = leverage * 2 + (3 if priority == "required" else 0) - effort
        notes = SOURCE_BUILD_NOTES.get(name, {})
        prioritized.append(
            {
                "name": name,
                "priority": priority,
                "status": status,
                "urgency_score": urgency,
                "leverage_score": leverage,
                "effort_score": effort,
                "collector_plan": notes.get("collector"),
                "first_artifact": notes.get("first_artifact"),
                "minimum_fields": spec.get("minimum_fields") or [],
                "purpose": spec.get("purpose"),
                "next_action": (
                    f"upgrade_{name}_from_proxy" if status == "proxy_only"
                    else f"collect_{name}" if status == "missing"
                    else "observe"
                ),
            }
        )

    prioritized.sort(
        key=lambda row: (
            0 if row["status"] in {"missing", "proxy_only"} else 1,
            -int(row["urgency_score"]),
            row["name"],
        )
    )

    top_research = [row for row in prioritized if row["status"] in {"missing", "proxy_only"}][:3]
    status = "ready" if top_research else "observe"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "required_sources_ready": required_ready,
        "required_sources_total": required_total,
        "top_research_sources": top_research,
        "sources": prioritized,
        "recommended_next_step": top_research[0]["next_action"] if top_research else "observe",
    }


def main() -> None:
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge data source research report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
