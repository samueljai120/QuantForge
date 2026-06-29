#!/usr/bin/env python3
"""QuantForge - competitiveness-gap rebuild plan artifact.

Turns repeated failed recovery outcomes into a concrete deeper-rebuild plan so
the research core can continue automatically after trials finish without falling back to
shallow tweak loops.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_rebuild_blueprint import build_rebuild_program

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
RECOVERY_FILE = os.path.join(BASE_DIR, "candidate-recovery.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "competitiveness-plan.json")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def build_plan():
    outcomes = read_json(OUTCOMES_FILE)
    recovery = read_json(RECOVERY_FILE)
    rebuild_program = build_rebuild_program()
    latest = (outcomes.get("latest") or {}) if isinstance(outcomes, dict) else {}
    history = (outcomes.get("history") or []) if isinstance(outcomes, dict) else []
    failed_types = [
        str(row.get("type", "") or "")
        for row in sorted(history, key=lambda r: str(r.get("recorded_at", "") or ""))[-5:]
        if str(row.get("assessment", "") or "").lower() == "fail"
    ]

    candidate_type = str(recovery.get("type") or "")
    status = "observe"
    if candidate_type in {"competitiveness_gap_rebuild", "quantforge_research_hold"}:
        status = "ready"
    elif {"quantforge_redesign", "model_recalibration"}.issubset(set(failed_types)):
        status = "ready"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_type": candidate_type or None,
        "latest_outcome_type": latest.get("type"),
        "latest_outcome_assessment": latest.get("assessment"),
        "recent_failed_types": failed_types,
        "lanes": [
            {
                "name": "data",
                "goal": "Upgrade market data quality with richer venue, spread, turnover, derivatives, breadth, and execution-quality context.",
            },
            {
                "name": "execution",
                "goal": "Make paper evaluation more realistic with spread/slippage and latency-aware assumptions.",
            },
            {
                "name": "model_architecture",
                "goal": "Separate prediction, regime detection, risk filter, and execution policy into distinct model layers.",
            },
            {
                "name": "strategy_scope",
                "goal": "Narrow QuantForge to slower, higher-conviction, major-symbol setups rather than pseudo-HFT behavior.",
            },
        ],
        "rebuild_program": rebuild_program,
        "success_gates": [
            "A rebuilt candidate must beat baseline after execution-realistic paper evaluation.",
            "The next lane should improve subgroup behavior, not only global PnL snapshots.",
            "Do not widen strategy scope again until the slower, cleaner scope proves itself.",
        ],
    }


def main():
    payload = build_plan()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge competitiveness plan")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
