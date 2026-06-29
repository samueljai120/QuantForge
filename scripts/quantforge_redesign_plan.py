#!/usr/bin/env python3
"""QuantForge - redesign blueprint artifact.

Turns the current recovery evidence into a structured redesign plan so the research core
can carry a coherent next-step object across dashboard, memory, and sync flows.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_rebuild_blueprint import build_rebuild_program

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTPUT_FILE = os.path.join(BASE_DIR, "redesign-plan.json")
RECOVERY_FILE = os.path.join(BASE_DIR, "candidate-recovery.json")
OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
DIAGNOSIS_FILE = os.path.join(BASE_DIR, "diagnosis-report.json")
MONITOR_FILE = os.path.join(BASE_DIR, "monitor-report.json")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
TARGET_REBUILD_FILE = os.path.join(BASE_DIR, "target-rebuild-report.json")
FEATURE_GAP_FILE = os.path.join(BASE_DIR, "feature-gap-report.json")
SEGMENTED_HOLDOUT_FILE = os.path.join(BASE_DIR, "segmented-holdout-report.json")
EXECUTION_REALISM_FILE = os.path.join(BASE_DIR, "execution-realism-report.json")
MARKET_DATA_GAP_FILE = os.path.join(BASE_DIR, "market-data-gap-report.json")
DATA_SOURCE_RESEARCH_FILE = os.path.join(BASE_DIR, "data-source-research-report.json")
MODEL_LAYER_FILE = os.path.join(BASE_DIR, "model-layer-report.json")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def build_plan():
    recovery = read_json(RECOVERY_FILE)
    outcomes = read_json(OUTCOMES_FILE)
    diagnosis = read_json(DIAGNOSIS_FILE)
    monitor = read_json(MONITOR_FILE)
    governance = read_json(GOVERNANCE_FILE)
    target_rebuild = read_json(TARGET_REBUILD_FILE)
    feature_gap = read_json(FEATURE_GAP_FILE)
    segmented_holdout = read_json(SEGMENTED_HOLDOUT_FILE)
    execution_realism = read_json(EXECUTION_REALISM_FILE)
    market_data_gap = read_json(MARKET_DATA_GAP_FILE)
    data_source_research = read_json(DATA_SOURCE_RESEARCH_FILE)
    model_layer = read_json(MODEL_LAYER_FILE)

    latest = (outcomes.get("latest") or {}) if isinstance(outcomes, dict) else {}
    active_type = str(recovery.get("type") or "none")
    causes = diagnosis.get("causes") or []
    drift_flags = monitor.get("drift_flags") or []
    regime = (monitor.get("regime") or {}).get("entropy_label") or "unknown"
    recommendation = governance.get("recommendation") or "unknown"
    rebuild_program = build_rebuild_program()
    weakest_segments = segmented_holdout.get("weakest_segments") or []
    weakest_positive_slice = weakest_segments[0] if weakest_segments else None

    status = "observe"
    workstreams = []
    success_gates = []
    if active_type == "quantforge_redesign":
        status = "ready"
        workstreams = [
            {
                "name": "setup_labels",
                "goal": "Replace unlabeled entries with explicit setup classes and cleaner failure tags.",
            },
            {
                "name": "training_targets",
                "goal": "Tighten outcome labels and holdout expectancy targets before retraining.",
            },
            {
                "name": "regime_features",
                "goal": "Promote entropy, trend, and volatility into first-class training inputs.",
            },
            {
                "name": "policy_split",
                "goal": "Separate prediction, risk filter, and execution policy so each can be evaluated independently.",
            },
            {
                "name": "entry_selection",
                "goal": "Require regime support and labeled-setup alignment before fresh entries.",
            },
        ]
        success_gates = [
            "Candidate must beat baseline in bounded paper-only evaluation.",
            "Governance must improve beyond REVIEW before promotion is considered.",
            "Durable history posture must stop reporting degraded before wider rollout.",
        ]
    elif active_type == "competitiveness_gap_rebuild":
        status = "handoff_to_competitiveness_lane"
        workstreams = [
            {
                "name": "data_competitiveness",
                "goal": "Upgrade data inputs beyond current thin OHLCV-style context.",
            },
            {
                "name": "execution_realism",
                "goal": "Rework paper assumptions around spread, slippage, and latency so evaluation is less naive.",
            },
            {
                "name": "model_specialization",
                "goal": "Split the monolithic decision path into prediction, regime, risk, and execution layers.",
            },
        ]
        success_gates = [
            "Only return to a live bounded trial after the competitiveness-gap rebuild has a concrete paper candidate.",
            "The rebuilt lane must beat baseline under execution-realistic assumptions.",
        ]
    elif active_type == "quantforge_research_hold":
        status = "rebuild_now"
        workstreams = [
            {
                "name": "labels_and_targets",
                "goal": "Redesign labels and targets around cleaner post-cost outcome definitions instead of the current thin 4h directional framing.",
            },
            {
                "name": "data_and_features",
                "goal": "Add richer market context, turnover, spread, and microstructure-aware features before the next candidate is trained.",
            },
            {
                "name": "execution_realism",
                "goal": "Tighten slippage, spread, latency, and fill assumptions so paper evaluation is harder to game.",
            },
            {
                "name": "model_layers",
                "goal": "Split prediction, regime detection, risk filter, and execution policy into distinct evaluated layers.",
            },
            {
                "name": "strategy_scope",
                "goal": "Keep QuantForge on slower, higher-conviction, major-symbol setups until a rebuilt candidate proves edge.",
            },
        ]
        success_gates = [
            "Do not queue another bounded paper trial until deeper rebuild artifacts exist for labels, data/features, and execution realism.",
            "The rebuilt candidate must beat baseline under execution-realistic assumptions, not just improve selectivity.",
            "QuantForge should remain flat or paper-constrained while the rebuild lane is active.",
        ]
    elif active_type == "quantforge_layered_trial":
        status = "layered_trial_ready"
        workstreams = [
            {
                "name": "prediction_layer",
                "goal": "Use the rebuilt, holdout-positive prediction layer in a bounded paper trial.",
            },
            {
                "name": "regime_gate",
                "goal": "Keep entries limited to regime-supported, labeled setups with rebuilt breadth and entropy context.",
            },
            {
                "name": "risk_and_execution",
                "goal": "Enforce aligned execution realism and bounded risk filtering during the layered paper lane.",
            },
        ]
        success_gates = [
            "The layered trial must beat baseline in paper-only mode before any wider progression.",
            "If the layered trial fails, QuantForge returns to research hold rather than another shallow rotation.",
        ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_id": recovery.get("candidate_id"),
        "candidate_type": active_type,
        "latest_failed_candidate": latest.get("candidate_id"),
        "latest_failed_type": latest.get("type"),
        "latest_failed_assessment": latest.get("assessment"),
        "context": {
            "governance_recommendation": recommendation,
            "diagnosis_causes": causes,
            "drift_flags": drift_flags,
            "regime_entropy_label": regime,
            "target_rebuild_status": target_rebuild.get("status"),
            "target_rebuild_overall_ready": ((target_rebuild.get("gates") or {}).get("overall_ready")),
            "feature_gap_status": feature_gap.get("status"),
            "feature_families_present": feature_gap.get("families_present"),
            "segmented_holdout_status": segmented_holdout.get("status"),
            "segmented_holdout_trade_count": segmented_holdout.get("trade_count"),
            "weakest_positive_slice": weakest_positive_slice,
            "execution_realism_status": execution_realism.get("status"),
            "execution_realism_verdict": execution_realism.get("verdict"),
            "market_data_gap_status": market_data_gap.get("status"),
            "required_data_sources_ready": market_data_gap.get("required_sources_ready"),
            "data_source_research_status": data_source_research.get("status"),
            "data_source_research_next": data_source_research.get("recommended_next_step"),
            "model_layer_status": model_layer.get("status"),
            "model_layer_next_step": model_layer.get("next_step"),
            "model_layers_ready": model_layer.get("ready_layers"),
        },
        "workstreams": workstreams,
        "success_gates": success_gates,
    }
    if target_rebuild:
        support_counts = target_rebuild.get("support_counts") or {}
        setup_target_summary = target_rebuild.get("setup_target_summary") or {}
        surviving_long_slices = [
            name for name, count in [
                ("trend_long", int(support_counts.get("long_trend_positive") or 0)),
                ("breakout_long", int(support_counts.get("long_breakout_positive") or 0)),
                ("rebound_long", int(support_counts.get("long_rebound_positive") or 0)),
            ]
            if count > 0
        ]
        payload["target_rebuild"] = {
            "status": target_rebuild.get("status"),
            "row_count": target_rebuild.get("row_count"),
            "summary": target_rebuild.get("summary") or {},
            "gates": target_rebuild.get("gates") or {},
            "support_counts": support_counts,
            "setup_target_summary": setup_target_summary,
            "surviving_long_slices": surviving_long_slices,
        }
    if feature_gap:
        payload["feature_gap"] = {
            "status": feature_gap.get("status"),
            "column_count": feature_gap.get("column_count"),
            "families_present": feature_gap.get("families_present"),
            "families_total": feature_gap.get("families_total"),
            "families": feature_gap.get("families") or [],
        }
    if segmented_holdout:
        payload["segmented_holdout"] = {
            "status": segmented_holdout.get("status"),
            "trade_count": segmented_holdout.get("trade_count"),
            "failing_segments": segmented_holdout.get("failing_segments") or [],
            "weakest_segments": weakest_segments,
            "strongest_segments": segmented_holdout.get("strongest_segments") or [],
        }
    if execution_realism:
        payload["execution_realism"] = {
            "status": execution_realism.get("status"),
            "verdict": execution_realism.get("verdict"),
            "comparison": execution_realism.get("comparison") or {},
            "next_actions": execution_realism.get("next_actions") or [],
        }
    if market_data_gap:
        payload["market_data_gap"] = {
            "status": market_data_gap.get("status"),
            "required_sources_ready": market_data_gap.get("required_sources_ready"),
            "required_sources_total": market_data_gap.get("required_sources_total"),
            "next_steps": market_data_gap.get("next_steps") or [],
            "sources": market_data_gap.get("sources") or [],
        }
    if data_source_research:
        payload["data_source_research"] = {
            "status": data_source_research.get("status"),
            "recommended_next_step": data_source_research.get("recommended_next_step"),
            "top_research_sources": data_source_research.get("top_research_sources") or [],
        }
    if model_layer:
        payload["model_layer"] = {
            "status": model_layer.get("status"),
            "ready_layers": model_layer.get("ready_layers"),
            "total_layers": model_layer.get("total_layers"),
            "next_step": model_layer.get("next_step"),
            "layers": model_layer.get("layers") or {},
        }
    if status in {"handoff_to_competitiveness_lane", "rebuild_now", "layered_trial_ready"}:
        payload["rebuild_program"] = rebuild_program
    return payload


def main():
    payload = build_plan()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge redesign plan")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
