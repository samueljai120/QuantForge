#!/usr/bin/env python3
"""QuantForge - explicit model-layer readiness report."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTPUT_FILE = os.path.join(BASE_DIR, "model-layer-report.json")
MODEL_META_FILE = os.path.join(BASE_DIR, "model", "model_meta.json")
MODEL_META_SHORT_FILE = os.path.join(BASE_DIR, "model", "model_meta_short.json")
SEGMENTED_HOLDOUT_FILE = os.path.join(BASE_DIR, "segmented-holdout-report.json")
EXECUTION_REALISM_FILE = os.path.join(BASE_DIR, "execution-realism-report.json")
MARKET_DATA_GAP_FILE = os.path.join(BASE_DIR, "market-data-gap-report.json")
MONITOR_FILE = os.path.join(BASE_DIR, "monitor-report.json")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")


def read_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _layer(status: str, score: float, rationale: str, details: dict | None = None) -> dict:
    return {
        "status": status,
        "score": round(float(score), 4),
        "rationale": rationale,
        "details": details or {},
    }


TRIALABLE_FAILING_SLICES = {
    ("regime_bucket", "fragile"),
    ("setup_tag", "exhaustion_short"),
    ("direction", "short"),
}


def _prediction_layer_ready(long_meta: dict, holdout: dict) -> tuple[bool, dict]:
    holdout_mean_edge = float((holdout.get("summary") or {}).get("net_edge_bps_mean", 0.0) or 0.0)
    holdout_trades = int(holdout.get("trade_count", 0) or 0)
    analysis_mode = str(holdout.get("analysis_mode", "") or "")
    failing_segments = holdout.get("failing_segments") or []
    failing_keys = {
        (str(row.get("dimension") or ""), str(row.get("segment") or ""))
        for row in failing_segments
    }
    legacy_gate_ready = bool(long_meta.get("gate_pass")) and holdout.get("status") == "ready" and holdout_mean_edge > 0
    executed_subset_ready = (
        holdout.get("status") == "ready"
        and analysis_mode == "executed_subset_ranked"
        and holdout_trades >= 5000
        and holdout_mean_edge > 0
        and bool(failing_keys)
        and failing_keys.issubset(TRIALABLE_FAILING_SLICES)
    )
    return legacy_gate_ready or executed_subset_ready, {
        "analysis_mode": analysis_mode,
        "holdout_trades": holdout_trades,
        "holdout_mean_net_edge_bps": round(holdout_mean_edge, 2),
        "failing_segment_count": len(failing_segments),
        "failing_segments": sorted(f"{dim}:{seg}" for dim, seg in failing_keys),
        "ready_basis": (
            "legacy_gate"
            if legacy_gate_ready
            else "executed_subset_ranked_positive"
            if executed_subset_ready
            else "not_ready"
        ),
    }


def build_report() -> dict:
    long_meta = read_json(MODEL_META_FILE)
    short_meta = read_json(MODEL_META_SHORT_FILE)
    holdout = read_json(SEGMENTED_HOLDOUT_FILE)
    realism = read_json(EXECUTION_REALISM_FILE)
    market_data_gap = read_json(MARKET_DATA_GAP_FILE)
    monitor = read_json(MONITOR_FILE)
    governance = read_json(GOVERNANCE_FILE)
    last_scan = read_json(LAST_SCAN_FILE)

    holdout_mean_edge = float((holdout.get("summary") or {}).get("net_edge_bps_mean", 0.0) or 0.0)
    holdout_trades = int(holdout.get("trade_count", 0) or 0)
    failing_segments = holdout.get("failing_segments") or []
    weakest_segments = holdout.get("weakest_segments") or []
    prediction_ready, prediction_basis = _prediction_layer_ready(long_meta, holdout)
    prediction_score = min(1.0, max(0.0, holdout_mean_edge / 150.0))
    prediction = _layer(
        "ready" if prediction_ready else "rebuild",
        prediction_score,
        "Primary prediction layer is ready for a bounded layered trial."
        if prediction_ready else
        "Primary prediction layer still needs stronger holdout expectancy.",
        {
            "long_gate_pass": bool(long_meta.get("gate_pass")),
            "short_gate_pass": bool(short_meta.get("gate_pass")),
            **prediction_basis,
        },
    )

    regime_entropy = (((monitor.get("regime") or {}).get("entropy_label")) or "unknown").lower()
    market_data_ready = str(market_data_gap.get("status") or "") == "ok"
    regime_ready = market_data_ready and regime_entropy != "unknown"
    regime = _layer(
        "ready" if regime_ready else "rebuild",
        1.0 if regime_ready else 0.4,
        "Regime layer has live entropy and breadth context."
        if regime_ready else
        "Regime layer is missing stable breadth/entropy context.",
        {
            "entropy_label": regime_entropy,
            "market_data_gap_status": market_data_gap.get("status"),
            "required_sources_ready": market_data_gap.get("required_sources_ready"),
        },
    )

    feedback = ((last_scan.get("feedback") or {}).get("summary") or {})
    risk_mult = float(feedback.get("risk_mult", 1.0) or 1.0)
    risk_ready = holdout.get("status") == "ready" and risk_mult <= 1.0
    risk_filter = _layer(
        "ready" if risk_ready else "observe",
        min(1.0, max(0.0, risk_mult)),
        "Risk filter is active and bounded."
        if risk_ready else
        "Risk filter is present, but still needs validation against the rebuilt holdout state.",
        {
            "risk_mult": round(risk_mult, 4),
            "governance_recommendation": governance.get("recommendation"),
            "weakest_segment": weakest_segments[0] if weakest_segments else None,
        },
    )

    realism_ready = str(realism.get("verdict") or "") == "aligned"
    execution = _layer(
        "ready" if realism_ready else "rebuild",
        1.0 if realism_ready else 0.5,
        "Execution policy is aligned with current rebuild economics."
        if realism_ready else
        "Execution policy still needs stricter realism alignment.",
        {
            "execution_realism_status": realism.get("status"),
            "execution_realism_verdict": realism.get("verdict"),
            "comparison": realism.get("comparison") or {},
        },
    )

    layers = {
        "prediction": prediction,
        "regime_gate": regime,
        "risk_filter": risk_filter,
        "execution_policy": execution,
    }
    ready_layers = sum(1 for layer in layers.values() if layer["status"] == "ready")
    overall_status = "ready_for_layered_trial" if ready_layers >= 3 and prediction_ready else "layer_rebuild_in_progress"

    if overall_status == "ready_for_layered_trial":
        next_step = "prepare_layered_trial_candidate"
    elif not prediction_ready:
        next_step = "retrain_prediction_layer_with_rebuilt_context"
    elif not regime_ready:
        next_step = "stabilize_regime_gate"
    elif not realism_ready:
        next_step = "tighten_execution_policy"
    else:
        next_step = "validate_risk_filter_contract"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "ready_layers": ready_layers,
        "total_layers": len(layers),
        "next_step": next_step,
        "layers": layers,
    }


def main() -> None:
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge model-layer report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
