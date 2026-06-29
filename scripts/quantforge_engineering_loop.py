#!/usr/bin/env python3
"""QuantForge — autonomous engineering loop scaffold.

Consumes diagnosis/governance/promotion artifacts and emits a safe, bounded
engineering action list. This does not patch code automatically; it prepares
the next concrete work packet.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_research_hold_support import summarize_top_alt_research_hold_support

BASE_DIR = os.path.join(cfg.data, "quantforge")
DIAGNOSIS_FILE = os.path.join(BASE_DIR, "diagnosis-report.json")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
PROMOTION_FILE = os.path.join(BASE_DIR, "model", "promotion_report.json")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "engineering-actions.json")
CANDIDATE_OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
TARGET_REBUILD_FILE = os.path.join(BASE_DIR, "target-rebuild-report.json")
FEATURE_GAP_FILE = os.path.join(BASE_DIR, "feature-gap-report.json")
SEGMENTED_HOLDOUT_FILE = os.path.join(BASE_DIR, "segmented-holdout-report.json")
EXECUTION_REALISM_FILE = os.path.join(BASE_DIR, "execution-realism-report.json")
MARKET_DATA_GAP_FILE = os.path.join(BASE_DIR, "market-data-gap-report.json")
DATA_SOURCE_RESEARCH_FILE = os.path.join(BASE_DIR, "data-source-research-report.json")
MODEL_LAYER_FILE = os.path.join(BASE_DIR, "model-layer-report.json")
HEAVY_REPORT_MAX_AGE_HOURS = 36
HEAVY_REPORT_SPECS = {
    "target_rebuild": TARGET_REBUILD_FILE,
    "feature_gap": FEATURE_GAP_FILE,
    "segmented_holdout": SEGMENTED_HOLDOUT_FILE,
    "execution_realism": EXECUTION_REALISM_FILE,
    "market_data_gap": MARKET_DATA_GAP_FILE,
    "data_source_research": DATA_SOURCE_RESEARCH_FILE,
    "model_layer": MODEL_LAYER_FILE,
}


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _file_age_hours(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return (datetime.now(timezone.utc).timestamp() - mtime) / 3600.0


def _load_heavy_reports():
    payloads = {}
    freshness = {}
    stale_inputs = []
    for label, path in HEAVY_REPORT_SPECS.items():
        if not os.path.exists(path):
            payloads[label] = {}
            freshness[label] = {"status": "missing", "age_hours": None, "path": os.path.basename(path)}
            stale_inputs.append(f"{label}:missing")
            continue
        age = _file_age_hours(path)
        payloads[label] = read_json(path)
        if age is None:
            freshness[label] = {"status": "unknown_age", "age_hours": None, "path": os.path.basename(path)}
            stale_inputs.append(f"{label}:unknown_age")
        elif age > HEAVY_REPORT_MAX_AGE_HOURS:
            freshness[label] = {"status": "stale", "age_hours": round(age, 1), "path": os.path.basename(path)}
            stale_inputs.append(f"{label}:{round(age, 1)}h")
        else:
            freshness[label] = {"status": "fresh", "age_hours": round(age, 1), "path": os.path.basename(path)}
    return payloads, freshness, stale_inputs


def target_rule_enabled(target_rebuild, rule_name, default=True):
    profile = target_rebuild.get("profile") or {}
    rules = profile.get("rules") or {}
    value = rules.get(rule_name)
    if value is None:
        return default
    return bool(value)


def build_actions():
    diagnosis = read_json(DIAGNOSIS_FILE)
    governance = read_json(GOVERNANCE_FILE)
    promotion = read_json(PROMOTION_FILE)
    last_scan = read_json(LAST_SCAN_FILE)
    outcomes = read_json(CANDIDATE_OUTCOMES_FILE)
    heavy_reports, heavy_freshness, stale_heavy_inputs = _load_heavy_reports()
    target_rebuild = heavy_reports["target_rebuild"]
    feature_gap = heavy_reports["feature_gap"]
    segmented_holdout = heavy_reports["segmented_holdout"]
    execution_realism = heavy_reports["execution_realism"]
    market_data_gap = heavy_reports["market_data_gap"]
    data_source_research = heavy_reports["data_source_research"]
    model_layer = heavy_reports["model_layer"]

    actions = []
    summary = []
    latest = (outcomes.get("latest") or {}) if isinstance(outcomes, dict) else {}
    history = (outcomes.get("history") or []) if isinstance(outcomes, dict) else []
    latest_type = str(latest.get("type", "") or "")
    latest_assessment = str(latest.get("assessment", "") or "").lower()
    failed_types = [
        str(row.get("type", "") or "")
        for row in sorted(history, key=lambda r: str(r.get("recorded_at", "") or ""))[-5:]
        if str(row.get("assessment", "") or "").lower() == "fail"
    ]

    if governance.get("recommendation") == "REVIEW":
        summary.append("governance_requires_review")
    if stale_heavy_inputs:
        summary.append("stale_rebuild_inputs")
        actions.append({
            "priority": "high",
            "type": "refresh_rebuild_artifacts",
            "why": "Fresh rebuild evidence is missing or stale, so deeper QuantForge conclusions should wait for regenerated heavy reports.",
            "inputs": [entry.split(":")[0] for entry in stale_heavy_inputs],
            "details": stale_heavy_inputs,
        })

    if promotion.get("overall_decision") == "DO_NOT_PROMOTE":
        actions.append({
            "priority": "high",
            "type": "hold_promotion",
            "why": "Current model is not promotion-ready.",
            "inputs": ["promotion_report.json", "governance-report.json"],
        })

    causes = set(diagnosis.get("causes", []))
    if "paper_underperformance" in causes or "negative_recent_trade_expectancy" in causes:
        actions.append({
            "priority": "high",
            "type": "tighten_live_selection",
            "why": "Recent paper outcomes remain negative.",
            "inputs": ["diagnosis-report.json", "last_scan.json"],
        })
    if "weak_recent_close_quality" in causes:
        actions.append({
            "priority": "high",
            "type": "review_recent_losers",
            "why": "Recent close win rate is weak.",
            "targets": [row.get("symbol") for row in diagnosis.get("worst_symbols", [])[:3]],
        })
    if "model_not_promotion_ready" in causes:
        actions.append({
            "priority": "medium",
            "type": "schedule_retrain_review",
            "why": "Model artifacts do not yet meet promotion criteria.",
            "inputs": ["model_meta.json", "best-params.json", "promotion_report.json"],
        })

    feedback = last_scan.get("feedback", {}).get("summary", {})
    if feedback.get("risk_mult", 1.0) < 1.0:
        actions.append({
            "priority": "medium",
            "type": "keep_risk_throttled",
            "why": f"Adaptive risk multiplier is {feedback.get('risk_mult')}.",
        })

    if {"quantforge_redesign", "model_recalibration"}.issubset(set(failed_types)):
        summary.append("competitiveness_gap_rebuild_needed")
        actions.extend([
            {
                "priority": "high",
                "type": "upgrade_market_data_lane",
                "why": "Recent redesign-grade failures suggest the current data/features are not competitive enough.",
                "inputs": ["candidate-outcomes.json", "governance-report.json"],
            },
            {
                "priority": "high",
                "type": "tighten_execution_realism",
                "why": "Paper evaluation needs stronger spread/slippage/latency realism before more model-local tweaks.",
                "inputs": ["candidate-outcomes.json", "last_scan.json"],
            },
            {
                "priority": "high",
                "type": "split_model_layers",
                "why": "Prediction, regime, risk filter, and execution policy should be evaluated as separate layers.",
                "inputs": ["candidate-outcomes.json", "promotion_report.json"],
            },
            {
                "priority": "medium",
                "type": "narrow_strategy_scope",
                "why": "QuantForge should bias toward slower, higher-conviction major-symbol setups instead of broad pseudo-HFT behavior.",
                "inputs": ["candidate-outcomes.json", "diagnosis-report.json"],
            },
        ])
    if "competitiveness_gap_rebuild" in set(failed_types):
        summary.append("quantforge_research_hold_needed")
        actions.extend([
            {
                "priority": "high",
                "type": "rebuild_labels_and_targets",
                "why": "Competitiveness-gap work still failed, so QuantForge needs a deeper label/target redesign before another trial.",
                "inputs": ["candidate-outcomes.json", "diagnosis-report.json", "promotion_report.json"],
            },
            {
                "priority": "high",
                "type": "freeze_new_trial_rotation",
                "why": "Stop rotating through low-leverage candidate classes until the deeper rebuild artifacts are ready.",
                "inputs": ["candidate-outcomes.json", "candidate-review.json"],
            },
            {
                "priority": "high",
                "type": "research_data_sources",
                "why": "The current data lane is not producing edge; research richer market and execution data before the next trial.",
                "inputs": ["candidate-outcomes.json", "governance-report.json"],
            },
        ])
    target_rebuild_fresh = heavy_freshness["target_rebuild"]["status"] == "fresh"
    target_status = str(target_rebuild.get("status") or "")
    target_gates = target_rebuild.get("gates") or {}
    support_counts = target_rebuild.get("support_counts") or {}
    setup_target_summary = target_rebuild.get("setup_target_summary") or {}
    if target_rebuild_fresh and target_status and not bool(target_gates.get("overall_ready")):
        summary.append("target_rebuild_not_ready")
        actions.append({
            "priority": "high",
            "type": "stabilize_labels_and_targets",
            "why": f"Target rebuild lane is {target_status} and not yet ready for another bounded trial.",
            "inputs": ["target-rebuild-report.json", "redesign-plan.json"],
        })
    elif target_rebuild_fresh and target_status == "ready":
        trend_positive = int(support_counts.get("long_trend_positive") or 0)
        breakout_positive = int(support_counts.get("long_breakout_positive") or 0)
        rebound_positive = int(support_counts.get("long_rebound_positive") or 0)
        if trend_positive > 0 and breakout_positive > 0 and rebound_positive == 0:
            summary.append("surviving_long_slices_are_setup_specific")
            actions.append({
                "priority": "high",
                "type": "train_per_setup_long_models",
                "why": (
                    "Target rebuild is ready and surviving long support is concentrated in "
                    f"trend_long={trend_positive} and breakout_long={breakout_positive} with rebound_long removed."
                ),
                "inputs": [
                    "target-rebuild-report.json",
                    "quantforge_ml.py",
                    "segmented-holdout-report.json",
                ],
                "context": {
                    "setup_target_summary": setup_target_summary,
                },
            })
        top_alt_support = summarize_top_alt_research_hold_support()
        if (
            bool(top_alt_support.get("expansion_supported"))
            and (
                "competitiveness_gap_rebuild" in set(failed_types)
                or (latest_type == "competitiveness_gap_rebuild" and latest_assessment == "fail")
            )
        ):
            summary.append("top_alt_long_expansion_supported")
            top_non_majors = top_alt_support.get("top_non_major_symbols") or []
            top_names = [str(row.get("symbol") or "") for row in top_non_majors[:3] if row.get("symbol")]
            actions.append({
                "priority": "high",
                "type": "prepare_major_liquidity_expansion_candidate",
                "execution_policy": "manual_only",
                "llm_eligible": False,
                "why": (
                    "Research-hold rebuild evidence now shows stronger bounded long support in "
                    f"{', '.join(top_names) or 'top-liquidity alts'} than in the current major-only survivors."
                ),
                "inputs": [
                    "target-rebuild-report.json",
                    "candidate-outcomes.json",
                    "features_all.parquet",
                ],
                "context": {
                    "setup_target_summary": setup_target_summary,
                    "top_alt_research_hold_support": top_alt_support,
                },
            })

    if heavy_freshness["feature_gap"]["status"] == "fresh" and feature_gap:
        families_total = int(feature_gap.get("families_total") or 0)
        families_present = int(feature_gap.get("families_present") or 0)
        if families_total and families_present < families_total:
            summary.append("feature_gap_open")
            actions.append({
                "priority": "high",
                "type": "fill_feature_family_gaps",
                "why": f"Only {families_present}/{families_total} rebuild feature families are represented in the current feature store.",
                "inputs": ["feature-gap-report.json", "competitiveness-plan.json"],
            })
    if heavy_freshness["segmented_holdout"]["status"] == "fresh" and segmented_holdout:
        failing_segments = segmented_holdout.get("failing_segments") or []
        if failing_segments:
            top_failure = failing_segments[0]
            summary.append("segmented_holdout_failure_visible")
            actions.append({
                "priority": "high",
                "type": "target_specific_failing_slice",
                "why": f"Segmented holdout shows {top_failure.get('dimension')}={top_failure.get('segment')} failing at {top_failure.get('avg_net_edge_bps')} bps.",
                "inputs": ["segmented-holdout-report.json", "target-rebuild-report.json"],
            })
        else:
            weakest_segments = segmented_holdout.get("weakest_segments") or []
            if weakest_segments:
                weakest = weakest_segments[0]
                weakest_dimension = str(weakest.get("dimension") or "")
                weakest_segment = str(weakest.get("segment") or "")
                rebound_quarantined = (
                    weakest_dimension == "setup_tag"
                    and weakest_segment == "rebound_long"
                    and not target_rule_enabled(target_rebuild, "enable_rebound_long", default=True)
                )
                if rebound_quarantined:
                    summary.append("weakest_slice_quarantined")
                else:
                    summary.append("segmented_holdout_positive_but_concentrated")
                    actions.append({
                        "priority": "medium",
                        "type": "verify_weakest_positive_slice",
                        "why": f"Segmented holdout is positive overall, but the weakest live slice is {weakest_dimension}={weakest_segment} at {weakest.get('avg_net_edge_bps')} bps.",
                        "inputs": ["segmented-holdout-report.json", "target-rebuild-report.json"],
                    })
    if heavy_freshness["execution_realism"]["status"] == "fresh" and execution_realism:
        verdict = str(execution_realism.get("verdict") or "")
        if verdict and verdict != "aligned":
            summary.append(f"execution_realism_{verdict}")
            actions.append({
                "priority": "high",
                "type": "tighten_execution_contract",
                "why": f"Execution realism lane is {verdict}; rebuild assumptions before another bounded trial.",
                "inputs": ["execution-realism-report.json", "target-rebuild-report.json"],
            })
    elif str(latest.get("next_candidate_hint", "") or "") == "competitiveness_gap_rebuild":
        summary.append("competitiveness_gap_candidate_queued")

    if heavy_freshness["market_data_gap"]["status"] == "fresh" and market_data_gap:
        required_total = int(market_data_gap.get("required_sources_total") or 0)
        required_ready = int(market_data_gap.get("required_sources_ready") or 0)
        if required_total and required_ready < required_total:
            summary.append("market_data_gap_open")
            actions.append({
                "priority": "high",
                "type": "close_market_data_contract_gaps",
                "why": f"Only {required_ready}/{required_total} required rebuild data sources are represented well enough for the current QuantForge lane.",
                "inputs": ["market-data-gap-report.json", "competitiveness-plan.json"],
            })
    if heavy_freshness["data_source_research"]["status"] == "fresh" and data_source_research:
        top_sources = data_source_research.get("top_research_sources") or []
        if top_sources:
            top = top_sources[0]
            actions.append({
                "priority": "high",
                "type": "build_top_data_source_collector",
                "why": f"Next highest-leverage source is {top.get('name')} ({top.get('status')}) via {top.get('collector_plan')}.",
                "inputs": ["data-source-research-report.json", "market-data-gap-report.json"],
            })
    if heavy_freshness["model_layer"]["status"] == "fresh" and model_layer:
        ready_layers = int(model_layer.get("ready_layers", 0) or 0)
        total_layers = int(model_layer.get("total_layers", 0) or 0)
        next_step = str(model_layer.get("next_step") or "")
        summary.append(f"model_layers_{ready_layers}_of_{total_layers}")
        if (
            next_step == "prepare_layered_trial_candidate"
            and not (
                "competitiveness_gap_rebuild" in set(failed_types)
                or (latest_type == "quantforge_layered_trial" and latest_assessment == "fail")
            )
        ):
            actions.append({
                "priority": "high",
                "type": "prepare_layered_trial_candidate",
                "why": "Prediction, regime, and execution layers look ready enough to package the next bounded layered trial candidate.",
                "inputs": ["model-layer-report.json", "segmented-holdout-report.json"],
            })
        elif next_step and not (
            next_step == "prepare_layered_trial_candidate"
            and (latest_type == "quantforge_layered_trial" and latest_assessment == "fail")
        ):
            actions.append({
                "priority": "medium",
                "type": "advance_model_layer_split",
                "why": f"Next highest-leverage layer step is {next_step}.",
                "inputs": ["model-layer-report.json", "redesign-plan.json"],
            })

    if not actions:
        actions.append({
            "priority": "low",
            "type": "observe",
            "why": "No urgent engineering action inferred from current diagnostics.",
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "actions": actions,
        "stale_inputs": stale_heavy_inputs,
        "heavy_inputs": heavy_freshness,
    }


def main():
    payload = build_actions()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge engineering loop")
    for action in payload["actions"]:
        print(f"- [{action['priority']}] {action['type']}: {action['why']}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
