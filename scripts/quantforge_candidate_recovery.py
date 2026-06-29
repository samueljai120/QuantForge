#!/usr/bin/env python3
"""QuantForge — bounded recovery candidate proposal.

Generates a concrete recovery candidate artifact from diagnosis, governance,
promotion, and durable operator history. This is a proposal only; execution
still obeys the local autopilot artifact and requires downstream validation.
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
AGI_HISTORY_FILE = os.path.join(cfg.data, "agi-operator-history-summary.json")
MONITOR_FILE = os.path.join(BASE_DIR, "monitor-report.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "candidate-recovery.json")
LANES_FILE = os.path.join(BASE_DIR, "experiment-lanes.json")
CANDIDATE_OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
MODEL_LAYER_FILE = os.path.join(BASE_DIR, "model-layer-report.json")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")
SETUP_QUALITY_ALLOWED_LONG_SETUPS = ("trend_long", "breakout_long")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _f(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _parse_ts(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _outcome_is_stale_for_candidate(latest_outcome: dict, candidate_lane: dict) -> bool:
    """True when the latest recorded trial predates the current candidate model."""

    if not isinstance(latest_outcome, dict) or not isinstance(candidate_lane, dict):
        return False
    outcome_dt = _parse_ts(latest_outcome.get("recorded_at") or latest_outcome.get("completed_at"))
    candidate_dt = _parse_ts(candidate_lane.get("model_trained_at"))
    if outcome_dt is None or candidate_dt is None:
        return False
    return outcome_dt < candidate_dt


def _setup_quality_recovery_symbol_allowlist(last_scan: dict) -> list[str]:
    if not isinstance(last_scan, dict):
        return []
    ranked: dict[str, float] = {}
    for row in last_scan.get("results") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "") or "").strip()
        setup_tag = str(row.get("setup_tag", "") or "").strip().lower()
        if not symbol or setup_tag not in SETUP_QUALITY_ALLOWED_LONG_SETUPS:
            continue
        status = str(row.get("status", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "")
        reason_l = reason.lower()
        score = None
        if status == "skip" and "restricts longs to major-liquidity symbols" in reason:
            score = 3.0
        elif status == "skip" and "limits longs to majors and top-liquidity alts" in reason_l:
            score = 2.5
        elif status in {"hold", "signal"}:
            try:
                long_conf = float(row.get("long_confidence") or 0.0)
            except Exception:
                long_conf = 0.0
            try:
                short_conf = float(row.get("short_confidence") or 0.0)
            except Exception:
                short_conf = 0.0
            if long_conf >= max(short_conf, 0.55):
                score = 1.0 + long_conf
        if score is None:
            continue
        ranked[symbol] = max(score, ranked.get(symbol, 0.0))
    return [
        symbol
        for symbol, _ in sorted(ranked.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]


def build_candidate():
    diagnosis = read_json(DIAGNOSIS_FILE)
    governance = read_json(GOVERNANCE_FILE)
    promotion = read_json(PROMOTION_FILE)
    agi_history = read_json(AGI_HISTORY_FILE)
    monitor = read_json(MONITOR_FILE)
    lanes = read_json(LANES_FILE)
    outcomes = read_json(CANDIDATE_OUTCOMES_FILE)
    model_layer = read_json(MODEL_LAYER_FILE)
    last_scan = read_json(LAST_SCAN_FILE)
    existing_candidate = read_json(OUTPUT_FILE)
    existing_trial = (lanes.get("candidate_trial") or {}) if isinstance(lanes, dict) else {}
    baseline_lane = (lanes.get("baseline") or {}) if isinstance(lanes, dict) else {}
    candidate_lane = (lanes.get("candidate") or {}) if isinstance(lanes, dict) else {}
    latest_outcome = (outcomes.get("latest") or {}) if isinstance(outcomes, dict) else {}
    outcome_history = (outcomes.get("history") or []) if isinstance(outcomes, dict) else []

    causes = set(diagnosis.get("causes", []))
    posture = agi_history.get("posture", "unknown")
    persistent_review = bool(agi_history.get("persistent_review", False))
    persistent_drift = bool(agi_history.get("persistent_drift", False))
    avg_risk = _f((agi_history.get("averages") or {}).get("adaptive_risk_mult"), 1.0)
    recent_win_rate = _f((governance.get("recent_closes") or {}).get("win_rate"), 0.0)
    total_pnl_pct = _f((governance.get("paper") or {}).get("total_pnl_pct"), 0.0)
    drift_flags = monitor.get("drift_flags") or []
    regime = monitor.get("regime") or {}
    worst_setups = diagnosis.get("worst_setups") or []
    unlabeled_setup = next((row for row in worst_setups if str(row.get("setup")) == "unlabeled"), {})
    unlabeled_count = int(unlabeled_setup.get("count", 0) or 0)
    baseline_pnl_pct = _f(baseline_lane.get("paper_total_pnl_pct"), total_pnl_pct)
    candidate_pnl_pct = _f(candidate_lane.get("paper_total_pnl_pct"), total_pnl_pct)
    pnl_gap_vs_baseline = round(candidate_pnl_pct - baseline_pnl_pct, 4)
    latest_outcome_type = str(latest_outcome.get("type", "") or "")
    latest_outcome_assessment = str(latest_outcome.get("assessment", "") or "").lower()
    latest_outcome_hint = str(latest_outcome.get("next_candidate_hint", "") or "")
    if (
        trial_status := str(existing_trial.get("status", "")).lower()
    ) == "completed" and str(existing_trial.get("assessment", "") or ""):
        latest_outcome_type = str(existing_trial.get("type", "") or latest_outcome_type)
        latest_outcome_assessment = str(existing_trial.get("assessment", "") or "").lower()
        latest_outcome_hint = str(existing_trial.get("next_candidate_hint", "") or latest_outcome_hint)
    model_layer_status = str(model_layer.get("status", "") or "")
    model_layer_next_step = str(model_layer.get("next_step", "") or "")
    recent_failed_types = [
        str(row.get("type", "") or "")
        for row in sorted(outcome_history, key=lambda r: str(r.get("recorded_at", "") or ""))[-4:]
        if str(row.get("assessment", "") or "").lower() == "fail"
    ]
    stale_latest_outcome = _outcome_is_stale_for_candidate(latest_outcome, candidate_lane)
    active_trial_preserves_hint = bool(
        stale_latest_outcome
        and trial_status in {"queued", "active"}
        and latest_outcome_hint
        and str(existing_trial.get("type", "") or "") == latest_outcome_hint
        and bool(existing_trial.get("paper_only", False))
    )
    if stale_latest_outcome and not active_trial_preserves_hint:
        latest_outcome_type = ""
        latest_outcome_assessment = ""
        latest_outcome_hint = ""
    elif active_trial_preserves_hint:
        stale_latest_outcome = False
    setup_quality_allowlist = _setup_quality_recovery_symbol_allowlist(last_scan)

    priority = "observe"
    candidate_type = "none"
    rationale = []
    changes = []
    validation = []
    blockers = []
    prior_trial_outcome = None

    if posture == "degraded" or persistent_review or persistent_drift:
        candidate_type = "defensive_recovery"
        priority = "high"
        rationale.append("Durable history remains degraded across repeated operator cycles.")
        changes.extend([
            {"key": "max_long_positions", "action": "reduce", "from": 3, "to": 2},
            {"key": "entry_profile", "action": "prefer", "value": "majors_only"},
            {"key": "generic_long_policy", "action": "tighten", "value": "require_top_quality"},
        ])
        validation.extend([
            "Candidate should reduce new-entry count while preserving management of existing winners.",
            "Candidate should improve paper drawdown or recent expectancy before any promotion.",
        ])

    if "persistent_underperformance_across_cycles" in causes or total_pnl_pct <= -5.0:
        candidate_type = "capital_preservation"
        priority = "high"
        changes.append({"key": "risk_mult_cap", "action": "cap", "value": round(min(avg_risk, 0.45), 2)})
        changes.append({"key": "non_major_entries", "action": "disable", "value": True})
        rationale.append("Paper underperformance is persistent enough to favor capital preservation.")
        if {"model_no_signal_bottleneck", "threshold_miss_bottleneck"} & causes:
            changes.append({"key": "max_short_positions", "action": "set", "value": 1})
            changes.append({"key": "allow_short_entries_in_adverse_regime", "action": "enable", "value": True})
            changes.append({"key": "trial_long_threshold_relief", "action": "set", "value": 0.10})
            changes.append({"key": "trial_short_threshold_relief", "action": "set", "value": 0.15})
            rationale.append("The paper lane is starved by threshold misses, so the bounded trial should allow one short slot and relax entry thresholds without widening the universe.")

    if "weak_recent_close_quality" in causes or recent_win_rate < 0.35:
        changes.append({"key": "candidate_focus", "action": "review", "value": "recent_losers_and_setup_tags"})
        rationale.append("Recent close quality remains weak and should be the first recovery target.")

    if "non_major_exposure" in drift_flags:
        changes.append({"key": "symbol_universe", "action": "restrict", "value": "major_liquidity_tier"})
    if str(regime.get("entropy_label", "")).upper() == "CHAOTIC":
        changes.append({"key": "regime_policy", "action": "tighten", "value": "chaos_rejects_generic_and_non_major_longs"})
        rationale.append("Entropy regime is chaotic, so recovery should avoid weaker long setups.")

    if promotion.get("overall_decision") == "DO_NOT_PROMOTE":
        blockers.append("Current promoted path remains blocked by governance/promotion.")
        validation.append("Do not promote this candidate until durable history posture improves from degraded.")

    if candidate_type == "none":
        rationale.append("No urgent candidate change inferred from current evidence.")
        validation.append("Continue observation until more durable cycles accumulate.")

    trial_type = str(existing_trial.get("type", "") or "")
    existing_candidate_type = str(existing_candidate.get("type", ""))
    existing_candidate_status = str(existing_candidate.get("status", "")).lower()
    completed_trial_failed = (
        trial_status == "completed"
        and int(existing_trial.get("cycles_run", 0) or 0) >= int(existing_trial.get("max_cycles", 0) or 0)
        and latest_outcome_assessment in {"fail", "insufficient"}
    )

    if completed_trial_failed:
        prior_trial_outcome = {
            "candidate_id": existing_trial.get("candidate_id"),
            "type": trial_type,
            "status": trial_status,
            "cycles_run": int(existing_trial.get("cycles_run", 0) or 0),
            "max_cycles": int(existing_trial.get("max_cycles", 0) or 0),
            "pnl_gap_vs_baseline": _f(existing_trial.get("pnl_gap_vs_baseline"), pnl_gap_vs_baseline),
            "assessment": latest_outcome_assessment,
        }

    if latest_outcome_assessment == "pass" and latest_outcome_type == "capital_preservation":
        candidate_type = "major_liquidity_expansion"
        priority = "medium"
        rationale = [
            "The capital-preservation lane stabilized paper behavior enough to allow a controlled expansion.",
            "The next bounded candidate should widen from majors-only into majors plus top-liquidity alts while keeping strict quality gates.",
        ]
        changes = [
            {"key": "entry_profile", "action": "expand", "value": "majors_plus_liquid_alts"},
            {"key": "symbol_universe", "action": "allow", "value": "major_and_top_alt_tier"},
            {"key": "allowed_long_setups", "action": "restrict", "value": ["trend_long", "breakout_long"]},
            {"key": "generic_long_policy", "action": "tighten", "value": "require_top_quality"},
            {"key": "risk_mult_cap", "action": "cap", "value": round(min(max(avg_risk, 0.45), 0.55), 2)},
            {"key": "max_fakeout_risk", "action": "cap", "value": 0.65},
            {"key": "max_long_positions", "action": "set", "value": 2},
        ]
        validation = [
            "Only majors and top-liquidity alts should be eligible in the next bounded trial.",
            "Long entries should stay restricted to trend_long and breakout_long instead of broad generic expansion.",
            "No promotion is allowed until the expansion lane beats the preservation baseline cleanly.",
        ]
        blockers = []

    if latest_outcome_assessment in {"fail", "insufficient"} and latest_outcome_hint:
        candidate_type = latest_outcome_hint
        priority = "high"
        rationale = [
            f"The previous {latest_outcome_type or 'recovery'} trial finished with assessment '{latest_outcome_assessment}'.",
            f"The next bounded recovery class should move to {latest_outcome_hint} instead of recycling the same idea.",
        ]
        if latest_outcome_hint == "setup_quality_recovery":
            changes = [
                {"key": "entry_profile", "action": "prefer", "value": "majors_only"},
                {"key": "allowed_long_setups", "action": "restrict", "value": list(SETUP_QUALITY_ALLOWED_LONG_SETUPS)},
                {"key": "entry_selection", "action": "require", "value": "require_regime_support_and_labeled_setup_alignment"},
                {"key": "generic_long_policy", "action": "tighten", "value": "require_labeled_setup_and_top_quality"},
                {"key": "max_fakeout_risk", "action": "cap", "value": 0.60},
                {"key": "max_long_positions", "action": "set", "value": 1},
                {"key": "max_short_positions", "action": "set", "value": 1},
                {"key": "allow_short_entries_in_adverse_regime", "action": "enable", "value": True},
                {"key": "trial_short_threshold_relief", "action": "set", "value": 0.15},
                {"key": "candidate_focus", "action": "review", "value": "setup_tags_and_failed_symbols"},
            ]
            if setup_quality_allowlist:
                changes.insert(
                    1,
                    {"key": "allowed_long_symbols", "action": "allow", "value": setup_quality_allowlist},
                )
            validation = [
                "Candidate should reduce or eliminate unlabeled long entries while preserving the strongest short setups.",
                "Candidate should improve expectancy by focusing on labeled long setups and explicitly learned recovery symbols.",
                "Do not promote this candidate until durable history posture improves from degraded.",
            ]
            highlighted = ", ".join(setup_quality_allowlist[:3])
            if latest_outcome_type == "major_liquidity_expansion" and highlighted:
                rationale = [
                    "The expansion lane failed, but the live scan still exposed labeled long setups that were blocked by overly defensive major-only scope.",
                    f"Setup-quality recovery should preserve the defensive posture while explicitly testing learned labeled-long recovery symbols: {highlighted}.",
                ]
            elif highlighted:
                rationale = [
                    f"The next bounded recovery class should focus on labeled setup quality while explicitly probing learned recovery symbols: {highlighted}.",
                    "This keeps QuantForge defensive on broad non-major exposure while still testing the blocked long slices that may carry edge.",
                ]
            else:
                rationale = [
                    f"The previous {latest_outcome_type or 'recovery'} trial finished with assessment '{latest_outcome_assessment}'.",
                    "The next bounded recovery class should focus on labeled setup quality instead of broad non-major exposure or generic-long churn.",
                ]
        elif latest_outcome_hint == "regime_sensitive_retraining":
            changes = [
                {"key": "retrain_focus", "action": "prioritize", "value": "entropy_regime_weighting"},
                {"key": "candidate_focus", "action": "review", "value": "entropy_regime_breakdowns"},
                {"key": "chaotic_regime_policy", "action": "tighten", "value": "reject_generic_long_and_reduce_size"},
                {"key": "regime_sensitive_thresholds", "action": "enable", "value": True},
            ]
            validation = [
                "Candidate should improve performance by separating orderly and chaotic regimes in both training and paper behavior.",
                "Entropy-aware recovery should reduce weak entries during mixed or chaotic conditions.",
            ]
        elif latest_outcome_hint == "model_recalibration":
            changes = [
                {"key": "retrain_focus", "action": "prioritize", "value": "threshold_and_probability_recalibration"},
                {"key": "candidate_focus", "action": "review", "value": "confidence_calibration_and_holdout_expectancy"},
                {"key": "long_confidence_floor", "action": "raise", "value": 0.86},
                {"key": "promotion_gate", "action": "tighten", "value": "require_supportive_paper_and_stable_history"},
            ]
            validation = [
                "Candidate should improve calibration rather than only filtering entries.",
                "Model recalibration should create more trustworthy probability-to-trade decisions.",
            ]
        elif latest_outcome_hint == "quantforge_redesign":
            changes = [
                {"key": "retrain_focus", "action": "redesign", "value": "setup_labels_targets_and_regime_feature_stack"},
                {"key": "candidate_focus", "action": "split", "value": "prediction_vs_risk_filter_vs_execution_policy"},
                {"key": "training_targets", "action": "tighten", "value": "cleaner_outcome_labels_and_holdout_expectancy"},
                {"key": "setup_labeling", "action": "expand", "value": "replace_unlabeled_with_explicit_setup_classes"},
                {"key": "regime_features", "action": "elevate", "value": "entropy_trend_volatility_as_first_class_inputs"},
                {"key": "entry_selection", "action": "tighten", "value": "require_regime_support_and_labeled_setup_alignment"},
            ]
            validation = [
                "Redesign candidate should be treated as a model/data architecture change, not a threshold tweak.",
                "Prediction, risk filter, and execution policy should become explicitly separable in the paper path.",
                "Do not promote the redesign candidate until it beats baseline in a bounded paper-only trial.",
            ]
        elif latest_outcome_hint == "competitiveness_gap_rebuild":
            changes = [
                {"key": "data_lane", "action": "upgrade", "value": "richer_market_data_and_microstructure_features"},
                {"key": "execution_lane", "action": "upgrade", "value": "spread_slippage_latency_realism"},
                {"key": "model_architecture", "action": "split", "value": "prediction_vs_regime_vs_risk_vs_execution"},
                {"key": "strategy_scope", "action": "narrow", "value": "slower_high_conviction_majors_only"},
                {"key": "backtest_policy", "action": "tighten", "value": "require_execution_realism_and_subgroup_eval"},
            ]
            validation = [
                "This candidate should be treated as a competitiveness-gap rebuild, not another threshold tweak.",
                "Execution assumptions must become more realistic before any claim of edge is accepted.",
                "The next paper-only lane should only continue if the rebuilt stack beats baseline with cleaner subgroup evidence.",
            ]
            rationale = [
                "Repeated bounded recovery candidates have failed to create enough edge against baseline.",
                "The next step is to address the competitiveness gap directly: data quality, execution realism, model layering, and strategy scope.",
            ]
        elif latest_outcome_hint == "quantforge_research_hold":
            changes = [
                {"key": "new_entries", "action": "disable", "value": True},
                {"key": "mode", "action": "set", "value": "research_only"},
                {"key": "rebuild_scope", "action": "prioritize", "value": "labels_targets_features_execution_data"},
                {"key": "candidate_focus", "action": "review", "value": "deep_rebuild_before_next_trial"},
            ]
            validation = [
                "Do not start another bounded paper trial until the deeper data/label rebuild has produced a materially different candidate.",
                "QuantForge should remain flat or paper-constrained while rebuild work is underway.",
            ]
            rationale = [
                "The competitiveness-gap rebuild also failed to beat baseline.",
                "QuantForge should now stop rotating through shallow recovery classes and enter an explicit rebuild/research hold.",
            ]
        elif latest_outcome_hint == "capital_preservation":
            changes = [
                {"key": "risk_mult_cap", "action": "cap", "value": round(min(avg_risk, 0.35), 2)},
                {"key": "max_long_positions", "action": "reduce", "value": 1},
                {"key": "non_major_entries", "action": "disable", "value": True},
            ]
            validation = [
                "Candidate should halt further degradation while the next training-quality pass is prepared.",
            ]
        blockers = []

    if latest_outcome_assessment == "fail" and {"quantforge_redesign", "model_recalibration"}.issubset(set(recent_failed_types)):
        candidate_type = "competitiveness_gap_rebuild"
        priority = "high"
        rationale = [
            "QuantForge has now failed through both model recalibration and redesign-grade recovery attempts.",
            "The next candidate should address the competitiveness gap directly instead of recycling strategy-local patches.",
        ]
        changes = [
            {"key": "data_lane", "action": "upgrade", "value": "richer_market_data_and_microstructure_features"},
            {"key": "execution_lane", "action": "upgrade", "value": "spread_slippage_latency_realism"},
            {"key": "model_architecture", "action": "split", "value": "prediction_vs_regime_vs_risk_vs_execution"},
            {"key": "strategy_scope", "action": "narrow", "value": "slower_high_conviction_majors_only"},
            {"key": "candidate_focus", "action": "review", "value": "competitiveness_gap_and_subgroup_edge"},
        ]
        validation = [
            "Do not return to shallow recovery classes until the competitiveness-gap rebuild has been evaluated.",
            "The rebuilt lane should explicitly test whether slower, more selective scope beats the current broad crypto posture.",
        ]

    if latest_outcome_assessment == "fail" and latest_outcome_type == "competitiveness_gap_rebuild":
        candidate_type = "quantforge_research_hold"
        priority = "critical"
        rationale = [
            "QuantForge has now failed through a competitiveness-gap rebuild without beating baseline.",
            "The system should freeze new recovery-class rotations and move into explicit research/rebuild mode.",
        ]
        changes = [
            {"key": "new_entries", "action": "disable", "value": True},
            {"key": "mode", "action": "set", "value": "research_only"},
            {"key": "rebuild_scope", "action": "prioritize", "value": "labels_targets_features_execution_data"},
            {"key": "candidate_focus", "action": "review", "value": "deep_rebuild_before_next_trial"},
        ]
        validation = [
            "Do not queue another bounded paper candidate until deeper rebuild artifacts exist.",
            "Use the research core for diagnosis, data design, and execution realism work rather than live trial churn.",
        ]

    if (
        candidate_type == "quantforge_research_hold"
        and model_layer_status == "ready_for_layered_trial"
        and model_layer_next_step == "prepare_layered_trial_candidate"
        and not (
            latest_outcome_type == "quantforge_layered_trial"
            and latest_outcome_assessment == "fail"
        )
    ):
        candidate_type = "quantforge_layered_trial"
        priority = "high"
        rationale = [
            "The rebuild lane has produced a model-layer report that is ready for a bounded layered trial.",
            "QuantForge should now move from generic research hold into a stricter paper-only layered-trial candidate.",
        ]
        changes = [
            {"key": "trial_profile", "action": "set", "value": "layered"},
            {"key": "entry_selection", "action": "require", "value": "regime_support_and_labeled_setup_alignment"},
            {"key": "execution_policy", "action": "enforce", "value": "rebuild_realism_aligned"},
            {"key": "candidate_focus", "action": "review", "value": "prediction_regime_risk_execution_contract"},
            {"key": "strategy_scope", "action": "narrow", "value": "major_symbols_and_positive_holdout_slices"},
        ]
        validation = [
            "Run the next candidate only as a bounded paper-only layered trial.",
            "Holdout-positive slices should remain favored, but no promotion is allowed until the layered trial beats baseline.",
            "If the layered trial fails, return to research hold instead of recycling shallow recovery classes.",
        ]
        blockers = []

    top_alt_support = None
    if candidate_type == "quantforge_research_hold":
        top_alt_support = summarize_top_alt_research_hold_support()
        if bool(top_alt_support.get("expansion_supported")):
            top_non_majors = top_alt_support.get("top_non_major_symbols") or []
            highlighted = [str(row.get("symbol") or "") for row in top_non_majors[:3] if row.get("symbol")]
            highlighted_text = ", ".join(highlighted) or "top-liquidity alts"
            candidate_type = "major_liquidity_expansion"
            priority = "high"
            rationale = [
                "Research-hold rebuild evidence now shows top-liquidity non-major long support beating the current major-only survivors.",
                f"The next bounded candidate should reopen a narrow majors-plus-liquidity expansion centered on {highlighted_text} while keeping strict quality gates.",
            ]
            changes = [
                {"key": "entry_profile", "action": "expand", "value": "majors_plus_liquid_alts"},
                {"key": "symbol_universe", "action": "allow", "value": "major_and_top_alt_tier"},
                {"key": "allowed_long_setups", "action": "restrict", "value": top_alt_support.get("allowed_long_setups") or ["trend_long", "breakout_long"]},
                {"key": "generic_long_policy", "action": "tighten", "value": "require_top_quality"},
                {"key": "max_fakeout_risk", "action": "cap", "value": 0.65},
                {"key": "candidate_focus", "action": "review", "value": "top_alt_research_hold_long_slices"},
                {"key": "risk_mult_cap", "action": "cap", "value": round(min(max(avg_risk, 0.45), 0.55), 2)},
                {"key": "max_long_positions", "action": "set", "value": 2},
            ]
            validation = [
                "Only majors and top-liquidity alts should survive the next bounded paper-only lane.",
                "Long entries should stay restricted to trend_long and breakout_long with top-quality filters.",
                "No promotion is allowed until the expansion lane beats baseline with cleaner subgroup evidence.",
            ]
            blockers = []

    sticky_trial = (
        candidate_type != "none"
        and existing_trial
        and trial_status in {"queued", "active"}
        and str(existing_trial.get("type", "")) == candidate_type
    )
    sticky_candidate = (
        candidate_type != "none"
        and existing_candidate
        and existing_candidate_type == candidate_type
        and existing_candidate_status in {"proposed", "queued", "active"}
    )

    if sticky_trial:
        candidate_id = str(existing_trial.get("candidate_id"))
    elif sticky_candidate:
        candidate_id = str(existing_candidate.get("candidate_id"))
    else:
        candidate_id = f"{candidate_type}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "priority": priority,
        "type": candidate_type,
        "status": "proposed" if candidate_type != "none" else "observe",
        "rationale": rationale,
        "changes": changes,
        "validation": validation,
        "blockers": blockers,
        "evidence": {
            "durable_posture": posture,
            "persistent_review": persistent_review,
            "persistent_drift": persistent_drift,
            "avg_adaptive_risk": avg_risk,
            "paper_total_pnl_pct": round(total_pnl_pct, 4),
            "baseline_pnl_pct": round(baseline_pnl_pct, 4),
            "candidate_pnl_pct": round(candidate_pnl_pct, 4),
            "pnl_gap_vs_baseline": pnl_gap_vs_baseline,
            "recent_close_win_rate": round(recent_win_rate, 4),
            "unlabeled_setup_count": unlabeled_count,
            "diagnosis_causes": sorted(causes),
            "drift_flags": drift_flags,
            "regime_label": regime.get("label"),
            "regime_entropy_label": regime.get("entropy_label"),
            "regime_entropy": round(_f(regime.get("entropy")), 4),
            "latest_outcome_type": latest_outcome_type,
            "latest_outcome_assessment": latest_outcome_assessment,
            "latest_outcome_hint": latest_outcome_hint,
            "stale_latest_outcome_ignored": stale_latest_outcome,
            "model_layer_status": model_layer_status,
            "model_layer_next_step": model_layer_next_step,
            "top_alt_research_hold_support": top_alt_support,
            "setup_quality_recovery_symbols": setup_quality_allowlist,
        },
    }
    if prior_trial_outcome:
        payload["prior_trial_outcome"] = prior_trial_outcome
    if sticky_trial:
        payload["status"] = "proposed"
        payload["sticky_reason"] = "preserve_existing_trial_until_completed_or_expired"
    elif sticky_candidate:
        payload["sticky_reason"] = "preserve_existing_candidate_until_trial_state_changes"
    return payload


def main():
    payload = build_candidate()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge recovery candidate")
    print(f"Type: {payload['type']}")
    print(f"Priority: {payload['priority']}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
