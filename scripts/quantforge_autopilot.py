#!/usr/bin/env python3
"""QuantForge — bounded autopilot decision artifact.

Combines governance, promotion, diagnosis, and experiment-lane state into a
single recommendation that other QuantForge loops can consume safely.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
PROMOTION_FILE = os.path.join(BASE_DIR, "model", "promotion_report.json")
DIAGNOSIS_FILE = os.path.join(BASE_DIR, "diagnosis-report.json")
LANES_FILE = os.path.join(BASE_DIR, "experiment-lanes.json")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history-summary.json")
MONITOR_FILE = os.path.join(BASE_DIR, "monitor-report.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "autopilot-report.json")
AGI_OPERATOR_HISTORY_FILE = os.path.join(cfg.data, "agi-operator-history-summary.json")
CANDIDATE_RECOVERY_FILE = os.path.join(BASE_DIR, "candidate-recovery.json")
CANDIDATE_OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
HARNESS_FILE = os.path.join(BASE_DIR, "harness-report.json")
MAX_PORTFOLIO_AGE_HOURS = 8
MAX_LAST_SCAN_AGE_HOURS = 8
MAX_GOVERNANCE_AGE_HOURS = 8
MAX_MONITOR_AGE_HOURS = 8


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


def _age_hours(value):
    dt = _parse_ts(value)
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def _outcome_is_stale_for_candidate(latest_outcome: dict, candidate_lane: dict) -> bool:
    if not isinstance(latest_outcome, dict) or not isinstance(candidate_lane, dict):
        return False
    outcome_dt = _parse_ts(latest_outcome.get("recorded_at") or latest_outcome.get("completed_at"))
    candidate_dt = _parse_ts(candidate_lane.get("model_trained_at"))
    if outcome_dt is None or candidate_dt is None:
        return False
    return outcome_dt < candidate_dt


def build_report():
    governance = read_json(GOVERNANCE_FILE)
    promotion = read_json(PROMOTION_FILE)
    diagnosis = read_json(DIAGNOSIS_FILE)
    lanes = read_json(LANES_FILE)
    portfolio = read_json(PORTFOLIO_FILE)
    history = read_json(HISTORY_FILE)
    monitor = read_json(MONITOR_FILE)
    agi_operator_history = read_json(AGI_OPERATOR_HISTORY_FILE)
    recovery_candidate = read_json(CANDIDATE_RECOVERY_FILE)
    candidate_outcomes = read_json(CANDIDATE_OUTCOMES_FILE)
    harness = read_json(HARNESS_FILE)

    candidate = lanes.get("candidate") or {}
    baseline = lanes.get("baseline") or {}
    candidate_trial = lanes.get("candidate_trial") or {}
    latest_trial_outcome = (candidate_outcomes.get("latest") or {}) if isinstance(candidate_outcomes, dict) else {}
    if (
        str(candidate_trial.get("status", "") or "").lower() == "completed"
        and str(candidate_trial.get("assessment", "") or "")
    ):
        latest_trial_outcome = {
            "candidate_id": candidate_trial.get("candidate_id"),
            "type": candidate_trial.get("type"),
            "assessment": candidate_trial.get("assessment"),
            "next_candidate_hint": candidate_trial.get("next_candidate_hint"),
            "pnl_gap_vs_baseline": candidate_trial.get("pnl_gap_vs_baseline"),
        }
    stale_latest_outcome = _outcome_is_stale_for_candidate(latest_trial_outcome, candidate)

    candidate_pnl = _f(candidate.get("paper_total_pnl_pct"))
    baseline_pnl = _f(baseline.get("paper_total_pnl_pct"))
    pnl_gap = candidate_pnl - baseline_pnl

    governance_reco = governance.get("recommendation")
    promotion_decision = promotion.get("overall_decision")
    causes = set(diagnosis.get("causes", []))
    open_positions = len((portfolio.get("positions") or {}))
    regime = monitor.get("regime") or {}
    regime_entropy_label = str(regime.get("entropy_label", "")).upper()
    actions = []
    reasons = []
    mode = "observe"
    rollback_trigger = None
    stale_inputs = []
    freshness_checks = [
        ("portfolio", portfolio.get("updated"), MAX_PORTFOLIO_AGE_HOURS),
        ("governance", governance.get("generated_at"), MAX_GOVERNANCE_AGE_HOURS),
        ("monitor", monitor.get("generated_at"), MAX_MONITOR_AGE_HOURS),
        ("last_scan", (history.get("window") or {}).get("latest") if history.get("status") == "ok" else None, 96),
    ]
    last_scan_ts = ((read_json(os.path.join(BASE_DIR, "last_scan.json"))).get("ts"))
    freshness_checks[-1] = ("last_scan", last_scan_ts, MAX_LAST_SCAN_AGE_HOURS)
    for label, value, max_age in freshness_checks:
        age = _age_hours(value)
        if age is None or age > max_age:
            stale_inputs.append(f"{label} age {age:.1f}h" if age is not None else f"{label} missing timestamp")

    if promotion_decision == "PROMOTE_CANDIDATE" and governance_reco in {"HOLD", "OBSERVE"}:
        mode = "promote_candidate"
        reasons.append("Promotion report and governance both support keeping the candidate active.")
    elif governance_reco == "DEMOTE":
        mode = "rollback_to_baseline"
        rollback_trigger = "governance_demote"
        reasons.append("Governance explicitly recommends demotion.")
    elif promotion_decision == "DO_NOT_PROMOTE":
        mode = "hold_in_paper"
        reasons.append("Promotion report blocks wider rollout.")
    elif promotion_decision == "DEMOTE_CANDIDATE":
        mode = "rollback_to_baseline"
        rollback_trigger = "promotion_demote_candidate"
        reasons.append("Promotion report indicates candidate should be demoted.")
    elif governance_reco == "REVIEW":
        mode = "review_required"
        reasons.append("Governance requires review before any wider rollout.")

    if open_positions > 0 and (
        governance_reco == "REVIEW" or "paper_underperformance" in causes or "weak_recent_close_quality" in causes
    ):
        mode = "pause_new_entries"
        reasons.append("Keep managing open positions, but pause fresh entries while performance is degraded.")

    if history.get("status") == "ok" and int(history.get("cycles_sampled", 0) or 0) >= 6:
        posture = history.get("posture")
        if posture == "degraded":
            mode = "pause_new_entries" if open_positions > 0 else "hold_in_paper"
            actions.append("respect_durable_history")
            reasons.append("Durable operator history remains degraded, so keep QuantForge constrained.")
        elif posture == "recovery_watch" and mode == "observe":
            mode = "hold_in_paper"
            actions.append("respect_durable_history")
            reasons.append("Durable operator history is still in recovery-watch mode.")

    if agi_operator_history.get("status") == "ok":
        if agi_operator_history.get("persistent_drift"):
            actions.append("respect_company_history")
            actions.append("keep_risk_throttled")
            reasons.append("Operator history shows persistent drifting/watch state.")
        if agi_operator_history.get("persistent_review"):
            actions.append("hold_promotion")
            if mode == "observe":
                mode = "review_required"
            reasons.append("Operator history shows repeated governance review/demote state.")
        avg_risk = _f((agi_operator_history.get("averages") or {}).get("adaptive_risk_mult"), 1.0)
        if avg_risk < 0.75:
            actions.append("favor_defensive_entries")
            reasons.append(f"Recent durable operator history keeps average adaptive risk low at {avg_risk:.2f}.")

    if recovery_candidate.get("status") == "proposed":
        actions.append("prepare_candidate_experiment")
        reasons.append(
            f"Recovery candidate {recovery_candidate.get('type', 'unknown')} is proposed "
            f"with priority {recovery_candidate.get('priority', 'unknown')}."
        )
    if str(recovery_candidate.get("type", "") or "") == "quantforge_research_hold":
        mode = "pause_new_entries"
        actions.append("freeze_for_rebuild")
        reasons.append("QuantForge is in research/rebuild hold after repeated failed recovery classes.")
    if str(recovery_candidate.get("type", "") or "") == "quantforge_layered_trial":
        actions.append("prepare_layered_trial")
        reasons.append("QuantForge has a bounded layered-trial candidate ready from the rebuild lane.")
    if str(harness.get("status", "")).lower() == "warn":
        actions.append("respect_harness")
        if mode not in {"rollback_to_baseline", "pause_new_entries"}:
            mode = "hold_in_paper"
        reasons.append("Harness guardrails found a candidate/trial mismatch that should be resolved before wider progression.")

    trial_status = str(candidate_trial.get("status", "")).lower()
    queued_or_active_paper_trial = trial_status in {"queued", "active"} and bool(candidate_trial.get("paper_only", False))
    if trial_status == "active":
        mode = "run_candidate_paper_trial"
        actions.append("respect_candidate_trial")
        reasons.append(
            f"Candidate trial {candidate_trial.get('candidate_id', 'unknown')} is active and must stay paper-only."
        )
    elif trial_status == "queued":
        if open_positions > 0:
            mode = "pause_new_entries"
            actions.append("respect_candidate_trial")
            reasons.append(
                f"Candidate trial {candidate_trial.get('candidate_id', 'unknown')} is queued; "
                "hold the current paper book steady until the bounded paper-only trial can start."
            )
        else:
            mode = "run_candidate_paper_trial"
            actions.append("activate_candidate_trial")
            reasons.append(
                f"Candidate trial {candidate_trial.get('candidate_id', 'unknown')} is queued and can start in paper-only mode."
            )
    elif trial_status == "completed":
        completed_assessment = str(candidate_trial.get("assessment", "") or "").lower()
        if completed_assessment == "fail":
            mode = "pause_new_entries"
            actions.append("freeze_for_rebuild")
            reasons.append(
                f"Candidate trial {candidate_trial.get('candidate_id', 'unknown')} completed with a fail assessment."
            )
        elif completed_assessment == "insufficient":
            mode = "hold_in_paper"
            actions.append("rotate_candidate_class")
            reasons.append(
                f"Candidate trial {candidate_trial.get('candidate_id', 'unknown')} completed without enough recovery evidence."
            )
        elif completed_assessment == "pass":
            mode = "hold_in_paper"
            actions.append("advance_candidate")
            reasons.append(
                f"Candidate trial {candidate_trial.get('candidate_id', 'unknown')} completed and passed baseline checks."
            )

    if "paper_underperformance" in causes:
        actions.append("tighten_live_selection")
        if candidate_pnl <= -5.0:
            rollback_trigger = rollback_trigger or "paper_underperformance"
    if "negative_recent_trade_expectancy" in causes:
        actions.append("keep_risk_throttled")
    if "chaotic_market_regime" in causes or regime_entropy_label == "CHAOTIC":
        actions.append("respect_entropy_regime")
        if open_positions > 0:
            mode = "pause_new_entries"
        elif mode == "observe":
            mode = "hold_in_paper"
        reasons.append("Entropy regime is chaotic, so new entries stay defensive.")
    if "model_not_promotion_ready" in causes:
        actions.append("retrain_candidate")
    if "weak_recent_close_quality" in causes:
        actions.append("quarantine_recent_losers")

    if pnl_gap <= -2.0 and baseline and not queued_or_active_paper_trial:
        mode = "rollback_to_baseline"
        rollback_trigger = rollback_trigger or "baseline_outperforming_candidate"
        reasons.append(f"Candidate trails baseline by {abs(pnl_gap):.2f} percentage points.")
    elif pnl_gap >= 2.0 and promotion_decision != "DO_NOT_PROMOTE":
        reasons.append(f"Candidate leads baseline by {pnl_gap:.2f} percentage points.")

    if (
        str(latest_trial_outcome.get("type", "") or "") == "competitiveness_gap_rebuild"
        and str(latest_trial_outcome.get("assessment", "") or "").lower() == "fail"
        and trial_status != "active"
        and not queued_or_active_paper_trial
        and str(recovery_candidate.get("type", "") or "") in {"", "quantforge_research_hold", "quantforge_layered_trial"}
        and not stale_latest_outcome
    ):
        mode = "pause_new_entries"
        actions.append("freeze_for_rebuild")
        reasons.append("Latest competitiveness-gap rebuild failed, so QuantForge should stay flat while the deeper rebuild is prepared.")
    if (
        str(latest_trial_outcome.get("type", "") or "") == "quantforge_layered_trial"
        and str(latest_trial_outcome.get("assessment", "") or "").lower() == "fail"
        and trial_status != "active"
        and not queued_or_active_paper_trial
        and str(recovery_candidate.get("type", "") or "") in {"", "quantforge_research_hold", "quantforge_layered_trial"}
        and not stale_latest_outcome
    ):
        mode = "pause_new_entries"
        actions.append("freeze_for_rebuild")
        reasons.append("Latest layered trial failed, so QuantForge should return to research/rebuild hold instead of lingering in trial mode.")

    # A queued or active paper-only recovery trial should keep control of the
    # execution mode unless stale inputs force a safer pause. Otherwise the
    # generic rollback path can permanently strand the trial in "queued".
    if queued_or_active_paper_trial and not stale_inputs:
        if trial_status == "active":
            mode = "run_candidate_paper_trial"
        elif open_positions > 0:
            mode = "pause_new_entries"
        else:
            mode = "run_candidate_paper_trial"

    if not actions:
        actions.append("observe")
    if stale_inputs:
        mode = "pause_new_entries"
        actions.append("respect_stale_inputs")
        reasons.append("Stale QuantForge control artifacts detected: " + ", ".join(stale_inputs))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "reasons": reasons,
        "actions": sorted(set(actions)),
        "stale_inputs": stale_inputs,
        "rollback_trigger": rollback_trigger,
        "baseline": {
            "label": baseline.get("label"),
            "model_trained_at": baseline.get("model_trained_at"),
            "paper_total_pnl_pct": baseline_pnl,
        },
        "candidate": {
            "label": candidate.get("label"),
            "model_trained_at": candidate.get("model_trained_at"),
            "paper_total_pnl_pct": candidate_pnl,
            "pnl_gap_vs_baseline": round(pnl_gap, 4),
        },
        "paper_state": {
            "open_positions": open_positions,
        },
        "regime": regime,
        "durable_history": {
            "status": history.get("status"),
            "cycles_sampled": int(history.get("cycles_sampled", 0) or 0),
            "posture": history.get("posture"),
            "averages": history.get("averages", {}),
            "ratios": history.get("ratios", {}),
        },
        "agi_operator_history": {
            "status": agi_operator_history.get("status"),
            "cycles_sampled": int(agi_operator_history.get("cycles_sampled", 0) or 0),
            "posture": agi_operator_history.get("posture"),
            "persistent_drift": bool(agi_operator_history.get("persistent_drift")),
            "persistent_review": bool(agi_operator_history.get("persistent_review")),
            "averages": agi_operator_history.get("averages", {}),
        },
        "recovery_candidate": {
            "candidate_id": recovery_candidate.get("candidate_id"),
            "type": recovery_candidate.get("type"),
            "priority": recovery_candidate.get("priority"),
            "status": recovery_candidate.get("status"),
            "changes": recovery_candidate.get("changes", []),
        },
        "candidate_trial": {
            "candidate_id": candidate_trial.get("candidate_id"),
            "type": candidate_trial.get("type"),
            "priority": candidate_trial.get("priority"),
            "status": candidate_trial.get("status"),
            "paper_only": bool(candidate_trial.get("paper_only", False)),
            "cycles_run": int(candidate_trial.get("cycles_run", 0) or 0),
            "max_cycles": int(candidate_trial.get("max_cycles", 0) or 0),
            "expires_at": candidate_trial.get("expires_at"),
            "changes": candidate_trial.get("changes", []),
        },
        "latest_trial_outcome": {
            "candidate_id": latest_trial_outcome.get("candidate_id"),
            "type": latest_trial_outcome.get("type"),
            "assessment": latest_trial_outcome.get("assessment"),
            "next_candidate_hint": latest_trial_outcome.get("next_candidate_hint"),
            "pnl_gap_vs_baseline": latest_trial_outcome.get("pnl_gap_vs_baseline"),
            "recorded_at": latest_trial_outcome.get("recorded_at"),
            "stale_for_candidate": stale_latest_outcome,
        },
        "harness": {
            "status": harness.get("status"),
            "recommendation": harness.get("recommendation"),
            "failed_count": int(harness.get("failed_count", 0) or 0),
            "summary": harness.get("summary", {}),
        },
        "inputs": {
            "governance_recommendation": governance_reco,
            "promotion_decision": promotion_decision,
            "diagnosis_causes": sorted(causes),
        },
    }


def main():
    cfg.require_production_runtime("quantforge_autopilot.py")
    report = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print("QuantForge autopilot")
    print(f"Mode: {report['mode']}")
    for reason in report["reasons"]:
        print(f"  - {reason}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
