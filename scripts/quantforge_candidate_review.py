#!/usr/bin/env python3
"""QuantForge — bounded reviewer artifact for recovery candidates and trial outcomes."""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
AUTOPILOT_FILE = os.path.join(BASE_DIR, "autopilot-report.json")
LANES_FILE = os.path.join(BASE_DIR, "experiment-lanes.json")
CANDIDATE_RECOVERY_FILE = os.path.join(BASE_DIR, "candidate-recovery.json")
CANDIDATE_OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "candidate-review.json")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_ts(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _outcome_is_stale_for_candidate(latest_outcome: dict, candidate_lane: dict) -> bool:
    if not isinstance(latest_outcome, dict) or not isinstance(candidate_lane, dict):
        return False
    outcome_dt = _parse_ts(latest_outcome.get("recorded_at") or latest_outcome.get("completed_at"))
    candidate_dt = _parse_ts(candidate_lane.get("model_trained_at"))
    if outcome_dt is None or candidate_dt is None:
        return False
    return outcome_dt < candidate_dt


def _f(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _active_trial_surface_summary(last_scan: dict, trial: dict) -> dict | None:
    if not isinstance(last_scan, dict) or not isinstance(trial, dict):
        return None
    if str(trial.get("status", "") or "").lower() != "active":
        return None
    trial_type = str(trial.get("type", "") or "")
    if trial_type not in {"major_liquidity_expansion", "setup_quality_recovery"}:
        return None
    flow = last_scan.get("flow") or {}
    rows = last_scan.get("results") or []
    long_skips = []
    blocked_labeled_longs = []
    strongest_long_hold = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason", "") or "")
        setup_tag = str(row.get("setup_tag", "") or "")
        if row.get("status") == "skip" and (setup_tag.endswith("_long") or "long" in reason.lower()):
            long_skips.append({
                "symbol": row.get("symbol"),
                "setup_tag": setup_tag,
                "reason": reason,
            })
            if (
                trial_type == "setup_quality_recovery"
                and setup_tag in {"trend_long", "breakout_long"}
                and "restricts longs to major-liquidity symbols" in reason
            ):
                blocked_labeled_longs.append({
                    "symbol": row.get("symbol"),
                    "setup_tag": setup_tag,
                    "reason": reason,
                })
        if row.get("status") == "hold":
            long_conf = _f(row.get("long_confidence"))
            short_conf = _f(row.get("short_confidence"))
            if long_conf < short_conf:
                continue
            if strongest_long_hold is None or long_conf > _f(strongest_long_hold.get("long_confidence")):
                strongest_long_hold = {
                    "symbol": row.get("symbol"),
                    "setup_tag": setup_tag,
                    "long_confidence": round(long_conf, 4),
                    "short_confidence": round(short_conf, 4),
                    "reason": reason,
                }
    buy_signals = int(flow.get("buy_signals", 0) or 0)
    sell_signals = int(flow.get("sell_signals", 0) or 0)
    threshold_miss = int(flow.get("threshold_miss", 0) or 0)
    selection_blocked = int(flow.get("selection_blocked", 0) or 0)
    no_target_long_surface = bool(
        buy_signals == 0
        and threshold_miss >= 10
        and strongest_long_hold is not None
        and _f(strongest_long_hold.get("long_confidence")) < 0.05
        and (sell_signals > 0 or selection_blocked > 0 or bool(long_skips))
    )
    setup_quality_scope_blocked = bool(
        trial_type == "setup_quality_recovery"
        and buy_signals == 0
        and bool(blocked_labeled_longs)
    )
    return {
        "scan_ts": last_scan.get("ts"),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "threshold_miss": threshold_miss,
        "selection_blocked": selection_blocked,
        "long_filtered_skips": long_skips[:5],
        "blocked_labeled_longs": blocked_labeled_longs[:5],
        "strongest_long_hold": strongest_long_hold,
        "no_target_long_surface": no_target_long_surface,
        "setup_quality_scope_blocked": setup_quality_scope_blocked,
    }


def build_review():
    autopilot = read_json(AUTOPILOT_FILE)
    lanes = read_json(LANES_FILE)
    recovery = read_json(CANDIDATE_RECOVERY_FILE)
    outcomes = read_json(CANDIDATE_OUTCOMES_FILE)
    last_scan = read_json(LAST_SCAN_FILE)

    trial = (lanes.get("candidate_trial") or {}) if isinstance(lanes, dict) else {}
    latest = (outcomes.get("latest") or {}) if isinstance(outcomes, dict) else {}
    candidate = (lanes.get("candidate") or {}) if isinstance(lanes, dict) else {}
    baseline = (lanes.get("baseline") or {}) if isinstance(lanes, dict) else {}

    trial_status = str(trial.get("status", "") or "").lower()
    trial_assessment = str(trial.get("assessment", "") or "").lower()
    assessment = str(latest.get("assessment", "") or "").lower()
    next_hint = str(latest.get("next_candidate_hint", "") or "")
    if _outcome_is_stale_for_candidate(latest, candidate):
        assessment = ""
        next_hint = ""
    if trial_status == "completed" and trial_assessment:
        assessment = trial_assessment
        if not next_hint:
            next_hint = str(trial.get("next_candidate_hint", "") or "")
    mode = str(autopilot.get("mode", "") or "")
    pnl_gap = _f(latest.get("pnl_gap_vs_baseline"), _f(candidate.get("paper_total_pnl_pct")) - _f(baseline.get("paper_total_pnl_pct")))
    trial_surface = _active_trial_surface_summary(last_scan, trial)

    recommendation = "observe"
    confidence = 0.55
    reasons = []

    if trial_status == "active":
        recommendation = "hold_active_trial"
        confidence = 0.78
        reasons.append("A bounded paper-only trial is active and should finish before more live-logic changes.")
    elif trial_status == "queued":
        recommendation = "queue_candidate_trial"
        confidence = 0.82
        reasons.append("A bounded paper-only trial is queued and should be allowed to start before rotating to another recovery path.")
    elif trial_status == "completed" and assessment == "blocked":
        recommendation = "freeze_and_repair"
        confidence = 0.95
        reasons.append("The queued trial never started and was retired as blocked, so QuantForge needs operator repair before another trial is queued.")
    elif trial_status == "completed" and assessment == "pass":
        recommendation = "advance_candidate"
        confidence = 0.84
        reasons.append("The completed bounded trial passed against baseline and is eligible for a stronger validation gate.")
    elif trial_status == "completed" and assessment == "fail":
        recommendation = "retire_candidate_and_rotate"
        confidence = 0.9
        reasons.append("The completed bounded trial failed and should no longer be treated as the active recovery answer.")
        if str(trial.get("type", "") or "") == "quantforge_layered_trial":
            recommendation = "freeze_and_rebuild"
            confidence = 0.94
            reasons.append("The layered-trial lane failed, so QuantForge should return to research/rebuild hold instead of lingering in trial mode.")
        elif str(trial.get("type", "") or "") == "competitiveness_gap_rebuild":
            recommendation = "freeze_and_rebuild"
            confidence = 0.93
            reasons.append("A competitiveness-gap rebuild already failed, so the next move should be research/rebuild mode instead of another shallow rotation.")
    elif trial_status == "completed" and assessment == "insufficient":
        recommendation = "rotate_candidate_class"
        confidence = 0.76
        reasons.append("The completed bounded trial was inconclusive and needs a materially different candidate class.")
    elif trial_status == "queued" and str(trial.get("type", "") or "") == "quantforge_layered_trial":
        recommendation = "queue_candidate_trial"
        confidence = 0.86
        reasons.append("The layered trial is already queued and should be allowed to start as the next bounded paper-only candidate.")
    elif assessment == "pass":
        recommendation = "advance_candidate"
        confidence = 0.8
        reasons.append("The latest bounded trial passed against the baseline and is eligible for a stronger validation gate.")
    elif assessment == "fail":
        recommendation = "retire_candidate_and_rotate"
        confidence = 0.88
        reasons.append("The latest bounded trial failed and should be retired as the active recovery answer.")
        if str(latest.get("type", "") or "") == "competitiveness_gap_rebuild":
            recommendation = "freeze_and_rebuild"
            confidence = 0.93
            reasons.append("A competitiveness-gap rebuild already failed, so the next move should be research/rebuild mode instead of another shallow rotation.")
    elif assessment == "insufficient":
        recommendation = "rotate_candidate_class"
        confidence = 0.72
        reasons.append("The last bounded trial was inconclusive and needs a materially different candidate class.")
    elif str(recovery.get("type", "") or "") == "quantforge_layered_trial" and str(recovery.get("status", "") or "").lower() == "proposed":
        recommendation = "queue_candidate_trial"
        confidence = 0.84
        reasons.append("The rebuild lane is ready for a bounded layered trial and should be queued as the next paper-only candidate.")
    elif str(recovery.get("status", "")).lower() == "proposed":
        recommendation = "queue_candidate_trial"
        confidence = 0.66
        reasons.append("A recovery candidate is proposed and ready to be queued into a bounded paper-only trial.")
    else:
        reasons.append("No stronger candidate-review action was inferred from current evidence.")

    if next_hint and trial_status not in {"queued", "active"}:
        reasons.append(f"Latest durable outcome points next toward {next_hint}.")
    if pnl_gap:
        reasons.append(f"Latest candidate-vs-baseline gap is {pnl_gap:+.2f} points.")
    if mode:
        reasons.append(f"Current autopilot mode is {mode}.")
    stale_inputs = autopilot.get("stale_inputs") if isinstance(autopilot.get("stale_inputs"), list) else []
    if stale_inputs:
        recommendation = "freeze_and_repair"
        confidence = max(confidence, 0.96)
        reasons.append("Autopilot reports stale control artifacts: " + ", ".join(stale_inputs))
    if trial_surface and trial_surface.get("no_target_long_surface"):
        strongest = trial_surface.get("strongest_long_hold") or {}
        reasons.append(
            "The active expansion trial is currently not surfacing its target long edge: "
            f"0 buy signals, {int(trial_surface.get('sell_signals', 0) or 0)} sell signals, and strongest long hold "
            f"{strongest.get('symbol', 'unknown')} only reached {float(strongest.get('long_confidence', 0.0) or 0.0):.4f}."
        )
    if trial_surface and trial_surface.get("setup_quality_scope_blocked"):
        blocked = trial_surface.get("blocked_labeled_longs") or []
        highlighted = ", ".join(str(row.get("symbol") or "unknown") for row in blocked[:3]) or "blocked labeled longs"
        reasons.append(
            "The active setup-quality recovery trial is still blocking labeled long recovery symbols under a stale major-only scope: "
            f"{highlighted}."
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "confidence": round(confidence, 2),
        "current_mode": mode,
        "active_trial": {
            "candidate_id": trial.get("candidate_id"),
            "type": trial.get("type"),
            "status": trial.get("status"),
            "cycles_run": int(trial.get("cycles_run", 0) or 0),
            "max_cycles": int(trial.get("max_cycles", 0) or 0),
        },
        "latest_outcome": {
            "candidate_id": latest.get("candidate_id"),
            "type": latest.get("type"),
            "assessment": latest.get("assessment"),
            "next_candidate_hint": next_hint or None,
            "pnl_gap_vs_baseline": round(pnl_gap, 4),
        },
        "proposed_candidate": {
            "candidate_id": recovery.get("candidate_id"),
            "type": recovery.get("type"),
            "priority": recovery.get("priority"),
            "status": recovery.get("status"),
        },
        "active_trial_surface": trial_surface,
        "reasons": reasons,
    }


def main():
    cfg.require_production_runtime("quantforge_candidate_review.py")
    payload = build_review()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge candidate review")
    print(f"Recommendation: {payload['recommendation']}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
