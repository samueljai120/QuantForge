#!/usr/bin/env python3
"""QuantForge - lightweight harness guardrail report.

Provides deterministic checks around candidate rotation so the research core can decide
whether a proposed recovery lane is coherent enough to queue or run.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTPUT_FILE = os.path.join(BASE_DIR, "harness-report.json")
AUTOPILOT_FILE = os.path.join(BASE_DIR, "autopilot-report.json")
RECOVERY_FILE = os.path.join(BASE_DIR, "candidate-recovery.json")
LANES_FILE = os.path.join(BASE_DIR, "experiment-lanes.json")
OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
REVIEW_FILE = os.path.join(BASE_DIR, "candidate-review.json")


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


def _queued_trial_is_fresh_retry(trial: dict, latest_outcome: dict) -> bool:
    if not isinstance(trial, dict) or not isinstance(latest_outcome, dict):
        return False
    queued_at = _parse_ts(trial.get("queued_at"))
    outcome_dt = _parse_ts(latest_outcome.get("recorded_at") or latest_outcome.get("completed_at"))
    if queued_at is None or outcome_dt is None:
        return False
    return queued_at > outcome_dt


def _rotation_matches_recovery(latest_hint: str, recovery_type: str, outcomes: dict) -> bool:
    if not latest_hint or not recovery_type:
        return True
    if recovery_type == latest_hint:
        return True
    if latest_hint == "quantforge_research_hold" and recovery_type == "quantforge_layered_trial":
        return True
    if latest_hint == "quantforge_research_hold" and recovery_type == "major_liquidity_expansion":
        return True

    history = (outcomes.get("history") or []) if isinstance(outcomes, dict) else []
    recent_failed_types = {
        str(row.get("type", "") or "")
        for row in sorted(history, key=lambda r: str(r.get("recorded_at", "") or ""))[-6:]
        if str(row.get("assessment", "") or "").lower() == "fail"
    }
    if (
        latest_hint == "setup_quality_recovery"
        and recovery_type == "competitiveness_gap_rebuild"
        and {"model_recalibration", "quantforge_redesign"}.issubset(recent_failed_types)
    ):
        return True
    return False


def build_report():
    autopilot = read_json(AUTOPILOT_FILE)
    recovery = read_json(RECOVERY_FILE)
    lanes = read_json(LANES_FILE)
    outcomes = read_json(OUTCOMES_FILE)
    review = read_json(REVIEW_FILE)

    latest = (outcomes.get("latest") or {}) if isinstance(outcomes, dict) else {}
    trial = (lanes.get("candidate_trial") or {}) if isinstance(lanes, dict) else {}
    candidate_lane = (lanes.get("candidate") or {}) if isinstance(lanes, dict) else {}

    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    rec_id = str(recovery.get("candidate_id") or "")
    trial_id = str(trial.get("candidate_id") or "")
    trial_status = str(trial.get("status") or "").lower()
    mode = str(autopilot.get("mode") or "").lower()
    latest_hint = str(latest.get("next_candidate_hint") or "")
    latest_assessment = str(latest.get("assessment") or "").lower()
    stale_latest_outcome = _outcome_is_stale_for_candidate(latest, candidate_lane)
    if stale_latest_outcome:
        latest_hint = ""
        latest_assessment = ""
    review_reco = str(review.get("recommendation") or "").lower()
    recovery_type = str(recovery.get("type") or "")
    stale_inputs = autopilot.get("stale_inputs") if isinstance(autopilot.get("stale_inputs"), list) else []

    check(
        "candidate_matches_trial",
        trial_status not in {"queued", "active"} or not rec_id or not trial_id or rec_id == trial_id,
        f"trial_status={trial_status or 'missing'} recovery={rec_id or 'missing'} trial={trial_id or 'missing'}",
    )
    check(
        "failed_outcome_rotates_cleanly",
        latest_assessment not in {"fail", "insufficient"}
        or not latest_hint
        or _rotation_matches_recovery(latest_hint, recovery_type, outcomes),
        f"latest_hint={latest_hint or 'none'} recovery_type={recovery_type or 'missing'}",
    )
    check(
        "paper_only_trial",
        trial_status not in {"queued", "active"} or not trial or bool(trial.get("paper_only", False)),
        f"trial_status={trial_status or 'missing'} paper_only={trial.get('paper_only') if trial else 'n/a'}",
    )
    check(
        "autopilot_respects_queued_trial",
        trial_status != "queued" or mode in {"pause_new_entries", "run_candidate_paper_trial", "hold_in_paper"},
        f"trial_status={trial_status or 'missing'} mode={mode or 'missing'}",
    )
    check(
        "reviewer_matches_state",
        not trial_status or review_reco in {
            "advance_candidate",
            "hold_active_trial",
            "rotate_candidate_class",
            "retire_candidate_and_rotate",
            "queue_candidate_trial",
            "freeze_and_repair",
            "freeze_and_rebuild",
            "observe",
        },
        f"review_recommendation={review_reco or 'missing'}",
    )
    check(
        "autopilot_inputs_fresh",
        not stale_inputs,
        ", ".join(stale_inputs) if stale_inputs else "fresh",
    )
    check(
        "blocked_trial_not_left_queued",
        not (trial_status == "queued" and latest_assessment == "blocked" and not _queued_trial_is_fresh_retry(trial, latest)),
        f"trial_status={trial_status or 'missing'} latest_assessment={latest_assessment or 'missing'}",
    )

    failed = [row for row in checks if not row["passed"]]
    status = "ok" if not failed else "warn"
    recommendation = "allow_progress"
    if failed:
        recommendation = "fix_harness_mismatch"
    elif latest_assessment == "blocked":
        recommendation = "repair_before_progress"
    elif trial_status == "queued":
        recommendation = "queue_ready"
    elif trial_status == "active":
        recommendation = "hold_trial"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "recommendation": recommendation,
        "check_count": len(checks),
        "failed_count": len(failed),
        "checks": checks,
        "summary": {
            "candidate_id": rec_id or None,
            "trial_id": trial_id or None,
            "trial_status": trial_status or None,
            "autopilot_mode": mode or None,
            "latest_outcome_assessment": latest_assessment or None,
            "latest_outcome_hint": latest_hint or None,
            "review_recommendation": review_reco or None,
        },
    }


def main():
    cfg.require_production_runtime("quantforge_harness_report.py")
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge harness report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
