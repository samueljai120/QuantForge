#!/usr/bin/env python3
"""QuantForge — baseline/candidate experiment lane registry.

Keeps a small, explicit record of which paper state is considered the baseline,
which candidate is under review, and what governance/diagnosis say about each.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
LANES_FILE = os.path.join(BASE_DIR, "experiment-lanes.json")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
DIAGNOSIS_FILE = os.path.join(BASE_DIR, "diagnosis-report.json")
PROMOTION_FILE = os.path.join(BASE_DIR, "model", "promotion_report.json")
MODEL_META_FILE = os.path.join(BASE_DIR, "model", "model_meta.json")
CANDIDATE_RECOVERY_FILE = os.path.join(BASE_DIR, "candidate-recovery.json")
CANDIDATE_OUTCOMES_FILE = os.path.join(BASE_DIR, "candidate-outcomes.json")
DEFAULT_TRIAL_MAX_CYCLES = 6
DEFAULT_TRIAL_DURATION_HOURS = 24


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _f(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _load_outcomes() -> dict:
    data = read_json(CANDIDATE_OUTCOMES_FILE)
    if isinstance(data, dict):
        data.setdefault("updated_at", None)
        data.setdefault("latest", None)
        data.setdefault("history", [])
        return data
    return {"updated_at": None, "latest": None, "history": []}


def _save_outcomes(data: dict) -> None:
    write_json(CANDIDATE_OUTCOMES_FILE, data)


def _score_trial_outcome(trial: dict, baseline: dict, candidate: dict) -> dict:
    trial_assessment = str(trial.get("assessment", "") or "").lower()
    if trial_assessment == "blocked":
        return {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "candidate_id": trial.get("candidate_id"),
            "type": str(trial.get("type", "") or ""),
            "status": str(trial.get("status", "")).lower(),
            "assessment": "blocked",
            "next_candidate_hint": trial.get("next_candidate_hint") or "capital_preservation",
            "queued_at": trial.get("queued_at"),
            "started_at": trial.get("started_at"),
            "completed_at": trial.get("completed_at"),
            "cycles_run": int(trial.get("cycles_run", 0) or 0),
            "max_cycles": int(trial.get("max_cycles", 0) or 0),
            "paper_only": bool(trial.get("paper_only", False)),
            "baseline_snapshot": trial.get("baseline_snapshot", {}),
            "trial_changes": trial.get("changes", []),
            "candidate_metrics": {
                "paper_total_pnl_pct": round(_f(candidate.get("paper_total_pnl_pct")), 4),
                "governance_recommendation": candidate.get("governance_recommendation"),
                "promotion_decision": candidate.get("promotion_decision"),
            },
            "baseline_metrics": {
                "paper_total_pnl_pct": round(_f(baseline.get("paper_total_pnl_pct")), 4),
                "governance_recommendation": baseline.get("governance_recommendation"),
                "promotion_decision": baseline.get("promotion_decision"),
            },
            "pnl_gap_vs_baseline": None,
            "reasons": [
                "Candidate trial never started and was retired as blocked.",
                f"Blocked reason: {trial.get('blocked_reason', 'unknown')}.",
            ],
        }

    baseline_pnl_pct = _f(baseline.get("paper_total_pnl_pct"))
    candidate_pnl_pct = _f(candidate.get("paper_total_pnl_pct"))
    pnl_gap_vs_baseline = round(candidate_pnl_pct - baseline_pnl_pct, 4)
    governance = str(candidate.get("governance_recommendation", "") or "").upper()
    promotion = str(candidate.get("promotion_decision", "") or "").upper()
    candidate_type = str(trial.get("type", "") or "")

    if trial_assessment == "fail" and str(trial.get("completion_reason", "") or "").lower() == "no_target_long_surface":
        summary = trial.get("completion_summary") if isinstance(trial.get("completion_summary"), dict) else {}
        strongest = summary.get("strongest_long_hold") if isinstance(summary, dict) else {}
        strongest = strongest if isinstance(strongest, dict) else {}
        reasons = []
        assessment_reason = str(trial.get("assessment_reason", "") or "").strip()
        if assessment_reason:
            reasons.append(assessment_reason)
        if strongest:
            reasons.append(
                "The expansion lane produced "
                f"{int(summary.get('buy_signals', 0) or 0)} buy signals versus "
                f"{int(summary.get('sell_signals', 0) or 0)} sell signals, while strongest long hold "
                f"{strongest.get('symbol', 'unknown')} only reached {float(strongest.get('long_confidence', 0.0) or 0.0):.4f}."
            )
        else:
            reasons.append("The expansion lane completed early after repeated scans failed to surface a viable target long edge.")
        return {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "candidate_id": trial.get("candidate_id"),
            "type": candidate_type,
            "status": str(trial.get("status", "")).lower(),
            "assessment": "fail",
            "next_candidate_hint": trial.get("next_candidate_hint") or "setup_quality_recovery",
            "queued_at": trial.get("queued_at"),
            "started_at": trial.get("started_at"),
            "completed_at": trial.get("completed_at"),
            "cycles_run": int(trial.get("cycles_run", 0) or 0),
            "max_cycles": int(trial.get("max_cycles", 0) or 0),
            "paper_only": bool(trial.get("paper_only", False)),
            "baseline_snapshot": trial.get("baseline_snapshot", {}),
            "trial_changes": trial.get("changes", []),
            "candidate_metrics": {
                "paper_total_pnl_pct": round(candidate_pnl_pct, 4),
                "governance_recommendation": candidate.get("governance_recommendation"),
                "promotion_decision": candidate.get("promotion_decision"),
            },
            "baseline_metrics": {
                "paper_total_pnl_pct": round(baseline_pnl_pct, 4),
                "governance_recommendation": baseline.get("governance_recommendation"),
                "promotion_decision": baseline.get("promotion_decision"),
            },
            "pnl_gap_vs_baseline": pnl_gap_vs_baseline,
            "reasons": reasons,
        }

    reasons = []
    assessment = "insufficient"
    next_candidate_hint = None

    if pnl_gap_vs_baseline >= 1.0 and governance in {"HOLD", "OBSERVE"}:
        assessment = "pass"
        reasons.append(
            f"Candidate beat baseline by {pnl_gap_vs_baseline:+.2f} points with governance {governance or 'UNKNOWN'}."
        )
    elif pnl_gap_vs_baseline <= 0.0:
        assessment = "fail"
        reasons.append(f"Candidate failed to beat baseline after a bounded trial ({pnl_gap_vs_baseline:+.2f} points).")
    elif governance in {"REVIEW", "DEMOTE"}:
        assessment = "fail"
        reasons.append(f"Governance remained {governance} after the bounded trial.")
    elif promotion == "DO_NOT_PROMOTE":
        assessment = "insufficient"
        reasons.append("Promotion still blocked after the bounded trial.")
    else:
        reasons.append("Bounded trial completed without enough evidence for a stronger verdict.")

    if assessment == "pass" and candidate_type == "capital_preservation":
        next_candidate_hint = "major_liquidity_expansion"
    elif assessment == "pass" and candidate_type == "major_liquidity_expansion":
        next_candidate_hint = "observe"
    if assessment != "pass":
        next_candidate_hint = {
            "capital_preservation": "setup_quality_recovery",
            "major_liquidity_expansion": "setup_quality_recovery",
            "setup_quality_recovery": "regime_sensitive_retraining",
            "regime_sensitive_retraining": "model_recalibration",
            "model_recalibration": "quantforge_redesign",
            "quantforge_redesign": "competitiveness_gap_rebuild",
            "competitiveness_gap_rebuild": "quantforge_research_hold",
            "quantforge_layered_trial": "quantforge_research_hold",
        }.get(candidate_type)

    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": trial.get("candidate_id"),
        "type": candidate_type,
        "status": str(trial.get("status", "")).lower(),
        "assessment": assessment,
        "next_candidate_hint": next_candidate_hint,
        "queued_at": trial.get("queued_at"),
        "started_at": trial.get("started_at"),
        "completed_at": trial.get("completed_at"),
        "cycles_run": int(trial.get("cycles_run", 0) or 0),
        "max_cycles": int(trial.get("max_cycles", 0) or 0),
        "paper_only": bool(trial.get("paper_only", False)),
        "baseline_snapshot": trial.get("baseline_snapshot", {}),
        "trial_changes": trial.get("changes", []),
        "candidate_metrics": {
            "paper_total_pnl_pct": round(candidate_pnl_pct, 4),
            "governance_recommendation": candidate.get("governance_recommendation"),
            "promotion_decision": candidate.get("promotion_decision"),
        },
        "baseline_metrics": {
            "paper_total_pnl_pct": round(baseline_pnl_pct, 4),
            "governance_recommendation": baseline.get("governance_recommendation"),
            "promotion_decision": baseline.get("promotion_decision"),
        },
        "pnl_gap_vs_baseline": pnl_gap_vs_baseline,
        "reasons": reasons,
    }


def _record_completed_trial_outcome(current: dict, baseline: dict, candidate: dict) -> dict:
    trial = current.get("candidate_trial") or {}
    if not isinstance(trial, dict) or str(trial.get("status", "")).lower() != "completed":
        return current

    candidate_id = str(trial.get("candidate_id", "") or "")
    if not candidate_id:
        return current

    outcomes = _load_outcomes()
    history = [row for row in (outcomes.get("history") or []) if str(row.get("candidate_id", "")) != candidate_id]
    outcome = _score_trial_outcome(trial, baseline, candidate)
    history.append(outcome)
    history = history[-24:]
    outcomes["history"] = history
    outcomes["latest"] = outcome
    outcomes["updated_at"] = outcome["recorded_at"]
    _save_outcomes(outcomes)

    trial["assessment"] = outcome["assessment"]
    trial["outcome_recorded_at"] = outcome["recorded_at"]
    trial["pnl_gap_vs_baseline"] = outcome["pnl_gap_vs_baseline"]
    trial["next_candidate_hint"] = outcome.get("next_candidate_hint")
    current["candidate_trial"] = trial
    return current


def _build_candidate_trial(existing: dict, recovery: dict, baseline: dict) -> dict | None:
    if not isinstance(recovery, dict):
        return existing or None

    if str(recovery.get("type", "") or "") == "quantforge_research_hold":
        return None

    existing = existing or {}
    existing_status = str(existing.get("status", "")).lower()
    if existing and existing_status in {"queued", "active"}:
        if str(existing.get("candidate_id", "")) == str(recovery.get("candidate_id", "")):
            trial = dict(existing)
            trial["changes"] = recovery.get("changes", [])
            trial["validation"] = recovery.get("validation", [])
            trial["priority"] = recovery.get("priority")
            trial["type"] = recovery.get("type")
            trial["paper_only"] = True
            trial["success_criteria"] = trial.get("success_criteria") or [
                "Paper drawdown should stabilize or improve versus the degraded baseline.",
                "Recent expectancy should stop worsening during the bounded trial window.",
                "No promotion is allowed from this lane until governance and durable history improve.",
            ]
            return trial
        return dict(existing)

    candidate_id = recovery.get("candidate_id")
    status = str(recovery.get("status", "")).lower()
    if not candidate_id or status not in {"proposed", "observe"}:
        return existing or None

    def _trial_success_criteria() -> list[str]:
        recovery_type = str(recovery.get("type", "") or "")
        if recovery_type == "major_liquidity_expansion":
            return [
                "The expansion lane must add trade breadth without reintroducing broad low-quality non-major exposure.",
                "Only majors and top-liquidity alts should survive the lane, and non-major entries must remain top-quality only.",
                "No promotion is allowed until the expansion lane beats the preservation baseline and durable history improves.",
            ]
        if recovery_type == "quantforge_layered_trial":
            return [
                "The layered trial must beat baseline using the rebuilt prediction, regime, risk, and execution contract.",
                "Only holdout-positive, labeled, regime-supported setups should survive this paper lane.",
                "No promotion is allowed from the layered trial until governance and durable history improve.",
            ]
        return [
            "Paper drawdown should stabilize or improve versus the degraded baseline.",
            "Recent expectancy should stop worsening during the bounded trial window.",
            "No promotion is allowed from this lane until governance and durable history improve.",
        ]

    def _new_trial() -> dict:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=DEFAULT_TRIAL_DURATION_HOURS)
        return {
            "candidate_id": candidate_id,
            "type": recovery.get("type"),
            "priority": recovery.get("priority"),
            "status": "queued" if status == "proposed" else "observe",
            "paper_only": True,
            "queued_at": now.isoformat(),
            "started_at": None,
            "completed_at": None,
            "expires_at": expires_at.isoformat(),
            "cycles_run": 0,
            "max_cycles": DEFAULT_TRIAL_MAX_CYCLES,
            "changes": recovery.get("changes", []),
            "validation": recovery.get("validation", []),
            "success_criteria": _trial_success_criteria(),
            "baseline_snapshot": {
                "label": baseline.get("label"),
                "model_trained_at": baseline.get("model_trained_at"),
                "paper_total_pnl_pct": baseline.get("paper_total_pnl_pct"),
                "governance_recommendation": baseline.get("governance_recommendation"),
                "promotion_decision": baseline.get("promotion_decision"),
            },
        }

    if existing.get("candidate_id") == candidate_id:
        if (
            existing_status == "completed"
            and str(existing.get("assessment", "") or "").lower() == "blocked"
            and str(existing.get("blocked_reason", "") or "").lower() == "queue_wait_timeout"
            and status == "proposed"
        ):
            return _new_trial()
        trial = dict(existing)
        trial["changes"] = recovery.get("changes", [])
        trial["validation"] = recovery.get("validation", [])
        trial["priority"] = recovery.get("priority")
        trial["type"] = recovery.get("type")
        trial["paper_only"] = True
        trial["success_criteria"] = trial.get("success_criteria") or [
            "Paper drawdown should stabilize or improve versus the degraded baseline.",
            "Recent expectancy should stop worsening during the bounded trial window.",
            "No promotion is allowed from this lane until governance and durable history improve.",
        ]
        return trial

    return _new_trial()


def snapshot_current_lane(label: str):
    portfolio = read_json(PORTFOLIO_FILE)
    governance = read_json(GOVERNANCE_FILE)
    diagnosis = read_json(DIAGNOSIS_FILE)
    promotion = read_json(PROMOTION_FILE)
    model_meta = read_json(MODEL_META_FILE)
    candidate_recovery = read_json(CANDIDATE_RECOVERY_FILE)

    lane = {
        "label": label,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model_trained_at": model_meta.get("trained_at"),
        "paper_total_pnl_pct": _f(governance.get("paper", {}).get("total_pnl_pct")),
        "realized_pnl": _f(portfolio.get("realized_pnl")),
        "open_positions": len((portfolio.get("positions") or {})),
        "governance_recommendation": governance.get("recommendation"),
        "promotion_decision": promotion.get("overall_decision"),
        "diagnosis_causes": diagnosis.get("causes", []),
        "diagnosis_actions": diagnosis.get("recommended_actions", []),
        "candidate_recovery": {
            "id": candidate_recovery.get("candidate_id"),
            "type": candidate_recovery.get("type"),
            "priority": candidate_recovery.get("priority"),
            "status": candidate_recovery.get("status"),
        },
    }
    return lane


def cmd_update():
    cfg.require_production_runtime("quantforge_experiment_lanes.py")
    current = read_json(LANES_FILE) or {
        "updated_at": None,
        "baseline": None,
        "candidate": None,
        "candidate_trial": None,
        "history": [],
    }

    baseline = current.get("baseline")
    candidate = snapshot_current_lane("candidate")
    candidate_recovery = read_json(CANDIDATE_RECOVERY_FILE)

    if baseline is None:
        baseline = dict(candidate)
        baseline["label"] = "baseline"

    if candidate:
        current.setdefault("history", []).append(candidate)
        current["history"] = current["history"][-12:]

    current["baseline"] = baseline
    current["candidate"] = candidate
    current["candidate_trial"] = _build_candidate_trial(current.get("candidate_trial"), candidate_recovery, baseline)
    current = _record_completed_trial_outcome(current, baseline, candidate)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(LANES_FILE, current)
    if not os.path.exists(CANDIDATE_OUTCOMES_FILE):
        _save_outcomes({"updated_at": None, "latest": None, "history": []})

    print("QuantForge experiment lanes")
    print(f"Baseline pnl:  {baseline.get('paper_total_pnl_pct', 0.0):+.2f}%")
    print(f"Candidate pnl: {candidate.get('paper_total_pnl_pct', 0.0):+.2f}%")
    if current.get("candidate_trial"):
        print(f"Trial:         {current['candidate_trial'].get('status')} ({current['candidate_trial'].get('type')})")
        if current["candidate_trial"].get("assessment"):
            print(f"Assessment:    {current['candidate_trial'].get('assessment')}")
    print(f"Saved:         {LANES_FILE}")


if __name__ == "__main__":
    cmd_update()
