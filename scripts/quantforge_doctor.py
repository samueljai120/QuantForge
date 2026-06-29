#!/usr/bin/env python3
"""QuantForge readiness and coherence doctor.

Runs a lightweight health gate across QuantForge control artifacts so operators
can tell whether the system is truly ready to trade, intentionally paused, or
blocked by stale / inconsistent state.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_params import load_merged_quantforge_params

BASE_DIR = os.path.join(cfg.data, "quantforge")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
MONITOR_FILE = os.path.join(BASE_DIR, "monitor-report.json")
AUTOPILOT_FILE = os.path.join(BASE_DIR, "autopilot-report.json")
LANES_FILE = os.path.join(BASE_DIR, "experiment-lanes.json")
HARNESS_FILE = os.path.join(BASE_DIR, "harness-report.json")
REVIEW_FILE = os.path.join(BASE_DIR, "candidate-review.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "doctor-report.json")
MAX_PORTFOLIO_AGE_HOURS = 8
MAX_LAST_SCAN_AGE_HOURS = 8
MAX_CONTROL_AGE_HOURS = 8

# --- Agent lane (READ-ONLY: agent + reflect daemon are frozen, never write) ---
AGENT_PORTFOLIO_FILE = os.path.join(BASE_DIR, "agent_portfolio.json")
REFLECT_DECISIONS_FILE = os.path.join(BASE_DIR, "reflect_decisions.jsonl")
ALLOCATOR_DECISIONS_FILE = os.path.join(BASE_DIR, "allocator_decisions.jsonl")
WATCHDOG_HEALTH_FILE = os.path.join(BASE_DIR, "watchdog_health.json")
LEGACY_HEALTH_FILE = os.path.join(BASE_DIR, "health.json")
INVARIANTS_STATE_FILE = os.path.join(BASE_DIR, "qf_invariants_state.json")
# Agent runs hourly via cron ("5 * * * *"); >6h means several missed cycles.
AGENT_PORTFOLIO_WARN_HOURS = 3
AGENT_PORTFOLIO_FAIL_HOURS = 6
# Reflect+allocator run once daily ("0 1 * * *"); >30h means a day was skipped.
DECISION_STALE_HOURS = 30
# Halt / kill flag files in the agent data dir. Presence is operationally
# meaningful (trading frozen / flattened), so surface them rather than fail.
HALT_KILL_FLAGS = ["dd_halt.flag", "agent_halt.flag", "KILL", "KILL_FLATTEN"]


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def age_hours(value):
    dt = parse_ts(value)
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def age_check(label, value, max_age):
    age = age_hours(value)
    if age is None:
        return {"name": label, "ok": False, "detail": "missing timestamp"}
    if age > max_age:
        return {"name": label, "ok": False, "detail": f"stale at {age:.1f}h"}
    return {"name": label, "ok": True, "detail": f"fresh at {age:.1f}h"}


def _paper_trial_can_override_entry_caps(trial: dict | None) -> bool:
    if not isinstance(trial, dict):
        return False
    if str(trial.get("status", "") or "").lower() not in {"queued", "active"}:
        return False
    if not bool(trial.get("paper_only", False)):
        return False
    for change in trial.get("changes", []) or []:
        if not isinstance(change, dict):
            continue
        if str(change.get("key", "") or "") in {"max_positions", "max_long_positions", "max_short_positions"}:
            try:
                override = change.get("to", change.get("value", 0))
                if float(override or 0) > 0:
                    return True
            except Exception:
                continue
        if (
            str(change.get("key", "") or "") == "strategy_scope"
            and str(change.get("value", "") or "").strip().lower()
            in {"slower_high_conviction_majors_only", "major_symbols_and_positive_holdout_slices"}
        ):
            return True
    return False


def file_age_hours(path):
    """Age of a file by mtime, for artifacts with no internal timestamp."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return (datetime.now(timezone.utc).timestamp() - mtime) / 3600.0


def is_bad_number(value):
    """True if value is a NaN/inf float (silent equity-bug signal)."""
    return isinstance(value, float) and not math.isfinite(value)


def build_agent_lane():
    """Read-only health + ledger-invariant checks for the frozen agent lane.

    NEVER writes any agent file. Missing files are reported, not invented.
    """
    checks = []

    # 1. Agent portfolio presence + freshness (hourly cron).
    if not os.path.exists(AGENT_PORTFOLIO_FILE):
        checks.append({
            "name": "agent_portfolio_present",
            "ok": False,
            "detail": "missing agent_portfolio.json",
        })
        port = {}
    else:
        port = read_json(AGENT_PORTFOLIO_FILE)
        age = file_age_hours(AGENT_PORTFOLIO_FILE)
        if age is None:
            checks.append({
                "name": "agent_portfolio_fresh",
                "ok": False,
                "detail": "unreadable mtime",
            })
        elif age > AGENT_PORTFOLIO_FAIL_HOURS:
            checks.append({
                "name": "agent_portfolio_fresh",
                "ok": False,
                "detail": f"stale at {age:.1f}h (> {AGENT_PORTFOLIO_FAIL_HOURS}h)",
            })
        elif age > AGENT_PORTFOLIO_WARN_HOURS:
            checks.append({
                "name": "agent_portfolio_fresh",
                "ok": True,
                "warn": True,
                "detail": f"aging at {age:.1f}h (> {AGENT_PORTFOLIO_WARN_HOURS}h warn)",
            })
        else:
            checks.append({
                "name": "agent_portfolio_fresh",
                "ok": True,
                "detail": f"fresh at {age:.1f}h",
            })

    # 2. Halt / kill flag files — presence is the signal, surface age too.
    active_flags = []
    for flag in HALT_KILL_FLAGS:
        path = os.path.join(BASE_DIR, flag)
        if os.path.exists(path):
            age = file_age_hours(path)
            age_str = f"{age:.1f}h" if age is not None else "unknown age"
            active_flags.append(f"{flag} ({age_str})")
    checks.append({
        "name": "halt_kill_flags",
        "ok": True,
        "warn": bool(active_flags),
        "detail": "; ".join(active_flags) if active_flags else "none active",
    })

    # 3. futures_kill state read from agent_portfolio.json.
    futures_kill = bool(port.get("futures_kill")) if isinstance(port, dict) else False
    checks.append({
        "name": "futures_kill",
        "ok": True,
        "warn": futures_kill,
        "detail": "ENGAGED (futures permanently disabled)" if futures_kill else "clear",
    })

    # 4. Reflect / allocator decision-log freshness (daily cron).
    for name, path in (
        ("reflect_decisions", REFLECT_DECISIONS_FILE),
        ("allocator_decisions", ALLOCATOR_DECISIONS_FILE),
    ):
        if not os.path.exists(path):
            checks.append({
                "name": f"{name}_fresh",
                "ok": True,
                "warn": True,
                "detail": "missing (never run?)",
            })
            continue
        age = file_age_hours(path)
        if age is None:
            checks.append({"name": f"{name}_fresh", "ok": False, "detail": "unreadable mtime"})
        elif age > DECISION_STALE_HOURS:
            checks.append({
                "name": f"{name}_fresh",
                "ok": True,
                "warn": True,
                "detail": f"stale at {age:.1f}h (> {DECISION_STALE_HOURS}h)",
            })
        else:
            checks.append({"name": f"{name}_fresh", "ok": True, "detail": f"fresh at {age:.1f}h"})

    # 5. Watchdog health — surface any issues/flags it reports.
    health_path = WATCHDOG_HEALTH_FILE if os.path.exists(WATCHDOG_HEALTH_FILE) else LEGACY_HEALTH_FILE
    if os.path.exists(health_path):
        wd = read_json(health_path)
        issues = []
        if isinstance(wd, dict):
            for key in ("issues", "flags", "alerts", "warnings", "errors"):
                val = wd.get(key)
                if isinstance(val, (list, tuple)) and val:
                    issues.extend(str(x) for x in val)
                elif isinstance(val, str) and val.strip():
                    issues.append(val.strip())
            status = str(wd.get("status", "") or "").lower()
            if status and status not in {"ok", "healthy", "green"}:
                issues.append(f"status={status}")
        checks.append({
            "name": "watchdog_health",
            "ok": not issues,
            "detail": "; ".join(issues) if issues else f"no issues reported ({os.path.basename(health_path)})",
        })
    else:
        checks.append({
            "name": "watchdog_health",
            "ok": True,
            "warn": True,
            "detail": "watchdog_health.json missing",
        })

    # 6. Invariants state — if the dedicated detector already found critical
    # money-conservation issues, surface them directly in doctor status.
    if os.path.exists(INVARIANTS_STATE_FILE):
        inv_state = read_json(INVARIANTS_STATE_FILE)
        n_critical = int(inv_state.get("n_critical", 0) or 0)
        n_warning = int(inv_state.get("n_warning", 0) or 0)
        violations = inv_state.get("violations") or []
        head = ", ".join(v.get("name", "?") for v in violations[:3]) if violations else "none"
        checks.append({
            "name": "invariants_state",
            "ok": n_critical == 0,
            "warn": n_critical == 0 and n_warning > 0,
            "detail": f"{n_critical} critical / {n_warning} warning ({head})",
        })
    else:
        checks.append({
            "name": "invariants_state",
            "ok": True,
            "warn": True,
            "detail": "qf_invariants_state.json missing",
        })

    # 7. Ledger invariants — cheap static guard against an 8th equity bug.
    checks.extend(build_ledger_invariants(port if isinstance(port, dict) else {}))

    return checks


def build_ledger_invariants(port):
    """Assert internal consistency of agent_portfolio.json. READ-ONLY.

    Each rule emits ok / warn / fail with the offending value. Never writes.
    """
    out = []

    def emit(name, ok, detail, warn=False):
        out.append({"name": name, "ok": ok, "warn": warn and ok, "detail": detail})

    if not port:
        emit("ledger_invariants", True, "agent_portfolio.json absent — skipped", warn=True)
        return out

    # starting_balance > 0
    sb = port.get("starting_balance")
    emit(
        "inv_starting_balance",
        isinstance(sb, (int, float)) and not is_bad_number(sb) and sb > 0,
        f"starting_balance={sb!r}",
    )

    # peak_equity >= 0
    pk = port.get("peak_equity")
    emit(
        "inv_peak_equity",
        isinstance(pk, (int, float)) and not is_bad_number(pk) and pk >= 0,
        f"peak_equity={pk!r}",
    )

    # cash >= 0 (negative cash is a real failure for a paper book)
    cash = port.get("cash")
    cash_ok = isinstance(cash, (int, float)) and not is_bad_number(cash) and cash >= 0
    emit("inv_cash_nonneg", cash_ok, f"cash={cash!r}")

    # n_trades >= 0
    nt = port.get("n_trades")
    emit(
        "inv_n_trades",
        isinstance(nt, int) and not isinstance(nt, bool) and nt >= 0,
        f"n_trades={nt!r}",
    )

    # total_fees_paid >= 0
    tf = port.get("total_fees_paid")
    emit(
        "inv_total_fees_paid",
        isinstance(tf, (int, float)) and not is_bad_number(tf) and tf >= 0,
        f"total_fees_paid={tf!r}",
    )

    # futures_position is a well-formed dict (present key must be a dict)
    has_fp = "futures_position" in port
    fp = port.get("futures_position")
    emit(
        "inv_futures_position",
        (not has_fp) or isinstance(fp, dict),
        "absent" if not has_fp else f"type={type(fp).__name__}",
    )

    # No NaN/inf anywhere in the numeric fields of the portfolio.
    bad = []

    def scan(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}{k}.")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}{i}.")
        elif is_bad_number(obj):
            bad.append(prefix.rstrip("."))

    scan(port)
    emit(
        "inv_no_nan_inf",
        not bad,
        "none" if not bad else "NaN/inf at: " + ", ".join(bad[:8]) + ("…" if len(bad) > 8 else ""),
    )

    return out


def build_control_param_conflict_check(autopilot: dict) -> dict:
    params = load_merged_quantforge_params()
    lanes = read_json(LANES_FILE)
    trial = (lanes.get("candidate_trial") or {}) if isinstance(lanes, dict) else {}
    conflicts = []
    mode = str((autopilot or {}).get("mode", "") or "").strip().lower()
    override = str(params.get("autopilot_override", "") or "").strip().lower()
    if override == "allow_entries" and mode in {"pause_new_entries", "review_required", "rollback_to_baseline"}:
        conflicts.append(f"autopilot_override=allow_entries bypasses autopilot mode={mode}")

    halted_caps = []
    for key in ("max_open_positions", "max_long_positions", "max_short_positions"):
        raw = params.get(key)
        if raw is None:
            continue
        try:
            if float(raw) <= 0:
                halted_caps.append(f"{key}={raw}")
        except Exception:
            continue
    if halted_caps and not _paper_trial_can_override_entry_caps(trial):
        conflicts.append("entry caps request hard halt: " + ", ".join(halted_caps))

    return {
        "name": "control_param_conflict",
        "ok": not conflicts,
        "detail": "; ".join(conflicts) if conflicts else "none",
    }


def build_report():
    portfolio = read_json(PORTFOLIO_FILE)
    last_scan = read_json(LAST_SCAN_FILE)
    governance = read_json(GOVERNANCE_FILE)
    monitor = read_json(MONITOR_FILE)
    autopilot = read_json(AUTOPILOT_FILE)
    lanes = read_json(LANES_FILE)
    harness = read_json(HARNESS_FILE)
    review = read_json(REVIEW_FILE)

    checks = [
        age_check("portfolio", portfolio.get("updated"), MAX_PORTFOLIO_AGE_HOURS),
        age_check("last_scan", last_scan.get("ts"), MAX_LAST_SCAN_AGE_HOURS),
        age_check("governance", governance.get("generated_at"), MAX_CONTROL_AGE_HOURS),
        age_check("monitor", monitor.get("generated_at"), MAX_CONTROL_AGE_HOURS),
        age_check("autopilot", autopilot.get("generated_at"), MAX_CONTROL_AGE_HOURS),
    ]

    trial = (lanes.get("candidate_trial") or {}) if isinstance(lanes, dict) else {}
    trial_status = str(trial.get("status", "") or "").lower()
    trial_assessment = str(trial.get("assessment", "") or "").lower()
    mode = str(autopilot.get("mode", "") or "").lower()
    monitor_health = str(monitor.get("health", "") or "").upper()
    harness_status = str(harness.get("status", "") or "").lower()
    review_reco = str(review.get("recommendation", "") or "").lower()

    checks.append({
        "name": "monitor_health",
        "ok": monitor_health not in {"STALLED"},
        "detail": monitor_health or "missing",
    })
    checks.append({
        "name": "harness_status",
        "ok": harness_status in {"ok", ""},
        "detail": harness_status or "missing",
    })
    checks.append({
        "name": "blocked_trial",
        "ok": trial_assessment != "blocked",
        "detail": f"trial_status={trial_status or 'missing'} assessment={trial_assessment or 'none'}",
    })
    checks.append({
        "name": "autopilot_stale_inputs",
        "ok": not bool(autopilot.get("stale_inputs")),
        "detail": ", ".join(autopilot.get("stale_inputs", [])) if autopilot.get("stale_inputs") else "none",
    })
    checks.append(build_control_param_conflict_check(autopilot))

    failed = [row for row in checks if not row["ok"]]

    # Agent lane (frozen / read-only) — surfaced as its own section so a control
    # PAUSE never masks an agent-lane integrity failure, and vice versa.
    agent_checks = build_agent_lane()
    agent_failed = [row for row in agent_checks if not row["ok"]]
    agent_warned = [row for row in agent_checks if row["ok"] and row.get("warn")]

    if failed or agent_failed:
        readiness = "BLOCKED"
    elif mode in {"pause_new_entries", "review_required", "rollback_to_baseline"}:
        readiness = "PAUSED"
    else:
        readiness = "READY"

    if agent_failed:
        agent_status = "FAIL"
    elif agent_warned:
        agent_status = "WARN"
    else:
        agent_status = "OK"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness,
        "autopilot_mode": mode or None,
        "monitor_health": monitor_health or None,
        "trial_status": trial_status or None,
        "trial_assessment": trial_assessment or None,
        "review_recommendation": review_reco or None,
        "check_count": len(checks),
        "failed_count": len(failed),
        "checks": checks,
        "agent_status": agent_status,
        "agent_check_count": len(agent_checks),
        "agent_failed_count": len(agent_failed),
        "agent_warned_count": len(agent_warned),
        "agent_checks": agent_checks,
    }


def main():
    cfg.require_production_runtime("quantforge_doctor.py")
    payload = build_report()
    write_json(OUTPUT_FILE, payload)
    print("QuantForge doctor")
    print(f"Readiness: {payload['readiness']}")
    print(f"Autopilot: {payload['autopilot_mode'] or 'missing'}")
    print(f"Monitor:   {payload['monitor_health'] or 'missing'}")
    print(f"Trial:     {(payload['trial_status'] or 'none')} / {(payload['trial_assessment'] or 'none')}")
    print(f"Failed:    {payload['failed_count']}/{payload['check_count']}")
    for check in payload["checks"]:
        marker = "OK" if check["ok"] else "FAIL"
        print(f"  [{marker}] {check['name']}: {check['detail']}")
    print(f"Agent lane: {payload['agent_status']} "
          f"({payload['agent_failed_count']} fail / {payload['agent_warned_count']} warn "
          f"/ {payload['agent_check_count']} checks)")
    for check in payload["agent_checks"]:
        if not check["ok"]:
            marker = "FAIL"
        elif check.get("warn"):
            marker = "WARN"
        else:
            marker = "OK"
        print(f"  [{marker}] {check['name']}: {check['detail']}")
    print(f"Saved:     {OUTPUT_FILE}")
    if len(sys.argv) > 1 and sys.argv[1] == "json":
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
