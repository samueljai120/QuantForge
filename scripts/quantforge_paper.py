#!/usr/bin/env python3
"""QuantForge Paper Trading Engine — QuantForge

Paper-trades on KuCoin public market data with technical analysis signals.
Proves profitability after fees before any live trading is enabled.

Usage:
    python3 quantforge_paper.py scan     # Scan market, generate signals
    python3 quantforge_paper.py status   # Show portfolio and stats
    python3 quantforge_paper.py run      # Full cycle: scan → signal → trade → update
    python3 quantforge_paper.py backtest # Backtest strategy gate (Sharpe, win rate)
    python3 quantforge_paper.py reconcile # Rebuild portfolio ledger from trade log
    python3 quantforge_paper.py repair   # Rebuild portfolio state from trade log
"""

import json
import math
import os
import sys
import time
import fcntl
from datetime import datetime, timedelta, timezone

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_params import LEGACY_PARAMS_FILE, load_merged_quantforge_params
from quantforge_signal_ranking import signal_rank_value

DATA_DIR = os.path.join(cfg.data, "quantforge")
TRADES_FILE = os.path.join(DATA_DIR, "paper-trades.jsonl")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
SIGNALS_FILE = os.path.join(DATA_DIR, "signals.jsonl")
RUN_LOCK_FILE = os.path.join(DATA_DIR, "quantforge-run.lock")
LAST_SCAN_FILE = os.path.join(DATA_DIR, "last_scan.json")
LAST_EXECUTION_FILE = os.path.join(DATA_DIR, "last_execution.json")
AUTOPILOT_FILE = os.path.join(DATA_DIR, "autopilot-report.json")
EXPERIMENT_LANES_FILE = os.path.join(DATA_DIR, "experiment-lanes.json")

KUCOIN_BASE = "https://api.kucoin.com"
KUCOIN_FUTURES_BASE = "https://api-futures.kucoin.com"
AUTOPILOT_MAX_AGE_HOURS = 8
QUEUED_TRIAL_MAX_WAIT_HOURS = 24
_TRIAL_MISSING = object()
RR_RATIO_TOLERANCE = 1e-6
NO_TARGET_LONG_SURFACE_MIN_STREAK = 2
NO_TARGET_LONG_SURFACE_CONFIDENCE_CEILING = 0.05
TRIAL_SCOPE_PRESETS = {
    "slower_high_conviction_majors_only": {
        "entry_profile": "majors_only",
        "symbol_universe": "major_liquidity_tier",
        "generic_long_policy": "require_top_quality",
        "entry_selection": "require_regime_support_and_labeled_setup_alignment",
        "non_major_entries": True,
        "max_long_positions": 1,
        "max_short_positions": 1,
    },
    "major_symbols_and_positive_holdout_slices": {
        "entry_profile": "majors_only",
        "symbol_universe": "major_liquidity_tier",
        "generic_long_policy": "require_top_quality",
        "entry_selection": "require_regime_support_and_labeled_setup_alignment",
        "non_major_entries": True,
        "max_fakeout_risk": 0.65,
        "allowed_short_setups": ["trend_short"],
        "allow_short_entries_in_adverse_regime": True,
        # Match the executed-subset holdout limits that justified this trial.
        "max_short_positions": 2,
    },
}

# Trading parameters — reads the parameter tuner tuned values from strategy-params.json
def _load_strategy_params():
    return load_merged_quantforge_params()


def _coerce_string_set(value) -> set[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return {part for part in parts if part}
    if isinstance(value, (list, tuple, set)):
        out = set()
        for item in value:
            text = str(item).strip()
            if text:
                out.add(text)
        return out
    return set()


def _coerce_float_map(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key).strip()] = float(raw)
        except Exception:
            continue
    return out


def _coerce_int_map(value) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key).strip()] = int(raw)
        except Exception:
            continue
    return out


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _long_setup_policy_reason(setup_tag: str) -> str | None:
    setup = str(setup_tag or "").strip()
    if BLOCKED_LONG_SETUPS and setup in BLOCKED_LONG_SETUPS:
        return f"blocked long setup policy rejects {setup}"
    if ALLOWED_LONG_SETUPS and setup not in ALLOWED_LONG_SETUPS:
        allowed = ", ".join(sorted(ALLOWED_LONG_SETUPS))
        return f"allowed long setup policy rejects {setup or 'unknown'}; allowed: {allowed}"
    return None


def _save_strategy_params(sp: dict):
    """Write updated strategy params back to disk."""
    # This is an operator/control-plane escape hatch, not an autotune-owned knob.
    # Do not let merged runtime state persist it back into legacy paper params.
    sp.pop("autopilot_override", None)
    sp["_last_modified_by"] = "quantforge_autotune"
    sp["_last_modified_at"] = datetime.now(timezone.utc).isoformat()
    with open(LEGACY_PARAMS_FILE, "w") as f:
        json.dump(sp, f, indent=2)


AUTOTUNE_MIN_INTERVAL_HOURS = 4.0  # never act more than once per 4 hours

def _auto_tune_thresholds(port: dict) -> dict | None:
    """Phase 1 auto-tune: auto-adjust entry thresholds based on recent trade performance.

    Returns dict describing the adjustment taken, or None if no change.
    Anti-spiral guardrails prevent runaway tightening or loosening.
    """
    if not AUTOTUNE_ENABLED:
        return None

    # Rate-limit: check last autotune decision timestamp, not strategy-params modification
    tune_file = os.path.join(DATA_DIR, "autotune-decisions.jsonl")
    if os.path.exists(tune_file):
        try:
            last_line = ""
            with open(tune_file) as f:
                for line in f:
                    if line.strip():
                        last_line = line
            if last_line:
                last_decision = json.loads(last_line)
                last_ts = last_decision.get("ts", "")
                if last_ts:
                    last_dt = datetime.fromisoformat(last_ts)
                    hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                    if hours_since < AUTOTUNE_MIN_INTERVAL_HOURS:
                        return None  # too soon — skip this cycle
        except Exception:
            pass

    trades_file = os.path.join(DATA_DIR, "paper-trades.jsonl")
    if not os.path.exists(trades_file):
        return None

    # Load recent closed trades
    closed = []
    with open(trades_file) as f:
        for line in f:
            try:
                t = json.loads(line)
                if t.get("type") in ("CLOSE", "PARTIAL_CLOSE") and "pnl" in t:
                    closed.append(t)
            except Exception:
                continue

    recent = closed[-AUTOTUNE_LOOKBACK:]
    sp = _load_strategy_params()
    current_thresh = float(sp.get("signal_confidence_threshold", 0.70))

    if len(recent) < AUTOTUNE_MIN_TRADES:
        # Cold start: if threshold is too high and we're not generating trades, relax
        if current_thresh > 0.72 and len(recent) < 8:
            sp["signal_confidence_threshold"] = 0.70
            sp["_autotune_reason"] = "cold start — relaxing to gather data"
            _save_strategy_params(sp)
            return {"action": "cold_start_relax", "new_threshold": 0.70}
        return None

    # Decay-weighted expectancy
    weights = [AUTOTUNE_DECAY_WEIGHT ** (len(recent) - 1 - i) for i in range(len(recent))]
    w_sum = sum(weights)
    weighted_pnl = sum(float(t.get("pnl", 0.0)) * w for t, w in zip(recent, weights))
    expectancy = weighted_pnl / w_sum

    wins = sum(1 for t in recent if float(t.get("pnl", 0.0)) > 0)
    win_rate = wins / len(recent)

    # Load tuning history for consecutive direction check
    tune_file = os.path.join(DATA_DIR, "autotune-decisions.jsonl")
    history = []
    if os.path.exists(tune_file):
        with open(tune_file) as f:
            for line in f:
                try:
                    history.append(json.loads(line))
                except Exception:
                    pass

    recent_dirs = [h.get("direction") for h in history[-AUTOTUNE_CONSECUTIVE:]]

    current_risk = float(sp.get("max_position_pct", 0.02))
    decision = None

    if expectancy < 0 and all(d == "tighten" for d in recent_dirs) and len(recent_dirs) >= AUTOTUNE_CONSECUTIVE:
        # Sustained negative expectancy — tighten
        new_thresh = min(current_thresh + AUTOTUNE_MAX_ADJUST, AUTOTUNE_CEIL_THRESH)
        new_risk = max(current_risk - 0.002, 0.01)
        if new_thresh != current_thresh or new_risk != current_risk:
            sp["signal_confidence_threshold"] = round(new_thresh, 4)
            sp["max_position_pct"] = round(new_risk, 4)
            sp["max_position_pct_for_quantforge"] = round(new_risk, 4)
            decision = {
                "action": "tighten", "direction": "tighten",
                "old_threshold": current_thresh, "new_threshold": round(new_thresh, 4),
                "old_risk": current_risk, "new_risk": round(new_risk, 4),
                "expectancy": round(expectancy, 4), "win_rate": round(win_rate, 4),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
    elif expectancy > 0 and win_rate > 0.50 and all(d == "relax" for d in recent_dirs) and len(recent_dirs) >= AUTOTUNE_CONSECUTIVE:
        # Sustained positive edge — relax cautiously
        new_thresh = max(current_thresh - 0.01, AUTOTUNE_FLOOR_THRESH)
        if new_thresh != current_thresh:
            sp["signal_confidence_threshold"] = round(new_thresh, 4)
            decision = {
                "action": "relax", "direction": "relax",
                "old_threshold": current_thresh, "new_threshold": round(new_thresh, 4),
                "expectancy": round(expectancy, 4), "win_rate": round(win_rate, 4),
                "ts": datetime.now(timezone.utc).isoformat(),
            }

    if decision is None:
        # No action — just record direction for next check
        direction = "tighten" if expectancy < 0 else ("relax" if expectancy > 0 and win_rate > 0.50 else "hold")
        decision = {
            "action": "hold", "direction": direction,
            "expectancy": round(expectancy, 4), "win_rate": round(win_rate, 4),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    with open(tune_file, "a") as f:
        f.write(json.dumps(decision) + "\n")

    if decision.get("action") not in ("hold", None):
        sp["_autotune_reason"] = f"{decision['action']}: expectancy={expectancy:.4f}, win_rate={win_rate:.2f}"
        _save_strategy_params(sp)
        # Append a reset-hold after every action so the consecutive window clears.
        # Without this, the next cycle sees N action records (all direction="tighten")
        # and immediately fires again.
        reset_hold = {
            "action": "hold", "direction": "hold",
            "expectancy": round(expectancy, 4), "win_rate": round(win_rate, 4),
            "ts": datetime.now(timezone.utc).isoformat(),
            "_note": "post-action reset",
        }
        with open(tune_file, "a") as f:
            f.write(json.dumps(reset_hold) + "\n")

    return decision


def _artifact_age_hours(payload: dict, *timestamp_keys: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in timestamp_keys:
        dt = _parse_iso_dt(payload.get(key))
        if dt:
            return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    return None


def load_autopilot_report(*, require_fresh: bool = False) -> dict:
    try:
        with open(AUTOPILOT_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        if require_fresh:
            age_hours = _artifact_age_hours(data, "generated_at")
            if age_hours is None or age_hours > AUTOPILOT_MAX_AGE_HOURS:
                raise RuntimeError(
                    f"Autopilot artifact is stale ({age_hours:.1f}h old)." if age_hours is not None
                    else "Autopilot artifact is missing generated_at."
                )
        return data
    except Exception:
        if require_fresh:
            raise
        return {}


def load_runtime_autopilot_report() -> tuple[dict, str | None]:
    """Load the freshest autopilot artifact we can without deadlocking runtime loops.

    Runtime safety rule:
    - Fresh autopilot: obey it as-is.
    - Stale or missing autopilot: keep managing exits and scanning, but force
      new entries to remain blocked until the review/control plane recovers.
    """
    try:
        return load_autopilot_report(require_fresh=True), None
    except Exception as exc:
        fallback = load_autopilot_report(require_fresh=False) or {}
        reasons = fallback.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        fallback["mode"] = "pause_new_entries"
        fallback["runtime_fallback"] = True
        fallback["runtime_warning"] = str(exc)
        if str(exc) not in reasons:
            reasons.insert(0, str(exc))
        fallback["reasons"] = reasons
        stale_inputs = fallback.get("stale_inputs")
        if not isinstance(stale_inputs, list):
            stale_inputs = []
        if str(exc) not in stale_inputs:
            stale_inputs.insert(0, str(exc))
        fallback["stale_inputs"] = stale_inputs
        return fallback, str(exc)


def load_experiment_lanes() -> dict:
    try:
        with open(EXPERIMENT_LANES_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_experiment_lanes(lanes: dict) -> None:
    with open(EXPERIMENT_LANES_FILE, "w") as f:
        json.dump(lanes, f, indent=2)


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def load_candidate_trial() -> dict:
    trial = (load_experiment_lanes().get("candidate_trial") or {})
    return trial if isinstance(trial, dict) else {}


def _candidate_trial_status(trial: dict | None) -> str:
    if not trial:
        return ""
    return str(trial.get("status", "")).strip().lower()


def candidate_trial_is_active(trial: dict | None) -> bool:
    return _candidate_trial_status(trial) in {"queued", "active"}


def _trial_override_value(trial: dict | None, key: str, default=None):
    value = _trial_change_value(trial, key, _TRIAL_MISSING)
    if value is not _TRIAL_MISSING:
        return value
    return default


def _trial_change_value(trial: dict | None, key: str, default=_TRIAL_MISSING):
    if not trial or not candidate_trial_is_active(trial):
        return default
    for change in trial.get("changes", []) or []:
        if change.get("key") != key:
            continue
        if "to" in change:
            return change.get("to")
        if "value" in change:
            return change.get("value")
    return default


def _trial_scope_name(trial: dict | None) -> str:
    scope = _trial_change_value(trial, "strategy_scope", "")
    return str(scope or "").strip().lower()


def _trial_runtime_value(trial: dict | None, key: str, default=None):
    explicit = _trial_change_value(trial, key, _TRIAL_MISSING)
    if explicit is not _TRIAL_MISSING:
        return explicit
    if not trial or not candidate_trial_is_active(trial):
        return default
    return TRIAL_SCOPE_PRESETS.get(_trial_scope_name(trial), {}).get(key, default)


def _trial_runtime_int(trial: dict | None, key: str, default: int) -> int:
    value = _trial_runtime_value(trial, key, default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _trial_runtime_string_set(trial: dict | None, key: str) -> set[str]:
    return _coerce_string_set(_trial_runtime_value(trial, key, []))


def _trial_override_float(trial: dict | None, key: str, default: float) -> float:
    value = _trial_override_value(trial, key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _trial_threshold_relief(trial: dict | None) -> tuple[float, float]:
    long_relief = max(0.0, _trial_override_float(trial, "trial_long_threshold_relief", TRIAL_LONG_THRESHOLD_RELIEF))
    short_relief = max(0.0, _trial_override_float(trial, "trial_short_threshold_relief", 0.0))
    return long_relief, short_relief


def _trial_allows_adverse_short_entries(trial: dict | None) -> bool:
    return bool(candidate_trial_is_active(trial) and _trial_runtime_value(trial, "allow_short_entries_in_adverse_regime", False))


def _trial_total_position_cap(trial: dict | None, direction: str, trial_short_cap: int) -> int:
    """Infer a bounded total-position cap for active paper-only trials.

    This is the only path allowed to reopen a zeroed legacy total cap. The
    trial must already be queued/active and explicitly paper-only.
    """
    if not candidate_trial_is_active(trial) or not bool((trial or {}).get("paper_only", False)):
        return 0
    explicit_total = _trial_runtime_value(trial, "max_positions", None)
    if explicit_total is not None:
        try:
            return max(0, min(int(explicit_total), 4))
        except Exception:
            return 0
    raw_long_cap = _trial_runtime_value(trial, "max_long_positions", None)
    raw_short_cap = _trial_runtime_value(trial, "max_short_positions", None)
    try:
        trial_long_cap = max(0, int(raw_long_cap)) if raw_long_cap is not None else 0
    except Exception:
        trial_long_cap = 0
    try:
        trial_short_cap = max(0, int(raw_short_cap)) if raw_short_cap is not None else 0
    except Exception:
        trial_short_cap = 0
    if str(direction or "").upper() == "LONG" and trial_long_cap <= 0:
        return 0
    if str(direction or "").upper() == "SHORT" and trial_short_cap <= 0:
        return 0
    inferred_total = trial_long_cap + trial_short_cap
    if inferred_total <= 0:
        inferred_total = max(trial_long_cap, trial_short_cap)
    return max(0, min(int(inferred_total), 4))


def _trial_bypasses_major_only_for_symbol(trial: dict | None, symbol: str, direction: str) -> bool:
    if not candidate_trial_is_active(trial) or not bool((trial or {}).get("paper_only", False)):
        return False
    if str(direction or "").upper() != "LONG":
        return False
    trial_entry_profile = str(_trial_runtime_value(trial, "entry_profile", "") or "").lower()
    trial_symbol_universe = str(_trial_runtime_value(trial, "symbol_universe", "") or "").lower()
    if trial_entry_profile != "majors_plus_liquid_alts" and trial_symbol_universe != "major_and_top_alt_tier":
        return False
    return symbol in TOP_ALT_EXPANSION_SYMBOLS


def _load_json_file(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _active_trial_surface_summary(last_scan: dict, trial: dict | None) -> dict | None:
    if not isinstance(last_scan, dict) or not isinstance(trial, dict):
        return None
    if _candidate_trial_status(trial) != "active":
        return None
    if str(trial.get("type", "") or "") != "major_liquidity_expansion":
        return None

    flow = last_scan.get("flow") or {}
    rows = last_scan.get("results") or []
    strongest_long_hold = None
    long_filtered_skips = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason", "") or "")
        setup_tag = str(row.get("setup_tag", "") or "")
        if row.get("status") == "skip" and (setup_tag.endswith("_long") or "long" in reason.lower()):
            long_filtered_skips.append(
                {
                    "symbol": row.get("symbol"),
                    "setup_tag": setup_tag,
                    "reason": reason,
                }
            )
        if row.get("status") != "hold":
            continue
        try:
            long_conf = float(row.get("long_confidence") or 0.0)
        except Exception:
            long_conf = 0.0
        try:
            short_conf = float(row.get("short_confidence") or 0.0)
        except Exception:
            short_conf = 0.0
        if long_conf < short_conf:
            continue
        if strongest_long_hold is None or long_conf > float(strongest_long_hold.get("long_confidence") or 0.0):
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
        and float(strongest_long_hold.get("long_confidence") or 0.0) < NO_TARGET_LONG_SURFACE_CONFIDENCE_CEILING
        and (sell_signals > 0 or selection_blocked > 0 or bool(long_filtered_skips))
    )
    return {
        "scan_ts": last_scan.get("ts"),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "threshold_miss": threshold_miss,
        "selection_blocked": selection_blocked,
        "long_filtered_skips": long_filtered_skips[:5],
        "strongest_long_hold": strongest_long_hold,
        "no_target_long_surface": no_target_long_surface,
    }


def _update_trial_surface_streak(trial: dict) -> None:
    if _candidate_trial_status(trial) != "active":
        return
    if str(trial.get("type", "") or "") != "major_liquidity_expansion":
        return

    summary = _active_trial_surface_summary(_load_json_file(LAST_SCAN_FILE), trial)
    if not summary:
        return

    trial["last_no_target_long_surface"] = summary
    if not summary.get("no_target_long_surface"):
        trial["no_target_long_surface_cycles"] = 0
        return

    streak = int(trial.get("no_target_long_surface_cycles", 0) or 0) + 1
    trial["no_target_long_surface_cycles"] = streak
    if streak < NO_TARGET_LONG_SURFACE_MIN_STREAK:
        return

    strongest = summary.get("strongest_long_hold") or {}
    strongest_symbol = strongest.get("symbol", "unknown")
    strongest_conf = float(strongest.get("long_confidence", 0.0) or 0.0)
    trial["status"] = "completed"
    trial["completed_at"] = datetime.now(timezone.utc).isoformat()
    trial["assessment"] = "fail"
    trial["completion_reason"] = "no_target_long_surface"
    trial["completion_summary"] = summary
    trial["next_candidate_hint"] = trial.get("next_candidate_hint") or "setup_quality_recovery"
    trial["assessment_reason"] = (
        "Active major-liquidity expansion never surfaced its target long edge "
        f"for {streak} cycles; strongest long hold {strongest_symbol} only reached {strongest_conf:.4f}."
    )


def _trial_effective_position_cap(
    trial: dict | None,
    direction: str,
    regime_max_positions: int,
    trial_short_cap: int,
) -> int:
    effective_max = int(regime_max_positions)
    trial_total_cap = _trial_total_position_cap(trial, direction, trial_short_cap)
    if trial_total_cap > 0:
        effective_max = max(effective_max, trial_total_cap)
    if (
        candidate_trial_is_active(trial)
        and bool((trial or {}).get("paper_only", False))
        and direction == "SHORT"
        and _trial_allows_adverse_short_entries(trial)
    ):
        return min(4, max(effective_max, int(trial_short_cap)))
    return min(4, effective_max)


def _allow_candidate_trial_gate_bypass(trial: dict | None) -> bool:
    return bool(candidate_trial_is_active(trial) and bool((trial or {}).get("paper_only", False)))


def _rr_gate_blocks(rr_ratio: float, min_rr: float) -> bool:
    return float(rr_ratio) + RR_RATIO_TOLERANCE < float(min_rr)


def _entry_exit_levels(entry_price: float, stop_pct: float, direction: str) -> tuple[float, float, float]:
    """Return rounded stop/target plus raw R:R computed before price rounding."""
    raw_sl = entry_price * (1 - stop_pct if direction == "LONG" else 1 + stop_pct)
    raw_tp = entry_price * (1 + TAKE_PROFIT_PCT if direction == "LONG" else 1 - TAKE_PROFIT_PCT)
    rr_ratio = abs(raw_tp - entry_price) / max(abs(entry_price - raw_sl), 1e-10)
    return round(raw_sl, 4), round(raw_tp, 4), rr_ratio


def _candidate_trial_expired(trial: dict | None) -> bool:
    if not trial:
        return False
    status = _candidate_trial_status(trial)
    if status == "queued" and not trial.get("started_at"):
        queued_at = _parse_iso_dt(trial.get("queued_at"))
        if queued_at and datetime.now(timezone.utc) >= queued_at + timedelta(hours=QUEUED_TRIAL_MAX_WAIT_HOURS):
            return True
        return False
    expires_at = _parse_iso_dt(trial.get("expires_at"))
    if expires_at and datetime.now(timezone.utc) >= expires_at:
        return True
    max_cycles = int(trial.get("max_cycles", 0) or 0)
    cycles_run = int(trial.get("cycles_run", 0) or 0)
    return max_cycles > 0 and cycles_run >= max_cycles


def sync_candidate_trial_state(autopilot: dict | None) -> dict:
    lanes = load_experiment_lanes()
    trial = lanes.get("candidate_trial") or {}
    if not isinstance(trial, dict) or not trial:
        return {}

    now = datetime.now(timezone.utc).isoformat()
    mode = str((autopilot or {}).get("mode", "")).strip().lower()
    status = _candidate_trial_status(trial)
    mutated = False

    if status in {"queued", "active"} and _candidate_trial_expired(trial):
        trial["status"] = "completed"
        trial["completed_at"] = now
        if status == "queued" and not trial.get("started_at"):
            trial["assessment"] = trial.get("assessment") or "blocked"
            trial["blocked_reason"] = trial.get("blocked_reason") or "queue_wait_timeout"
            trial["next_candidate_hint"] = trial.get("next_candidate_hint") or "capital_preservation"
        mutated = True
    elif mode == "run_candidate_paper_trial" and status == "queued":
        trial["status"] = "active"
        trial["started_at"] = now
        trial.setdefault("cycles_run", 0)
        expires_at = _parse_iso_dt(trial.get("expires_at"))
        if not expires_at or expires_at <= datetime.now(timezone.utc):
            trial["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        mutated = True

    if mutated:
        lanes["candidate_trial"] = trial
        lanes["updated_at"] = now
        save_experiment_lanes(lanes)

    return trial


def finalize_candidate_trial_cycle(autopilot: dict | None) -> dict:
    lanes = load_experiment_lanes()
    trial = lanes.get("candidate_trial") or {}
    if not isinstance(trial, dict) or _candidate_trial_status(trial) != "active":
        return trial if isinstance(trial, dict) else {}

    mode = str((autopilot or {}).get("mode", "")).strip().lower()
    if mode != "run_candidate_paper_trial":
        return trial

    trial["cycles_run"] = int(trial.get("cycles_run", 0) or 0) + 1
    _update_trial_surface_streak(trial)
    if _candidate_trial_status(trial) == "active" and _candidate_trial_expired(trial):
        trial["status"] = "completed"
        trial["completed_at"] = datetime.now(timezone.utc).isoformat()
    lanes["candidate_trial"] = trial
    lanes["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_experiment_lanes(lanes)
    return trial


def autopilot_blocks_new_entries(autopilot: dict | None) -> bool:
    if not autopilot:
        return False
    # Manual override: strategy-params.json can set "autopilot_override": "allow_entries"
    # to bypass the pause while new safety layers (ATR stops, regime gate, daily breaker,
    # correlation filter) prove themselves. The autopilot report still reflects true
    # system health — this just lets the bot trade through it with its own risk controls.
    sp = _load_strategy_params()
    if str(sp.get("autopilot_override", "")).strip().lower() == "allow_entries":
        return False
    return str(autopilot.get("mode", "")).strip().lower() in AUTOPILOT_ENTRY_BLOCK_MODES

_sp = _load_strategy_params()

# --- Risk-param allowlist bounds  -------------------------------
# strategy-params.json is writable by autotune/reflection/self-heal daemons.
# Every risk-critical key is clamped here at load: a daemon (or LLM) can tune
# WITHIN these bounds but can never loosen risk past them. Widening a bound is
# a human code change.
_SP_BOUNDS = {
    "max_position_pct": (0.001, 0.02),
    "max_position_pct_for_quantforge": (0.001, 0.02),
    "leverage": (1, 2),
    # Zero is a valid human/operator hard halt and must not be coerced back to 1.
    "max_open_positions": (0, 4),
    "max_long_positions": (0, 3),
    "max_short_positions": (0, 2),
    "min_cash_reserve_pct": (0.10, 0.90),
    "max_hold_hours": (1, 72),
    "paper_entry_slippage_bps": (5, 200),
    "paper_exit_slippage_bps": (5, 200),
    "paper_spread_bps": (2, 100),
    "reentry_cooldown_hours": (1, 48),
    "atr_min_stop_pct": (0.005, 0.03),
    "atr_max_stop_pct": (0.010, 0.05),
    "short_trailing_min_r": (0.5, 2.0),
    "funding_rate_long_skip": (-0.02, -0.001),
}
for _k, (_lo, _hi) in _SP_BOUNDS.items():
    if _k in _sp:
        try:
            _v = float(_sp[_k])
        except Exception:
            _sp.pop(_k, None)
            continue
        _cl = min(max(_v, _lo), _hi)
        if _cl != _v:
            print(f"[SAFETY] strategy-params {_k}={_v} outside [{_lo}, {_hi}] — clamped to {_cl}", file=sys.stderr)
        _sp[_k] = _cl


def _normalize_position_caps(total_cap: int, long_cap: int, short_cap: int) -> tuple[int, int, int]:
    total = max(0, min(int(total_cap), 4))
    long = max(0, min(int(long_cap), min(total, 3)))
    short = max(0, min(int(short_cap), min(total, 2)))
    return total, long, short

STARTING_BALANCE = 1000.0
MAKER_FEE = 0.001
TAKER_FEE = 0.001
MAX_RISK_PCT = _sp.get("max_position_pct", _sp.get("max_position_pct_for_quantforge", 0.02))
STOP_LOSS_PCT = _sp.get("trailing_stop_pct", 0.03)
TAKE_PROFIT_PCT = _sp.get("take_profit_pct", 0.05)
ENABLE_TRAILING_PROFIT = _coerce_bool(_sp.get("enable_trailing_profit", True), True)
TRAILING_PROFIT_ACTIVATE_PCT = float(_sp.get("trailing_profit_activate_pct", 0.010))  # activate trailing at +1.0%
TRAILING_PROFIT_GIVEBACK_PCT = float(_sp.get("trailing_profit_giveback_pct", max(STOP_LOSS_PCT * 0.5, 0.01)))
TRAILING_PROFIT_TIER2_ACTIVATE_PCT = float(
    _sp.get("trailing_profit_tier2_activate_pct", max(TAKE_PROFIT_PCT, TRAILING_PROFIT_ACTIVATE_PCT * 1.5))
)
TRAILING_PROFIT_TIER3_ACTIVATE_PCT = float(
    _sp.get("trailing_profit_tier3_activate_pct", max(TAKE_PROFIT_PCT * 1.4, TRAILING_PROFIT_ACTIVATE_PCT * 2.0))
)
TRAILING_PROFIT_TIER2_GIVEBACK_PCT = float(
    _sp.get("trailing_profit_tier2_giveback_pct", max(TRAILING_PROFIT_GIVEBACK_PCT * 0.75, 0.0075))
)
TRAILING_PROFIT_TIER3_GIVEBACK_PCT = float(
    _sp.get("trailing_profit_tier3_giveback_pct", max(TRAILING_PROFIT_GIVEBACK_PCT * 0.5, 0.005))
)
MAX_HOLD_HOURS = int(_sp.get("max_hold_hours", 48))  # time-stop: close after N hours regardless
LEVERAGE      = int(_sp.get("leverage", 2))          # futures leverage multiplier
MAX_POSITIONS, MAX_LONG_POSITIONS, MAX_SHORT_POSITIONS = _normalize_position_caps(
    _sp.get("max_open_positions", 4),
    _sp.get("max_long_positions", 3),
    _sp.get("max_short_positions", 2),
)
MIN_CASH_RESERVE    = float(_sp.get("min_cash_reserve_pct", 0.20))  # always keep 20% cash
# HARD CIRCUIT BREAKER — cannot be overridden by strategy-params.json.
# No single position may use more than this fraction of equity as margin.
# With 2× leverage this caps notional at 50% of equity.  Even if risk-based
# sizing requests more, this cap prevents one gapping stop from wiping out the
# account.  The 5%/2% param combo that caused TAO's $148 loss would have been
# capped here.
MAX_MARGIN_PER_POSITION_PCT = 0.25
# ATR-based stop configuration (replaces fixed STOP_LOSS_PCT for entry sizing)
ATR_STOP_MULT_LONG  = float(_sp.get("atr_stop_mult_long", 2.5))
ATR_STOP_MULT_SHORT = float(_sp.get("atr_stop_mult_short", 3.0))
ATR_MIN_STOP_PCT    = float(_sp.get("atr_min_stop_pct", 0.015))   # never tighter than 1.5%
ATR_MAX_STOP_PCT    = float(_sp.get("atr_max_stop_pct", 0.020))   # never wider than 2.0% — tighter for better R:R
MIN_RR_RATIO        = float(_sp.get("min_rr_ratio", 2.0))         # minimum reward:risk ratio to enter
ATR_COMPRESSION_THRESHOLD = float(_sp.get("atr_compression_threshold", 0.70))
ATR_COMPRESSION_WIDEN = float(_sp.get("atr_compression_widen", 1.4))
MAX_DAILY_LOSS_PCT = 0.05  # 5% equity — hard stop for the day
# --- Hard safety breakers (non-negotiable) ----------------------------------
# HARD breakers. Deliberately NOT read from strategy-params.json so neither
# the autotuner nor the reflection daemon can loosen them. Changing these is
# a human decision.
MAX_WEEKLY_LOSS_PCT = 0.06        # 6% equity realized over trailing 7 days — halt new entries
MAX_DRAWDOWN_HALT_PCT = 0.10      # 10% peak-to-trough equity — latching halt, needs `reset-halt`
PRICE_GAP_HALT_PCT = 0.08         # BTC moving >8% in one 1h bar = market dislocation, pause entries
BTC_STRESS_VOL_24H = 0.05         # BTC 24h realized vol >=5%: all crypto correlates to ~1 — collapse groups, halve size
SIGNAL_DRIFT_MAX_PCT = 0.01       # reject entry if live price drifted >1% from the signal price
FUNDING_INTERVAL_HOURS = 8.0      # KuCoin funding interval
FUNDING_FALLBACK_RATE = 0.0001    # 0.01%/8h accrual assumption when live rate unavailable
KILL_FILE = os.path.join(DATA_DIR, "KILL")                  # touch → block all new entries
KILL_FLATTEN_FILE = os.path.join(DATA_DIR, "KILL_FLATTEN")  # touch → close everything + block entries
DD_HALT_FILE = os.path.join(DATA_DIR, "dd_halt.flag")       # written when drawdown halt latches
CORRELATION_GROUPS = {
    "large_cap": {"BTC-USDT", "ETH-USDT"},
    "l1_alts": {"SOL-USDT", "AVAX-USDT", "DOT-USDT", "ADA-USDT", "NEAR-USDT"},
    "defi": {"UNI-USDT", "AAVE-USDT", "CAKE-USDT", "CRV-USDT"},
    "meme": {"DOGE-USDT", "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT", "MOODENG-USDT"},
    "privacy": {"XMR-USDT", "ZEC-USDT", "DASH-USDT"},
}
MAX_PER_CORRELATION_GROUP = 1
AUTOTUNE_ENABLED       = _coerce_bool(_sp.get("autotune_enabled", True), True)
AUTOTUNE_MIN_TRADES    = 15       # minimum trades before tuning activates
AUTOTUNE_LOOKBACK      = 50       # number of trades to analyze
AUTOTUNE_DECAY_WEIGHT  = 0.95     # exponential decay — recent trades matter more
AUTOTUNE_FLOOR_THRESH  = 0.60     # never relax threshold below this
AUTOTUNE_CEIL_THRESH   = 0.85     # never tighten threshold above this
AUTOTUNE_MAX_ADJUST    = 0.02     # max change per adjustment
AUTOTUNE_CONSECUTIVE   = 3        # need 3 consecutive checks in same direction
FUNDING_RATE_LONG_SKIP_THRESHOLD = float(_sp.get("funding_rate_long_skip", -0.005))  # -0.5% per 8h
REENTRY_COOLDOWN_HOURS = float(_sp.get("reentry_cooldown_hours", 4))
SCAN_TOP_N = int(_sp.get("scan_top_n", 20))
PAPER_ENTRY_SLIPPAGE_BPS = float(_sp.get("paper_entry_slippage_bps", 10))
PAPER_EXIT_SLIPPAGE_BPS = float(_sp.get("paper_exit_slippage_bps", 10))
PAPER_SPREAD_BPS = float(_sp.get("paper_spread_bps", 4))
REBUILD_TRIAL_ENTRY_SLIPPAGE_BPS = float(_sp.get("rebuild_trial_entry_slippage_bps", 35))
REBUILD_TRIAL_EXIT_SLIPPAGE_BPS = float(_sp.get("rebuild_trial_exit_slippage_bps", 30))
REBUILD_TRIAL_SPREAD_BPS = float(_sp.get("rebuild_trial_spread_bps", 12))
REBUILD_TRIAL_STOP_GAP_BPS = float(_sp.get("rebuild_trial_stop_gap_bps", 18))
REBUILD_TRIAL_MARK_HAIRCUT_BPS = float(_sp.get("rebuild_trial_mark_haircut_bps", 18))
RESEARCH_HOLD_ENTRY_SLIPPAGE_BPS = float(_sp.get("research_hold_entry_slippage_bps", 30))
RESEARCH_HOLD_EXIT_SLIPPAGE_BPS = float(_sp.get("research_hold_exit_slippage_bps", 35))
RESEARCH_HOLD_SPREAD_BPS = float(_sp.get("research_hold_spread_bps", 14))
RESEARCH_HOLD_STOP_GAP_BPS = float(_sp.get("research_hold_stop_gap_bps", 18))
RESEARCH_HOLD_MARK_HAIRCUT_BPS = float(_sp.get("research_hold_mark_haircut_bps", 28))
NON_MAJOR_EXECUTION_PENALTY_BPS = float(_sp.get("non_major_execution_penalty_bps", 12))
LOW_PRICE_EXECUTION_PENALTY_BPS = float(_sp.get("low_price_execution_penalty_bps", 10))
LOW_QUALITY_EXECUTION_PENALTY_BPS = float(_sp.get("low_quality_execution_penalty_bps", 8))
MAX_CONSECUTIVE_LOSSES_PER_SYMBOL = int(_sp.get("max_consecutive_losses_per_symbol", 2))
LOSS_STREAK_COOLOFF_HOURS = float(_sp.get("loss_streak_cooloff_hours", 12))
SINGLE_LOSS_COOLOFF_HOURS = float(_sp.get("single_loss_cooloff_hours", min(LOSS_STREAK_COOLOFF_HOURS, 6)))
ML_ONLY_TRAINED_PAIRS = _coerce_bool(_sp.get("ml_only_trained_pairs", True), True)
REGIME_ADJUST_ENABLED = True  # dynamic threshold + sizing via regime detector
MIN_VOLUME_USDT = _sp.get("min_volume_usdt", 500_000)
SIGNAL_CONFIDENCE_THRESHOLD = _sp.get("signal_confidence_threshold",
                                       _sp.get("signal_confidence_threshold_for_quantforge", 0.6))
LONG_LIVE_THRESHOLD_CAP = float(_sp.get("long_live_threshold_cap", 0.72))
LONG_LIVE_THRESHOLD_OFFSET = float(_sp.get("long_live_threshold_offset", -0.08))
TRIAL_LONG_THRESHOLD_RELIEF = float(_sp.get("trial_long_threshold_relief", 0.0))
TRIAL_ENTRY_SCORE_THRESHOLD = float(_sp.get("trial_entry_score_threshold", max(0.50, SIGNAL_CONFIDENCE_THRESHOLD - TRIAL_LONG_THRESHOLD_RELIEF)))
SHORT_LIVE_THRESHOLD_CAP = float(_sp.get("short_live_threshold_cap", 0.55))
QUALITY_FILTER_ENABLED = _coerce_bool(_sp.get("quality_filter_enabled", True), True)
QUALITY_MIN_24H_TURNOVER_USDT = float(_sp.get("quality_min_24h_turnover_usdt", max(MIN_VOLUME_USDT * 5, 2_500_000)))
QUALITY_MIN_RECENT_24H_TURNOVER_USDT = float(_sp.get("quality_min_recent_24h_turnover_usdt", 500_000))
QUALITY_MIN_HISTORY_CANDLES = int(_sp.get("quality_min_history_candles", 5000))
QUALITY_MIN_HISTORY_CANDLES_MAJOR = int(
    _sp.get("quality_min_history_candles_major", min(QUALITY_MIN_HISTORY_CANDLES, 1500))
)
QUALITY_MIN_HISTORY_CANDLES_TOP_ALT = int(
    _sp.get("quality_min_history_candles_top_alt", min(QUALITY_MIN_HISTORY_CANDLES, 2500))
)
QUALITY_MIN_PRICE = float(_sp.get("quality_min_price", 0.01))
QUALITY_MAX_ABS_24H_MOVE_PCT = float(_sp.get("quality_max_abs_24h_move_pct", 30.0))
QUALITY_MAX_REALIZED_VOL_24H = float(_sp.get("quality_max_realized_vol_24h", 0.35))
QUALITY_MIN_SCORE = float(_sp.get("quality_min_score", 0.58))
ENABLE_RULE_FALLBACK_WHEN_ML_DEGENERATE = _coerce_bool(_sp.get("enable_rule_fallback_when_ml_degenerate", True), True)
RULE_FALLBACK_MAX_ML_CONFIDENCE = float(_sp.get("rule_fallback_max_ml_confidence", 0.01))
RULE_FALLBACK_MIN_SCORE = float(_sp.get("rule_fallback_min_score", 2.0 / 3.0))
PANIC_SHORT_RET24H_PCT = float(_sp.get("panic_short_ret24h_pct", -0.15))
PANIC_SHORT_RET4H_PCT = float(_sp.get("panic_short_ret4h_pct", -0.08))
PANIC_SHORT_THRESHOLD_ADJ = float(_sp.get("panic_short_threshold_adj", -0.12))
ENABLE_SIGNAL_ROTATION = _coerce_bool(_sp.get("enable_signal_rotation", True), True)
ROTATION_MIN_SCORE_ADVANTAGE = float(_sp.get("rotation_min_score_advantage", 0.05))
ROTATION_MIN_OPEN_LOSS_PCT = float(_sp.get("rotation_min_open_loss_pct", 0.003))
AUTO_FEEDBACK_ENABLED = _coerce_bool(_sp.get("auto_feedback_enabled", True), True)
FEEDBACK_LOOKBACK_HOURS = float(_sp.get("feedback_lookback_hours", 72))
FEEDBACK_RECENT_CLOSE_LIMIT = int(_sp.get("feedback_recent_close_limit", 60))
SYMBOL_QUARANTINE_MIN_LOSSES = int(_sp.get("symbol_quarantine_min_losses", 2))
SYMBOL_QUARANTINE_HOURS = float(_sp.get("symbol_quarantine_hours", 24))
SETUP_QUARANTINE_MIN_TRADES = int(_sp.get("setup_quarantine_min_trades", 3))
SETUP_QUARANTINE_HOURS = float(_sp.get("setup_quarantine_hours", 18))
SETUP_BAD_WIN_RATE = float(_sp.get("setup_bad_win_rate", 0.34))
SETUP_BAD_AVG_PNL = float(_sp.get("setup_bad_avg_pnl", -2.5))
ADAPTIVE_RISK_ENABLED = _coerce_bool(_sp.get("adaptive_risk_enabled", True), True)
ADAPTIVE_RISK_FLOOR = float(_sp.get("adaptive_risk_floor", 0.45))
ADAPTIVE_RISK_CEIL = float(_sp.get("adaptive_risk_ceil", 1.0))
GENERIC_LONG_MIN_SETUP_SCORE = float(_sp.get("generic_long_min_setup_score", 0.50))
GENERIC_LONG_MIN_QUALITY_SCORE = float(_sp.get("generic_long_min_quality_score", 0.94))
GENERIC_LONG_NON_MAJOR_MIN_QUALITY_SCORE = float(_sp.get("generic_long_non_major_min_quality_score", 0.97))
NON_MAJOR_LONG_SCORE_PENALTY = float(_sp.get("non_major_long_score_penalty", 0.08))
MAJOR_LONG_SCORE_BOOST = float(_sp.get("major_long_score_boost", 0.03))
BLOCKED_LONG_SETUPS = _coerce_string_set(_sp.get("blocked_long_setups", ""))
ALLOWED_LONG_SETUPS = _coerce_string_set(_sp.get("allowed_long_setups", ""))
LONG_SETUP_SIZE_MULTIPLIERS = _coerce_float_map(_sp.get("long_setup_size_multipliers", {}))
LONG_SETUP_MIN_SCORE = _coerce_float_map(_sp.get("long_setup_min_score", {}))
LONG_SETUP_MIN_QUALITY = _coerce_float_map(_sp.get("long_setup_min_quality", {}))
LONG_SETUP_MAJOR_ONLY = _coerce_string_set(_sp.get("long_setup_major_only", ""))
RESEARCH_HOLD_NON_MAJOR_ALLOWED_LONG_SETUPS = _coerce_string_set(
    _sp.get("research_hold_non_major_allowed_long_setups", "trend_long,breakout_long")
)
RESEARCH_HOLD_NON_MAJOR_MIN_SETUP_SCORE = float(_sp.get("research_hold_non_major_min_setup_score", 0.67))
RESEARCH_HOLD_NON_MAJOR_MIN_QUALITY_SCORE = float(_sp.get("research_hold_non_major_min_quality_score", 0.96))
RESEARCH_HOLD_NON_MAJOR_SIZE_MULT = float(_sp.get("research_hold_non_major_size_mult", 0.9))
SYMBOL_SETUP_QUARANTINE_MIN_TRADES = int(_sp.get("symbol_setup_quarantine_min_trades", 2))
SYMBOL_SETUP_QUARANTINE_HOURS = float(_sp.get("symbol_setup_quarantine_hours", 18))
SYMBOL_SETUP_BAD_AVG_PNL = float(_sp.get("symbol_setup_bad_avg_pnl", -1.5))
SYMBOL_SETUP_BAD_WIN_RATE = float(_sp.get("symbol_setup_bad_win_rate", 0.34))
EXECUTION_USE_FILTERED_SCORE = _coerce_bool(_sp.get("execution_use_filtered_score", True), True)
BREAK_EVEN_PROTECT_PCT = float(_sp.get("break_even_protect_pct", 0.015))
BREAK_EVEN_BUFFER_PCT = float(_sp.get("break_even_buffer_pct", 0.002))
TRAILING_LOCK_PROFIT_SHARE_TIER2 = float(_sp.get("trailing_lock_profit_share_tier2", 0.45))
TRAILING_LOCK_PROFIT_SHARE_TIER3 = float(_sp.get("trailing_lock_profit_share_tier3", 0.65))
SHORT_TRAILING_MIN_R = float(_sp.get("short_trailing_min_r", 1.0))

STABLECOINS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD",
    "FRAX", "LUSD", "SUSD", "FDUSD", "PYUSD", "UST",
}

AUTOPILOT_ENTRY_BLOCK_MODES = {
    "pause_new_entries",
    "rollback_to_baseline",
    "review_required",
}
MAJOR_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BCH-USDT", "TRX-USDT"}
TOP_ALT_EXPANSION_SYMBOLS = MAJOR_SYMBOLS | {"ADA-USDT", "DOGE-USDT", "LINK-USDT", "AVAX-USDT", "LTC-USDT", "XMR-USDT", "TAO-USDT"}
RESEARCH_HOLD_FRAGILE_FAKEOUT_CAP = 0.65

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# ATR stop helper
# ---------------------------------------------------------------------------

def _trend_filter_close(candle) -> float:
    """Return candle close across dict and KuCoin list payloads."""
    if isinstance(candle, dict):
        if "close" in candle and candle["close"] is not None:
            return float(candle["close"])
        if "raw_close" in candle and candle["raw_close"] is not None:
            return float(candle["raw_close"])
        if "price" in candle and candle["price"] is not None:
            return float(candle["price"])
        return 0.0
    if len(candle) > 2:
        return float(candle[2])
    if candle:
        return float(candle[-1])
    return 0.0


def _compute_atr_stop_pct(candles: list, price: float, direction: str = "LONG") -> float:
    """Compute ATR-based stop distance as a percentage of price.

    Uses ATR-14 from candle data with direction-aware multiplier.
    Falls back to fixed STOP_LOSS_PCT if insufficient candle data.
    """
    if len(candles) < 20:
        return STOP_LOSS_PCT  # fallback
    try:
        import numpy as np
        highs  = np.array([float(c[3]) for c in candles[-20:]])   # index 3 = high
        lows   = np.array([float(c[4]) for c in candles[-20:]])   # index 4 = low
        closes = np.array([float(c[2]) for c in candles[-20:]])   # index 2 = close

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        # ATR-14 (simple mean as approximation)
        atr_14 = float(np.mean(tr[-14:]))
        # ATR-slow for compression detection (use all available)
        atr_slow = float(np.mean(tr)) if len(tr) >= 19 else atr_14
        atr_ratio = atr_14 / max(atr_slow, 1e-10)

        mult = ATR_STOP_MULT_LONG if direction == "LONG" else ATR_STOP_MULT_SHORT
        stop_dist = atr_14 * mult

        # Compression guard: widen stop in low-volatility squeeze
        if atr_ratio < ATR_COMPRESSION_THRESHOLD:
            stop_dist *= ATR_COMPRESSION_WIDEN

        stop_pct = stop_dist / max(price, 1e-10)

        # Hard clamp
        return max(ATR_MIN_STOP_PCT, min(stop_pct, ATR_MAX_STOP_PCT))
    except Exception:
        return STOP_LOSS_PCT


# ---------------------------------------------------------------------------
# Daily loss circuit breaker
# ---------------------------------------------------------------------------

def _daily_loss_breaker_active(port: dict) -> bool:
    """Return True if realized losses today exceed MAX_DAILY_LOSS_PCT of equity."""
    try:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_date_str = today_start.strftime("%Y-%m-%d")
        if not os.path.exists(TRADES_FILE):
            return False
        today_pnl = 0.0
        with open(TRADES_FILE) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if t.get("type") not in ("CLOSE", "PARTIAL_CLOSE"):
                        continue
                    ts = t.get("ts", "")
                    if ts[:10] >= today_date_str:
                        today_pnl += float(t.get("pnl", 0.0))
                except Exception:
                    continue
        eq = float(port.get("cash", 0.0))
        for pos in port.get("positions", {}).values():
            eq += float(pos.get("margin", 0.0))
            eq += float(pos.get("unrealized_pnl", 0.0))
        if eq <= 0:
            return True  # safety: no equity = stop trading
        return today_pnl / eq < -MAX_DAILY_LOSS_PCT
    except Exception:
        return False


def _portfolio_equity_quick(port: dict) -> float:
    """Cash + margin + last-known unrealized PnL, no network calls."""
    eq = float(port.get("cash", 0.0))
    for pos in port.get("positions", {}).values():
        eq += float(pos.get("margin", 0.0))
        eq += float(pos.get("unrealized_pnl", 0.0))
    return eq


def _refresh_portfolio_equity_snapshot(port: dict, current_equity: float | None = None) -> float:
    """Persist a non-null equity snapshot for downstream truth consumers."""
    if current_equity is None:
        current_equity = _portfolio_equity_quick(port)
    try:
        equity_value = round(float(current_equity), 4)
    except Exception:
        equity_value = round(float(port.get("starting_balance", STARTING_BALANCE) or STARTING_BALANCE), 4)
    port["equity"] = equity_value
    return equity_value


def _weekly_loss_breaker_active(port: dict) -> bool:
    """True if realized losses over the trailing 7 days exceed MAX_WEEKLY_LOSS_PCT of equity."""
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        if not os.path.exists(TRADES_FILE):
            return False
        week_pnl = 0.0
        with open(TRADES_FILE) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if t.get("type") not in ("CLOSE", "PARTIAL_CLOSE"):
                        continue
                    if str(t.get("ts", "")) >= cutoff:
                        week_pnl += float(t.get("pnl", 0.0))
                except Exception:
                    continue
        eq = _portfolio_equity_quick(port)
        if eq <= 0:
            return True
        return week_pnl / eq < -MAX_WEEKLY_LOSS_PCT
    except Exception:
        return False


def _drawdown_halt_active(port: dict) -> bool:
    """Latching max-drawdown halt.

    Triggers when equity falls MAX_DRAWDOWN_HALT_PCT below peak. Writes a flag
    file so the halt persists even if equity bounces back — clearing it is a
    human decision via the `reset-halt` CLI command.
    """
    if os.path.exists(DD_HALT_FILE):
        return True
    try:
        peak = float(port.get("peak_equity", STARTING_BALANCE) or STARTING_BALANCE)
        if peak <= 0:
            return False
        eq = _portfolio_equity_quick(port)
        dd = (peak - eq) / peak
        if dd >= MAX_DRAWDOWN_HALT_PCT:
            with open(DD_HALT_FILE, "w") as f:
                json.dump({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "drawdown": round(dd, 6),
                    "peak_equity": round(peak, 4),
                    "equity": round(eq, 4),
                    "note": "Latched. Clear with: quantforge_paper.py reset-halt",
                }, f, indent=2)
            return True
    except Exception:
        pass
    return False


def _kill_switch_state() -> str | None:
    """Operator kill switch via marker files: 'flatten' closes everything, 'halt' blocks entries."""
    if os.path.exists(KILL_FLATTEN_FILE):
        return "flatten"
    if os.path.exists(KILL_FILE):
        return "halt"
    return None


def _market_gap_halt() -> str | None:
    """Pause entries when BTC printed a >PRICE_GAP_HALT_PCT move in the last closed 1h bar."""
    try:
        candles = get_klines("BTC-USDT", "1hour", 3)
        if not candles or len(candles) < 2:
            return None
        prev_close = float(candles[-2][2])
        last_close = float(candles[-1][2])
        if prev_close <= 0:
            return None
        move = abs(last_close / prev_close - 1.0)
        if move >= PRICE_GAP_HALT_PCT:
            return f"BTC moved {move * 100:.1f}% in one 1h bar"
    except Exception:
        return None
    return None


EVENT_REPORT_FILE = os.path.join(DATA_DIR, "events", "event-overlap-report.json")
EVENT_BLOCK_MIN_SCORE = 0.7  # block entries while a high-impact event window is active


def _event_risk_block() -> str | None:
    """Reason string when a high-impact event window is active (funding window,
    internal incident overlap). Missing/stale report fails open — collector
    freshness is covered by the doctor gates."""
    try:
        with open(EVENT_REPORT_FILE) as f:
            report = json.load(f)
        gen_age = _age_hours_iso(report.get("generated_at"))
        if gen_age is None or gen_age > 6:
            return None
        hot = [
            e for e in (report.get("active_events") or [])
            if float(e.get("event_score", 0.0) or 0.0) >= EVENT_BLOCK_MIN_SCORE
        ]
        if hot:
            kinds = ", ".join(sorted({str(e.get("event_type")) for e in hot}))
            return f"event risk window active ({kinds})"
    except Exception:
        return None
    return None


def _age_hours_iso(value) -> float | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return None


def _btc_stress_mode() -> bool:
    """True when BTC 24h realized vol >= BTC_STRESS_VOL_24H.

    In a crypto stress event everything correlates toward 1, so the static
    correlation groups stop protecting. Callers collapse to one position max
    and halve sizing while this is active. Fails open to False (other halts
    cover data loss).
    """
    try:
        candles = get_klines("BTC-USDT", "1hour", 26)
        closes = [float(c[2]) for c in candles]
        if len(closes) < 24:
            return False
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]
        recent = rets[-24:]
        mean = sum(recent) / len(recent)
        var = sum((r - mean) ** 2 for r in recent) / len(recent)
        vol_24h = (var ** 0.5) * (24 ** 0.5)
        return vol_24h >= BTC_STRESS_VOL_24H
    except Exception:
        return False


def _accrue_funding(sym: str, pos: dict, price: float, now_iso: str, port: dict | None = None) -> float:
    """Accrue funding cost on a leveraged position since the last accrual.

    Longs pay positive funding, shorts receive it. Uses the live rate when
    available, otherwise a conservative fallback. Accrued cost is stored on the
    position and subtracted from PnL at (partial) close.
    """
    if not is_leveraged_position(pos):
        return 0.0
    try:
        last_iso = pos.get("funding_accrued_ts") or pos.get("open_ts") or pos.get("opened")
        last_dt = datetime.fromisoformat(str(last_iso).replace("Z", "+00:00"))
        now_dt = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
        hours = (now_dt - last_dt).total_seconds() / 3600.0
        if hours <= 0:
            return 0.0
        rate = _get_funding_rate(sym)
        if rate is None:
            rate = FUNDING_FALLBACK_RATE
        notional = float(pos.get("qty", 0.0)) * float(price)
        cost = rate * notional * (hours / FUNDING_INTERVAL_HOURS)
        if str(pos.get("direction", "LONG")).upper() == "SHORT":
            cost = -cost
        pos["funding_paid"] = round(float(pos.get("funding_paid", 0.0)) + cost, 8)
        pos["funding_accrued_ts"] = now_iso
        if port is not None:
            port["total_funding_paid"] = round(float(port.get("total_funding_paid", 0.0)) + cost, 8)
        return cost
    except Exception:
        return 0.0


def _correlation_group_full(symbol: str, port: dict) -> tuple:
    """Check if adding this symbol would over-concentrate in one correlation group.
    Returns (is_full: bool, reason: str).
    """
    open_syms = set(port.get("positions", {}).keys())
    for group_name, group_syms in CORRELATION_GROUPS.items():
        if symbol in group_syms:
            already_open = open_syms & group_syms
            if len(already_open) >= MAX_PER_CORRELATION_GROUP:
                return True, f"correlation group '{group_name}' already has {already_open}"
    return False, ""


# ---------------------------------------------------------------------------
# KuCoin public API helpers
# ---------------------------------------------------------------------------

def kucoin_get(path: str, params: dict | None = None) -> dict:
    """GET from KuCoin public REST API with retry."""
    url = KUCOIN_BASE + path
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            body = r.json()
            if body.get("code") == "200000":
                return body["data"]
            raise ValueError(f"KuCoin API error: {body.get('msg', body)}")
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def kucoin_futures_get(path: str, params: dict | None = None) -> dict | list:
    """GET from KuCoin Futures public REST API with retry."""
    url = KUCOIN_FUTURES_BASE + path
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            body = r.json()
            if body.get("code") == "200000":
                return body["data"]
            raise ValueError(f"KuCoin Futures API error: {body.get('msg', body)}")
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


_SPOT_TO_FUTURES = {"BTC-USDT": "XBTUSDTM"}


def to_futures_symbol(spot_sym: str) -> str:
    """Convert BTC-USDT → XBTUSDTM; others → ETHUSDTM etc."""
    return _SPOT_TO_FUTURES.get(spot_sym, spot_sym.replace("-USDT", "USDTM"))


def get_futures_tickers() -> list[dict]:
    """Get all active futures contracts with price + 24h volume (uses contracts/active)."""
    try:
        return kucoin_futures_get("/api/v1/contracts/active") or []
    except Exception:
        return []


def get_tickers() -> list[dict]:
    """Get all spot tickers (kept for backtest historical fetches)."""
    return kucoin_get("/api/v1/market/allTickers").get("ticker", [])


def get_klines(symbol: str, kline_type: str = "1hour", limit: int = 300) -> list[list]:
    """Fetch 1h futures klines — paginates to overcome KuCoin's 200-candle API cap.
    KuCoin caps /kline/query at 200 rows per request. We make ceil(limit/200) requests
    each covering a 200h window offset backwards, then merge + dedup by timestamp.
    """
    fsym = to_futures_symbol(symbol)
    end_ms   = int(time.time() * 1000)
    PAGE     = 200               # KuCoin hard cap per request
    all_rows: dict = {}          # ts_s → row  (dedup key)
    pages_needed = -(-limit // PAGE)  # ceil division
    for page in range(pages_needed):
        page_end   = end_ms   - page * PAGE * 3600 * 1000
        page_start = page_end - PAGE  * 3600 * 1000
        try:
            data = kucoin_futures_get("/api/v1/kline/query", {
                "symbol": fsym, "granularity": 60,
                "from": page_start, "to": page_end,
            })
            if not data:
                break
            for c in data:
                ts_s = int(c[0]) // 1000
                all_rows[ts_s] = [ts_s, c[1], c[4], c[2], c[3], c[5], c[5]]
        except Exception as e:
            print(f"[WARN] futures klines page={page} {fsym}: {e}", file=sys.stderr)
            break
        time.sleep(0.1)  # be polite between pages
    if not all_rows:
        return []
    result = sorted(all_rows.values(), key=lambda x: x[0])
    return result[-limit:]  # return exactly limit rows (newest)


def _get_funding_rate(symbol: str) -> float | None:
    """Fetch current funding rate from KuCoin Futures. Returns None on failure.

    Positive funding = longs pay shorts. Negative = shorts pay longs.
    Skip longs when deeply negative (cost of carry erodes profit).
    Accepts spot-style symbols (BTC-USDT) and maps them to futures contracts.
    """
    try:
        url = f"https://api-futures.kucoin.com/api/v1/funding-rate/{to_futures_symbol(symbol)}/current"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return float(data.get("data", {}).get("value", 0.0))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Technical analysis (pure Python, no pandas/numpy dependency)
# ---------------------------------------------------------------------------

def to_floats(candles: list[list], idx: int) -> list[float]:
    """Extract float column from candle data."""
    return [float(c[idx]) for c in candles]


def sma(values: list[float], period: int) -> list[float | None]:
    """Simple Moving Average."""
    result = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1: i + 1]) / period
    return result


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential Moving Average."""
    result: list[float | None] = [None] * len(values)
    k = 2.0 / (period + 1)
    # Seed with SMA
    if len(values) < period:
        return result
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        val = values[i] * k + prev * (1 - k)
        result[i] = val
        prev = val
    return result


def rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Relative Strength Index."""
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0)
        loss = max(-delta, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd(closes: list[float], fast: int = 12, slow: int = 26, signal_period: int = 9):
    """MACD line, signal line, histogram. Returns three lists."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    # Signal line = EMA of MACD line
    macd_vals = [v if v is not None else 0.0 for v in macd_line]
    sig = ema(macd_vals, signal_period)
    hist: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if macd_line[i] is not None and sig[i] is not None:
            hist[i] = macd_line[i] - sig[i]
    return macd_line, sig, hist


def volume_spike(volumes: list[float], lookback: int = 20, threshold: float = 2.0) -> list[bool]:
    """True if current volume > threshold * average of last `lookback` bars."""
    result = [False] * len(volumes)
    for i in range(lookback, len(volumes)):
        avg = sum(volumes[i - lookback: i]) / lookback
        if avg > 0 and volumes[i] > threshold * avg:
            result[i] = True
    return result


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(symbol: str, candles: list[list]) -> dict | None:
    """Analyze candles and return a signal dict or None."""
    if len(candles) < 52:
        return None

    closes = to_floats(candles, 2)  # close price
    volumes = to_floats(candles, 5)

    rsi_vals = rsi(closes)
    macd_line, macd_sig, macd_hist = macd(closes)
    sma50 = sma(closes, 50)
    sma20 = sma(closes, 20)
    vol_spike = volume_spike(volumes)

    # Latest values
    i = len(closes) - 1
    cur_rsi = rsi_vals[i]
    cur_hist = macd_hist[i]
    prev_hist = macd_hist[i - 1] if i > 0 else None
    cur_sma50 = sma50[i]
    cur_sma20 = sma20[i]
    cur_vol_spike = vol_spike[i]
    cur_price = closes[i]

    if cur_rsi is None or cur_hist is None or prev_hist is None:
        return None
    if cur_sma50 is None or cur_sma20 is None:
        return None

    # Score each indicator independently: +1 bullish, -1 bearish, 0 neutral
    reasons = []

    # RSI
    if cur_rsi < 30:
        rsi_vote = 1
        reasons.append(f"RSI oversold ({cur_rsi:.1f})")
    elif cur_rsi > 70:
        rsi_vote = -1
        reasons.append(f"RSI overbought ({cur_rsi:.1f})")
    else:
        rsi_vote = 0  # neutral — does not block other signals

    # MACD crossover (strongest momentum signal)
    if prev_hist < 0 and cur_hist > 0:
        macd_vote = 1
        reasons.append("MACD bullish crossover")
    elif prev_hist > 0 and cur_hist < 0:
        macd_vote = -1
        reasons.append("MACD bearish crossover")
    elif cur_hist > 0:
        macd_vote = 1
        reasons.append("MACD histogram positive (bullish trend)")
    else:
        macd_vote = -1
        reasons.append("MACD histogram negative (bearish trend)")

    # SMA cross (20 vs 50)
    if cur_sma20 > cur_sma50:
        sma_vote = 1
        reasons.append("SMA20 > SMA50 (bullish)")
    else:
        sma_vote = -1
        reasons.append("SMA20 < SMA50 (bearish)")

    # Volume spike (amplifier — not a vote, but boosts confidence)
    if cur_vol_spike:
        reasons.append("Volume spike detected")

    # 2-of-3 consensus: need at least 2 indicators agreeing on direction
    # Neutral votes (RSI=0) don't count against — only active votes matter
    votes = [rsi_vote, macd_vote, sma_vote]
    bullish_votes = sum(1 for v in votes if v > 0)
    bearish_votes = sum(1 for v in votes if v < 0)

    if bullish_votes >= 2:
        side = "BUY"
        raw_score = bullish_votes / 3.0
    elif bearish_votes >= 2:
        side = "SELL"
        raw_score = bearish_votes / 3.0
    else:
        return None  # no consensus

    # Volume spike amplifies confidence
    if cur_vol_spike:
        raw_score = min(1.0, raw_score * 1.3)

    raw_score = round(raw_score, 4)

    return {
        "symbol": symbol,
        "side": side,
        "score": round(raw_score, 4),
        "price": cur_price,
        "rsi": round(cur_rsi, 2),
        "macd_hist": round(cur_hist, 6),
        "sma20": round(cur_sma20, 4),
        "sma50": round(cur_sma50, 4),
        "volume_spike": cur_vol_spike,
        "reasons": reasons,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# ML signal engine (XGBoost+LightGBM ensemble, falls back to rule-based)
# ---------------------------------------------------------------------------

try:
    import pandas as _pd
    from quantforge_ml import generate_signal as _ml_generate_signal
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False


def _get_regime():
    """Load regime with 4h cache — returns neutral dict on any failure."""
    try:
        from quantforge_regime import get_regime
        return get_regime()
    except Exception:
        return {"score": 0, "label": "NEUTRAL", "long_adj": 0.04, "short_adj": 0.04,
                "size_mult": 0.80, "signals": {}}


def _regime_controls(regime_score: float) -> dict:
    """Regime-gated trading controls with linear interpolation.

    Returns dict with:
      size_mult: float (0.40 to 1.0, linear)
      max_positions: int (0 to MAX_POSITIONS)
      majors_only: bool
    """
    score = float(regime_score)

    # Linear size multiplier: 0.40 at score=-0.20, 1.0 at score=0.60
    size_mult = max(0.40, min(1.0, 0.40 + (score + 0.20) * 0.75))

    # Stepped max_positions with hysteresis buffer
    if score > 0.40:
        max_pos = min(MAX_POSITIONS, 3)
    elif score > 0.10:
        max_pos = min(MAX_POSITIONS, 2)
    elif score > -0.15:
        max_pos = min(MAX_POSITIONS, 1)
    else:
        max_pos = 0  # manage existing only, no new entries

    majors_only = score < 0.10

    return {
        "size_mult": round(size_mult, 3),
        "max_positions": max_pos,
        "majors_only": majors_only,
    }


def _non_actionable_signal(symbol: str, regime: dict, price: float, result: dict, *, decision_stage: str) -> dict:
    return {
        "symbol": symbol,
        "side": "HOLD",
        "actionable": False,
        "decision_stage": decision_stage,
        "score": float(result.get("confidence", 0.0) or 0.0),
        "raw_score": float(result.get("confidence", 0.0) or 0.0),
        "ml_confidence": float(result.get("confidence", 0.0) or 0.0),
        "long_confidence": float(result.get("long_confidence", result.get("confidence", 0.0)) or 0.0),
        "short_confidence": float(result.get("short_confidence", 0.0) or 0.0),
        "long_threshold": float(result.get("long_threshold", result.get("threshold", 0.0)) or 0.0),
        "short_threshold": float(result.get("short_threshold", result.get("threshold", 0.0)) or 0.0),
        "price": float(price or 0.0),
        "reasons": list(result.get("reason", []) or ["no actionable signal"]),
        "setup_tag": result.get("setup_tag", "unknown"),
        "setup_score": float(result.get("setup_score", 0.0) or 0.0),
        "dominant_setup_tag": result.get("dominant_setup_tag", result.get("setup_tag", "unknown")),
        "dominant_setup_score": float(result.get("dominant_setup_score", result.get("setup_score", 0.0)) or 0.0),
        "dominant_setup_direction": result.get("dominant_setup_direction"),
        "dominant_long_setup_tag": result.get("dominant_long_setup_tag"),
        "dominant_long_setup_score": float(result.get("dominant_long_setup_score", 0.0) or 0.0),
        "dominant_short_setup_tag": result.get("dominant_short_setup_tag"),
        "dominant_short_setup_score": float(result.get("dominant_short_setup_score", 0.0) or 0.0),
        "policy_score": float(result.get("policy_score", 0.0) or 0.0),
        "risk_filter_profile": result.get("risk_filter_profile", "legacy"),
        "redesign_active": bool(result.get("redesign_active", False)),
        "regime": regime["label"],
        "regime_score": regime.get("score"),
        "regime_entropy": regime.get("entropy"),
        "regime_entropy_label": regime.get("entropy_label"),
        "regime_volatility": regime.get("volatility"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def generate_signals_ml(symbol: str, candles: list, regime: dict | None = None, trial: dict | None = None) -> dict | None:
    """ML-powered signal generation with regime-aware adaptive thresholds.

    The regime detector classifies the current market as BULL/BEAR/NEUTRAL
    and adjusts confidence thresholds + position sizing accordingly:
      - BEAR regime: LONG threshold raised (harder to go long against trend)
      - BULL regime: LONG threshold slightly lowered (ride the trend)
      - High vol:    all thresholds raised, size_mult shrunk
    """
    if not _ML_AVAILABLE:
        return generate_signals(symbol, candles)

    model_meta_path = os.path.join(DATA_DIR, "model", "model_meta.json")
    if not os.path.exists(model_meta_path):
        return generate_signals(symbol, candles)

    # Load regime (4h cached — shared across all symbols in same run)
    if regime is None:
        regime = _get_regime()

    try:
        import pandas as pd
        # KuCoin candle format: [ts, open, CLOSE, high, low, volume, turnover]
        df = pd.DataFrame(candles, columns=["ts", "open", "raw_close", "high", "low", "volume", "turnover"])
        df["close"] = df["raw_close"].astype(float)
        for col in ["open", "high", "low", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        df["ts"] = df["ts"].astype(int)
        df = df[["ts", "open", "high", "low", "close", "volume", "turnover"]].sort_values("ts").reset_index(drop=True)

        # Load model thresholds then apply regime adjustment
        import json as _json
        meta = _json.loads(open(model_meta_path).read())
        trained_pairs = set(meta.get("pairs", []))
        if ML_ONLY_TRAINED_PAIRS and trained_pairs and symbol not in trained_pairs:
            return _non_actionable_signal(
                symbol,
                regime,
                float(df["close"].iloc[-1]),
                {
                    "confidence": 0.0,
                    "long_confidence": 0.0,
                    "short_confidence": 0.0,
                    "long_threshold": 0.0,
                    "short_threshold": 0.0,
                    "reason": ["symbol not in trained_pairs allowlist"],
                },
                decision_stage="trained_pair_blocked",
            )
        base_long_thr  = float(meta.get("optimal_threshold", 0.80))
        live_long_thr = max(0.50, base_long_thr + LONG_LIVE_THRESHOLD_OFFSET)
        adj_long_thr   = min(LONG_LIVE_THRESHOLD_CAP, live_long_thr + regime.get("long_adj", 0.0))
        trial_long_relief, trial_short_relief = _trial_threshold_relief(trial)
        if candidate_trial_is_active(trial):
            adj_long_thr = max(0.50, adj_long_thr - trial_long_relief)
        short_meta_path = os.path.join(DATA_DIR, "model", "model_meta_short.json")
        base_short_thr = 0.80
        if os.path.exists(short_meta_path):
            try:
                base_short_thr = float(_json.loads(open(short_meta_path).read()).get("optimal_threshold", 0.80))
            except Exception:
                base_short_thr = 0.80
        adj_short_thr = min(0.95, max(0.50, base_short_thr + regime.get("short_adj", 0.0)))
        adj_short_thr = min(adj_short_thr, SHORT_LIVE_THRESHOLD_CAP)
        if candidate_trial_is_active(trial):
            adj_short_thr = max(0.35, adj_short_thr - trial_short_relief)

        close_now = float(df["close"].iloc[-1])
        ret_4h = (close_now / float(df["close"].iloc[-5]) - 1.0) if len(df) >= 5 else 0.0
        ret_24h = (close_now / float(df["close"].iloc[-25]) - 1.0) if len(df) >= 25 else 0.0
        if ret_24h <= PANIC_SHORT_RET24H_PCT or ret_4h <= PANIC_SHORT_RET4H_PCT:
            adj_short_thr = max(0.35, adj_short_thr + PANIC_SHORT_THRESHOLD_ADJ)

        result = _ml_generate_signal(symbol, df,
                                     long_threshold_override=adj_long_thr,
                                     short_threshold_override=adj_short_thr,
                                     allow_gate_bypass=_allow_candidate_trial_gate_bypass(trial))
        if result["signal"] in ("BUY", "SELL"):
            size_mult = regime.get("size_mult", 1.0)
            threshold_line = f"LONG thr {adj_long_thr:.3f}, SHORT thr {adj_short_thr:.3f}"
            setup_tag = result.get("setup_tag")
            setup_score = float(result.get("setup_score", 0.0) or 0.0)
            if not setup_tag:
                setup_tag, setup_score = _infer_setup_context(result["signal"], candles)
            return {
                "symbol": symbol,
                "side": result["signal"],  # BUY=LONG, SELL=SHORT
                "score": result["confidence"],
                "raw_score": result["confidence"],
                "entry_threshold": adj_long_thr if result["signal"] == "BUY" else adj_short_thr,
                "policy_score": float(result.get("policy_score", result["confidence"])),
                "price": float(df["close"].iloc[-1]),
                "rsi": 0.0,
                "macd_hist": 0.0,
                "sma20": 0.0,
                "sma50": 0.0,
                "volume_spike": False,
                "reasons": result["reason"] + [
                    f"Regime: {regime['label']} (score={regime['score']})",
                    f"{threshold_line} from regime adjustments",
                ],
                "ts": datetime.now(timezone.utc).isoformat(),
                "ml_confidence": result["confidence"],
                "regime": regime["label"],
                "regime_score": regime.get("score"),
                "regime_entropy": regime.get("entropy"),
                "regime_entropy_label": regime.get("entropy_label"),
                "regime_volatility": regime.get("volatility"),
                "size_mult": size_mult,
                "setup_tag": setup_tag,
                "setup_score": setup_score,
                "fakeout_risk": float(result.get("fakeout_risk", 0.0) or 0.0),
                "risk_filter_profile": result.get("risk_filter_profile", "legacy"),
                "redesign_active": bool(result.get("redesign_active", False)),
            }
        result = dict(result)
        result["long_threshold"] = adj_long_thr
        result["short_threshold"] = adj_short_thr
        if (
            ENABLE_RULE_FALLBACK_WHEN_ML_DEGENERATE
            and max(float(result.get("long_confidence", 0.0) or 0.0), float(result.get("short_confidence", 0.0) or 0.0)) <= RULE_FALLBACK_MAX_ML_CONFIDENCE
            and symbol in TOP_ALT_EXPANSION_SYMBOLS
        ):
            fallback = generate_signals(symbol, candles)
            if fallback and fallback.get("side") == "BUY" and float(fallback.get("score", 0.0) or 0.0) >= RULE_FALLBACK_MIN_SCORE:
                setup_tag, setup_score = _infer_setup_context("BUY", candles)
                fallback_reasons = list(fallback.get("reasons", []))
                fallback_reasons.append(
                    f"ML confidence degenerate ({float(result.get('long_confidence', 0.0) or 0.0):.3f}); using rule fallback on liquid symbol"
                )
                fallback_reasons.append(
                    f"Rule fallback threshold {RULE_FALLBACK_MIN_SCORE:.3f}"
                )
                fallback["entry_threshold"] = RULE_FALLBACK_MIN_SCORE
                fallback["raw_score"] = fallback.get("score")
                fallback["policy_score"] = fallback.get("score")
                fallback["ml_confidence"] = fallback.get("score")
                fallback["setup_tag"] = setup_tag
                fallback["setup_score"] = setup_score
                fallback["regime"] = regime["label"]
                fallback["regime_score"] = regime.get("score")
                fallback["regime_entropy"] = regime.get("entropy")
                fallback["regime_entropy_label"] = regime.get("entropy_label")
                fallback["regime_volatility"] = regime.get("volatility")
                fallback["size_mult"] = regime.get("size_mult", 1.0)
                fallback["risk_filter_profile"] = "rule_fallback_when_ml_degenerate"
                fallback["reasons"] = fallback_reasons
                return fallback
        decision_stage = "threshold_miss"
        hold_reasons = " ".join(str(r) for r in (result.get("reason") or []))
        if "gate failed" in hold_reasons.lower():
            decision_stage = "ml_gate_blocked"
        return _non_actionable_signal(
            symbol,
            regime,
            float(df["close"].iloc[-1]),
            result,
            decision_stage=decision_stage,
        )
    except Exception:
        return generate_signals(symbol, candles)


# ---------------------------------------------------------------------------
# Coin screening
# ---------------------------------------------------------------------------

def screen_coins(top_n: int = 20) -> list[dict]:
    """Screen top futures contracts by 24h USDT turnover."""
    pairs = []
    for c in get_futures_tickers():
        sym = c.get("symbol", "")
        if not sym.endswith("USDTM") or c.get("status") != "Open":
            continue
        base = c.get("baseCurrency", sym.replace("USDTM", ""))
        if base == "XBT": base = "BTC"
        if base[:1].isdigit() or len(base) > 8 or base in STABLECOINS:
            continue
        vol = float(c.get("turnoverOf24h", 0) or 0)   # USDT turnover
        price = float(c.get("lastTradePrice", c.get("markPrice", 0)) or 0)
        if vol < MIN_VOLUME_USDT or price <= 0:
            continue
        pairs.append({"symbol": f"{base}-USDT", "futures_symbol": sym,
                      "base": base, "price": price,
                      "change_pct": float(c.get("priceChgPct", 0)) * 100,
                      "vol_usdt": vol,
                      "open_interest": float(c.get("openInterest", 0) or 0)})
    pairs.sort(key=lambda x: x["vol_usdt"], reverse=True)
    return pairs[:top_n]


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(max(var, 0.0))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _historical_1h_candle_count(symbol: str) -> int:
    safe = symbol.replace("-", "_")
    path = os.path.join(cfg.data, "quantforge", "historical", f"{safe}_1h.csv")
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            # subtract header if present
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def _passes_symbol_quality_filters(symbol_info: dict, candles: list[list]) -> tuple[bool, list[str], dict]:
    """Reject low-quality symbols before they become actionable trades."""
    symbol = str(symbol_info.get("symbol", "") or "")
    if symbol in MAJOR_SYMBOLS:
        min_history_candles = QUALITY_MIN_HISTORY_CANDLES_MAJOR
    elif symbol in TOP_ALT_EXPANSION_SYMBOLS:
        min_history_candles = QUALITY_MIN_HISTORY_CANDLES_TOP_ALT
    else:
        min_history_candles = QUALITY_MIN_HISTORY_CANDLES
    metrics = {
        "quality_recent_24h_turnover": 0.0,
        "quality_realized_vol_24h": 0.0,
        "quality_abs_move_24h_pct": 0.0,
        "quality_history_candles": _historical_1h_candle_count(symbol),
        "quality_score": 0.0,
    }
    if not QUALITY_FILTER_ENABLED:
        return True, [], metrics
    if not candles:
        return False, ["No candle history"], metrics

    reasons = []
    price = float(symbol_info.get("price", 0.0) or 0.0)
    turnover_24h = float(symbol_info.get("vol_usdt", 0.0) or 0.0)
    if turnover_24h < QUALITY_MIN_24H_TURNOVER_USDT:
        reasons.append(f"24h turnover ${turnover_24h:,.0f} < ${QUALITY_MIN_24H_TURNOVER_USDT:,.0f}")
    if price < QUALITY_MIN_PRICE:
        reasons.append(f"Price ${price:.4f} < ${QUALITY_MIN_PRICE:.4f}")
    if metrics["quality_history_candles"] < min_history_candles:
        reasons.append(f"History {metrics['quality_history_candles']} candles < {min_history_candles}")

    closes = [float(c[2]) for c in candles if len(c) > 2]
    recent_turnover = [float(c[6]) for c in candles[-24:] if len(c) > 6]
    metrics["quality_recent_24h_turnover"] = float(sum(recent_turnover))
    if metrics["quality_recent_24h_turnover"] < QUALITY_MIN_RECENT_24H_TURNOVER_USDT:
        reasons.append(
            f"Recent 24h turnover ${metrics['quality_recent_24h_turnover']:,.0f} < ${QUALITY_MIN_RECENT_24H_TURNOVER_USDT:,.0f}"
        )

    if len(closes) >= 25 and closes[-25] > 0 and closes[-1] > 0:
        metrics["quality_abs_move_24h_pct"] = abs((closes[-1] / closes[-25] - 1.0) * 100.0)
        if metrics["quality_abs_move_24h_pct"] > QUALITY_MAX_ABS_24H_MOVE_PCT:
            reasons.append(f"24h move {metrics['quality_abs_move_24h_pct']:.1f}% > {QUALITY_MAX_ABS_24H_MOVE_PCT:.1f}%")
        log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - 24, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
        metrics["quality_realized_vol_24h"] = _stddev(log_rets) * math.sqrt(max(len(log_rets), 1))
        if metrics["quality_realized_vol_24h"] > QUALITY_MAX_REALIZED_VOL_24H:
            reasons.append(f"24h realized vol {metrics['quality_realized_vol_24h']:.2f} > {QUALITY_MAX_REALIZED_VOL_24H:.2f}")

    history_score = _clamp(metrics["quality_history_candles"] / max(min_history_candles * 2, 1), 0.0, 1.0)
    turnover_score = _clamp(turnover_24h / max(QUALITY_MIN_24H_TURNOVER_USDT * 3, 1), 0.0, 1.0)
    recent_turnover_score = _clamp(metrics["quality_recent_24h_turnover"] / max(QUALITY_MIN_RECENT_24H_TURNOVER_USDT * 3, 1), 0.0, 1.0)
    price_score = _clamp(price / max(QUALITY_MIN_PRICE * 20, 1e-6), 0.0, 1.0)
    move_penalty = _clamp(metrics["quality_abs_move_24h_pct"] / max(QUALITY_MAX_ABS_24H_MOVE_PCT, 1.0), 0.0, 1.0)
    vol_penalty = _clamp(metrics["quality_realized_vol_24h"] / max(QUALITY_MAX_REALIZED_VOL_24H, 1e-6), 0.0, 1.0)
    metrics["quality_score"] = round(
        0.25 * history_score
        + 0.25 * turnover_score
        + 0.20 * recent_turnover_score
        + 0.10 * price_score
        + 0.10 * (1.0 - move_penalty)
        + 0.10 * (1.0 - vol_penalty),
        4,
    )
    if metrics["quality_score"] < QUALITY_MIN_SCORE:
        reasons.append(f"Quality score {metrics['quality_score']:.2f} < {QUALITY_MIN_SCORE:.2f}")

    return len(reasons) == 0, reasons, metrics


def _infer_setup_context(side: str, candles: list[list]) -> tuple[str, float]:
    closes = [float(c[2]) for c in candles if len(c) > 2]
    highs = [float(c[3]) for c in candles if len(c) > 3]
    lows = [float(c[4]) for c in candles if len(c) > 4]
    if len(closes) < 50:
        return ("unknown", 0.0)

    close_now = closes[-1]
    ret_4h = close_now / closes[-5] - 1.0 if closes[-5] > 0 else 0.0
    ret_24h = close_now / closes[-25] - 1.0 if closes[-25] > 0 else 0.0
    range_24h = (max(highs[-24:]) / min(lows[-24:]) - 1.0) if min(lows[-24:]) > 0 else 0.0
    sma20 = sum(closes[-20:]) / 20.0
    sma50 = sum(closes[-50:]) / 50.0
    high_24h = max(highs[-24:])
    low_24h = min(lows[-24:])
    close_vs_high = close_now / high_24h - 1.0 if high_24h > 0 else 0.0
    close_vs_low = close_now / low_24h - 1.0 if low_24h > 0 else 0.0
    pulled_back = closes[-12] / max(closes[-25], 1e-9) - 1.0 if closes[-25] > 0 else 0.0
    recovered = close_now / max(closes[-12], 1e-9) - 1.0 if closes[-12] > 0 else 0.0

    if side == "BUY":
        if sma20 > sma50 and ret_24h >= 0.04 and ret_4h >= 0.01:
            trend_score = 0.58 + max(0.0, ret_24h) * 2.0 + max(0.0, ret_4h) * 3.0
            return ("trend_long", round(min(1.0, trend_score), 4))
        if sma20 >= sma50 and ret_4h >= 0.018 and close_vs_high >= -0.01 and range_24h <= 0.16:
            breakout_score = 0.52 + max(0.0, ret_4h) * 5.0 + max(0.0, close_vs_high + 0.01) * 4.0
            return ("breakout_long", round(min(1.0, breakout_score), 4))
        if pulled_back <= -0.02 and recovered >= 0.015 and close_vs_low >= 0.03:
            rebound_score = 0.50 + abs(min(0.0, pulled_back)) * 2.0 + max(0.0, recovered) * 3.0
            return ("rebound_long", round(min(1.0, rebound_score), 4))
        return ("generic_long", round(min(1.0, 0.40 + max(ret_4h, 0.0) * 6.0), 4))

    if ret_24h <= -0.05 and ret_4h <= -0.015:
        return ("trend_short", round(min(1.0, 0.55 + abs(ret_24h) * 3.0), 4))
    if ret_24h >= 0.06 and ret_4h <= 0.0:
        return ("exhaustion_short", round(min(1.0, 0.50 + ret_24h * 2.5), 4))
    return ("generic_short", round(min(1.0, 0.40 + max(-ret_4h, 0.0) * 6.0), 4))


# ---------------------------------------------------------------------------
# Portfolio management
# ---------------------------------------------------------------------------

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            port = json.load(f)
        _normalize_position_state(port)
        _reconcile_cash_ledger(port)
        _repair_loss_lockout_state(port)
        if port.get("equity") is None:
            _refresh_portfolio_equity_snapshot(port)
        return port
    return {
        "cash": STARTING_BALANCE,
        "equity": STARTING_BALANCE,
        "starting_balance": STARTING_BALANCE,
        "positions": {},
        "last_exit_ts_by_symbol": {},
        "loss_streak_by_symbol": {},
        "loss_lockout_until_by_symbol": {},
        "realized_pnl": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_fees_paid": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "peak_equity": STARTING_BALANCE,
        "max_drawdown": 0.0,
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
    }


def save_portfolio(port: dict):
    if port.get("equity") is None:
        _refresh_portfolio_equity_snapshot(port)
    port["updated"] = datetime.now(timezone.utc).isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(port, f, indent=2)


def append_trade(trade: dict):
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(trade) + "\n")


class _RunLock:
    """Simple non-blocking process lock to prevent overlapping paper runs."""

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        self._fh = open(self.path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fh.close()
            self._fh = None
            return False
        self._fh.write(f"{os.getpid()}\n")
        self._fh.flush()
        return True

    def release(self):
        if not self._fh:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def append_signal(sig: dict):
    with open(SIGNALS_FILE, "a") as f:
        f.write(json.dumps(sig) + "\n")


def _signal_rank_value(row: dict) -> float:
    return signal_rank_value(row)


def _top_pick_rows(signals: list[dict], limit: int = 10) -> list[dict]:
    ranked = sorted(signals, key=_signal_rank_value, reverse=True)
    picks = []
    for row in ranked[:limit]:
        picks.append({
            "symbol": row.get("symbol"),
            "side": row.get("side"),
            "score": round(float(row.get("score", 0.0) or 0.0), 4),
            "edge_rank": round(float(row.get("edge_rank", _signal_rank_value(row)) or 0.0), 6),
            "raw_score": round(float(row.get("raw_score", row.get("score", 0.0)) or 0.0), 4),
            "setup_tag": row.get("setup_tag"),
            "quality_score": round(float((row.get("quality_metrics") or {}).get("quality_score", 0.0) or 0.0), 4),
            "regime": row.get("regime"),
            "ts": row.get("ts"),
            "reasons": (row.get("reasons") or [])[:4],
        })
    return picks


def _summarize_scan_results(results: list[dict]) -> dict:
    counts = {"signal": 0, "hold": 0, "skip": 0, "error": 0, "open_position": 0}
    blocked_reasons = {}
    for row in results:
        status = str(row.get("status", "") or "")
        if status in counts:
            counts[status] += 1
        reason = str(row.get("reason", "") or "")
        if status in {"skip", "hold"} and reason:
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
    return {
        "counts": counts,
        "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
    }


def save_last_scan(report: dict):
    signals = report.get("signals") or []
    results = report.get("results") or []
    flow = report.get("flow") or {}
    ts = report.get("ts") or datetime.now(timezone.utc).isoformat()
    report["ts"] = ts
    report["generated_at"] = ts
    report["top_picks"] = _top_pick_rows(signals)
    report["pick_count"] = len(report["top_picks"])
    report["summary"] = _summarize_scan_results(results)
    if isinstance(flow, dict):
        for key, value in flow.items():
            report[key] = value
        if "actionable" not in report and "actionable_signals" in flow:
            report["actionable"] = int(flow.get("actionable_signals", 0) or 0)
    counts = (report.get("summary") or {}).get("counts") or {}
    report["signal_count"] = int(counts.get("signal", len(signals)) or 0)
    report["result_count"] = int(sum(int(v or 0) for v in counts.values()))
    with open(LAST_SCAN_FILE, "w") as f:
        json.dump(report, f, indent=2)


def save_last_execution(report: dict):
    ts = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    report["generated_at"] = ts
    report["ts"] = report.get("ts") or ts
    if not report.get("mode"):
        report["mode"] = str(report.get("autopilot_mode", "") or "")
    with open(LAST_EXECUTION_FILE, "w") as f:
        json.dump(report, f, indent=2)


def _save_idle_execution_report(
    *,
    autopilot: dict | None,
    trial: dict | None,
    signal_count: int,
    execution_permission: str,
    idle_reason: str,
    details: list[str] | None = None,
):
    save_last_execution({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trial_active": candidate_trial_is_active(trial),
        "autopilot_mode": str((autopilot or {}).get("mode", "") or ""),
        "execution_permission": execution_permission,
        "signal_count": int(signal_count),
        "executed_count": 0,
        "executed_symbols": [],
        "skipped_count": 0,
        "skip_summary": {},
        "skipped": [],
        "idle_reason": idle_reason,
        "details": list(details or [])[:10],
    })


def normalize_last_scan_artifact() -> dict:
    try:
        with open(LAST_SCAN_FILE) as f:
            report = json.load(f)
    except Exception:
        return {}
    if not isinstance(report, dict):
        return {}
    save_last_scan(report)
    return report


def _open_margin_total(port: dict) -> float:
    return sum(float(pos.get("margin", 0.0)) for pos in port.get("positions", {}).values())


def _open_entry_fee_total(port: dict) -> float:
    return sum(float(pos.get("entry_fee", 0.0)) for pos in port.get("positions", {}).values())


def _open_spot_cost_total(port: dict) -> float:
    total = 0.0
    for pos in port.get("positions", {}).values():
        if is_leveraged_position(pos):
            continue
        total += float(pos.get("qty", 0.0)) * float(pos.get("entry_price", 0.0))
    return total


def _reconcile_cash_ledger(port: dict):
    """Repair free-cash drift from earlier spot-style long close accounting."""
    positions = port.get("positions", {})
    if not positions:
        return
    expected_cash = (
        float(port.get("starting_balance", STARTING_BALANCE))
        + float(port.get("realized_pnl", 0.0))
        - _open_margin_total(port)
        - _open_spot_cost_total(port)
        - _open_entry_fee_total(port)
    )
    if abs(float(port.get("cash", 0.0)) - expected_cash) > 0.01:
        port["cash"] = round(expected_cash, 4)
        port["cash_reconciled_at"] = datetime.now(timezone.utc).isoformat()


def _recent_close_feedback(now_iso: str) -> dict:
    summary = {
        "window_hours": FEEDBACK_LOOKBACK_HOURS,
        "recent_closes": 0,
        "wins": 0,
        "losses": 0,
        "avg_pnl": 0.0,
        "win_rate": 0.0,
        "risk_mult": 1.0,
    }
    if not AUTO_FEEDBACK_ENABLED or not os.path.exists(TRADES_FILE):
        return {"summary": summary, "by_symbol": {}, "by_setup": {}, "by_symbol_setup": {}}

    try:
        now_dt = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
    except Exception:
        now_dt = datetime.now(timezone.utc)

    recent = []
    with open(TRADES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("type") != "CLOSE":
                continue
            ts = row.get("ts")
            try:
                ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except Exception:
                continue
            if (now_dt - ts_dt).total_seconds() > FEEDBACK_LOOKBACK_HOURS * 3600:
                continue
            row["_ts_dt"] = ts_dt
            recent.append(row)

    recent = recent[-FEEDBACK_RECENT_CLOSE_LIMIT:]
    if not recent:
        return {"summary": summary, "by_symbol": {}, "by_setup": {}, "by_symbol_setup": {}}

    def _bucket(target: dict, key: str, row: dict):
        bucket = target.setdefault(key, {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "last_ts": None})
        pnl = float(row.get("pnl", 0.0) or 0.0)
        bucket["trades"] += 1
        bucket["total_pnl"] += pnl
        if pnl > 0:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
        bucket["last_ts"] = row.get("ts")

    by_symbol = {}
    by_setup = {}
    by_symbol_setup = {}
    total_pnl = 0.0
    wins = 0
    for row in recent:
        pnl = float(row.get("pnl", 0.0) or 0.0)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        _bucket(by_symbol, row.get("symbol", "unknown"), row)
        setup_tag = row.get("setup_tag")
        if setup_tag:
            _bucket(by_setup, setup_tag, row)
            _bucket(by_symbol_setup, f"{row.get('symbol', 'unknown')}::{setup_tag}", row)

    for buckets in (by_symbol, by_setup, by_symbol_setup):
        for stats in buckets.values():
            stats["avg_pnl"] = round(stats["total_pnl"] / max(stats["trades"], 1), 4)
            stats["win_rate"] = round(stats["wins"] / max(stats["trades"], 1), 4)
            stats["total_pnl"] = round(stats["total_pnl"], 4)

    losses = len(recent) - wins
    avg_pnl = total_pnl / max(len(recent), 1)
    risk_mult = 1.0
    if ADAPTIVE_RISK_ENABLED:
        if avg_pnl < 0:
            risk_mult -= min(0.40, abs(avg_pnl) / 20.0)
        if wins / max(len(recent), 1) < 0.35:
            risk_mult -= 0.15
        risk_mult = _clamp(risk_mult, ADAPTIVE_RISK_FLOOR, ADAPTIVE_RISK_CEIL)

    summary.update({
        "recent_closes": len(recent),
        "wins": wins,
        "losses": losses,
        "avg_pnl": round(avg_pnl, 4),
        "win_rate": round(wins / max(len(recent), 1), 4),
        "risk_mult": round(risk_mult, 4),
    })
    return {"summary": summary, "by_symbol": by_symbol, "by_setup": by_setup, "by_symbol_setup": by_symbol_setup}


def _feedback_adjustments(symbol: str, setup_tag: str, feedback: dict, now_iso: str) -> tuple[bool, list[str], float, float]:
    if not AUTO_FEEDBACK_ENABLED:
        return True, [], 0.0, 1.0

    reasons = []
    score_penalty = 0.0
    size_mult = float(feedback.get("summary", {}).get("risk_mult", 1.0) or 1.0)

    try:
        now_dt = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
    except Exception:
        now_dt = datetime.now(timezone.utc)

    symbol_setup_key = f"{symbol}::{setup_tag}"
    symbol_setup_stats = feedback.get("by_symbol_setup", {}).get(symbol_setup_key)
    if symbol_setup_stats:
        if (
            symbol_setup_stats.get("trades", 0) >= SYMBOL_SETUP_QUARANTINE_MIN_TRADES
            and symbol_setup_stats.get("win_rate", 1.0) <= SYMBOL_SETUP_BAD_WIN_RATE
            and symbol_setup_stats.get("avg_pnl", 0.0) <= SYMBOL_SETUP_BAD_AVG_PNL
            and symbol_setup_stats.get("last_ts")
        ):
            try:
                last_dt = datetime.fromisoformat(str(symbol_setup_stats["last_ts"]).replace("Z", "+00:00"))
                if now_dt < last_dt + timedelta(hours=SYMBOL_SETUP_QUARANTINE_HOURS):
                    return False, [f"symbol+setup quarantine — {symbol} {setup_tag} weak recently"], 1.0, ADAPTIVE_RISK_FLOOR
            except Exception:
                pass
        if symbol_setup_stats.get("avg_pnl", 0.0) < 0:
            score_penalty += min(0.12, abs(float(symbol_setup_stats["avg_pnl"])) / 35.0)
            size_mult *= 0.85
            reasons.append(f"symbol+setup drag {symbol} {setup_tag} avg_pnl {symbol_setup_stats['avg_pnl']:.2f}")

    sym_stats = feedback.get("by_symbol", {}).get(symbol)
    if sym_stats:
        recent_lossy = (
            sym_stats.get("losses", 0) >= SYMBOL_QUARANTINE_MIN_LOSSES
            and sym_stats.get("wins", 0) == 0
            and sym_stats.get("total_pnl", 0.0) < 0
        )
        if recent_lossy and sym_stats.get("last_ts"):
            try:
                last_dt = datetime.fromisoformat(str(sym_stats["last_ts"]).replace("Z", "+00:00"))
                if now_dt < last_dt + timedelta(hours=SYMBOL_QUARANTINE_HOURS):
                    return False, [f"auto quarantine — recent symbol losses ({sym_stats['losses']})"], 1.0, ADAPTIVE_RISK_FLOOR
            except Exception:
                pass
        if sym_stats.get("avg_pnl", 0.0) < 0:
            score_penalty += min(0.15, abs(float(sym_stats["avg_pnl"])) / 50.0)
            size_mult *= 0.9

    setup_stats = feedback.get("by_setup", {}).get(setup_tag)
    if setup_stats:
        if (
            setup_stats.get("trades", 0) >= SETUP_QUARANTINE_MIN_TRADES
            and setup_stats.get("win_rate", 1.0) <= SETUP_BAD_WIN_RATE
            and setup_stats.get("avg_pnl", 0.0) <= SETUP_BAD_AVG_PNL
            and setup_stats.get("last_ts")
        ):
            try:
                last_dt = datetime.fromisoformat(str(setup_stats["last_ts"]).replace("Z", "+00:00"))
                if now_dt < last_dt + timedelta(hours=SETUP_QUARANTINE_HOURS):
                    return False, [f"setup quarantine — {setup_tag} weak recently"], 1.0, ADAPTIVE_RISK_FLOOR
            except Exception:
                pass
        if setup_stats.get("avg_pnl", 0.0) < 0:
            score_penalty += min(0.10, abs(float(setup_stats["avg_pnl"])) / 40.0)
            size_mult *= 0.9
            reasons.append(f"setup drag {setup_tag} avg_pnl {setup_stats['avg_pnl']:.2f}")

    if feedback.get("summary", {}).get("recent_closes", 0) >= 4 and feedback["summary"].get("avg_pnl", 0.0) < 0:
        reasons.append(f"recent avg pnl {feedback['summary']['avg_pnl']:.2f}")

    return True, reasons, round(score_penalty, 4), round(_clamp(size_mult, ADAPTIVE_RISK_FLOOR, ADAPTIVE_RISK_CEIL), 4)


def _selection_adjustments(
    symbol: str, sig: dict, quality_metrics: dict, trial: dict | None = None, regime: dict | None = None
) -> tuple[bool, list[str], float, float]:
    reasons = []
    score_adj = 0.0
    size_mult = 1.0
    setup_tag = sig.get("setup_tag", "unknown")
    setup_score = float(sig.get("setup_score", 0.0) or 0.0)
    quality_score = float(quality_metrics.get("quality_score", 0.0) or 0.0)
    is_major = symbol in MAJOR_SYMBOLS
    in_top_alt_expansion = symbol in TOP_ALT_EXPANSION_SYMBOLS
    trial_active = candidate_trial_is_active(trial)
    trial_entry_profile = str(_trial_runtime_value(trial, "entry_profile", "") or "").lower()
    trial_disable_non_majors = bool(_trial_runtime_value(trial, "non_major_entries", False))
    trial_generic_policy = str(_trial_runtime_value(trial, "generic_long_policy", "") or "").lower()
    trial_symbol_universe = str(_trial_runtime_value(trial, "symbol_universe", "") or "").lower()
    trial_allowed_long_setups = {s.lower() for s in _trial_runtime_string_set(trial, "allowed_long_setups")}
    trial_allowed_long_symbols = {s for s in _trial_runtime_string_set(trial, "allowed_long_symbols")}
    trial_allowed_short_setups = {s.lower() for s in _trial_runtime_string_set(trial, "allowed_short_setups")}
    trial_max_fakeout_risk = _trial_runtime_value(trial, "max_fakeout_risk", None)
    trial_type = str((trial or {}).get("type", "") or "").lower()
    regime_label = str((regime or {}).get("label", "NEUTRAL")).upper()
    entropy_label = str((regime or {}).get("entropy_label", "MIXED")).upper()
    entropy_penalty = float((regime or {}).get("entropy_penalty", 0.0) or 0.0)
    regime_size_mult = float((regime or {}).get("size_mult", 1.0) or 1.0)
    fakeout_risk = float(sig.get("fakeout_risk", 0.0) or 0.0)
    try:
        trial_max_fakeout_risk = float(trial_max_fakeout_risk) if trial_max_fakeout_risk is not None else None
    except Exception:
        trial_max_fakeout_risk = None
    redesign_active = bool(sig.get("redesign_active", False)) or (trial_active and trial_type == "quantforge_redesign")
    entry_selection_policy = str(_trial_runtime_value(trial, "entry_selection", "") or "").lower()
    labeled_setup_required = redesign_active or (trial_active and entry_selection_policy == "require_regime_support_and_labeled_setup_alignment")
    execution_mode = _execution_realism_mode(trial, load_autopilot_report())
    blocks_fragile_context = execution_mode in {"research_hold", "rebuild_trial"}
    trial_allows_non_major_long = bool(symbol in trial_allowed_long_symbols)

    def _setup_min_score(setup: str) -> float:
        if setup in LONG_SETUP_MIN_SCORE:
            return float(LONG_SETUP_MIN_SCORE[setup])
        if setup == "generic_long":
            return GENERIC_LONG_MIN_SETUP_SCORE
        return 0.0

    def _setup_min_quality(setup: str) -> float:
        if setup in LONG_SETUP_MIN_QUALITY:
            return float(LONG_SETUP_MIN_QUALITY[setup])
        if setup == "generic_long":
            return GENERIC_LONG_MIN_QUALITY_SCORE
        return 0.0

    if sig.get("side") == "BUY":
        if blocks_fragile_context and fakeout_risk >= RESEARCH_HOLD_FRAGILE_FAKEOUT_CAP:
            return False, [f"{execution_mode.replace('_', '/')} blocks fragile context (fakeout_risk {fakeout_risk:.2f} >= {RESEARCH_HOLD_FRAGILE_FAKEOUT_CAP:.2f})"], 0.0, 1.0
        setup_policy_reason = _long_setup_policy_reason(setup_tag)
        if setup_policy_reason:
            return False, [setup_policy_reason], 0.0, 1.0
        if trial_active and trial_generic_policy == "require_labeled_setup_and_top_quality" and setup_tag in {"generic_long", "unknown"}:
            return False, [f"candidate trial blocks unlabeled long setup {setup_tag} during setup-quality recovery"], 0.0, 1.0
        if setup_tag in LONG_SETUP_MAJOR_ONLY and not is_major:
            return False, [f"{setup_tag} restricted to major symbols"], 0.0, 1.0
        if execution_mode == "research_hold" and not is_major:
            if not in_top_alt_expansion:
                return False, ["research hold restricts non-major longs to top-liquidity expansion symbols"], 0.0, 1.0
            if setup_tag not in RESEARCH_HOLD_NON_MAJOR_ALLOWED_LONG_SETUPS:
                allowed = ", ".join(sorted(RESEARCH_HOLD_NON_MAJOR_ALLOWED_LONG_SETUPS)) or "none"
                return False, [f"research hold rejects non-major long setup {setup_tag}; allowed: {allowed}"], 0.0, 1.0
            if setup_score < RESEARCH_HOLD_NON_MAJOR_MIN_SETUP_SCORE:
                return False, [f"research hold non-major long requires setup >= {RESEARCH_HOLD_NON_MAJOR_MIN_SETUP_SCORE:.2f} (got {setup_score:.2f})"], 0.0, 1.0
            if quality_score < RESEARCH_HOLD_NON_MAJOR_MIN_QUALITY_SCORE:
                return False, [f"research hold non-major long requires quality >= {RESEARCH_HOLD_NON_MAJOR_MIN_QUALITY_SCORE:.2f} (got {quality_score:.2f})"], 0.0, 1.0
            size_mult *= RESEARCH_HOLD_NON_MAJOR_SIZE_MULT
            reasons.append(f"research hold top-alt long size {RESEARCH_HOLD_NON_MAJOR_SIZE_MULT:.2f}x")
        if execution_mode == "research_hold" and setup_tag == "rebound_long":
            return False, ["research hold blocks rebound_long until slice is rebuilt"], 0.0, 1.0
        if execution_mode == "rebuild_trial" and setup_tag == "rebound_long":
            return False, ["rebuild/layered trial blocks rebound_long until slice is rebuilt"], 0.0, 1.0
        if trial_active and trial_allowed_long_symbols and not is_major and not trial_allows_non_major_long:
            return False, ["candidate trial restricts non-major longs to approved recovery symbols"], 0.0, 1.0
        if trial_active and (trial_disable_non_majors or trial_entry_profile == "majors_only" or trial_symbol_universe == "major_liquidity_tier") and not is_major and not trial_allows_non_major_long:
            return False, ["candidate trial restricts longs to major-liquidity symbols"], 0.0, 1.0
        if trial_active and trial_max_fakeout_risk is not None and fakeout_risk > trial_max_fakeout_risk:
            return False, [f"candidate trial rejects fragile long context (fakeout_risk {fakeout_risk:.2f} > {trial_max_fakeout_risk:.2f})"], 0.0, 1.0
        if trial_active and (trial_entry_profile == "majors_plus_liquid_alts" or trial_symbol_universe == "major_and_top_alt_tier") and not in_top_alt_expansion:
            return False, ["candidate expansion limits longs to majors and top-liquidity alts"], 0.0, 1.0
        if trial_active and trial_allowed_long_setups and setup_tag.lower() not in trial_allowed_long_setups:
            allowed = ", ".join(sorted(trial_allowed_long_setups))
            return False, [f"candidate trial rejects long setup {setup_tag}; allowed: {allowed}"], 0.0, 1.0
        if labeled_setup_required and setup_tag in {"generic_long", "unknown"}:
            return False, [f"redesign rejects unlabeled long setup ({setup_tag})"], 0.0, 1.0
        if labeled_setup_required and setup_score < 0.60:
            return False, [f"redesign requires stronger labeled long setup ({setup_score:.2f})"], 0.0, 1.0
        if entropy_label == "CHAOTIC" and (setup_tag == "generic_long" or not is_major):
            return False, [f"entropy regime {entropy_label.lower()} rejects weaker long setup"], 0.0, 1.0
        if redesign_active and entropy_label == "MIXED" and quality_score < 0.96:
            return False, [f"redesign requires stronger quality in mixed entropy regime ({quality_score:.2f})"], 0.0, 1.0
        required_setup_score = _setup_min_score(setup_tag)
        if required_setup_score and setup_score < required_setup_score:
            return False, [f"{setup_tag} setup score {setup_score:.2f} < {required_setup_score:.2f}"], 0.0, 1.0
        required_quality_score = _setup_min_quality(setup_tag)
        if required_quality_score and quality_score < required_quality_score:
            return False, [f"{setup_tag} quality {quality_score:.2f} < {required_quality_score:.2f}"], 0.0, 1.0
        if setup_tag == "generic_long":
            if trial_active and (trial_entry_profile == "majors_plus_liquid_alts" or trial_symbol_universe == "major_and_top_alt_tier") and not is_major:
                if quality_score < max(GENERIC_LONG_NON_MAJOR_MIN_QUALITY_SCORE, 0.985) or setup_score < max(GENERIC_LONG_MIN_SETUP_SCORE, 0.60):
                    return False, [f"candidate expansion rejects weaker top-alt long (setup {setup_score:.2f}, quality {quality_score:.2f})"], 0.0, 1.0
            if trial_active and trial_generic_policy == "require_top_quality":
                if quality_score < max(GENERIC_LONG_MIN_QUALITY_SCORE, 0.97) or setup_score < max(GENERIC_LONG_MIN_SETUP_SCORE, 0.55):
                    return False, [f"candidate trial rejects weak generic long (setup {setup_score:.2f}, quality {quality_score:.2f})"], 0.0, 1.0
            if quality_score < GENERIC_LONG_MIN_QUALITY_SCORE and setup_score < GENERIC_LONG_MIN_SETUP_SCORE:
                return False, [f"generic long too weak (setup {setup_score:.2f}, quality {quality_score:.2f})"], 0.0, 1.0
            if not is_major and quality_score < GENERIC_LONG_NON_MAJOR_MIN_QUALITY_SCORE:
                return False, [f"generic long non-major quality {quality_score:.2f} < {GENERIC_LONG_NON_MAJOR_MIN_QUALITY_SCORE:.2f}"], 0.0, 1.0
            if not is_major:
                score_adj -= NON_MAJOR_LONG_SCORE_PENALTY
                size_mult *= 0.85
                reasons.append("non-major generic long penalty")
            # Damp size for weak setup_score regardless of quality passing —
            # low setup confidence means the entry thesis is thin.
            if setup_score < 0.30:
                size_mult *= 0.60
                reasons.append(f"generic long weak setup damp 0.60x (setup {setup_score:.2f})")
            elif setup_score < 0.40:
                size_mult *= 0.75
                reasons.append(f"generic long low setup damp 0.75x (setup {setup_score:.2f})")
        elif setup_tag == "trend_long":
            score_adj += max(0.0, setup_score - 0.65) * 0.10
            reasons.append("trend-long preference")
            if regime_label == "BULL":
                score_adj += 0.02
                reasons.append("trend-long regime tailwind")
        elif setup_tag == "breakout_long":
            score_adj -= 0.03
            reasons.append("breakout-long skepticism")
            if regime_label != "BULL":
                size_mult *= 0.80
                reasons.append(f"breakout-long size damp in {regime_label.lower()} regime")
        elif setup_tag == "rebound_long":
            score_adj -= 0.02
            reasons.append("rebound-long caution until rebuild proves edge")
        elif is_major:
            score_adj += MAJOR_LONG_SCORE_BOOST
            reasons.append("major-symbol preference")

        setup_size_mult = float(LONG_SETUP_SIZE_MULTIPLIERS.get(setup_tag, 1.0) or 1.0)
        if setup_size_mult <= 0:
            return False, [f"long setup size policy disabled {setup_tag}"], 0.0, 1.0
        if setup_size_mult != 1.0:
            size_mult *= setup_size_mult
            reasons.append(f"long setup size policy {setup_tag} {setup_size_mult:.2f}x")
        if not is_major and quality_score < 0.95:
            size_mult *= 0.9
        if redesign_active:
            size_mult *= max(0.70, min(regime_size_mult, 0.90))
            score_adj += max(0.0, setup_score - 0.60) * 0.08
            reasons.append("redesign prediction/risk split active")
        if entropy_penalty > 0:
            score_adj -= entropy_penalty
            size_mult *= regime_size_mult
            reasons.append(f"entropy regime {entropy_label.lower()} penalty")
    else:
        if blocks_fragile_context and fakeout_risk >= RESEARCH_HOLD_FRAGILE_FAKEOUT_CAP:
            return False, [f"{execution_mode.replace('_', '/')} blocks fragile context (fakeout_risk {fakeout_risk:.2f} >= {RESEARCH_HOLD_FRAGILE_FAKEOUT_CAP:.2f})"], 0.0, 1.0
        if execution_mode in {"research_hold", "rebuild_trial"} and setup_tag == "exhaustion_short":
            return False, [f"{execution_mode.replace('_', '/')} blocks exhaustion_short until that slice is rebuilt"], 0.0, 1.0
        if trial_active and trial_allowed_short_setups and setup_tag.lower() not in trial_allowed_short_setups:
            allowed = ", ".join(sorted(trial_allowed_short_setups))
            return False, [f"candidate trial rejects short setup {setup_tag}; allowed: {allowed}"], 0.0, 1.0
        if redesign_active and setup_tag in {"generic_short", "unknown"} and entropy_label == "CHAOTIC":
            return False, [f"redesign rejects weak short setup during {entropy_label.lower()} regime"], 0.0, 1.0
        if entropy_penalty > 0:
            score_adj -= entropy_penalty * 0.5
            size_mult *= max(0.7, regime_size_mult)
            reasons.append(f"entropy regime {entropy_label.lower()} caution")
        if redesign_active:
            score_adj += max(0.0, setup_score - 0.55) * 0.05
            reasons.append("redesign prediction/risk split active")

    return True, reasons, round(score_adj, 4), round(size_mult, 4)


def _slippage_multiplier(direction: str, is_entry: bool) -> float:
    """Return a price multiplier that worsens simulated fills."""
    bps = PAPER_ENTRY_SLIPPAGE_BPS if is_entry else PAPER_EXIT_SLIPPAGE_BPS
    slip = max(0.0, bps) / 10_000.0
    if direction == "LONG":
        return 1.0 + slip if is_entry else 1.0 - slip
    return 1.0 - slip if is_entry else 1.0 + slip


def _apply_slippage(price: float, direction: str, *, is_entry: bool) -> float:
    return round(price * _slippage_multiplier(direction, is_entry), 8)


def _execution_realism_mode(trial: dict | None = None, autopilot: dict | None = None) -> str:
    trial_type = str((trial or {}).get("type", "") or "").lower()
    actions = {str(x).strip().lower() for x in ((autopilot or {}).get("actions") or []) if str(x).strip()}
    if candidate_trial_is_active(trial) and trial_type in {"competitiveness_gap_rebuild", "quantforge_redesign", "quantforge_layered_trial"}:
        return "rebuild_trial"
    if "freeze_for_rebuild" in actions:
        return "research_hold"
    return "normal"


def _execution_realism_profile(
    symbol: str,
    direction: str,
    *,
    is_entry: bool,
    trial: dict | None = None,
    autopilot: dict | None = None,
    quality_metrics: dict | None = None,
    pos: dict | None = None,
    trigger: str | None = None,
) -> dict:
    mode = _execution_realism_mode(trial, autopilot)
    trigger_norm = str(trigger or "").upper()
    price = float(
        (quality_metrics or {}).get("price")
        or (pos or {}).get("entry_price")
        or 0.0
    )
    quality_score = float(
        (quality_metrics or {}).get("quality_score")
        or (pos or {}).get("quality_score")
        or 0.0
    )
    is_major = symbol in MAJOR_SYMBOLS

    slippage_bps = PAPER_ENTRY_SLIPPAGE_BPS if is_entry else PAPER_EXIT_SLIPPAGE_BPS
    spread_bps = PAPER_SPREAD_BPS
    mark_haircut_bps = 0.0
    stop_gap_bps = 0.0

    if mode == "rebuild_trial":
        slippage_bps = max(
            slippage_bps,
            REBUILD_TRIAL_ENTRY_SLIPPAGE_BPS if is_entry else REBUILD_TRIAL_EXIT_SLIPPAGE_BPS,
        )
        spread_bps = max(spread_bps, REBUILD_TRIAL_SPREAD_BPS)
        mark_haircut_bps = max(mark_haircut_bps, REBUILD_TRIAL_MARK_HAIRCUT_BPS)
        if not is_entry and trigger_norm in {"STOP_LOSS", "TIME_STOP", "SIGNAL_ROTATION"}:
            stop_gap_bps += REBUILD_TRIAL_STOP_GAP_BPS
    elif mode == "research_hold":
        slippage_bps = max(
            slippage_bps,
            RESEARCH_HOLD_ENTRY_SLIPPAGE_BPS if is_entry else RESEARCH_HOLD_EXIT_SLIPPAGE_BPS,
        )
        spread_bps = max(spread_bps, RESEARCH_HOLD_SPREAD_BPS)
        mark_haircut_bps = max(mark_haircut_bps, RESEARCH_HOLD_MARK_HAIRCUT_BPS)
        if not is_entry and trigger_norm in {"STOP_LOSS", "TIME_STOP", "SIGNAL_ROTATION"}:
            stop_gap_bps += RESEARCH_HOLD_STOP_GAP_BPS

    if not is_major:
        spread_bps += NON_MAJOR_EXECUTION_PENALTY_BPS
    if 0 < price < 1.0:
        spread_bps += LOW_PRICE_EXECUTION_PENALTY_BPS
    if quality_score and quality_score < 0.95:
        spread_bps += LOW_QUALITY_EXECUTION_PENALTY_BPS

    fill_bps = max(0.0, slippage_bps) + max(0.0, spread_bps) / 2.0 + max(0.0, stop_gap_bps)
    liquidation_bps = max(0.0, slippage_bps) + max(0.0, spread_bps) / 2.0 + max(0.0, mark_haircut_bps)
    return {
        "mode": mode,
        "slippage_bps": round(slippage_bps, 4),
        "spread_bps": round(spread_bps, 4),
        "stop_gap_bps": round(stop_gap_bps, 4),
        "mark_haircut_bps": round(mark_haircut_bps, 4),
        "fill_bps": round(fill_bps, 4),
        "liquidation_bps": round(liquidation_bps, 4),
    }


def _worsen_price(price: float, direction: str, bps: float, *, is_entry: bool) -> float:
    worsen = max(0.0, float(bps)) / 10_000.0
    if direction == "LONG":
        return round(price * (1.0 + worsen if is_entry else 1.0 - worsen), 8)
    return round(price * (1.0 - worsen if is_entry else 1.0 + worsen), 8)


def _apply_execution_price(
    price: float,
    symbol: str,
    direction: str,
    *,
    is_entry: bool,
    trial: dict | None = None,
    autopilot: dict | None = None,
    quality_metrics: dict | None = None,
    pos: dict | None = None,
    trigger: str | None = None,
) -> tuple[float, dict]:
    profile = _execution_realism_profile(
        symbol,
        direction,
        is_entry=is_entry,
        trial=trial,
        autopilot=autopilot,
        quality_metrics=quality_metrics,
        pos=pos,
        trigger=trigger,
    )
    adjusted = _worsen_price(price, direction, profile["fill_bps"], is_entry=is_entry)
    return adjusted, profile


def _mark_to_exit_price(
    symbol: str,
    pos: dict,
    market_price: float,
    *,
    trial: dict | None = None,
    autopilot: dict | None = None,
) -> tuple[float, dict]:
    profile = _execution_realism_profile(
        symbol,
        pos.get("direction", "LONG"),
        is_entry=False,
        trial=trial,
        autopilot=autopilot,
        pos=pos,
        trigger="MARK",
    )
    adjusted = _worsen_price(
        market_price,
        pos.get("direction", "LONG"),
        profile["liquidation_bps"],
        is_entry=False,
    )
    return adjusted, profile


def _mark_last_exit(port: dict, symbol: str, ts: str, pnl: float = 0.0):
    # Store both timestamp and whether the exit was a winner so cooldown
    # logic can apply a longer hold-off after profitable exits.
    port.setdefault("last_exit_ts_by_symbol", {})[symbol] = ts
    port.setdefault("last_exit_winner_by_symbol", {})[symbol] = pnl > 0


def _in_reentry_cooldown(port: dict, symbol: str, now_iso: str) -> bool:
    if REENTRY_COOLDOWN_HOURS <= 0:
        return False
    last_exit = port.get("last_exit_ts_by_symbol", {}).get(symbol)
    if not last_exit:
        return False
    try:
        exit_dt = datetime.fromisoformat(str(last_exit).replace("Z", "+00:00"))
        now_dt = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
    except Exception:
        return False
    was_winner = bool(port.get("last_exit_winner_by_symbol", {}).get(symbol, False))
    # After a winning trade the move is likely exhausted — wait 2× as long
    # before re-entering the same coin.
    cooldown_mult = 2.0 if was_winner else 1.0
    return (now_dt - exit_dt).total_seconds() < REENTRY_COOLDOWN_HOURS * 3600 * cooldown_mult


def _register_close_outcome(port: dict, symbol: str, pnl: float, ts: str):
    streaks = port.setdefault("loss_streak_by_symbol", {})
    lockouts = port.setdefault("loss_lockout_until_by_symbol", {})
    if pnl > 0:
        streaks[symbol] = 0
        lockouts.pop(symbol, None)
        return

    try:
        ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        ts_dt = datetime.now(timezone.utc)

    next_streak = int(streaks.get(symbol, 0)) + 1
    streaks[symbol] = next_streak
    if SINGLE_LOSS_COOLOFF_HOURS > 0:
        candidate_until = ts_dt + timedelta(hours=SINGLE_LOSS_COOLOFF_HOURS)
        existing = lockouts.get(symbol)
        if existing:
            try:
                existing_dt = datetime.fromisoformat(str(existing).replace("Z", "+00:00"))
                candidate_until = max(candidate_until, existing_dt)
            except Exception:
                pass
        lockouts[symbol] = candidate_until.isoformat()
    if next_streak >= MAX_CONSECUTIVE_LOSSES_PER_SYMBOL:
        lockouts[symbol] = (ts_dt + timedelta(hours=LOSS_STREAK_COOLOFF_HOURS)).isoformat()


def _in_loss_lockout(port: dict, symbol: str, now_iso: str) -> bool:
    lockout_until = port.get("loss_lockout_until_by_symbol", {}).get(symbol)
    if not lockout_until:
        return False
    try:
        now_dt = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
        lockout_dt = datetime.fromisoformat(str(lockout_until).replace("Z", "+00:00"))
    except Exception:
        return False
    return now_dt < lockout_dt


def _repair_loss_lockout_state(port: dict):
    lockouts = port.setdefault("loss_lockout_until_by_symbol", {})
    streaks = port.setdefault("loss_streak_by_symbol", {})
    last_exits = port.setdefault("last_exit_ts_by_symbol", {})
    now_dt = datetime.now(timezone.utc)

    for symbol, until in list(lockouts.items()):
        try:
            if datetime.fromisoformat(str(until).replace("Z", "+00:00")) <= now_dt:
                lockouts.pop(symbol, None)
        except Exception:
            lockouts.pop(symbol, None)

    for symbol, streak in streaks.items():
        if int(streak or 0) <= 0 or symbol in lockouts:
            continue
        last_exit = last_exits.get(symbol)
        if not last_exit:
            continue
        try:
            last_exit_dt = datetime.fromisoformat(str(last_exit).replace("Z", "+00:00"))
        except Exception:
            continue

        hours = LOSS_STREAK_COOLOFF_HOURS if int(streak or 0) >= MAX_CONSECUTIVE_LOSSES_PER_SYMBOL else SINGLE_LOSS_COOLOFF_HOURS
        if hours <= 0:
            continue
        candidate = last_exit_dt + timedelta(hours=float(hours))
        if candidate > now_dt:
            lockouts[symbol] = candidate.isoformat()


def _normalize_position_state(port: dict):
    for pos in port.get("positions", {}).values():
        entry = float(pos.get("entry_price", 0.0) or 0.0)
        if entry <= 0:
            continue
        pos["original_qty"] = float(pos.get("original_qty", pos.get("qty", 0.0)) or 0.0)
        pos["best_price"] = float(pos.get("best_price", entry) or entry)
        pos["trailing_active"] = bool(pos.get("trailing_active", False))
        pos["entry_fee"] = float(pos.get("entry_fee", 0.0) or 0.0)
        pos["initial_stop_loss"] = float(pos.get("initial_stop_loss", pos.get("stop_loss", entry)) or entry)
        risk_per_unit = abs(entry - float(pos.get("initial_stop_loss", pos.get("stop_loss", entry)) or entry))
        pos["initial_risk_per_unit"] = float(pos.get("initial_risk_per_unit", risk_per_unit) or risk_per_unit)
        pos["highest_unrealized_pct"] = float(pos.get("highest_unrealized_pct", 0.0) or 0.0)
        pos["highest_r"] = float(pos.get("highest_r", 0.0) or 0.0)
        pos["stop_stage"] = str(pos.get("stop_stage", "initial") or "initial")
        partials = pos.get("partial_take_profits") or {}
        pos["partial_take_profits"] = {
            "tp1_done": bool(partials.get("tp1_done", False)),
            "tp2_done": bool(partials.get("tp2_done", False)),
        }


def _position_move_pct(pos: dict, price: float) -> float:
    entry = float(pos.get("entry_price", 0.0) or 0.0)
    if entry <= 0 or price <= 0:
        return 0.0
    direction = pos.get("direction", "LONG")
    if direction == "SHORT":
        return (entry - price) / entry
    return (price - entry) / entry


def _position_move_r(pos: dict, price: float) -> float:
    risk_per_unit = float(pos.get("initial_risk_per_unit", 0.0) or 0.0)
    if risk_per_unit <= 0:
        return 0.0
    entry = float(pos.get("entry_price", 0.0) or 0.0)
    direction = pos.get("direction", "LONG")
    if direction == "SHORT":
        move = entry - price
    else:
        move = price - entry
    return move / risk_per_unit


def _update_position_extrema(pos: dict, price: float) -> None:
    move_pct = _position_move_pct(pos, price)
    move_r = _position_move_r(pos, price)
    pos["highest_unrealized_pct"] = max(float(pos.get("highest_unrealized_pct", 0.0) or 0.0), move_pct)
    pos["highest_r"] = max(float(pos.get("highest_r", 0.0) or 0.0), move_r)


def _partial_tp_plan(pos: dict) -> list[tuple[str, float, float]]:
    """Quality-scaled partial take-profit plan.

    High-conviction entries let more ride (smaller early takes).
    Low-conviction entries bank profit faster (larger early takes).
    """
    setup = str(pos.get("setup_tag", "") or "").lower()
    signal_score = float(pos.get("signal_score", 0.0) or 0.0)

    # Quality-based scaling
    if signal_score >= 0.85:
        # High conviction — let it run
        if "breakout" in setup:
            return [("tp1_done", 1.5, 0.25), ("tp2_done", 3.0, 0.25)]
        return [("tp1_done", 1.2, 0.25), ("tp2_done", 2.5, 0.25)]
    elif signal_score >= 0.70:
        # Medium — standard plan (preserve original setup_tag logic)
        if "rebound" in setup:
            return [("tp1_done", 0.8, 0.35), ("tp2_done", 1.5, 0.30)]
        if "breakout" in setup:
            return [("tp1_done", 1.25, 0.30), ("tp2_done", 2.5, 0.30)]
        return [("tp1_done", 1.0, 0.33), ("tp2_done", 2.0, 0.33)]
    else:
        # Low conviction — bank early (also covers signal_score == 0 / no score)
        if signal_score == 0.0:
            # Legacy path: no signal_score stored — use original setup_tag routing
            if "rebound" in setup:
                return [("tp1_done", 0.8, 0.40), ("tp2_done", 1.5, 0.35)]
            if "breakout" in setup:
                return [("tp1_done", 1.25, 0.30), ("tp2_done", 2.5, 0.30)]
            return [("tp1_done", 1.0, 0.33), ("tp2_done", 2.0, 0.33)]
        return [("tp1_done", 0.8, 0.40), ("tp2_done", 1.5, 0.30)]


def _close_trigger_label(pos: dict, triggered: str) -> str:
    if triggered != "STOP_LOSS":
        return triggered
    if bool(pos.get("trailing_active", False)):
        return "TRAILING_STOP"
    stage = str(pos.get("stop_stage", "initial") or "initial")
    if stage in {"breakeven", "locked_0_5r", "locked_1r"}:
        return "BREAK_EVEN_STOP"
    return "INITIAL_STOP_LOSS"


def _execute_partial_take_profit(port: dict, sym: str, pos: dict, price: float, stage_name: str, fraction: float, now: str, *, autopilot=None, trial=None) -> dict | None:
    qty = float(pos.get("qty", 0.0) or 0.0)
    if qty <= 0:
        return None
    close_qty = round(min(qty, max(qty * fraction, 0.0)), 8)
    if close_qty <= 0 or close_qty >= qty:
        return None
    exit_price, exit_profile = _apply_execution_price(
        price,
        sym,
        pos.get("direction", "LONG"),
        is_entry=False,
        trial=trial,
        autopilot=autopilot,
        pos=pos,
        trigger=stage_name.upper(),
    )
    exit_fee = close_qty * exit_price * TAKER_FEE
    entry_fee_remaining = float(pos.get("entry_fee", 0.0) or 0.0)
    entry_fee_alloc = round(entry_fee_remaining * (close_qty / qty), 8)
    funding_remaining = float(pos.get("funding_paid", 0.0) or 0.0) if is_leveraged_position(pos) else 0.0
    funding_alloc = round(funding_remaining * (close_qty / qty), 8)
    unit_pnl = position_unrealized_pnl({**pos, "qty": 1.0}, exit_price)
    pnl = round(unit_pnl * close_qty - exit_fee - entry_fee_alloc - funding_alloc, 4)
    margin_release = float(pos.get("margin", 0.0) or 0.0) * (close_qty / qty)
    if is_leveraged_position(pos):
        port["cash"] += margin_release + pnl
    else:
        port["cash"] += close_qty * exit_price - exit_fee
    _record_realized_pnl(port, pnl, count_closed_trade=False)
    port["total_fees_paid"] += exit_fee
    pos["qty"] = round(qty - close_qty, 8)
    pos["margin"] = round(float(pos.get("margin", 0.0) or 0.0) - margin_release, 4)
    pos["entry_fee"] = round(entry_fee_remaining - entry_fee_alloc, 8)
    if funding_alloc:
        pos["funding_paid"] = round(funding_remaining - funding_alloc, 8)
    if stage_name == "tp1_done":
        pos["stop_stage"] = "breakeven"
    elif stage_name == "tp2_done":
        pos["stop_stage"] = "locked_0_5r"
    trade = {
        "type": "PARTIAL_CLOSE",
        "symbol": sym,
        "side": "BUY" if pos.get("direction", "LONG") == "SHORT" else "SELL",
        "trigger": stage_name.upper(),
        "qty": close_qty,
        "remaining_qty": pos["qty"],
        "entry_price": pos["entry_price"],
        "exit_price": exit_price,
        "entry_fee_allocated": round(entry_fee_alloc, 8),
        "exit_fee": round(exit_fee, 4),
        "pnl": pnl,
        "setup_tag": pos.get("setup_tag"),
        "setup_score": pos.get("setup_score"),
        "quality_score": pos.get("quality_score"),
        "best_price": pos.get("best_price"),
        "stop_loss": pos.get("stop_loss"),
        "take_profit": pos.get("take_profit"),
        "stop_stage": pos.get("stop_stage"),
        "execution_mode": exit_profile.get("mode"),
        "execution_bps": exit_profile.get("fill_bps"),
        "ts": now,
    }
    append_trade(trade)
    return trade


def _apply_partial_profit_stop_lock(pos: dict, stage_name: str) -> None:
    entry_price = float(pos.get("entry_price", 0.0) or 0.0)
    if entry_price <= 0:
        return
    direction = str(pos.get("direction", "LONG")).upper()
    current_stop = float(pos.get("stop_loss", entry_price) or entry_price)
    risk_per_unit = float(pos.get("initial_risk_per_unit", 0.0) or 0.0)

    if stage_name == "tp2_done" and risk_per_unit > 0:
        lock_distance = 0.5 * risk_per_unit
    else:
        lock_distance = entry_price * BREAK_EVEN_BUFFER_PCT

    if direction == "SHORT":
        candidate_stop = round(entry_price - lock_distance, 4)
        pos["stop_loss"] = min(current_stop, candidate_stop)
    else:
        candidate_stop = round(entry_price + lock_distance, 4)
        pos["stop_loss"] = max(current_stop, candidate_stop)


def _trailing_profile(move_pct: float, pos: dict = None, candles: list = None) -> tuple[str, float, float]:
    """Momentum-adaptive trailing profile.

    Uses rolling 3-candle rate of change when candle data available,
    falls back to move_pct tiers otherwise.
    """
    roc_3 = None
    if candles and len(candles) >= 4:
        try:
            # Resolve position open timestamp as Unix seconds for comparison
            open_ts_s = None
            if pos:
                raw_ts = pos.get("open_ts", pos.get("ts", None))
                if raw_ts:
                    _dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                    open_ts_s = int(_dt.timestamp())

            # candles[i][0] is Unix seconds (int) — see get_klines()
            c4_ts_s = int(candles[-4][0])

            # Only compute roc_3 if the 4-candle lookback window is post-entry
            if open_ts_s is None or c4_ts_s >= open_ts_s:
                c_now = float(candles[-1][2])
                c_prev = float(candles[-4][2])
                roc_3 = (c_now - c_prev) / max(c_prev, 1e-10)
        except Exception:
            pass

    if roc_3 is not None:
        if roc_3 > 0.03:       # explosive spike >3% in 3h
            return ("spike_lock", 0.005, 0.60)
        elif roc_3 > 0.01:     # healthy momentum
            giveback = max(0.005, 0.015 - move_pct * 0.08)
            return ("momentum", round(giveback, 4), 0.40)
        elif roc_3 < -0.005 and move_pct > 0.02:  # momentum fading while in profit
            return ("fade_lock", 0.008, 0.50)
        else:
            return ("grind", 0.02, 0.30)

    # Fallback: original tier logic when no candle data
    if move_pct >= TRAILING_PROFIT_TIER3_ACTIVATE_PCT:
        return ("tier3", TRAILING_PROFIT_TIER3_GIVEBACK_PCT, TRAILING_LOCK_PROFIT_SHARE_TIER3)
    if move_pct >= TRAILING_PROFIT_TIER2_ACTIVATE_PCT:
        return ("tier2", TRAILING_PROFIT_TIER2_GIVEBACK_PCT, TRAILING_LOCK_PROFIT_SHARE_TIER2)
    return ("tier1", TRAILING_PROFIT_GIVEBACK_PCT, 0.0)


def _build_trailing_update_payload(
    *,
    trailing_tier: str,
    move_pct: float,
    move_r: float,
    direction: str,
    entry_price: float,
    price: float,
    giveback_pct: float,
    lock_share: float,
    best_price_before: float,
    best_price_after: float,
    stop_loss_before: float,
    stop_loss_after: float,
    trailing_was_active: bool,
    trailing_is_active: bool,
) -> dict:
    return {
        "event": "TRAILING_UPDATE",
        "trailing_tier": trailing_tier,
        "move_pct": round(move_pct, 6),
        "move_r": round(move_r, 3),
        "direction": direction,
        "entry_price": entry_price,
        "price": price,
        "giveback_pct": round(giveback_pct, 6),
        "lock_share": round(lock_share, 4),
        "best_price_before": best_price_before,
        "best_price_after": best_price_after,
        "stop_loss_before": stop_loss_before,
        "stop_loss_after": stop_loss_after,
        "trailing_was_active": trailing_was_active,
        "trailing_is_active": trailing_is_active,
    }


def _update_trailing_exit(pos: dict, price: float, candles: list = None) -> dict | None:
    """R-multiple staged trailing: lock gains as multiples of initial risk.

    Stage transitions (LONG example, risk_per_unit = $1):
      Move >= 1R ($1 profit/unit): lock stop at entry + 0.25R (guaranteed small win)
      Move >= 1.5R:                lock stop at entry + 0.75R
      Move >= 2R:                  lock stop at entry + 1.0R (guaranteed 1:1 R:R)
      Move >= 3R:                  lock stop at entry + 2.0R
      Additionally: classic trailing with tight giveback from best_price.
    """
    if not ENABLE_TRAILING_PROFIT:
        return None
    entry = float(pos.get("entry_price", 0.0) or 0.0)
    if entry <= 0 or price <= 0:
        return None
    direction = pos.get("direction", "LONG")
    risk_per_unit = float(pos.get("initial_risk_per_unit", 0.0) or 0.0)
    if risk_per_unit <= 0:
        risk_per_unit = abs(entry - float(pos.get("initial_stop_loss", pos.get("stop_loss", entry))))
    if risk_per_unit <= 0:
        risk_per_unit = entry * 0.02  # fallback 2%

    best_price = float(pos.get("best_price", entry) or entry)
    prev_best_price = best_price
    prev_stop_loss = float(pos.get("stop_loss", entry))
    prev_trailing_active = bool(pos.get("trailing_active", False))

    # R-multiple lock levels: (R_threshold, lock_at_R_fraction)
    R_LOCK_LEVELS = [(1.0, 0.25), (1.5, 0.75), (2.0, 1.0), (3.0, 2.0), (4.0, 3.0)]

    if direction == "SHORT":
        best_price = min(best_price, price)
        pos["best_price"] = best_price
        move_r = (entry - best_price) / risk_per_unit
        move_pct = (entry - best_price) / entry

        if move_r < SHORT_TRAILING_MIN_R:
            return None
        pos["trailing_active"] = True

        # R-multiple lock: compute the highest lock level reached
        r_lock_stop = None
        for r_thresh, lock_r in R_LOCK_LEVELS:
            if move_r >= r_thresh:
                r_lock_stop = round(entry - lock_r * risk_per_unit, 4)

        trailing_tier, trailing_giveback_pct, lock_share = _trailing_profile(move_pct, pos=pos, candles=candles)
        trailing_stop = round(best_price * (1 + trailing_giveback_pct), 4)

        # Take the tightest (lowest for shorts) of R-lock and trailing
        candidates = [c for c in [r_lock_stop, trailing_stop] if c is not None]
        if not candidates:
            return None
        candidate_stop = min(candidates)
        # Never worse than previous stop
        if candidate_stop < prev_stop_loss:
            pos["stop_loss"] = candidate_stop
            pos["stop_stage"] = "r_lock_%.1fR" % move_r
        if candidate_stop < prev_stop_loss or (not prev_trailing_active and pos["trailing_active"]):
            return _build_trailing_update_payload(
                trailing_tier=trailing_tier,
                move_pct=move_pct,
                move_r=move_r,
                direction=direction,
                entry_price=entry,
                price=price,
                giveback_pct=trailing_giveback_pct,
                lock_share=lock_share,
                best_price_before=prev_best_price,
                best_price_after=best_price,
                stop_loss_before=prev_stop_loss,
                stop_loss_after=float(pos["stop_loss"]),
                trailing_was_active=prev_trailing_active,
                trailing_is_active=True,
            )
        return None

    # LONG direction
    best_price = max(best_price, price)
    pos["best_price"] = best_price
    move_r = (best_price - entry) / risk_per_unit
    move_pct = (best_price - entry) / entry

    if move_pct < TRAILING_PROFIT_ACTIVATE_PCT and move_r < 0.5:
        return None
    pos["trailing_active"] = True

    # R-multiple lock
    r_lock_stop = None
    for r_thresh, lock_r in R_LOCK_LEVELS:
        if move_r >= r_thresh:
            r_lock_stop = round(entry + lock_r * risk_per_unit, 4)

    # Classic trailing from best price (tight 0.5% giveback)
    trailing_stop = round(best_price * (1 - 0.005), 4)

    candidates = [c for c in [r_lock_stop, trailing_stop] if c is not None]
    if not candidates:
        return None
    candidate_stop = max(candidates)
    # Never worse than previous stop
    if candidate_stop > prev_stop_loss:
        pos["stop_loss"] = candidate_stop
        pos["stop_stage"] = "r_lock_%.1fR" % move_r
    if candidate_stop > prev_stop_loss or (not prev_trailing_active and pos["trailing_active"]):
        return _build_trailing_update_payload(
            trailing_tier="r_lock",
            move_pct=move_pct,
            move_r=move_r,
            direction=direction,
            entry_price=entry,
            price=price,
            giveback_pct=0.005,
            lock_share=0.0,
            best_price_before=prev_best_price,
            best_price_after=best_price,
            stop_loss_before=prev_stop_loss,
            stop_loss_after=float(pos["stop_loss"]),
            trailing_was_active=prev_trailing_active,
            trailing_is_active=True,
        )
    return None


def is_leveraged_position(pos: dict) -> bool:
    """True when the position uses futures-style margin accounting."""
    return "margin" in pos


def position_unrealized_pnl(pos: dict, price: float) -> float:
    """Mark-to-market PnL for both legacy spot and leveraged futures positions."""
    if pos.get("direction") == "SHORT":
        return (pos["entry_price"] - price) * pos["qty"]
    return (price - pos["entry_price"]) * pos["qty"]


def position_equity_value(pos: dict, price: float) -> float:
    """Position contribution to account equity (net of accrued funding)."""
    if is_leveraged_position(pos):
        return pos.get("margin", 0.0) + position_unrealized_pnl(pos, price) - float(pos.get("funding_paid", 0.0))
    return pos["qty"] * price


def close_position_accounting(pos: dict, exit_price: float, exit_fee: float) -> tuple[float, float]:
    """Return (pnl, cash_delta) when closing a position.

    PnL is net of exit fee, remaining entry fee, and accrued funding cost.
    """
    funding = float(pos.get("funding_paid", 0.0)) if is_leveraged_position(pos) else 0.0
    pnl = position_unrealized_pnl(pos, exit_price) - exit_fee - pos.get("entry_fee", 0.0) - funding
    if is_leveraged_position(pos):
        return pnl, pos.get("margin", 0.0) + pnl
    return pnl, pos["qty"] * exit_price - exit_fee


def _record_realized_pnl(port: dict, pnl: float, *, count_closed_trade: bool) -> None:
    """Apply realized PnL to portfolio-level aggregates."""
    port["realized_pnl"] += pnl
    if pnl > 0:
        port["gross_profit"] += pnl
        if count_closed_trade:
            port["wins"] += 1
    else:
        port["gross_loss"] += abs(pnl)
        if count_closed_trade:
            port["losses"] += 1
    if count_closed_trade:
        port["total_trades"] += 1


def _entry_score_threshold(sig: dict, trial: dict | None = None) -> float:
    signal_threshold = sig.get("entry_threshold")
    if signal_threshold is not None:
        try:
            threshold = float(signal_threshold)
        except Exception:
            threshold = float(SIGNAL_CONFIDENCE_THRESHOLD)
    else:
        threshold = float(SIGNAL_CONFIDENCE_THRESHOLD)
    if candidate_trial_is_active(trial) and sig.get("side") == "BUY":
        threshold = min(max(0.50, threshold - TRIAL_LONG_THRESHOLD_RELIEF), TRIAL_ENTRY_SCORE_THRESHOLD)
    return threshold


def equity(
    port: dict,
    prices: dict[str, float] | None = None,
    *,
    autopilot: dict | None = None,
    trial: dict | None = None,
) -> float:
    """Calculate futures-style equity as cash + margin + unrealized PnL."""
    total = port["cash"]
    for sym, pos in port.get("positions", {}).items():
        price = (prices or {}).get(sym, pos["entry_price"])
        if autopilot or trial:
            price, _ = _mark_to_exit_price(sym, pos, price, trial=trial, autopilot=autopilot)
        total += position_equity_value(pos, price)
    return total


def rebuild_portfolio_from_trades() -> dict:
    """Rebuild the portfolio ledger from the trade log to repair accounting drift."""
    port = {
        "cash": STARTING_BALANCE,
        "equity": STARTING_BALANCE,
        "starting_balance": STARTING_BALANCE,
        "positions": {},
        "realized_pnl": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_fees_paid": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "peak_equity": STARTING_BALANCE,
        "max_drawdown": 0.0,
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
        "last_exit_ts_by_symbol": {},
        "loss_streak_by_symbol": {},
        "loss_lockout_until_by_symbol": {},
    }
    if not os.path.exists(TRADES_FILE):
        return port

    first_ts = None
    with open(TRADES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trade = json.loads(line)
            ts = trade.get("open_ts") or trade.get("ts")
            if ts and first_ts is None:
                first_ts = ts

            if trade.get("type") == "OPEN":
                qty = float(trade["qty"])
                entry_price = float(trade["entry_price"])
                entry_fee = float(trade.get("entry_fee", 0.0))
                leverage = int(trade.get("leverage", 1) or 1)
                leveraged = leverage > 1 or "direction" in trade
                if leveraged:
                    notional = qty * entry_price
                    margin = round(notional / leverage, 4)
                    port["cash"] -= (margin + entry_fee)
                    port["positions"][trade["symbol"]] = {
                        "direction": trade.get("direction", "LONG"),
                        "qty": qty,
                        "original_qty": float(trade.get("original_qty", qty) or qty),
                        "margin": margin,
                        "entry_price": entry_price,
                        "entry_fee": entry_fee,
                        "stop_loss": float(trade.get("stop_loss", entry_price)),
                        "initial_stop_loss": float(trade.get("initial_stop_loss", trade.get("stop_loss", entry_price)) or entry_price),
                        "initial_risk_per_unit": float(trade.get("initial_risk_per_unit", abs(entry_price - float(trade.get("stop_loss", entry_price)))) or 0.0),
                        "take_profit": float(trade.get("take_profit", entry_price)),
                        "open_ts": ts,
                        "opened": ts,
                        "signal_score": float(trade.get("signal_score", trade.get("score", 0.0))),
                        "raw_signal_score": float(trade.get("raw_signal_score", trade.get("signal_score", trade.get("score", 0.0)))),
                        "setup_tag": trade.get("setup_tag"),
                        "setup_score": float(trade.get("setup_score", 0.0) or 0.0),
                        "quality_score": float(trade.get("quality_score", 0.0) or 0.0),
                        "best_price": float(trade.get("best_price", entry_price)),
                        "trailing_active": bool(trade.get("trailing_active", False)),
                        "highest_unrealized_pct": float(trade.get("highest_unrealized_pct", 0.0) or 0.0),
                        "highest_r": float(trade.get("highest_r", 0.0) or 0.0),
                        "stop_stage": str(trade.get("stop_stage", "initial") or "initial"),
                        "partial_take_profits": {
                            "tp1_done": bool(trade.get("tp1_done", False)),
                            "tp2_done": bool(trade.get("tp2_done", False)),
                        },
                    }
                else:
                    cost = qty * entry_price
                    port["cash"] -= (cost + entry_fee)
                    port["positions"][trade["symbol"]] = {
                        "qty": qty,
                        "original_qty": float(trade.get("original_qty", qty) or qty),
                        "entry_price": entry_price,
                        "entry_fee": entry_fee,
                        "stop_loss": float(trade.get("stop_loss", entry_price)),
                        "initial_stop_loss": float(trade.get("initial_stop_loss", trade.get("stop_loss", entry_price)) or entry_price),
                        "initial_risk_per_unit": float(trade.get("initial_risk_per_unit", abs(entry_price - float(trade.get("stop_loss", entry_price)))) or 0.0),
                        "take_profit": float(trade.get("take_profit", entry_price)),
                        "opened": ts,
                        "signal_score": float(trade.get("signal_score", trade.get("score", 0.0))),
                        "raw_signal_score": float(trade.get("raw_signal_score", trade.get("signal_score", trade.get("score", 0.0)))),
                        "setup_tag": trade.get("setup_tag"),
                        "setup_score": float(trade.get("setup_score", 0.0) or 0.0),
                        "quality_score": float(trade.get("quality_score", 0.0) or 0.0),
                        "best_price": float(trade.get("best_price", entry_price)),
                        "trailing_active": bool(trade.get("trailing_active", False)),
                        "highest_unrealized_pct": float(trade.get("highest_unrealized_pct", 0.0) or 0.0),
                        "highest_r": float(trade.get("highest_r", 0.0) or 0.0),
                        "stop_stage": str(trade.get("stop_stage", "initial") or "initial"),
                        "partial_take_profits": {
                            "tp1_done": bool(trade.get("tp1_done", False)),
                            "tp2_done": bool(trade.get("tp2_done", False)),
                        },
                    }
                port["total_fees_paid"] += entry_fee
                continue

            if trade.get("type") == "PARTIAL_CLOSE":
                sym = trade["symbol"]
                pos = port["positions"].get(sym)
                if not pos:
                    continue
                qty_before = float(pos.get("qty", 0.0) or 0.0)
                close_qty = float(trade.get("qty", 0.0) or 0.0)
                if qty_before <= 0 or close_qty <= 0:
                    continue
                pnl = float(trade.get("pnl", 0.0) or 0.0)
                exit_fee = float(trade.get("exit_fee", 0.0) or 0.0)
                entry_fee_alloc = float(trade.get("entry_fee_allocated", 0.0) or 0.0)
                margin_release = float(pos.get("margin", 0.0) or 0.0) * (close_qty / qty_before)
                if is_leveraged_position(pos):
                    port["cash"] += margin_release + pnl
                else:
                    exit_price = float(trade.get("exit_price", pos.get("entry_price", 0.0)) or pos.get("entry_price", 0.0))
                    port["cash"] += close_qty * exit_price - exit_fee
                _record_realized_pnl(port, pnl, count_closed_trade=False)
                port["total_fees_paid"] += exit_fee
                pos["qty"] = round(max(0.0, qty_before - close_qty), 8)
                pos["margin"] = round(max(0.0, float(pos.get("margin", 0.0) or 0.0) - margin_release), 4)
                pos["entry_fee"] = round(max(0.0, float(pos.get("entry_fee", 0.0) or 0.0) - entry_fee_alloc), 8)
                pos["stop_loss"] = float(trade.get("stop_loss", pos.get("stop_loss", pos.get("entry_price", 0.0))) or pos.get("stop_loss", 0.0))
                pos["best_price"] = float(trade.get("best_price", pos.get("best_price", pos.get("entry_price", 0.0))) or pos.get("best_price", 0.0))
                pos["stop_stage"] = str(trade.get("stop_stage", pos.get("stop_stage", "initial")) or pos.get("stop_stage", "initial"))
                partials = pos.setdefault("partial_take_profits", {"tp1_done": False, "tp2_done": False})
                trigger = str(trade.get("trigger", "") or "").upper()
                if trigger == "TP1_DONE":
                    partials["tp1_done"] = True
                elif trigger == "TP2_DONE":
                    partials["tp2_done"] = True
                continue

            if trade.get("type") != "CLOSE":
                continue

            sym = trade["symbol"]
            pos = port["positions"].get(sym)
            if not pos:
                continue

            exit_price = float(trade["exit_price"])
            exit_fee = float(trade.get("exit_fee", 0.0))
            pnl = float(trade.get("pnl", 0.0))

            if is_leveraged_position(pos):
                port["cash"] += pos.get("margin", 0.0) + pnl
            else:
                proceeds = float(trade["qty"]) * exit_price - exit_fee
                port["cash"] += proceeds

            _record_realized_pnl(port, pnl, count_closed_trade=True)
            port["total_fees_paid"] += exit_fee
            _mark_last_exit(port, sym, trade.get("ts", ts or datetime.now(timezone.utc).isoformat()), pnl=pnl)
            _register_close_outcome(port, sym, pnl, trade.get("ts", ts or datetime.now(timezone.utc).isoformat()))
            del port["positions"][sym]

    port["created"] = first_ts or port["created"]
    port["updated"] = datetime.now(timezone.utc).isoformat()
    return port


# ---------------------------------------------------------------------------
# Paper trade execution
# ---------------------------------------------------------------------------

def execute_paper_trades(
    signals: list[dict], port: dict, autopilot: dict | None = None, trial: dict | None = None,
    regime: dict | None = None,
) -> list[dict]:
    """Execute paper trades based on signals. Returns list of executed trades."""
    executed = []
    skipped = []
    now = datetime.now(timezone.utc).isoformat()
    signals = sorted(signals, key=_signal_rank_value, reverse=True)
    block_new_entries = autopilot_blocks_new_entries(autopilot)
    trial_active = candidate_trial_is_active(trial)
    trial_max_long_positions = _trial_runtime_int(trial, "max_long_positions", MAX_LONG_POSITIONS)
    trial_max_short_positions = _trial_runtime_int(trial, "max_short_positions", MAX_SHORT_POSITIONS)
    trial_disable_non_majors = bool(_trial_runtime_value(trial, "non_major_entries", False))
    trial_allow_adverse_short = _trial_allows_adverse_short_entries(trial)
    trial_risk_mult_cap = _trial_override_value(trial, "risk_mult_cap", None)
    try:
        trial_risk_mult_cap = float(trial_risk_mult_cap) if trial_risk_mult_cap is not None else None
    except Exception:
        trial_risk_mult_cap = None
    price_map = {}
    try:
        for c in get_futures_tickers():
            base = c.get("baseCurrency", c.get("symbol", "").replace("USDTM", ""))
            if base == "XBT":
                base = "BTC"
            price_map[f"{base}-USDT"] = float(c.get("lastTradePrice", c.get("markPrice", 0)) or 0)
    except Exception:
        price_map = {}

    def _direction_count(direction: str) -> int:
        return sum(1 for pos in port["positions"].values() if pos.get("direction", "LONG") == direction)

    def _record_skip(sig: dict, reason: str, **extra) -> None:
        skipped.append({
            "symbol": sig.get("symbol"),
            "side": sig.get("side"),
            "score": round(float(sig.get("score", 0.0) or 0.0), 6),
            "reason": reason,
            **extra,
        })

    # --- Entry halt stack: computed once per cycle. Exits are NEVER blocked. ---
    _entry_halt_reason = None
    _kill_mode = _kill_switch_state()
    if _kill_mode:
        _entry_halt_reason = f"operator kill switch active ({_kill_mode})"
    elif not price_map:
        # Ticker API failed: without live prices the drift gate can't run and
        # fills would anchor to stale signal prices. Refuse entries; exits and
        # stop management degrade gracefully elsewhere.
        _entry_halt_reason = "no live ticker data — exchange connectivity suspect"
    elif _daily_loss_breaker_active(port):
        _entry_halt_reason = "daily loss breaker (>%.0f%% equity lost today)" % (MAX_DAILY_LOSS_PCT * 100)
    elif _weekly_loss_breaker_active(port):
        _entry_halt_reason = "weekly loss breaker (>%.0f%% equity lost in 7d)" % (MAX_WEEKLY_LOSS_PCT * 100)
    elif _drawdown_halt_active(port):
        _entry_halt_reason = "max drawdown halt latched (>=%.0f%% from peak — run reset-halt after review)" % (MAX_DRAWDOWN_HALT_PCT * 100)
    else:
        _gap_reason = _market_gap_halt()
        _event_reason = None if _gap_reason else _event_risk_block()
        if _gap_reason:
            _entry_halt_reason = f"market gap halt — {_gap_reason}"
        elif _event_reason:
            _entry_halt_reason = _event_reason
        elif str((regime or {}).get("source", "")).lower() == "fallback":
            _entry_halt_reason = "regime detector in fallback mode — no fresh market state"
    if _entry_halt_reason:
        print(f"  ENTRY HALT: {_entry_halt_reason}")
    _stress_mode = False if _entry_halt_reason else _btc_stress_mode()
    if _stress_mode:
        print(f"  STRESS MODE: BTC 24h vol >= {BTC_STRESS_VOL_24H:.0%} — one position max, size halved.")
    # Compute regime gate once per cycle (score doesn't change mid-loop)
    _regime_score = float((regime or {}).get("score", 0.0))
    regime_gate = _regime_controls(_regime_score)
    for sig in signals:
        sym = sig["symbol"]

        # --- Close position if signal opposes current direction ---
        _existing = port["positions"].get(sym)
        _closes = (
            (_existing and sig["side"] == "SELL" and _existing.get("direction", "LONG") == "LONG") or
            (_existing and sig["side"] == "BUY"  and _existing.get("direction") == "SHORT")
        )
        if _closes and sym in port["positions"]:
            pos = port["positions"][sym]
            # Close at the live ticker when available (no drift gate — closing reduces risk).
            _close_base = float(price_map.get(sym, 0.0) or 0.0) or float(sig.get("price", 0.0) or 0.0)
            _accrue_funding(sym, pos, _close_base, now, port=port)
            exit_price, exit_profile = _apply_execution_price(
                _close_base,
                sym,
                pos.get("direction", "LONG"),
                is_entry=False,
                trial=trial,
                autopilot=autopilot,
                pos=pos,
                trigger="SIGNAL_FLIP",
            )
            exit_fee = pos["qty"] * exit_price * TAKER_FEE
            pnl, cash_delta = close_position_accounting(pos, exit_price, exit_fee)
            port["cash"] += cash_delta
            pnl = round(pnl, 4)

            trade = {
                "type": "CLOSE",
                "symbol": sym,
                "side": "SELL",
                "qty": pos["qty"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "entry_fee": pos["entry_fee"],
                "exit_fee": round(exit_fee, 4),
                "funding_paid": round(float(pos.get("funding_paid", 0.0)), 8),
                "pnl": round(pnl, 4),
                "setup_tag": pos.get("setup_tag"),
                "setup_score": pos.get("setup_score"),
                "execution_mode": exit_profile.get("mode"),
                "execution_bps": exit_profile.get("fill_bps"),
                "reason": sig["reasons"],
                "ts": now,
            }
            append_trade(trade)
            executed.append(trade)

            port["realized_pnl"] += pnl
            port["total_fees_paid"] += exit_fee
            port["total_trades"] += 1
            if pnl > 0:
                port["wins"] += 1
                port["gross_profit"] += pnl
            else:
                port["losses"] += 1
                port["gross_loss"] += abs(pnl)
            _mark_last_exit(port, sym, now, pnl=pnl)
            _register_close_outcome(port, sym, pnl, now)
            del port["positions"][sym]
            continue

        # --- Open LONG (BUY) or SHORT (SELL) if no position exists ---
        _dir = "LONG" if sig["side"] == "BUY" else "SHORT"
        if sym not in port["positions"] and sig["side"] in ("BUY", "SELL"):
            if _entry_halt_reason:
                # Skip (not break) so close-on-flip signals later in the list still process.
                _record_skip(sig, f"entry halt — {_entry_halt_reason}")
                continue
            if block_new_entries:
                _record_skip(sig, "autopilot blocked new entries")
                continue
            # Use the filtered execution score so entry admission matches the
            # same selection/feedback stack that produced the actionable signal.
            ml_confidence = float(sig.get("ml_confidence", sig.get("raw_score", sig.get("score", 0.0))) or 0.0)
            execution_score = float(
                sig.get(
                    "execution_score",
                    sig.get("score", ml_confidence) if EXECUTION_USE_FILTERED_SCORE else ml_confidence,
                ) or 0.0
            )
            threshold = _entry_score_threshold(sig, trial)
            if execution_score < threshold:
                _record_skip(
                    sig,
                    "execution score below threshold",
                    execution_score=round(execution_score, 4),
                    ml_conf=round(ml_confidence, 4),
                    threshold=round(threshold, 6),
                )
                continue
            if not bool(sig.get("quality_pass", True)):
                _record_skip(sig, "quality gate failed at execution")
                continue
            if trial_active and trial_disable_non_majors and _dir == "LONG" and sym not in MAJOR_SYMBOLS:
                _record_skip(sig, "trial restricts longs to major symbols")
                continue
            long_cap = trial_max_long_positions if trial_active and _dir == "LONG" else MAX_LONG_POSITIONS
            if _dir == "LONG" and _direction_count("LONG") >= long_cap:
                _record_skip(sig, "max long positions reached", current=_direction_count("LONG"), cap=long_cap)
                continue
            if _dir == "SHORT" and _direction_count("SHORT") >= trial_max_short_positions:
                _record_skip(sig, "max short positions reached", current=_direction_count("SHORT"), cap=trial_max_short_positions)
                continue
            if _in_reentry_cooldown(port, sym, now):
                _record_skip(sig, "re-entry cooldown active")
                continue
            if _in_loss_lockout(port, sym, now):
                _record_skip(sig, "loss lockout active")
                continue
            # Regime-gated position cap (replaces flat MAX_POSITIONS check)
            effective_max = _trial_effective_position_cap(
                trial,
                _dir,
                regime_gate["max_positions"],
                trial_max_short_positions,
            )
            if len(port["positions"]) >= effective_max:
                if effective_max == 0:
                    _record_skip(sig, "regime adverse — no new entries", regime_score=round(_regime_score, 4))
                    continue
                if not ENABLE_SIGNAL_ROTATION:
                    _record_skip(sig, "regime-gated position cap reached", max=effective_max)
                    continue
                rotation = None
                for open_sym, pos in port["positions"].items():
                    if pos.get("direction", "LONG") == _dir:
                        continue
                    live_price = price_map.get(open_sym, pos["entry_price"])
                    effective_live_price, _ = _mark_to_exit_price(
                        open_sym,
                        pos,
                        live_price,
                        trial=trial,
                        autopilot=autopilot,
                    )
                    entry_notional = float(pos["qty"]) * float(pos["entry_price"])
                    if entry_notional <= 0:
                        continue
                    pnl = position_unrealized_pnl(pos, effective_live_price)
                    pnl_pct = pnl / entry_notional
                    score_gap = float(sig.get("score", 0.0)) - float(pos.get("signal_score", 0.0))
                    if pnl_pct > -ROTATION_MIN_OPEN_LOSS_PCT:
                        continue
                    if score_gap < ROTATION_MIN_SCORE_ADVANTAGE:
                        continue
                    candidate = {
                        "symbol": open_sym,
                        "position": pos,
                        "live_price": live_price,
                        "effective_live_price": effective_live_price,
                        "pnl_pct": pnl_pct,
                        "score_gap": score_gap,
                    }
                    if rotation is None or candidate["pnl_pct"] < rotation["pnl_pct"]:
                        rotation = candidate
                if rotation is None:
                    _record_skip(sig, "regime-gated position cap reached and no rotation candidate", current=len(port["positions"]), cap=effective_max)
                    continue
                pos = rotation["position"]
                exit_price, exit_profile = _apply_execution_price(
                    rotation["live_price"],
                    rotation["symbol"],
                    pos.get("direction", "LONG"),
                    is_entry=False,
                    trial=trial,
                    autopilot=autopilot,
                    pos=pos,
                    trigger="SIGNAL_ROTATION",
                )
                exit_fee = pos["qty"] * exit_price * TAKER_FEE
                pnl, cash_delta = close_position_accounting(pos, exit_price, exit_fee)
                port["cash"] += cash_delta
                pnl = round(pnl, 4)
                trade = {
                    "type": "CLOSE",
                    "symbol": rotation["symbol"],
                    "side": "SELL",
                    "trigger": "SIGNAL_ROTATION",
                    "qty": pos["qty"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "entry_fee": pos["entry_fee"],
                    "exit_fee": round(exit_fee, 4),
                    "pnl": round(pnl, 4),
                    "setup_tag": pos.get("setup_tag"),
                    "setup_score": pos.get("setup_score"),
                    "execution_mode": exit_profile.get("mode"),
                    "execution_bps": exit_profile.get("fill_bps"),
                    "reason": [
                        f"Rotated into stronger {_dir} signal for {sym}",
                        f"Signal score gap {rotation['score_gap']:.3f}",
                        f"Open PnL {rotation['pnl_pct']:.2%}",
                    ],
                    "ts": now,
                }
                append_trade(trade)
                executed.append(trade)
                port["realized_pnl"] += pnl
                port["total_fees_paid"] += exit_fee
                port["total_trades"] += 1
                if pnl > 0:
                    port["wins"] += 1
                    port["gross_profit"] += pnl
                else:
                    port["losses"] += 1
                    port["gross_loss"] += abs(pnl)
                _mark_last_exit(port, rotation["symbol"], now, pnl=pnl)
                _register_close_outcome(port, rotation["symbol"], pnl, now)
                del port["positions"][rotation["symbol"]]
            # Majors-only gate — weak regimes OR strategy-params override restrict to major symbols
            _majors_only_override = str(_load_strategy_params().get("majors_only_override", "")).strip().lower() == "true"
            bypass_majors_only = trial_active and _dir == "SHORT" and trial_allow_adverse_short
            if _trial_bypasses_major_only_for_symbol(trial, sym, _dir):
                bypass_majors_only = True
            if not bypass_majors_only and (regime_gate["majors_only"] or _majors_only_override) and sym not in MAJOR_SYMBOLS:
                _record_skip(sig, "majors-only restriction active", regime_score=round(_regime_score, 4))
                continue
            # --- EMA trend-direction filter: only trade WITH the trend ---
            _trend_candles = sig.get("candles") or []
            if len(_trend_candles) >= 50:
                _closes = [_trend_filter_close(c) for c in _trend_candles[-50:]]
                def _ema(vals, span):
                    k = 2 / (span + 1)
                    out = vals[0]
                    for v in vals[1:]:
                        out = v * k + out * (1 - k)
                    return out
                _ema21 = _ema(_closes[-21:], 21)
                _ema50 = _ema(_closes, 50)
                _last_close = _closes[-1]
                if _dir == "LONG" and (_last_close < _ema21 or _last_close < _ema50):
                    _record_skip(sig, "EMA trend filter — price below EMA21/50 for long", ema21=round(_ema21, 2), ema50=round(_ema50, 2), close=round(_last_close, 2))
                    continue
                if _dir == "SHORT" and not trial_allow_adverse_short and (_last_close > _ema21 or _last_close > _ema50):
                    _record_skip(sig, "EMA trend filter — price above EMA21/50 for short", ema21=round(_ema21, 2), ema50=round(_ema50, 2), close=round(_last_close, 2))
                    continue
            corr_full, corr_reason = _correlation_group_full(sym, port)
            if corr_full:
                _record_skip(sig, f"correlation filter — {corr_reason}")
                continue
            if _stress_mode and len(port.get("positions", {})) >= 1:
                _record_skip(sig, "stress mode — all-crypto one correlation group, position already open")
                continue
            if _dir == "LONG":
                _funding = _get_funding_rate(sym)
                if _funding is not None and _funding < FUNDING_RATE_LONG_SKIP_THRESHOLD:
                    _record_skip(sig, f"funding rate {_funding:.4f} too negative for long", funding_rate=_funding)
                    continue
            # --- Execution realism: fill at the LIVE ticker, not the signal-bar close.
            # Reject if the market already ran away from the signal price.
            _sig_px = float(sig.get("price", 0.0) or 0.0)
            _live_px = float(price_map.get(sym, 0.0) or 0.0)
            if _live_px > 0 and _sig_px > 0:
                _drift = abs(_live_px - _sig_px) / _sig_px
                if _drift > SIGNAL_DRIFT_MAX_PCT:
                    _record_skip(
                        sig,
                        "signal drift %.2f%% exceeds %.1f%% — price moved since signal" % (_drift * 100, SIGNAL_DRIFT_MAX_PCT * 100),
                        signal_price=_sig_px,
                        live_price=_live_px,
                    )
                    continue
                _fill_base = _live_px
            else:
                _fill_base = _sig_px
            current_eq = equity(port, autopilot=autopilot, trial=trial)
            # Cash reserve guard — always keep MIN_CASH_RESERVE of equity in cash
            min_cash = current_eq * MIN_CASH_RESERVE
            if port["cash"] <= min_cash:
                _record_skip(sig, "cash reserve guard active", cash=round(port["cash"], 4), min_cash=round(min_cash, 4))
                continue  # not enough dry powder
            available_cash_for_entry = max(0.0, port["cash"] - min_cash)
            # Cap per-entry to leave room for other positions.
            # Divide available cash by remaining open slots so earlier entries
            # don't hog all the capital.
            open_count = len(port.get("positions", {}))
            remaining_slots = max(1, effective_max - open_count)
            per_slot_cash = available_cash_for_entry / remaining_slots
            available_cash_for_entry = min(available_cash_for_entry, per_slot_cash)
            _sig_candle_data = sig.get("candles", [])
            _dynamic_stop_pct = _compute_atr_stop_pct(_sig_candle_data, float(sig.get("price", 0.0) or 0.0), _dir)
            stop_dist = float(sig.get("price", 0.0) or 0.0) * _dynamic_stop_pct
            if stop_dist <= 0:
                _record_skip(sig, "invalid stop distance from signal price", signal_price=round(float(sig.get("price", 0.0) or 0.0), 8))
                continue
            # Apply regime size multiplier (shrinks position in bear/high-vol regimes)
            size_mult = float(sig.get("size_mult", 1.0))
            # Apply regime-gated size multiplier (linear interpolation based on regime score)
            size_mult *= regime_gate["size_mult"]
            if _stress_mode:
                size_mult *= 0.5
            if trial_active and trial_risk_mult_cap is not None:
                size_mult = min(size_mult, trial_risk_mult_cap)
            effective_risk = MAX_RISK_PCT * size_mult
            entry_price, entry_profile = _apply_execution_price(
                _fill_base,
                sym,
                _dir,
                is_entry=True,
                trial=trial,
                autopilot=autopilot,
                quality_metrics=sig.get("quality_metrics"),
                trigger="OPEN",
            )
            # Dynamic ATR-based stop distance (recomputed on entry_price post-slippage)
            _dynamic_stop_pct = _compute_atr_stop_pct(_sig_candle_data, entry_price, _dir)
            stop_dist = entry_price * _dynamic_stop_pct
            qty = (current_eq * effective_risk) / stop_dist
            notional = qty * entry_price
            margin = notional / LEVERAGE
            # --- CIRCUIT BREAKER: hard cap margin per position ---
            max_margin_allowed = current_eq * MAX_MARGIN_PER_POSITION_PCT
            if margin > max_margin_allowed:
                margin = max_margin_allowed
                notional = margin * LEVERAGE
                qty = notional / entry_price
            entry_fee = notional * TAKER_FEE
            if margin + entry_fee > available_cash_for_entry:
                avail = available_cash_for_entry * 0.98 - entry_fee
                if avail <= 0:
                    _record_skip(
                        sig,
                        "insufficient free cash after reserve and fees",
                        cash=round(port["cash"], 4),
                        min_cash=round(min_cash, 4),
                        entry_fee=round(entry_fee, 4),
                    )
                    continue
                margin, notional = avail, avail * LEVERAGE
                qty = notional / entry_price
                entry_fee = notional * TAKER_FEE
            if notional < 10:
                _record_skip(sig, "order notional below minimum", notional=round(notional, 6))
                continue
            sl, tp, _rr_ratio = _entry_exit_levels(entry_price, _dynamic_stop_pct, _dir)
            # --- R:R gate: reject trades where reward < MIN_RR_RATIO × risk ---
            if _rr_gate_blocks(_rr_ratio, MIN_RR_RATIO):
                _record_skip(sig, "R:R ratio %.2f < minimum %.1f" % (_rr_ratio, MIN_RR_RATIO), rr=round(_rr_ratio, 2), min_rr=MIN_RR_RATIO)
                continue
            port["cash"] -= (margin + entry_fee)
            risk_per_unit = abs(entry_price - sl)
            port["total_fees_paid"] += entry_fee
            port["positions"][sym] = {
                "direction": _dir, "qty": round(qty, 8), "margin": round(margin, 4),
                "original_qty": round(qty, 8),
                "entry_price": entry_price, "entry_fee": round(entry_fee, 4),
                "stop_loss": sl, "take_profit": tp,
                "initial_stop_loss": sl, "initial_risk_per_unit": round(risk_per_unit, 8),
                "open_ts": now, "opened": now, "signal_score": sig["score"],
                "raw_signal_score": sig.get("raw_score", sig["score"]),
                "setup_tag": sig.get("setup_tag"),
                "setup_score": sig.get("setup_score"),
                "quality_score": float(sig.get("quality_metrics", {}).get("quality_score", 0.0) or 0.0),
                "execution_mode": entry_profile.get("mode"),
                "entry_execution_bps": entry_profile.get("fill_bps"),
                "best_price": entry_price, "trailing_active": False,
                "highest_unrealized_pct": 0.0, "highest_r": 0.0,
                "stop_stage": "initial",
                "partial_take_profits": {"tp1_done": False, "tp2_done": False},
                "trial_candidate_id": trial.get("candidate_id") if trial_active else None,
                "dynamic_stop_pct": round(_dynamic_stop_pct, 6),
            }
            trade = {
                "type": "OPEN", "symbol": sym, "direction": _dir,
                "side": sig["side"], "leverage": LEVERAGE,
                "qty": round(qty, 8), "entry_price": entry_price,
                "entry_fee": round(entry_fee, 4), "stop_loss": sl, "take_profit": tp,
                "original_qty": round(qty, 8),
                "initial_stop_loss": sl,
                "initial_risk_per_unit": round(risk_per_unit, 8),
                "best_price": entry_price, "trailing_active": False,
                "highest_unrealized_pct": 0.0, "highest_r": 0.0,
                "stop_stage": "initial",
                "tp1_done": False, "tp2_done": False,
                "signal_score": sig["score"], "raw_signal_score": sig.get("raw_score", sig["score"]),
                "setup_tag": sig.get("setup_tag"), "setup_score": sig.get("setup_score"),
                "quality_score": float(sig.get("quality_metrics", {}).get("quality_score", 0.0) or 0.0),
                "execution_mode": entry_profile.get("mode"),
                "entry_execution_bps": entry_profile.get("fill_bps"),
                "trial_candidate_id": trial.get("candidate_id") if trial_active else None,
                "open_ts": now, "reason": sig["reasons"], "ts": now,
            }
            append_trade(trade)
            executed.append(trade)

    skip_summary = {}
    for row in skipped:
        reason = str(row.get("reason", "") or "")
        if reason:
            skip_summary[reason] = skip_summary.get(reason, 0) + 1
    save_last_execution({
        "generated_at": now,
        "trial_active": trial_active,
        "autopilot_mode": str((autopilot or {}).get("mode", "") or ""),
        "execution_permission": "blocked" if block_new_entries else "allowed",
        "signal_count": len(signals),
        "executed_count": len(executed),
        "executed_symbols": [row.get("symbol") for row in executed],
        "skipped_count": len(skipped),
        "skip_summary": dict(sorted(skip_summary.items(), key=lambda kv: kv[1], reverse=True)),
        "skipped": skipped[:25],
    })
    return executed


def check_stops(
    port: dict,
    *,
    autopilot: dict | None = None,
    trial: dict | None = None,
) -> list[dict]:
    """Check stop-loss and take-profit on open positions using live prices."""
    if not port.get("positions"):
        return []

    price_map = {}
    for c in get_futures_tickers():
        base = c.get("baseCurrency", c.get("symbol", "").replace("USDTM", ""))
        if base == "XBT": base = "BTC"
        try:
            price_map[f"{base}-USDT"] = float(c.get("lastTradePrice", c.get("markPrice", 0)) or 0)
        except (ValueError, TypeError):
            price_map[f"{base}-USDT"] = 0.0

    closed = []
    now = datetime.now(timezone.utc).isoformat()
    to_remove = []
    _kill_flatten = _kill_switch_state() == "flatten"
    if _kill_flatten:
        print("  KILL_FLATTEN: operator kill switch — closing all open positions.")
    _daily_flatten = (not _kill_flatten) and _daily_loss_breaker_active(port)
    if _daily_flatten:
        print("  DAILY LOSS BREAKER: flattening all open positions to stop the bleed for today.")

    for sym, pos in port["positions"].items():
        price = price_map.get(sym)
        if not price:
            print(f"[WARN] Missing live price for open position {sym}; skipping stop check for this symbol this cycle.", file=sys.stderr)
            continue

        _accrue_funding(sym, pos, price, now, port=port)

        policy_trigger_reason = None
        if str(pos.get("direction", "LONG")).upper() == "LONG":
            policy_trigger_reason = _long_setup_policy_reason(pos.get("setup_tag", "unknown"))
        if _kill_flatten:
            triggered = "KILL_SWITCH"
        elif _daily_flatten:
            triggered = "DAILY_LOSS_STOP"
        elif policy_trigger_reason:
            triggered = "POLICY_EXIT"
        else:
            triggered = None

        _update_position_extrema(pos, price)
        _pos_candles = None
        try:
            _pos_candles = get_klines(sym, "1hour", 50)
        except Exception as _e:
            print(f"[WARN] check_stops: candle fetch failed for {sym}: {_e}", file=sys.stderr)
        trail_update = _update_trailing_exit(pos, price, candles=_pos_candles)
        if trail_update:
            append_trade({
                "type": "TRAILING_UPDATE",
                "symbol": sym,
                "direction": pos.get("direction", "LONG"),
                "trailing_tier": trail_update.get("trailing_tier"),
                "move_pct": trail_update.get("move_pct"),
                "giveback_pct": trail_update.get("giveback_pct"),
                "lock_share": trail_update.get("lock_share"),
                "entry_price": round(float(trail_update["entry_price"]), 8),
                "price": round(float(trail_update["price"]), 8),
                "best_price_before": round(float(trail_update["best_price_before"]), 8),
                "best_price_after": round(float(trail_update["best_price_after"]), 8),
                "stop_loss_before": round(float(trail_update["stop_loss_before"]), 8),
                "stop_loss_after": round(float(trail_update["stop_loss_after"]), 8),
                "trailing_was_active": bool(trail_update["trailing_was_active"]),
                "trailing_is_active": bool(trail_update["trailing_is_active"]),
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        partials = pos.setdefault("partial_take_profits", {"tp1_done": False, "tp2_done": False})
        move_r = _position_move_r(pos, price)
        for stage_name, target_r, fraction in _partial_tp_plan(pos):
            if partials.get(stage_name):
                continue
            if move_r < target_r:
                continue
            partial_trade = _execute_partial_take_profit(
                port,
                sym,
                pos,
                price,
                stage_name,
                fraction,
                now,
                autopilot=autopilot,
                trial=trial,
            )
            if partial_trade:
                partials[stage_name] = True
                closed.append(partial_trade)
                _apply_partial_profit_stop_lock(pos, stage_name)
                if pos.get("qty", 0.0) <= 0:
                    to_remove.append(sym)
                    break
        if sym in to_remove:
            continue
        _d = pos.get("direction", "LONG")
        trailing_active = bool(pos.get("trailing_active", False))
        if not triggered:
            if _d == "SHORT":
                if price >= pos["stop_loss"]:
                    triggered = "STOP_LOSS"
                elif not trailing_active and price <= pos["take_profit"]:
                    triggered = "TAKE_PROFIT"
            else:
                if price <= pos["stop_loss"]:
                    triggered = "STOP_LOSS"
                elif not trailing_active and price >= pos["take_profit"]:
                    triggered = "TAKE_PROFIT"
        opened_at = pos.get("open_ts") or pos.get("opened")
        if not triggered and opened_at:
            try:
                opened_dt = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
                age_h = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
                if age_h >= MAX_HOLD_HOURS:
                    triggered = "TIME_STOP"
            except Exception:
                pass

        if triggered:
            trigger_label = _close_trigger_label(pos, triggered)
            exit_price, exit_profile = _apply_execution_price(
                price,
                sym,
                pos.get("direction", "LONG"),
                is_entry=False,
                trial=trial,
                autopilot=autopilot,
                pos=pos,
                trigger=trigger_label,
            )
            exit_fee = pos["qty"] * exit_price * TAKER_FEE
            pnl, cash_delta = close_position_accounting(pos, exit_price, exit_fee)
            port["cash"] += cash_delta
            pnl = round(pnl, 4)

            trade = {
                "type": "CLOSE",
                "symbol": sym,
                "side": "SELL",
                "trigger": trigger_label,
                "qty": pos["qty"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "entry_fee": pos["entry_fee"],
                "exit_fee": round(exit_fee, 4),
                "funding_paid": round(float(pos.get("funding_paid", 0.0)), 8),
                "pnl": round(pnl, 4),
                "setup_tag": pos.get("setup_tag"),
                "setup_score": pos.get("setup_score"),
                "quality_score": pos.get("quality_score"),
                "best_price": pos.get("best_price"),
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit"),
                "trailing_active": pos.get("trailing_active"),
                "stop_stage": pos.get("stop_stage"),
                "highest_unrealized_pct": pos.get("highest_unrealized_pct"),
                "highest_r": pos.get("highest_r"),
                "execution_mode": exit_profile.get("mode"),
                "execution_bps": exit_profile.get("fill_bps"),
                "reason": [policy_trigger_reason] if policy_trigger_reason else pos.get("reason"),
                "ts": now,
            }
            append_trade(trade)
            closed.append(trade)

            _record_realized_pnl(port, pnl, count_closed_trade=True)
            port["total_fees_paid"] += exit_fee
            _mark_last_exit(port, sym, now, pnl=pnl)
            _register_close_outcome(port, sym, pnl, now)
            to_remove.append(sym)

    for sym in to_remove:
        del port["positions"][sym]

    return closed


def update_drawdown(port: dict, current_equity: float):
    """Track peak equity and max drawdown."""
    current_equity = _refresh_portfolio_equity_snapshot(port, current_equity)
    if current_equity > port.get("peak_equity", STARTING_BALANCE):
        port["peak_equity"] = current_equity
    peak = port["peak_equity"]
    if peak > 0:
        dd = (peak - current_equity) / peak
        if dd > port.get("max_drawdown", 0):
            port["max_drawdown"] = round(dd, 6)


def run_debate_on_signals(signals: list[dict]) -> list[dict]:
    """Placeholder adversarial review hook for signals.

    Some deployed production copies already call this hook from `cmd_scan()`.
    Keep it as a no-op until the debate layer is implemented end-to-end so the
    paper trader never crashes on an undefined symbol.
    """
    return signals


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_scan():
    """Scan market and generate signals."""
    print("=== QuantForge Paper Trading — Market Scan ===\n")
    print("Screening coins...")
    coins = screen_coins(SCAN_TOP_N)
    port = load_portfolio()
    autopilot = load_autopilot_report()
    candidate_trial = sync_candidate_trial_state(autopilot)
    now = datetime.now(timezone.utc).isoformat()
    print(f"Found {len(coins)} coins above ${MIN_VOLUME_USDT:,.0f} 24h volume\n")

    if not coins:
        print("No coins passed screening. Check network/API.")
        return []

    print(f"{'Symbol':<14} {'Price':>12} {'24h%':>8} {'Vol($M)':>10}")
    print("-" * 48)
    for c in coins[:10]:
        print(f"{c['symbol']:<14} ${c['price']:>10,.4f} {c['change_pct']:>+7.2f}% {c['vol_usdt']/1e6:>9.1f}M")
    print()

    signals = []
    scan_rows = []
    flow = {
        "scan_top_n": SCAN_TOP_N,
        "screened_universe": len(coins),
        "open_position_skips": 0,
        "reentry_cooldown_skips": 0,
        "loss_lockout_skips": 0,
        "quality_passed": 0,
        "quality_blocked": 0,
        "model_no_signal": 0,
        "threshold_miss": 0,
        "trained_pair_blocked": 0,
        "ml_gate_blocked": 0,
        "selection_blocked": 0,
        "feedback_blocked": 0,
        "error_count": 0,
        "actionable_signals": 0,
        "buy_signals": 0,
        "sell_signals": 0,
    }
    # Fetch regime ONCE for the whole scan — all symbols use same macro context
    _regime = _get_regime()
    feedback = _recent_close_feedback(now)
    print(f"Analyzing signals... [{_regime['label']} regime, score={_regime['score']}]")
    for c in coins:
        try:
            if c["symbol"] in port.get("positions", {}):
                flow["open_position_skips"] += 1
                scan_rows.append({"symbol": c["symbol"], "status": "open_position"})
                continue
            if _in_reentry_cooldown(port, c["symbol"], now):
                print(f"  {c['symbol']:<14} SKIP   re-entry cooldown active")
                flow["reentry_cooldown_skips"] += 1
                scan_rows.append({"symbol": c["symbol"], "status": "skip", "reason": "re-entry cooldown active"})
                continue
            if _in_loss_lockout(port, c["symbol"], now):
                print(f"  {c['symbol']:<14} SKIP   loss lockout active")
                flow["loss_lockout_skips"] += 1
                scan_rows.append({"symbol": c["symbol"], "status": "skip", "reason": "loss lockout active"})
                continue
            candles = get_klines(c["symbol"], "1hour", 300)
            quality_pass, quality_reasons, quality_metrics = _passes_symbol_quality_filters(c, candles)
            if not quality_pass:
                print(f"  {c['symbol']:<14} SKIP   quality filter — {quality_reasons[0]}")
                flow["quality_blocked"] += 1
                scan_rows.append({
                    "symbol": c["symbol"],
                    "status": "skip",
                    "reason": f"quality filter — {quality_reasons[0]}",
                    "quality_metrics": quality_metrics,
                })
                continue
            flow["quality_passed"] += 1
            sig = generate_signals_ml(c["symbol"], candles, regime=_regime, trial=candidate_trial)
            if sig:
                if sig.get("side") not in ("BUY", "SELL") or not bool(sig.get("actionable", sig.get("side") in ("BUY", "SELL"))):
                    decision_stage = str(sig.get("decision_stage", "model_no_signal") or "model_no_signal")
                    flow["model_no_signal"] += 1
                    if decision_stage != "model_no_signal" and decision_stage in flow:
                        flow[decision_stage] += 1
                    scan_rows.append({
                        "symbol": c["symbol"],
                        "status": "hold",
                        "reason": (sig.get("reasons") or ["no actionable signal"])[0],
                        "decision_stage": decision_stage,
                        "long_confidence": sig.get("long_confidence"),
                        "short_confidence": sig.get("short_confidence"),
                        "long_threshold": sig.get("long_threshold"),
                        "short_threshold": sig.get("short_threshold"),
                        "quality_metrics": quality_metrics,
                        "setup_tag": sig.get("setup_tag"),
                    })
                    continue
                sel_allowed, sel_reasons, score_adj, selection_size_mult = _selection_adjustments(
                    c["symbol"], sig, quality_metrics, candidate_trial, _regime
                )
                if not sel_allowed:
                    print(f"  {c['symbol']:<14} SKIP   {sel_reasons[0]}")
                    flow["selection_blocked"] += 1
                    scan_rows.append({
                        "symbol": c["symbol"],
                        "status": "skip",
                        "reason": sel_reasons[0],
                        "setup_tag": sig.get("setup_tag"),
                        "quality_metrics": quality_metrics,
                    })
                    continue
                allowed, feedback_reasons, score_penalty, feedback_size_mult = _feedback_adjustments(
                    c["symbol"], sig.get("setup_tag", "unknown"), feedback, now
                )
                if not allowed:
                    print(f"  {c['symbol']:<14} SKIP   {feedback_reasons[0]}")
                    flow["feedback_blocked"] += 1
                    scan_rows.append({
                        "symbol": c["symbol"],
                        "status": "skip",
                        "reason": feedback_reasons[0],
                        "setup_tag": sig.get("setup_tag"),
                        "quality_metrics": quality_metrics,
                    })
                    continue
                sig["quality_pass"] = quality_pass
                sig["quality_metrics"] = quality_metrics
                sig["candles"] = candles  # carry candles for ATR stop sizing in execute_paper_trades
                sig["selection_size_mult"] = selection_size_mult
                sig["feedback_risk_mult"] = feedback_size_mult
                sig["feedback_score_penalty"] = score_penalty
                sig["size_mult"] = round(float(sig.get("size_mult", 1.0)) * selection_size_mult * feedback_size_mult, 4)
                if score_adj != 0:
                    sig["score"] = round(max(0.0, float(sig.get("score", 0.0)) + score_adj), 4)
                    sig["reasons"].append(f"Selection score adj {score_adj:+.3f}")
                if score_penalty > 0:
                    sig["score"] = round(max(0.0, float(sig.get("score", 0.0)) - score_penalty), 4)
                    sig["reasons"].append(f"Feedback penalty {score_penalty:.3f}")
                sig["execution_score"] = round(float(sig.get("score", 0.0) or 0.0), 4)
                sig["edge_rank"] = round(_signal_rank_value(sig), 6)
                for reason in sel_reasons:
                    sig["reasons"].append(reason)
                for reason in feedback_reasons:
                    sig["reasons"].append(reason)
                signals.append(sig)
                flow["actionable_signals"] += 1
                if sig["side"] == "BUY":
                    flow["buy_signals"] += 1
                elif sig["side"] == "SELL":
                    flow["sell_signals"] += 1
                append_signal(sig)
                direction = "LONG" if sig["side"] == "BUY" else "SHORT"
                print(f"  {sig['symbol']:<14} {direction:<6} score={sig['score']:>+.3f}  RSI={sig['rsi']:.1f}  {', '.join(sig['reasons'][:2])}")
                scan_rows.append({
                    "symbol": sig["symbol"],
                    "status": "signal",
                    "side": sig["side"],
                    "score": sig["score"],
                    "edge_rank": sig.get("edge_rank"),
                    "raw_score": sig.get("raw_score", sig["score"]),
                    "setup_tag": sig.get("setup_tag"),
                    "quality_score": quality_metrics.get("quality_score"),
                    "regime": _regime.get("label"),
                    "regime_entropy_label": _regime.get("entropy_label"),
                    "reasons": sig.get("reasons", [])[:4],
                })
        except Exception as e:
            print(f"  {c['symbol']}: error — {e}")
            flow["error_count"] += 1
            scan_rows.append({"symbol": c["symbol"], "status": "error", "reason": str(e)})
        time.sleep(0.15)  # rate limit courtesy

    if not signals:
        print("\n  No actionable signals found.")
        print(
            "  Flow:"
            f" quality_passed={flow['quality_passed']},"
            f" quality_blocked={flow['quality_blocked']},"
            f" model_no_signal={flow['model_no_signal']},"
            f" threshold_miss={flow['threshold_miss']},"
            f" selection_blocked={flow['selection_blocked']},"
            f" cooldown_skips={flow['reentry_cooldown_skips']},"
            f" loss_lockout_skips={flow['loss_lockout_skips']}"
        )
    else:
        buys = sum(1 for s in signals if s["side"] == "BUY")
        sells = sum(1 for s in signals if s["side"] == "SELL")
        print(f"\nSignals: {buys} BUY, {sells} SELL")
        print(
            "Flow:"
            f" quality_passed={flow['quality_passed']},"
            f" model_no_signal={flow['model_no_signal']},"
            f" selection_blocked={flow['selection_blocked']},"
            f" feedback_blocked={flow['feedback_blocked']},"
            f" actionable={flow['actionable_signals']}"
        )

    save_last_scan({
        "ts": now,
        "regime": _regime,
        "feedback": feedback,
        "signals": signals,
        "results": scan_rows,
        "flow": flow,
    })
    normalize_last_scan_artifact()

    return signals


def cmd_status():
    """Show portfolio status and stats."""
    port = load_portfolio()
    autopilot = load_autopilot_report()
    candidate_trial = load_candidate_trial()

    # Get live prices for unrealized PnL
    prices = {}
    if port.get("positions"):
        try:
            for c in get_futures_tickers():
                base = c.get("baseCurrency", c.get("symbol", "").replace("USDTM", ""))
                if base == "XBT": base = "BTC"
                prices[f"{base}-USDT"] = float(c.get("lastTradePrice", c.get("markPrice", 0)) or 0)
        except Exception:
            pass

    current_eq = equity(port, prices, autopilot=autopilot, trial=candidate_trial)
    unrealized = 0.0
    reserved_margin = 0.0
    for sym, pos in port.get("positions", {}).items():
        price = prices.get(sym, pos["entry_price"])
        price, _ = _mark_to_exit_price(sym, pos, price, trial=candidate_trial, autopilot=autopilot)
        unrealized += position_unrealized_pnl(pos, price)
        reserved_margin += float(pos.get("margin", 0.0))

    open_entry_fees = _open_entry_fee_total(port)
    total_pnl = port["realized_pnl"] + unrealized - open_entry_fees
    pnl_pct = (total_pnl / port["starting_balance"]) * 100
    win_rate = (port["wins"] / port["total_trades"] * 100) if port["total_trades"] > 0 else 0
    avg_win = (port["gross_profit"] / port["wins"]) if port["wins"] > 0 else 0
    avg_loss = (port["gross_loss"] / port["losses"]) if port["losses"] > 0 else 0
    profit_factor = (port["gross_profit"] / port["gross_loss"]) if port["gross_loss"] > 0 else float("inf")

    print("=" * 55)
    print("    QuantForge Paper Trading — Portfolio Status")
    print("=" * 55)
    print(f"  Starting Balance:   ${port['starting_balance']:>10,.2f}")
    print(f"  Current Equity:     ${current_eq:>10,.2f}")
    print(f"  Cash Available:     ${port['cash']:>10,.2f}")
    print(f"  Reserved Margin:    ${reserved_margin:>10,.2f}")
    print(f"  Realized PnL:       ${port['realized_pnl']:>+10,.2f}")
    print(f"  Unrealized PnL:     ${unrealized:>+10,.2f}")
    print(f"  Open Entry Fees:    ${open_entry_fees:>10,.4f}")
    print(f"  Total PnL:          ${total_pnl:>+10,.2f}  ({pnl_pct:>+.2f}%)")
    print(f"  Total Fees Paid:    ${port['total_fees_paid']:>10,.4f}")
    print(f"  Max Drawdown:       {port.get('max_drawdown', 0) * 100:>10.2f}%")
    print()
    print(f"  Total Trades:       {port['total_trades']:>10}")
    print(f"  Wins / Losses:      {port['wins']:>5} / {port['losses']}")
    print(f"  Win Rate:           {win_rate:>10.1f}%")
    print(f"  Avg Win:            ${avg_win:>10,.2f}")
    print(f"  Avg Loss:           ${avg_loss:>10,.2f}")
    print(f"  Profit Factor:      {profit_factor:>10.2f}")
    print()

    if port.get("positions"):
        print(f"  Open Positions ({len(port['positions'])}):")
        print(f"  {'Symbol':<14} {'Dir':<8} {'Qty':>12} {'Entry':>10} {'Current':>10} {'PnL':>10}")
        print("  " + "-" * 72)
        for sym, pos in port["positions"].items():
            cur_price = prices.get(sym, pos["entry_price"])
            cur_price, _ = _mark_to_exit_price(sym, pos, cur_price, trial=candidate_trial, autopilot=autopilot)
            _d = pos.get("direction", "LONG")
            pos_pnl = position_unrealized_pnl(pos, cur_price)
            print(f"  {sym:<14} {_d:<8} {pos['qty']:>12.6f} ${pos['entry_price']:>9,.2f} ${cur_price:>9,.2f} ${pos_pnl:>+9,.2f}")
        print()
    else:
        print("  No open positions.\n")

    print(f"  Updated: {port.get('updated', 'never')}")
    print(f"  Created: {port.get('created', 'unknown')}")
    print("=" * 55)


def cmd_run():
    """Full cycle: scan → signals → check stops → execute → update."""
    run_lock = _RunLock(RUN_LOCK_FILE)
    if not run_lock.acquire():
        print("=== QuantForge Paper Trading — Full Cycle ===\n")
        print("Another QuantForge run is already active. Skipping this cycle.")
        return

    try:
        print("=== QuantForge Paper Trading — Full Cycle ===\n")

        port = load_portfolio()
        autopilot, runtime_warning = load_runtime_autopilot_report()
        candidate_trial = sync_candidate_trial_state(autopilot)
        autopilot_mode = str(autopilot.get("mode", "")).strip().lower()
        block_new_entries = autopilot_blocks_new_entries(autopilot)

        if runtime_warning:
            print(f"[warn] Autopilot control artifact degraded: {runtime_warning}")
            print("       New entries remain blocked, but exits/scans will continue so QuantForge can recover.")

        # 1) Check existing stop-loss/take-profit
        if port.get("positions"):
            print("[1/4] Checking stops on open positions...")
            stopped = check_stops(port, autopilot=autopilot, trial=candidate_trial)
            if stopped:
                for t in stopped:
                    print(f"  {t['trigger']}: {t['symbol']} PnL=${t['pnl']:+.2f}")
            else:
                print("  No stops triggered.")
        else:
            print("[1/4] No open positions to check.")

        # 2) Scan market
        print("\n[2/4] Scanning market...")
        signals = cmd_scan()

        # 3) Execute paper trades
        print("\n[3/4] Executing paper trades...")
        regime = _get_regime()
        if block_new_entries:
            reasons = autopilot.get("reasons", []) if isinstance(autopilot.get("reasons"), list) else []
            print(f"  Autopilot mode '{autopilot_mode or 'unknown'}' is active. Managing existing positions only.")
            for reason in reasons[:2]:
                print(f"   - {reason}")
            _save_idle_execution_report(
                autopilot=autopilot,
                trial=candidate_trial,
                signal_count=len(signals),
                execution_permission="blocked",
                idle_reason="autopilot_blocked_new_entries",
                details=reasons[:5],
            )
            executed = []
        elif signals:
            executed = execute_paper_trades(signals, port, autopilot=autopilot, trial=candidate_trial, regime=regime)
            if executed:
                for t in executed:
                    if t["type"] == "OPEN":
                        print(f"  OPENED {t['symbol']} qty={t['qty']:.6f} @ ${t['entry_price']:.4f} SL=${t['stop_loss']:.4f} TP=${t['take_profit']:.4f}")
                    else:
                        print(f"  CLOSED {t['symbol']} PnL=${t['pnl']:+.4f}")
            else:
                print("  No trades executed (positions full or signals not strong enough).")
                try:
                    last_execution = read_json_safe(LAST_EXECUTION_FILE) or {}
                    for reason, count in list((last_execution.get("skip_summary") or {}).items())[:3]:
                        print(f"   - {reason} ({count})")
                except Exception:
                    pass
        else:
            print("  No signals to trade.")
            _save_idle_execution_report(
                autopilot=autopilot,
                trial=candidate_trial,
                signal_count=0,
                execution_permission="allowed",
                idle_reason="no_actionable_signals",
            )

        # 4) Update equity and drawdown
        print("\n[4/4] Updating portfolio...")
        prices = {}
        try:
            for t in get_futures_tickers():
                base = t.get("symbol", "").replace("USDTM", "")
                if base == "XBT": base = "BTC"
                prices[f"{base}-USDT"] = float(t.get("price", t.get("lastTradePrice", 0)) or 0)
        except Exception:
            pass
        current_eq = equity(port, prices, autopilot=autopilot, trial=candidate_trial)
        update_drawdown(port, current_eq)
        save_portfolio(port)
        if AUTOTUNE_ENABLED:
            tune_result = _auto_tune_thresholds(port)
            if tune_result and tune_result.get("action") not in ("hold", None):
                print(f"  AUTO-TUNE: {tune_result['action']} — threshold → {tune_result.get('new_threshold', '?')}")
        finalized_trial = finalize_candidate_trial_cycle(autopilot)
        print(f"  Equity: ${current_eq:,.2f}  Drawdown: {port['max_drawdown'] * 100:.2f}%")
        if _candidate_trial_status(finalized_trial):
            print(
                f"  Candidate trial: {finalized_trial.get('status')} "
                f"({int(finalized_trial.get('cycles_run', 0) or 0)}/{int(finalized_trial.get('max_cycles', 0) or 0)} cycles)"
            )
        print("\nDone. Run 'quantforge_paper.py status' for full stats.")
    finally:
        run_lock.release()


def cmd_manage():
    """Lightweight risk loop: manage open positions without running a full market scan."""
    run_lock = _RunLock(RUN_LOCK_FILE)
    if not run_lock.acquire():
        print("=== QuantForge Paper Trading — Position Management ===\n")
        print("Another QuantForge run is already active. Skipping management cycle.")
        return

    try:
        print("=== QuantForge Paper Trading — Position Management ===\n")

        port = load_portfolio()
        autopilot, runtime_warning = load_runtime_autopilot_report()
        candidate_trial = sync_candidate_trial_state(autopilot)

        if runtime_warning:
            print(f"[warn] Autopilot control artifact degraded: {runtime_warning}")
            print("       Managing exits only until the review/autopilot loop refreshes.")

        if not port.get("positions"):
            print("No open positions to manage.")
            save_portfolio(port)
            return

        print(f"Managing {len(port['positions'])} open position(s)...")
        stopped = check_stops(port, autopilot=autopilot, trial=candidate_trial)
        if stopped:
            for trade in stopped:
                print(f"  {trade['trigger']}: {trade['symbol']} PnL=${trade['pnl']:+.2f}")
        else:
            print("  No exits triggered.")

        prices = {}
        try:
            for t in get_futures_tickers():
                base = t.get("symbol", "").replace("USDTM", "")
                if base == "XBT":
                    base = "BTC"
                prices[f"{base}-USDT"] = float(t.get("price", t.get("lastTradePrice", 0)) or 0)
        except Exception:
            pass

        current_eq = equity(port, prices, autopilot=autopilot, trial=candidate_trial)
        update_drawdown(port, current_eq)
        save_portfolio(port)
        print(f"  Equity: ${current_eq:,.2f}  Drawdown: {port['max_drawdown'] * 100:.2f}%")
    finally:
        run_lock.release()


def cmd_reconcile():
    """Repair the saved portfolio ledger from the recorded trade history."""
    print("=== QuantForge Paper Trading — Reconcile Ledger ===\n")
    port = rebuild_portfolio_from_trades()
    _reconcile_cash_ledger(port)

    prices = {}
    try:
        for t in get_futures_tickers():
            base = t.get("baseCurrency", t.get("symbol", "").replace("USDTM", ""))
            if base == "XBT":
                base = "BTC"
            prices[f"{base}-USDT"] = float(t.get("lastTradePrice", t.get("markPrice", 0)) or 0)
    except Exception:
        pass

    current_eq = equity(port, prices)
    update_drawdown(port, current_eq)
    save_portfolio(port)

    print(f"  Cash:    ${port['cash']:,.2f}")
    print(f"  Equity:  ${current_eq:,.2f}")
    print(f"  Trades:  {port['total_trades']}")
    print(f"  Open positions: {len(port['positions'])}")
    print("\nLedger rebuilt from trade history.")


# ---------------------------------------------------------------------------
# Backtesting gate
# ---------------------------------------------------------------------------

BACKTEST_PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT"]
BACKTEST_DAYS = 90
BACKTEST_MIN_TRADES = 100
BACKTEST_MIN_SHARPE = 1.0
BACKTEST_MIN_WIN_RATE = 55.0  # percent


def _fetch_historical_candles(symbol: str, days: int = 90, bar_type: str = "1hour") -> list[list]:
    """
    Fetch historical candles from KuCoin public API, oldest first.
    KuCoin returns max 1500 candles per request; we page backward to cover `days`.
    bar_type: "1day" | "1hour" | "4hour" etc.
    """
    bars_per_day = {"1day": 1, "4hour": 6, "1hour": 24, "30min": 48, "15min": 96}.get(bar_type, 24)
    bar_seconds = 86400 // bars_per_day

    end_ts = int(time.time())
    start_ts = end_ts - days * 86400

    all_candles: list[list] = []
    chunk_end = end_ts

    # Page through time in 1500-bar windows
    while chunk_end > start_ts:
        chunk_start = max(start_ts, chunk_end - 1500 * bar_seconds)
        try:
            chunk = kucoin_get("/api/v1/market/candles", {
                "symbol": symbol,
                "type": bar_type,
                "startAt": chunk_start,
                "endAt": chunk_end,
            })
        except Exception:
            chunk = []
        if not chunk:
            break
        # KuCoin returns newest first
        chunk.reverse()
        all_candles = chunk + all_candles
        chunk_end = chunk_start - bar_seconds
        time.sleep(0.2)

    # Deduplicate by timestamp, keep sorted oldest-first
    seen = set()
    unique = []
    for c in all_candles:
        ts = c[0]
        if ts not in seen:
            seen.add(ts)
            unique.append(c)
    unique.sort(key=lambda x: x[0])
    return unique


def _backtest_signal_on_bar(candles: list[list], i: int) -> str | None:
    """
    Run the same indicator logic as generate_signals() on a historical bar.
    Returns "BUY", "SELL", or None.  Requires at least 52 bars up to index i.
    """
    if i < 51:
        return None

    window = candles[: i + 1]
    closes = to_floats(window, 2)
    volumes = to_floats(window, 5)

    rsi_vals = rsi(closes)
    macd_line, macd_sig, macd_hist = macd(closes)
    sma50_vals = sma(closes, 50)
    sma20_vals = sma(closes, 20)
    vol_spikes = volume_spike(volumes)

    idx = len(closes) - 1
    cur_rsi = rsi_vals[idx]
    cur_hist = macd_hist[idx]
    prev_hist = macd_hist[idx - 1] if idx > 0 else None
    cur_sma50 = sma50_vals[idx]
    cur_sma20 = sma20_vals[idx]
    cur_vol_spike = vol_spikes[idx]

    if None in (cur_rsi, cur_hist, prev_hist, cur_sma50, cur_sma20):
        return None

    # RSI vote
    if cur_rsi < 30:
        rsi_vote = 1
    elif cur_rsi > 70:
        rsi_vote = -1
    else:
        rsi_vote = 0

    # MACD vote
    if prev_hist < 0 and cur_hist > 0:
        macd_vote = 1
    elif prev_hist > 0 and cur_hist < 0:
        macd_vote = -1
    elif cur_hist > 0:
        macd_vote = 1
    else:
        macd_vote = -1

    # SMA vote
    sma_vote = 1 if cur_sma20 > cur_sma50 else -1

    bullish = sum(1 for v in [rsi_vote, macd_vote, sma_vote] if v > 0)
    bearish = sum(1 for v in [rsi_vote, macd_vote, sma_vote] if v < 0)

    if bullish >= 2:
        return "BUY"
    elif bearish >= 2:
        return "SELL"
    return None


def backtest_strategy() -> dict:
    """
    Pull 90 days of daily OHLCV for each pair in BACKTEST_PAIRS, run the
    RSI/MACD/SMA signal logic bar-by-bar, simulate entry/exit with fees,
    and return a results dict with Sharpe, win_rate, trade count, etc.
    """
    import math

    all_returns: list[float] = []
    total_trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    pair_results = {}

    for symbol in BACKTEST_PAIRS:
        print(f"  Fetching {BACKTEST_DAYS}d hourly candles for {symbol}...")
        try:
            candles = _fetch_historical_candles(symbol, BACKTEST_DAYS, bar_type="1hour")
        except Exception as e:
            print(f"    ERROR fetching {symbol}: {e}")
            pair_results[symbol] = {"error": str(e)}
            continue

        if len(candles) < 52:
            print(f"    SKIP {symbol}: only {len(candles)} bars (need >=52)")
            pair_results[symbol] = {"skipped": f"only {len(candles)} bars"}
            continue

        print(f"    Got {len(candles)} bars.")
        time.sleep(0.3)  # be polite to KuCoin rate limits

        position = None   # dict with entry_price, entry_idx when open
        pair_trades = 0
        pair_wins = 0

        # Execution realism: signals are computed on bar i's CLOSE, so fills
        # happen at bar i+1's OPEN plus slippage+half-spread — never at the
        # same bar the signal was computed on. Loop stops at len-1 so i+1 is
        # always a real bar.
        _bt_slip = (max(0.0, PAPER_ENTRY_SLIPPAGE_BPS) + max(0.0, PAPER_SPREAD_BPS) / 2.0) / 10_000.0
        _bt_slip_exit = (max(0.0, PAPER_EXIT_SLIPPAGE_BPS) + max(0.0, PAPER_SPREAD_BPS) / 2.0) / 10_000.0

        for i in range(51, len(candles) - 1):
            close_price = float(candles[i][2])
            next_open = float(candles[i + 1][1])
            sig = _backtest_signal_on_bar(candles, i)

            # Close position
            if position is not None:
                should_close = False
                close_reason = None
                _update_trailing_exit(position, close_price, candles=candles[:i + 1])
                trailing_active = bool(position.get("trailing_active", False))

                if sig == "SELL":
                    should_close = True
                    close_reason = "signal"
                elif close_price <= position["stop_loss"]:
                    should_close = True
                    close_reason = "stop_loss"
                elif not trailing_active and close_price >= position["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"

                if should_close:
                    entry = position["entry_price"]
                    exit_fill = next_open * (1 - _bt_slip_exit)
                    # Round-trip fees: taker in + taker out
                    entry_cost = entry * (1 + TAKER_FEE)
                    exit_proceeds = exit_fill * (1 - TAKER_FEE)
                    ret_pct = (exit_proceeds - entry_cost) / entry_cost

                    all_returns.append(ret_pct)
                    total_trades += 1
                    pair_trades += 1

                    if ret_pct > 0:
                        wins += 1
                        pair_wins += 1
                        gross_profit += ret_pct
                    else:
                        gross_loss += abs(ret_pct)

                    position = None

            # Open position (only if flat) — fill at NEXT bar's open + slippage
            if position is None and sig == "BUY":
                entry_fill = next_open * (1 + _bt_slip)
                position = {
                    "entry_price": entry_fill,
                    "entry_idx": i + 1,
                    "stop_loss": entry_fill * (1 - STOP_LOSS_PCT),
                    "take_profit": entry_fill * (1 + TAKE_PROFIT_PCT),
                    "best_price": entry_fill,
                    "trailing_active": False,
                }

        # Force-close any open position at last bar (exit slippage still applies)
        if position is not None:
            last_close = float(candles[-1][2]) * (1 - _bt_slip_exit)
            entry = position["entry_price"]
            entry_cost = entry * (1 + TAKER_FEE)
            exit_proceeds = last_close * (1 - TAKER_FEE)
            ret_pct = (exit_proceeds - entry_cost) / entry_cost
            all_returns.append(ret_pct)
            total_trades += 1
            pair_trades += 1
            if ret_pct > 0:
                wins += 1
                pair_wins += 1
                gross_profit += ret_pct
            else:
                gross_loss += abs(ret_pct)

        pair_win_rate = (pair_wins / pair_trades * 100) if pair_trades > 0 else 0.0
        pair_results[symbol] = {
            "trades": pair_trades,
            "wins": pair_wins,
            "win_rate_pct": round(pair_win_rate, 2),
        }
        print(f"    {symbol}: {pair_trades} trades, win_rate={pair_win_rate:.1f}%")

    # --- Aggregate stats ---
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_pct = ((1 + sum(all_returns)) - 1) * 100  # cumulative compounded-style

    # Annualized Sharpe (per-trade returns, risk-free = 0).
    # Annualize by the ACTUAL trade frequency over the tested window, not by
    # bars-per-year — sqrt(8760) assumed a trade every hour and overstated
    # Sharpe by sqrt(8760/actual) (~10x at typical trade counts).
    sharpe = 0.0
    if len(all_returns) >= 2:
        n = len(all_returns)
        mean_r = sum(all_returns) / n
        variance = sum((r - mean_r) ** 2 for r in all_returns) / (n - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        trades_per_year = total_trades * (365.0 / max(BACKTEST_DAYS, 1))
        if std_r > 0 and trades_per_year > 0:
            sharpe = (mean_r / std_r) * math.sqrt(trades_per_year)

    gate_pass = sharpe >= BACKTEST_MIN_SHARPE and win_rate >= BACKTEST_MIN_WIN_RATE and total_trades >= BACKTEST_MIN_TRADES

    results = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "pairs": BACKTEST_PAIRS,
        "backtest_days": BACKTEST_DAYS,
        "total_trades": total_trades,
        "wins": wins,
        "losses": total_trades - wins,
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 4),
        "profit_pct": round(profit_pct, 4),
        "gross_profit_pct": round(gross_profit * 100, 4),
        "gross_loss_pct": round(gross_loss * 100, 4),
        "gate": {
            "pass": gate_pass,
            "min_sharpe": BACKTEST_MIN_SHARPE,
            "min_win_rate_pct": BACKTEST_MIN_WIN_RATE,
            "min_trades": BACKTEST_MIN_TRADES,
        },
        "pair_results": pair_results,
    }

    return results


def cmd_backtest():
    """Run backtesting gate and report results."""
    print("=" * 60)
    print("  QuantForge — Backtesting Gate")
    print(f"  Pairs : {', '.join(BACKTEST_PAIRS)}")
    print(f"  Period: {BACKTEST_DAYS} days of 1-hour bars")
    print(f"  Gate  : Sharpe > {BACKTEST_MIN_SHARPE}  AND  Win rate > {BACKTEST_MIN_WIN_RATE}%  AND  Trades >= {BACKTEST_MIN_TRADES}")
    print("=" * 60)
    print()

    results = backtest_strategy()

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total trades   : {results['total_trades']}")
    print(f"  Wins / Losses  : {results['wins']} / {results['losses']}")
    print(f"  Win rate       : {results['win_rate_pct']:.2f}%  (gate >= {BACKTEST_MIN_WIN_RATE}%)")
    print(f"  Sharpe ratio   : {results['sharpe_ratio']:.4f}  (gate >= {BACKTEST_MIN_SHARPE})")
    print(f"  Profit         : {results['profit_pct']:.2f}%")
    print()

    gate = results["gate"]
    if gate["pass"]:
        print("  GATE: *** PASS *** — Strategy meets live-trading criteria.")
    else:
        reasons = []
        if results["sharpe_ratio"] < BACKTEST_MIN_SHARPE:
            reasons.append(f"Sharpe {results['sharpe_ratio']:.4f} < {BACKTEST_MIN_SHARPE}")
        if results["win_rate_pct"] < BACKTEST_MIN_WIN_RATE:
            reasons.append(f"Win rate {results['win_rate_pct']:.2f}% < {BACKTEST_MIN_WIN_RATE}%")
        if results["total_trades"] < BACKTEST_MIN_TRADES:
            reasons.append(f"Trade count {results['total_trades']} < {BACKTEST_MIN_TRADES}")
        print(f"  GATE: FAIL — {'; '.join(reasons)}")

    print()

    # Save results
    os.makedirs(DATA_DIR, exist_ok=True)
    results_file = os.path.join(DATA_DIR, "backtest-results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_file}")
    print("=" * 60)


def cmd_repair():
    """Backward-compatible alias for the ledger reconcile command."""
    cmd_reconcile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cmd_reset_halt():
    """Clear the latched drawdown halt after human review. Kill files must be removed manually."""
    cleared = []
    if os.path.exists(DD_HALT_FILE):
        try:
            with open(DD_HALT_FILE) as f:
                print("Latched halt details:")
                print(f.read())
        except Exception:
            pass
        os.remove(DD_HALT_FILE)
        cleared.append("dd_halt.flag")
    for path, name in ((KILL_FILE, "KILL"), (KILL_FLATTEN_FILE, "KILL_FLATTEN")):
        if os.path.exists(path):
            print(f"NOTE: {name} kill file is present at {path} — remove it manually if intended.")
    if cleared:
        print(f"Cleared: {', '.join(cleared)}. New entries are allowed again next cycle.")
    else:
        print("No latched halt found. Nothing to clear.")


def main():
    cfg.require_production_runtime("quantforge_paper.py")
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "scan":
        cmd_scan()
    elif cmd == "status":
        cmd_status()
    elif cmd == "run":
        cmd_run()
    elif cmd == "manage":
        cmd_manage()
    elif cmd == "reconcile":
        cmd_reconcile()
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "repair":
        cmd_repair()
    elif cmd == "reset-halt":
        cmd_reset_halt()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
