#!/usr/bin/env python3
"""Minimal live health shim for QuantForge agent self-heal hooks.

The heavier historical self-heal module is absent from this checkout, but the
agent still imports `quantforge_self_heal` each cycle. This shim restores the
health artifact and surfaces critical lane failures without silently crashing
the import path. It is intentionally conservative: it reports and persists
health, but does not mutate params or trading state on its own.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
HEALTH_FILE = os.path.join(DATA_DIR, "health.json")
WATCHDOG_HEALTH_FILE = os.path.join(DATA_DIR, "watchdog_health.json")
INVARIANTS_STATE_FILE = os.path.join(DATA_DIR, "qf_invariants_state.json")


@dataclass
class HealthReport:
    status: str = "healthy"
    alerts: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    recovery_level: int = 0
    recovery_reason: str = ""


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def check_health(*, portfolio, trades, regime_history, equity, peak_equity, current_regime):
    now = datetime.now(timezone.utc).isoformat()
    alerts: list[str] = []
    status = "healthy"

    inv_state = _read_json(INVARIANTS_STATE_FILE)
    n_critical = int(inv_state.get("n_critical", 0) or 0)
    n_warning = int(inv_state.get("n_warning", 0) or 0)
    if n_critical > 0:
        status = "critical"
        names = ", ".join(v.get("name", "?") for v in (inv_state.get("violations") or [])[:3])
        alerts.append(f"CRITICAL invariants active: {names or 'unknown'}")
    elif n_warning > 0:
        status = "degraded"
        alerts.append(f"WARNING invariants active: {n_warning}")

    if portfolio.get("futures_kill"):
        status = "critical"
        alerts.append(f"CRITICAL futures_kill engaged ({portfolio.get('futures_kill_reason', 'unknown')})")
    if portfolio.get("panic_halted"):
        status = "critical"
        alerts.append(f"CRITICAL panic_halted ({portfolio.get('panic_halt_reason', 'unknown')})")

    dd_pct = 0.0
    if peak_equity:
        try:
            dd_pct = max(0.0, (float(peak_equity) - float(equity)) / float(peak_equity))
        except Exception:
            dd_pct = 0.0
    if status == "healthy" and dd_pct >= 0.08:
        status = "degraded"
        alerts.append(f"WARNING drawdown elevated at {dd_pct:.1%}")

    return HealthReport(
        status=status,
        alerts=alerts,
        metrics={
            "checked_at": now,
            "current_regime": current_regime,
            "equity": round(float(equity or 0.0), 2),
            "peak_equity": round(float(peak_equity or 0.0), 2),
            "drawdown_pct": round(dd_pct * 100.0, 2),
            "recent_trade_count": len(trades or []),
            "regime_history_len": len(regime_history or []),
            "invariant_critical_count": n_critical,
            "invariant_warning_count": n_warning,
            "futures_kill": bool(portfolio.get("futures_kill")),
            "panic_halted": bool(portfolio.get("panic_halted")),
        },
        recovery_level=0,
        recovery_reason="",
    )


def apply_recovery(level, reason, port, params_file):
    return {
        "applied": False,
        "reason": reason,
        "level": level,
        "params_file": params_file,
    }


def _save_health(payload):
    os.makedirs(DATA_DIR, exist_ok=True)
    for path in (HEALTH_FILE, WATCHDOG_HEALTH_FILE):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
