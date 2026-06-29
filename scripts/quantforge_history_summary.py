#!/usr/bin/env python3
"""QuantForge — durable history summary from Supabase operator cycles.

Reads recent operator history from the configured Supabase project and writes
a compact local artifact that governance/diagnosis/autopilot can consume without
putting remote dependencies in the trading hot path.
"""

import datetime
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTPUT_FILE = os.path.join(BASE_DIR, "history-summary.json")
LOOKBACK_LIMIT = 96


def _f(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def write_summary(payload):
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def fetch_cycles(limit=LOOKBACK_LIMIT):
    if not cfg.supabase_url or not cfg.supabase_service_key:
        return None, "missing_credentials"

    query = urllib.parse.urlencode(
        {
            "select": "cycle_timestamp,quantforge_autopilot_mode,quantforge_governance,quantforge_monitor_health,payload",
            "agent_key": f"eq.{os.environ.get('QF_SUPABASE_AGENT_KEY', 'quantforge')}",
            "order": "cycle_timestamp.desc",
            "limit": str(limit),
        }
    )
    table = os.environ.get("QF_SUPABASE_TABLE", "quantforge_operator_cycles")
    url = f"{cfg.supabase_url}/rest/v1/{table}?{query}"
    headers = {
        "apikey": cfg.supabase_service_key,
        "Authorization": f"Bearer {cfg.supabase_service_key}",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else [], None
    except Exception as exc:
        return None, str(exc)


def _classify_posture(avg_pnl_pct, avg_recent_pnl, review_ratio, pause_ratio, drifting_ratio):
    if pause_ratio >= 0.5 or review_ratio >= 0.5 or avg_pnl_pct <= -4.0:
        return "degraded"
    if drifting_ratio >= 0.35 or avg_recent_pnl < 0:
        return "recovery_watch"
    if avg_pnl_pct > 0 and avg_recent_pnl > 0 and review_ratio <= 0.2:
        return "supportive"
    return "observe"


def build_summary():
    rows, error = fetch_cycles()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if rows is None:
        return {
            "generated_at": now,
            "status": "error" if error != "missing_credentials" else "missing_credentials",
            "error": error,
            "cycles_sampled": 0,
            "posture": "unknown",
        }

    autopilot_counts = Counter()
    governance_counts = Counter()
    promotion_counts = Counter()
    diagnosis_counts = Counter()
    monitor_counts = Counter()

    pnl_values = []
    recent_win_rates = []
    recent_avg_pnls = []

    for row in rows:
        autopilot_counts[str(row.get("quantforge_autopilot_mode") or "unknown")] += 1
        governance_counts[str(row.get("quantforge_governance") or "unknown")] += 1
        monitor_counts[str(row.get("quantforge_monitor_health") or "unknown")] += 1
        payload = row.get("payload") or {}
        promotion = payload.get("quantforge_promotion") or {}
        diagnosis = payload.get("quantforge_diagnosis") or {}
        gov_payload = payload.get("quantforge_governance") or {}
        paper = gov_payload.get("paper") or {}
        recent_closes = gov_payload.get("recent_closes") or {}

        promotion_counts[str(promotion.get("overall_decision") or "unknown")] += 1
        for cause in diagnosis.get("causes") or []:
            diagnosis_counts[str(cause)] += 1

        if paper.get("total_pnl_pct") is not None:
            pnl_values.append(_f(paper.get("total_pnl_pct")))
        if recent_closes.get("win_rate") is not None:
            recent_win_rates.append(_f(recent_closes.get("win_rate")))
        if recent_closes.get("avg_pnl") is not None:
            recent_avg_pnls.append(_f(recent_closes.get("avg_pnl")))

    count = len(rows)
    pause_ratio = autopilot_counts.get("pause_new_entries", 0) / max(count, 1)
    review_ratio = (
        governance_counts.get("REVIEW", 0) + governance_counts.get("DEMOTE", 0)
    ) / max(count, 1)
    drifting_ratio = monitor_counts.get("DRIFTING", 0) / max(count, 1)
    avg_pnl_pct = sum(pnl_values) / max(len(pnl_values), 1) if pnl_values else 0.0
    avg_recent_win_rate = sum(recent_win_rates) / max(len(recent_win_rates), 1) if recent_win_rates else 0.0
    avg_recent_pnl = sum(recent_avg_pnls) / max(len(recent_avg_pnls), 1) if recent_avg_pnls else 0.0
    posture = _classify_posture(avg_pnl_pct, avg_recent_pnl, review_ratio, pause_ratio, drifting_ratio)

    return {
        "generated_at": now,
        "status": "ok",
        "cycles_sampled": count,
        "window": {
            "latest": rows[0].get("cycle_timestamp") if rows else None,
            "oldest": rows[-1].get("cycle_timestamp") if rows else None,
        },
        "counts": {
            "autopilot_modes": dict(autopilot_counts),
            "governance": dict(governance_counts),
            "promotion": dict(promotion_counts),
            "monitor_health": dict(monitor_counts),
            "diagnosis_causes": dict(diagnosis_counts.most_common(10)),
        },
        "averages": {
            "paper_total_pnl_pct": round(avg_pnl_pct, 4),
            "recent_close_win_rate": round(avg_recent_win_rate, 4),
            "recent_close_avg_pnl": round(avg_recent_pnl, 4),
        },
        "ratios": {
            "pause_new_entries": round(pause_ratio, 4),
            "review_or_demote": round(review_ratio, 4),
            "drifting": round(drifting_ratio, 4),
        },
        "posture": posture,
    }


def main():
    summary = build_summary()
    write_summary(summary)
    print("QuantForge durable history summary")
    print(f"Status: {summary['status']}")
    print(f"Posture: {summary.get('posture', 'unknown')}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
