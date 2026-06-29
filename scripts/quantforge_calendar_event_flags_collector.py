#!/usr/bin/env python3
"""QuantForge calendar and event flags collector.

Builds a lightweight event-overlap lane from internal alerts plus recurring
market timing windows that matter for crypto trading.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
PARQUET_FILE = os.path.join(EVENTS_DIR, "event_flags_latest.parquet")
HISTORY_FILE = os.path.join(EVENTS_DIR, "event_flags_history.jsonl")
REPORT_FILE = os.path.join(EVENTS_DIR, "event-overlap-report.json")
ALERT_SUMMARY_FILE = os.path.join(cfg.data, "alerts-summary.json")
ALERT_LOG_FILE = os.path.join(cfg.data, "alerts.log")
CYBER_LOG_FILE = os.path.join(cfg.data, "cybersecurity.log")
PRIMARY_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BCH-USDT", "TRX-USDT"]


def ensure_dirs() -> None:
    os.makedirs(EVENTS_DIR, exist_ok=True)


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tail_lines(path: str, count: int = 50) -> list[str]:
    try:
        with open(path) as f:
            return [line.strip() for line in f.readlines()[-count:] if line.strip()]
    except Exception:
        return []


def _hours_until(target: datetime, now: datetime) -> float:
    return (target - now).total_seconds() / 3600.0


def _build_clock_events(now: datetime) -> list[dict]:
    rows = []
    next_funding_hour = ((now.hour // 8) + 1) * 8
    funding_day = now
    if next_funding_hour >= 24:
        next_funding_hour -= 24
        funding_day = now + timedelta(days=1)
    next_funding = funding_day.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)
    next_utc_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    next_us_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
    if next_us_open <= now:
        next_us_open += timedelta(days=1)
    next_us_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
    if next_us_close <= now:
        next_us_close += timedelta(days=1)

    windows = [
        ("funding_window", "exchange", next_funding, 0.8),
        ("utc_daily_rollover", "market", next_utc_day, 0.4),
        ("us_session_open", "macro", next_us_open, 0.5),
        ("us_session_close", "macro", next_us_close, 0.35),
    ]
    for event_type, severity, target, score in windows:
        hours_to = _hours_until(target, now)
        rows.append(
            {
                "timestamp": int(now.timestamp()),
                "timestamp_iso": now.isoformat(),
                "event_type": event_type,
                "event_severity": severity,
                "event_score": score,
                "symbol_scope": "market",
                "hours_to_event": round(hours_to, 4),
                "event_active": 1 if abs(hours_to) <= 0.5 else 0,
                "source": "clock_rules",
            }
        )

    if now.weekday() >= 5:
        rows.append(
            {
                "timestamp": int(now.timestamp()),
                "timestamp_iso": now.isoformat(),
                "event_type": "weekend_liquidity",
                "event_severity": "market",
                "event_score": 0.45,
                "symbol_scope": "market",
                "hours_to_event": 0.0,
                "event_active": 1,
                "source": "clock_rules",
            }
        )
    return rows


def _build_incident_events(now: datetime) -> list[dict]:
    rows = []
    alert_summary = _read_json(ALERT_SUMMARY_FILE)
    active_alerts = alert_summary.get("active") or []
    recent_lines = []
    recent_lines.extend(_tail_lines(ALERT_LOG_FILE, 20))
    recent_lines.extend(_tail_lines(CYBER_LOG_FILE, 20))
    incident_active = bool(active_alerts) or any("alert" in line.lower() or "critical" in line.lower() for line in recent_lines)
    if incident_active:
        rows.append(
            {
                "timestamp": int(now.timestamp()),
                "timestamp_iso": now.isoformat(),
                "event_type": "internal_incident_overlap",
                "event_severity": "incident",
                "event_score": 0.8,
                "symbol_scope": "market",
                "hours_to_event": 0.0,
                "event_active": 1,
                "source": "internal_alerts",
            }
        )
    return rows


def _build_symbol_rows(rows: list[dict]) -> list[dict]:
    expanded = []
    for row in rows:
        scope = row.get("symbol_scope")
        if scope == "market":
            for symbol in PRIMARY_SYMBOLS:
                item = dict(row)
                item["symbol_scope"] = symbol
                expanded.append(item)
        else:
            expanded.append(row)
    return expanded


def _build_report(rows: list[dict]) -> dict:
    active = [row for row in rows if int(row.get("event_active", 0) or 0) > 0]
    upcoming = sorted(rows, key=lambda row: abs(float(row.get("hours_to_event", 999) or 999)))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if rows else "missing",
        "row_count": len(rows),
        "active_count": len(active),
        "active_events": [
            {
                "event_type": row.get("event_type"),
                "event_severity": row.get("event_severity"),
                "symbol_scope": row.get("symbol_scope"),
                "event_score": row.get("event_score"),
            }
            for row in active[:8]
        ],
        "upcoming_events": [
            {
                "event_type": row.get("event_type"),
                "event_severity": row.get("event_severity"),
                "symbol_scope": row.get("symbol_scope"),
                "hours_to_event": row.get("hours_to_event"),
            }
            for row in upcoming[:8]
        ],
    }


def main() -> None:
    ensure_dirs()
    now = datetime.now(timezone.utc)
    base_rows = _build_clock_events(now) + _build_incident_events(now)
    rows = _build_symbol_rows(base_rows)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(PARQUET_FILE, index=False)
        with open(HISTORY_FILE, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    report = _build_report(rows)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("QuantForge calendar/event collector")
    print(f"Status: {report['status']}")
    print(f"Rows:   {report['row_count']}")
    print(f"Saved:  {PARQUET_FILE}")


if __name__ == "__main__":
    main()
