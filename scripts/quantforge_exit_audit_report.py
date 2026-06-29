#!/usr/bin/env python3
"""QuantForge exit audit report.

Summarizes how exits behaved recently so we can distinguish:
- good trailing / partial-profit capture
- stop losses that were too loose
- trades that never moved enough to justify a looser stop

The report groups trades by position lifecycle, not only by symbol, so repeated
same-day re-entries are audited separately.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
TRADES_FILE = os.path.join(BASE_DIR, "paper-trades.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "exit-audit-report.json")


def read_trades() -> list[dict]:
    rows = []
    try:
        with open(TRADES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> None:
    cfg.require_production_runtime("quantforge_exit_audit_report.py")
    trades = read_trades()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

    recent = [t for t in trades if (_parse_ts(t.get("ts")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    closed_rows = []
    summary = {
        "closed_trades": 0,
        "trailing_stop_closes": 0,
        "initial_stop_loss_closes": 0,
        "stop_loss_closes": 0,
        "take_profit_closes": 0,
        "time_stop_closes": 0,
        "gross_realized_recent": 0.0,
        "partial_profit_realized": 0.0,
        "largest_loss": None,
        "largest_win": None,
    }

    open_positions = {}
    position_counter = defaultdict(int)
    for row in sorted(recent, key=lambda r: str(r.get("ts") or "")):
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        ttype = str(row.get("type") or "")
        if ttype == "OPEN":
            position_counter[symbol] += 1
            position_id = f"{symbol}#{position_counter[symbol]}"
            open_positions[symbol] = {
                "position_id": position_id,
                "symbol": symbol,
                "open_ts": row.get("ts"),
                "setup_tag": row.get("setup_tag"),
                "setup_score": row.get("setup_score"),
                "quality_score": row.get("quality_score"),
                "partials": [],
                "partial_pnl": 0.0,
            }
            continue

        active = open_positions.get(symbol)
        if not active:
            continue

        if ttype == "PARTIAL_CLOSE":
            pnl = float(row.get("pnl", 0.0) or 0.0)
            active["partial_pnl"] += pnl
            active["partials"].append({
                "trigger": row.get("trigger"),
                "pnl": round(pnl, 4),
                "remaining_qty": row.get("remaining_qty"),
                "stop_stage": row.get("stop_stage"),
            })
            continue

        if ttype != "CLOSE":
            continue

        pnl = float(row.get("pnl", 0.0) or 0.0)
        trigger = str(row.get("trigger") or "")
        highest_unrealized_pct = float(row.get("highest_unrealized_pct", 0.0) or 0.0)
        highest_r = float(row.get("highest_r", 0.0) or 0.0)
        classification = "neutral"
        if trigger == "TRAILING_STOP":
            classification = "captured_winner"
        elif trigger == "INITIAL_STOP_LOSS":
            classification = "never_worked"
        elif trigger == "STOP_LOSS" and highest_r >= 1.0:
            classification = "gave_back_winner"
        elif trigger == "STOP_LOSS":
            classification = "stopped_before_edge"
        elif trigger == "TIME_STOP":
            classification = "stale_exit"

        partial_pnl = float(active.get("partial_pnl", 0.0) or 0.0)
        audit_row = {
            "position_id": active["position_id"],
            "symbol": symbol,
            "trigger": trigger,
            "classification": classification,
            "pnl": round(pnl, 4),
            "partial_pnl_before_close": round(partial_pnl, 4),
            "net_pnl_with_partials": round(partial_pnl + pnl, 4),
            "highest_unrealized_pct": round(highest_unrealized_pct, 6),
            "highest_r": round(highest_r, 6),
            "setup_tag": row.get("setup_tag") or active.get("setup_tag"),
            "setup_score": row.get("setup_score") or active.get("setup_score"),
            "quality_score": row.get("quality_score") if row.get("quality_score") is not None else active.get("quality_score"),
            "stop_stage": row.get("stop_stage"),
            "partial_events": active.get("partials", []),
            "opened_at": active.get("open_ts"),
            "closed_at": row.get("ts"),
        }
        closed_rows.append(audit_row)

        summary["closed_trades"] += 1
        summary["gross_realized_recent"] += pnl
        summary["partial_profit_realized"] += partial_pnl
        if trigger == "TRAILING_STOP":
            summary["trailing_stop_closes"] += 1
        elif trigger == "INITIAL_STOP_LOSS":
            summary["initial_stop_loss_closes"] += 1
        elif trigger == "STOP_LOSS":
            summary["stop_loss_closes"] += 1
        elif trigger == "TAKE_PROFIT":
            summary["take_profit_closes"] += 1
        elif trigger == "TIME_STOP":
            summary["time_stop_closes"] += 1

        marker = {"symbol": symbol, "pnl": round(pnl, 4), "trigger": trigger}
        if summary["largest_loss"] is None or pnl < float(summary["largest_loss"]["pnl"]):
            summary["largest_loss"] = marker
        if summary["largest_win"] is None or pnl > float(summary["largest_win"]["pnl"]):
            summary["largest_win"] = marker
        open_positions.pop(symbol, None)

    closed_rows.sort(key=lambda r: str(r.get("closed_at") or ""), reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": 48,
        "summary": {
            **summary,
            "gross_realized_recent": round(summary["gross_realized_recent"], 4),
            "partial_profit_realized": round(summary["partial_profit_realized"], 4),
        },
        "rows": closed_rows[:20],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("QuantForge exit audit report")
    print(f"Closed trades audited: {payload['summary']['closed_trades']}")
    print(f"Saved:                {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
