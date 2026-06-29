#!/usr/bin/env python3
"""QuantForge — monitoring and drift snapshot.

Produces a compact monitoring artifact for the current paper-trading state.
This works without optional observability packages, and records whether richer
tooling like Evidently is available at runtime.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

try:
    import evidently  # type: ignore
except Exception:
    evidently = None


BASE_DIR = os.path.join(cfg.data, "quantforge")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
TRADES_FILE = os.path.join(BASE_DIR, "paper-trades.jsonl")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
DIAGNOSIS_FILE = os.path.join(BASE_DIR, "diagnosis-report.json")
MONITOR_FILE = os.path.join(BASE_DIR, "monitor-report.json")
MAX_PORTFOLIO_AGE_HOURS = 8
MAX_LAST_SCAN_AGE_HOURS = 8

MAJOR_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BCH-USDT", "TRX-USDT"}
TOP_LIQUIDITY_EXPANSION_SYMBOLS = MAJOR_SYMBOLS | {
    "ADA-USDT",
    "DOGE-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "LTC-USDT",
    "XMR-USDT",
    "TAO-USDT",
}


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def _f(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _parse_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _age_hours(value):
    dt = _parse_ts(value)
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def _summarize_rows(rows):
    pnls = [_f(row.get("pnl")) for row in rows]
    wins = sum(1 for pnl in pnls if pnl > 0)
    signal_scores = [_f(row.get("signal_score")) for row in rows if row.get("signal_score") is not None]
    quality_scores = [_f(row.get("quality_score")) for row in rows if row.get("quality_score") is not None]
    return {
        "count": len(rows),
        "win_rate": round(wins / max(len(rows), 1), 4) if rows else 0.0,
        "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 4) if pnls else 0.0,
        "avg_signal_score": round(sum(signal_scores) / max(len(signal_scores), 1), 4) if signal_scores else None,
        "avg_quality_score": round(sum(quality_scores) / max(len(quality_scores), 1), 4) if quality_scores else None,
    }


def build_report():
    now = datetime.now(timezone.utc)
    portfolio = read_json(PORTFOLIO_FILE)
    trades = read_jsonl(TRADES_FILE)
    last_scan = read_json(LAST_SCAN_FILE)
    governance = read_json(GOVERNANCE_FILE)
    diagnosis = read_json(DIAGNOSIS_FILE)

    recent_cutoff = now - timedelta(hours=72)
    prior_cutoff = now - timedelta(hours=144)
    closes = [row for row in trades if row.get("type") == "CLOSE"]
    recent_closes = []
    prior_closes = []
    for row in closes:
        ts = _parse_ts(row.get("ts"))
        if not ts:
            continue
        if ts >= recent_cutoff:
            recent_closes.append(row)
        elif prior_cutoff <= ts < recent_cutoff:
            prior_closes.append(row)

    recent_summary = _summarize_rows(recent_closes)
    prior_summary = _summarize_rows(prior_closes)
    feedback_summary = (last_scan.get("feedback") or {}).get("summary", {})
    regime = last_scan.get("regime") or {}
    flow = last_scan.get("flow") or {}
    open_positions = portfolio.get("positions") or {}

    blocked_counts = defaultdict(int)
    for row in last_scan.get("results", []):
        if row.get("status") in {"skip", "hold"}:
            blocked_counts[str(row.get("reason", "unknown"))] += 1

    drift_flags = []
    if _f(governance.get("paper", {}).get("total_pnl_pct")) <= -5.0:
        drift_flags.append("paper_drawdown")
    if recent_summary["count"] >= 4 and recent_summary["win_rate"] < 0.35:
        drift_flags.append("recent_close_quality")
    if recent_summary["count"] >= 4 and recent_summary["avg_pnl"] < 0:
        drift_flags.append("negative_recent_expectancy")
    if prior_summary["count"] >= 3 and recent_summary["count"] >= 3:
        if recent_summary["win_rate"] + 0.15 < prior_summary["win_rate"]:
            drift_flags.append("win_rate_drift")
        if recent_summary["avg_pnl"] + 2.0 < prior_summary["avg_pnl"]:
            drift_flags.append("pnl_drift")
    if _f(feedback_summary.get("risk_mult"), 1.0) < 1.0:
        drift_flags.append("risk_throttle_active")
    if len(last_scan.get("signals", [])) == 0:
        drift_flags.append("low_signal_activity")
    if int(flow.get("model_no_signal", 0) or 0) > 0:
        drift_flags.append("model_no_signal_bottleneck")
    if int(flow.get("threshold_miss", 0) or 0) > 0:
        drift_flags.append("threshold_miss_bottleneck")
    if str(regime.get("entropy_label", "")).upper() == "CHAOTIC":
        drift_flags.append("high_entropy_regime")

    non_major_positions = []
    for sym, pos in open_positions.items():
        if sym not in TOP_LIQUIDITY_EXPANSION_SYMBOLS:
            non_major_positions.append(
                {
                    "symbol": sym,
                    "setup_tag": pos.get("setup_tag"),
                    "quality_score": _f(pos.get("quality_score")),
                    "signal_score": _f(pos.get("signal_score")),
                }
            )
    if non_major_positions:
        drift_flags.append("non_major_exposure")
    stale_inputs = []
    portfolio_age = _age_hours(portfolio.get("updated"))
    last_scan_age = _age_hours(last_scan.get("ts"))
    if portfolio_age is None or portfolio_age > MAX_PORTFOLIO_AGE_HOURS:
        drift_flags.append("stale_portfolio")
        stale_inputs.append(f"portfolio age {portfolio_age:.1f}h" if portfolio_age is not None else "portfolio missing updated")
    if last_scan_age is None or last_scan_age > MAX_LAST_SCAN_AGE_HOURS:
        drift_flags.append("stale_last_scan")
        stale_inputs.append(f"last scan age {last_scan_age:.1f}h" if last_scan_age is not None else "last scan missing ts")

    if stale_inputs:
        health = "STALLED"
    elif drift_flags:
        health = "DRIFTING"
    elif governance.get("recommendation") in {"REVIEW", "DEMOTE"}:
        health = "WATCH"
    else:
        health = "STABLE"

    report = {
        "generated_at": now.isoformat(),
        "health": health,
        "tooling": {
            "evidently_available": evidently is not None,
            "evidently_used": False,
        },
        "summary": {
            "governance_recommendation": governance.get("recommendation"),
            "paper_total_pnl_pct": round(_f(governance.get("paper", {}).get("total_pnl_pct")), 4),
            "recent_close_count": recent_summary["count"],
            "recent_close_win_rate": recent_summary["win_rate"],
            "recent_close_avg_pnl": recent_summary["avg_pnl"],
            "prior_close_count": prior_summary["count"],
            "prior_close_win_rate": prior_summary["win_rate"],
            "prior_close_avg_pnl": prior_summary["avg_pnl"],
            "signal_count": len(last_scan.get("signals", [])),
            "model_no_signal": int(flow.get("model_no_signal", 0) or 0),
            "threshold_miss": int(flow.get("threshold_miss", 0) or 0),
            "open_positions": len(open_positions),
            "adaptive_risk_mult": round(_f(feedback_summary.get("risk_mult"), 1.0), 4),
            "diagnosis_causes": diagnosis.get("causes", []),
            "regime_label": regime.get("label"),
            "regime_score": round(_f(regime.get("score")), 4),
            "regime_entropy": round(_f(regime.get("entropy")), 4),
            "regime_entropy_label": regime.get("entropy_label"),
        },
        "regime": regime,
        "stale_inputs": stale_inputs,
        "drift_flags": sorted(set(drift_flags)),
        "blocked_reasons": dict(sorted(blocked_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "non_major_positions": non_major_positions,
        "recent_setup_pnl": sorted(
            [
                {
                    "setup_tag": setup,
                    "count": stats["count"],
                    "avg_pnl": round(stats["pnl"] / max(stats["count"], 1), 4),
                    "total_pnl": round(stats["pnl"], 4),
                }
                for setup, stats in _aggregate_by_setup(recent_closes).items()
            ],
            key=lambda row: (row["total_pnl"], row["count"]),
        )[:8],
    }
    return report


def _aggregate_by_setup(rows):
    by_setup = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for row in rows:
        setup = row.get("setup_tag") or "unlabeled"
        by_setup[setup]["count"] += 1
        by_setup[setup]["pnl"] += _f(row.get("pnl"))
    return by_setup


def main():
    cfg.require_production_runtime("quantforge_monitor.py")
    report = build_report()
    with open(MONITOR_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print("QuantForge monitor report")
    print(f"Health: {report['health']}")
    print(f"Drift flags: {', '.join(report['drift_flags']) if report['drift_flags'] else 'none'}")
    print(f"Saved: {MONITOR_FILE}")


if __name__ == "__main__":
    main()
