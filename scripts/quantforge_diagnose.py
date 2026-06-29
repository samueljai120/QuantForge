#!/usr/bin/env python3
"""QuantForge — paper-trading diagnosis report.

Turns the current paper portfolio, trade history, scan feedback, and governance
signals into a compact diagnosis artifact for downstream automation.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
TRADES_FILE = os.path.join(BASE_DIR, "paper-trades.jsonl")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")
GOVERNANCE_FILE = os.path.join(BASE_DIR, "governance-report.json")
PROMOTION_FILE = os.path.join(BASE_DIR, "model", "promotion_report.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history-summary.json")
REPORT_FILE = os.path.join(BASE_DIR, "diagnosis-report.json")


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
                if line:
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


def build_report():
    portfolio = read_json(PORTFOLIO_FILE)
    trades = read_jsonl(TRADES_FILE)
    last_scan = read_json(LAST_SCAN_FILE)
    governance = read_json(GOVERNANCE_FILE)
    promotion = read_json(PROMOTION_FILE)
    history = read_json(HISTORY_FILE)

    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=72)
    recent_closes = []
    for row in trades:
        if row.get("type") != "CLOSE":
            continue
        try:
            ts = datetime.fromisoformat(str(row.get("ts")).replace("Z", "+00:00"))
        except Exception:
            continue
        if ts >= recent_cutoff:
            recent_closes.append(row)

    causes = []
    actions = []
    evidence = {}

    total_pnl_pct = _f(governance.get("paper", {}).get("total_pnl_pct"), 0.0)
    recent_win_rate = _f(governance.get("recent_closes", {}).get("win_rate"), 0.0)
    recent_avg_pnl = _f(governance.get("recent_closes", {}).get("avg_pnl"), 0.0)
    feedback = last_scan.get("feedback", {}).get("summary", {})
    regime = last_scan.get("regime", {}) or {}
    flow = last_scan.get("flow", {}) or {}
    risk_mult = _f(feedback.get("risk_mult"), 1.0)

    if total_pnl_pct <= -5.0:
        causes.append("paper_underperformance")
        actions.append("keep_in_review_mode")
    if recent_win_rate < 0.35 and len(recent_closes) >= 4:
        causes.append("weak_recent_close_quality")
        actions.append("tighten_entry_selection")
    if recent_avg_pnl < 0:
        causes.append("negative_recent_trade_expectancy")
        actions.append("keep_adaptive_risk_reduced")
    if risk_mult < 1.0:
        causes.append("risk_throttle_active")
    if promotion.get("overall_decision") == "DO_NOT_PROMOTE":
        causes.append("model_not_promotion_ready")
        actions.append("retrain_before_promotion")
    if str(regime.get("entropy_label", "")).upper() == "CHAOTIC":
        causes.append("chaotic_market_regime")
        actions.append("tighten_regime_filtering")
    if history.get("status") == "ok" and int(history.get("cycles_sampled", 0) or 0) >= 6:
        posture = history.get("posture")
        if posture == "degraded":
            causes.append("persistent_underperformance_across_cycles")
            actions.append("keep_autopilot_restricted")
        if posture in {"degraded", "recovery_watch"}:
            actions.append("prefer_major_symbols_only")

    by_symbol = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    by_setup = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for row in recent_closes:
        symbol = row.get("symbol", "unknown")
        setup = row.get("setup_tag") or "unlabeled"
        pnl = _f(row.get("pnl"))
        by_symbol[symbol]["count"] += 1
        by_symbol[symbol]["pnl"] += pnl
        by_symbol[symbol]["wins"] += 1 if pnl > 0 else 0
        by_setup[setup]["count"] += 1
        by_setup[setup]["pnl"] += pnl
        by_setup[setup]["wins"] += 1 if pnl > 0 else 0

    worst_symbols = sorted(
        [{"symbol": k, **v} for k, v in by_symbol.items()],
        key=lambda x: (x["pnl"], -x["count"])
    )[:5]
    worst_setups = sorted(
        [{"setup": k, **v} for k, v in by_setup.items()],
        key=lambda x: (x["pnl"], -x["count"])
    )[:5]

    blocked = [row for row in last_scan.get("results", []) if row.get("status") == "skip"]
    holds = [row for row in last_scan.get("results", []) if row.get("status") == "hold"]
    blocked_summary = defaultdict(int)
    for row in blocked:
        reason = str(row.get("reason", "unknown"))
        blocked_summary[reason] += 1
    hold_summary = defaultdict(int)
    hold_stage_summary = defaultdict(int)
    for row in holds:
        reason = str(row.get("reason", "unknown"))
        hold_summary[reason] += 1
        hold_stage_summary[str(row.get("decision_stage", "model_no_signal"))] += 1

    if blocked_summary:
        causes.append("selection_filter_active")
        evidence["blocked_reasons"] = dict(sorted(blocked_summary.items(), key=lambda kv: kv[1], reverse=True))
    if holds:
        causes.append("model_no_signal_bottleneck")
        evidence["hold_reasons"] = dict(sorted(hold_summary.items(), key=lambda kv: kv[1], reverse=True))
        evidence["hold_stages"] = dict(sorted(hold_stage_summary.items(), key=lambda kv: kv[1], reverse=True))
    if int(flow.get("threshold_miss", 0) or 0) > 0:
        causes.append("threshold_miss_bottleneck")
    if int(flow.get("trained_pair_blocked", 0) or 0) > 0:
        causes.append("trained_pair_restriction")

    report = {
        "generated_at": now.isoformat(),
        "summary": {
            "paper_total_pnl_pct": round(total_pnl_pct, 4),
            "recent_close_win_rate": round(recent_win_rate, 4),
            "recent_close_avg_pnl": round(recent_avg_pnl, 4),
            "adaptive_risk_mult": round(risk_mult, 4),
            "model_no_signal": int(flow.get("model_no_signal", 0) or 0),
            "threshold_miss": int(flow.get("threshold_miss", 0) or 0),
            "governance_recommendation": governance.get("recommendation"),
            "promotion_decision": promotion.get("overall_decision"),
            "durable_history_posture": history.get("posture"),
            "durable_cycles_sampled": int(history.get("cycles_sampled", 0) or 0),
            "regime_label": regime.get("label"),
            "regime_entropy_label": regime.get("entropy_label"),
            "regime_entropy": round(_f(regime.get("entropy")), 4),
        },
        "causes": sorted(set(causes)),
        "recommended_actions": sorted(set(actions)),
        "worst_symbols": worst_symbols,
        "worst_setups": worst_setups,
        "open_positions": [
            {
                "symbol": sym,
                "direction": pos.get("direction", "LONG"),
                "signal_score": _f(pos.get("signal_score")),
                "setup_tag": pos.get("setup_tag"),
                "quality_score": _f(pos.get("quality_score")),
            }
            for sym, pos in (portfolio.get("positions") or {}).items()
        ],
        "evidence": evidence,
        "durable_history": {
            "status": history.get("status"),
            "posture": history.get("posture"),
            "averages": history.get("averages", {}),
            "ratios": history.get("ratios", {}),
            "common_causes": (history.get("counts") or {}).get("diagnosis_causes", {}),
        },
    }
    return report


def main():
    report = build_report()
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print("QuantForge diagnosis report")
    print(f"Summary: {report['summary']}")
    print(f"Causes: {', '.join(report['causes']) if report['causes'] else 'none'}")
    print(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
