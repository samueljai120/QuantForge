#!/usr/bin/env python3
"""QuantForge — evaluation and governance snapshot.

Produces a compact report that helps decide whether the current model/paper
deployment should be observed, held, reviewed, or demoted.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_paper import get_futures_tickers, equity

BASE_DIR = os.path.join(cfg.data, "quantforge")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
TRADES_FILE = os.path.join(BASE_DIR, "paper-trades.jsonl")
MODEL_META_FILE = os.path.join(BASE_DIR, "model", "model_meta.json")
MODEL_META_SHORT_FILE = os.path.join(BASE_DIR, "model", "model_meta_short.json")
LAST_SCAN_FILE = os.path.join(BASE_DIR, "last_scan.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history-summary.json")
REPORT_FILE = os.path.join(BASE_DIR, "governance-report.json")
MAX_PORTFOLIO_AGE_HOURS = 8
MAX_LAST_SCAN_AGE_HOURS = 8


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


def _parse_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _age_hours(value) -> float | None:
    dt = _parse_ts(value)
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def _live_prices() -> dict[str, float]:
    prices = {}
    try:
        for c in get_futures_tickers():
            base = c.get("baseCurrency", c.get("symbol", "").replace("USDTM", ""))
            if base == "XBT":
                base = "BTC"
            prices[f"{base}-USDT"] = float(c.get("lastTradePrice", c.get("markPrice", 0)) or 0)
    except Exception:
        return {}
    return prices


def build_report():
    portfolio = read_json(PORTFOLIO_FILE)
    model_meta = read_json(MODEL_META_FILE)
    short_meta = read_json(MODEL_META_SHORT_FILE)
    last_scan = read_json(LAST_SCAN_FILE)
    history = read_json(HISTORY_FILE)
    trades = read_jsonl(TRADES_FILE)

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

    wins = sum(1 for row in recent_closes if float(row.get("pnl", 0.0) or 0.0) > 0)
    losses = len(recent_closes) - wins
    avg_recent_pnl = (
        sum(float(row.get("pnl", 0.0) or 0.0) for row in recent_closes) / max(len(recent_closes), 1)
        if recent_closes else 0.0
    )
    prices = _live_prices()
    portfolio_equity = equity(portfolio, prices or None)
    starting_balance = float(portfolio.get("starting_balance", 1000.0) or 1000.0)
    total_pnl_pct = ((portfolio_equity - starting_balance) / starting_balance * 100.0) if starting_balance else 0.0
    stale_inputs = []
    portfolio_age = _age_hours(portfolio.get("updated"))
    last_scan_age = _age_hours(last_scan.get("ts"))
    if portfolio_age is None or portfolio_age > MAX_PORTFOLIO_AGE_HOURS:
        stale_inputs.append(f"portfolio age {portfolio_age:.1f}h" if portfolio_age is not None else "portfolio missing updated")
    if last_scan_age is None or last_scan_age > MAX_LAST_SCAN_AGE_HOURS:
        stale_inputs.append(f"last scan age {last_scan_age:.1f}h" if last_scan_age is not None else "last scan missing ts")

    recommendation = "OBSERVE"
    reasons = []
    if not bool(model_meta.get("gate_pass", False)):
        recommendation = "DEMOTE"
        reasons.append("Long model gate failed")
    elif total_pnl_pct <= -5.0:
        recommendation = "REVIEW"
        reasons.append(f"Paper PnL {total_pnl_pct:.2f}% is below -5%")
    elif recent_closes and wins / max(len(recent_closes), 1) < 0.35:
        recommendation = "REVIEW"
        reasons.append("Recent close win rate below 35%")
    elif len(recent_closes) >= 6 and avg_recent_pnl > 0 and total_pnl_pct > 0:
        recommendation = "HOLD"
        reasons.append("Recent closes and total PnL are positive")
    else:
        reasons.append("Need more clean paper evidence")

    history_cycles = int(history.get("cycles_sampled", 0) or 0)
    history_posture = history.get("posture")
    history_avg_pnl = _safe_history(history, "paper_total_pnl_pct")
    history_review_ratio = _safe_history(history, "review_or_demote", kind="ratios")
    if history.get("status") == "ok" and history_cycles >= 6:
        if history_posture == "degraded":
            recommendation = "REVIEW"
            reasons.append(
                f"Durable history remains degraded across {history_cycles} cycles "
                f"(avg paper pnl {history_avg_pnl:+.2f}%, review ratio {history_review_ratio:.0%})."
            )
        elif history_posture == "supportive" and recommendation == "OBSERVE":
            recommendation = "HOLD"
            reasons.append("Durable history is supportive across recent operator cycles.")
    if stale_inputs:
        recommendation = "REVIEW"
        reasons.append("Stale QuantForge inputs detected: " + ", ".join(stale_inputs))

    report = {
        "generated_at": now.isoformat(),
        "recommendation": recommendation,
        "reasons": reasons,
        "stale_inputs": stale_inputs,
        "paper": {
            "starting_balance": starting_balance,
            "cash": float(portfolio.get("cash", 0.0) or 0.0),
            "realized_pnl": float(portfolio.get("realized_pnl", 0.0) or 0.0),
            "total_pnl_pct": round(total_pnl_pct, 4),
            "open_positions": len(portfolio.get("positions", {})),
            "total_trades": int(portfolio.get("total_trades", 0) or 0),
            "max_drawdown": float(portfolio.get("max_drawdown", 0.0) or 0.0),
        },
        "recent_closes": {
            "count": len(recent_closes),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(len(recent_closes), 1), 4) if recent_closes else 0.0,
            "avg_pnl": round(avg_recent_pnl, 4),
        },
        "model": {
            "trained_at": model_meta.get("trained_at"),
            "auc": model_meta.get("overall_auc"),
            "gate_pass": model_meta.get("gate_pass"),
            "threshold": model_meta.get("optimal_threshold"),
        },
        "short_model": {
            "trained_at": short_meta.get("trained_at"),
            "auc": short_meta.get("overall_auc"),
            "gate_pass": short_meta.get("gate_pass"),
            "threshold": short_meta.get("optimal_threshold"),
        },
        "last_scan": {
            "ts": last_scan.get("ts"),
            "signal_count": len(last_scan.get("signals", [])),
            "feedback_summary": last_scan.get("feedback", {}).get("summary", {}),
            "regime": last_scan.get("regime", {}),
        },
        "durable_history": {
            "status": history.get("status"),
            "cycles_sampled": history_cycles,
            "posture": history_posture,
            "averages": history.get("averages", {}),
            "ratios": history.get("ratios", {}),
        },
    }
    return report


def _safe_history(history, field, kind="averages"):
    try:
        return float((history.get(kind) or {}).get(field, 0.0) or 0.0)
    except Exception:
        return 0.0


def cmd_report():
    cfg.require_production_runtime("quantforge_governance.py")
    report = build_report()
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print("QuantForge governance report")
    print(f"Recommendation: {report['recommendation']}")
    for reason in report["reasons"]:
        print(f"  - {reason}")
    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    cmd_report()
