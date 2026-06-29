#!/usr/bin/env python3
"""QuantForge — promotion/demotion scorecard.

Builds a model-promotion report from saved ML metadata and paper-trading artifacts.
This does not change execution behavior; it only summarizes readiness.

Usage:
    python3 quantforge_promotion_report.py
    python3 quantforge_promotion_report.py --json
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

QUANTFORGE_DIR = os.path.join(cfg.data, "quantforge")
MODEL_DIR = os.path.join(QUANTFORGE_DIR, "model")
OPTIMIZATION_DIR = os.path.join(QUANTFORGE_DIR, "optimization")
REPORT_PATH = os.path.join(MODEL_DIR, "promotion_report.json")
AGI_OPERATOR_HISTORY_PATH = os.path.join(cfg.data, "agi-operator-history-summary.json")

MIN_PAPER_CLOSED_TRADES = 10
MIN_PAPER_PROFIT_FACTOR = 1.05
MIN_PAPER_REALIZED_PNL = 0.0
MAX_PAPER_DRAWDOWN = 0.12


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        return []
    return rows


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def summarize_paper():
    portfolio = read_json(os.path.join(QUANTFORGE_DIR, "portfolio.json")) or {}
    trades = read_jsonl(os.path.join(QUANTFORGE_DIR, "paper-trades.jsonl"))
    closed = [t for t in trades if t.get("type") == "CLOSE"]

    realized_pnl = _safe_float(portfolio.get("realized_pnl"), 0.0)
    gross_profit = _safe_float(portfolio.get("gross_profit"), 0.0)
    gross_loss = abs(_safe_float(portfolio.get("gross_loss"), 0.0))
    portfolio_closed_trades = int(portfolio.get("total_trades", 0) or 0)
    closed_trades = max(len(closed), portfolio_closed_trades)
    wins = int(portfolio.get("wins", 0) or 0)
    losses = int(portfolio.get("losses", 0) or 0)
    if wins + losses == 0 and closed:
        wins = sum(1 for t in closed if _safe_float(t.get("pnl")) > 0)
        losses = sum(1 for t in closed if _safe_float(t.get("pnl")) <= 0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    max_drawdown = _safe_float(portfolio.get("max_drawdown"), 0.0)

    verdict = "INSUFFICIENT"
    reasons = []
    if closed_trades < MIN_PAPER_CLOSED_TRADES:
        reasons.append(f"Only {closed_trades} closed paper trades < {MIN_PAPER_CLOSED_TRADES}")
    if closed_trades >= MIN_PAPER_CLOSED_TRADES and realized_pnl <= MIN_PAPER_REALIZED_PNL:
        reasons.append(f"Realized PnL {realized_pnl:+.2f} <= {MIN_PAPER_REALIZED_PNL:+.2f}")
    if closed_trades >= MIN_PAPER_CLOSED_TRADES and profit_factor < MIN_PAPER_PROFIT_FACTOR:
        reasons.append(f"Profit factor {profit_factor:.2f} < {MIN_PAPER_PROFIT_FACTOR:.2f}")
    if max_drawdown > MAX_PAPER_DRAWDOWN:
        reasons.append(f"Max drawdown {max_drawdown:.1%} > {MAX_PAPER_DRAWDOWN:.1%}")

    if closed_trades >= MIN_PAPER_CLOSED_TRADES and not reasons:
        verdict = "SUPPORTIVE"
    elif closed_trades >= MIN_PAPER_CLOSED_TRADES:
        verdict = "WEAK"

    return {
        "verdict": verdict,
        "closed_trades": closed_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / (wins + losses)) if (wins + losses) else None,
        "realized_pnl": realized_pnl,
        "profit_factor": profit_factor if profit_factor != float("inf") else None,
        "max_drawdown": max_drawdown,
        "open_positions": len((portfolio.get("positions") or {})),
        "reasons": reasons,
    }


def summarize_side(name, meta, *, min_auc=None):
    if not meta:
        return {
            "side": name,
            "decision": "MISSING",
            "reasons": ["No saved model metadata found"],
        }

    gate = bool(meta.get("gate_pass", False))
    gate_eval = meta.get("gate_evaluation") or {}
    holdout_trades = int(meta.get("holdout_trades", 0) or 0)
    holdout_auc = meta.get("holdout_auc")

    reasons = []
    decision = "KEEP_IN_PAPER"

    if not gate:
        decision = "BLOCKED"
        reasons.extend(gate_eval.get("reasons") or ["Model gate failed"])
    elif holdout_trades <= 0:
        reasons.append("No hold-out trade sample saved")
    elif holdout_trades < (gate_eval.get("criteria", {}).get("min_holdout_trades") or 25):
        reasons.append(f"Hold-out trade count still thin at {holdout_trades}")

    if min_auc is not None and holdout_auc is not None and holdout_auc < min_auc:
        reasons.append(f"Hold-out AUC {holdout_auc:.4f} < {min_auc:.2f}")
        decision = "BLOCKED"

    if gate and holdout_trades >= (gate_eval.get("criteria", {}).get("min_holdout_trades") or 25) and not reasons:
        decision = "PROMOTE_CANDIDATE"

    return {
        "side": name,
        "decision": decision,
        "trained_at": meta.get("trained_at"),
        "overall_auc": meta.get("overall_auc"),
        "optimal_threshold": meta.get("optimal_threshold"),
        "holdout_auc": holdout_auc,
        "holdout_win_rate": meta.get("holdout_win_rate"),
        "holdout_sharpe": meta.get("holdout_sharpe"),
        "holdout_trades": holdout_trades,
        "gate_pass": gate,
        "gate_evaluation": gate_eval,
        "reasons": reasons,
    }


def summarize_operator_history():
    history = read_json(AGI_OPERATOR_HISTORY_PATH) or {}
    if not history:
        return {
            "status": "missing",
            "cycles_sampled": 0,
            "posture": "unknown",
            "persistent_review": False,
            "persistent_drift": False,
            "averages": {},
            "reasons": ["No durable operator history found"],
        }

    reasons = []
    if history.get("persistent_review"):
        reasons.append("Durable operator history shows repeated review/demote state")
    if history.get("persistent_drift"):
        reasons.append("Durable operator history shows repeated drifting/watch state")

    return {
        "status": history.get("status", "unknown"),
        "cycles_sampled": int(history.get("cycles_sampled", 0) or 0),
        "posture": history.get("posture", "unknown"),
        "persistent_review": bool(history.get("persistent_review", False)),
        "persistent_drift": bool(history.get("persistent_drift", False)),
        "averages": history.get("averages", {}),
        "reasons": reasons,
    }


def overall_decision(long_side, short_side, paper, operator_history):
    reasons = []
    promotable_sides = [s["side"] for s in (long_side, short_side) if s["decision"] == "PROMOTE_CANDIDATE"]
    blocked_sides = [s["side"] for s in (long_side, short_side) if s["decision"] == "BLOCKED"]
    history_ready = operator_history.get("status") == "ok" and operator_history.get("cycles_sampled", 0) >= 3
    history_degraded = bool(operator_history.get("persistent_review")) or bool(operator_history.get("persistent_drift"))

    if history_ready and history_degraded:
        reasons.extend(operator_history.get("reasons", []))
        if promotable_sides:
            reasons.append(
                f"Promotable sides exist ({', '.join(promotable_sides)}), but durable operator history still blocks rollout"
            )
            return "KEEP_IN_PAPER", reasons
        reasons.append("Promotion blocked until durable operator history improves")
        return "DO_NOT_PROMOTE", reasons

    if not promotable_sides:
        reasons.append("No side currently clears model gate + hold-out requirements")
        return "DO_NOT_PROMOTE", reasons

    if paper["verdict"] == "SUPPORTIVE":
        reasons.append(f"Paper trading is supportive with {paper['closed_trades']} closed trades")
        reasons.append(f"Promotable sides: {', '.join(promotable_sides)}")
        return "PROMOTE_CANDIDATE", reasons

    if paper["verdict"] == "WEAK":
        reasons.extend(paper["reasons"])
        reasons.append(f"Promotable sides exist but paper performance is weak ({', '.join(promotable_sides)})")
        return "DEMOTE_CANDIDATE", reasons

    reasons.extend(paper["reasons"])
    reasons.append(f"Promotable sides exist but paper evidence is still thin ({', '.join(promotable_sides)})")
    if blocked_sides:
        reasons.append(f"Blocked sides: {', '.join(blocked_sides)}")
    return "KEEP_IN_PAPER", reasons


def build_report():
    long_meta = read_json(os.path.join(MODEL_DIR, "model_meta.json"))
    short_meta = read_json(os.path.join(MODEL_DIR, "model_meta_short.json"))
    best_params = read_json(os.path.join(OPTIMIZATION_DIR, "best-params.json"))
    paper = summarize_paper()
    operator_history = summarize_operator_history()
    long_side = summarize_side("LONG", long_meta)
    short_side = summarize_side("SHORT", short_meta, min_auc=0.55)
    decision, reasons = overall_decision(long_side, short_side, paper, operator_history)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_decision": decision,
        "overall_reasons": reasons,
        "paper": paper,
        "operator_history": operator_history,
        "sides": {
            "long": long_side,
            "short": short_side,
        },
        "optimizer": {
            "available": bool(best_params),
            "optimized_at": best_params.get("optimized_at") if best_params else None,
            "threshold": best_params.get("threshold") if best_params else None,
            "score": best_params.get("score") if best_params else None,
            "holdout_trades": best_params.get("holdout_trades") if best_params else None,
            "cv_trades": best_params.get("cv_trades") if best_params else None,
        },
    }
    return report


def print_report(report):
    print("=" * 62)
    print("QuantForge Promotion Report")
    print("=" * 62)
    print(f"Generated:         {report['generated_at']}")
    print(f"Overall decision:  {report['overall_decision']}")
    for reason in report["overall_reasons"]:
        print(f"  - {reason}")

    paper = report["paper"]
    print("\nPaper evidence")
    print(f"  Verdict:         {paper['verdict']}")
    print(f"  Closed trades:   {paper['closed_trades']}")
    if paper["win_rate"] is not None:
        print(f"  Win rate:        {paper['win_rate']:.1%}")
    print(f"  Realized PnL:    {paper['realized_pnl']:+.2f}")
    if paper["profit_factor"] is not None:
        print(f"  Profit factor:   {paper['profit_factor']:.2f}")
    print(f"  Max drawdown:    {paper['max_drawdown']:.1%}")
    print(f"  Open positions:  {paper['open_positions']}")
    for reason in paper["reasons"]:
        print(f"  - {reason}")

    history = report["operator_history"]
    print("\nDurable operator history")
    print(f"  Status:          {history['status']}")
    print(f"  Cycles sampled:  {history['cycles_sampled']}")
    print(f"  Posture:         {history['posture']}")
    for reason in history["reasons"]:
        print(f"  - {reason}")

    for key in ("long", "short"):
        side = report["sides"][key]
        print(f"\n{side['side']} side")
        print(f"  Decision:        {side['decision']}")
        print(f"  Gate:            {'PASS' if side.get('gate_pass') else 'FAIL'}")
        if side.get("overall_auc") is not None:
            print(f"  CV AUC:          {float(side['overall_auc']):.4f}")
        if side.get("holdout_auc") is not None:
            print(f"  Hold-out AUC:    {float(side['holdout_auc']):.4f}")
        if side.get("holdout_win_rate") is not None:
            print(f"  Hold-out WR:     {float(side['holdout_win_rate']):.1%}")
        if side.get("holdout_sharpe") is not None:
            print(f"  Hold-out Sharpe: {float(side['holdout_sharpe']):.2f}")
        if side.get("holdout_trades") is not None:
            print(f"  Hold-out trades: {int(side['holdout_trades'])}")
        for reason in side["reasons"]:
            print(f"  - {reason}")

    optimizer = report["optimizer"]
    print("\nOptimizer")
    print(f"  Available:       {'yes' if optimizer['available'] else 'no'}")
    if optimizer["available"]:
        print(f"  Optimized at:    {optimizer['optimized_at']}")
        if optimizer["score"] is not None:
            print(f"  Score:           {optimizer['score']}")
        if optimizer["threshold"] is not None:
            print(f"  Threshold:       {optimizer['threshold']}")
        if optimizer["holdout_trades"] is not None:
            print(f"  Hold-out trades: {optimizer['holdout_trades']}")
        if optimizer["cv_trades"] is not None:
            print(f"  CV trades:       {optimizer['cv_trades']}")

    print(f"\nSaved:             {REPORT_PATH}")


def main():
    report = build_report()
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    if "--json" in sys.argv[1:]:
        print(json.dumps(report, indent=2))
        return
    print_report(report)


if __name__ == "__main__":
    main()
