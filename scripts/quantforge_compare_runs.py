#!/usr/bin/env python3
"""QuantForge — compare archived and live paper runs.

Usage:
    python3 quantforge_compare_runs.py
    python3 quantforge_compare_runs.py /path/to/runA /path/to/runB /path/to/live
"""

import json
import math
import os
import sys
from typing import Any


_QF_BASE = os.path.expanduser(os.environ.get("QF_BASE_DIR", "~/quantforge"))
DEFAULT_RUNS = [
    (os.path.join(_QF_BASE, "data/quantforge/archive/run-A"), "baseline-reset"),
    (os.path.join(_QF_BASE, "data/quantforge/archive/run-B"), "pre-tuned-archive"),
    (os.path.join(_QF_BASE, "data/quantforge"), "tuned-live"),
]


def load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_trades(path: str) -> list[dict]:
    trades = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trades.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return trades


def summarize_run(path: str, label: str) -> dict:
    portfolio = load_json(os.path.join(path, "portfolio.json"), {})
    trades = load_trades(os.path.join(path, "paper-trades.jsonl"))
    signals = load_trades(os.path.join(path, "signals.jsonl"))

    starting = float(portfolio.get("starting_balance", 1000.0) or 1000.0)
    cash = float(portfolio.get("cash", starting) or starting)
    positions = portfolio.get("positions", {}) or {}
    realized = float(portfolio.get("realized_pnl", 0.0) or 0.0)
    fees = float(portfolio.get("total_fees_paid", 0.0) or 0.0)
    max_dd = float(portfolio.get("max_drawdown", 0.0) or 0.0)
    wins = int(portfolio.get("wins", 0) or 0)
    losses = int(portfolio.get("losses", 0) or 0)
    total_trades = int(portfolio.get("total_trades", 0) or 0)

    close_trades = [trade for trade in trades if trade.get("type") == "CLOSE"]
    open_trades = [trade for trade in trades if trade.get("type") == "OPEN"]
    trailing_updates = sum(1 for trade in trades if trade.get("type") == "TRAILING_UPDATE")
    signal_count = len(signals)

    gross_profit = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in close_trades if float(trade.get("pnl", 0.0) or 0.0) > 0)
    gross_loss = abs(sum(float(trade.get("pnl", 0.0) or 0.0) for trade in close_trades if float(trade.get("pnl", 0.0) or 0.0) < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else math.inf
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0

    return {
        "label": label,
        "path": path,
        "starting_balance": starting,
        "cash": cash,
        "realized_pnl": realized,
        "total_pnl_pct": (realized / starting * 100.0) if starting else 0.0,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "open_positions": len(positions),
        "open_trade_events": len(open_trades),
        "close_trade_events": len(close_trades),
        "signal_count": signal_count,
        "trailing_updates": trailing_updates,
        "fees_paid": fees,
        "max_drawdown_pct": max_dd * 100.0,
        "created": portfolio.get("created"),
        "updated": portfolio.get("updated"),
    }


def fmt_float(value: float, digits: int = 2) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def print_run(run: dict) -> None:
    print(run["label"])
    print(f"  path:             {run['path']}")
    print(f"  realized pnl:     ${fmt_float(run['realized_pnl'])} ({fmt_float(run['total_pnl_pct'])}%)")
    print(f"  trades:           {run['total_trades']}  wins/losses={run['wins']}/{run['losses']}  win_rate={fmt_float(run['win_rate'] * 100)}%")
    print(f"  profit factor:    {fmt_float(run['profit_factor'])}")
    print(f"  open positions:   {run['open_positions']}")
    print(f"  signals logged:   {run['signal_count']}")
    print(f"  trailing updates: {run['trailing_updates']}")
    print(f"  fees paid:        ${fmt_float(run['fees_paid'])}")
    print(f"  max drawdown:     {fmt_float(run['max_drawdown_pct'])}%")
    print(f"  updated:          {run['updated']}")


def main():
    if len(sys.argv) > 1:
        run_specs = [(path, os.path.basename(path.rstrip("/")) or path) for path in sys.argv[1:]]
    else:
        run_specs = DEFAULT_RUNS

    runs = [summarize_run(path, label) for path, label in run_specs]
    print("QuantForge run comparison")
    print("=" * 60)
    for run in runs:
        print_run(run)
        print("-" * 60)


if __name__ == "__main__":
    main()
