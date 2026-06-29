#!/usr/bin/env python3
"""QuantForge Daily Summary — paper trading performance snapshot.

Reads portfolio.json + paper-trades.jsonl + model_meta.json and
writes a human-readable summary to data/quantforge/daily-summary.log.
Also writes data/quantforge/daily-summary-latest.json for CEO digest.

Usage:
    python3 quantforge_daily_summary.py        # Print + log summary
    python3 quantforge_daily_summary.py json   # Print JSON only
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from quantforge_params import load_merged_quantforge_params

try:
    from quantforge_paper import get_futures_tickers, position_equity_value, position_unrealized_pnl
except Exception:
    get_futures_tickers = None
    position_equity_value = None
    position_unrealized_pnl = None

BASE_DIR = os.path.expanduser("~/quantforge")
QF_DIR   = os.path.join(BASE_DIR, "data", "quantforge")

PORTFOLIO_FILE = os.path.join(QF_DIR, "portfolio.json")
TRADES_FILE    = os.path.join(QF_DIR, "paper-trades.jsonl")
MODEL_FILE     = os.path.join(QF_DIR, "model", "model_meta.json")
LOG_FILE       = os.path.join(QF_DIR, "daily-summary.log")
LATEST_FILE    = os.path.join(QF_DIR, "daily-summary-latest.json")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_trades(path, since_hours=24):
    """Return trades from the last N hours."""
    trades = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                ts_str = t.get("ts", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts >= cutoff:
                        trades.append(t)
                except Exception:
                    pass
    except Exception:
        pass
    return trades


def load_all_trades(path):
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
                    pass
    except Exception:
        pass
    return trades


def _sorted_top(mapping, key, limit=5):
    rows = list(mapping.values())
    rows.sort(key=key, reverse=True)
    return rows[:limit]


def build_attribution(trades):
    open_by_symbol = {}
    by_symbol = {}
    by_trigger = {}
    by_direction = {
        "LONG": {"direction": "LONG", "closed": 0, "wins": 0, "pnl": 0.0},
        "SHORT": {"direction": "SHORT", "closed": 0, "wins": 0, "pnl": 0.0},
    }
    trailing_updates = 0
    trailing_activations = 0
    trailing_stop_closes = {"count": 0, "wins": 0, "pnl": 0.0}

    for trade in trades:
        ttype = trade.get("type")
        sym = trade.get("symbol")
        if ttype == "OPEN" and sym:
            open_by_symbol[sym] = trade
            continue
        if ttype == "TRAILING_UPDATE":
            trailing_updates += 1
            if not trade.get("trailing_was_active") and trade.get("trailing_is_active"):
                trailing_activations += 1
            continue
        if ttype != "CLOSE" or not sym:
            continue

        pnl = float(trade.get("pnl", 0.0) or 0.0)
        trigger = trade.get("trigger", "UNKNOWN")
        open_trade = open_by_symbol.pop(sym, {})
        direction = open_trade.get("direction", trade.get("direction", "LONG"))

        sym_row = by_symbol.setdefault(sym, {"symbol": sym, "closed": 0, "wins": 0, "pnl": 0.0})
        sym_row["closed"] += 1
        sym_row["pnl"] += pnl
        if pnl > 0:
            sym_row["wins"] += 1

        trigger_row = by_trigger.setdefault(trigger, {"trigger": trigger, "closed": 0, "wins": 0, "pnl": 0.0})
        trigger_row["closed"] += 1
        trigger_row["pnl"] += pnl
        if pnl > 0:
            trigger_row["wins"] += 1

        dir_row = by_direction.setdefault(direction, {"direction": direction, "closed": 0, "wins": 0, "pnl": 0.0})
        dir_row["closed"] += 1
        dir_row["pnl"] += pnl
        if pnl > 0:
            dir_row["wins"] += 1

        if trigger == "STOP_LOSS" and pnl > 0:
            trailing_stop_closes["count"] += 1
            trailing_stop_closes["wins"] += 1
            trailing_stop_closes["pnl"] += pnl

    top_winners = _sorted_top(by_symbol, key=lambda row: row["pnl"], limit=5)
    top_losers = sorted(by_symbol.values(), key=lambda row: row["pnl"])[:5]

    return {
        "by_symbol": sorted(by_symbol.values(), key=lambda row: row["pnl"], reverse=True),
        "by_trigger": sorted(by_trigger.values(), key=lambda row: row["pnl"], reverse=True),
        "by_direction": list(by_direction.values()),
        "top_winners": top_winners,
        "top_losers": top_losers,
        "trailing": {
            "updates": trailing_updates,
            "activations": trailing_activations,
            "profitable_stop_closes": trailing_stop_closes,
        },
    }


def get_live_prices():
    """Get current futures prices for open positions."""
    if get_futures_tickers is None:
        return {}
    try:
        prices = {}
        for c in get_futures_tickers():
            base = c.get("baseCurrency", c.get("symbol", "").replace("USDTM", ""))
            if base == "XBT":
                base = "BTC"
            prices[f"{base}-USDT"] = float(c.get("lastTradePrice", c.get("markPrice", 0)) or 0)
        return prices
    except Exception:
        return {}


def build_summary():
    now = datetime.now(timezone.utc)
    portfolio = load_json(PORTFOLIO_FILE)
    model     = load_json(MODEL_FILE)
    params    = load_merged_quantforge_params()

    # ── Account snapshot ──────────────────────────────────────────────────
    starting   = portfolio.get("starting_balance", 1000.0)
    cash       = portfolio.get("cash", 0.0)
    realized   = portfolio.get("realized_pnl", 0.0)
    fees_paid  = portfolio.get("total_fees_paid", 0.0)
    total_trades = portfolio.get("total_trades", 0)
    wins       = portfolio.get("wins", 0)
    losses     = portfolio.get("losses", 0)
    peak_eq    = portfolio.get("peak_equity", starting)
    max_dd     = portfolio.get("max_drawdown", 0.0)
    positions  = portfolio.get("positions", {})

    win_rate = wins / max(total_trades, 1) * 100

    # ── Live prices for open positions ────────────────────────────────────
    prices = get_live_prices()
    unrealized = 0.0
    pos_rows = []
    for sym, pos in positions.items():
        qty        = pos.get("qty", 0)
        entry_px   = pos.get("entry_price", 0)
        entry_fee  = pos.get("entry_fee", 0)
        direction  = pos.get("direction", "LONG")
        take_profit = pos.get("take_profit", 0)
        stop_loss   = pos.get("stop_loss", 0)
        score       = pos.get("signal_score", 0)
        cur_px = prices.get(sym, None)
        if cur_px is None:
            cur_px = entry_px
        if position_unrealized_pnl is not None:
            pnl = position_unrealized_pnl(pos, cur_px)
        else:
            pnl = (entry_px - cur_px) * qty if direction == "SHORT" else (cur_px - entry_px) * qty
        unrealized += pnl
        pnl_pct = ((entry_px - cur_px) / entry_px * 100) if direction == "SHORT" else ((cur_px - entry_px) / entry_px * 100)
        pos_rows.append({
            "symbol": sym,
            "entry": entry_px,
            "current": cur_px,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "stop": stop_loss,
            "tp": take_profit,
            "score": score,
            "direction": direction,
        })

    if position_equity_value is not None:
        equity = cash + sum(position_equity_value(pos, prices.get(sym, pos.get("entry_price", 0))) for sym, pos in positions.items())
    else:
        equity = starting + realized + unrealized
    total_pnl = realized + unrealized
    total_pnl_pct = total_pnl / starting * 100

    # ── Last 24h trades ───────────────────────────────────────────────────
    recent = load_trades(TRADES_FILE, since_hours=24)
    all_trades = load_all_trades(TRADES_FILE)
    recent_closed = [t for t in recent if t.get("type") == "CLOSE"]
    recent_pnl = sum(t.get("pnl", 0) for t in recent_closed)
    recent_wins = sum(1 for t in recent_closed if t.get("pnl", 0) > 0)
    recent_attribution = build_attribution(recent)
    all_time_attribution = build_attribution(all_trades)

    # ── Model stats ───────────────────────────────────────────────────────
    threshold    = params.get("signal_confidence_threshold", 0.80)
    model_wr     = model.get("win_rate_at_threshold", 0.0) * 100
    model_ev     = model.get("ev_at_threshold", 0.0)
    model_auc    = model.get("overall_auc", 0.0)
    model_gate   = model.get("gate_pass", False)
    model_trained = model.get("trained_at", "unknown")
    pos_size_pct = params.get("max_position_pct_for_quantforge", 0.015) * 100

    # ── Build result dict ─────────────────────────────────────────────────
    result = {
        "generated_at": now.isoformat(),
        "account": {
            "starting_balance": starting,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "peak_equity": round(peak_eq, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "fees_paid": round(fees_paid, 2),
        },
        "trades": {
            "total": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 1),
            "last_24h_closed": len(recent_closed),
            "last_24h_pnl": round(recent_pnl, 2),
            "last_24h_wins": recent_wins,
        },
        "attribution": {
            "last_24h": recent_attribution,
            "all_time": all_time_attribution,
        },
        "open_positions": pos_rows,
        "model": {
            "auc": model_auc,
            "threshold": threshold,
            "cv_win_rate_pct": round(model_wr, 1),
            "ev_per_trade": round(model_ev, 5),
            "gate_pass": model_gate,
            "trained_at": model_trained,
            "pos_size_pct": pos_size_pct,
        },
    }

    return result


def format_text(r):
    """Human-readable daily summary."""
    a  = r["account"]
    tr = r["trades"]
    m  = r["model"]
    attr = r["attribution"]
    day_attr = attr["last_24h"]
    all_attr = attr["all_time"]
    ts = r["generated_at"][:10]

    pnl_sign  = "+" if a["total_pnl"] >= 0 else ""
    day_sign  = "+" if tr["last_24h_pnl"] >= 0 else ""
    gate_flag = " PASS" if m["gate_pass"] else " FAIL"

    lines = [
        "=" * 55,
        f"  QuantForge Daily Summary — {ts}",
        "=" * 55,
        "",
        "  ACCOUNT",
        f"  Equity:          ${a['equity']:>10,.2f}",
        f"  Realized PnL:    ${a['realized_pnl']:>+10,.2f}",
        f"  Unrealized PnL:  ${a['unrealized_pnl']:>+10,.2f}",
        f"  Total PnL:       ${a['total_pnl']:>+10,.2f}  ({pnl_sign}{a['total_pnl_pct']:.2f}%)",
        f"  Max Drawdown:    {a['max_drawdown_pct']:.2f}%",
        f"  Fees Paid:       ${a['fees_paid']:.4f}",
        "",
        "  ALL-TIME TRADES",
        f"  Total:           {tr['total']}   (W={tr['wins']} / L={tr['losses']})",
        f"  Win Rate:        {tr['win_rate_pct']:.1f}%",
        "",
        "  LAST 24 HOURS",
        f"  Closed trades:   {tr['last_24h_closed']}   (W={tr['last_24h_wins']})",
        f"  24h PnL:         ${tr['last_24h_pnl']:>+.2f}",
        "",
    ]

    if r["open_positions"]:
        lines.append("  OPEN POSITIONS")
        lines.append(f"  {'Symbol':<14} {'Entry':>9} {'Current':>9} {'PnL':>8}  Stop→TP")
        for p in r["open_positions"]:
            lines.append(
                f"  {p['symbol']:<14} {p['entry']:>9.4f} {p['current']:>9.4f} "
                f"  {p['pnl']:>+7.2f}  {p['stop']:.3f}→{p['tp']:.3f}"
            )
        lines.append("")

    lines += [
        "  ATTRIBUTION (24H)",
    ]
    for row in day_attr["by_direction"]:
        if row["closed"] <= 0:
            continue
        win_rate = (row["wins"] / row["closed"] * 100) if row["closed"] else 0.0
        lines.append(
            f"  {row['direction']:<14} closed={row['closed']:<3} pnl=${row['pnl']:+.2f}  win_rate={win_rate:.1f}%"
        )
    for row in day_attr["by_trigger"][:4]:
        if row["closed"] <= 0:
            continue
        lines.append(
            f"  trigger:{row['trigger']:<8} closed={row['closed']:<3} pnl=${row['pnl']:+.2f}"
        )
    trailing_day = day_attr["trailing"]
    lines.append(
        f"  trailing events: {trailing_day['updates']} updates, {trailing_day['activations']} activations, "
        f"{trailing_day['profitable_stop_closes']['count']} profitable stop exits"
    )
    lines.append("")

    lines += [
        "  TOP CONTRIBUTORS (ALL-TIME)",
    ]
    for row in all_attr["top_winners"][:3]:
        if row["closed"] <= 0:
            continue
        lines.append(f"  + {row['symbol']:<12} pnl=${row['pnl']:+.2f} over {row['closed']} closes")
    for row in all_attr["top_losers"][:3]:
        if row["closed"] <= 0:
            continue
        lines.append(f"  - {row['symbol']:<12} pnl=${row['pnl']:+.2f} over {row['closed']} closes")
    lines.append("")

    lines += [
        "  ML MODEL",
        f"  AUC:             {m['auc']:.4f}",
        f"  Signal threshold:{m['threshold']:.2f}",
        f"  CV Win Rate:     {m['cv_win_rate_pct']:.1f}%  (at threshold)",
        f"  EV per trade:    {m['ev_per_trade']:.4f}",
        f"  Pos size:        {m['pos_size_pct']:.1f}% per trade",
        f"  Gate:            {gate_flag}",
        f"  Trained:         {m['trained_at'][:16]}",
        "",
        "=" * 55,
    ]

    return "\n".join(lines)


def main():
    r = build_summary()
    os.makedirs(QF_DIR, exist_ok=True)

    # Save JSON for CEO digest
    with open(LATEST_FILE, "w") as f:
        json.dump(r, f, indent=2)

    text = format_text(r)

    # Append to rolling log
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n\n")

    if len(sys.argv) > 1 and sys.argv[1] == "json":
        print(json.dumps(r, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
