#!/usr/bin/env python3
"""QuantForge Moonshot Sleeve — barbell satellite lane . PAPER ONLY.

The barbell structure: the core book (quantforge_agent, $5,000) stays
disciplined at <=5x with stops. This sleeve is the OTHER end of the barbell —
a small, walled-off ledger ($250 paper, ~5% satellite) that is allowed to use
10x leverage because its TOTAL LOSS is already budgeted. It is an option
premium, not an investment: expect it to hit zero sometimes; the asymmetry
must come from selective entries at structural dislocations.

THE WALL (what makes a barbell a barbell and not just more risk):
  - Separate ledger file. Never touches the agent portfolio, by construction.
  - Entries ONLY at measured dislocations (liquidation cascades) — the one
    regime where "the time is right" is a measurement, not a feeling.
  - One position at a time. Margin = 20% of sleeve. 10x paper leverage.
  - Binary, pre-committed lifecycle: TP +100% of margin, SL -50% of margin,
    simulated liquidation at -9.5% price (checked against intrabar 1h
    extremes, worst-case ordering — no optimistic fills).
  - 4 consecutive losses -> frozen 7 days. Sleeve < 40% of start -> frozen
    until governance review. No automatic refills, ever.
  - Respects operator KILL files.

Runs every 2h piggybacked on the derivatives collector cron (fresh data,
no new cron entry). Usage: quantforge_moonshot.py run|status
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta


def _http_json(url, params, timeout):
    qs = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{qs}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
STATE_PATH = os.path.join(DATA_DIR, "moonshot_state.json")
# 2h-cadence per-symbol snapshots from quantforge_collect_all.py; the
# liquidation_*_usd fields are running totals, so cascades are detected as
# interval DELTAS vs their own trailing 7d average. (The *_latest.parquet
# has no history/averages — reading it alone can never detect a spike.)
DERIV_HIST_PATH = os.path.join(DATA_DIR, "derivatives", "derivatives_state_history.jsonl")
KILL_FILES = [os.path.join(DATA_DIR, "KILL"), os.path.join(DATA_DIR, "KILL_FLATTEN")]

STARTING_BALANCE = 250.0        # 5% satellite of the $5k core — separate book
LEVERAGE = 10.0                 # paper; total loss per attempt = margin
MARGIN_FRAC = 0.20              # 20% of sleeve per attempt
TP_MARGIN_PCT = 1.00            # take profit at +100% of margin (+10% price)
SL_MARGIN_PCT = -0.50           # stop at -50% of margin (-5% price)
LIQ_PRICE_PCT = 0.095           # simulated liquidation distance at 10x
MAX_HOLD_HOURS = 48
LIQ_SPIKE_MULT = 3.0            # latest 2h liq delta > 3x its 7d average
MIN_BASELINE_SAMPLES = 36       # >= 3 days of 2h deltas before arming
BASELINE_WINDOW_S = 7 * 86400
ENTRY_COOLDOWN_H = 12           # one cascade must not chain multiple entries
FREEZE_AFTER_LOSSES = 4
FREEZE_DAYS = 7
DEAD_FLOOR_FRAC = 0.40          # sleeve below 40% of start -> dead until review
SYMBOL = "BTC-USDT"


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "balance": STARTING_BALANCE,
        "starting_balance": STARTING_BALANCE,
        "position": None,
        "consecutive_losses": 0,
        "frozen_until": None,
        "dead": False,
        "stats": {"wins": 0, "losses": 0, "liquidations": 0, "total_pnl": 0.0},
        "history": [],
        "created_at": now_iso(),
    }


def save_state(s):
    s["updated_at"] = now_iso()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, STATE_PATH)


def get_price():
    data = _http_json("https://api.kucoin.com/api/v1/market/orderbook/level1",
                      {"symbol": SYMBOL}, timeout=10)
    return float(data["data"]["price"])


def get_klines_since(start_ts):
    """1h candles since start_ts: list of (ts, high, low)."""
    data = _http_json("https://api.kucoin.com/api/v1/market/candles",
                      {"type": "1hour", "symbol": SYMBOL,
                       "startAt": int(start_ts), "endAt": int(time.time())},
                      timeout=15)
    out = []
    for row in data.get("data", []):
        # kucoin order: [time, open, close, high, low, volume, turnover]
        out.append((int(row[0]), float(row[3]), float(row[4])))
    return sorted(out)


def read_dislocation():
    """Detect a liquidation cascade from the collector's history JSONL.

    Computes the latest 2h liquidation DELTA for BTC and compares it to the
    trailing-7d average of deltas. Returns (direction, detail) or
    (None, reason). LONG = fade a long-flush (forced sellers exhausted),
    SHORT = fade a short-squeeze pump.
    """
    if not os.path.exists(DERIV_HIST_PATH):
        return None, "no derivatives history"
    cutoff = time.time() - BASELINE_WINDOW_S - 7200
    rows = []
    try:
        with open(DERIV_HIST_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or f'"{SYMBOL}"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("symbol") != SYMBOL or e.get("timestamp", 0) < cutoff:
                    continue
                rows.append((e["timestamp"],
                             float(e.get("liquidation_long_usd", 0) or 0),
                             float(e.get("liquidation_short_usd", 0) or 0),
                             float(e.get("funding_rate", 0) or 0)))
    except Exception as e:
        return None, f"deriv history read failed: {e}"

    rows.sort()
    if len(rows) < MIN_BASELINE_SAMPLES + 1:
        return None, f"baseline warming up ({len(rows)}/{MIN_BASELINE_SAMPLES + 1} samples)"

    # interval deltas; totals can reset or jitter down -> clamp at 0
    d_long = [max(0.0, rows[i][1] - rows[i - 1][1]) for i in range(1, len(rows))]
    d_short = [max(0.0, rows[i][2] - rows[i - 1][2]) for i in range(1, len(rows))]
    last_long, last_short = d_long[-1], d_short[-1]
    avg_long = sum(d_long[:-1]) / len(d_long[:-1])
    avg_short = sum(d_short[:-1]) / len(d_short[:-1])
    funding = rows[-1][3]

    if avg_long > 0 and last_long > LIQ_SPIKE_MULT * avg_long:
        return "LONG", (f"long-flush cascade: 2h liq ${last_long:,.0f} vs 7d-avg "
                        f"${avg_long:,.0f} ({last_long / avg_long:.1f}x), "
                        f"funding {funding * 100:.4f}% — fading the flush")
    if avg_short > 0 and last_short > LIQ_SPIKE_MULT * avg_short:
        return "SHORT", (f"short-squeeze cascade: 2h liq ${last_short:,.0f} vs 7d-avg "
                         f"${avg_short:,.0f} ({last_short / avg_short:.1f}x), "
                         f"funding {funding * 100:.4f}% — fading the pump")
    lr = last_long / avg_long if avg_long > 0 else 0.0
    sr = last_short / avg_short if avg_short > 0 else 0.0
    return None, f"no cascade (2h deltas: long {lr:.2f}x, short {sr:.2f}x of 7d avg)"


def close_position(s, exit_price, reason, pnl_margin_pct):
    """Settle the open position. pnl_margin_pct is PnL as fraction of margin."""
    pos = s["position"]
    pnl = pos["margin"] * pnl_margin_pct
    s["balance"] = round(s["balance"] + pos["margin"] + pnl, 2)
    s["stats"]["total_pnl"] = round(s["stats"]["total_pnl"] + pnl, 2)
    won = pnl > 0
    if won:
        s["stats"]["wins"] += 1
        s["consecutive_losses"] = 0
    else:
        s["stats"]["losses"] += 1
        s["consecutive_losses"] += 1
        if reason == "liquidation":
            s["stats"]["liquidations"] += 1
    s["history"].append({
        "ts": now_iso(), "direction": pos["direction"],
        "entry": pos["entry_price"], "exit": exit_price,
        "margin": pos["margin"], "pnl": round(pnl, 2), "reason": reason,
    })
    s["history"] = s["history"][-200:]
    s["position"] = None
    log(f"  CLOSE {pos['direction']} @ {exit_price:,.1f} | {reason} | "
        f"PnL ${pnl:+.2f} | sleeve ${s['balance']:.2f}")

    if s["consecutive_losses"] >= FREEZE_AFTER_LOSSES:
        until = (datetime.now(timezone.utc) + timedelta(days=FREEZE_DAYS)).isoformat()
        s["frozen_until"] = until
        log(f"   {FREEZE_AFTER_LOSSES} consecutive losses — frozen until {until[:10]}")
    if s["balance"] < s["starting_balance"] * DEAD_FLOOR_FRAC:
        s["dead"] = True
        log(f"   sleeve below {DEAD_FLOOR_FRAC:.0%} of start — DEAD until "
            f"governance review (budgeted loss realized; no auto-refill)")


def manage_position(s, price):
    """Check liquidation / SL / TP / time-box against intrabar extremes.

    Worst-case ordering: if both an adverse level and TP were touched since
    entry, the adverse exit wins. No optimistic fills.
    """
    pos = s["position"]
    entry = pos["entry_price"]
    opened = datetime.fromisoformat(pos["opened_at"])
    sign = 1.0 if pos["direction"] == "LONG" else -1.0

    liq_price = entry * (1 - sign * LIQ_PRICE_PCT)
    sl_price = entry * (1 + sign * SL_MARGIN_PCT / LEVERAGE)
    tp_price = entry * (1 + sign * TP_MARGIN_PCT / LEVERAGE)

    try:
        candles = get_klines_since(opened.timestamp())
    except Exception as e:
        log(f"  klines failed ({e}) — using spot price only")
        candles = [(int(time.time()), price, price)]

    for _, high, low in candles:
        adverse = low if sign > 0 else high
        favorable = high if sign > 0 else low
        if (adverse - liq_price) * sign <= 0:
            close_position(s, liq_price, "liquidation", -1.0)
            return
        if (adverse - sl_price) * sign <= 0:
            close_position(s, sl_price, "stop_loss", SL_MARGIN_PCT)
            return
        if (favorable - tp_price) * sign >= 0:
            close_position(s, tp_price, "take_profit", TP_MARGIN_PCT)
            return

    held_h = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
    if held_h >= MAX_HOLD_HOURS:
        pnl_pct = sign * (price / entry - 1) * LEVERAGE
        close_position(s, price, "time_box_48h", pnl_pct)
        return
    upnl = pos["margin"] * sign * (price / entry - 1) * LEVERAGE
    log(f"  holding {pos['direction']} @ {entry:,.1f} ({held_h:.1f}h) | "
        f"mark {price:,.1f} | uPnL ${upnl:+.2f}")


def run():
    for kf in KILL_FILES:
        if os.path.exists(kf):
            log(f"KILL file present ({kf}) — moonshot idle")
            return
    s = load_state()
    log(f"=== moonshot cycle | sleeve ${s['balance']:.2f} "
        f"(start ${s['starting_balance']:.2f}) ===")

    if s["dead"]:
        log("   sleeve dead — awaiting governance review/refill decision")
        save_state(s)
        return

    price = get_price()

    if s["position"]:
        manage_position(s, price)
        save_state(s)
        return

    if s["frozen_until"]:
        if datetime.fromisoformat(s["frozen_until"]) > datetime.now(timezone.utc):
            log(f"   frozen until {s['frozen_until'][:16]}")
            save_state(s)
            return
        s["frozen_until"] = None
        s["consecutive_losses"] = 0
        log("  freeze expired — re-armed")

    if s.get("last_entry_at"):
        since_h = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(s["last_entry_at"])).total_seconds() / 3600
        if since_h < ENTRY_COOLDOWN_H:
            log(f"  cooldown: {since_h:.1f}h since last entry (< {ENTRY_COOLDOWN_H}h)")
            save_state(s)
            return

    direction, detail = read_dislocation()
    if direction is None:
        log(f"  idle: {detail}")
        save_state(s)
        return

    margin = round(s["balance"] * MARGIN_FRAC, 2)
    if margin < 10:
        log(f"  margin ${margin:.2f} too small — sleeve effectively dead")
        s["dead"] = True
        save_state(s)
        return
    s["balance"] = round(s["balance"] - margin, 2)
    s["last_entry_at"] = now_iso()
    s["position"] = {
        "direction": direction,
        "margin": margin,
        "notional": round(margin * LEVERAGE, 2),
        "entry_price": price,
        "leverage": LEVERAGE,
        "opened_at": now_iso(),
        "signal": detail,
    }
    log(f"   OPEN {direction} | margin ${margin:.2f} | notional "
        f"${margin * LEVERAGE:,.2f} (10x paper) @ {price:,.1f}")
    log(f"     signal: {detail}")
    save_state(s)


def status():
    s = load_state()
    st = s["stats"]
    n = st["wins"] + st["losses"]
    wr = st["wins"] / n if n else 0.0
    print(f"Moonshot sleeve (PAPER, barbell satellite)")
    print(f"  balance:   ${s['balance']:.2f} / start ${s['starting_balance']:.2f} "
          f"({(s['balance'] / s['starting_balance'] - 1) * 100:+.1f}%)")
    print(f"  trades:    {n} | WR {wr:.0%} | liquidations {st['liquidations']} "
          f"| total PnL ${st['total_pnl']:+.2f}")
    print(f"  state:     {'DEAD' if s['dead'] else ('FROZEN until ' + s['frozen_until'][:16]) if s['frozen_until'] else 'armed'}")
    if s["position"]:
        p = s["position"]
        print(f"  position:  {p['direction']} @ {p['entry_price']:,.1f} "
              f"margin ${p['margin']:.2f} since {p['opened_at'][:16]}")
    for h in s["history"][-5:]:
        print(f"    {h['ts'][:16]} {h['direction']} {h['reason']} ${h['pnl']:+.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        status()
    else:
        run()
