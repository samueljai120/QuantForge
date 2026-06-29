#!/usr/bin/env python3
"""Live benchmark-gate evaluator (Phase E capital protection).

Computes the agent's realized return since the portfolio reset vs the only
honest default — buying and HOLDING BTC over the same window — and applies
qf_mlops.benchmark_gate. Writes a verdict to benchmark_gate_state.json and
prints it. The rule the whole session earned: active trading must out-earn
holding, on live evidence, or it has not justified its risk.

Read-only: this evaluates and reports. Enforcement (the agent defaulting to
hold while SHADOW) is a separate, explicitly-approved step.
"""
from collections import deque
import json
import math
import os
import urllib.request
from datetime import datetime, timedelta, timezone

from qf_mlops.benchmark_gate import benchmark_gate
from quantforge_equity import compute_spot_equity, compute_true_equity

DATA = os.path.expanduser("~/quantforge/data/quantforge")
PORT = os.path.join(DATA, "agent_portfolio.json")
STATE = os.path.join(DATA, "benchmark_gate_state.json")
TRADES = os.path.join(DATA, "agent_trades.jsonl")
MIN_TRADES = 20
MIN_EDGE_PCT = 0.0  # must at least match holding
EARLY_HOLD_MIN_TRADES = 10     # magnitude-based early trip needs a meaningful (not thin) sample
CATASTROPHIC_EDGE_PCT = -15.0  # active trailing HODL by >=15 pts = real divergence, not noise
STALE_METADATA_GAP_DAYS = 7    # ancient reset metadata + thin sample => trust recent live activity instead


def compute_enforce_hold(status, n_trades, active_ret, hodl_ret):
    """Whether the agent should default to HODL (capital protection).

    Two evidence-based triggers; fails OPEN (returns False) on any missing/unknown input:
      - full-sample : proven underperformance over the full window
                      (status SHADOW, n_trades >= MIN_TRADES, active < hodl)
      - early-trip  : catastrophic *realized* divergence on a still-meaningful sample
                      (n_trades >= EARLY_HOLD_MIN_TRADES, active trails hodl by >= 15 pts)

    Never an override: both require active to actually be trailing HODL on real trades.
    Never fires on a thin sample (< EARLY_HOLD_MIN_TRADES) or UNKNOWN status (price fetch failed).
    """
    if status not in ("SHADOW", "PROMOTED"):    # UNKNOWN / error -> keep trading (fail open)
        return False
    if active_ret is None or hodl_ret is None:
        return False
    try:
        active_ret, hodl_ret = float(active_ret), float(hodl_ret)
    except (TypeError, ValueError):             # non-numeric -> fail open
        return False
    if not (math.isfinite(active_ret) and math.isfinite(hodl_ret)):  # NaN/inf -> fail open
        return False
    if active_ret >= hodl_ret:                  # matching or beating the benchmark -> never hold
        return False
    edge_pct = active_ret - hodl_ret            # < 0 here
    full_sample = (status == "SHADOW" and n_trades >= MIN_TRADES)
    early_trip = (n_trades >= EARLY_HOLD_MIN_TRADES and edge_pct <= CATASTROPHIC_EDGE_PCT)
    return bool(full_sample or early_trip)


def parse_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def recent_spot_trade_times(path, max_count=None):
    """Return the timestamps of the most recent spot BUY/SELL trades."""
    times = deque(maxlen=max_count if max_count and max_count > 0 else None)
    try:
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("type") not in {"BUY", "SELL"}:
                    continue
                dt = parse_ts(row.get("ts"))
                if dt is not None:
                    times.append(dt)
    except Exception:
        return []
    return list(times)


def infer_start_anchor(port, trades_path=TRADES):
    """Pick the fairest anchor for benchmark comparison.

    Prefer explicit reset metadata. But if a thin live sample is paired with
    obviously stale metadata (for example, a legacy 2023 created_at attached to
    a fresh 10-trade run), fall back to recent live activity so the benchmark
    compares like-for-like windows.
    """
    n_trades = int(port.get("n_trades", 0) or port.get("total_trades", 0) or 0)
    max_count = n_trades if n_trades > 0 else None

    metadata_candidates = [
        ("_reset_at", parse_ts(port.get("_reset_at"))),
        ("created_at", parse_ts(port.get("created_at"))),
        ("last_panic_reset_at", parse_ts(port.get("last_panic_reset_at"))),
    ]
    metadata_source, metadata_dt = next(
        ((source, dt) for source, dt in metadata_candidates if dt is not None),
        (None, None),
    )

    activity_candidates = []
    trade_times = recent_spot_trade_times(trades_path, max_count=max_count)
    if trade_times:
        activity_candidates.append(("recent_spot_trades", trade_times[0]))

    rebalance_times = [parse_ts(ts) for ts in (port.get("rebalance_log") or [])]
    rebalance_times = [dt for dt in rebalance_times if dt is not None]
    if rebalance_times:
        if max_count:
            rebalance_times = rebalance_times[-max_count:]
        activity_candidates.append(("rebalance_log", min(rebalance_times)))

    activity_source, activity_dt = min(activity_candidates, key=lambda item: item[1]) if activity_candidates else (None, None)

    if metadata_dt and activity_dt and 0 < n_trades < MIN_TRADES:
        if metadata_dt < activity_dt - timedelta(days=STALE_METADATA_GAP_DAYS):
            return activity_dt, f"{activity_source}_fallback"

    if metadata_dt is not None:
        return metadata_dt, metadata_source
    if activity_dt is not None:
        return activity_dt, f"{activity_source}_fallback"
    return None, None


def btc_price_at(ms):
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime={ms}&limit=1"
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "QF/1"}), timeout=15).read())
        return float(d[0][4]) if d else None
    except Exception:
        return None


def btc_price_now():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "QF/1"}), timeout=15).read())
        return float(d["price"])
    except Exception:
        return None


def main():
    p = json.load(open(PORT))
    start_bal = float(p.get("starting_balance", 5000))
    n_trades = int(p.get("n_trades", 0) or p.get("total_trades", 0))
    start_dt, start_source = infer_start_anchor(p)
    current_price = btc_price_now()
    cur_eq_true = compute_true_equity(p, current_price) if current_price is not None else None
    cur_eq_spot = compute_spot_equity(p, current_price) if current_price is not None else None
    active_ret = ((cur_eq_true - start_bal) / start_bal * 100.0) if cur_eq_true is not None else None
    active_ret_spot = ((cur_eq_spot - start_bal) / start_bal * 100.0) if cur_eq_spot is not None else None

    reset_ms = int(start_dt.timestamp() * 1000) if start_dt else None
    p0 = btc_price_at(reset_ms) if reset_ms else None
    hodl_ret = (current_price - p0) / p0 * 100.0 if (p0 and current_price) else None

    if active_ret is None or hodl_ret is None:
        verdict = {"status": "UNKNOWN", "allowed": False, "reason": "could not fetch BTC benchmark prices"}
    else:
        verdict = benchmark_gate(signal_return_pct=active_ret, benchmark_return_pct=hodl_ret,
                                 n_trades=n_trades, min_trades=MIN_TRADES, min_edge_pct=MIN_EDGE_PCT)
        # enforce_hold: force the benchmark posture only on evidence-based underperformance
        # — full-sample (n>=MIN_TRADES) OR a catastrophic early-trip (n>=EARLY_HOLD_MIN_TRADES
        # and active trailing HODL by >=15 pts). Thin samples keep trading to gather evidence.
        enforce_hold = compute_enforce_hold(verdict["status"], n_trades, active_ret, hodl_ret)
        verdict.update({
            "active_return_pct": round(active_ret, 3),
            "active_return_pct_spot_only": round(active_ret_spot, 3) if active_ret_spot is not None else None,
            "active_equity": round(cur_eq_true, 2),
            "active_equity_spot_only": round(cur_eq_spot, 2) if cur_eq_spot is not None else None,
            "active_equity_mode": "true_equity",
            "hodl_return_pct": round(hodl_ret, 3),
            "n_trades": n_trades,
            "enforce_hold": enforce_hold,
            "benchmark_start_at": start_dt.isoformat() if start_dt else None,
            "benchmark_start_source": start_source,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })

    with open(STATE, "w") as f:
        json.dump(verdict, f, indent=2)
    active_label = f"{active_ret:+.2f}%" if active_ret is not None else "n/a"
    hodl_label = f"{hodl_ret:+.2f}%" if hodl_ret is not None else "n/a"
    print("BENCHMARK GATE: active %s vs HODL %s over %d trades -> %s"
          % (active_label, hodl_label, n_trades, verdict["status"]))
    print("  %s" % verdict.get("reason", ""))


if __name__ == "__main__":
    main()
