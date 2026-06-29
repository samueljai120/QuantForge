#!/usr/bin/env python3
"""Money-conservation self-DETECTION runner.

Loads the live portfolio + trade ledger, runs qf_safety.invariants, persists a snapshot
(for cross-cycle reconciliation) + a state file, prints any violations, and exits non-zero
on a CRITICAL one so a caller (the agent cycle, the report, or self-heal) can escalate.

READ-ONLY w.r.t. the trading path — it never writes the portfolio or trades. This is the
layer that was missing: it would have flagged the margin-orphan bug in one cycle instead
of letting it bleed silently for weeks. ``evaluate()`` is the importable entry point the
agent calls in-cycle; ``main()`` is the CLI/report runner.
"""
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qf_safety.invariants import run_checks, equity

DATA = os.path.expanduser("~/quantforge/data/quantforge")
PORT = os.path.join(DATA, "agent_portfolio.json")
TRADES = os.path.join(DATA, "agent_trades.jsonl")
STATE = os.path.join(DATA, "qf_invariants_state.json")
SNAP = os.path.join(DATA, "qf_invariants_snapshot.json")
CARRY = os.path.join(DATA, "carry_harvester_state.json")


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _btc_price():
    try:
        from quantforge_agent import get_btc_price
        p = get_btc_price()
        if p:
            return float(p)
    except Exception:
        pass
    try:
        return float((_read_json(SNAP, {}) or {}).get("price") or 0) or None
    except Exception:
        return None


def _trades_since_reset(port):
    reset_dt = _infer_reset_anchor(port)
    reset = reset_dt.isoformat() if reset_dt is not None else str(port.get("created_at", "") or "")
    out = []
    try:
        with open(TRADES) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if str(r.get("ts", "")) >= reset:
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def _parse_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _recent_spot_trade_times(path, max_count=None):
    times = deque(maxlen=max_count if max_count and max_count > 0 else None)
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("type") not in {"BUY", "SELL"}:
                    continue
                dt = _parse_ts(row.get("ts"))
                if dt is not None:
                    times.append(dt)
    except Exception:
        return []
    return list(times)


def _infer_reset_anchor(port):
    """Choose the fairest anchor for invariant evaluation.

    Prefer explicit reset metadata when present. This avoids letting ancient
    legacy `created_at` values keep old, already-reset runs permanently pinned
    into current money-conservation checks.
    """

    n_trades = int(port.get("n_trades", 0) or port.get("total_trades", 0) or 0)
    max_count = n_trades if n_trades > 0 else None

    for key in ("_reset_at", "last_panic_reset_at", "created_at"):
        dt = _parse_ts(port.get(key))
        if dt is not None:
            if key != "created_at":
                return dt
            break
    else:
        dt = None

    created_at = _parse_ts(port.get("created_at"))
    trade_times = _recent_spot_trade_times(TRADES, max_count=max_count)
    if created_at is not None and trade_times and 0 < n_trades < 20:
        first_trade = trade_times[0]
        if created_at < first_trade:
            return first_trade
    return created_at


def evaluate(port, price, *, persist=True):
    """Run all invariants against the live ledger + prior snapshot. Optionally persist the
    state file + a fresh snapshot for the next cross-cycle reconciliation. Returns
    (violations, state_dict). Used by both the CLI runner and the in-cycle agent hook.
    Never raises on I/O — persistence failures are swallowed so a check can't break a cycle."""
    trades = _trades_since_reset(port)
    prev = _read_json(SNAP, None)
    carry = _read_json(CARRY, None)
    violations = run_checks(port, trades, prev, cur_price=price, carry_state=carry) if price else []
    crit = [v for v in violations if v.severity == "critical"]
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "evaluated_at": now, "price": price, "ok": not violations,
        "n_critical": len(crit), "n_warning": len(violations) - len(crit),
        "violations": [{"name": v.name, "severity": v.severity, "detail": v.detail, "hint": v.hint}
                       for v in violations],
    }
    if persist:
        try:
            with open(STATE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
        if price:
            last_ts = max((str(t.get("ts", "")) for t in trades), default="")
            try:
                with open(SNAP, "w") as f:
                    json.dump({"equity": equity(port, price), "price": price,
                               "btc_qty": float(port.get("btc_qty", 0) or 0),
                               "last_trade_ts": last_ts, "ts": now}, f, indent=2)
            except Exception:
                pass
    return violations, state


def main():
    port = _read_json(PORT, {})
    price = _btc_price()
    violations, _ = evaluate(port, price)
    if not price:
        print("QuantForge Invariants: SKIPPED (no BTC price available)", flush=True)
        return 0
    crit = [v for v in violations if v.severity == "critical"]
    print("QuantForge Invariants: " + ("OK (money conserves)" if not violations
          else f"{len(crit)} CRITICAL / {len(violations) - len(crit)} warning"), flush=True)
    for v in violations:
        print(f"  [{v.severity.upper()}] {v.name}: {v.detail}", flush=True)
        if v.hint:
            print(f"      -> {v.hint}", flush=True)
    return 1 if crit else 0


if __name__ == "__main__":
    sys.exit(main())
