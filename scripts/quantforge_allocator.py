#!/usr/bin/env python3
"""QuantForge — Stage 4 cross-strategy capital allocator (meta-controller).

Routes capital toward what wins, per regime, using the agent's own
`regime_perf` attribution as evidence. Acts ONLY through the bounded
`regime_weight_table` tunable in qf_strategy_params.json — the same governed
channel the reflection daemon uses. It cannot touch safety parameters, cannot
exceed the agent's per-key weight bounds (enforced again agent-side), and
follows the reflect-daemon discipline:

  - max ONE regime row changed per UTC day
  - max 20% relative change per weight per day
  - minimum evidence before acting (>=48h and >=24 visits in the regime)
  - every proposal logged to allocator_decisions.jsonl with full reasoning
  - training wheels: PROPOSE-ONLY until allocator_auto_apply.flag exists

evidence rule (per regime, vs passive HODL):
  - negative alpha  -> scale active weights (mr/futures/ml/funding) x0.80 and
    pull spot allocation 20% of the way toward the HODL baseline (0.65)
  - positive alpha  -> scale active weights x1.10 (bounded by per-key caps)
Only the eligible regime with the largest |alpha| moves each day.
"""

import json
import os
import sys
import fcntl
from datetime import datetime, timezone

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "agent_portfolio.json")
QF_PARAMS_FILE = os.path.join(DATA_DIR, "qf_strategy_params.json")
DECISIONS_FILE = os.path.join(DATA_DIR, "allocator_decisions.jsonl")
AUTO_APPLY_FLAG = os.path.join(DATA_DIR, "allocator_auto_apply.flag")
LOCK_FILE = os.path.join(DATA_DIR, "allocator.lock")

REGIMES = ("STRONG_BEAR", "BEAR", "CHOP", "NEUTRAL", "BULL", "STRONG_BULL")
ACTIVE_KEYS = ("mr_weight", "futures_weight", "ml_scanner_weight", "funding_arb_weight")

# Must mirror the agent's _TABLE_BOUNDS — the agent re-clamps on load anyway,
# this is defense in depth.
BOUNDS = {
    "spot_alloc_pct": (0.40, 0.85),
    "futures_weight": (0.0, 0.30),
    "mr_weight": (0.0, 0.50),
    "ml_scanner_weight": (0.0, 0.15),
    "funding_arb_weight": (0.0, 0.30),
}

HODL_BASELINE_SPOT = 0.65
MIN_HOURS = 48.0          # evidence floor per regime
MIN_VISITS = 24
SHRINK = 0.80             # negative-alpha regimes: active weights x0.80 (max -20%/day)
GROW = 1.10               # positive-alpha regimes: active weights x1.10 (max +10%/day)
SPOT_PULL = 0.20          # negative-alpha: pull spot 20% of gap toward baseline


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def clamp(key, value):
    lo, hi = BOUNDS[key]
    return min(max(float(value), lo), hi)


def already_ran_today() -> bool:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with open(DECISIONS_FILE) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if str(row.get("ts", ""))[:10] == today:
                    return True
    except FileNotFoundError:
        pass
    return False


def current_weights(params: dict) -> dict:
    """Effective flat weights from params (reflect-daemon output) or agent defaults."""
    return {
        "spot_alloc_pct": clamp("spot_alloc_pct", params.get("fixed_alloc_pct", HODL_BASELINE_SPOT)),
        "mr_weight": clamp("mr_weight", params.get("mr_weight", 0.12)),
        "futures_weight": clamp("futures_weight", params.get("futures_weight", 0.05)),
        "ml_scanner_weight": clamp("ml_scanner_weight", params.get("ml_scanner_weight", 0.10)),
        "funding_arb_weight": clamp("funding_arb_weight", params.get("funding_arb_weight", 0.01)),
    }


def seed_table(params: dict) -> dict:
    """Behavior-neutral seed: every regime row = current effective weights.

    Mirrors the Stage 2 discipline — the allocator's first write changes
    nothing until evidence moves a single row.
    """
    base = current_weights(params)
    return {r: dict(base) for r in REGIMES}


def propose(table: dict, regime_perf: dict) -> dict | None:
    """Pick the single eligible regime with the strongest |alpha| evidence and
    produce a bounded adjustment for its row. Returns a decision dict or None."""
    candidates = []
    for regime in REGIMES:
        perf = regime_perf.get(regime) or {}
        hours = float(perf.get("hours", 0.0) or 0.0)
        visits = int(perf.get("visits", 0) or 0)
        alpha = float(perf.get("alpha", (perf.get("our_pnl", 0.0) or 0.0) - (perf.get("hodl_pnl", 0.0) or 0.0)))
        if hours < MIN_HOURS or visits < MIN_VISITS:
            continue
        candidates.append((abs(alpha), alpha, regime, hours, visits))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, alpha, regime, hours, visits = candidates[0]

    row = dict(table.get(regime) or {})
    if not row:
        return None
    before = dict(row)
    factor = SHRINK if alpha < 0 else GROW
    for key in ACTIVE_KEYS:
        if key in row:
            row[key] = round(clamp(key, float(row[key]) * factor), 4)
    if alpha < 0 and "spot_alloc_pct" in row:
        spot = float(row["spot_alloc_pct"])
        row["spot_alloc_pct"] = round(clamp("spot_alloc_pct", spot + (HODL_BASELINE_SPOT - spot) * SPOT_PULL), 4)

    if row == before:
        return None  # bounds already saturated — nothing to change

    direction = "shrink_active_toward_hodl" if alpha < 0 else "grow_active"
    return {
        "regime": regime,
        "action": direction,
        "alpha_evidence": round(alpha, 2),
        "hours": round(hours, 1),
        "visits": visits,
        "row_before": before,
        "row_after": row,
        "reasoning": (
            f"{regime}: alpha {alpha:+.2f} over {hours:.0f}h/{visits} visits vs passive HODL. "
            + ("Active strategies are losing to passive here — cut their weights 20% and pull spot toward baseline."
               if alpha < 0 else
               "Active strategies are beating passive here — grow their weights 10% within bounds.")
        ),
    }


def apply_to_params(params: dict, table: dict) -> dict:
    out = dict(params)
    out["regime_adaptive"] = True
    out["regime_weight_table"] = table
    out["_last_modified_by"] = "quantforge_allocator"
    out["_allocator_updated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("SKIP: allocator already running")
        return 0

    if already_ran_today():
        print("SKIP: allocator already decided today (max 1 change/day)")
        return 0

    port = read_json(PORTFOLIO_FILE)
    regime_perf = port.get("regime_perf") or {}
    if not regime_perf:
        print("SKIP: no regime_perf attribution yet")
        return 0

    params = read_json(QF_PARAMS_FILE)
    table = params.get("regime_weight_table")
    seeded = False
    if not isinstance(table, dict) or not all(r in table for r in REGIMES):
        table = seed_table(params)
        seeded = True

    decision = propose(table, regime_perf)
    auto_apply = os.path.exists(AUTO_APPLY_FLAG)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "seeded_table": seeded,
        "auto_apply": auto_apply,
        "decision": decision,
    }

    applied = False
    if decision:
        table[decision["regime"]] = decision["row_after"]
    if auto_apply and (decision or seeded):
        new_params = apply_to_params(params, table)
        tmp = QF_PARAMS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(new_params, f, indent=2)
        os.replace(tmp, QF_PARAMS_FILE)
        applied = True
    record["applied"] = applied

    with open(DECISIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    print("QuantForge Allocator (Stage 4)")
    if seeded:
        print("  Seeded behavior-neutral regime_weight_table from current weights.")
    if decision:
        print(f"  {decision['regime']}: {decision['action']}  (alpha {decision['alpha_evidence']:+.2f} over {decision['hours']}h)")
        print(f"  {decision['reasoning']}")
    else:
        print("  No regime has sufficient evidence (or bounds saturated) — no change proposed.")
    print(f"  Applied to params: {applied} ({'auto-apply ON' if auto_apply else 'training wheels — propose only'})")
    print(f"  Decision logged: {DECISIONS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
