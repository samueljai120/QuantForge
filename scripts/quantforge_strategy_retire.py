#!/usr/bin/env python3
"""QuantForge — strategy auto-retirement system.

audit(): scans agent_trades.jsonl for tagged strategy PnL, falling back
         to agent.log cycle-end lines + strategy presence to estimate
         win rate, age, and cumulative PnL per strategy.

retire(): sets weight to 0 in qf_strategy_params.json for strategies
          with WR < 45% for 30+ days OR negative cumulative PnL for 20+ days.
          Adds retired name to 'retired_strategies' list in params.

Minimal, self-contained. No external dependencies beyond stdlib.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
TRADES_FILE = os.path.join(DATA_DIR, "agent_trades.jsonl")
LOG_FILE = os.path.join(DATA_DIR, "agent.log")
PARAMS_FILE = os.path.join(DATA_DIR, "qf_strategy_params.json")


# ── Strategies we know about ─────────────────────────────────────────
BASE_STRATEGIES = [
    "hodl",
    "mean_revert_chop",
    "futures_lane",
    "liquidation_dip",
    "funding_mr",
    "cvd_momentum",
    "vol_breakout",
    "cross_asset",
    "liq_scalp",
    "oi_divergence",
    "ml_scanner",
]

# Strategies with weight keys in qf_strategy_params.json
PARAMS_WEIGHT_KEYS = {
    "mean_revert_chop": "mr_weight",
    "futures_lane": "futures_weight",
    "ml_scanner": "ml_scanner_weight",
    "funding_mr": "funding_arb_weight",
}

# Strategies with hardcoded weights in _rebuild_strategy_registry()
HARDCODED_WEIGHTS = {
    "liquidation_dip": 0.03,
    "funding_mr": 0.01,
    "cvd_momentum": 0.01,
    "vol_breakout": 0.01,
    "cross_asset": 0.01,
    "liq_scalp": 0.01,
    "oi_divergence": 0.01,
}


def _parse_log_cycles(log_path: str) -> list[dict]:
    """Parse agent.log into a list of cycle dicts.

    Each dict: {ts, equity, pnl, regime, strategies: {name: {weight, notes}}}
    """
    cycles = []
    current_cycle = None

    if not os.path.exists(log_path):
        return cycles

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ts_match = re.match(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2})\]\s+(.*)", line)
            if not ts_match:
                continue

            ts_str = ts_match.group(1)
            body = ts_match.group(2)

            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            # Cycle start
            if "=== Agent cycle start ===" in body:
                current_cycle = {"ts": ts, "equity": None, "pnl": 0.0, "regime": "", "strategies": {}}
                continue

            if current_cycle is None:
                continue

            # Strategy lines: "  Strategy 'hodl' (weight 74%): REGIME_ACTIVE target=80% regime=STRONG_BULL"
            strat_match = re.match(r"^\s*Strategy\s+'(\S+)'\s+\(weight\s+([\d.]+)%\):\s+(.*)", body)
            if strat_match:
                name = strat_match.group(1)
                weight = float(strat_match.group(2)) / 100.0
                notes = strat_match.group(3)
                current_cycle["strategies"][name] = {
                    "weight": weight,
                    "notes": notes,
                }
                # track strategy names we discover (auto-generated: *_genN)
                if name not in BASE_STRATEGIES and name not in PARAMS_WEIGHT_KEYS:
                    pass  # unknown strategy — will be discovered
                continue

            # Cycle end line: "=== Cycle end. Equity $9,635.01  PnL $+4635.01 (+92.70%)  Regime STRONG_BULL  Trades total 43 ==="
            end_match = re.match(
                r"=== Cycle end\.\s+Equity\s+\$([\d,.]+)\s+PnL\s+\$([+\-][\d,.]+)\s+\([^)]+\)\s+Regime\s+(\S+)\s+Trades\s+total\s+\d+\s+===",
                body,
            )
            if end_match:
                try:
                    current_cycle["equity"] = float(end_match.group(1).replace(",", ""))
                    current_cycle["pnl"] = float(end_match.group(2).replace(",", ""))
                    current_cycle["regime"] = end_match.group(3)
                    cycles.append(current_cycle)
                except (ValueError, IndexError):
                    pass
                current_cycle = None

    return cycles


def _parse_trades(trades_path: str) -> list[dict]:
    """Parse agent_trades.jsonl and return list of trade dicts."""
    trades = []
    if not os.path.exists(trades_path):
        return trades
    with open(trades_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return trades


def _is_strategy_active(notes: str) -> bool:
    """A strategy is 'active' if it's not explicitly INACTIVE/passive."""
    if not notes:
        return True
    notes_upper = notes.upper()
    # Inactive signals
    if "INACTIVE" in notes_upper:
        return False
    if notes.startswith("LIQ_DIP watching"):  # liquidation_dip is always watching
        return False
    # ml_scanner is active if it's not just watching
    if notes.startswith("ML_SCANNER active"):
        return True
    if "no picks" in notes:
        return False
    return True


def _estimate_strategy_pnl(cycles: list[dict], strategy_name: str) -> tuple[float, int, int, float]:
    """Estimate cumulative PnL, wins, total active cycles, and win rate for a strategy.

    For each cycle where the strategy was active, we attribute a share of
    the cycle-level equity change proportional to its weight among active strategies.

    Returns (cumulative_pnl, wins, active_cycles, win_rate).
    """
    cumulative = 0.0
    wins = 0
    active_cycles = 0

    for i, cycle in enumerate(cycles):
        if strategy_name not in cycle["strategies"]:
            continue

        strat_info = cycle["strategies"][strategy_name]
        if not _is_strategy_active(strat_info.get("notes", "")):
            continue

        # Compute equity change from this cycle to next
        if i + 1 >= len(cycles):
            continue
        next_cycle = cycles[i + 1]
        if cycle["equity"] is None or next_cycle["equity"] is None:
            continue

        equity_delta = next_cycle["equity"] - cycle["equity"]

        # Compute total active weight in this cycle
        total_active_weight = sum(
            s["weight"]
            for name, s in cycle["strategies"].items()
            if _is_strategy_active(s.get("notes", ""))
        ) or 1.0

        # Proportional share
        share = strat_info["weight"] / max(total_active_weight, 0.001)
        attributed = equity_delta * share

        cumulative += attributed
        if attributed > 0:
            wins += 1
        active_cycles += 1

    wr = (wins / active_cycles) if active_cycles > 0 else 0.0
    return cumulative, wins, active_cycles, wr


def _compute_from_trade_tags(trades: list[dict]) -> dict[str, dict]:
    """Attempt to compute per-strategy stats from trade tags.

    Looks for strategy names in trade 'reason' or 'strategy' fields.
    Falls back to empty dict if no tags found.
    """
    per_strat = {}
    for trade in trades:
        # Try explicit strategy tag
        strat_tag = trade.get("strategy", "")
        # Also check reason for strategy names
        reason = trade.get("reason", "")

        matched = None
        for sname in BASE_STRATEGIES:
            if sname in strat_tag or sname in reason:
                matched = sname
                break

        if not matched:
            continue

        if matched not in per_strat:
            per_strat[matched] = {"pnl": 0.0, "wins": 0, "total": 0, "first_ts": trade.get("ts", "")}

        pnl = trade.get("pnl_usd", 0.0) or 0.0
        per_strat[matched]["pnl"] += pnl
        per_strat[matched]["total"] += 1
        if pnl > 0:
            per_strat[matched]["wins"] += 1
        if trade.get("ts", "") < per_strat[matched].get("first_ts", "z"):
            per_strat[matched]["first_ts"] = trade["ts"]

    # Compute WR
    for sname, sdata in per_strat.items():
        sdata["wr"] = (sdata["wins"] / sdata["total"]) if sdata["total"] > 0 else 0.0

    return per_strat


def _discover_strategies(cycles: list[dict]) -> list[str]:
    """Discover all strategy names from log cycles."""
    all_names = set(BASE_STRATEGIES)
    for cycle in cycles:
        for name in cycle.get("strategies", {}):
            all_names.add(name)
    return sorted(all_names)


def _compute_age_days(strategy_name: str, cycles: list[dict]) -> float:
    """Compute age of a strategy in days based on first log appearance."""
    first_ts = None
    for cycle in cycles:
        if strategy_name in cycle.get("strategies", {}):
            first_ts = cycle["ts"]
            break
    if first_ts is None:
        return 0.0
    now = datetime.now(timezone.utc)
    return (now - first_ts).total_seconds() / 86400.0


def audit() -> list[dict]:
    """Print audit table and return results list.

    Returns list of dicts with keys: name, wr, age_days, cumulative_pnl,
    active_cycles, active (bool).
    """
    cycles = _parse_log_cycles(LOG_FILE)
    trades = _parse_trades(TRADES_FILE)

    # First try trade-tag-based computation
    trade_stats = _compute_from_trade_tags(trades)

    # Discover all strategies
    strategies = _discover_strategies(cycles)

    results = []

    print()
    print("=" * 80)
    print("  QuantForge Strategy Audit — Auto-Retirement Eligibility Check")
    print("=" * 80)
    print(f"  {'Strategy':<22} {'WR':>7} {'Age(d)':>8} {'Cum PnL':>10} {'ActiveCyc':>10} {'Status':>10}")
    print("  " + "-" * 78)

    for sname in strategies:
        age_days = _compute_age_days(sname, cycles)

        if sname in trade_stats:
            # Use trade-tag data
            ts = trade_stats[sname]
            wr = ts["wr"]
            cumulative_pnl = ts["pnl"]
            active_cycles = ts.get("total", 0)
        else:
            # Estimate from log cycles
            cumulative_pnl, wins, active_cycles, wr = _estimate_strategy_pnl(cycles, sname)

        is_active = True
        status = "active"
        retire_reason = ""

        if wr < 0.45 and age_days >= 30:
            status = "RETIRE"
            is_active = False
            retire_reason = f"WR {wr:.1%} < 45% for {age_days:.0f}d"
        elif cumulative_pnl < 0 and age_days >= 20:
            status = "RETIRE"
            is_active = False
            retire_reason = f"neg PnL ${cumulative_pnl:,.2f} for {age_days:.0f}d"
        elif wr < 0.45:
            status = "WATCH"
            retire_reason = f"WR {wr:.1%} < 45% (age {age_days:.0f}d, need 30d)"
        elif cumulative_pnl < 0:
            status = "WATCH"
            retire_reason = f"neg PnL ${cumulative_pnl:,.2f} (age {age_days:.0f}d, need 20d)"

        print(f"  {sname:<22} {wr:>6.1%} {age_days:>7.1f} "
              f"${cumulative_pnl:>9,.2f} {active_cycles:>10} {status:>10}")
        if retire_reason:
            print(f"    ↳ {retire_reason}")

        results.append({
            "name": sname,
            "wr": round(wr, 4),
            "age_days": round(age_days, 1),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "active_cycles": active_cycles,
            "active": is_active,
            "status": status,
            "retire_reason": retire_reason,
        })

    print("=" * 80)
    retireable = [r for r in results if r["status"] == "RETIRE"]
    if retireable:
        print(f"    {len(retireable)} strategies eligible for retirement:")
        for r in retireable:
            print(f"     • {r['name']}: {r['retire_reason']}")
    else:
        print(f"   All {len(results)} strategies healthy — no retirements needed")
    print()

    return results


def retire(dry_run: bool = True) -> list[str]:
    """Retire strategies that fail criteria. Returns list of retired names.

    If dry_run=True, prints what WOULD happen but doesn't modify files.
    """
    results = audit()
    retireable = [r for r in results if r["status"] == "RETIRE"]

    if not retireable:
        print("  No strategies to retire.")
        return []

    retired_names = [r["name"] for r in retireable]

    if dry_run:
        print(f"\n  [DRY RUN] Would retire: {', '.join(retired_names)}")
        print(f"  [DRY RUN] No files modified. Run with 'retire' to apply.\n")
        return retired_names

    # Load params
    params = {}
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE) as f:
                params = json.load(f)
        except Exception as e:
            print(f"    Could not load {PARAMS_FILE}: {e}")
            return []

    # Load retired list
    retired_list = params.get("retired_strategies", [])
    if isinstance(retired_list, list):
        retired_set = set(retired_list)
    else:
        retired_set = set()

    modified = False

    for rname in retired_names:
        # Set individual weight key to 0 if it exists
        key = PARAMS_WEIGHT_KEYS.get(rname)
        if key and key in params:
            old = params[key]
            params[key] = 0.0
            print(f"   Retired {rname}: {key} {old} → 0.0")
            modified = True
        elif key and rname not in PARAMS_WEIGHT_KEYS:
            # auto-generated strategy — might have its own weight key
            gen_key = f"{rname}_weight"
            if gen_key in params:
                old = params[gen_key]
                params[gen_key] = 0.0
                print(f"   Retired {rname}: {gen_key} {old} → 0.0")
                modified = True

        # Zero out in regime_weight_table
        rwt = params.get("regime_weight_table", {})
        table_modified = False
        for regime, weights in rwt.items():
            if not isinstance(weights, dict):
                continue
            # Map strategy name to table key
            table_key_map = {
                "mean_revert_chop": "mr_weight",
                "futures_lane": "futures_weight",
                "ml_scanner": "ml_scanner_weight",
                "funding_mr": "funding_arb_weight",
                "hodl": "spot_alloc_pct",
            }
            tk = table_key_map.get(rname)
            if tk and tk in weights:
                old_val = weights[tk]
                weights[tk] = 0.0
                table_modified = True
            # auto strategies might have their own keys
            gen_tk = f"{rname}_weight"
            if gen_tk in weights:
                old_val = weights[gen_tk]
                weights[gen_tk] = 0.0
                table_modified = True

        if table_modified:
            print(f"   Zeroed {rname} in regime_weight_table")
            modified = True

        # Add to retired list
        if rname not in retired_set:
            retired_set.add(rname)
            retired_list.append(rname)
            modified = True

    # Update params
    params["retired_strategies"] = sorted(retired_list)
    params["_retire_modified_at"] = datetime.now(timezone.utc).isoformat()
    params["_retire_modified_by"] = "quantforge_strategy_retire"

    if modified:
        # Backup
        backup_path = PARAMS_FILE + ".bak"
        try:
            import shutil
            shutil.copy2(PARAMS_FILE, backup_path)
            print(f"   Backup saved: {backup_path}")
        except Exception as e:
            print(f"    Could not backup: {e}")

        with open(PARAMS_FILE, "w") as f:
            json.dump(params, f, indent=2)
        print(f"\n   Retired {len(retired_names)} strategies, params updated: {PARAMS_FILE}")
    else:
        print(f"  ℹ  No params changes needed (weights may be hardcoded in source)")

    # For hardcoded strategies, note them
    hardcoded_retired = [r for r in retired_names if r in HARDCODED_WEIGHTS]
    if hardcoded_retired:
        print(f"\n    {len(hardcoded_retired)} retired strategies have hardcoded weights in quantforge_agent.py:")
        for hr in hardcoded_retired:
            print(f"     • {hr}: weight {HARDCODED_WEIGHTS[hr]} hardcoded in _rebuild_strategy_registry()")
        print(f"     → These need manual code removal or a params-based override guard.")

    return retired_names


def trigger_alert(retired_names: list[str]) -> None:
    """Write alert agent trigger file so LLM review is queued."""
    trigger_file = os.path.expanduser(
        os.environ.get("QF_ALERT_TRIGGER_FILE", "~/.quantforge/alert_trigger.json")
    )
    os.makedirs(os.path.dirname(trigger_file), exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reasons": [f"strategy_retired:{','.join(retired_names)}"],
        "consumed": False,
        "cooldown_h": 0.0,
        "source": "quantforge_strategy_retire",
    }
    with open(trigger_file, "w") as f:
        json.dump(payload, f)
    print(f"   alert agent monitor trigger written for retirement review")


# ── Wire-in helpers for quantforge_agent.py ──────────────────────────

def check_and_retire(dry_run: bool = False) -> tuple[list[dict], list[str]]:
    """Audit + retire in one call. Returns (audit_results, retired_names)."""
    results = audit()
    retiring = [r for r in results if r["status"] == "RETIRE"]
    if retiring:
        retired = retire(dry_run=dry_run)
        if not dry_run and retired:
            trigger_alert(retired)
        return results, retired
    return results, []


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "audit"

    if cmd == "audit":
        audit()
    elif cmd == "retire":
        dry_run = "--apply" not in sys.argv
        retired = retire(dry_run=dry_run)
        if not dry_run and retired:
            trigger_alert(retired)
    elif cmd == "check":
        # audit + retire (dry run), returns exit code 1 if retirements found
        results, retired = check_and_retire(dry_run=True)
        if retired:
            sys.exit(1)
    elif cmd == "check-and-retire":
        results, retired = check_and_retire(dry_run=False)
        if retired:
            sys.exit(1)
    else:
        print(f"Usage: {sys.argv[0]} [audit|retire|check|check-and-retire]")
        print(f"  audit            — print strategy audit table")
        print(f"  retire           — dry-run retirement (add --apply to execute)")
        print(f"  check            — audit + dry-run retire, exit 1 if retirements found")
        print(f"  check-and-retire — audit + execute retirement")
        sys.exit(1)
