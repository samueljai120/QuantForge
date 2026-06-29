#!/usr/bin/env python3
"""QuantForge — Stage 4 allocator live-sanity readout (READ-ONLY observability).

The cross-strategy allocator (quantforge_allocator.py) now runs with auto-apply
ENABLED (allocator_auto_apply.flag present as of ~2026-06-15). Auto-apply means
the allocator can write bounded changes into qf_strategy_params.json without a
human in the loop. This script is the ongoing sanity readout that watches that
behavior — it is NOT a go/no-go gate, it confirms the allocator keeps acting
within its own discipline.

It reads allocator_decisions.jsonl (one JSON object per line, written by
quantforge_allocator.py) and summarizes the last N days (default 7):
  - current auto_apply state (from the most-recent record)
  - decision-bearing days vs null / no-op days
  - how many decisions were actually applied to params
  - which regimes were moved (and how often)
  - MAX per-day relative weight change vs the allocator's ±20% per-day cap,
    derived from row_before -> row_after on each decision
  - whether any weight landed on a hard bound (saturation)

Output: a human-readable summary + a one-line verdict
  STABLE   — decisions are within bounds and look sane
  REVIEW   — bounds are saturating, or the allocator is unusually active
  NO-DATA  — no decisions file / no records in window
and a small machine-readable JSON at allocator-readout.json.

READ-ONLY: this script never writes to any agent/reflect/allocator state file.
It only reads allocator_decisions.jsonl and writes its own readout JSON.

Designed to be foldable into daily-ceo-digest.py later (S7) via build_readout().
Stdlib only.
"""

import json
import os
import sys
from datetime import datetime, timezone

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
DECISIONS_FILE = os.path.join(DATA_DIR, "allocator_decisions.jsonl")
READOUT_FILE = os.path.join(DATA_DIR, "allocator-readout.json")

# Mirror of quantforge_allocator.py discipline. The allocator scales active
# weights by GROW=1.10 / SHRINK=0.80, i.e. at most a 20% per-weight relative
# move per day. We compare observed moves against that single cap.
PER_DAY_REL_CAP = 0.20

# Mirror of quantforge_allocator.BOUNDS — used only to detect when an applied
# weight is sitting on a hard edge (saturation). Read-only; never enforced here.
BOUNDS = {
    "spot_alloc_pct": (0.40, 0.85),
    "futures_weight": (0.0, 0.30),
    "mr_weight": (0.0, 0.50),
    "ml_scanner_weight": (0.0, 0.15),
    "funding_arb_weight": (0.0, 0.30),
}
# Float tolerance for "is this value on the bound" / "is this move at the cap".
EPS = 1e-6
CAP_NEAR_FRAC = 0.95  # >=95% of the cap counts as "near the cap" for REVIEW


def _parse_ts(value):
    """Parse an ISO ts to an aware UTC datetime, or None if unparseable."""
    if not value:
        return None
    s = str(value)
    try:
        # Python's fromisoformat handles the allocator's datetime.isoformat()
        # output (with offset). Normalize a trailing 'Z' just in case.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def load_records(path=DECISIONS_FILE):
    """Read all valid JSON records. Tolerates a missing/empty/partly-corrupt file."""
    records = []
    if not os.path.exists(path):
        return records
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    # skip a malformed line rather than aborting the readout
                    continue
    except Exception:
        return records
    return records


def _on_bound(key, value):
    """True if `value` sits on (or beyond) a hard BOUNDS edge for `key`."""
    bound = BOUNDS.get(key)
    if bound is None:
        return False
    lo, hi = bound
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v <= lo + EPS or v >= hi - EPS


def _row_moves(before, after):
    """Yield (key, before_val, after_val, rel_change) for each changed weight.

    rel_change is |after-before| / |before|. When before == 0 the relative
    change is undefined; we report None for it (still flag the absolute move).
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        return
    for key in after:
        if key not in before:
            continue
        try:
            b = float(before[key])
            a = float(after[key])
        except (TypeError, ValueError):
            continue
        if abs(a - b) <= EPS:
            continue
        rel = (abs(a - b) / abs(b)) if abs(b) > EPS else None
        yield key, b, a, rel


def analyze(records, days=7, now=None):
    """Summarize the last `days` of allocator decisions. Pure / testable."""
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - days * 86400

    in_window = []
    undated = []
    for rec in records:
        dt = _parse_ts(rec.get("ts"))
        if dt is None:
            undated.append(rec)
            continue
        if dt.timestamp() >= cutoff:
            in_window.append((dt, rec))
    in_window.sort(key=lambda x: x[0])

    # auto_apply state comes from the most-recent record overall (not just window)
    latest = None
    latest_dt = None
    for rec in records:
        dt = _parse_ts(rec.get("ts"))
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt, latest = dt, rec
    auto_apply_state = bool(latest.get("auto_apply")) if latest else None

    total = len(in_window)
    decision_days = 0
    noop_days = 0
    applied_count = 0
    seeded_count = 0
    regimes_moved = {}          # regime -> count
    max_rel_change = 0.0        # largest relative per-weight move seen
    max_rel_detail = None       # (regime, key, before, after, rel)
    saturated_keys = {}         # "regime.key" -> count of applied moves landing on bound
    moves = []                  # per-decision flat summary for JSON

    for dt, rec in in_window:
        if rec.get("seeded_table"):
            seeded_count += 1
        applied = bool(rec.get("applied"))
        if applied:
            applied_count += 1
        decision = rec.get("decision")
        if not decision or not isinstance(decision, dict):
            noop_days += 1
            continue
        decision_days += 1
        regime = decision.get("regime", "?")
        regimes_moved[regime] = regimes_moved.get(regime, 0) + 1

        before = decision.get("row_before")
        after = decision.get("row_after")
        decision_max_rel = 0.0
        changed_keys = []
        for key, b, a, rel in _row_moves(before, after):
            changed_keys.append(key)
            if rel is not None and rel > decision_max_rel:
                decision_max_rel = rel
            if rel is not None and rel > max_rel_change:
                max_rel_change = rel
                max_rel_detail = {
                    "regime": regime,
                    "key": key,
                    "before": round(b, 6),
                    "after": round(a, 6),
                    "rel_change": round(rel, 4),
                }
            # saturation only meaningful when the move actually took effect
            if applied and _on_bound(key, a):
                tag = f"{regime}.{key}"
                saturated_keys[tag] = saturated_keys.get(tag, 0) + 1

        moves.append({
            "ts": dt.isoformat(),
            "regime": regime,
            "action": decision.get("action"),
            "alpha_evidence": decision.get("alpha_evidence"),
            "applied": applied,
            "changed_keys": changed_keys,
            "max_rel_change": round(decision_max_rel, 4),
        })

    # Verdict
    cap_breached = max_rel_change > PER_DAY_REL_CAP + EPS
    near_cap = max_rel_change >= PER_DAY_REL_CAP * CAP_NEAR_FRAC
    # "unusually active": more than one decision per ~2 days in the window
    unusually_active = decision_days > max(1, days // 2)

    if total == 0:
        verdict = "NO-DATA"
        verdict_reason = "No allocator decisions in the window."
    elif cap_breached or saturated_keys or near_cap or unusually_active:
        verdict = "REVIEW"
        reasons = []
        if cap_breached:
            reasons.append(f"per-day move {max_rel_change:.1%} exceeds {PER_DAY_REL_CAP:.0%} cap")
        elif near_cap:
            reasons.append(f"per-day move {max_rel_change:.1%} near {PER_DAY_REL_CAP:.0%} cap")
        if saturated_keys:
            reasons.append(f"{len(saturated_keys)} weight(s) saturating a hard bound")
        if unusually_active:
            reasons.append(f"{decision_days} decisions in {days}d (active)")
        verdict_reason = "; ".join(reasons)
    else:
        verdict = "STABLE"
        verdict_reason = "Decisions within bounds and cadence; allocator looks sane."

    return {
        "generated_at": now.isoformat(),
        "window_days": days,
        "auto_apply": auto_apply_state,
        "latest_decision_ts": latest_dt.isoformat() if latest_dt else None,
        "records_total_all_time": len(records),
        "records_undated_skipped": len(undated),
        "records_in_window": total,
        "decision_days": decision_days,
        "noop_days": noop_days,
        "applied_count": applied_count,
        "seeded_count": seeded_count,
        "regimes_moved": regimes_moved,
        "per_day_rel_cap": PER_DAY_REL_CAP,
        "max_rel_change": round(max_rel_change, 4),
        "max_rel_change_detail": max_rel_detail,
        "saturated_bounds": saturated_keys,
        "moves": moves,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def build_readout(days=7, path=None, now=None):
    """Foldable entry point for daily-ceo-digest (S7). Returns the summary dict.

    `path` resolves to the module-level DECISIONS_FILE at call time (not at
    def time) so callers/tests can repoint DECISIONS_FILE before invoking.
    """
    if path is None:
        path = DECISIONS_FILE
    return analyze(load_records(path), days=days, now=now)


def format_human(summary):
    """Render the summary dict as a human-readable block of text."""
    lines = []
    lines.append("QuantForge Allocator — live-sanity readout")
    lines.append(f"  window:            last {summary['window_days']} day(s)")
    aa = summary["auto_apply"]
    aa_str = "ON (auto-apply enabled)" if aa else ("OFF (training wheels)" if aa is False else "unknown")
    lines.append(f"  auto_apply:        {aa_str}")
    lines.append(f"  latest decision:   {summary['latest_decision_ts'] or 'n/a'}")
    lines.append(
        f"  records:           {summary['records_in_window']} in window "
        f"({summary['records_total_all_time']} all-time"
        + (f", {summary['records_undated_skipped']} undated skipped" if summary['records_undated_skipped'] else "")
        + ")"
    )
    lines.append(
        f"  decision days:     {summary['decision_days']} "
        f"(no-op/null: {summary['noop_days']}, applied: {summary['applied_count']}, "
        f"seeded: {summary['seeded_count']})"
    )
    if summary["regimes_moved"]:
        moved = ", ".join(f"{r}x{n}" if n > 1 else r for r, n in sorted(summary["regimes_moved"].items()))
        lines.append(f"  regimes moved:     {moved}")
    else:
        lines.append("  regimes moved:     none")

    cap = summary["per_day_rel_cap"]
    mx = summary["max_rel_change"]
    detail = summary["max_rel_change_detail"]
    if detail:
        lines.append(
            f"  max per-day move:  {mx:.1%}  (cap {cap:.0%})  "
            f"[{detail['regime']}.{detail['key']}: {detail['before']} -> {detail['after']}]"
        )
    else:
        lines.append(f"  max per-day move:  {mx:.1%}  (cap {cap:.0%})  [no weight moves]")

    if summary["saturated_bounds"]:
        sat = ", ".join(f"{k} (x{n})" if n > 1 else k for k, n in sorted(summary["saturated_bounds"].items()))
        lines.append(f"  bound saturation:  {sat}")
    else:
        lines.append("  bound saturation:  none")

    lines.append(f"  VERDICT: {summary['verdict']} — {summary['verdict_reason']}")
    return "\n".join(lines)


def _parse_days_arg(argv):
    days = 7
    for arg in argv[1:]:
        if arg.startswith("--days="):
            arg = arg.split("=", 1)[1]
        if arg.startswith("-"):
            continue
        try:
            n = int(arg)
            if n > 0:
                days = n
        except ValueError:
            continue
    return days


def main(argv=None):
    argv = sys.argv if argv is None else argv
    days = _parse_days_arg(argv)
    summary = build_readout(days=days)

    print(format_human(summary))

    # Write the machine-readable readout (best-effort; never fatal).
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = READOUT_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, READOUT_FILE)
        print(f"  readout written:   {READOUT_FILE}")
    except Exception as exc:
        print(f"  (could not write readout JSON: {exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
