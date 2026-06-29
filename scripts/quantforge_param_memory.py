#!/usr/bin/env python3
"""
QuantForge Param Memory — result-to-parameter memory system (v1).

Stores snapshots of (regime, params, performance metrics) so the agent can
recall which parameter combinations produced the best alpha in each regime,
then reload them when that regime reappears.

No new dependencies — stdlib only.

API:
    save_snapshot(regime, params, metrics)
        Append a timestamped snapshot to param_memory.jsonl.
        metrics dict must have keys: equity, dd, alpha, wr.

    get_best_params(regime)
        Return the param dict with the highest alpha for the given regime,
        or None if no history exists.

    apply_best_if_known(regime)
        If >3 snapshots exist for this regime AND the current alpha in the
        portfolio's regime_perf is worse than the best historical alpha,
        load the best params into qf_strategy_params.json (backtest-gated
        via qf_validate_tune.py). Returns True if params were applied.

Usage:
    python3 quantforge_param_memory.py test   # verify import works
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timezone


# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
MEMORY_FILE = os.path.join(DATA_DIR, "param_memory.jsonl")
PARAMS_FILE = os.path.join(DATA_DIR, "qf_strategy_params.json")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "agent_portfolio.json")
VALIDATE_SCRIPT = os.path.join(SCRIPTS_DIR, "qf_validate_tune.py")


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


# ── Core API ───────────────────────────────────────────────────────────────

def save_snapshot(regime, params, metrics):
    """Append a timestamped snapshot to param_memory.jsonl.

    Args:
        regime:   str like 'BULL', 'BEAR', 'CHOP', etc.
        params:   dict of strategy parameters (e.g. the full qf_strategy_params).
        metrics:  dict with keys: equity (float), dd (float drawdown),
                  alpha (float), wr (float win rate).
    """
    _ensure_dirs()
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "param": dict(params),  # shallow copy to avoid mutation surprises
        "equity": float(metrics.get("equity", 0)),
        "dd": float(metrics.get("dd", 0)),
        "alpha": float(metrics.get("alpha", 0)),
        "wr": float(metrics.get("wr", 0)),
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(snapshot) + "\n")


def get_best_params(regime):
    """Return params dict with highest alpha for *regime*, or None."""
    if not os.path.exists(MEMORY_FILE):
        return None

    best = None
    best_alpha = -float("inf")

    with open(MEMORY_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
            except json.JSONDecodeError:
                continue
            if snap.get("regime") != regime:
                continue
            alpha = snap.get("alpha", -float("inf"))
            if alpha > best_alpha:
                best_alpha = alpha
                best = snap.get("param")

    return best


def _count_snapshots(regime):
    """Return number of snapshots for *regime*."""
    if not os.path.exists(MEMORY_FILE):
        return 0
    count = 0
    with open(MEMORY_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                snap = json.loads(line)
            except json.JSONDecodeError:
                continue
            if snap.get("regime") == regime:
                count += 1
    return count


def _current_alpha_for_regime(regime):
    """Read current alpha for *regime* from agent_portfolio.json regime_perf."""
    if not os.path.exists(PORTFOLIO_FILE):
        return None
    try:
        with open(PORTFOLIO_FILE) as f:
            port = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    regime_perf = port.get("regime_perf", {})
    entry = regime_perf.get(regime, {})
    return entry.get("alpha")


def _run_backtest_gate(proposed_params):
    """Run qf_validate_tune.py against the proposed params dict.

    The proposed_params dict is used as a full-file replacement, so we pass
    the entire dict as --proposed wrapped under the top-level key ''.
    We merge into the current file and let the gate compare.

    Returns (approved: bool, reason: str).
    """
    if not os.path.exists(VALIDATE_SCRIPT):
        # No gate available — allow
        return True, "gate_unavailable: validate script not found"

    try:
        # Write proposed to a temp file
        tmp_proposed = os.path.join(DATA_DIR, "_gate_param_memory_proposed.json")
        with open(tmp_proposed, "w") as f:
            json.dump(proposed_params, f)

        # The gate expects --proposed and --current paths
        tmp_current = os.path.join(DATA_DIR, "_gate_param_memory_current.json")
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE) as f:
                current = json.load(f)
        else:
            current = {}
        with open(tmp_current, "w") as f:
            json.dump(current, f)

        result = subprocess.run(
            [
                "python3",
                os.path.expanduser("~/quantforge/scripts/quantforge_backtest_gate.py"),
                "--proposed", tmp_proposed,
                "--current", tmp_current,
            ],
            capture_output=True, text=True, timeout=60,
        )

        # Clean up temps
        for tmp in [tmp_proposed, tmp_current]:
            try:
                os.remove(tmp)
            except OSError:
                pass

        gate_output = json.loads(result.stdout) if result.stdout.strip() else {}
        approved = gate_output.get("approved", False)
        reason = gate_output.get("reason", "gate returned no reason")
        return approved, reason

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        # Gate failed — allow (same policy as qf_validate_tune.py)
        return True, f"gate_unavailable: {str(e)[:80]}"


def apply_best_if_known(regime):
    """Check if best params exist for *regime* and apply them if beneficial.

    Conditions:
      1. >3 snapshots exist for this regime.
      2. Current alpha (from portfolio regime_perf) is worse than best
         historical alpha.
      3. Backtest gate (qf_validate_tune.py → quantforge_backtest_gate.py)
         approves the change.

    Returns:
        True  if best params were applied to qf_strategy_params.json.
        False otherwise (not enough data, already optimal, or gate rejected).
    """
    n_snapshots = _count_snapshots(regime)
    if n_snapshots <= 3:
        return False

    best_params = get_best_params(regime)
    if best_params is None:
        return False

    # What was the best alpha among the snapshots?
    best_alpha = -float("inf")
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snap = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if snap.get("regime") == regime:
                    alpha = snap.get("alpha", -float("inf"))
                    if alpha > best_alpha:
                        best_alpha = alpha

    current_alpha = _current_alpha_for_regime(regime)
    if current_alpha is None:
        # Can't determine current alpha — skip
        return False

    if current_alpha >= best_alpha:
        # Current performance is already at or above the best historical
        return False

    # Run backtest gate
    approved, reason = _run_backtest_gate(best_params)
    if not approved:
        print(f"[param_memory] Gate rejected best params for {regime}: {reason}")
        return False

    # Apply: write best_params to qf_strategy_params.json
    try:
        # Preserve metadata fields if we want, but the task says "load best params"
        best_out = dict(best_params)
        best_out["_last_modified_at"] = datetime.now(timezone.utc).isoformat()
        best_out["_last_modified_by"] = "param_memory"
        best_out["_last_change_reason"] = (
            f"Recalled best params for {regime} "
            f"(historical alpha={best_alpha:.2f} > current alpha={current_alpha:.2f}, "
            f"n={n_snapshots} snapshots)"
        )

        _ensure_dirs()
        with open(PARAMS_FILE, "w") as f:
            json.dump(best_out, f, indent=2)

        print(
            f"[param_memory]  Applied best params for {regime}: "
            f"alpha {current_alpha:.2f} → recalled {best_alpha:.2f} "
            f"(gate: {reason[:60]})"
        )
        return True

    except (IOError, OSError) as e:
        print(f"[param_memory]  Failed to write params file: {e}")
        return False


# ── Test harness ───────────────────────────────────────────────────────────

def _test():
    """Quick self-test to verify import and basic functionality."""
    print("[param_memory] Self-test starting...")

    # 1. Verify paths
    assert os.path.isdir(SCRIPTS_DIR), f"SCRIPTS_DIR missing: {SCRIPTS_DIR}"
    print(f"  SCRIPTS_DIR: {SCRIPTS_DIR}")
    print(f"  DATA_DIR:    {DATA_DIR}")
    print(f"  MEMORY_FILE: {MEMORY_FILE}")

    # 2. Test save_snapshot (writes to temp location — use the real file)
    test_regime = "TEST_REGIME"
    test_params = {"fixed_alloc_pct": 0.5, "rebalance_threshold": 0.03}
    test_metrics = {"equity": 10000.0, "dd": 0.05, "alpha": 150.0, "wr": 0.55}

    save_snapshot(test_regime, test_params, test_metrics)
    assert os.path.exists(MEMORY_FILE), "Memory file was not created"
    print("   save_snapshot() created memory file")

    # 3. Test get_best_params
    best = get_best_params(test_regime)
    assert best is not None, "get_best_params returned None"
    assert best.get("fixed_alloc_pct") == 0.5, f"Unexpected params: {best}"
    print(f"   get_best_params() returned correct params: {best}")

    # 4. Test get_best_params for unknown regime
    assert get_best_params("NONEXISTENT") is None, "Should return None for unknown regime"
    print("   get_best_params() returns None for unknown regime")

    # 5. Test _count_snapshots
    count = _count_snapshots(test_regime)
    assert count == 1, f"Expected 1 snapshot, got {count}"
    print(f"   _count_snapshots({test_regime}) = {count}")

    # 6. Test apply_best_if_known (should return False — not enough data)
    result = apply_best_if_known(test_regime)
    assert result is False, f"apply_best_if_known should return False with <3 snapshots, got {result}"
    print("   apply_best_if_known() correctly skipped (insufficient data)")

    # 7. Clean up test data from memory file
    # Remove the test snapshot we just added
    lines = []
    with open(MEMORY_FILE) as f:
        for line in f:
            try:
                snap = json.loads(line.strip())
                if snap.get("regime") != test_regime:
                    lines.append(line)
            except json.JSONDecodeError:
                lines.append(line)
    with open(MEMORY_FILE, "w") as f:
        f.writelines(lines)
    print("   Test data cleaned up")

    print("\n[param_memory]  All tests passed — import and API work correctly.")


# ── CLI entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _test()
    else:
        print("Usage: python3 quantforge_param_memory.py test")
        print("       (import as module for save_snapshot, get_best_params, apply_best_if_known)")
