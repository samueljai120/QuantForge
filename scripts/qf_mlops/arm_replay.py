"""Replay recorded prediction outcomes into honest arm returns — Phase 1.1.

Consumes resolved records of the form
``{"regime_predicted": ..., "pct_change": <realized % move>}`` (as written to
``data/quantforge/signal_outcomes.jsonl``) and replays each arm under IDENTICAL
per-trade costs. Because the signal and random arms pay the same cost, the
*difference* `signal - random` is a clean edge measure even when the absolute
cost level is imperfect (funding/slippage gaps in the live model don't bias the
delta).

Scope note: this measures the **directional regime signal's** economic edge, not
the XGBoost coin-outperformance model (which needs its own prediction log). It
answers: does the directional signal make money net of costs, and does it beat a
randomized predictor?

Pure functions — no I/O, deterministic given inputs.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Callable, Dict, List

_BULL = {"BULL", "STRONG_BULL"}
_BEAR = {"BEAR", "STRONG_BEAR"}


def signal_direction(record: Dict[str, Any]) -> int:
    """Map a predicted regime to a position in {-1, 0, +1}. Unknown -> flat."""
    r = str(record.get("regime_predicted", "")).upper()
    if r in _BULL:
        return 1
    if r in _BEAR:
        return -1
    return 0


def replay_directional(
    outcomes: List[Dict[str, Any]],
    *,
    cost_bps: float,
    policy: Callable[[Dict[str, Any], int], int],
) -> Dict[str, Any]:
    """Compound the return of taking ``policy`` positions over ``outcomes``.

    Each non-flat position realizes ``pos * pct_change`` and pays ``cost_bps``
    (round-trip, in basis points). ``pct_change`` is a percent (e.g. 2.0 == +2%).
    """
    equity = 1.0
    cost = cost_bps / 10000.0
    n_trades = 0
    hits = 0
    for i, rec in enumerate(outcomes):
        pos = policy(rec, i)
        pct = float(rec.get("pct_change", 0.0)) / 100.0
        if pos != 0:
            equity *= (1.0 + pos * pct - cost)
            n_trades += 1
            if pos * pct > 0:
                hits += 1
    return {
        "cum_return_pct": (equity - 1.0) * 100.0,
        "n_trades": n_trades,
        "hit_rate": (hits / n_trades) if n_trades else 0.0,
    }


def random_baseline(
    outcomes: List[Dict[str, Any]], *, cost_bps: float, n_seeds: int = 200
) -> Dict[str, Any]:
    """Mean/std cumulative return of a randomized predictor over ``n_seeds`` runs.

    Deterministic (seeds 0..n_seeds-1) so the baseline is reproducible.
    """
    results = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        dirs = [rng.choice((-1, 0, 1)) for _ in outcomes]
        r = replay_directional(outcomes, cost_bps=cost_bps, policy=lambda rec, i: dirs[i])
        results.append(r["cum_return_pct"])
    return {
        "mean_cum_return_pct": statistics.fmean(results) if results else 0.0,
        "std_cum_return_pct": statistics.pstdev(results) if len(results) > 1 else 0.0,
        "n_seeds": n_seeds,
    }


def frequency_matched_random(
    outcomes: List[Dict[str, Any]],
    signal_policy: Callable[[Dict[str, Any], int], int],
    *,
    cost_bps: float,
    n_seeds: int = 200,
) -> Dict[str, Any]:
    """Randomized control that trades EXACTLY where the signal trades, with random
    sign. Both arms pay identical cost (same trade count), so the difference is
    pure directional skill — not a trade-frequency artifact.
    """
    trade_idx = [i for i, rec in enumerate(outcomes) if signal_policy(rec, i) != 0]
    results = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        signs = {i: rng.choice((-1, 1)) for i in trade_idx}
        r = replay_directional(
            outcomes, cost_bps=cost_bps, policy=lambda rec, i: signs.get(i, 0)
        )
        results.append(r["cum_return_pct"])
    return {
        "mean_cum_return_pct": statistics.fmean(results) if results else 0.0,
        "std_cum_return_pct": statistics.pstdev(results) if len(results) > 1 else 0.0,
        "n_seeds": n_seeds,
        "n_trades": len(trade_idx),
    }


def signal_edge_report(
    outcomes: List[Dict[str, Any]], *, cost_bps: float, n_seeds: int = 200
) -> Dict[str, Any]:
    """Honest verdict: directional signal vs a FREQUENCY-MATCHED randomized control.

    The frequency-matched control is the fair one: it isolates directional skill
    from trade frequency. The naive independent-random number is reported too, but
    only for transparency — it conflates 'predicts worse' with 'trades more'.
    """
    resolved = [o for o in outcomes if "pct_change" in o and o.get("pct_change") is not None]
    sig = replay_directional(resolved, cost_bps=cost_bps, policy=lambda rec, i: signal_direction(rec))
    fm = frequency_matched_random(
        resolved, lambda rec, i: signal_direction(rec), cost_bps=cost_bps, n_seeds=n_seeds
    )
    indep = random_baseline(resolved, cost_bps=cost_bps, n_seeds=n_seeds)
    delta = sig["cum_return_pct"] - fm["mean_cum_return_pct"]
    std = fm["std_cum_return_pct"]
    return {
        "n_resolved": len(resolved),
        "cost_bps": cost_bps,
        "control": "frequency_matched",
        "signal_cum_return_pct": sig["cum_return_pct"],
        "signal_n_trades": sig["n_trades"],
        "signal_hit_rate": sig["hit_rate"],
        "random_mean_cum_return_pct": fm["mean_cum_return_pct"],
        "random_std_cum_return_pct": std,
        "random_independent_mean_cum_return_pct": indep["mean_cum_return_pct"],
        "cash_return_pct": 0.0,
        "signal_minus_random_pct": delta,
        # How many control-std's above the control mean the signal sits (rough
        # significance gauge — not a formal test).
        "signal_z_vs_random": (delta / std) if std > 0 else 0.0,
        "signal_beats_random": delta > 0,
    }
