"""Phase 1 — counterfactual arm backtest.

Run RULES_ONLY vs RULES_ML (and HODL / ML_ONLY / RANDOM_ML controls) over
IDENTICAL bars, costs and realized forward returns, then feed the arm totals to
``qf_mlops.baselines.decompose`` for an honest ``incremental_ml_value``.

Cost convention: round-trip cost is charged ONCE per contiguous holding period
(on entry or flip), never per bar — so a buy-and-hold arm is not unfairly
overcharged versus a sparse-signal arm.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Sequence

from qf_mlops.baselines import Arm, decompose, incremental_ml_value


def arm_total_return(
    positions: Sequence[int],
    fwd_rets_pct: Sequence[float],
    cost_frac: float,
) -> float:
    """Compound an arm's net return over its per-bar positions.

    ``positions[i]`` in {-1, 0, +1} is the position held over bar ``i``;
    ``fwd_rets_pct[i]`` is that bar's forward return in percent (2.0 == +2%).
    ``cost_frac`` is the round-trip execution cost as a fraction (0.0015 == 15bps),
    charged once when a new (or flipped) position is entered.

    Returns the cumulative return in percent.
    """
    equity = 1.0
    prev = 0
    for pos, ret in zip(positions, fwd_rets_pct):
        if pos != 0:
            equity *= 1.0 + pos * ret / 100.0
            if pos != prev:  # entering or flipping into this position
                equity *= 1.0 - cost_frac
        prev = pos
    return (equity - 1.0) * 100.0


def build_positions(
    arm: Arm,
    *,
    rule_long: Sequence[bool],
    ml_prob: Sequence[float],
    threshold: float,
    seed: int = 0,
) -> List[int]:
    """Positions (long-only: 0/1) for ``arm`` over the same bars.

    ``rule_long[i]`` — did any long rule-setup fire on bar i.
    ``ml_prob[i]``   — the ensemble's BUY probability on bar i.
    RANDOM_ML places the SAME number of long entries as RULES_ML, at seeded
    random bars — the frequency-matched control that strips out trade-timing
    luck so only genuine ML selection skill can show up in ``RULES_ML``.
    """
    n = len(rule_long)
    if arm == Arm.HODL:
        return [1] * n
    if arm == Arm.CASH:
        return [0] * n
    if arm == Arm.RULES_ONLY:
        return [1 if r else 0 for r in rule_long]
    if arm == Arm.ML_ONLY:
        return [1 if p >= threshold else 0 for p in ml_prob]
    if arm == Arm.RULES_ML:
        return [1 if (r and p >= threshold) else 0 for r, p in zip(rule_long, ml_prob)]
    if arm == Arm.RANDOM_ML:
        k = sum(1 for r, p in zip(rule_long, ml_prob) if r and p >= threshold)
        rng = random.Random(seed)
        chosen = set(rng.sample(range(n), k)) if 0 < k <= n else set()
        return [1 if i in chosen else 0 for i in range(n)]
    raise ValueError(f"unsupported arm: {arm}")


_ALL_ARMS = [Arm.HODL, Arm.RULES_ONLY, Arm.RULES_ML, Arm.ML_ONLY, Arm.RANDOM_ML, Arm.CASH]

# The OOF record schema — the single contract between the training pipeline
# (producer) and run_backtest_from_oof (consumer).
OOF_KEYS = ("ts", "symbol", "prob", "rule_long", "fwd_ret_4h", "target", "fold")


def oof_rows_from_fold(
    *,
    ts: Sequence,
    symbol: Sequence[str],
    prob: Sequence[float],
    fwd_ret: Sequence[float],
    target: Sequence[int],
    rule_long: Sequence[bool],
    fold: int,
) -> List[dict]:
    """Build one OOF record per validation-fold bar, in the schema the backtest
    replay consumes. Called from the training CV loop with the predictions of a
    model that did NOT train on these bars (hence leak-free)."""
    rows = []
    for i in range(len(ts)):
        rows.append(
            {
                "ts": int(ts[i]),
                "symbol": str(symbol[i]),
                "prob": float(prob[i]),
                "rule_long": bool(rule_long[i]),
                "fwd_ret_4h": float(fwd_ret[i]),
                "target": int(target[i]),
                "fold": int(fold),
            }
        )
    return rows


def run_backtest_from_oof(
    records: Sequence[dict],
    *,
    threshold: float,
    costs_by_symbol: Dict[str, float],
    fwd_key: str = "fwd_ret_4h",
    seed: int = 0,
) -> dict:
    """Replay out-of-fold predictions into honest arm returns and decompose.

    Each record is one OOF bar with keys ``symbol``, ``ts`` (sortable),
    ``prob`` (ensemble BUY probability), ``rule_long`` (did a long setup fire),
    and ``fwd_key`` (realized forward return as a FRACTION, 0.02 == +2%).
    Because every prediction came from a model that did not train on that bar,
    the resulting ``incremental_ml_value`` is leak-free.

    Arm returns are computed per symbol (with that symbol's round-trip cost) and
    equal-weighted across symbols — a simple, honest portfolio interpretation.
    """
    by_sym: Dict[str, list] = defaultdict(list)
    for r in records:
        by_sym[r["symbol"]].append(r)

    per_arm: Dict[Arm, list] = {a: [] for a in _ALL_ARMS}
    trade_counts: Dict[Arm, int] = {a: 0 for a in _ALL_ARMS}
    n_records = 0

    for sym, rs in by_sym.items():
        rs = sorted(rs, key=lambda r: r["ts"])
        rule_long = [bool(r["rule_long"]) for r in rs]
        ml_prob = [float(r["prob"]) for r in rs]
        fwd_pct = [float(r[fwd_key]) * 100.0 for r in rs]
        cost = float(costs_by_symbol.get(sym, 0.0))
        n_records += len(rs)
        for a in _ALL_ARMS:
            pos = build_positions(a, rule_long=rule_long, ml_prob=ml_prob, threshold=threshold, seed=seed)
            per_arm[a].append(arm_total_return(pos, fwd_pct, cost))
            trade_counts[a] += sum(1 for p in pos if p != 0)

    arms = {a: (sum(v) / len(v) if v else 0.0) for a, v in per_arm.items()}
    return {
        "arms": {a.value: arms[a] for a in _ALL_ARMS},
        "incremental_ml_value": incremental_ml_value(arms[Arm.RULES_ML], arms[Arm.RULES_ONLY]),
        "decompose": decompose(arms),
        "trade_counts": {a.value: trade_counts[a] for a in _ALL_ARMS},
        "n_records": n_records,
        "n_symbols": len(by_sym),
    }


# --- EWAA P0: honest significance helpers ---------------------------------

def hac_tstat(returns: Sequence[float]) -> float:
    """Newey-West HAC-corrected t-stat of the mean of ``returns``.

    Overlapping holding windows make realized returns autocorrelated; the naive
    t-stat overstates significance. The HAC variance adds positively-weighted
    autocovariances so positive autocorrelation correctly *lowers* the t-stat.
    """
    xs = [float(x) for x in returns]
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    dev = [x - m for x in xs]
    g0 = sum(d * d for d in dev) / n
    if g0 <= 0:
        return 0.0 if m == 0 else (float("inf") if m > 0 else float("-inf"))
    lags = max(1, int(4 * (n / 100.0) ** (2.0 / 9.0)))
    var = g0
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        gl = sum(dev[t] * dev[t - lag] for t in range(lag, n)) / n
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * gl
    if var <= 0:
        var = g0  # guard: HAC variance can go negative on small samples
    se = math.sqrt(var / n)
    if se <= 0:
        return 0.0 if m == 0 else (float("inf") if m > 0 else float("-inf"))
    return m / se


def random_control_percentile(pool, n_picks, *, seeds: int = 30, pct: float = 95.0) -> float:
    """Realized-trade null control: the ``pct``-th percentile of the mean of
    ``n_picks`` random draws (with replacement) from ``pool`` over ``seeds`` runs.

    A strategy with ``n_picks`` trades and observed mean M has beaten the null
    iff M exceeds this percentile — i.e. its selection beat random selection of
    the same count from the same opportunity pool (carry's proven 30/30 standard).
    """
    pool = [float(x) for x in pool]
    n_picks = int(n_picks)
    if not pool or n_picks <= 0:
        return 0.0
    means = []
    for s in range(int(seeds)):
        rng = random.Random(s)
        means.append(sum(rng.choice(pool) for _ in range(n_picks)) / n_picks)
    means.sort()
    if len(means) == 1:
        return means[0]
    k = (len(means) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return means[lo]
    return means[lo] + (means[hi] - means[lo]) * (k - lo)
