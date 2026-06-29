"""EWAA P0 — honest per-strategy edge attribution.

Consumes canonical closed-trade records (agent trades attributed by OPENING
strategy_id per P-1b, plus carry episodes from carry_ingest) and produces, per
strategy, a gated edge score. Pure: no I/O, no price fetches — the caller
attaches benchmark_return_pct (BTC-HODL over the real [entry_ts,exit_ts] window
for directional strategies; 0 for carry, which is delta-neutral / cash-benchmark).

Controls (all must clear -> PROMOTED, else SHADOW), via graduated_gate:
  edge > fee hurdle AND HAC t >= 2 AND beats null AND survives cost AND n >= 20.
Edge is decay-weighted (recent trades count more). Reconciliation is total-pnl
conservation: a mismatch sets reconciled=False so the allocator fails closed.
Unknown strategy_ids route to an unattributed zero bucket (never earn capital).
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Sequence

from qf_mlops.arm_backtest import hac_tstat, random_control_percentile
from qf_mlops.benchmark_gate import graduated_gate

# The opening strategies the agent tags (P-1b) + carry. Anything else -> unattributed.
STRATEGY_IDS = {
    "hodl", "mean_revert", "futures_lane", "ml_scanner", "liquidation_dip",
    "funding_arb", "cvd_momentum", "vol_breakout", "cross_asset", "liq_scalp",
    "oi_divergence", "timesfm",
}


def _parse(ts):
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _decay_weight(exit_ts, now, half_life_days):
    try:
        age = (now - _parse(exit_ts)).total_seconds() / 86400.0
    except Exception:
        age = 0.0
    if age < 0:
        age = 0.0
    return 0.5 ** (age / half_life_days) if half_life_days > 0 else 1.0


def _cap(t):
    if t == float("inf"):
        return 99.0
    if t == float("-inf"):
        return -99.0
    return round(t, 3)


def attribute_edges(
    records: Sequence[dict],
    *,
    portfolio_realized_total: float = None,
    now_ts=None,
    half_life_days: float = 21.0,
    min_trades: int = 20,
    fee_hurdle_pct: float = 0.0,
    min_t_stat: float = 2.0,
    null_seeds: int = 30,
    null_pct: float = 95.0,
    recon_tol_abs: float = 0.01,
    recon_tol_frac: float = 0.005,
) -> dict:
    now = _parse(now_ts) if now_ts else datetime.now(timezone.utc)

    by: Dict[str, list] = defaultdict(list)
    unattributed: List[dict] = []
    for r in records:
        sid = r.get("strategy_id")
        (by[sid] if sid in STRATEGY_IDS else unattributed).append(r)

    # Reconciliation: total attributed pnl must conserve to the portfolio total.
    total_attr = sum(float(r.get("pnl_usd", 0) or 0) for r in records)
    reconciled, recon_err = True, 0.0
    if portfolio_realized_total is not None:
        recon_err = abs(total_attr - float(portfolio_realized_total))
        tol = max(recon_tol_abs, recon_tol_frac * abs(float(portfolio_realized_total)))
        reconciled = recon_err <= tol

    # Null pool = every attributed trade's excess (the opportunity set).
    pool = [
        float(r.get("return_pct", 0) or 0) - float(r.get("benchmark_return_pct", 0) or 0)
        for rs in by.values() for r in rs
    ]

    strategies: Dict[str, dict] = {}
    for sid, rs in by.items():
        excess = [float(r.get("return_pct", 0) or 0) - float(r.get("benchmark_return_pct", 0) or 0) for r in rs]
        weights = [_decay_weight(r.get("exit_ts"), now, half_life_days) for r in rs]
        wsum = sum(weights) or 1.0
        edge_pct = sum(e * w for e, w in zip(excess, weights)) / wsum
        raw_mean_return = statistics.fmean([float(r.get("return_pct", 0) or 0) for r in rs])
        gross_excess = statistics.fmean([
            float(r.get("gross_return_pct", r.get("return_pct", 0)) or 0) - float(r.get("benchmark_return_pct", 0) or 0)
            for r in rs
        ])
        mean_cost = statistics.fmean([float(r.get("cost_pct", 0) or 0) for r in rs])

        t = hac_tstat(excess)
        npct = random_control_percentile(pool, len(rs), seeds=null_seeds, pct=null_pct) if pool else 0.0
        beats_null = edge_pct > npct
        survives_cost = gross_excess > mean_cost
        reduced_beta = raw_mean_return > 0 and edge_pct <= 0

        gate = graduated_gate(
            edge_pct=edge_pct, hac_t_stat=t, beats_null=beats_null,
            survives_cost=survives_cost, n_trades=len(rs),
            min_trades=min_trades, min_t_stat=min_t_stat, fee_hurdle_pct=fee_hurdle_pct,
        )
        strategies[sid] = {
            "strategy_id": sid, "n_trades": len(rs), "edge_pct": round(edge_pct, 4),
            "edge_var": round(statistics.pvariance(excess), 6) if len(excess) > 1 else 0.0,
            "hac_t_stat": _cap(t), "null_pct": round(npct, 4), "beats_null": beats_null,
            "survives_cost": survives_cost, "reduced_beta_flag": reduced_beta,
            "confidence": gate["confidence"], "status": gate["status"], "reasons": gate["reasons"],
        }

    return {
        "reconciled": reconciled,
        "recon_error": round(recon_err, 4),
        "unattributed": {"n": len(unattributed), "pnl": round(sum(float(r.get("pnl_usd", 0) or 0) for r in unattributed), 4)},
        "pool_size": len(pool),
        "strategies": strategies,
    }
