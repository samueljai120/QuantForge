"""EWAA P0 — map per-strategy edge scores to a bounded regime_weight_table.

Per double-eval must-fix #1, the agent reads spot allocation from the per-row
spot_alloc_pct inside regime_weight_table (and fixed_alloc_pct), NOT a flat
spot_alloc_pct key. So this emits the FULL table object (all 6 regime rows), and
the HODL residual is each row's spot_alloc_pct = 1 - sum(active weights).

Weights are fractional-Kelly (lambda * edge/variance * confidence), clamped to
the agent's _TABLE_BOUNDS caps, and the active sleeve is normalized to leave at
least the spot floor (>= 40% in BTC). Only PROMOTED capital-lane strategies earn
weight; everything else (SHADOW, or signals without a capital lane) -> 0, so the
default is emergent HODL.

CAPS NOTE: LANE_CAPS mirror quantforge_agent._TABLE_BOUNDS exactly; the agent
re-clamps on load (defense in depth). Keep these in sync (single-source-of-truth
follow-up tracked in the plan).
"""
from __future__ import annotations

from typing import Dict

REGIMES = ["STRONG_BEAR", "BEAR", "CHOP", "NEUTRAL", "BULL", "STRONG_BULL"]

# strategy_id -> the agent's regime_weight_table key (only these 5 are capital lanes)
LANE_KEY = {
    "futures_lane": "futures_weight",
    "mean_revert": "mr_weight",
    "ml_scanner": "ml_scanner_weight",
    "funding_arb": "funding_arb_weight",
}
# must match quantforge_agent._TABLE_BOUNDS
LANE_CAPS = {
    "futures_weight": 0.30,
    "mr_weight": 0.50,
    "ml_scanner_weight": 0.15,
    "funding_arb_weight": 0.30,
}
SPOT_FLOOR, SPOT_CEIL = 0.40, 0.85
DEFAULT_VAR = 1.0  # conservative variance when none reported


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def map_edges_to_table(strategies: Dict[str, dict], *, kelly_lambda: float = 0.25) -> dict:
    """strategies: {strategy_id: {status, edge_pct, confidence, edge_var}}.
    Returns a regime_weight_table {regime: {spot_alloc_pct, *_weight}}."""
    active: Dict[str, float] = {k: 0.0 for k in LANE_CAPS}
    for sid, sc in strategies.items():
        key = LANE_KEY.get(sid)
        if key is None or sc.get("status") != "PROMOTED":
            continue
        edge = float(sc.get("edge_pct", 0.0)) / 100.0  # % -> fraction
        var = max(float(sc.get("edge_var", DEFAULT_VAR) or DEFAULT_VAR), 1e-9)
        conf = float(sc.get("confidence", 0.0))
        kelly_f = edge / var
        active[key] = _clamp(kelly_lambda * kelly_f * conf, 0.0, LANE_CAPS[key])

    # Normalize the active sleeve to leave at least the spot floor in BTC.
    total = sum(active.values())
    max_active = 1.0 - SPOT_FLOOR
    if total > max_active and total > 0:
        scale = max_active / total
        active = {k: v * scale for k, v in active.items()}
        total = sum(active.values())

    spot = _clamp(1.0 - total, SPOT_FLOOR, SPOT_CEIL)
    row = {"spot_alloc_pct": round(spot, 4), **{k: round(v, 4) for k, v in active.items()}}
    return {reg: dict(row) for reg in REGIMES}
