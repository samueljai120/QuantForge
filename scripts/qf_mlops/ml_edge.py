"""Cost-adjusted ML edge — Phase 1.1.

The honest embargo'd cross-validation (model_meta.json) gives the model a real
but tiny statistical edge: AUC ~0.532 on n=813k (significant, not noise) and a
GROSS expected value per trade. The decisive question is economic — every trade
pays round-trip cost (fees + spread + slippage), so:

    EV_after_cost = EV_gross - round_trip_cost
    breakeven_cost = EV_gross expressed in bps

A model can have a real predictive edge and still lose money once costs are
charged. This module makes that explicit instead of citing AUC in a vacuum.

All percentages are in percent units (0.2257 means 0.2257%); 1% == 100 bps.
"""

from __future__ import annotations

from typing import Dict, List


def ev_after_cost(ev_gross_pct: float, cost_pct: float) -> float:
    """Expected value per trade after subtracting round-trip cost (both percent)."""
    return ev_gross_pct - cost_pct


def breakeven_cost_bps(ev_gross_pct: float) -> float:
    """Round-trip cost (bps) at which the gross edge is exactly consumed."""
    return ev_gross_pct * 100.0  # 1% == 100 bps


def ml_edge_report(
    *,
    ev_gross_pct: float,
    auc: float,
    win_rate: float,
    base_rate: float,
    n_samples: int,
    cost_bps_levels: List[float],
    realistic_cost_bps: float,
) -> Dict:
    """Whether the model's edge survives realistic costs, with the full curve."""
    curve = {cb: ev_after_cost(ev_gross_pct, cb / 100.0) for cb in cost_bps_levels}
    survives = ev_after_cost(ev_gross_pct, realistic_cost_bps / 100.0) > 0
    return {
        "auc": auc,
        "auc_edge_over_random": auc - 0.5,
        "win_rate": win_rate,
        "base_rate": base_rate,
        "n_samples": n_samples,
        "ev_gross_pct": ev_gross_pct,
        "breakeven_cost_bps": breakeven_cost_bps(ev_gross_pct),
        "realistic_cost_bps": realistic_cost_bps,
        "ev_after_cost_by_bps": curve,
        "edge_survives_costs": survives,
    }
