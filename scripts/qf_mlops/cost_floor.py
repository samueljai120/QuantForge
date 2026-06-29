"""Per-symbol cost-floor verdict — Phase 1.2.

Given each symbol's realistic round-trip cost (bps) and the model's breakeven
(EV_gross expressed in bps), decide where the ML edge actually survives. A symbol
survives only if its cost is STRICTLY below breakeven (at breakeven, after-cost
EV is exactly zero — not profitable).

Pure logic. The per-symbol costs are produced by the system's own
``quantforge_execution_realism.get_realistic_roundtrip_cost`` in the run step, so
this verdict uses the trading system's own cost assumptions, not invented ones.
"""

from __future__ import annotations

from typing import Dict


def survives_cost_floor(round_trip_bps: float, breakeven_bps: float) -> bool:
    return round_trip_bps < breakeven_bps


def cost_floor_report(symbol_costs: Dict[str, float], breakeven_bps: float) -> Dict:
    per_symbol = {
        sym: {
            "round_trip_bps": cost,
            "survives": survives_cost_floor(cost, breakeven_bps),
            "margin_bps": breakeven_bps - cost,
        }
        for sym, cost in symbol_costs.items()
    }
    survivors = sorted(s for s, r in per_symbol.items() if r["survives"])
    n_total = len(symbol_costs)
    return {
        "breakeven_bps": breakeven_bps,
        "n_total": n_total,
        "n_survive": len(survivors),
        "fraction_survive": (len(survivors) / n_total) if n_total else 0.0,
        "survivors": survivors,
        "per_symbol": per_symbol,
    }
