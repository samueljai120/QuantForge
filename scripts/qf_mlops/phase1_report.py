"""Unified Phase 1 verdict — the standing cost-honest answer.

Synthesizes the three analyses into one verdict + actions, so the daily report
can carry "does the system have a cost-surviving edge?" continuously instead of
it being a one-off investigation. Pure synthesis; inputs come from
``arm_replay.signal_edge_report``, ``ml_edge.ml_edge_report``, and
``cost_floor.cost_floor_report``.
"""

from __future__ import annotations

from typing import Dict, List

# A directional edge is only "demonstrated" if it beats random AND is significant.
_SIGNIFICANCE_Z = 2.0


def phase1_verdict(*, signal: Dict, ml: Dict, cost: Dict) -> Dict:
    sig_has_edge = bool(signal.get("signal_beats_random")) and abs(
        signal.get("signal_z_vs_random", 0.0)
    ) >= _SIGNIFICANCE_Z

    ml_survives = bool(ml.get("edge_survives_costs"))
    frac = cost.get("fraction_survive", 0.0)

    actions: List[str] = []
    if not ml_survives:
        actions.append(
            "ML edge does not survive realistic costs — reduce round-trip cost "
            "(maker fills, liquid-only universe, lower turnover) or do not trade ML."
        )
    elif frac < 0.1:
        actions.append(
            f"ML edge survives only on {cost.get('n_survive', 0)}/{cost.get('n_total', 0)} "
            f"symbols — restrict the scanner universe to the cost-survivors "
            f"({', '.join(cost.get('survivors', [])[:6])})."
        )
        actions.append("Gate each trade on per-symbol edge > per-symbol cost.")
    if not sig_has_edge:
        actions.append(
            "Directional regime signal shows no demonstrated edge — do not size up on it "
            "until a larger multi-regime sample is significant."
        )

    overall = _overall(sig_has_edge, ml_survives, frac)

    return {
        "directional_signal": {
            "has_edge": sig_has_edge,
            "hit_rate": signal.get("signal_hit_rate"),
            "z_vs_random": signal.get("signal_z_vs_random"),
            "n_resolved": signal.get("n_resolved"),
        },
        "ml_model": {
            "survives_costs": ml_survives,
            "auc": ml.get("auc"),
            "n_samples": ml.get("n_samples"),
            "ev_gross_pct": ml.get("ev_gross_pct"),
            "breakeven_cost_bps": ml.get("breakeven_cost_bps"),
        },
        "tradeable_universe": {
            "n_survive": cost.get("n_survive"),
            "n_total": cost.get("n_total"),
            "fraction_survive": frac,
            "survivors": cost.get("survivors", []),
        },
        "overall": overall,
        "recommended_actions": actions,
    }


def _overall(sig_has_edge: bool, ml_survives: bool, frac: float) -> str:
    if ml_survives and frac >= 0.1:
        return "System shows a cost-surviving edge across part of its universe."
    if ml_survives and frac < 0.1:
        return (
            "ML has a real but razor-thin edge that only survives costs on a few "
            "liquid symbols; the system is structurally unprofitable on its full universe."
        )
    return "No cost-surviving edge demonstrated — current trading loses to costs."


def format_verdict(v: Dict) -> str:
    ds = v["directional_signal"]
    ml = v["ml_model"]
    tu = v["tradeable_universe"]
    lines = [
        "QuantForge Phase 1 — cost-honest verdict",
        f"  Overall: {v['overall']}",
        f"  Directional signal: edge={ds['has_edge']} "
        f"(hit {(ds['hit_rate'] or 0)*100:.0f}%, z={ds['z_vs_random']}, n={ds['n_resolved']})",
        f"  ML model: AUC {ml['auc']} on n={ml['n_samples']:,}; gross EV {ml['ev_gross_pct']}%/trade; "
        f"breakeven {ml['breakeven_cost_bps']} bps; survives_costs={ml['survives_costs']}",
        f"  Tradeable universe: {tu['n_survive']}/{tu['n_total']} symbols clear the cost floor "
        f"({tu['fraction_survive']*100:.1f}%)",
    ]
    if v["recommended_actions"]:
        lines.append("  Actions:")
        lines.extend(f"    - {a}" for a in v["recommended_actions"])
    return "\n".join(lines)
