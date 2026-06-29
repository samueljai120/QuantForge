"""Phase E — benchmark-beating gate.

Decide whether a signal has earned REAL capital allocation. A signal is promoted
only when, on live out-of-sample evidence, it beats the benchmark (holding) by a
required margin over a minimum number of trades. Otherwise it stays in shadow
and the default posture remains the benchmark. This is the capital-protection
rule that follows from the session's evidence: do not deploy capital into
anything that has not out-earned simply holding.
"""

from __future__ import annotations


def benchmark_gate(
    *,
    signal_return_pct: float,
    benchmark_return_pct: float,
    n_trades: int,
    min_trades: int = 20,
    min_edge_pct: float = 0.0,
) -> dict:
    """Return a promotion decision for a signal vs the benchmark.

    ``signal_return_pct`` / ``benchmark_return_pct`` are realized returns over the
    SAME live window. ``min_edge_pct`` is the margin the signal must clear above
    the benchmark to justify the risk of trading instead of holding.
    """
    edge = signal_return_pct - benchmark_return_pct
    if n_trades < min_trades:
        return {"allowed": False, "status": "SHADOW", "edge_pct": edge,
                "reason": f"insufficient trades ({n_trades}<{min_trades}) — keep gathering live evidence"}
    if edge < min_edge_pct:
        return {"allowed": False, "status": "SHADOW", "edge_pct": edge,
                "reason": f"does not beat benchmark by required margin (+{edge:.2f}% < +{min_edge_pct:.2f}%)"}
    return {"allowed": True, "status": "PROMOTED", "edge_pct": edge,
            "reason": f"beats benchmark by +{edge:.2f}% over {n_trades} trades"}


def graduated_gate(
    *,
    edge_pct: float,
    hac_t_stat: float,
    beats_null: bool,
    survives_cost: bool,
    n_trades: int,
    min_trades: int = 20,
    min_t_stat: float = 2.0,
    fee_hurdle_pct: float = 0.0,
) -> dict:
    """Multiplicative promotion verdict for one strategy (EWAA).

    PROMOTED (earns weight) only if it clears EVERY control: beats the benchmark
    by the fee hurdle, HAC t >= min_t_stat, beats the null control, survives the
    cost floor, and has >= min_trades. Any failure -> SHADOW (zero weight). This
    is the graduated extension of the binary benchmark_gate(); it does not touch
    that function. ``confidence`` scales the weight in edge_weight_map.
    """
    reasons = []
    if n_trades < min_trades:
        reasons.append(f"insufficient trades ({n_trades}<{min_trades})")
    if edge_pct <= fee_hurdle_pct:
        reasons.append(f"edge {edge_pct:.3f}% <= hurdle {fee_hurdle_pct:.3f}%")
    if hac_t_stat < min_t_stat:
        reasons.append(f"HAC t {hac_t_stat:.2f} < {min_t_stat}")
    if not beats_null:
        reasons.append("does not beat null control")
    if not survives_cost:
        reasons.append("below cost floor")
    promoted = not reasons
    confidence = round(min(1.0, n_trades / (2.0 * min_trades)), 4) if promoted else 0.0
    return {
        "status": "PROMOTED" if promoted else "SHADOW",
        "allowed": promoted,
        "confidence": confidence,
        "edge_pct": edge_pct,
        "reasons": reasons,
    }
