"""Phase E — funding-carry backtest (structurally-sound edge candidate).

Simulate a delta-neutral funding-carry sleeve over a per-interval funding series:
enter when |funding| >= enter_thresh on the collecting side, accrue |funding|
each interval while it stays >= exit_thresh and keeps its sign, exit otherwise.
A two-leg round-trip cost is charged once per episode (entry). Price P&L is
assumed to cancel under the delta hedge — basis drift is a separate refinement.

P&L is accumulated additively (per-interval funding is tiny, so compounding is
negligible and additive keeps the accounting transparent).
"""

from __future__ import annotations

from typing import Sequence


def carry_decision(
    funding: float,
    *,
    in_pos: bool,
    side: int,
    enter_thresh: float,
    exit_thresh: float,
) -> tuple:
    """One live carry decision. Returns (action, new_side).

    action in {ENTER, HOLD, EXIT, FLAT}; new_side in {+1, -1, 0}. +1 = positioned
    to collect POSITIVE funding (short perp / long spot); -1 = collect NEGATIVE
    funding (long perp / short spot). Exit on normalization OR sign flip (a flip
    means you would start paying). This is the live analog of carry_pnl's loop —
    same rule the 4.5yr backtest validated.
    """
    if in_pos:
        if abs(funding) < exit_thresh or (funding > 0) != (side > 0):
            return ("EXIT", 0)
        return ("HOLD", side)
    if abs(funding) >= enter_thresh:
        return ("ENTER", 1 if funding > 0 else -1)
    return ("FLAT", 0)


def carry_pnl(
    funding: Sequence[float],
    *,
    enter_thresh: float,
    exit_thresh: float,
    cost_frac: float,
) -> dict:
    """Return total net carry return (%) and trade stats over ``funding``."""
    equity = 0.0
    in_pos = False
    side = 0
    episodes = 0
    periods_in = 0

    for f in funding:
        if in_pos and (abs(f) < exit_thresh or (f > 0) != (side > 0)):
            in_pos = False  # normalized or sign-flipped -> close (round-trip already paid)
        if not in_pos:
            if abs(f) >= enter_thresh:
                in_pos = True
                side = 1 if f > 0 else -1
                episodes += 1
                equity -= cost_frac      # two-leg round-trip cost on entry
                equity += abs(f)          # collect this interval's funding
                periods_in += 1
        else:
            equity += abs(f)
            periods_in += 1

    return {
        "total_return_pct": equity * 100.0,
        "n_episodes": episodes,
        "periods_in_market": periods_in,
    }
