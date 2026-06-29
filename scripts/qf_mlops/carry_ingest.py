"""EWAA P-1c — funding-carry ingestion into the canonical edge schema.

The one proven edge (extreme-funding carry) writes to carry_harvester_state.json,
NOT agent_trades.jsonl — so without this path it is invisible to the allocator.
This reads the harvester's closed-episode history and emits records in the same
canonical shape edge_attribution consumes for every strategy.

Two corrections baked in from the double-eval (must-fix #6):
  * history[].pnl_usd is ALREADY net (pnl = collected - cost). We treat it as
    final and DERIVE cost = collected - pnl. We never re-apply a cost-floor.
  * carry is delta-neutral, so its benchmark is cash / 0% (a market-neutral
    hurdle), not BTC-HODL — recorded as benchmark="cash".

Fail-closed: a missing/unreadable state file yields no records, so funding_arb
correctly reports SHADOW rather than a fabricated edge.
"""
from __future__ import annotations

import json
from typing import List

STRATEGY_ID = "funding_arb"
DEFAULT_NOTIONAL = 150.0  # matches quantforge_carry_harvester.NOTIONAL


def ingest_carry(state, *, notional: float = DEFAULT_NOTIONAL) -> List[dict]:
    """Map carry harvester closed episodes -> canonical edge records."""
    if not isinstance(state, dict):
        return []
    history = state.get("history") or []
    out: List[dict] = []
    for ep in history:
        if not isinstance(ep, dict):
            continue
        try:
            pnl = float(ep["pnl_usd"])
        except (KeyError, TypeError, ValueError):
            continue
        collected = float(ep.get("collected", pnl))
        out.append({
            "strategy_id": STRATEGY_ID,
            "symbol": str(ep.get("symbol", "")),
            "entry_ts": ep.get("entry_ts"),
            "exit_ts": ep.get("exit_ts"),
            "pnl_usd": pnl,                                          # already net
            "return_pct": (pnl / notional * 100.0) if notional else 0.0,
            "cost_usd": collected - pnl,                             # derived, not re-applied
            "benchmark": "cash",                                    # delta-neutral -> 0% hurdle
        })
    return out


def load_and_ingest(path: str, *, notional: float = DEFAULT_NOTIONAL) -> List[dict]:
    """Read carry_harvester_state.json and ingest; fail-closed to [] on any error."""
    try:
        with open(path) as f:
            state = json.load(f)
    except Exception:
        return []
    return ingest_carry(state, notional=notional)
