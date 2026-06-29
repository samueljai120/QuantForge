#!/usr/bin/env python3
"""QuantForge meme / volatile-coin paper scout lane.

Creates a lightweight artifact for volatile, liquid, non-major symbols that may
be worth paper-only exploration under much tighter limits than the core lane.
This does not execute trades. It only proposes bounded candidates and safety
constraints.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_paper import (
    MAJOR_SYMBOLS,
    QUALITY_MIN_PRICE,
    SCAN_TOP_N,
    _get_regime,
    _passes_symbol_quality_filters,
    get_klines,
    screen_coins,
)

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTPUT_FILE = os.path.join(BASE_DIR, "meme-lane-report.json")

MEME_MIN_VOLUME_USDT = 750_000
MEME_MIN_ABS_24H_MOVE_PCT = 8.0
MEME_MAX_ABS_24H_MOVE_PCT = 35.0
MEME_MIN_HISTORY_CANDLES = 1500
MEME_MAX_PRICE = 15.0
MEME_MIN_QUALITY_SCORE = 0.72


def _risk_bucket(abs_move_pct: float, quality_score: float, history_candles: int) -> str:
    if abs_move_pct >= 25 or quality_score < 0.78 or history_candles < 2000:
        return "extreme"
    if abs_move_pct >= 15 or quality_score < 0.84:
        return "high"
    return "elevated"


def _paper_safety_profile(risk_bucket: str) -> dict:
    profile = {
        "paper_only": True,
        "max_position_pct": 0.0075,
        "leverage": 1,
        "take_profit_pct": 0.035,
        "secondary_take_profit_pct": 0.055,
        "trailing_stop_pct": 0.015,
        "max_hold_hours": 8,
        "notes": [
            "Keep size tiny relative to the core lane.",
            "Secure profit earlier than the core trend lane.",
            "Do not promote to live use without standalone paper expectancy.",
        ],
    }
    if risk_bucket == "extreme":
        profile["max_position_pct"] = 0.005
        profile["take_profit_pct"] = 0.03
        profile["secondary_take_profit_pct"] = 0.045
        profile["trailing_stop_pct"] = 0.012
        profile["max_hold_hours"] = 6
    elif risk_bucket == "high":
        profile["max_position_pct"] = 0.006
        profile["take_profit_pct"] = 0.032
        profile["secondary_take_profit_pct"] = 0.05
        profile["trailing_stop_pct"] = 0.013
        profile["max_hold_hours"] = 7
    return profile


def main() -> None:
    cfg.require_production_runtime("quantforge_meme_lane_report.py")
    regime = _get_regime()
    universe = screen_coins(max(40, SCAN_TOP_N))
    rows = []

    for coin in universe:
        symbol = str(coin.get("symbol") or "")
        if not symbol or symbol in MAJOR_SYMBOLS:
            continue
        price = float(coin.get("price", 0.0) or 0.0)
        vol_usdt = float(coin.get("vol_usdt", 0.0) or 0.0)
        abs_move_pct = abs(float(coin.get("change_pct", 0.0) or 0.0))
        if vol_usdt < MEME_MIN_VOLUME_USDT:
            continue
        if abs_move_pct < MEME_MIN_ABS_24H_MOVE_PCT or abs_move_pct > MEME_MAX_ABS_24H_MOVE_PCT:
            continue
        if price < QUALITY_MIN_PRICE or price > MEME_MAX_PRICE:
            continue
        try:
            candles = get_klines(symbol, "1hour", 300)
            history_candles = len(candles)
            quality_pass, quality_reasons, quality_metrics = _passes_symbol_quality_filters(coin, candles)
        except Exception as exc:
            rows.append({
                "symbol": symbol,
                "status": "error",
                "reason": str(exc),
            })
            continue

        quality_score = float(quality_metrics.get("quality_score", 0.0) or 0.0)
        if history_candles < MEME_MIN_HISTORY_CANDLES:
            rows.append({
                "symbol": symbol,
                "status": "blocked",
                "reason": f"history {history_candles} < {MEME_MIN_HISTORY_CANDLES}",
                "abs_move_pct": round(abs_move_pct, 2),
                "quality_score": round(quality_score, 4),
            })
            continue
        if quality_score < MEME_MIN_QUALITY_SCORE:
            rows.append({
                "symbol": symbol,
                "status": "blocked",
                "reason": f"quality {quality_score:.2f} < {MEME_MIN_QUALITY_SCORE:.2f}",
                "abs_move_pct": round(abs_move_pct, 2),
                "quality_score": round(quality_score, 4),
            })
            continue

        risk_bucket = _risk_bucket(abs_move_pct, quality_score, history_candles)
        scout_score = round(
            (min(abs_move_pct, 30.0) / 30.0) * 0.35
            + min(vol_usdt / 20_000_000.0, 1.0) * 0.20
            + min(max(quality_score, 0.0), 1.0) * 0.45,
            4,
        )
        rows.append({
            "symbol": symbol,
            "status": "paper_probe_candidate",
            "price": round(price, 8),
            "abs_move_pct": round(abs_move_pct, 2),
            "vol_usdt": round(vol_usdt, 2),
            "history_candles": history_candles,
            "quality_score": round(quality_score, 4),
            "quality_pass_core": bool(quality_pass),
            "core_quality_reason": quality_reasons[0] if quality_reasons else None,
            "risk_bucket": risk_bucket,
            "scout_score": scout_score,
            "paper_safety_profile": _paper_safety_profile(risk_bucket),
        })

    candidates = [r for r in rows if r.get("status") == "paper_probe_candidate"]
    candidates.sort(key=lambda r: (float(r.get("scout_score", 0.0) or 0.0), float(r.get("quality_score", 0.0) or 0.0)), reverse=True)
    blocked = [r for r in rows if r.get("status") == "blocked"]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "status": "watch_candidates" if candidates else "observe_only",
        "summary": {
            "universe_size": len(universe),
            "candidate_count": len(candidates),
            "blocked_count": len(blocked),
            "top_candidate": (candidates[0].get("symbol") if candidates else None),
            "policy": "paper_only_high_volatility_lane",
        },
        "guardrails": {
            "paper_only": True,
            "segregated_from_core_lane": True,
            "max_concurrent_positions": 1,
            "allow_live_promotion": False,
            "required_before_live": [
                "positive post-fee expectancy in standalone paper lane",
                "acceptable drawdown relative to tiny position sizing",
                "evidence that exits capture volatility better than the core lane",
            ],
        },
        "rows": candidates[:8] + blocked[:8],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("QuantForge meme lane report")
    print(f"Candidates: {len(candidates)}")
    print(f"Saved:      {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
