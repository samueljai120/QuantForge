#!/usr/bin/env python3
"""QuantForge target profile helpers.

Keeps alternate label/target construction out of the main training loop so
training modes can switch target semantics without duplicating plumbing.
"""

from __future__ import annotations

import pandas as pd

RESEARCH_HOLD_LONG_TARGET_COL = "target_4h_research_hold_long"
RESEARCH_HOLD_SHORT_TARGET_COL = "target_4h_research_hold_short"
RESEARCH_HOLD_LONG_NET_RET_COL = "rebuild_net_ret_4h_long"
RESEARCH_HOLD_SHORT_NET_RET_COL = "rebuild_net_ret_4h_short"
RESEARCH_HOLD_TREND_LONG_TARGET_COL = "target_4h_research_hold_trend_long"
RESEARCH_HOLD_BREAKOUT_LONG_TARGET_COL = "target_4h_research_hold_breakout_long"
RESEARCH_HOLD_REBOUND_LONG_TARGET_COL = "target_4h_research_hold_rebound_long"
RESEARCH_HOLD_TREND_SHORT_TARGET_COL = "target_4h_research_hold_trend_short"
RESEARCH_HOLD_EXHAUSTION_SHORT_TARGET_COL = "target_4h_research_hold_exhaustion_short"
SETUP_QUALITY_LONG_TARGET_COL = "target_4h_setup_quality_long"
SETUP_QUALITY_SHORT_TARGET_COL = "target_4h_setup_quality_short"
SETUP_QUALITY_TREND_LONG_TARGET_COL = "target_4h_setup_quality_trend_long"
SETUP_QUALITY_BREAKOUT_LONG_TARGET_COL = "target_4h_setup_quality_breakout_long"

LONG_SETUP_SCORE_COLS = (
    "setup_trend_long_score",
    "setup_breakout_long_score",
    "setup_rebound_long_score",
)
SHORT_SETUP_SCORE_COLS = (
    "setup_trend_short_score",
    "setup_exhaustion_short_score",
)
MAJOR_SYMBOLS = {"BTC", "ETH", "SOL", "XRP", "BCH", "TRX"}

RESEARCH_SLICE_PROFILE_MAJORS_NON_FRAGILE = "majors_non_fragile"
RESEARCH_SLICE_PROFILE_MAJORS_POSITIVE_LONG_SLICES = "majors_positive_long_slices"

ROUND_TRIP_COST_PROXY = 0.0040
EXECUTION_REALISM_HAIRCUT = 0.0030
LONG_NET_RET_FLOOR = 0.0025
SHORT_NET_RET_FLOOR = 0.0025
MIN_DIRECTIONAL_POSITIVES = 250
MAX_POSITIVE_RATE = 0.20
RESEARCH_HOLD_ENABLE_REBOUND_LONG = False


def _series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return df[col].fillna(default)


def _max_available(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[available].fillna(0.0).max(axis=1)


def _major_symbol_mask(df: pd.DataFrame) -> pd.Series:
    if "symbol" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    tokens = df["symbol"].fillna("").astype(str).str.split("-").str[0].str.upper()
    return tokens.isin(MAJOR_SYMBOLS)


def apply_research_rebuild_slice(
    df: pd.DataFrame,
    *,
    slice_profile: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Apply non-leaky research-only row filters for rebuild experiments."""
    profile_name = str(slice_profile or "").strip().lower()
    before_rows = int(len(df))
    if not profile_name:
        return df.copy(), {
            "profile": "all_rows",
            "active": False,
            "before_rows": before_rows,
            "kept_rows": before_rows,
            "kept_ratio": 1.0,
        }

    major_mask = _major_symbol_mask(df)
    non_fragile_mask = _series(df, "fakeout_risk") < 0.65
    mask = major_mask & non_fragile_mask
    rules = {
        "major_symbols": sorted(MAJOR_SYMBOLS),
        "fragile_fakeout_cap": 0.65,
    }

    if profile_name == RESEARCH_SLICE_PROFILE_MAJORS_POSITIVE_LONG_SLICES:
        long_core = pd.concat(
            [
                _series(df, "setup_trend_long_score").rename("trend_long"),
                _series(df, "setup_breakout_long_score").rename("breakout_long"),
            ],
            axis=1,
        ).max(axis=1)
        short_core = _max_available(df, SHORT_SETUP_SCORE_COLS)
        adx = _series(df, "adx")
        mask &= (
            (adx >= 18.0)
            & (long_core >= 0.67)
            & (long_core >= short_core)
        )
        rules.update({
            "min_adx": 18.0,
            "min_long_core_setup_score": 0.67,
            "require_long_setup_not_weaker_than_short": True,
            "allowed_long_setup_family": ["trend_long", "breakout_long"],
        })
    elif profile_name != RESEARCH_SLICE_PROFILE_MAJORS_NON_FRAGILE:
        raise ValueError(f"unknown research slice profile: {slice_profile}")

    filtered = df.loc[mask].copy().reset_index(drop=True)
    kept_rows = int(mask.sum())
    return filtered, {
        "profile": profile_name,
        "active": True,
        "before_rows": before_rows,
        "kept_rows": kept_rows,
        "kept_ratio": round((kept_rows / before_rows), 6) if before_rows else 0.0,
        "rules": rules,
        "support": {
            "major_rows": int(major_mask.sum()),
            "non_fragile_rows": int(non_fragile_mask.sum()),
        },
    }


def apply_research_hold_target_profile(
    df: pd.DataFrame,
    *,
    horizon: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """Build materially stricter setup-conditioned composite targets.

    This path is intentionally different from the default "price up/down in 4h"
    labels. It only rewards rows where a specific setup was present *and* the
    realized move survived a stronger quality filter.
    """
    target_h = f"target_{horizon}h"
    target_short_h = f"target_{horizon}h_short"
    target_trend_long = f"target_{horizon}h_trend_long"
    target_breakout_long = f"target_{horizon}h_breakout_long"
    target_rebound_long = f"target_{horizon}h_rebound_long"
    target_trend_short = f"target_{horizon}h_trend_short"
    target_exhaustion_short = f"target_{horizon}h_exhaustion_short"
    fwd_ret_col = f"fwd_ret_{horizon}h"

    out = df.copy()

    long_setup_max = _max_available(out, LONG_SETUP_SCORE_COLS)
    short_setup_max = _max_available(out, SHORT_SETUP_SCORE_COLS)

    setup_trend_long_score = _series(out, "setup_trend_long_score")
    setup_breakout_long_score = _series(out, "setup_breakout_long_score")
    setup_rebound_long_score = _series(out, "setup_rebound_long_score")
    setup_trend_short_score = _series(out, "setup_trend_short_score")
    setup_exhaustion_short_score = _series(out, "setup_exhaustion_short_score")

    fakeout_risk = _series(out, "fakeout_risk")
    squeeze_risk = _series(out, "squeeze_risk")
    turnover_z = _series(out, "turnover_z_48h")
    adx = _series(out, "adx")
    upper_wick = _series(out, "upper_wick_pct")
    fomo_score = _series(out, "fomo_score")
    bb_pct_b = _series(out, "bb_pct_b", 0.5)
    long_fwd_ret = _series(out, fwd_ret_col)
    long_net_ret = long_fwd_ret - (ROUND_TRIP_COST_PROXY + EXECUTION_REALISM_HAIRCUT)
    short_net_ret = (-long_fwd_ret) - (ROUND_TRIP_COST_PROXY + EXECUTION_REALISM_HAIRCUT)
    out[RESEARCH_HOLD_LONG_NET_RET_COL] = long_net_ret
    out[RESEARCH_HOLD_SHORT_NET_RET_COL] = short_net_ret

    trend_long_ok = (
        (_series(out, target_trend_long) > 0)
        & (setup_trend_long_score >= 0.67)
        & (adx >= 18)
        & (fakeout_risk <= 0.55)
        & (long_net_ret >= LONG_NET_RET_FLOOR)
    )
    breakout_long_ok = (
        (_series(out, target_breakout_long) > 0)
        & (setup_breakout_long_score >= 0.68)
        & (fakeout_risk <= 0.42)
        & (turnover_z >= -0.25)
        & (bb_pct_b >= 0.65)
        & (long_net_ret >= LONG_NET_RET_FLOOR + 0.001)
    )
    if RESEARCH_HOLD_ENABLE_REBOUND_LONG:
        rebound_long_ok = (
            (_series(out, target_rebound_long) > 0)
            & (setup_rebound_long_score >= 0.74)
            & (squeeze_risk <= 0.45)
            & (bb_pct_b <= 0.40)
            & (turnover_z >= 0.0)
            & (long_net_ret >= LONG_NET_RET_FLOOR + 0.0015)
        )
    else:
        rebound_long_ok = pd.Series(False, index=out.index, dtype=bool)

    trend_short_ok = (
        (_series(out, target_trend_short) > 0)
        & (setup_trend_short_score >= 0.67)
        & (adx >= 18)
        & (short_net_ret >= SHORT_NET_RET_FLOOR)
    )
    exhaustion_short_ok = (
        (_series(out, target_exhaustion_short) > 0)
        & (setup_exhaustion_short_score >= 0.66)
        & (upper_wick >= 0.18)
        & (fomo_score >= 0.02)
        & (short_net_ret >= SHORT_NET_RET_FLOOR)
    )

    out[RESEARCH_HOLD_TREND_LONG_TARGET_COL] = trend_long_ok.astype(float)
    out[RESEARCH_HOLD_BREAKOUT_LONG_TARGET_COL] = breakout_long_ok.astype(float)
    out[RESEARCH_HOLD_REBOUND_LONG_TARGET_COL] = rebound_long_ok.astype(float)
    out[RESEARCH_HOLD_TREND_SHORT_TARGET_COL] = trend_short_ok.astype(float)
    out[RESEARCH_HOLD_EXHAUSTION_SHORT_TARGET_COL] = exhaustion_short_ok.astype(float)
    out[RESEARCH_HOLD_LONG_TARGET_COL] = (
        (long_setup_max >= 0.62)
        & (trend_long_ok | breakout_long_ok | rebound_long_ok)
    ).astype(float)
    out[RESEARCH_HOLD_SHORT_TARGET_COL] = (
        (short_setup_max >= 0.60)
        & (trend_short_ok | exhaustion_short_ok)
    ).astype(float)

    long_positive_rate = float(out[RESEARCH_HOLD_LONG_TARGET_COL].mean())
    short_positive_rate = float(out[RESEARCH_HOLD_SHORT_TARGET_COL].mean())
    long_positive_count = int(out[RESEARCH_HOLD_LONG_TARGET_COL].sum())
    short_positive_count = int(out[RESEARCH_HOLD_SHORT_TARGET_COL].sum())
    long_ready = (
        long_positive_count >= MIN_DIRECTIONAL_POSITIVES
        and 0.0 < long_positive_rate <= MAX_POSITIVE_RATE
    )
    short_ready = (
        short_positive_count >= MIN_DIRECTIONAL_POSITIVES
        and 0.0 < short_positive_rate <= MAX_POSITIVE_RATE
    )

    profile = {
        "profile": "research_hold_setup_composite",
        "horizon_hours": int(horizon),
        "long_target_col": RESEARCH_HOLD_LONG_TARGET_COL,
        "short_target_col": RESEARCH_HOLD_SHORT_TARGET_COL,
        "setup_target_cols": {
            "trend_long": RESEARCH_HOLD_TREND_LONG_TARGET_COL,
            "breakout_long": RESEARCH_HOLD_BREAKOUT_LONG_TARGET_COL,
            "rebound_long": RESEARCH_HOLD_REBOUND_LONG_TARGET_COL,
            "trend_short": RESEARCH_HOLD_TREND_SHORT_TARGET_COL,
            "exhaustion_short": RESEARCH_HOLD_EXHAUSTION_SHORT_TARGET_COL,
        },
        "base_target_col": target_h,
        "base_short_target_col": target_short_h,
        "fwd_ret_col": fwd_ret_col,
        "rules": {
            "long_min_setup_score": 0.62,
            "short_min_setup_score": 0.60,
            "round_trip_cost_proxy": ROUND_TRIP_COST_PROXY,
            "execution_realism_haircut": EXECUTION_REALISM_HAIRCUT,
            "long_net_ret_floor": LONG_NET_RET_FLOOR,
            "short_net_ret_floor": SHORT_NET_RET_FLOOR,
            "breakout_fakeout_cap": 0.42,
            "rebound_max_squeeze_risk": 0.45,
            "rebound_max_bb_pct_b": 0.40,
            "enable_rebound_long": RESEARCH_HOLD_ENABLE_REBOUND_LONG,
        },
        "class_balance": {
            "base_long_positive_rate": round(float(_series(out, target_h).mean()), 6),
            "research_long_positive_rate": round(long_positive_rate, 6),
            "base_short_positive_rate": round(float(_series(out, target_short_h).mean()), 6),
            "research_short_positive_rate": round(short_positive_rate, 6),
        },
        "support_counts": {
            "long_total": int(len(out)),
            "long_positive": long_positive_count,
            "short_positive": short_positive_count,
            "long_trend_positive": int(trend_long_ok.sum()),
            "long_breakout_positive": int(breakout_long_ok.sum()),
            "long_rebound_positive": int(rebound_long_ok.sum()),
            "short_trend_positive": int(trend_short_ok.sum()),
            "short_exhaustion_positive": int(exhaustion_short_ok.sum()),
        },
        "setup_target_summary": {
            "trend_long_positive_rate": round(float(out[RESEARCH_HOLD_TREND_LONG_TARGET_COL].mean()), 6),
            "breakout_long_positive_rate": round(float(out[RESEARCH_HOLD_BREAKOUT_LONG_TARGET_COL].mean()), 6),
            "rebound_long_positive_rate": round(float(out[RESEARCH_HOLD_REBOUND_LONG_TARGET_COL].mean()), 6),
            "trend_short_positive_rate": round(float(out[RESEARCH_HOLD_TREND_SHORT_TARGET_COL].mean()), 6),
            "exhaustion_short_positive_rate": round(float(out[RESEARCH_HOLD_EXHAUSTION_SHORT_TARGET_COL].mean()), 6),
        },
        "viability": {
            "min_directional_positives": MIN_DIRECTIONAL_POSITIVES,
            "max_positive_rate": MAX_POSITIVE_RATE,
            "long_ready": long_ready,
            "short_ready": short_ready,
            "overall_status": (
                "ready"
                if long_ready and short_ready
                else "partial" if long_ready or short_ready
                else "too_sparse"
            ),
        },
    }
    return out, profile


def apply_setup_quality_target_profile(
    df: pd.DataFrame,
    *,
    horizon: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """Build a labeled-setup recovery profile for the setup-quality lane.

    This profile keeps the short target tradable, but redefines the long target
    around labeled trend/breakout rows with stronger setup confirmation so the
    next retrain learns away from generic-long churn.
    """
    target_h = f"target_{horizon}h"
    target_short_h = f"target_{horizon}h_short"
    fwd_ret_col = f"fwd_ret_{horizon}h"

    out = df.copy()
    long_setup_max = _max_available(out, LONG_SETUP_SCORE_COLS)
    short_setup_max = _max_available(out, SHORT_SETUP_SCORE_COLS)

    setup_trend_long_score = _series(out, "setup_trend_long_score")
    setup_breakout_long_score = _series(out, "setup_breakout_long_score")
    setup_trend_short_score = _series(out, "setup_trend_short_score")
    setup_exhaustion_short_score = _series(out, "setup_exhaustion_short_score")

    fakeout_risk = _series(out, "fakeout_risk")
    turnover_z = _series(out, "turnover_z_48h")
    adx = _series(out, "adx")
    bb_pct_b = _series(out, "bb_pct_b", 0.5)
    upper_wick = _series(out, "upper_wick_pct")
    fomo_score = _series(out, "fomo_score")
    long_fwd_ret = _series(out, fwd_ret_col)
    long_net_ret = long_fwd_ret - (ROUND_TRIP_COST_PROXY + EXECUTION_REALISM_HAIRCUT)
    short_net_ret = (-long_fwd_ret) - (ROUND_TRIP_COST_PROXY + EXECUTION_REALISM_HAIRCUT)

    trend_long_ok = (
        (_series(out, target_h) > 0)
        & (setup_trend_long_score >= 0.60)
        & (setup_trend_long_score >= short_setup_max)
        & (adx >= 16.0)
        & (fakeout_risk <= 0.62)
        & (long_net_ret >= LONG_NET_RET_FLOOR)
    )
    breakout_long_ok = (
        (_series(out, target_h) > 0)
        & (setup_breakout_long_score >= 0.62)
        & (setup_breakout_long_score >= short_setup_max)
        & (fakeout_risk <= 0.50)
        & (turnover_z >= -0.35)
        & (bb_pct_b >= 0.58)
        & (long_net_ret >= LONG_NET_RET_FLOOR)
    )
    short_ok = (
        (_series(out, target_short_h) > 0)
        & (
            ((setup_trend_short_score >= 0.60) & (short_net_ret >= SHORT_NET_RET_FLOOR))
            | (
                (setup_exhaustion_short_score >= 0.60)
                & (upper_wick >= 0.16)
                & (fomo_score >= 0.015)
                & (short_net_ret >= SHORT_NET_RET_FLOOR)
            )
        )
    )

    out[SETUP_QUALITY_TREND_LONG_TARGET_COL] = trend_long_ok.astype(float)
    out[SETUP_QUALITY_BREAKOUT_LONG_TARGET_COL] = breakout_long_ok.astype(float)
    out[SETUP_QUALITY_LONG_TARGET_COL] = (
        (long_setup_max >= 0.58)
        & (trend_long_ok | breakout_long_ok)
    ).astype(float)
    out[SETUP_QUALITY_SHORT_TARGET_COL] = (
        ((short_setup_max >= 0.58) & short_ok)
        | ((_series(out, target_short_h) > 0) & (short_net_ret >= SHORT_NET_RET_FLOOR + 0.0005))
    ).astype(float)

    long_positive_rate = float(out[SETUP_QUALITY_LONG_TARGET_COL].mean())
    short_positive_rate = float(out[SETUP_QUALITY_SHORT_TARGET_COL].mean())
    long_positive_count = int(out[SETUP_QUALITY_LONG_TARGET_COL].sum())
    short_positive_count = int(out[SETUP_QUALITY_SHORT_TARGET_COL].sum())

    profile = {
        "profile": "setup_quality_labeled_directional",
        "horizon_hours": int(horizon),
        "long_target_col": SETUP_QUALITY_LONG_TARGET_COL,
        "short_target_col": SETUP_QUALITY_SHORT_TARGET_COL,
        "setup_target_cols": {
            "trend_long": SETUP_QUALITY_TREND_LONG_TARGET_COL,
            "breakout_long": SETUP_QUALITY_BREAKOUT_LONG_TARGET_COL,
        },
        "base_target_col": target_h,
        "base_short_target_col": target_short_h,
        "fwd_ret_col": fwd_ret_col,
        "rules": {
            "long_min_setup_score": 0.58,
            "trend_long_min_setup_score": 0.60,
            "breakout_long_min_setup_score": 0.62,
            "trend_long_min_adx": 16.0,
            "breakout_long_fakeout_cap": 0.50,
            "round_trip_cost_proxy": ROUND_TRIP_COST_PROXY,
            "execution_realism_haircut": EXECUTION_REALISM_HAIRCUT,
            "long_net_ret_floor": LONG_NET_RET_FLOOR,
            "short_net_ret_floor": SHORT_NET_RET_FLOOR,
        },
        "class_balance": {
            "base_long_positive_rate": round(float(_series(out, target_h).mean()), 6),
            "profile_long_positive_rate": round(long_positive_rate, 6),
            "base_short_positive_rate": round(float(_series(out, target_short_h).mean()), 6),
            "profile_short_positive_rate": round(short_positive_rate, 6),
        },
        "support_counts": {
            "long_total": int(len(out)),
            "long_positive": long_positive_count,
            "short_positive": short_positive_count,
            "long_trend_positive": int(trend_long_ok.sum()),
            "long_breakout_positive": int(breakout_long_ok.sum()),
            "short_quality_positive": int(short_ok.sum()),
        },
        "viability": {
            "min_directional_positives": MIN_DIRECTIONAL_POSITIVES,
            "max_positive_rate": MAX_POSITIVE_RATE,
            "long_ready": long_positive_count >= MIN_DIRECTIONAL_POSITIVES and 0.0 < long_positive_rate <= MAX_POSITIVE_RATE,
            "short_ready": short_positive_count >= MIN_DIRECTIONAL_POSITIVES and 0.0 < short_positive_rate <= MAX_POSITIVE_RATE,
        },
    }
    return out, profile
