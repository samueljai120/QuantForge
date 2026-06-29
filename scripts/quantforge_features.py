#!/usr/bin/env python3
"""QuantForge — Feature Engineering

Loads historical OHLCV CSVs and computes ML features for each candle.
Output: ~/quantforge/data/quantforge/features/<PAIR>_features.parquet

Features computed (per 1h candle, using 4h + 1d for context):
  - Returns: 1h, 4h, 12h, 24h, 48h, 168h
  - RSI: 7, 14, 21 period
  - MACD: line, signal, histogram (12/26/9)
  - Bollinger Bands: %B position, bandwidth (20 period)
  - EMAs: 8, 21, 55, 200 — distance from price, crossover states
  - ATR: 14 period (normalized by price)
  - Volume: ratio to 20-period avg, OBV direction
  - Stochastic: %K, %D (14/3)
  - Trend: ADX (14), +DI, -DI
  - Multi-timeframe context from 4h + 1d (RSI, EMA position, trend)
  - Coin behavior profile features (trendiness, mean reversion, crash/rebound tendency)
  - Move-type state features (panic flush, breakout, rebound, fakeout, squeeze risk)
  - Cross-asset context from BTC + ETH
  - Crowd/whale proxy features from turnover, range expansion, and impulse candles
  - Target: price up after 4h? (binary classification label)

Usage:
    python3 quantforge_features.py build          # Build all pairs
    python3 quantforge_features.py build BTC-USDT # Single pair
    python3 quantforge_features.py research-tsfresh # Build optional research-stage tsfresh features
    python3 quantforge_features.py status         # Show output files
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_params import load_merged_quantforge_params


def _load_strategy_params():
    return load_merged_quantforge_params()

HISTORICAL_DIR = os.path.join(cfg.data, "quantforge", "historical")
FEATURES_DIR = os.path.join(cfg.data, "quantforge", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)
BOOK_HISTORY_FILE = os.path.join(cfg.data, "quantforge", "book", "book_snapshot_history.jsonl")
DERIV_HISTORY_FILE = os.path.join(cfg.data, "quantforge", "derivatives", "derivatives_state_history.jsonl")
BREADTH_HISTORY_FILE = os.path.join(cfg.data, "quantforge", "breadth", "breadth_context_history.jsonl")
EVENT_HISTORY_FILE = os.path.join(cfg.data, "quantforge", "events", "event_flags_history.jsonl")
MICRO_HISTORY_FILE = os.path.join(cfg.data, "quantforge", "microstructure", "trade_tape_proxy_history.jsonl")

TARGET_HORIZONS = [1, 2, 4, 8]  # Hours ahead to predict
PRIMARY_TARGET = 4               # Main target horizon for training
MIN_1H_CANDLES = int(os.environ.get("QF_MIN_1H_CANDLES", "500"))
_sp = _load_strategy_params()
SHORT_TARGET_DROP_PCT = float(os.environ.get(
    "QF_SHORT_TARGET_DROP_PCT",
    _sp.get("short_target_drop_pct", 0.025),
))
BENCHMARK_SYMBOLS = ("BTC-USDT", "ETH-USDT")


def ROUND_TRIP_PROXY(hours_ahead: int) -> float:
    """Simple hurdle rate proxy for setup-conditioned target labeling.

    We use a slightly wider hurdle than raw zero-return so setup labels reflect
    moves that are more likely to survive fees/slippage in practice.
    """
    base = 0.0025
    return base if hours_ahead <= 4 else base * 1.4


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _dedupe_latest(rows, key_fields):
    ordered = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        ordered[key] = row
    return list(ordered.values())


def _prepare_symbol_context(rows, keep_fields):
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    if frame.empty or "timestamp" not in frame.columns or "symbol" not in frame.columns:
        return None
    cols = ["timestamp", "symbol"] + [field for field in keep_fields if field in frame.columns]
    frame = frame[cols].copy()
    frame.rename(columns={"timestamp": "ts"}, inplace=True)
    frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values(["symbol", "ts"]).drop_duplicates(subset=["symbol", "ts"], keep="last")
    return frame


def _prepare_market_context(rows, keep_fields):
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    if frame.empty or "timestamp" not in frame.columns:
        return None
    cols = ["timestamp"] + [field for field in keep_fields if field in frame.columns]
    frame = frame[cols].copy()
    frame.rename(columns={"timestamp": "ts"}, inplace=True)
    frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return frame


def _merge_symbol_context(df, context):
    if context is None or context.empty:
        return df
    merged = pd.merge_asof(
        df.sort_values("ts"),
        context.sort_values("ts"),
        on="ts",
        by="symbol",
        direction="backward",
    )
    return merged


def _merge_market_context(df, context):
    if context is None or context.empty:
        return df
    merged = pd.merge_asof(
        df.sort_values("ts"),
        context.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    return merged


def _merge_prefixed_market_context(df, context, prefix):
    if context is None or context.empty:
        return df
    renamed = context.copy()
    rename_map = {col: f"{prefix}_{col}" for col in renamed.columns if col != "ts"}
    renamed = renamed.rename(columns=rename_map)
    return _merge_market_context(df, renamed)


def _fill_feature_column(df, col, fallback):
    if col not in df.columns:
        df[col] = fallback
    else:
        df[col] = df[col].fillna(fallback)
    return df


def load_rebuild_context():
    book_rows = _dedupe_latest(_read_jsonl(BOOK_HISTORY_FILE), ("timestamp", "symbol"))
    derivatives_rows = _dedupe_latest(_read_jsonl(DERIV_HISTORY_FILE), ("timestamp", "symbol"))
    breadth_rows = _dedupe_latest(_read_jsonl(BREADTH_HISTORY_FILE), ("timestamp",))
    micro_rows = _dedupe_latest(_read_jsonl(MICRO_HISTORY_FILE), ("timestamp", "symbol"))
    event_rows = _dedupe_latest(_read_jsonl(EVENT_HISTORY_FILE), ("timestamp", "event_type", "symbol_scope"))

    event_market = []
    for row in event_rows:
        symbol_scope = row.get("symbol_scope")
        if symbol_scope in ("market", None, ""):
            event_market.append(row)
            continue
        event_market.append(
            {
                "timestamp": row.get("timestamp"),
                "event_active_any": row.get("event_active", 0),
                "event_score_any": row.get("event_score", 0.0),
                "hours_to_event_min": row.get("hours_to_event", 999.0),
            }
        )
    if event_market:
        event_frame = pd.DataFrame(event_market)
        if not event_frame.empty and "timestamp" in event_frame.columns:
            agg = event_frame.groupby("timestamp", as_index=False).agg(
                event_active_any=("event_active_any", "max"),
                event_score_any=("event_score_any", "max"),
                hours_to_event_min=("hours_to_event_min", "min"),
            )
            event_market_rows = agg.to_dict("records")
        else:
            event_market_rows = []
    else:
        event_market_rows = []

    return {
        "book": _prepare_symbol_context(
            book_rows,
            [
                "quoted_spread_bps",
                "effective_spread_proxy_bps",
                "book_depth_proxy",
                "top5_depth_imbalance",
                "expected_impact_10k_bps",
                "expected_impact_25k_bps",
                "book_staleness_seconds",
            ],
        ),
        "derivatives": _prepare_symbol_context(
            derivatives_rows,
            [
                "funding_rate",
                "open_interest",
                "basis_bps",
                "long_short_ratio",
                "liquidation_long_usd",
                "liquidation_short_usd",
                "crowding_score",
            ],
        ),
        "micro": _prepare_symbol_context(
            micro_rows,
            [
                "aggressive_buy_volume_proxy",
                "aggressive_sell_volume_proxy",
                "micro_return_30s_proxy",
                "micro_vol_30s_proxy",
                "pressure_imbalance_proxy",
            ],
        ),
        "breadth": _prepare_market_context(
            breadth_rows,
            [
                "btc_return_1h",
                "eth_return_1h",
                "majors_breadth",
                "alts_breadth",
                "market_volume_breadth",
                "stablecoin_dominance_proxy",
                "majors_volume_breadth",
                "alts_volume_breadth",
                "majors_volume_share",
            ],
        ),
        "events": _prepare_market_context(
            event_market_rows,
            ["event_active_any", "event_score_any", "hours_to_event_min"],
        ),
    }


def apply_rebuild_context(df, symbol, rebuild_context):
    """Merge rebuild-stage external market context using backward-only joins.

    The external collectors are newer than much of the historical OHLCV, so we
    fall back to conservative local proxies where appropriate rather than
    letting the feature space become mostly sparse.
    """
    if not rebuild_context:
        return df

    out = df.copy()

    book_context = rebuild_context.get("book")
    if book_context is not None and not book_context.empty:
        symbol_book = book_context[book_context["symbol"] == symbol].drop(columns=["symbol"], errors="ignore")
        out = _merge_prefixed_market_context(out, symbol_book, "book")
    out = _fill_feature_column(out, "book_quoted_spread_bps", out["quoted_spread_proxy_bps"])
    out = _fill_feature_column(out, "book_effective_spread_proxy_bps", out["effective_spread_proxy_bps"])
    out = _fill_feature_column(out, "book_book_depth_proxy", out["book_depth_proxy"])
    out = _fill_feature_column(out, "book_top5_depth_imbalance", out["top5_depth_imbalance_proxy"])
    out = _fill_feature_column(out, "book_expected_impact_10k_bps", out["expected_impact_proxy_bps"])
    out = _fill_feature_column(out, "book_expected_impact_25k_bps", out["expected_impact_proxy_bps"] * 1.75)
    out = _fill_feature_column(out, "book_book_staleness_seconds", out["book_staleness_proxy"] * 60.0)

    derivatives_context = rebuild_context.get("derivatives")
    if derivatives_context is not None and not derivatives_context.empty:
        symbol_deriv = derivatives_context[derivatives_context["symbol"] == symbol].drop(columns=["symbol"], errors="ignore")
        out = _merge_prefixed_market_context(out, symbol_deriv, "deriv")
    for col in (
        "deriv_funding_rate",
        "deriv_open_interest",
        "deriv_basis_bps",
        "deriv_long_short_ratio",
        "deriv_liquidation_long_usd",
        "deriv_liquidation_short_usd",
        "deriv_crowding_score",
    ):
        out = _fill_feature_column(out, col, 0.0)

    micro_context = rebuild_context.get("micro")
    if micro_context is not None and not micro_context.empty:
        symbol_micro = micro_context[micro_context["symbol"] == symbol].drop(columns=["symbol"], errors="ignore")
        out = _merge_prefixed_market_context(out, symbol_micro, "micro")
    for col in (
        "micro_aggressive_buy_volume_proxy",
        "micro_aggressive_sell_volume_proxy",
        "micro_micro_return_30s_proxy",
        "micro_micro_vol_30s_proxy",
        "micro_pressure_imbalance_proxy",
    ):
        out = _fill_feature_column(out, col, 0.0)

    breadth_context = rebuild_context.get("breadth")
    out = _merge_prefixed_market_context(out, breadth_context, "breadth")
    for col in (
        "breadth_btc_return_1h",
        "breadth_eth_return_1h",
        "breadth_majors_breadth",
        "breadth_alts_breadth",
        "breadth_market_volume_breadth",
        "breadth_stablecoin_dominance_proxy",
        "breadth_majors_volume_breadth",
        "breadth_alts_volume_breadth",
        "breadth_majors_volume_share",
    ):
        out = _fill_feature_column(out, col, 0.0)

    events_context = rebuild_context.get("events")
    out = _merge_prefixed_market_context(out, events_context, "event")
    out = _fill_feature_column(out, "event_event_active_any", 0.0)
    out = _fill_feature_column(out, "event_event_score_any", 0.0)
    out = _fill_feature_column(out, "event_hours_to_event_min", 999.0)

    return out


# ---------------------------------------------------------------------------
# Indicator functions (vectorized with pandas)
# ---------------------------------------------------------------------------

def add_returns(df):
    """Price returns at multiple lookbacks."""
    for h in [1, 4, 12, 24, 48, 168]:
        df[f"ret_{h}h"] = df["close"].pct_change(h)
    df["log_ret_1h"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_rsi(df, periods=(7, 14, 21)):
    """RSI for multiple periods."""
    delta = df["close"].diff()
    for p in periods:
        gain = delta.clip(lower=0).ewm(com=p-1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
        # Normalized: distance from 50
        df[f"rsi_{p}_norm"] = (df[f"rsi_{p}"] - 50) / 50
    return df


def add_macd(df):
    """MACD (12/26/9)."""
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    # Normalized by price
    df["macd_line_norm"] = df["macd_line"] / df["close"]
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]
    # Histogram direction change
    df["macd_hist_dir"] = np.sign(df["macd_hist"])
    df["macd_hist_cross"] = (df["macd_hist_dir"] != df["macd_hist_dir"].shift(1)).astype(int)
    return df


def add_bollinger(df, period=20):
    """Bollinger Bands: %B and bandwidth."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    df["bb_pct_b"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    df["bb_width"] = (upper - lower) / sma.replace(0, np.nan)
    return df


def add_emas(df, periods=(8, 21, 55, 200)):
    """EMA distances and crossover states."""
    ema_vals = {}
    for p in periods:
        col = f"ema_{p}"
        ema_vals[p] = df["close"].ewm(span=p, adjust=False).mean()
        # Distance from price (normalized)
        df[f"ema_{p}_dist"] = (df["close"] - ema_vals[p]) / ema_vals[p].replace(0, np.nan)

    # Crossover states: only for pairs that exist in the requested periods
    all_cross_pairs = [(8, 21), (21, 55), (55, 200)]
    cross_pairs = [(s, l) for s, l in all_cross_pairs if s in periods and l in periods]
    for short, long in cross_pairs:
        col = f"ema_{short}_{long}_above"
        df[col] = (ema_vals[short] > ema_vals[long]).astype(int)
        # Just crossed?
        prev = (ema_vals[short].shift(1) > ema_vals[long].shift(1))
        df[f"ema_{short}_{long}_cross_up"] = ((df[col] == 1) & (prev == False)).astype(int)
        df[f"ema_{short}_{long}_cross_dn"] = ((df[col] == 0) & (prev == True)).astype(int)

    # Price position: above EMA200 (only if 200 in periods)
    if 200 in ema_vals:
        df["above_ema200"] = (df["close"] > ema_vals[200]).astype(int)
    return df


def add_atr(df, period=14):
    """Average True Range (normalized)."""
    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - df["close"].shift(1)).abs()
    low_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    df["atr_norm"] = atr / df["close"]  # Normalized ATR
    # ATR ratio: current vs 30-period ATR (volatility regime)
    atr_slow = tr.ewm(span=30, adjust=False).mean()
    df["atr_ratio"] = atr / atr_slow.replace(0, np.nan)
    return df


def add_volume_features(df):
    """Volume indicators."""
    vol_ma20 = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / vol_ma20.replace(0, np.nan)

    # OBV direction (5-period EMA of OBV)
    obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    obv_ema = obv.ewm(span=5, adjust=False).mean()
    df["obv_trend"] = np.sign(obv_ema.diff(3))

    # Price × volume surge
    df["turnover_ratio"] = df["turnover"] / df["turnover"].rolling(20).mean().replace(0, np.nan)
    return df


def add_stochastic(df, k_period=14, d_period=3):
    """Stochastic oscillator %K and %D."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    df["stoch_k_norm"] = (df["stoch_k"] - 50) / 50
    return df


def add_adx(df, period=14):
    """ADX (trend strength) + DI lines."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out where the other DM is larger
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2 = minus_dm < high.diff().clip(lower=0)
    minus_dm[mask2] = 0

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(span=period, adjust=False).mean()
    df["plus_di_norm"] = (plus_di - minus_di) / 100  # Normalized DI spread
    return df


def add_candle_patterns(df):
    """Simple candle pattern features."""
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body_pct"] = body / range_
    df["candle_direction"] = np.sign(df["close"] - df["open"])
    # Upper/lower wick ratio
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    df["upper_wick_pct"] = upper_wick / range_
    df["lower_wick_pct"] = lower_wick / range_
    # 3-candle pattern: consecutive direction
    df["dir_streak"] = df["candle_direction"].rolling(3).sum() / 3
    return df


def add_behavior_profile_features(df):
    """Rolling behavior profile using only historically realized price action."""
    signed_ret = np.sign(df["ret_1h"].fillna(0))
    df["behavior_trend_persist_24h"] = (signed_ret == signed_ret.shift(1)).astype(float).rolling(24).mean()
    df["behavior_mean_revert_24h"] = (signed_ret == -signed_ret.shift(1)).astype(float).rolling(24).mean()

    down_shock = (df["ret_4h"].shift(1) < -0.05).astype(float)
    up_shock = (df["ret_4h"].shift(1) > 0.05).astype(float)
    rebound_realized = ((df["ret_4h"].shift(1) < -0.05) & (df["ret_1h"] > 0)).astype(float)
    continuation_realized = ((df["ret_4h"].shift(1) < -0.05) & (df["ret_1h"] < 0)).astype(float)
    breakout_follow = ((df["ret_24h"].shift(1) > 0.08) & (df["ret_1h"] > 0)).astype(float)
    spike_fade = ((df["ret_24h"].shift(1) > 0.08) & (df["ret_1h"] < 0)).astype(float)

    df["behavior_crash_rebound_30d"] = rebound_realized.rolling(24 * 30).sum() / down_shock.rolling(24 * 30).sum().replace(0, np.nan)
    df["behavior_crash_continue_30d"] = continuation_realized.rolling(24 * 30).sum() / down_shock.rolling(24 * 30).sum().replace(0, np.nan)
    df["behavior_breakout_follow_30d"] = breakout_follow.rolling(24 * 30).sum() / up_shock.rolling(24 * 30).sum().replace(0, np.nan)
    df["behavior_spike_fade_30d"] = spike_fade.rolling(24 * 30).sum() / up_shock.rolling(24 * 30).sum().replace(0, np.nan)
    return df


def add_flow_and_move_features(df):
    """Proxy features for crowd FOMO, large-flow candles, and move classification."""
    rolling_ret_std = df["ret_1h"].rolling(48).std().replace(0, np.nan)
    rolling_turn_mean = df["turnover"].rolling(48).mean()
    rolling_turn_std = df["turnover"].rolling(48).std().replace(0, np.nan)
    candle_range_pct = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    df["turnover_z_48h"] = (df["turnover"] - rolling_turn_mean) / rolling_turn_std
    df["range_pct"] = candle_range_pct
    df["range_z_48h"] = (candle_range_pct - candle_range_pct.rolling(48).mean()) / candle_range_pct.rolling(48).std().replace(0, np.nan)
    df["impulse_body_signed"] = ((df["close"] - df["open"]) / df["open"].replace(0, np.nan)) * df["candle_body_pct"]

    # Crowd/FOMO proxy: strong upside momentum + volume/turnover expansion + close near highs.
    df["fomo_score"] = (
        0.35 * df["ret_4h"].clip(-0.25, 0.25)
        + 0.25 * df["ret_24h"].clip(-0.5, 0.5)
        + 0.20 * (df["vol_ratio"] - 1.0).clip(-2, 4)
        + 0.20 * (df["bb_pct_b"] - 0.5).clip(-1, 1)
    )

    # Whale/large-flow proxy: abnormal turnover + range expansion + directional impulse.
    df["whale_flow_proxy"] = (
        0.45 * df["turnover_z_48h"].clip(-4, 6)
        + 0.30 * df["range_z_48h"].clip(-4, 6)
        + 0.25 * (df["impulse_body_signed"] * 100).clip(-5, 5)
    )

    # Move-type state features. These are soft indicators, not hard trade labels.
    df["panic_flush_signal"] = (
        (df["ret_4h"] < -0.08).astype(float)
        + (df["ret_24h"] < -0.15).astype(float)
        + (df["rsi_14"] < 35).astype(float)
        + (df["atr_ratio"] > 1.4).astype(float)
        + (df["turnover_z_48h"] > 1.5).astype(float)
    ) / 5.0
    df["rebound_signal"] = (
        (df["ret_4h"] < -0.08).astype(float)
        + (df["lower_wick_pct"] > 0.45).astype(float)
        + (df["candle_direction"] > 0).astype(float)
        + (df["rsi_14"] < 40).astype(float)
        + (df["bb_pct_b"] < 0.2).astype(float)
    ) / 5.0
    df["breakout_signal"] = (
        (df["ret_24h"] > 0.10).astype(float)
        + (df["vol_ratio"] > 1.5).astype(float)
        + (df["bb_pct_b"] > 0.8).astype(float)
        + (df["ema_21_55_above"] > 0).astype(float)
        + (df["adx"] > 20).astype(float)
    ) / 5.0
    df["fakeout_risk"] = (
        (df["upper_wick_pct"] > 0.40).astype(float)
        + (df["vol_ratio"] < 0.9).astype(float)
        + (df["ret_1h"] < 0).astype(float)
        + (df["breakout_signal"] > 0.5).astype(float)
    ) / 4.0
    df["squeeze_risk"] = (
        (df["atr_ratio"] > 1.2).astype(float)
        + (df["turnover_z_48h"] > 1.0).astype(float)
        + (df["dir_streak"].abs() > 0.66).astype(float)
        + (df["adx"] > 25).astype(float)
    ) / 4.0
    return df


def add_execution_quality_features(df):
    """Cheap execution-quality proxies until real book snapshots exist.

    These features are intentionally conservative approximations from OHLCV so
    the model and rebuild reports can reason about fill quality before the
    proper top-of-book lane is online.
    """
    price = df["close"].replace(0, np.nan)
    rng = (df["high"] - df["low"]).clip(lower=0)
    turnover = df["turnover"].clip(lower=0)
    volume = df["volume"].clip(lower=0)
    body = (df["close"] - df["open"]).abs()

    # Proxy quoted spread from the intrabar range and wickiness.
    df["quoted_spread_proxy_bps"] = ((rng / price) * 10000).clip(0, 500)
    df["effective_spread_proxy_bps"] = (
        0.6 * df["quoted_spread_proxy_bps"].fillna(0)
        + 0.4 * (df["candle_body_pct"].fillna(0) * 10000)
    ).clip(0, 500)

    # Depth/impact proxies: how much turnover is needed to move price.
    notional_per_range = turnover / rng.replace(0, np.nan)
    df["book_depth_proxy"] = np.log1p(notional_per_range.clip(lower=0))
    df["top5_depth_imbalance_proxy"] = (
        (df["turnover_ratio"].fillna(1.0) - 1.0).clip(-3, 5) * 0.35
        + (df["vol_ratio"].fillna(1.0) - 1.0).clip(-3, 5) * 0.35
        - df["range_z_48h"].fillna(0).clip(-4, 6) * 0.30
    )

    impact_base = (body / price.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    impact_denom = np.log1p(turnover).replace(0, np.nan)
    df["expected_impact_proxy_bps"] = (
        (impact_base / impact_denom).replace([np.inf, -np.inf], np.nan).fillna(0) * 100000
    ).clip(0, 500)

    # Staleness proxy: low turnover / low activity / narrow bars tend to be harder fills.
    df["book_staleness_proxy"] = (
        (1.0 / np.log1p(turnover).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        + (1.0 / np.log1p(volume).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        + (1.0 / (1.0 + df["range_pct"].fillna(0).clip(lower=0) * 100))
    ).clip(0, 5)
    return df


def add_setup_context_features(df):
    """Encode reusable market setup context so the model can learn by setup type.

    These are not direct trade signals. They are structured tags describing the
    kind of move currently underway, which helps the model distinguish:
      - trend continuation
      - panic/rebound mean reversion
      - breakout extension
      - blow-off / squeeze fade risk
      - downside continuation
    """
    adx = df["adx"].fillna(0)
    ret_4h = df["ret_4h"].fillna(0)
    ret_24h = df["ret_24h"].fillna(0)
    rsi = df["rsi_14"].fillna(50)
    bb = df["bb_pct_b"].fillna(0.5)
    vol_ratio = df["vol_ratio"].fillna(1.0)
    turnover_z = df["turnover_z_48h"].fillna(0)
    lower_wick = df["lower_wick_pct"].fillna(0)
    upper_wick = df["upper_wick_pct"].fillna(0)
    candle_dir = df["candle_direction"].fillna(0)
    above_ema200 = df.get("above_ema200", pd.Series(0, index=df.index)).fillna(0)
    ema_trend = df.get("ema_21_55_above", pd.Series(0, index=df.index)).fillna(0)
    fakeout_risk = df["fakeout_risk"].fillna(0)
    squeeze_risk = df["squeeze_risk"].fillna(0)
    panic_flush = df["panic_flush_signal"].fillna(0)
    rebound_signal = df["rebound_signal"].fillna(0)
    breakout_signal = df["breakout_signal"].fillna(0)
    fomo_score = df["fomo_score"].fillna(0)
    whale_flow = df["whale_flow_proxy"].fillna(0)
    crash_rebound_bias = df["behavior_crash_rebound_30d"].fillna(0.5)
    crash_continue_bias = df["behavior_crash_continue_30d"].fillna(0.5)
    breakout_follow_bias = df["behavior_breakout_follow_30d"].fillna(0.5)
    spike_fade_bias = df["behavior_spike_fade_30d"].fillna(0.5)
    trend_persist_bias = df["behavior_trend_persist_24h"].fillna(0.5)

    df["setup_trend_long_score"] = (
        0.30 * (ret_24h > 0.06).astype(float)
        + 0.20 * (ret_4h > 0.02).astype(float)
        + 0.15 * (adx > 22).astype(float)
        + 0.15 * (above_ema200 > 0).astype(float)
        + 0.10 * (ema_trend > 0).astype(float)
        + 0.10 * trend_persist_bias.clip(0, 1)
    )
    df["setup_breakout_long_score"] = (
        0.25 * breakout_signal.clip(0, 1)
        + 0.20 * (bb > 0.85).astype(float)
        + 0.15 * (vol_ratio > 1.3).astype(float)
        + 0.15 * (turnover_z > 1.0).astype(float)
        + 0.15 * breakout_follow_bias.clip(0, 1)
        + 0.10 * (fakeout_risk < 0.45).astype(float)
    )
    df["setup_rebound_long_score"] = (
        0.30 * rebound_signal.clip(0, 1)
        + 0.20 * (panic_flush > 0.55).astype(float)
        + 0.15 * (lower_wick > 0.35).astype(float)
        + 0.15 * (candle_dir > 0).astype(float)
        + 0.10 * crash_rebound_bias.clip(0, 1)
        + 0.10 * (rsi < 42).astype(float)
    )
    df["setup_trend_short_score"] = (
        0.30 * (ret_24h < -0.06).astype(float)
        + 0.20 * (ret_4h < -0.02).astype(float)
        + 0.15 * (adx > 22).astype(float)
        + 0.15 * (above_ema200 == 0).astype(float)
        + 0.10 * crash_continue_bias.clip(0, 1)
        + 0.10 * trend_persist_bias.clip(0, 1)
    )
    df["setup_exhaustion_short_score"] = (
        0.25 * squeeze_risk.clip(0, 1)
        + 0.20 * (fomo_score > 0.08).astype(float)
        + 0.20 * (whale_flow > 1.2).astype(float)
        + 0.15 * (upper_wick > 0.35).astype(float)
        + 0.10 * spike_fade_bias.clip(0, 1)
        + 0.10 * (rsi > 68).astype(float)
    )

    df["setup_trend_long"] = (df["setup_trend_long_score"] >= 0.60).astype(float)
    df["setup_breakout_long"] = (df["setup_breakout_long_score"] >= 0.60).astype(float)
    df["setup_rebound_long"] = (df["setup_rebound_long_score"] >= 0.60).astype(float)
    df["setup_trend_short"] = (df["setup_trend_short_score"] >= 0.60).astype(float)
    df["setup_exhaustion_short"] = (df["setup_exhaustion_short_score"] >= 0.60).astype(float)

    return df


def add_targets(df):
    """Forward return labels for multiple horizons + SHORT targets for futures."""
    for h in TARGET_HORIZONS:
        future_close = df["close"].shift(-h)
        df[f"target_{h}h"] = (future_close > df["close"]).astype(float)
        df[f"fwd_ret_{h}h"] = (future_close - df["close"]) / df["close"]
        # SHORT target: configurable down-move requirement for continuation setups.
        df[f"target_{h}h_short"] = (df[f"fwd_ret_{h}h"] < -SHORT_TARGET_DROP_PCT).astype(float)
        # Setup-conditioned realized labels for later segmented training/analysis.
        df[f"target_{h}h_trend_long"] = ((df["setup_trend_long"] > 0) & (df[f"fwd_ret_{h}h"] > ROUND_TRIP_PROXY(h))).astype(float)
        df[f"target_{h}h_breakout_long"] = ((df["setup_breakout_long"] > 0) & (df[f"fwd_ret_{h}h"] > ROUND_TRIP_PROXY(h))).astype(float)
        df[f"target_{h}h_rebound_long"] = ((df["setup_rebound_long"] > 0) & (df[f"fwd_ret_{h}h"] > ROUND_TRIP_PROXY(h))).astype(float)
        df[f"target_{h}h_trend_short"] = ((df["setup_trend_short"] > 0) & (df[f"fwd_ret_{h}h"] < -ROUND_TRIP_PROXY(h))).astype(float)
        df[f"target_{h}h_exhaustion_short"] = ((df["setup_exhaustion_short"] > 0) & (df[f"fwd_ret_{h}h"] < -ROUND_TRIP_PROXY(h))).astype(float)
    return df


# ---------------------------------------------------------------------------
# Multi-timeframe context
# ---------------------------------------------------------------------------

def build_4h_context(df_4h):
    """Compute 4h indicators and return resampled series."""
    df = df_4h.copy()
    df = add_rsi(df, periods=(14,))
    df = add_macd(df)
    df = add_emas(df, periods=(21, 55))
    context = df[["ts", "rsi_14", "macd_hist_norm", "ema_21_dist", "ema_21_55_above"]].copy()
    context.columns = ["ts", "ctx_4h_rsi", "ctx_4h_macd_hist", "ctx_4h_ema21_dist", "ctx_4h_trend"]
    return context.set_index("ts")


def build_1d_context(df_1d):
    """Compute daily indicators."""
    df = df_1d.copy()
    df = add_rsi(df, periods=(14,))
    df = add_emas(df, periods=(50, 200))
    df["ctx_1d_above_ema200"] = (df["close"] > df["close"].ewm(span=200, adjust=False).mean()).astype(int)
    context = df[["ts", "rsi_14", "ctx_1d_above_ema200"]].copy()
    context.columns = ["ts", "ctx_1d_rsi", "ctx_1d_above_ema200"]
    return context.set_index("ts")


def merge_context(df_1h, df_4h, df_1d):
    """Forward-fill higher-timeframe context onto 1h dataframe."""
    ctx_4h = build_4h_context(df_4h)
    ctx_1d = build_1d_context(df_1d)

    # Convert 1h timestamps to match 4h/1d bucket
    df_1h = df_1h.copy()

    # 4h context: assign each 1h candle to its containing 4h bucket
    ts_series = pd.to_datetime(df_1h["ts"], unit="s", utc=True)
    ts_4h_bucket = (df_1h["ts"] // 14400) * 14400  # floor to 4h
    ts_1d_bucket = (df_1h["ts"] // 86400) * 86400  # floor to day

    # Map context values
    for col in ctx_4h.columns:
        df_1h[col] = ts_4h_bucket.map(ctx_4h[col]).ffill()

    for col in ctx_1d.columns:
        df_1h[col] = ts_1d_bucket.map(ctx_1d[col]).ffill()

    return df_1h


def build_benchmark_context(symbols=BENCHMARK_SYMBOLS):
    """Build BTC/ETH context frames for relative-strength and market-state features."""
    ctx = None
    for sym in symbols:
        df = load_ohlcv(sym, "1h")
        if df is None or len(df) < MIN_1H_CANDLES:
            continue
        b = df[["ts", "close", "volume", "turnover"]].copy()
        prefix = sym.split("-")[0].lower()
        b[f"{prefix}_ret_1h"] = b["close"].pct_change(1)
        b[f"{prefix}_ret_4h"] = b["close"].pct_change(4)
        b[f"{prefix}_ret_24h"] = b["close"].pct_change(24)
        b[f"{prefix}_vol_ratio"] = b["volume"] / b["volume"].rolling(24).mean().replace(0, np.nan)
        b[f"{prefix}_turnover_ratio"] = b["turnover"] / b["turnover"].rolling(24).mean().replace(0, np.nan)
        keep = ["ts", f"{prefix}_ret_1h", f"{prefix}_ret_4h", f"{prefix}_ret_24h", f"{prefix}_vol_ratio", f"{prefix}_turnover_ratio"]
        b = b[keep]
        ctx = b if ctx is None else ctx.merge(b, on="ts", how="outer")
    return ctx.sort_values("ts").reset_index(drop=True) if ctx is not None else None


def merge_benchmark_context(df, benchmark_ctx):
    if benchmark_ctx is None or benchmark_ctx.empty:
        return df
    df = df.merge(benchmark_ctx, on="ts", how="left")
    for col in benchmark_ctx.columns:
        if col != "ts":
            df[col] = df[col].ffill()

    if "btc_ret_1h" in df.columns:
        df["rel_btc_ret_1h"] = df["ret_1h"] - df["btc_ret_1h"]
        df["rel_btc_ret_4h"] = df["ret_4h"] - df["btc_ret_4h"]
        df["btc_beta_proxy"] = df["ret_24h"].rolling(24).corr(df["btc_ret_24h"])
    if "eth_ret_1h" in df.columns:
        df["rel_eth_ret_1h"] = df["ret_1h"] - df["eth_ret_1h"]
        df["rel_eth_ret_4h"] = df["ret_4h"] - df["eth_ret_4h"]
        df["eth_beta_proxy"] = df["ret_24h"].rolling(24).corr(df["eth_ret_24h"])

    benchmark_cols = [c for c in ("btc_ret_24h", "eth_ret_24h") if c in df.columns]
    if benchmark_cols:
        df["market_breadth_proxy"] = df[benchmark_cols].mean(axis=1)
    return df


# ---------------------------------------------------------------------------
# Build feature set for one pair
# ---------------------------------------------------------------------------

def csv_path(symbol, tf):
    safe = symbol.replace("-", "_")
    return os.path.join(HISTORICAL_DIR, f"{safe}_{tf}.csv")


def load_ohlcv(symbol, tf):
    path = csv_path(symbol, tf)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["ts"] = df["ts"].astype(int)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def discover_symbols_from_history():
    """Infer the trainable universe from downloaded historical files."""
    if not os.path.exists(HISTORICAL_DIR):
        return []
    symbols = set()
    for fname in os.listdir(HISTORICAL_DIR):
        if not fname.endswith("_1h.csv"):
            continue
        symbol = fname[:-7].replace("_", "-")
        symbols.add(symbol)
    return sorted(symbols)


def build_features(symbol, benchmark_ctx=None, rebuild_context=None):
    """Build full feature matrix for one symbol. Returns DataFrame."""
    print(f"  Building features for {symbol}...")

    df_1h = load_ohlcv(symbol, "1h")
    df_4h = load_ohlcv(symbol, "4h")
    df_1d = load_ohlcv(symbol, "1d")

    if df_1h is None or len(df_1h) < MIN_1H_CANDLES:
        print(f"    Skipping: not enough 1h data ({len(df_1h) if df_1h is not None else 0} candles)")
        return None

    # Add all 1h features
    df = df_1h.copy()
    df = add_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_emas(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_stochastic(df)
    df = add_adx(df)
    df = add_candle_patterns(df)
    df = add_behavior_profile_features(df)
    df = add_flow_and_move_features(df)
    df = add_execution_quality_features(df)
    df = add_setup_context_features(df)
    df = apply_rebuild_context(df, symbol, rebuild_context)

    # Add multi-timeframe context
    if df_4h is not None and len(df_4h) > 100:
        df = merge_context(df, df_4h, df_1d if df_1d is not None else df_4h)
    df = merge_benchmark_context(df, benchmark_ctx)

    # Add LLM-suggested custom features + on-chain features (plugin)
    try:
        from quantforge_features_custom import apply_custom_features
        df = apply_custom_features(df, symbol=symbol)
    except ImportError:
        pass  # Plugin not yet created — normal on first run
    except Exception as _e:
        print(f"    [WARN] custom features failed: {_e}")

    # Snapshot news features to history JSONL (accumulates daily for future ML training)
    try:
        from quantforge_news import get_news_features
        get_news_features()  # refreshes cache + appends timestamped entry to news_history.jsonl
    except Exception:
        pass

    # Add targets
    df = add_targets(df)

    # Add symbol column
    df["symbol"] = symbol

    # Drop rows with NaN in critical columns (initial lookback period)
    feature_cols = [c for c in df.columns if c not in ["ts", "symbol"] and not c.startswith("target_") and not c.startswith("fwd_ret_")]
    df = df.dropna(subset=feature_cols[:10])  # Drop rows where first 10 features are NaN
    df = df.reset_index(drop=True)

    print(f"    {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_build(pairs=None):
    if pairs is None:
        pairs = discover_symbols_from_history()
    benchmark_ctx = build_benchmark_context()
    rebuild_context = load_rebuild_context()

    all_dfs = []
    for symbol in pairs:
        df = build_features(symbol, benchmark_ctx=benchmark_ctx, rebuild_context=rebuild_context)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No data available. Run quantforge_data.py fetch first.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(FEATURES_DIR, "features_all.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined):,} rows to {out_path}")

    # Also save per-pair
    for df in all_dfs:
        sym = df["symbol"].iloc[0]
        safe = sym.replace("-", "_")
        path = os.path.join(FEATURES_DIR, f"{safe}_features.parquet")
        df.to_parquet(path, index=False)

    print(f"Feature columns: {[c for c in combined.columns if c not in ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover']]}")


def cmd_status():
    files = sorted(os.listdir(FEATURES_DIR)) if os.path.exists(FEATURES_DIR) else []
    if not files:
        print("No feature files yet. Run: python3 quantforge_features.py build")
        return
    for fname in files:
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(FEATURES_DIR, fname)
        df = pd.read_parquet(path)
        print(f"  {fname:<40} {len(df):>8,} rows  {len(df.columns):>4} cols")


def cmd_research_tsfresh(pairs=None):
    """Build research-stage tsfresh features in a separate output directory."""
    try:
        from quantforge_tsfresh import cmd_build as build_tsfresh
    except Exception as e:
        print(f"Unable to load tsfresh research scaffold: {e}")
        return
    build_tsfresh(pairs)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "build":
        pairs = sys.argv[2:] if len(sys.argv) > 2 else None
        cmd_build(pairs)
    elif cmd == "research-tsfresh":
        pairs = sys.argv[2:] if len(sys.argv) > 2 else None
        cmd_research_tsfresh(pairs)
    elif cmd == "status":
        cmd_status()
    else:
        print(__doc__)
        sys.exit(1)
