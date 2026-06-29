#!/usr/bin/env python3
"""QuantForge Agent — regime-aware adaptive BTC allocator.

Replaces the static HODL bot with a smart system that dynamically adjusts BTC
exposure based on market regime:

  BULL regime    → target 70% BTC, 30% cash    (lean in, capture trend)
  NEUTRAL regime → target 45% BTC, 55% cash    (balanced)
  CHOP regime    → target 35% BTC, 65% cash    (defensive, dry powder)
  BEAR regime    → target 15% BTC, 85% cash    (preserve capital)

Each cycle (hourly):
  1. Pull latest BTC price + recent history
  2. Compute regime classification
  3. Compute current allocation
  4. If allocation drifts > 5% from target → rebalance (buy/sell to target)
  5. Apply drawdown circuit breakers (auto-trim if equity drops > 8%)
  6. Apply profit ladder (sell 5% of position at every +20% gain in equity)

Why this works (real research):
  - Position sizing matters more than entry timing for retail traders
  - Regime-aware allocation captures upside while limiting bear drawdown
  - Disciplined rebalancing forces "buy low / sell high" mechanically
  - No predictions, no signals — just match exposure to current market state

Strategy edge comes from:
  - DRAWDOWN AVOIDANCE during bear regimes (most retail blow up here)
  - DCA effect during chop regimes (forces buying dips)
  - TRIM-THE-WINNER during euphoria (prevents giving back gains)
"""
import json
import os
import sys
import urllib.request
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import math
from quantforge_equity import compute_true_equity

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "agent_portfolio.json")
TRADES_FILE = os.path.join(DATA_DIR, "agent_trades.jsonl")
LOG_FILE = os.path.join(DATA_DIR, "agent.log")
REGIME_FILE = os.path.join(DATA_DIR, "agent_regime.json")
HALT_FILE = os.path.join(DATA_DIR, "agent_halt.flag")   # if exists, all trading is frozen
QF_PARAMS_FILE = os.path.join(DATA_DIR, "qf_strategy_params.json")  # v6: runtime tunables written by reflect daemon

KLINE_URL = "https://api-futures.kucoin.com/api/v1/kline/query"
TICKER_URL = "https://api-futures.kucoin.com/api/v1/ticker"
SYMBOL = "XBTUSDTM"

# ---------------------------------------------------------------------------
# Tunables (all overridable via strategy-params.json if present)
# ---------------------------------------------------------------------------
STARTING_BALANCE = 5000.0
LEVERAGE = 1                       # No leverage — pure HODL with smart sizing

# Regime → target BTC allocation
# v5 (2026-05-15): HODL-WITH-SAFETY-NETS mode.
#
# After 5 days of live paper-trading we had clear per-regime alpha data and it was
# unambiguous: every regime visited produced negative alpha vs passive HODL.
# CHOP: -$45 alpha over 14 visits. BULL: -$11 alpha over 2 visits. Total
# -$57 vs HODL while paying $10 in fees.
#
# The lagging-indicator failure mode is structural — MA20>MA50 + RSI 70 means
# "the rally already happened," so we buy at local tops, then sell at local
# bottoms when regime mean-reverts. Tightening deltas would shrink the bleed
# but not stop it. So: flat 65% allocation across ALL regimes. The bot still
# detects regime (for logging + attribution learning) but never acts on it.
#
# Why 65%: max upside capture while keeping $1,750 dry powder for DD-trim
# re-entry, panic-halt recovery, and opportunistic buy-the-dip on big moves.
# Drawdown trim (-8%), panic halt (-15%), profit ladder (+20%) all still
# active — those react to equity, not regime.
#
# Reversion path: if pure HODL also underperforms or one regime later shows
# clearly positive alpha, restore per-regime values in this dict.
TARGET_ALLOC = {
    "STRONG_BULL": 0.65,
    "BULL":        0.65,
    "NEUTRAL":     0.65,
    "CHOP":        0.65,
    "BEAR":        0.65,
    "STRONG_BEAR": 0.65,
}
HODL_MODE = False                  # REGIME_ACTIVE — agent acts on regime signals (profit mandate)

REBALANCE_THRESHOLD = 0.08         # 5% → 8% drift before rebalance (less over-trading)
DRAWDOWN_TRIM_PCT = 0.08           # Auto-trim BTC if equity drops 8% from peak
DRAWDOWN_TRIM_FACTOR = 0.5         # Sell half of BTC position on drawdown trip
# After a drawdown trim, suppress rebalance BUYS for this many hours so the
# rebalancer doesn't immediately buy back the BTC we just defensively sold.
# Sells are NEVER suppressed — if the crash continues we still de-risk.
# 48h gives the daily reflection daemon time to set a proper lower target.
DRAWDOWN_TRIM_BUYBACK_SUPPRESS_HOURS = 6   # v12: 24h→6h — crypto moves too fast for 24h lockout
PANIC_HALT_PCT = 0.15              # Full halt + liquidation if equity drops 15% from peak
PANIC_HALT_ABS_PCT = 0.35  # v29: disabled — BTC moves 15% normally          # OR if total PnL drops below -12% of starting balance

# === Tail-risk caps (2026-06-22 — NON-auto-tunable hard floor) ===============
# Hard ceilings the agent cannot loosen (NOT in TUNABLE_KEYS, like PANIC_HALT_PCT).
# They harden the DISCIPLINED CORE futures lane after the v28 bleed era. The moonshot
# sleeve (quantforge_moonshot.py) is a separate, downside-budgeted barbell satellite
# and is intentionally EXEMPT — do NOT extend these caps to it.
MAX_EFFECTIVE_LEVERAGE = 2.0       # hard ceiling on core futures leverage AFTER all scaling
DD_VELOCITY_TRIP_PCT = 0.05        # single-cycle equity drop that trips the circuit breaker
LEVERAGE_COOLDOWN_HOURS = 6        # suppress NEW leveraged opens this long after a breaker trip
PROFIT_TAKE_PCT = 0.10             # At +10% from start, sell 5% of position (lowered from 20%)
PROFIT_TAKE_INCREMENT = 0.10       # Continue selling 5% every additional +10% gain
# v12: Trailing stop for spot BTC — locks in profits on pullbacks
TRAIL_STOP_PCT = 0.05              # Sell if price drops 5% below highest since entry
TRAIL_STOP_ACTIVATE_PCT = 0.03     # Only activate trail when up 3% from cost basis
TAKER_FEE = 0.0006                 # KuCoin futures taker (0.06%) — charged on notional
MAKER_FEE = 0.0002                 # KuCoin futures maker (0.02%)
SPOT_FEE  = 0.001                  # KuCoin SPOT taker (0.10%) — spot fills are NOT futures-priced;
                                   # cost-honest: no assumed KCS/VIP discount (worst-case base rate)

# Performance tracking cache for auto force-trigger (v22)
# Shared across cycles via module-level dict — survives between cron invocations
# because the Python process is long-lived (imported once per agent run cycle).
_PERF_HISTORY_CACHE = {}  # keys: '_qf_perf_history' → list of equity values

# OVER-TRADING FIX (2026-05-14):
# After 4 days of live paper cycles, the agent did 10 trades (mostly rebalancing on regime flips
# every few hours) and lost $30 of simulated equity to fee drag. Adding three safeguards:
#   1. Hysteresis: regime must persist for N consecutive cycles before acting
#   2. Cooldown: minimum hours between any two rebalances
#   3. Daily cap: max 2 rebalances per 24h window
# Together: trade frequency drops from ~2.5/day to <=1/day.
REGIME_HYSTERESIS_CYCLES = 3        # Regime must be same for 3 cycles before acting
REBALANCE_COOLDOWN_HOURS = 6        # Min 6h between rebalances
MAX_REBALANCES_PER_DAY = 2          # Hard cap on daily trade frequency

# Mean-reversion strategy runtime tunables (v6.1 — hands-free auto-tuning)
MR_OVERSOLD_Z = -1.0                # z-score below which MR goes max-long
MR_OVERBOUGHT_Z = 1.0               # z-score above which MR goes max-short
MR_WEIGHT = 0.30                    # MR strategy weight

# Futures lane tunables (v7 — leveraged directional trading)
FUTURES_WEIGHT = 0.05               # fraction of equity allocated to futures lane
FUTURES_LEVERAGE = 3                # leverage multiplier for futures positions (auto-tunable 1-5)
VOLATILITY_GATE_ATR = 0.035         # ATR threshold above which leverage is reduced by 1x

# Conviction-scaled leverage thresholds (auto-tunable)
CONSENSUS_IRONCLAD = 8             # strategies needed for max 5x leverage
CONSENSUS_STRONG = 5               # strategies needed for 3x leverage
CONSENSUS_MODERATE = 3             # strategies needed for 2x leverage

# ML Scanner lane tunables (v8 — multi-coin ML-driven selection)
ML_SCANNER_WEIGHT = 0.05            # fraction of equity allocated to ML picks (spot)
ML_SCANNER_TOP_N = 5                # max number of coins to hold
ML_SCANNER_MIN_CONFIDENCE = 0.54    # minimum ensemble confidence for BUY signal
ML_SCANNER_MAX_PER_COIN = 0.02      # max % of equity per individual coin
ML_SCANNER_VENV_PYTHON = os.path.expanduser(  # Python with xgboost/lightgbm
    "~/quantforge/.venvs/quant-ops/bin/python"
)

# ML BTC directional signal (v31) — thin-edge XGBoost predictor
# Only acts when confidence > 0.55; model CV win rate 52.1% (base 50.6%)
ML_BTC_WEIGHT = 0.05               # max ±5% of equity adjustment (0.0-0.15)
ML_BTC_VENV_PYTHON = os.path.expanduser(  # quant-ops venv for xgboost
    "~/.venvs/quant-ops/bin/python"
)
ML_BTC_PREDICTOR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "quantforge_btc_predictor.py")

# TimesFM directional signal lane (v32)
TIMESFM_SIGNAL_WEIGHT = 0.03        # 0 disables the lane entirely

# Funding rate arbitrage lane tunables (v9)
FUNDING_ARB_WEIGHT = 0.05           # fraction of equity allocated to funding arb
FUNDING_ARB_MAX_POSITIONS = 3       # max concurrent funding arb positions
FUNDING_ARB_POSITION_SIZE = 50.0    # USD per position (paper)

# Regime-adaptive weight table (v10 — 2026-06-01)
# When REGIME_ADAPTIVE=True, strategy weights auto-swap based on the active regime.
# This prevents the agent from fighting itself: in BEAR/STRONG_BEAR, futures short
# gets heavier and spot HODL lighter. In BULL/STRONG_BULL, spot HODL gets heavier
# and futures long gets heavier. MR dominates in CHOP. Funding arb ramps in BEAR.
# Values are applied BEFORE strategy evaluation each cycle and override the
# module-level defaults (or params-file values).
REGIME_ADAPTIVE = True
REGIME_WEIGHT_TABLE = {
    #            spot_alloc  futures  mr      ml      funding_arb
    "STRONG_BEAR":  {"spot_alloc_pct": 0.38, "futures_weight": 0.18, "mr_weight": 0.20, "ml_scanner_weight": 0.06, "funding_arb_weight": 0.05},
    "BEAR":         {"spot_alloc_pct": 0.45, "futures_weight": 0.07, "mr_weight": 0.25, "ml_scanner_weight": 0.08, "funding_arb_weight": 0.05},
    "CHOP":         {"spot_alloc_pct": 0.50, "futures_weight": 0.05, "mr_weight": 0.30, "ml_scanner_weight": 0.10, "funding_arb_weight": 0.05},
    "NEUTRAL":      {"spot_alloc_pct": 0.55, "futures_weight": 0.05, "mr_weight": 0.25, "ml_scanner_weight": 0.10, "funding_arb_weight": 0.02},
    "BULL":         {"spot_alloc_pct": 0.62, "futures_weight": 0.07, "mr_weight": 0.15, "ml_scanner_weight": 0.10, "funding_arb_weight": 0.00},
    "STRONG_BULL":  {"spot_alloc_pct": 0.65, "futures_weight": 0.15, "mr_weight": 0.05, "ml_scanner_weight": 0.10, "funding_arb_weight": 0.00},
}
# Remember last applied regime so we don't log "no change" noise every cycle
_LAST_APPLIED_REGIME = None

# Param memory: track last regime we checked for best-param recall (v22)
_LAST_PARAM_MEMORY_REGIME = None

# Swarm consensus regime detection (v11)
# When True, replaces single-model detect_regime() with 7-voter swarm consensus.
# More robust against regime misreads — requires 4+ voters to agree.
SWARM_REGIME = True
SWARM_MIN_AGREEMENT = 0.50  # v12: 0.35→0.50 — fewer flips, more conviction
SWARM_MIN_VOTERS = 3
SWARM_CONFIDENCE_THRESHOLD = 0.30  # v12: 0.20→0.30 — tighter consensus required

# Microstructure-first regime detection (v12)
# When True, replaces both single-model and swarm with forward-looking order flow
# analysis. Reads CVD, pressure imbalance, depth walls, and micro returns from
# existing collector data — no new infrastructure. This is the fix for the
# lagging-indicator problem documented in the v5 code comments.
MICRO_REGIME = True
MICRO_REGIME_MIN_CONFIDENCE = 0.20  # v12: lower threshold while data is sparse; tune up later

# History window for regime detection
REGIME_LOOKBACK_HOURS = 200        # need this many 1h candles for MA200 etc


# ---------------------------------------------------------------------------
# Runtime tunables (v6 Stage 1) — written by quantforge_reflect.py
# ---------------------------------------------------------------------------
# At the start of every cycle, the agent reads qf_strategy_params.json (if
# present) and overrides the module-level constants below. This is the
# control plane that lets the reflection daemon adjust the bot's behavior
# without redeploying code.
#
# IMPORTANT: This loader ONLY accepts keys in TUNABLE_KEYS. Anything outside
# this allowlist is ignored — this is the second line of defense against the
# daemon (or a malicious params file) altering safety-critical constants like
# PANIC_HALT_PCT or DRAWDOWN_TRIM_PCT.
TUNABLE_KEYS = {
    "hodl_mode",
    "fixed_alloc_pct",
    "rebalance_threshold",
    "rebalance_cooldown_hours",
    "max_rebalances_per_day",
    "regime_hysteresis_cycles",
    "profit_take_pct",
    "profit_take_increment",
    "mr_oversold_z",
    "mr_overbought_z",
    "mr_weight",
    "futures_weight",
    "futures_leverage",
    "volatility_gate_atr",
    "consensus_ironclad",
    "consensus_strong",
    "consensus_moderate",
    "ml_scanner_weight",
    "ml_scanner_top_n",
    "ml_scanner_min_confidence",
    "ml_scanner_max_per_coin",
    "funding_arb_weight",
    "funding_arb_max_positions",
    "funding_arb_position_size",
    "regime_adaptive",
    "regime_weight_table",
    "swarm_regime",
    "micro_regime",
    "ml_btc_weight",
    "timesfm_signal_weight",
}

PARAM_KEY_ALIASES = {
    "REGIME_ADAPTIVE": "regime_adaptive",
}


def load_runtime_params():
    """Read qf_strategy_params.json and override module-level constants.

    Returns dict of {applied_key: applied_value} for logging. Silently
    ignores any key not in TUNABLE_KEYS — including PANIC_HALT_PCT and
    other safety constants which CANNOT be overridden via this path.
    """
    global HODL_MODE, TARGET_ALLOC, REBALANCE_THRESHOLD
    global REBALANCE_COOLDOWN_HOURS, MAX_REBALANCES_PER_DAY
    global REGIME_HYSTERESIS_CYCLES, PROFIT_TAKE_PCT, PROFIT_TAKE_INCREMENT
    global MR_OVERSOLD_Z, MR_OVERBOUGHT_Z, MR_WEIGHT
    global FUTURES_WEIGHT, FUTURES_LEVERAGE, CONSENSUS_IRONCLAD, CONSENSUS_STRONG, CONSENSUS_MODERATE, CONSENSUS_IRONCLAD, CONSENSUS_STRONG, CONSENSUS_MODERATE
    global ML_SCANNER_WEIGHT, ML_SCANNER_TOP_N, ML_SCANNER_MIN_CONFIDENCE
    global ML_SCANNER_MAX_PER_COIN
    global FUNDING_ARB_WEIGHT, FUNDING_ARB_MAX_POSITIONS, FUNDING_ARB_POSITION_SIZE
    global REGIME_ADAPTIVE, REGIME_WEIGHT_TABLE
    global SWARM_REGIME
    global MICRO_REGIME
    global ML_BTC_WEIGHT, TIMESFM_SIGNAL_WEIGHT
    if not os.path.exists(QF_PARAMS_FILE):
        return {}
    try:
        with open(QF_PARAMS_FILE) as f:
            params = json.load(f)
    except Exception as e:
        log(f"  ⚠️ Could not load {QF_PARAMS_FILE}: {e} — using defaults")
        return {}
    applied = {}
    for raw_key, value in params.items():
        if raw_key.startswith("_"):  # metadata fields like _last_modified_by
            continue
        key = PARAM_KEY_ALIASES.get(raw_key, raw_key)
        if key not in TUNABLE_KEYS:
            log(f"  ⚠️ Ignoring non-tunable param '{raw_key}' in params file")
            continue
        if key == "hodl_mode":
            HODL_MODE = bool(value)
            applied[key] = HODL_MODE
        elif key == "fixed_alloc_pct":
            v = float(value)
            if 0.40 <= v <= 0.85:
                TARGET_ALLOC = {r: v for r in TARGET_ALLOC}
                applied[key] = v
            else:
                log(f"  ⚠️ fixed_alloc_pct {v} out of bounds [0.40, 0.85] — ignored")
        elif key == "rebalance_threshold":
            v = float(value)
            if 0.02 <= v <= 0.20:
                REBALANCE_THRESHOLD = v
                applied[key] = v
        elif key == "rebalance_cooldown_hours":
            v = int(value)
            if 0 <= v <= 48:  # allow 0 for emergency bypass
                REBALANCE_COOLDOWN_HOURS = v
                applied[key] = v
        elif key == "max_rebalances_per_day":
            v = int(value)
            if 1 <= v <= 5:
                MAX_REBALANCES_PER_DAY = v
                applied[key] = v
        elif key == "regime_hysteresis_cycles":
            v = int(value)
            if 2 <= v <= 6:
                REGIME_HYSTERESIS_CYCLES = v
                applied[key] = v
        elif key == "profit_take_pct":
            v = float(value)
            if 0.10 <= v <= 0.30:
                PROFIT_TAKE_PCT = v
                applied[key] = v
        elif key == "profit_take_increment":
            v = float(value)
            if 0.05 <= v <= 0.15:
                PROFIT_TAKE_INCREMENT = v
                applied[key] = v
        elif key == "mr_oversold_z":
            v = float(value)
            if -2.5 <= v <= -0.3:
                MR_OVERSOLD_Z = v
                applied[key] = v
        elif key == "mr_overbought_z":
            v = float(value)
            if 0.3 <= v <= 2.5:
                MR_OVERBOUGHT_Z = v
                applied[key] = v
        elif key == "mr_weight":
            v = float(value)
            if 0.0 <= v <= 0.50:
                MR_WEIGHT = v
                applied[key] = v
        elif key == "futures_weight":
            v = float(value)
            if 0.0 <= v <= 0.30:
                FUTURES_WEIGHT = v
                applied[key] = v
        elif key == "futures_leverage":
            v = int(value)
            if 1 <= v <= 5:
                FUTURES_LEVERAGE = v
                applied[key] = v
        elif key == "volatility_gate_atr":
            v = float(value)
            if 0.02 <= v <= 0.08:
                VOLATILITY_GATE_ATR = v
                applied[key] = v
        elif key == "consensus_ironclad":
            v = int(value)
            if 5 <= v <= 10:
                CONSENSUS_IRONCLAD = v
                applied[key] = v
        elif key == "consensus_strong":
            v = int(value)
            if 3 <= v <= params.get("consensus_ironclad", globals()['CONSENSUS_IRONCLAD']):
                CONSENSUS_STRONG = v
                applied[key] = v
        elif key == "consensus_moderate":
            v = int(value)
            if 2 <= v <= params.get("consensus_strong", globals()['CONSENSUS_STRONG']):
                CONSENSUS_MODERATE = v
                applied[key] = v
        elif key == "ml_scanner_weight":
            v = float(value)
            if 0.0 <= v <= 0.15:
                ML_SCANNER_WEIGHT = v
                applied[key] = v
        elif key == "ml_scanner_top_n":
            v = int(value)
            if 1 <= v <= 10:
                ML_SCANNER_TOP_N = v
                applied[key] = v
        elif key == "ml_scanner_min_confidence":
            v = float(value)
            if 0.50 <= v <= 0.80:
                ML_SCANNER_MIN_CONFIDENCE = v
                applied[key] = v
        elif key == "ml_scanner_max_per_coin":
            v = float(value)
            if 0.01 <= v <= 0.05:
                ML_SCANNER_MAX_PER_COIN = v
                applied[key] = v
        elif key == "funding_arb_weight":
            v = float(value)
            if 0.0 <= v <= 0.30:
                FUNDING_ARB_WEIGHT = v
                applied[key] = v
        elif key == "funding_arb_max_positions":
            v = int(value)
            if 1 <= v <= 10:
                FUNDING_ARB_MAX_POSITIONS = v
                applied[key] = v
        elif key == "funding_arb_position_size":
            v = float(value)
            if 10.0 <= v <= 500.0:
                FUNDING_ARB_POSITION_SIZE = v
                applied[key] = v
        elif key == "regime_adaptive":
            REGIME_ADAPTIVE = bool(value)
            applied[key] = REGIME_ADAPTIVE
        elif key == "regime_weight_table":
            v = value
            if isinstance(v, dict) and all(r in v for r in ("STRONG_BEAR", "BEAR", "CHOP", "NEUTRAL", "BULL", "STRONG_BULL")):
                REGIME_WEIGHT_TABLE = v
                applied[key] = f"<{len(v)} regimes>"
            else:
                log(f"  ⚠️ regime_weight_table missing required regimes — ignored")
        elif key == "swarm_regime":
            SWARM_REGIME = bool(value)
            applied[key] = SWARM_REGIME
        elif key == "micro_regime":
            MICRO_REGIME = bool(value)
            applied[key] = MICRO_REGIME
        elif key == "ml_btc_weight":
            v = float(value)
            if 0.0 <= v <= 0.15:
                ML_BTC_WEIGHT = v
                applied[key] = v
            else:
                log(f"  ⚠️ ml_btc_weight {v} out of bounds [0.0, 0.15] — ignored")
        elif key == "timesfm_signal_weight":
            v = float(value)
            if 0.0 <= v <= 0.10:
                TIMESFM_SIGNAL_WEIGHT = v
                applied[key] = v
            else:
                log(f"  ⚠️ timesfm_signal_weight {v} out of bounds [0.0, 0.10] — ignored")
    # Rebuild strategy registry with current MR_WEIGHT
    _rebuild_strategy_registry()
    return applied


def _apply_regime_weights(active_regime):
    """Override strategy weights based on active regime (v10).

    When REGIME_ADAPTIVE=True, reads the REGIME_WEIGHT_TABLE for the active
    regime and overrides module-level globals: TARGET_ALLOC, FUTURES_WEIGHT,
    MR_WEIGHT, ML_SCANNER_WEIGHT, FUNDING_ARB_WEIGHT.

    Returns a dict of {key: "old→new"} for logging. Returns {} if no change.
    Silently no-ops if REGIME_ADAPTIVE=False or regime not in table.
    """
    global TARGET_ALLOC, FUTURES_WEIGHT, MR_WEIGHT
    global ML_SCANNER_WEIGHT, FUNDING_ARB_WEIGHT, _LAST_APPLIED_REGIME

    if not REGIME_ADAPTIVE:
        return {}

    overrides = REGIME_WEIGHT_TABLE.get(active_regime)
    if not overrides:
        return {}

    # Skip logging if same regime as last cycle (noise reduction)
    if _LAST_APPLIED_REGIME == active_regime:
        return {}

    # Per-key hard bounds — the table path must respect the SAME bounds as the
    # flat tunable keys, otherwise a bad table write bypasses every limit.
    _TABLE_BOUNDS = {
        "spot_alloc_pct": (0.40, 0.85),
        "futures_weight": (0.0, 0.30),
        "mr_weight": (0.0, 0.50),
        "ml_scanner_weight": (0.0, 0.15),
        "funding_arb_weight": (0.0, 0.30),
    }

    changes = {}
    spot_pct = overrides.get("spot_alloc_pct")
    if spot_pct is not None and isinstance(spot_pct, (int, float)):
        lo, hi = _TABLE_BOUNDS["spot_alloc_pct"]
        clamped = min(max(float(spot_pct), lo), hi)
        if clamped != float(spot_pct):
            log(f"  ⚠️ regime table spot_alloc_pct {spot_pct} out of bounds [{lo}, {hi}] — clamped to {clamped}")
        old = TARGET_ALLOC.get(active_regime, TARGET_ALLOC.get("NEUTRAL", 0.55))
        TARGET_ALLOC[active_regime] = clamped
        changes["spot_alloc"] = f"{old:.0%}→{clamped:.0%}"

    for key, var_name in [
        ("futures_weight", "FUTURES_WEIGHT"),
        ("mr_weight", "MR_WEIGHT"),
        ("ml_scanner_weight", "ML_SCANNER_WEIGHT"),
        ("funding_arb_weight", "FUNDING_ARB_WEIGHT"),
    ]:
        val = overrides.get(key)
        if val is not None and isinstance(val, (int, float)):
            old_val = globals().get(var_name)
            if old_val is not None:
                lo, hi = _TABLE_BOUNDS[key]
                clamped = min(max(float(val), lo), hi)
                if clamped != float(val):
                    log(f"  ⚠️ regime table {key} {val} out of bounds [{lo}, {hi}] — clamped to {clamped}")
                globals()[var_name] = clamped
                changes[key] = f"{old_val:.0%}→{clamped:.0%}"

    _LAST_APPLIED_REGIME = active_regime
    
    # v28 SAFETY CLAMP: prevent strategy factory from going nuclear
    # Net exposure = spot - futures_weight * futures_leverage
    # Must never go below -10% (max 10% net short)
    spot = TARGET_ALLOC.get(active_regime, 0.45)
    fw = globals().get("FUTURES_WEIGHT", 0.05)
    fle = float(overrides.get("futures_leverage", FUTURES_LEVERAGE))
    net_exposure = spot - fw * fle
    
    if net_exposure < -0.10:
        # Too aggressive — clamp futures weight
        max_fw = (spot + 0.10) / max(fle, 1)
        clamped_fw = min(max_fw, 0.15)
        globals()["FUTURES_WEIGHT"] = clamped_fw
        log(f"  🛡️ SAFETY CLAMP: net exposure {net_exposure:.0%} below -10% → futures {fw:.0%}→{clamped_fw:.0%}")
    if active_regime in ("BEAR", "STRONG_BEAR") and spot < 0.35:
        TARGET_ALLOC[active_regime] = 0.35
        log(f"  🛡️ SAFETY CLAMP: STRONG_BEAR/BEAR spot floor 35% → {spot:.0%}→35%")
    
    return changes


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def get_btc_price():
    url = f"{TICKER_URL}?symbol={SYMBOL}"
    req = urllib.request.Request(url, headers={"User-Agent": "qf-agent"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != "200000":
        raise RuntimeError(f"ticker error: {data}")
    return float(data["data"]["price"])


def get_btc_klines_1h(hours=REGIME_LOOKBACK_HOURS):
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - hours * 3600 * 1000
    params = {
        "symbol": SYMBOL,
        "granularity": 60,
        "from": start_ms,
        "to": end_ms,
    }
    url = KLINE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "qf-agent"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != "200000":
        raise RuntimeError(f"klines error: {data}")
    rows = data.get("data", [])
    # Format: [time(ms), open, high, low, close, volume, turnover]
    out = []
    for r in rows:
        out.append({
            "ts": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        })
    out.sort(key=lambda c: c["ts"])
    return out


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------
def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def atr_pct(candles, period=24):
    """Average True Range as % of close — measures volatility."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, period + 1):
        c = candles[-i]
        prev = candles[-i - 1]
        tr = max(
            c["high"] - c["low"],
            abs(c["high"] - prev["close"]),
            abs(c["low"] - prev["close"]),
        )
        trs.append(tr / c["close"])
    return sum(trs) / period


# Cache for external API calls (avoid rate limits)
_SENTIMENT_CACHE = {"ts": datetime.min.replace(tzinfo=timezone.utc), "data": {}}
_SENTIMENT_CACHE_TTL = timedelta(hours=1)
_NEWS_CACHE = {"ts": datetime.min.replace(tzinfo=timezone.utc), "data": {}}
_NEWS_CACHE_TTL = timedelta(minutes=30)


def _fetch_micro_signals(closes):
    """Compute microstructure signals from candles for auto strategies (v19).
    Returns: CVD proxy, ETH/SOL proxy, liquidation proxy, OI proxy, ATR."""
    sig = {}
    n = len(closes)
    if n < 24:
        return sig

    cur = closes[-1]
    # CVD proxy: net volume * price change over 4h and 8h
    # Approximate from candle closes — real CVD needs trade tape
    cvd_4h = 0.0; cvd_8h = 0.0
    for i in range(max(0, n-4), n):
        cvd_4h += 0  # placeholder — no volume data in closes list
    sig["cvd_4h"] = cvd_4h
    sig["cvd_8h"] = cvd_8h

    # ATR-14
    if n >= 15:
        trs = []
        for i in range(max(0, n-14), n):
            trs.append(abs(closes[i] - closes[i-1]) if i > 0 else 0)
        sig["atr_14"] = sum(trs) / len(trs) if trs else 0
        sig["atr_mean_24h"] = sig["atr_14"]  # approximate

    # MA20
    sig["MA20"] = sum(closes[-20:]) / min(n, 20) if n >= 20 else cur

    # Price change 4h
    if n >= 4:
        sig["price_change_4h"] = (cur - closes[-4]) / closes[-4] if closes[-4] > 0 else 0

    # ETH/SOL proxy: use BTC itself as lead (imperfect but functional)
    sig["eth_return_4h"] = sig.get("price_change_4h", 0) * 1.1  # ETH typically 1.1x beta
    sig["sol_return_4h"] = sig.get("price_change_4h", 0) * 1.3   # SOL typically 1.3x beta

    # Liquidation proxy: compute from volatility spike
    if n >= 24:
        recent_vol = 0.0
        for i in range(max(0, n-24), n-1):
            recent_vol += abs(closes[i+1] - closes[i]) / closes[i] if closes[i] > 0 else 0
        recent_vol /= min(n, 24)
        sig["liq_long_1h"] = recent_vol * 100000  # scale to liquidation magnitude
        sig["liq_short_1h"] = recent_vol * 50000

    # OI change proxy: volume * price change direction
    sig["oi_change_4h"] = sig.get("price_change_4h", 0) * 0.5

    # Funding rate proxy: from sentiment cache
    sentiment = _fetch_sentiment_signals()
    sig["funding_rate"] = sentiment.get("funding_rate", 0.0001)

    return sig


def _fetch_sentiment_signals():
    """Fetch Fear & Greed Index + KuCoin funding rate. Cached for 1 hour."""
    global _SENTIMENT_CACHE
    now = datetime.now(timezone.utc)
    if now - _SENTIMENT_CACHE["ts"] < _SENTIMENT_CACHE_TTL and _SENTIMENT_CACHE["data"]:
        return _SENTIMENT_CACHE["data"]

    data = {"fear_greed": 50, "funding_rate": 0.0}  # defaults on failure
    try:
        # Fear & Greed Index (alternative.me — free, no key needed)
        req = urllib.request.Request("https://api.alternative.me/fng/?limit=1", headers={"User-Agent": "qf-agent"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            fng = json.loads(resp.read().decode())
        data["fear_greed"] = int(fng["data"][0]["value"])
    except Exception as e:
        log(f"  ⚠️ Fear & Greed fetch failed: {e}")

    try:
        # KuCoin funding rate (already using this API for price)
        req = urllib.request.Request("https://api-futures.kucoin.com/api/v1/contracts/XBTUSDTM", headers={"User-Agent": "qf-agent"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            contract = json.loads(resp.read().decode())
        data["funding_rate"] = float(contract["data"]["fundingFeeRate"])
    except Exception as e:
        log(f"  ⚠️ Funding rate fetch failed: {e}")

    _SENTIMENT_CACHE = {"ts": now, "data": data}
    return data


def _fetch_news_signal():
    """Fetch crypto news sentiment signal from RSS feeds. Cached for 30 min.
    Uses quantforge_news.py module — zero API keys, pure RSS.
    Returns dict: {signal, confidence, score, highlights}
    """
    global _NEWS_CACHE
    now = datetime.now(timezone.utc)
    if now - _NEWS_CACHE["ts"] < _NEWS_CACHE_TTL and _NEWS_CACHE["data"]:
        return _NEWS_CACHE["data"]

    data = {"signal": "NEUTRAL", "confidence": 0.0, "score": 0.0, "highlights": []}
    try:
        from quantforge_news import get_news_signal
        data = get_news_signal()
    except Exception as e:
        log(f"  ⚠️ News signal fetch failed: {e}")

    _NEWS_CACHE = {"ts": now, "data": data}
    return data


def _apply_sentiment_modifier(target_alloc_pct, signals):
    """Adjust target allocation based on Fear & Greed contrarian signal.

    Proven crypto alpha: buy when crowd is fearful, sell when greedy.
    Returns adjusted target_alloc_pct.
    """
    fg = signals.get("fear_greed", 50)
    if fg == 50:
        return target_alloc_pct  # no data, no modifier

    # Contrarian adjustment: fear → increase, greed → decrease
    if fg <= 25:           # Extreme Fear: +10% allocation
        adj = +0.10
        tag = "EXTREME_FEAR"
    elif fg <= 40:         # Fear: +5%
        adj = +0.05
        tag = "FEAR"
    elif fg <= 60:         # Neutral: no change
        adj = 0.0
        tag = "NEUTRAL"
    elif fg <= 75:         # Greed: -5%
        adj = -0.05
        tag = "GREED"
    else:                  # Extreme Greed: -10%
        adj = -0.10
        tag = "EXTREME_GREED"

    adjusted = target_alloc_pct + adj
    # Clamp to safe range
    adjusted = max(0.25, min(0.80, adjusted))
    if adj != 0:
        log(f"  📊 Sentiment: {tag} (F&G={fg}) → alloc {target_alloc_pct*100:.0f}% → {adjusted*100:.0f}%")
    return adjusted


def detect_regime(candles):
    """Classify market into one of 6 regimes based on multi-timeframe signals.
    Also fetches external sentiment signals (Fear & Greed, funding rate)."""
    closes = [c["close"] for c in candles]
    n = len(closes)
    if n < 50:
        return "NEUTRAL", {"reason": "insufficient_history", "n_candles": n}

    cur = closes[-1]
    ma20 = sma(closes, 20) or cur
    ma50 = sma(closes, 50) or cur
    ma200 = sma(closes, 200) if n >= 200 else (ma50 or cur)
    rsi14 = rsi(closes, 14) or 50
    vol = atr_pct(candles, 24) or 0.01

    # 7-day price change
    if n >= 168:
        change_7d = (cur / closes[-168] - 1)
    else:
        change_7d = (cur / closes[0] - 1) if closes[0] > 0 else 0

    # 30-day slope (cur vs 30d ago)
    if n >= 30 * 24:
        change_30d = (cur / closes[-30 * 24] - 1)
    else:
        change_30d = change_7d

    # 24h price statistics — used by mean-reversion strategy (Stage 3).
    # We compute mean & std over the last 24 closes (24h on 1h candles).
    # When n < 24, fall back to whatever we have (n >= 50 by guard above).
    window = closes[-24:] if n >= 24 else closes
    price_mean_24h = sum(window) / len(window) if window else cur
    if len(window) > 1:
        variance = sum((p - price_mean_24h) ** 2 for p in window) / len(window)
        price_std_24h = variance ** 0.5
    else:
        price_std_24h = 0.0
    price_z_24h = (cur - price_mean_24h) / price_std_24h if price_std_24h > 0 else 0.0

    signals = {
        "price": cur,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi14": rsi14,
        "atr_pct": vol,
        "change_7d": change_7d,
        "change_30d": change_30d,
        "price_mean_24h": price_mean_24h,
        "price_std_24h": price_std_24h,
        "price_z_24h": price_z_24h,
    }
    # Inject external sentiment signals (free APIs, cached for 1h)
    sentiment = _fetch_sentiment_signals()
    signals.update(sentiment)

    # Inject microstructure signals for auto-generated strategies (v19)
    micro_signals = _fetch_micro_signals(closes)
    signals.update(micro_signals)

    # Inject news sentiment signal (v23 — zero-cost RSS, no API keys)
    news_signal = _fetch_news_signal()
    signals["news_signal"] = news_signal

    # STRONG BEAR: MA20 < MA50 < MA200 AND price falling AND RSI 30-50
    if ma20 < ma50 < ma200 and change_7d < -0.05 and rsi14 < 50:
        return "STRONG_BEAR", signals
    # BEAR: MA20 < MA50 AND 7d change negative
    if ma20 < ma50 and change_7d < -0.02:
        return "BEAR", signals
    # STRONG BULL: MA20 > MA50 > MA200 AND price rising 5%+ AND RSI 50-70
    if ma20 > ma50 > ma200 and change_7d > 0.05 and 50 <= rsi14 <= 75:
        return "STRONG_BULL", signals
    # BULL: MA20 > MA50 AND positive 7d change
    if ma20 > ma50 and change_7d > 0.02:
        return "BULL", signals
    # CHOP: low ATR, MAs tangled
    if vol < 0.008 and abs(change_7d) < 0.03:
        return "CHOP", signals
    return "NEUTRAL", signals


# ---------------------------------------------------------------------------
# Portfolio I/O
# ---------------------------------------------------------------------------
PORTFOLIO_START_STALE_GAP_DAYS = 7
PORTFOLIO_START_THIN_SAMPLE_MAX_TRADES = 20


def _parse_portfolio_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _recent_spot_trade_times(path, max_count=None):
    times = deque(maxlen=max_count if max_count and max_count > 0 else None)
    try:
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("type") not in {"BUY", "SELL"}:
                    continue
                dt = _parse_portfolio_ts(row.get("ts"))
                if dt is not None:
                    times.append(dt)
    except Exception:
        return []
    return list(times)


def infer_portfolio_start_anchor(port, trades_path=TRADES_FILE):
    """Pick the fairest portfolio-start timestamp for user-facing reporting."""
    n_trades = int(port.get("n_trades", 0) or port.get("total_trades", 0) or 0)
    max_count = n_trades if n_trades > 0 else None

    metadata_candidates = [
        ("_reset_at", _parse_portfolio_ts(port.get("_reset_at"))),
        ("created_at", _parse_portfolio_ts(port.get("created_at"))),
        ("last_panic_reset_at", _parse_portfolio_ts(port.get("last_panic_reset_at"))),
    ]
    metadata_source, metadata_dt = next(
        ((source, dt) for source, dt in metadata_candidates if dt is not None),
        (None, None),
    )

    activity_candidates = []
    trade_times = _recent_spot_trade_times(trades_path, max_count=max_count)
    if trade_times:
        activity_candidates.append(("recent_spot_trades", trade_times[0]))

    rebalance_times = [_parse_portfolio_ts(ts) for ts in (port.get("rebalance_log") or [])]
    rebalance_times = [dt for dt in rebalance_times if dt is not None]
    if rebalance_times:
        if max_count:
            rebalance_times = rebalance_times[-max_count:]
        activity_candidates.append(("rebalance_log", min(rebalance_times)))

    activity_source, activity_dt = min(activity_candidates, key=lambda item: item[1]) if activity_candidates else (None, None)

    if metadata_dt and activity_dt and 0 < n_trades < PORTFOLIO_START_THIN_SAMPLE_MAX_TRADES:
        if metadata_dt < activity_dt - timedelta(days=PORTFOLIO_START_STALE_GAP_DAYS):
            return activity_dt, f"{activity_source}_fallback"

    if metadata_dt is not None:
        return metadata_dt, metadata_source
    if activity_dt is not None:
        return activity_dt, f"{activity_source}_fallback"
    return None, None


def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return None
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


def save_portfolio(port):
    port["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(port, f, indent=2)


def append_trade(trade):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(trade) + "\n")


# ---------------------------------------------------------------------------
# Panic halt — hard circuit breaker. When tripped, the agent fully liquidates
# BTC to cash and writes a halt marker file. Subsequent cycles see the marker
# and refuse to trade until a human runs `quantforge_agent.py panic-reset`.
# This is the last line of defense — protects against catastrophic regime
# misreads, flash crashes, or strategy logic bugs spiraling into large losses.
# ---------------------------------------------------------------------------
def is_halted():
    return os.path.exists(HALT_FILE)


def _benchmark_hold_active():
    """Phase E benchmark gate: True only when quantforge_benchmark_gate.py has
    PROVEN (over >=20 live trades) that active trading underperforms simply
    holding BTC. Absent / insufficient evidence -> False, so the agent keeps
    trading to earn its track record. Fails open to normal trading on any read
    error — an advisory capital-protection flag must never silently freeze the
    agent on a malformed or missing file."""
    try:
        with open(os.path.join(DATA_DIR, "benchmark_gate_state.json")) as f:
            return bool(json.load(f).get("enforce_hold", False))
    except Exception:
        return False


def write_halt_marker(reason, equity, peak_equity, drawdown_pct):
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "halted_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "equity": round(equity, 2),
        "peak_equity": round(peak_equity, 2),
        "drawdown_pct": round(drawdown_pct * 100, 2),
        "starting_balance": STARTING_BALANCE,
    }
    with open(HALT_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def clear_halt_marker():
    if os.path.exists(HALT_FILE):
        os.remove(HALT_FILE)
        return True
    return False


def _true_equity(port, price):
    """v29: Net liquidation value of the agent ledger — single source of truth."""
    return compute_true_equity(port, price)


def _true_drawdown(port, price):
    """v28: Single source of truth for drawdown from peak."""
    eq = _true_equity(port, price)
    peak = port.get("peak_equity", eq)
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - eq) / peak)


def _auto_fix_stale_peak(port, price):
    """v29: No-op, retained for call-site compatibility.

    The v28 version reset peak_equity DOWNWARD whenever drawdown exceeded 10%
    while the account was profitable overall — which made the 8% drawdown trim
    and 15% panic halt unreachable from any profitable peak (a real 11% crash
    from highs would silently re-baseline instead of halting). The phantom
    drawdowns it papered over were caused by _true_equity excluding futures
    margin; v29 includes margin, so position opens no longer dent equity and
    the safety breakers must measure from the genuine peak.
    """
    return False


def trip_panic_halt(port, price, reason, drawdown_pct):
    """Liquidate all BTC, close every leveraged lane, and freeze trading."""
    # v29: single source of truth for equity (was a third inline formula that
    # double-counted leverage and ignored SHORT direction)
    true_equity = _true_equity(port, price)
    log(f"🚨🚨🚨 PANIC HALT TRIPPED: {reason}")
    log(f"   True equity: ${true_equity:,.2f}  Peak: ${port.get('peak_equity', STARTING_BALANCE):,.2f}  DD: {drawdown_pct*100:.2f}%")
    if port["btc_qty"] > 0:
        log(f"   Liquidating {port['btc_qty']:.6f} BTC to cash")
        sell_btc(port, price, port["btc_qty"], reason=f"panic_halt:{reason}")
    # v29: close leveraged lanes too. A halted cycle returns before the
    # futures stop-losses run, so anything left open here is UNMANAGED
    # leveraged exposure for the entire halt duration.
    fp = port.get("futures_position") or {}
    if fp.get("direction") and fp.get("notional", 0) > 0:
        entry = fp.get("entry_price", price)
        if fp["direction"] == "LONG":
            pnl = fp["notional"] * (price / entry - 1.0)
        else:
            pnl = fp["notional"] * (1.0 - price / entry)
        port["cash"] += fp["margin"] + pnl
        port["futures_pnl"] = port.get("futures_pnl", 0.0) + pnl
        log(f"   Closing futures {fp['direction']} | PnL ${pnl:+.2f}")
        append_trade({
            "ts": datetime.now(timezone.utc).isoformat(),
            "side": "close_" + fp["direction"].lower(),
            "type": "FUTURES_CLOSE", "strategy_id": "futures_lane", "entry_ts": fp.get("opened_at"), "_entry_price": fp.get("entry_price"),
            "reason": f"panic_halt:{reason}",
            "price": price,
            "qty": fp["notional"],
            "usd": fp["margin"] + pnl,
            "fee": 0.0,
            "pnl_usd": round(pnl, 2),
            "direction": fp["direction"],
        })
        port["futures_position"] = {"direction": None, "margin": 0, "notional": 0, "entry_price": 0, "opened_at": None}
    if (port.get("prehedge") or {}).get("open"):
        try:
            from quantforge_prehedge import close_position as _ph_close
            res = _ph_close(port, price)
            log(f"   Closing prehedge | PnL ${res.get('pnl', 0):+.2f}")
        except Exception as e:
            log(f"   ⚠️ Prehedge close failed during halt: {e}")
    lp = port.get("liq_dip_position") or {}
    if lp.get("direction") and lp.get("notional", 0) > 0:
        entry = lp.get("entry_price", price)
        if lp["direction"] == "LONG":
            pnl = lp["notional"] * (price / entry - 1.0)
        else:
            pnl = lp["notional"] * (1.0 - price / entry)
        port["cash"] += lp["margin"] + pnl
        log(f"   Closing liq-dip {lp['direction']} | PnL ${pnl:+.2f}")
        port["liq_dip_position"] = {}
    # Recompute after liquidation (all proceeds now sit in cash)
    final_equity = _true_equity(port, price)
    port["panic_halted"] = True
    port["panic_halted_at"] = datetime.now(timezone.utc).isoformat()
    port["panic_halt_reason"] = reason
    write_halt_marker(reason, final_equity, port.get("peak_equity", STARTING_BALANCE), drawdown_pct)
    save_portfolio(port)
    log(f"   Final equity all-cash: ${final_equity:,.2f}")
    log(f"   Halt marker written: {HALT_FILE}")
    log(f"   Run 'quantforge_agent.py panic-reset' to resume trading after review.")


def _in_leverage_cooldown(port):
    """True while a dd-velocity breaker cooldown is active. Suppresses NEW leveraged
    opens (never closes). Fail-safe: missing/bad timestamp -> not in cooldown, so a
    malformed field can never wedge the lane permanently shut."""
    until = port.get("leverage_cooldown_until")
    if not until:
        return False
    try:
        return datetime.now(timezone.utc) < datetime.fromisoformat(until)
    except Exception:
        return False


def check_dd_velocity_breaker(port, price):
    """Fast first-responder BELOW the panic halt: if true equity dropped >=
    DD_VELOCITY_TRIP_PCT in a SINGLE cycle, flatten the leveraged lanes
    (futures / prehedge / liq-dip), KEEP spot BTC (don't dump the core holding at the
    bottom), and start a LEVERAGE_COOLDOWN_HOURS cooldown that suppresses new leveraged
    opens. Catches fast intra-trend bleeds the once-per-cycle, 15%-from-peak panic halt
    is too high/slow to stop. Returns True if it tripped.

    The moonshot sleeve is a separate, downside-budgeted module and is untouched here.
    Fail-safe: any error -> no trip (the breaker must never break the cycle)."""
    try:
        prev = port.get("prev_cycle_equity")
        if not prev or prev <= 0:
            return False
        eq = _true_equity(port, price)
        drop = (prev - eq) / prev
        if drop < DD_VELOCITY_TRIP_PCT:
            return False
        log(f"🧯 DD-VELOCITY BREAKER: equity ${prev:,.2f} → ${eq:,.2f} "
            f"({drop*100:.1f}% in one cycle ≥ {DD_VELOCITY_TRIP_PCT*100:.0f}%) — "
            f"flattening leverage, keeping spot BTC.")
        # Futures lane — tested helper: no-op when flat, fail-safe on bad price, ledgers the close.
        _close_futures_position(port, price, "dd_velocity_breaker")
        # Prehedge lane
        if (port.get("prehedge") or {}).get("open"):
            try:
                from quantforge_prehedge import close_position as _ph_close
                res = _ph_close(port, price)
                log(f"   Closing prehedge | PnL ${res.get('pnl', 0):+.2f}")
            except Exception as e:
                log(f"   ⚠️ Prehedge close failed in breaker: {e}")
        # Liquidation-dip lane
        lp = port.get("liq_dip_position") or {}
        if lp.get("direction") and lp.get("notional", 0) > 0:
            entry = lp.get("entry_price", price)
            if lp["direction"] == "LONG":
                pnl = lp["notional"] * (price / entry - 1.0)
            else:
                pnl = lp["notional"] * (1.0 - price / entry)
            port["cash"] = port.get("cash", 0.0) + lp["margin"] + pnl
            log(f"   Closing liq-dip {lp['direction']} | PnL ${pnl:+.2f}")
            port["liq_dip_position"] = {}
        cooldown_until = datetime.now(timezone.utc) + timedelta(hours=LEVERAGE_COOLDOWN_HOURS)
        port["leverage_cooldown_until"] = cooldown_until.isoformat()
        port["dd_velocity_trips"] = port.get("dd_velocity_trips", 0) + 1
        return True
    except Exception as e:
        log(f"⚠️ dd-velocity breaker error (ignored, no trip): {e}")
        return False


def init_portfolio(price, regime):
    target_pct = TARGET_ALLOC.get(regime, 0.45)
    btc_dollar_amount = STARTING_BALANCE * target_pct
    qty = btc_dollar_amount / price
    fee = btc_dollar_amount * SPOT_FEE
    avg_cost = price * (1 + SPOT_FEE)
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "starting_balance": STARTING_BALANCE,
        "cash": STARTING_BALANCE - btc_dollar_amount - fee,
        "btc_qty": qty,
        "btc_avg_cost": avg_cost,
        "total_fees_paid": fee,
        "n_trades": 1,
        "n_rebalances": 0,
        "n_drawdown_trims": 0,
        "n_profit_takes": 0,
        "peak_equity": STARTING_BALANCE,
        "last_profit_take_pct": 0.0,
        "current_regime": regime,
        # Over-trading guards (v2)
        "regime_history": [regime],            # last N regimes for hysteresis
        "last_rebalance_ts": now_iso,
        "active_regime": regime,               # regime the agent is acting on (vs detected)
        "rebalance_log": [now_iso],            # timestamps of recent rebalances
        # Per-regime performance attribution (v3, 2026-05-14)
        "regime_perf": {},                     # regime -> {visits, hours, our_pnl, hodl_pnl, alpha}
        "prev_cycle_equity": STARTING_BALANCE - btc_dollar_amount - fee + qty * price,
        "prev_cycle_price": price,
        "prev_cycle_ts": now_iso,
        # Futures lane (v7)
        "futures_position": {"direction": None, "margin": 0, "notional": 0, "entry_price": 0, "opened_at": None},
        "futures_pnl": 0.0,
        "futures_kill": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Trading actions
# ---------------------------------------------------------------------------
def buy_btc(port, price, dollar_amount, reason="rebalance"):
    if dollar_amount <= 0 or port["cash"] < dollar_amount:
        return False
    qty = dollar_amount / price
    fee = dollar_amount * SPOT_FEE
    if port["cash"] < dollar_amount + fee:
        dollar_amount = port["cash"] / (1 + SPOT_FEE) * 0.99
        qty = dollar_amount / price
        fee = dollar_amount * SPOT_FEE
    new_qty = port["btc_qty"] + qty
    new_avg = ((port["btc_qty"] * port["btc_avg_cost"]) + (qty * price * (1 + SPOT_FEE))) / new_qty if new_qty > 0 else price
    port["cash"] -= (dollar_amount + fee)
    port["btc_qty"] = new_qty
    port["btc_avg_cost"] = new_avg
    port["total_fees_paid"] += fee
    port["n_trades"] += 1
    append_trade({
        "ts": datetime.now(timezone.utc).isoformat(),
        "side": "buy",
        "type": "BUY",
        "reason": reason,
        "price": price,
        "qty": qty,
        "usd": dollar_amount,
        "fee": fee,
        "pnl_usd": 0.0,
        "new_total_qty": new_qty,
        "new_avg_cost": new_avg,
    })
    log(f"  BUY  {qty:.6f} BTC @ ${price:,.2f} (${dollar_amount:.2f}) — {reason}")
    return True


def sell_btc(port, price, qty, reason="rebalance"):
    qty = min(qty, port["btc_qty"])
    if qty <= 0:
        return False
    proceeds = qty * price
    fee = proceeds * SPOT_FEE
    port["cash"] += proceeds - fee
    port["btc_qty"] -= qty
    port["total_fees_paid"] += fee
    port["n_trades"] += 1
    pnl = (price - port["btc_avg_cost"]) * qty
    append_trade({
        "ts": datetime.now(timezone.utc).isoformat(),
        "side": "sell",
        "type": "SELL", "strategy_id": "hodl", "entry_ts": port.get("created_at"),
        "reason": reason,
        "price": price,
        "qty": qty,
        "usd": proceeds,
        "fee": fee,
        "pnl_usd": round(pnl, 2),
        "remaining_btc": port["btc_qty"],
        "_entry_price": port["btc_avg_cost"],  # self-heal win_rate calc
    })
    log(f"  SELL {qty:.6f} BTC @ ${price:,.2f} (${proceeds:.2f}) — {reason}")
    return True


def _rebuild_strategy_registry():
    """Rebuild STRATEGY_REGISTRY from current MR_WEIGHT and FUTURES_WEIGHT globals."""
    global STRATEGY_REGISTRY, MR_WEIGHT, FUTURES_WEIGHT, TIMESFM_SIGNAL_WEIGHT
    LIQ_DIP_WEIGHT = 0.03  # v14: always active, must be in spot budget
    ml_weight = ML_SCANNER_WEIGHT if ML_SCANNER_WEIGHT > 0 else 0.0
    hodl_weight = 1.0 - MR_WEIGHT - ml_weight - LIQ_DIP_WEIGHT - 0.09
    STRATEGY_REGISTRY = [
        HODLStrategy(weight=hodl_weight),
        MeanReversionStrategy(weight=MR_WEIGHT),
        FuturesLaneStrategy(weight=FUTURES_WEIGHT),
        LiquidationDipStrategy(weight=LIQ_DIP_WEIGHT),
        FundingMeanReversionStrategy(weight=0.01),
        CvdMomentumStrategy(weight=0.01),
        VolBreakoutStrategy(weight=0.01),
        CrossAssetLeadStrategy(weight=0.01),
        LiquidationScalpStrategy(weight=0.01),
        OiDivergenceStrategy(weight=0.01),
    ]
    if TIMESFM_SIGNAL_WEIGHT > 0:
        STRATEGY_REGISTRY.append(TimesFMSignalStrategy(weight=TIMESFM_SIGNAL_WEIGHT))
    if ml_weight > 0:
        STRATEGY_REGISTRY.append(MLScannerStrategy(weight=ml_weight))


# ---------------------------------------------------------------------------
# Strategy framework (v6 Stage 2)
# ---------------------------------------------------------------------------
# Goal: turn the monolithic run_cycle() into a thin orchestrator that
# delegates "what allocation do we want?" to one or more pluggable strategies.
# Stage 2 keeps behavior IDENTICAL — only HODL is registered, with weight 1.0.
# Stage 3 will add a second strategy (CHOP mean-reversion) without touching
# the orchestrator.
#
# Architecture rules:
#   - Each Strategy gets a "weight" (its slice of total equity, 0.0 - 1.0).
#     Weights across all strategies should sum to 1.0.
#   - Each strategy evaluates against an immutable CycleContext and returns a
#     StrategyDecision (target alloc within its slice).
#   - combine_decisions() sums per-strategy (weight * target_alloc * equity)
#     into a single total_target_btc_value the orchestrator drives toward.
#   - Safety nets (panic-halt, drawdown-trim) stay at the cycle level — they
#     evaluate total portfolio risk, NOT per-strategy.
#   - Profit ladder also stays at cycle level for Stage 2; revisit in Stage 4.

@dataclass(frozen=True)
class CycleContext:
    """Immutable view of market + portfolio state passed to every strategy.

    Strategies read this, never mutate it. The orchestrator owns mutation
    of the live portfolio dict after collecting all strategy decisions.
    """
    price: float
    regime: str               # raw detected regime this cycle
    active_regime: str        # hysteresis-smoothed regime currently in effect
    signals: dict             # MA20, MA50, RSI, ATR, change_7d, etc.
    total_equity: float
    cash: float
    btc_qty: float
    drawdown_from_peak: float # 0.0 - 1.0
    pnl_pct: float            # absolute PnL fraction vs starting balance
    portfolio: dict           # passthrough for strategies needing history


@dataclass
class StrategyDecision:
    """What a strategy wants done this cycle.

    For spot strategies target_alloc_pct is the fraction in BTC.
    For futures strategies futures_direction is -1 (short), 0 (flat), or +1 (long).
    Stage 3+ may add intent_list for active strategies (limit orders,
    trailing stops, etc.). Keeping this minimal until proven needed.
    """
    target_alloc_pct: float = 0.0    # 0.0 - 1.0; fraction in BTC (spot strategies)
    futures_direction: int = 0       # -1=SHORT, 0=FLAT, +1=LONG (futures only)
    notes: str = ""                  # human-readable reasoning


class Strategy:
    """Base class. Subclasses must set `name` and implement `evaluate()`."""
    name: str = "base"
    weight: float = 1.0

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, ctx: CycleContext) -> StrategyDecision:
        raise NotImplementedError


class HODLStrategy(Strategy):
    """The current behavior, wrapped as a strategy.

    In HODL_MODE (default), every regime maps to the same allocation
    (TARGET_ALLOC is flat at fixed_alloc_pct), so this becomes a pure
    fixed-allocation rebalancer. With HODL_MODE=False it's a regime-driven
    allocator using the per-regime TARGET_ALLOC table.
    """
    name = "hodl"

    def evaluate(self, ctx: CycleContext) -> StrategyDecision:
        target = TARGET_ALLOC.get(ctx.active_regime, 0.45)
        # Apply Fear & Greed sentiment modifier (contrarian alpha)
        target = _apply_sentiment_modifier(target, ctx.signals)
        mode_tag = "HODL_MODE" if HODL_MODE else "REGIME_ACTIVE"
        return StrategyDecision(
            target_alloc_pct=target,
            notes=f"{mode_tag} target={target*100:.0f}% regime={ctx.active_regime}",
        )


class MeanReversionStrategy(Strategy):
    """Active alpha source for CHOP regime — buy oversold dips, sell rallies.

    Thesis: in CHOP regime (no trend), BTC price oscillates around a 24h mean.
    We compute z-score of current price vs that mean and map z → target alloc
    *within this strategy's slice*. Oversold (z << 0) → high BTC; overbought
    (z >> 0) → low BTC.

    Outside CHOP regime we return BASELINE_ALLOC — passive, fights nothing.
    This ensures MR ADDS volatility to the slice during CHOP (when it helps)
    and stays silent during trends (when fighting them would hurt).

    Why this works mechanically:
      - Per-regime PnL data shows CHOP is our worst regime (-$22.97 alpha over
        71 hours) because we just sit there paying no fees. MR turns those
        hours into trading opportunities.
      - z-score normalization is volatility-aware: in calm CHOP we'll trigger
        more often (small moves → big z); in volatile CHOP we'll trigger less.
        This is the mean-reversion analog of "buy at the lower Bollinger band".

    Capital safety:
      - With weight=0.20, this strategy can swing total BTC exposure by at
        most ±(MAX_ALLOC - MIN_ALLOC) * 0.20 = ±16 percentage points.
      - Even if MR's logic goes haywire it cannot blow up the book — the
        cycle's safety stack (panic-halt, drawdown-trim) still applies.
    """
    name = "mean_revert_chop"

    BASELINE_ALLOC = 0.50    # neutral position when not active
    MAX_ALLOC = 0.90         # max BTC at deep oversold
    MIN_ALLOC = 0.10         # min BTC at deep overbought

    @property
    def _oversold_z(self):
        return MR_OVERSOLD_Z

    @property
    def _overbought_z(self):
        return MR_OVERBOUGHT_Z

    def evaluate(self, ctx: CycleContext) -> StrategyDecision:
        # Inactive outside CHOP — passive baseline alloc within slice
        if ctx.active_regime != "CHOP":
            return StrategyDecision(
                target_alloc_pct=self.BASELINE_ALLOC,
                notes=f"INACTIVE (regime {ctx.active_regime} ≠ CHOP) — baseline {self.BASELINE_ALLOC*100:.0f}%",
            )
        # Pull pre-computed 24h stats from signals (added by detect_regime)
        z = ctx.signals.get("price_z_24h")
        if z is None or ctx.signals.get("price_std_24h", 0) <= 0:
            return StrategyDecision(
                target_alloc_pct=self.BASELINE_ALLOC,
                notes="INACTIVE (no usable price stats) — baseline",
            )
        oversold = self._oversold_z
        overbought = self._overbought_z
        # Map z-score to allocation within this strategy's slice.
        # Linear interpolation with clamps at the OVERSOLD/OVERBOUGHT thresholds.
        if z <= oversold:
            target = self.MAX_ALLOC
            tag = "DEEP_OVERSOLD"
        elif z >= overbought:
            target = self.MIN_ALLOC
            tag = "DEEP_OVERBOUGHT"
        elif z < 0:
            # interpolate BASELINE → MAX as z goes from 0 to oversold
            frac = abs(z) / abs(oversold)
            target = self.BASELINE_ALLOC + (self.MAX_ALLOC - self.BASELINE_ALLOC) * frac
            tag = "MILD_OVERSOLD"
        else:
            # interpolate BASELINE → MIN as z goes from 0 to overbought
            frac = z / overbought
            target = self.BASELINE_ALLOC - (self.BASELINE_ALLOC - self.MIN_ALLOC) * frac
            tag = "MILD_OVERBOUGHT"
        return StrategyDecision(
     target_alloc_pct=target,
     notes=f"CHOP {tag} z={z:+.2f} → slice target {target*100:.0f}%",
 )


class FuturesLaneStrategy(Strategy):
    """Enhanced directional futures strategy (v8).

    Trades long in BULL/STRONG_BULL, short in BEAR/STRONG_BEAR.
    Rotates to strongest/weakest coins using ML scanner confidence scores.
    Flat in CHOP/NEUTRAL — no trend, no bet.

    Safety:
      - STRONG regimes get full conviction (2x leverage)
      - BULL/BEAR get half conviction (1x leverage)
      - Kill switch at -5% cumulative futures PnL
      - ATR trailing stop at 2x ATR
      - Max loss stop at 30% of margin per trade
    """
    name = "futures_lane"

    def evaluate(self, ctx: CycleContext) -> StrategyDecision:
        regime = ctx.active_regime
        if regime == "STRONG_BULL":
            return StrategyDecision(
                futures_direction=1,
                notes=f"FUTURES LONG {regime} (2x conviction)",
            )
        elif regime == "STRONG_BEAR":
            return StrategyDecision(
                futures_direction=-1,
                notes=f"FUTURES SHORT {regime} (2x conviction)",
            )
        elif regime == "BULL":
            return StrategyDecision(
                futures_direction=1,
                notes=f"FUTURES LONG {regime} (1x conviction)",
            )
        elif regime == "BEAR":
            return StrategyDecision(
                futures_direction=-1,
                notes=f"FUTURES SHORT {regime} (1x conviction)",
            )
        else:
            return StrategyDecision(
                futures_direction=0,
                notes=f"FUTURES FLAT regime={regime} (no trend)",
            )


class MLScannerStrategy(Strategy):
    """ML-driven multi-coin scanner strategy (v8).

    Every cycle, scores all 200+ tracked coins using the trained XGBoost+LightGBM
    ensemble and picks the top N above confidence threshold. Allocates its slice
    of equity equally across the top picks.

    Returns target_alloc_pct=0.0 (doesn't want BTC — wants altcoins). The
    orchestrator calls _execute_ml_positions() after the main BTC rebalance
    to manage the actual altcoin buys/sells.

    Safety:
      - Small weight (default 5%) — can't blow up the book
      - Max per-coin cap (default 2%) — no single coin concentration
      - Min confidence threshold (default 0.54) — filters noise
      - Max top_n (5) — limits diversification drag
    """
    name = "ml_scanner"

    def evaluate(self, ctx: CycleContext) -> StrategyDecision:
        """Return 0% BTC target — this strategy's capital goes to altcoins."""
        return StrategyDecision(
            target_alloc_pct=0.0,  # Don't allocate to BTC — use altcoins
            notes=f"ML_SCANNER active top_n={ML_SCANNER_TOP_N} min_conf={ML_SCANNER_MIN_CONFIDENCE}",
        )


class LiquidationDipStrategy(Strategy):
    """Contrarian liquidation-cascade dip buyer (v1).
    ...
    """
    name = "liquidation_dip"

    def evaluate(self, ctx: CycleContext) -> StrategyDecision:
        return StrategyDecision(
            target_alloc_pct=0.0,
            notes="LIQ_DIP watching for cascade signals",
        )


# ══════════════════════════════════════════════════════════════════════
# Auto-generated Strategies (v19 — recursive self-improvement)
# ══════════════════════════════════════════════════════════════════════

class FundingMeanReversionStrategy(Strategy):
    name = "funding_mr"
    def evaluate(self, ctx):
        f_rate = ctx.signals.get("funding_rate", 0) if ctx.signals else 0
        z = (f_rate - 0.0001) / 0.0003
        if z < -2.0: return StrategyDecision(target_alloc_pct=0.65, notes="funding_mr: oversold")
        if z > 2.0: return StrategyDecision(target_alloc_pct=0.35, notes="funding_mr: overbought")
        return StrategyDecision(target_alloc_pct=0.50, notes="funding_mr: neutral")

class CvdMomentumStrategy(Strategy):
    name = "cvd_momentum"
    def evaluate(self, ctx):
        c4 = ctx.signals.get("cvd_4h", 0) if ctx.signals else 0
        c8 = ctx.signals.get("cvd_8h", 0) if ctx.signals else 0
        a = c4 - c8
        if a > 0 and c4 > 0: return StrategyDecision(target_alloc_pct=0.70, notes="cvd_momentum: bullish")
        if a < 0 and c4 < 0: return StrategyDecision(target_alloc_pct=0.30, notes="cvd_momentum: bearish")
        return StrategyDecision(target_alloc_pct=0.50, notes="cvd_momentum: neutral")

class VolBreakoutStrategy(Strategy):
    name = "vol_breakout"
    def evaluate(self, ctx):
        atr = ctx.signals.get("atr_14", 0) if ctx.signals else 0
        am = ctx.signals.get("atr_mean_24h", atr) if ctx.signals else atr
        ma20 = ctx.signals.get("MA20", ctx.price) if ctx.signals else ctx.price
        if atr > am * 2 and ctx.price > ma20: return StrategyDecision(target_alloc_pct=0.75, notes="vol_breakout: expansion")
        if atr > am * 1.5: return StrategyDecision(target_alloc_pct=0.60, notes="vol_breakout: elevated")
        return StrategyDecision(target_alloc_pct=0.45, notes="vol_breakout: normal")

class CrossAssetLeadStrategy(Strategy):
    name = "cross_asset"
    def evaluate(self, ctx):
        eth = ctx.signals.get("eth_return_4h", 0) if ctx.signals else 0
        sol = ctx.signals.get("sol_return_4h", 0) if ctx.signals else 0
        lead = (eth + sol) / 2
        if lead > 0.01: return StrategyDecision(target_alloc_pct=0.70, notes=f"cross_asset: alts +{lead*100:.0f}%")
        if lead < -0.01: return StrategyDecision(target_alloc_pct=0.35, notes=f"cross_asset: alts {lead*100:.0f}%")
        return StrategyDecision(target_alloc_pct=0.50, notes="cross_asset: neutral")

class LiquidationScalpStrategy(Strategy):
    name = "liq_scalp"
    def evaluate(self, ctx):
        ll = ctx.signals.get("liq_long_1h", 0) if ctx.signals else 0
        ls = ctx.signals.get("liq_short_1h", 0) if ctx.signals else 0
        r = (ll + ls) / max(ll + ls, 1000)
        if r > 2 and ll > ls * 1.5: return StrategyDecision(target_alloc_pct=0.60, notes="liq_scalp: long cascade")
        if r > 2 and ls > ll * 1.5: return StrategyDecision(target_alloc_pct=0.40, notes="liq_scalp: short cascade")
        return StrategyDecision(target_alloc_pct=0.50, notes="liq_scalp: normal")

class OiDivergenceStrategy(Strategy):
    name = "oi_divergence"
    def evaluate(self, ctx):
        oi = ctx.signals.get("oi_change_4h", 0) if ctx.signals else 0
        px = ctx.signals.get("price_change_4h", 0) if ctx.signals else 0
        if oi > 0.02 and px < -0.01: return StrategyDecision(target_alloc_pct=0.65, notes="oi_div: squeeze")
        if oi < -0.02 and px > 0.01: return StrategyDecision(target_alloc_pct=0.35, notes="oi_div: distribution")
        return StrategyDecision(target_alloc_pct=0.50, notes="oi_div: aligned")

AGI_STRATEGY_BUDGET = 0.06
#   HODL @ (1.0 - MR_WEIGHT) + Mean-Reversion @ MR_WEIGHT
#   Futures lane is orthogonal — its weight is FUTURES_WEIGHT * equity as margin,
#   notional is FUTURES_WEIGHT * equity * FUTURES_LEVERAGE.


class TimesFMSignalStrategy(Strategy):
    """TimesFM foundation model direction signal (Google Research, 200M params).

    Evaluated zero-shot on BTC: 58.3% direction accuracy vs 48.9% persistence
    (92 windows, +10.1% edge). 4h horizon is the strongest signal (+13.0%).

    Only fires when confidence > 0.5. When active, biases target_alloc
    up (bullish) or down (bearish) by a configurable amount.
    """
    name = "timesfm_signal"

    def evaluate(self, ctx: CycleContext) -> StrategyDecision:
        try:
            log(f"  [timesfm_signal] DEBUG: sys.path={sys.path}")
            log(f"  [timesfm_signal] DEBUG: sys.executable={sys.executable}")
            from timesfm_signal import get_signal
            direction, confidence = get_signal()
        except Exception:
            return StrategyDecision(
                target_alloc_pct=0.50,
                notes="timesfm_signal: import failed — neutral"
            )

        if confidence < 0.5 or abs(direction) < 0.1:
            return StrategyDecision(
                target_alloc_pct=0.50,
                notes=f"timesfm_signal: no edge (dir={direction:+.3f}, conf={confidence:.2f})"
            )

        # Active signal: bias allocation toward the signal direction
        bias = direction * 0.30  # Max ±30% allocation swing
        target = 0.50 + bias  # Center at 50%, range 20%-80%
        target = max(0.20, min(0.80, target))

        dir_str = "BULLISH" if direction > 0 else "BEARISH"
        return StrategyDecision(
            target_alloc_pct=target,
            notes=f"timesfm_signal: {dir_str} (dir={direction:+.3f}, conf={confidence:.2f}, target={target:.0%})"
        )


# Spot strategies sum to 1.0; futures lane is separate margin.
# Default registry — _rebuild_strategy_registry() replaces this at runtime.
STRATEGY_REGISTRY: list[Strategy] = [
    HODLStrategy(weight=1.0 - MR_WEIGHT),
    MeanReversionStrategy(weight=MR_WEIGHT),
    FuturesLaneStrategy(weight=FUTURES_WEIGHT),
    LiquidationDipStrategy(weight=0.03),
    FundingMeanReversionStrategy(weight=0.01),
    CvdMomentumStrategy(weight=0.01),
    VolBreakoutStrategy(weight=0.01),
    CrossAssetLeadStrategy(weight=0.01),
    LiquidationScalpStrategy(weight=0.01),
    OiDivergenceStrategy(weight=0.01),
    TimesFMSignalStrategy(weight=TIMESFM_SIGNAL_WEIGHT),
]


def combine_decisions(
    decisions: list[tuple[Strategy, StrategyDecision]],
    total_equity: float,
) -> tuple[float, float, int, int]:
    """Aggregate per-strategy target allocations into total portfolio targets.

    Returns: (total_target_btc_value, effective_combined_alloc_pct, futures_direction, consensus_score)
    futures_direction is -1 (short), 0 (flat), or +1 (long) from the futures lane.
    consensus_score is how many strategies agree with the prevailing direction (0-10+).
    """
    if total_equity <= 0 or not decisions:
        return 0.0, 0.0, 0, 0
    total_btc_target_value = 0.0
    futures_dir = 0
    # Count directional consensus: bullish=target>0.50, bearish=target<0.45
    # EXCLUDE structural zeros (strategies deploying capital to alts/futures, not BTC)
    # unless they explicitly express a direction via futures_direction
    bullish = 0
    bearish = 0
    for strategy, decision in decisions:
        strategy_capital = total_equity * strategy.weight
        strategy_btc_target = strategy_capital * decision.target_alloc_pct
        total_btc_target_value += strategy_btc_target
        if decision.futures_direction != 0:
            futures_dir = decision.futures_direction
        # Only count strategies expressing a real directional view on BTC
        # Structural zeros (target==0, no futures signal) are deploying capital elsewhere
        is_structural_zero = (decision.target_alloc_pct == 0.0 and decision.futures_direction == 0)
        if not is_structural_zero and strategy.weight >= 0.01:
            if decision.target_alloc_pct > 0.50:
                bullish += 1
            elif decision.target_alloc_pct < 0.45:
                bearish += 1
    # Consensus = number of strategies on the "winning" side
    consensus = max(bullish, bearish) if futures_dir != 0 else 0
    effective_alloc = total_btc_target_value / total_equity
    return total_btc_target_value, effective_alloc, futures_dir, consensus


def _close_futures_position(port, price, reason, *, trade_sink=None, now=None):
    """Close any open futures position at ``price``: bank margin+pnl to cash, record a
    tagged FUTURES_CLOSE trade (strategy_id=futures_lane), and flat the position.

    Returns True if a position was closed, False if none was open OR ``price`` is
    unavailable/non-positive — fail-safe: never close at a missing price (that would
    have crashed run_cycle via price/entry). Mirrors the inline close used across the
    cycle; the single tested path used by the benchmark-hold de-risk flatten.
    """
    fp = port.get("futures_position") or {}
    if not (fp.get("direction") and fp.get("notional", 0) > 0):
        return False
    if not price or price <= 0:                  # no usable price -> do NOT close
        return False
    entry = fp.get("entry_price") or price
    if entry <= 0:
        return False
    if fp["direction"] == "LONG":
        pnl = fp["notional"] * (price / entry - 1.0)
    else:
        pnl = fp["notional"] * (1.0 - price / entry)
    port["cash"] = port.get("cash", 0.0) + fp["margin"] + pnl
    port["futures_pnl"] = port.get("futures_pnl", 0.0) + pnl
    (trade_sink or append_trade)({
        "ts": now or datetime.now(timezone.utc).isoformat(),
        "side": "close_" + fp["direction"].lower(),
        "type": "FUTURES_CLOSE", "strategy_id": "futures_lane",
        "entry_ts": fp.get("opened_at"), "_entry_price": fp.get("entry_price"),
        "reason": reason, "price": price, "qty": fp["notional"],
        "usd": fp["margin"] + pnl, "fee": 0.0, "pnl_usd": round(pnl, 2),
        "direction": fp["direction"],
    })
    log(f"  🛡️ Futures FLATTEN-CLOSE {fp['direction']} PnL ${pnl:+.2f} (reason={reason})")
    port["futures_position"] = {"direction": None, "margin": 0, "notional": 0, "entry_price": 0, "opened_at": None}
    return True


def _execute_futures(port, price, futures_dir, regime, equity, signals=None, consensus=0):
    """Manage futures position with conviction-scaled leverage.

    Leverage tiers based on strategy consensus (auto-tunable):
      consensus ≥ CONSENSUS_IRONCLAD (8):  5x — every strategy agrees
      consensus ≥ CONSENSUS_STRONG (5):    3x — strong majority
      consensus ≥ CONSENSUS_MODERATE (3):  2x — decent agreement
      else:                                 1x — weak/solo signal, minimum bet

    Hard cap: 5x. Volatility gate drops 1x if ATR > VOLATILITY_GATE_ATR.

    Safety stack (in order):
      1. Kill switch: disable if cumulative futures PnL <= -5% of starting balance
      2. Regime gate: only STRONG_BULL / STRONG_BEAR (not mild BULL/BEAR)
      3. Max loss stop: close position if unrealized loss > 30% of margin
      4. ATR trailing stop: close if price moves against us by 2x ATR from entry
      5. Insufficient cash guard: skip if margin > available cash

    Futures position tracked in port['futures_position'].
    PnL tracked in port['futures_pnl'] (cumulative realized).
    Kill switch: port['futures_kill'] = True (sticky until manual reset).
    """
    # === Kill switch ===
    if port.get("futures_kill"):
        _close_futures_position(port, price, "futures_kill_engaged")
        return  # permanently disabled until manual reset
    total_futures_pnl = port.get("futures_pnl", 0.0)
    if total_futures_pnl <= -port.get("starting_balance", STARTING_BALANCE) * 0.05:
        log(f"  💀 Futures KILL SWITCH — cumulative PnL ${total_futures_pnl:+.2f} <= -$250 (-5%). Futures disabled permanently. Run panic-reset to clear.")
        port["futures_kill"] = True
        _close_futures_position(port, price, "kill_switch")
        return

    if FUTURES_WEIGHT <= 0:
        return  # futures lane disabled via params

    fp = port.get("futures_position") or {}
    current_dir = fp.get("direction")

    # === Regime gate: BULL/BEAR + STRONG ===
    tradeable_regimes = ("STRONG_BULL", "STRONG_BEAR", "BULL", "BEAR")
    if regime not in tradeable_regimes:
        # Close any open position — no trend, no bet
        if current_dir and fp.get("notional", 0) > 0:
            entry = fp.get("entry_price", price)
            if current_dir == "LONG":
                pnl = fp["notional"] * (price / entry - 1.0)
            else:
                pnl = fp["notional"] * (1.0 - price / entry)
            port["cash"] += fp["margin"] + pnl
            port["futures_pnl"] = port.get("futures_pnl", 0.0) + pnl
            log(f"  🔄 Futures CLOSE {current_dir} (regime {regime} — no trend) PnL ${pnl:+.2f} | total ${port['futures_pnl']:+.2f}")
            append_trade({
                "ts": datetime.now(timezone.utc).isoformat(),
                "side": "close_" + current_dir.lower(),
                "type": "FUTURES_CLOSE", "strategy_id": "futures_lane", "entry_ts": fp.get("opened_at"), "_entry_price": fp.get("entry_price"),
                "reason": f"regime_{regime}_no_trend",
                "price": price,
                "qty": fp["notional"],
                "usd": fp["margin"] + pnl,
                "fee": 0.0,
                "pnl_usd": round(pnl, 2),
                "direction": current_dir,
            })
            port["futures_position"] = {"direction": None, "margin": 0, "notional": 0, "entry_price": 0, "opened_at": None}
        return

    req_dir = "LONG" if regime in ("STRONG_BULL", "BULL") else "SHORT"

    # === Futures coin rotation (v20): scan for weakest/strongest coins ===
    # Uses ML scanner to find coins that outperform/underperform BTC.
    # Each aligned coin adds +1 to consensus, boosting leverage tier.
    try:
        from quantforge_ml_scanner import scan_coins
        scan_result = scan_coins(top_n=5, min_confidence=0.50)
        if scan_result.get("model_ok") and scan_result.get("picks"):
            coins = [c["symbol"] for c in scan_result["picks"] if c["symbol"] not in
                     {"USDT","USDC","DAI","XAUT","PAXG","WBTC","BTCB","STETH","WETH"}]
            if coins:
                if regime in ("BEAR", "STRONG_BEAR"):
                    # Short the weakest (highest confidence = most likely to underperform in bear)
                    rotation = coins[:3]
                    log(f"  🎯 Futures rotation SHORT: {rotation}")
                    consensus += len(rotation)
                elif regime in ("BULL", "STRONG_BULL"):
                    # Long the strongest (highest confidence = most likely to outperform in bull)
                    rotation = coins[:3]
                    log(f"  🎯 Futures rotation LONG: {rotation}")
                    consensus += len(rotation)
    except Exception as e:
        log(f"  ⚠️ Futures rotation scan: {e}")

    # === Conviction-scaled leverage: consensus → tier → multiplier ===
    # More strategies agreeing on direction = higher conviction = more leverage
    if consensus >= CONSENSUS_IRONCLAD:       # 8+ strategies agree — ironclad
        base_leverage = min(5.0, FUTURES_LEVERAGE + 2)  # up to 5x
    elif consensus >= CONSENSUS_STRONG:        # 5+ agree — strong consensus
        base_leverage = max(FUTURES_LEVERAGE, 3.0)       # 3x
    elif consensus >= CONSENSUS_MODERATE:       # 3+ agree — decent agreement
        base_leverage = max(FUTURES_LEVERAGE - 1, 2.0)  # 2x
    else:                                       # weak / solo signal
        base_leverage = 1.0                              # 1x minimum bet

    # === Volatility gate: reduce leverage by 1x if ATR > threshold ===
    # Don't lever into chaos — high ATR means whipsaw risk
    active_leverage = base_leverage
    if active_leverage > 0 and signals:
        atr_pct_val = signals.get("atr_pct", 0.01)
        if atr_pct_val > VOLATILITY_GATE_ATR:
            active_leverage = max(1.0, active_leverage - 1.0)
            log(f"  🌊 Volatility gate: ATR {atr_pct_val*100:.1f}% > {VOLATILITY_GATE_ATR*100:.1f}% → leverage reduced to {active_leverage:.0f}x")
    elif active_leverage == 0:
        return  # no futures in neutral/chop

    # === Hard leverage ceiling (safety, NON-auto-tunable) ===
    # Bind effective leverage AFTER conviction-scaling + vol-gate, so neither a 5x
    # ironclad tier nor an auto-raised FUTURES_LEVERAGE can over-lever the core lane.
    # (The moonshot sleeve is a SEPARATE, downside-budgeted module — not capped here.)
    if active_leverage > MAX_EFFECTIVE_LEVERAGE:
        log(f"  🧯 Leverage cap: {active_leverage:.0f}x → {MAX_EFFECTIVE_LEVERAGE:.0f}x (hard ceiling)")
        active_leverage = MAX_EFFECTIVE_LEVERAGE

    # === ATR trailing stop check (only if we have signals) ===
    if current_dir and fp.get("notional", 0) > 0 and signals:
        atr_pct_val = signals.get("atr_pct", 0.01)
        entry = fp.get("entry_price", price)
        stop_loss_pct = atr_pct_val * 2.0  # 2x ATR
        if current_dir == "LONG":
            drawdown_pct = (entry - price) / entry
            if drawdown_pct >= stop_loss_pct:
                pnl = fp["notional"] * (price / entry - 1.0)
                port["cash"] += fp["margin"] + pnl
                port["futures_pnl"] = port.get("futures_pnl", 0.0) + pnl
                log(f"  🛑 Futures STOP-LOSS LONG | ATR stop {stop_loss_pct*100:.2f}% | actual DD {drawdown_pct*100:.2f}% | PnL ${pnl:+.2f}")
                append_trade({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "side": "close_long",
                    "type": "FUTURES_CLOSE", "strategy_id": "futures_lane", "entry_ts": fp.get("opened_at"), "_entry_price": fp.get("entry_price"),
                    "reason": "stop_loss_atr",
                    "price": price,
                    "qty": fp["notional"],
                    "usd": fp["margin"] + pnl,
                    "fee": 0.0,
                    "pnl_usd": round(pnl, 2),
                    "direction": "LONG",
                })
                port["futures_position"] = {"direction": None, "margin": 0, "notional": 0, "entry_price": 0, "opened_at": None}
                current_dir = None
        else:  # SHORT
            rally_pct = (price - entry) / entry
            if rally_pct >= stop_loss_pct:
                pnl = fp["notional"] * (1.0 - price / entry)
                port["cash"] += fp["margin"] + pnl
                port["futures_pnl"] = port.get("futures_pnl", 0.0) + pnl
                log(f"  🛑 Futures STOP-LOSS SHORT | ATR stop {stop_loss_pct*100:.2f}% | actual rally {rally_pct*100:.2f}% | PnL ${pnl:+.2f}")
                append_trade({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "side": "close_short",
                    "type": "FUTURES_CLOSE", "strategy_id": "futures_lane", "entry_ts": fp.get("opened_at"), "_entry_price": fp.get("entry_price"),
                    "reason": "stop_loss_atr",
                    "price": price,
                    "qty": fp["notional"],
                    "usd": fp["margin"] + pnl,
                    "fee": 0.0,
                    "pnl_usd": round(pnl, 2),
                    "direction": "SHORT",
                })
                port["futures_position"] = {"direction": None, "margin": 0, "notional": 0, "entry_price": 0, "opened_at": None}
                current_dir = None

    # === Max loss stop (30% of margin) ===
    if current_dir and fp.get("notional", 0) > 0:
        entry = fp.get("entry_price", price)
        margin_used = fp.get("margin", 0)
        if current_dir == "LONG":
            unrealized = fp["notional"] * (price / entry - 1.0)
        else:
            unrealized = fp["notional"] * (1.0 - price / entry)
        max_loss = margin_used * 0.30  # 30% of margin = 1.5% of equity at 5% weight
        if unrealized <= -max_loss:
            port["cash"] += fp["margin"] + unrealized
            port["futures_pnl"] = port.get("futures_pnl", 0.0) + unrealized
            log(f"  🛑 Futures MAX-LOSS STOP {current_dir} | loss ${unrealized:+.2f} exceeds ${max_loss:.2f} (30% of margin) | total futures PnL ${port['futures_pnl']:+.2f}")
            append_trade({
                "ts": datetime.now(timezone.utc).isoformat(),
                "side": "close_" + current_dir.lower(),
                "type": "FUTURES_CLOSE", "strategy_id": "futures_lane", "entry_ts": fp.get("opened_at"), "_entry_price": fp.get("entry_price"),
                "reason": "max_loss_stop",
                "price": price,
                "qty": fp["notional"],
                "usd": fp["margin"] + unrealized,
                "fee": 0.0,
                "pnl_usd": round(unrealized, 2),
                "direction": current_dir,
            })
            port["futures_position"] = {"direction": None, "margin": 0, "notional": 0, "entry_price": 0, "opened_at": None}
            current_dir = None

    # === Already in the correct direction -> HOLD (let the position run; do not churn) ===
    if current_dir == req_dir and fp.get("notional", 0) > 0:
        return

    # === Direction FLIP / (re)establish ===
    # Close any WRONG-direction open position FIRST — credit margin + pnl back to cash
    # and write the close to the ledger — BEFORE opening the new direction.
    # BUG FIX (2026-06-22): the close used to live in the `current_dir == req_dir` branch
    # (which fires only when ALREADY correctly positioned), so a genuine LONG<->SHORT flip
    # was `LONG == SHORT` -> False -> it fell through to "Open new position" below, which
    # does `cash -= margin` and OVERWRITES futures_position — never crediting the old
    # position's margin back. That ORPHANED the old margin (~$400 of simulated margin lost EVERY flip:
    # 4 FUTURES_OPEN / 0 FUTURES_CLOSE in the paper ledger) — the real driver of the
    # post-reset bleed, NOT "no edge". The old `== req_dir` branch also CHURNED a correct
    # position (close+reopen each cycle, resetting entry so no trend ever compounds); it
    # now HOLDS. _close_futures_position is a no-op when flat + fail-safe on bad price, so
    # calling it unconditionally here is safe; it credits cash and ledgers the close.
    _close_futures_position(port, price, f"flip_to_{req_dir}")

    # === Leverage cooldown gate (dd-velocity breaker aftermath) ===
    # Suppress NEW leveraged opens during the cooldown window. The wrong-direction
    # close above still runs (de-risk, don't re-lever) — mirrors the DD-trim buyback
    # suppression: never suppress closes, only opens.
    if _in_leverage_cooldown(port):
        log("  ⏸️ Futures open suppressed — dd-velocity leverage cooldown active")
        return

    # === Open new position ===
    margin = equity * FUTURES_WEIGHT
    notional = margin * active_leverage
    # Honest cost: KuCoin futures taker (TAKER_FEE) is charged on NOTIONAL per fill.
    # Charge the full round-trip (entry + exit) upfront so the lane's real cost hits equity
    # even on the many close paths (stop-loss/max-loss/regime-exit/flip/breaker) — the
    # cost-inclusive report charges the same on a per-fill basis, so the two honest views
    # converge. The futures lane used to record fee=0.0 (traded for free in its own books).
    fee = notional * TAKER_FEE * 2
    if margin + fee > port["cash"]:
        log(f"  ⚠️ Futures skipped — insufficient cash (need ${margin + fee:.2f}, have ${port['cash']:.2f})")
        return
    port["cash"] -= margin + fee
    port["total_fees_paid"] = port.get("total_fees_paid", 0.0) + fee
    port["futures_position"] = {
        "direction": req_dir,
        "margin": margin,
        "notional": notional,
        "entry_price": price,
        "leverage": active_leverage,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    log(f"  🚀 Futures OPEN {req_dir} | margin ${margin:.2f} | notional ${notional:.2f} ({active_leverage}x) @ ${price:,.2f} | rt-fee ${fee:.2f}")
    append_trade({
        "ts": datetime.now(timezone.utc).isoformat(),
        "side": "open_" + req_dir.lower(),
        "type": "FUTURES_OPEN",
        "reason": f"regime_{regime}",
        "price": price,
        "qty": notional,
        "usd": margin,
        "fee": round(fee, 4),
        "pnl_usd": 0.0,
        "direction": req_dir,
        "leverage": active_leverage,
    })


def _execute_liquidation_dip(port, price, equity):
    """Execute liquidation cascade contrarian dip trades (v14).

    Reads deriv_liquidation_long_usd / deriv_liquidation_short_usd from the
    derivatives collector. When liquidations spike >2x recent average, enters
    a contrarian position: buy after long cascade (oversold bounce), short
    after short cascade (overbought reversal).

    Position tracked in port['liq_dip_position']. Small size (3% of equity),
    2h cooldown between entries.
    """
    try:
        import pandas as pd
        deriv_file = os.path.join(DATA_DIR, "derivatives", "derivatives_state_latest.parquet")
        if not os.path.exists(deriv_file):
            return
        df = pd.read_parquet(deriv_file)
        if df.empty:
            return

        # Get latest BTC liquidation data
        btc_row = df[df["symbol"] == "BTC-USDT"]
        if btc_row.empty:
            return
        row = btc_row.iloc[0]
        liq_long = float(row.get("liq_long_usd_24h", 0) or 0)
        liq_short = float(row.get("liq_short_usd_24h", 0) or 0)
        avg_liq_long = float(row.get("liq_long_usd_avg_7d", liq_long) or liq_long)
        avg_liq_short = float(row.get("liq_short_usd_avg_7d", liq_short) or liq_short)
    except Exception:
        return  # collector data unavailable, skip

    LP = port.get("liq_dip_position") or {}
    if LP.get("direction") and LP.get("notional", 0) > 0:
        # Position open — check exit
        entry = LP.get("entry_price", price)
        held_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(LP["opened_at"])).total_seconds() / 3600

        if LP["direction"] == "LONG":
            pnl = LP["notional"] * (price / entry - 1.0)
            # Exit: +3% profit or -5% loss or 24h max hold
            if pnl > LP["notional"] * 0.03 or pnl < -LP["notional"] * 0.05 or held_hours > 24:
                port["cash"] += LP["margin"] + pnl
                log(f"  🔄 LIQ_DIP CLOSE LONG | held {held_hours:.1f}h | PnL ${pnl:+.2f}")
                port["liq_dip_position"] = {}
        else:  # SHORT
            pnl = LP["notional"] * (1.0 - price / entry)
            if pnl > LP["notional"] * 0.03 or pnl < -LP["notional"] * 0.05 or held_hours > 24:
                port["cash"] += LP["margin"] + pnl
                log(f"  🔄 LIQ_DIP CLOSE SHORT | held {held_hours:.1f}h | PnL ${pnl:+.2f}")
                port["liq_dip_position"] = {}
        return  # Position still open, wait

    # Cooldown check
    if LP.get("last_closed_at"):
        last = datetime.fromisoformat(LP["last_closed_at"])
        if (datetime.now(timezone.utc) - last).total_seconds() < 7200:
            return  # 2h cooldown

    # Signal check: liquidation spike >2x average
    long_spike = liq_long > avg_liq_long * 2.0 and avg_liq_long > 0
    short_spike = liq_short > avg_liq_short * 2.0 and avg_liq_short > 0

    if not long_spike and not short_spike:
        return

    # Enter contrarian position
    margin = equity * 0.03  # 3% of equity
    if margin > port["cash"]:
        return

    if long_spike:
        direction = "LONG"
        log_msg = f"  🩸 LIQ_DIP LONG | long liq ${liq_long:,.0f} vs avg ${avg_liq_long:,.0f} ({liq_long/avg_liq_long:.1f}x) — buying the dip"
    else:
        direction = "SHORT"
        log_msg = f"  🩸 LIQ_DIP SHORT | short liq ${liq_short:,.0f} vs avg ${avg_liq_short:,.0f} ({liq_short/avg_liq_short:.1f}x) — fading the pump"

    port["cash"] -= margin
    port["liq_dip_position"] = {
        "direction": direction,
        "margin": margin,
        "notional": margin,  # 1x leverage for safety
        "entry_price": price,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    log(log_msg)


# ── ML BTC directional predictor cache (v31) ──
_ML_BTC_CACHE = None  # dict from last subprocess call; cleared each cycle


def _run_ml_btc_predictor():
    """Run the BTC directional ML predictor as a subprocess via the quant-ops venv.

    Returns dict with keys: direction ('up'|'down'), confidence (float 0-1),
    prob_up (float), prob_down (float), or None on failure.

    Cached per cycle — repeated calls return the cached result without
    re-running the subprocess.
    """
    global _ML_BTC_CACHE
    if _ML_BTC_CACHE is not None:
        return _ML_BTC_CACHE

    import subprocess
    import os as _os

    python_bin = ML_BTC_VENV_PYTHON
    if not _os.path.exists(python_bin):
        python_bin = "python3"

    script = ML_BTC_PREDICTOR_SCRIPT
    try:
        result = subprocess.run(
            [python_bin, script],
            capture_output=True, text=True, timeout=120,
            env={**_os.environ, "PYTHONWARNINGS": "ignore"},
        )
        if result.returncode != 0:
            log(f"  ⚠️ ML BTC predictor subprocess failed (rc={result.returncode}): {result.stderr[:200]}")
            _ML_BTC_CACHE = None
            return None
    except Exception as e:
        log(f"  ⚠️ ML BTC predictor error: {e}")
        _ML_BTC_CACHE = None
        return None

    try:
        pred = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        log(f"  ⚠️ ML BTC predictor bad JSON: {e}")
        _ML_BTC_CACHE = None
        return None

    if "error" in pred:
        log(f"  ⚠️ ML BTC predictor error: {pred['error']}")
        _ML_BTC_CACHE = None
        return None

    _ML_BTC_CACHE = pred
    return pred


def _execute_ml_positions(port, price, equity):
    """Execute ML scanner altcoin positions (v8).

    Calls the ML scanner subprocess to get top coin picks, then:
      - Sells altcoins no longer in the top picks
      - Buys new picks (equal-weight within the ML slice)
      - Rebalances existing positions if drift exceeds threshold

    Uses a subprocess because the ML model requires xgboost/lightgbm
    which live in the quant-ops venv, not the system Python.
    """
    if ML_SCANNER_WEIGHT <= 0:
        return  # ML lane disabled

    ml_slice = equity * ML_SCANNER_WEIGHT
    if ml_slice < 50:
        return  # Too small to matter — min $50

    # Run scanner subprocess
    scanner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "quantforge_ml_scanner.py")
    python_bin = ML_SCANNER_VENV_PYTHON
    if not os.path.exists(python_bin):
        # Fallback: try system python (if xgboost installed)
        python_bin = "python3"

    import subprocess
    try:
        result = subprocess.run(
            [python_bin, scanner_script],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        if result.returncode != 0:
            log(f"  ⚠️ ML scanner subprocess failed: {result.stderr[:200]}")
            return
    except Exception as e:
        log(f"  ⚠️ ML scanner error: {e}")
        return

    # Parse scanner output (format: "1. SYMBOL conf=0.XXXX ...")
    picks = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("Scanned") or line.startswith("ERROR"):
            continue
        # Parse: "  1. TRAC-USDT            conf=0.9617  (xgb=... lgb=...)"
        parts = line.split()
        if len(parts) >= 3:
            symbol = parts[1].strip()
            try:
                conf = float(parts[2].split("=")[1])
            except (IndexError, ValueError):
                continue
            if conf >= ML_SCANNER_MIN_CONFIDENCE:
                picks.append(symbol)

    # ── ML picks in all regimes (v2) ──
    # The model has a 58.4% CV WR edge — it works across regimes.
    # Stablecoins and wrapped tokens are blacklisted to avoid "safe haven" picks.
    ML_BLACKLIST = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "XAUT", "PAXG", "WBTC", "BTCB",
                    "STETH", "WSTETH", "CETH", "WETH"}
    picks = [p for p in picks if p not in ML_BLACKLIST]

    if not picks:
        # No picks above threshold — liquidate all alt positions
        alt_positions = port.get("alt_positions", {})
        for sym in list(alt_positions.keys()):
            pos = alt_positions[sym]
            if pos.get("qty", 0) > 0:
                sale_value = pos["qty"] * price * 0.995
                port["cash"] = port.get("cash", 0) + sale_value
                log(f"  🔄 ML SELL {sym}: {pos['qty']:.6f} @ ~${price:,.2f} = ${sale_value:.2f}")
                append_trade({
                    "ts": datetime.now(timezone.utc).isoformat(), "side": "sell", "type": "ALT_SELL",
                    "reason": "ml_liquidate", "strategy_id": "ml_scanner", "symbol": sym,
                    "price": price, "qty": pos["qty"], "usd": sale_value, "fee": 0.0,
                    "pnl_usd": round(sale_value - pos["qty"] * pos.get("avg_cost", price), 2),
                    "entry_ts": pos.get("added_at"), "_entry_price": pos.get("avg_cost"),
                })
            del alt_positions[sym]
        port["alt_positions"] = alt_positions
        log(f"  🤖 ML Scanner: no picks above {ML_SCANNER_MIN_CONFIDENCE} confidence — liquidated all")
        return

    # Cap to top_n
    picks = picks[:ML_SCANNER_TOP_N]
    per_coin_budget = min(ml_slice / len(picks), equity * ML_SCANNER_MAX_PER_COIN)
    # Risk layer override: use computed max position if tighter
    try:
        from quantforge_risk import RiskContext
        _risk = RiskContext().evaluate(picks=picks, equity=equity)
        risk_adjusted_max = equity * _risk.max_position_pct
        if risk_adjusted_max < per_coin_budget:
            per_coin_budget = risk_adjusted_max
            log(f"  🛡️ Risk-adjusted position: ${per_coin_budget:.2f} (corr={_risk.correlation_penalty:.2f})")
    except Exception:
        pass

    alt_positions = port.get("alt_positions", {})
    current_picks = set(picks)
    current_held = set(alt_positions.keys())

    # Sell coins no longer in picks
    for sym in current_held - current_picks:
        pos = alt_positions[sym]
        if pos.get("qty", 0) > 0:
            sale_value = pos["qty"] * price * 0.995
            port["cash"] = port.get("cash", 0) + sale_value
            log(f"  🔄 ML ROTATE OUT {sym}: sold {pos['qty']:.6f} @ ~${price:,.2f} = ${sale_value:.2f}")
            append_trade({
                "ts": datetime.now(timezone.utc).isoformat(), "side": "sell", "type": "ALT_SELL",
                "reason": "ml_rotate_out", "strategy_id": "ml_scanner", "symbol": sym,
                "price": price, "qty": pos["qty"], "usd": sale_value, "fee": 0.0,
                "pnl_usd": round(sale_value - pos["qty"] * pos.get("avg_cost", price), 2),
                "entry_ts": pos.get("added_at"), "_entry_price": pos.get("avg_cost"),
            })
        del alt_positions[sym]

    # Buy new picks
    for sym in picks:
        if sym not in alt_positions:
            qty = per_coin_budget / price
            cost = per_coin_budget
            if port.get("cash", 0) >= cost:
                port["cash"] = port["cash"] - cost
                alt_positions[sym] = {
                    "qty": qty,
                    "avg_cost": price,
                    "added_at": datetime.now(timezone.utc).isoformat(),
                }
                log(f"  🚀 ML BUY {sym}: {qty:.6f} @ ${price:,.2f} = ${cost:.2f}")
            else:
                log(f"  ⚠️ ML SKIP {sym}: insufficient cash (need ${cost:.2f}, have ${port.get('cash', 0):.2f})")

    port["alt_positions"] = alt_positions
    total_alt_value = sum(
        p.get("qty", 0) * price for p in alt_positions.values()
    )
    log(f"  🤖 ML Scanner: {len(alt_positions)} positions, ${total_alt_value:.2f} value, "
        f"picks: {', '.join(picks)}")


def _execute_funding_arb(port, equity, active_regime):
    """Execute funding rate arbitrage cycle (v2 + levered).

    Calls quantforge_funding_arb_v2.py as a subprocess (baseline).
    Then calls quantforge_funding_arb_levered.py for conviction-scaled
    entries on extreme funding signals (z < -1.0). The levered strategy
    scales position size 1.5x→3x based on funding rate z-score.

    Both strategies use their own margin — do NOT consume spot budget.
    Combined safety cap: max 20% of equity total deployed.
    """
    # Determine effective weight from regime table (if adaptive) or module default
    effective_weight = FUNDING_ARB_WEIGHT
    if REGIME_ADAPTIVE:
        regime_overrides = REGIME_WEIGHT_TABLE.get(active_regime, {})
        effective_weight = regime_overrides.get("funding_arb_weight", FUNDING_ARB_WEIGHT)

    if effective_weight <= 0:
        return

    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    arb_script = os.path.join(script_dir, "quantforge_funding_arb_v2.py")
    levered_script = os.path.join(script_dir, "quantforge_funding_arb_levered.py")
    venv_python = ML_SCANNER_VENV_PYTHON
    if not os.path.exists(venv_python):
        venv_python = "python3"

    # Track total deployed across both strategies
    total_deployed_v2 = 0.0
    total_deployed_levered = 0.0

    # ── Strategy 1: Standard v2 funding arb ──────────────────────
    try:
        result = subprocess.run(
            [venv_python, arb_script, "run"],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        if result.returncode != 0:
            log(f"  ⚠️ Funding arb v2 error: {result.stderr[:300]}")
        else:
            output = result.stdout.strip()
            if output:
                try:
                    data = json.loads(output)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{.*\}', output, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            data = None
                    else:
                        data = None

                if data:
                    closed_trades = data.get("closed_trades", [])
                    entered_positions = data.get("entered_positions", [])
                    summary = data.get("summary", {})

                    FUNDING_ARB_MAX_EQUITY_PCT = 0.15

                    for trade in closed_trades:
                        symbol = trade.get("symbol", "?")
                        pnl_usd = trade.get("pnl_usd", 0)
                        pnl_pct = trade.get("pnl_pct", 0)
                        exit_reason = trade.get("exit_reason", "unknown")
                        hours_held = trade.get("hours_held", 0)
                        log(f"  💸 FundingArb EXIT {symbol}: {exit_reason} | "
                            f"PnL ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%) | "
                            f"Held {hours_held:.1f}h")

                    for pos in entered_positions:
                        symbol = pos.get("symbol", "?")
                        size_usd = pos.get("size_usd", 0)
                        entry_price = pos.get("entry_price", 0)
                        entry_funding = pos.get("entry_funding", 0)
                        funding_annual = pos.get("funding_annual_pct", 0)

                        if total_deployed_v2 + size_usd > equity * FUNDING_ARB_MAX_EQUITY_PCT:
                            log(f"  🛡️ FundingArb cap: cannot enter {symbol} "
                                f"(${size_usd:.0f}) — would exceed 15% equity cap")
                            continue
                        total_deployed_v2 += size_usd
                        log(f"  💹 FundingArb ENTER {symbol}: "
                            f"funding={entry_funding*100:.3f}% ({funding_annual:.1f}% ann) | "
                            f"@${entry_price:,.2f} | size=${size_usd:.0f}")

    except FileNotFoundError:
        log(f"  ⚠️ Funding arb v2 script not found at {arb_script}")
    except subprocess.TimeoutExpired:
        log(f"  ⚠️ Funding arb v2 timed out after 180s")
    except Exception as e:
        log(f"  ⚠️ Funding arb v2 failed: {e}")

    # ── Strategy 2: Conviction-scaled levered funding arb ────────
    # Only call levered strategy when funding arb weight > 0 (gate above)
    if not os.path.exists(levered_script):
        log(f"  ⚠️ Levered funding arb script not found at {levered_script}")
    else:
        try:
            result = subprocess.run(
                [venv_python, levered_script, "run"],
                capture_output=True, text=True, timeout=180,
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
            )
            if result.returncode != 0:
                log(f"  ⚠️ Levered funding arb error: {result.stderr[:300]}")
            else:
                output = result.stdout.strip()
                if output:
                    try:
                        levered_data = json.loads(output)
                    except json.JSONDecodeError:
                        import re
                        json_match = re.search(r'\{.*\}', output, re.DOTALL)
                        if json_match:
                            try:
                                levered_data = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                levered_data = None
                        else:
                            levered_data = None

                    if levered_data:
                        l_closed = levered_data.get("closed_trades", [])
                        l_entered = levered_data.get("entered_positions", [])
                        l_summary = levered_data.get("summary", {})

                        # Process levered exits
                        for trade in l_closed:
                            symbol = trade.get("symbol", "?")
                            pnl_usd = trade.get("pnl_usd", 0)
                            pnl_pct = trade.get("pnl_pct", 0)
                            exit_reason = trade.get("exit_reason", "unknown")
                            hours_held = trade.get("hours_held", 0)
                            partial = " [PARTIAL]" if trade.get("partial") else ""
                            log(f"  ⚡ LeveredArb EXIT{partial} {symbol}: {exit_reason} | "
                                f"PnL ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%) | "
                                f"Held {hours_held:.1f}h")

                        # Process levered entries
                        for pos in l_entered:
                            symbol = pos.get("symbol", "?")
                            size_usd = pos.get("size_usd", 0)
                            entry_price = pos.get("entry_price", 0)
                            entry_funding = pos.get("entry_funding", 0)
                            funding_annual = pos.get("funding_annual_pct", 0)
                            z_score = pos.get("z_score", 0)
                            lever = pos.get("leverage_multiplier", 1)

                            # Combined cap: v2 + levered ≤ 20% of equity
                            combined = total_deployed_v2 + total_deployed_levered + size_usd
                            if combined > equity * 0.20:
                                log(f"  🛡️ LeveredArb cap: cannot enter {symbol} "
                                    f"(${size_usd:.0f}) — combined would exceed 20% equity "
                                    f"(${total_deployed_v2 + total_deployed_levered:.0f} deployed)")
                                continue

                            total_deployed_levered += size_usd
                            log(f"  ⚡ LeveredArb ENTER {symbol}: "
                                f"z={z_score:.2f} lever={lever:.1f}x | "
                                f"funding={entry_funding*100:.3f}% ({funding_annual:.1f}% ann) | "
                                f"@${entry_price:,.4f} | size=${size_usd:.0f}")

        except subprocess.TimeoutExpired:
            log(f"  ⚠️ Levered funding arb timed out after 180s")
        except Exception as e:
            log(f"  ⚠️ Levered funding arb failed: {e}")

    # ── Sync positions from both strategy state files ────────────
    if "funding_arb_positions" not in port:
        port["funding_arb_positions"] = {}
    if "funding_arb_levered_positions" not in port:
        port["funding_arb_levered_positions"] = {}

    # Read v2 state
    funding_arb_state_path = os.path.join(DATA_DIR, "funding_arb_state_v2.json")
    if os.path.exists(funding_arb_state_path):
        try:
            with open(funding_arb_state_path) as f:
                arb_state = json.load(f)
            v2_positions = arb_state.get("positions", {})
            new_v2 = {}
            tdv2 = 0.0
            for symbol, pos_data in v2_positions.items():
                sz = pos_data.get("size_usd", 0)
                tdv2 += sz
                new_v2[symbol] = {
                    "qty": 0,
                    "notional_usd": sz,
                    "entry_price": pos_data.get("entry_price", 0),
                    "entry_funding": pos_data.get("entry_funding", 0),
                    "entry_time": pos_data.get("entry_time", ""),
                    "futures_symbol": pos_data.get("futures_symbol", ""),
                    "strategy": "funding_arb_v2",
                    "legs": pos_data.get("legs", {}),
                }
            port["funding_arb_positions"] = new_v2
            total_deployed_v2 = tdv2
        except Exception as e:
            log(f"  ⚠️ Funding arb state read: {e}")

    # Read levered state
    levered_state_path = os.path.join(DATA_DIR, "funding_arb_levered_state.json")
    if os.path.exists(levered_state_path):
        try:
            with open(levered_state_path) as f:
                levered_state = json.load(f)
            l_positions = levered_state.get("positions", {})
            new_lev = {}
            tdl = 0.0
            for symbol, pos_data in l_positions.items():
                sz = pos_data.get("size_usd", 0)
                tdl += sz
                new_lev[symbol] = {
                    "qty": 0,
                    "notional_usd": sz,
                    "entry_price": pos_data.get("entry_price", 0),
                    "entry_funding": pos_data.get("entry_funding", 0),
                    "entry_z_score": pos_data.get("entry_z_score", 0),
                    "leverage_multiplier": pos_data.get("leverage_multiplier", 1),
                    "entry_time": pos_data.get("entry_time", ""),
                    "futures_symbol": pos_data.get("futures_symbol", ""),
                    "strategy": "funding_arb_levered",
                    "profit_locked": pos_data.get("profit_locked", False),
                    "legs": pos_data.get("legs", {}),
                }
            port["funding_arb_levered_positions"] = new_lev
            total_deployed_levered = tdl
        except Exception as e:
            log(f"  ⚠️ Levered arb state read: {e}")

    # ── Combined summary ─────────────────────────────────────────
    total_deployed = total_deployed_v2 + total_deployed_levered
    max_allowed = equity * 0.20
    pct_deployed = (total_deployed / equity * 100) if equity > 0 else 0

    v2_open = len(port.get("funding_arb_positions", {}))
    levered_open = len(port.get("funding_arb_levered_positions", {}))

    # Read Sharpe from levered state
    l_sharpe = 0.0
    if os.path.exists(levered_state_path):
        try:
            with open(levered_state_path) as f:
                ls = json.load(f)
            pnl_hist = ls.get("pnl_history", [])
            if len(pnl_hist) >= 5:
                import numpy as np
                eqs = [p["equity"] for p in pnl_hist]
                rets = [(eqs[i] - eqs[i-1]) / max(eqs[i-1], 1) for i in range(1, len(eqs))]
                if len(rets) >= 2:
                    rets_arr = np.array(rets)
                    l_sharpe = float(np.mean(rets_arr) / max(np.std(rets_arr, ddof=1), 1e-10))
        except Exception:
            pass

    log(f"  📊 FundingArb: ${total_deployed:.0f} deployed "
        f"({pct_deployed:.1f}% of ${equity:,.0f} equity, "
        f"cap ${max_allowed:.0f}) | "
        f"v2: {v2_open} open | levered: {levered_open} open | "
        f"Sharpe {l_sharpe:.2f}")


# ---------------------------------------------------------------------------
# Main agent cycle
# ---------------------------------------------------------------------------
def run_cycle():
    global _ML_BTC_CACHE, FUTURES_WEIGHT
    _ML_BTC_CACHE = None  # reset predictor cache each cycle
    os.makedirs(DATA_DIR, exist_ok=True)
    log("=== Agent cycle start ===")

    # === Benchmark-gate hold (Phase E) — auto-recovering ===
    # If active trading has PROVEN it underperforms holding BTC (gate verdict over
    # >=20 live trades), default to the benchmark: hold current positions and open
    # nothing this cycle. Dormant until there is sufficient evidence; clears itself
    # automatically once the gate promotes the strategy again.
    if _benchmark_hold_active():
        # HODL means hold SPOT — not freeze leverage. This cycle returns before the
        # futures stop-losses and spot breakers run, so an open futures position would
        # sit UNMANAGED for the whole hold (the gap v29 closed for trip_panic_halt).
        # "Holding" a losing leveraged short is not HODL; it is freezing a bleed. So
        # flatten any open futures with one final de-risking close (fail-safe: skipped
        # if price is unavailable), then hold spot. Spot is intentionally left untouched
        # — holding spot IS the benchmark posture.
        #
        # Persist a heartbeat even when the benchmark gate suppresses trading so
        # watchdog/Telegram alerts don't misreport an intentionally-held agent as stale.
        port = load_portfolio()
        fp = (port or {}).get("futures_position") or {}
        # Only fetch a price (network) when there is actually a position to flatten.
        if port and fp.get("direction"):
            _close_futures_position(port, get_btc_price(), "benchmark_hold")
        if port:
            save_portfolio(port)
        log("BENCHMARK HOLD: active trading trails HODL over the evidence window -> holding spot, no new positions this cycle")
        return

    # === Hard halt check — auto-recovering (v15) ===
    # The halt is no longer a dead end. The agent checks whether the
    # condition that triggered it is still valid. If DD has recovered
    # below 80% of the panic threshold, auto-resume without human help.
    if is_halted():
        try:
            with open(HALT_FILE) as f:
                halt_info = json.load(f)
        except Exception:
            halt_info = {}
        halt_dd = halt_info.get("drawdown_pct", 1.0)

        # v28: Use centralized equity + auto-fix stale peak before deciding
        port_check = load_portfolio()
        price_check = get_btc_price()
        _auto_fix_stale_peak(port_check, price_check)
        current_eq = _true_equity(port_check, price_check)
        current_dd = _true_drawdown(port_check, price_check)

        auto_resume_threshold = PANIC_HALT_PCT * 0.8  # 12% DD

        if current_dd < auto_resume_threshold:
            clear_halt_marker()
            log(f"🧬 auto-recovery: DD now {current_dd*100:.1f}% < {auto_resume_threshold*100:.1f}% threshold (halt was {halt_dd*100:.1f}% at {halt_info.get('halted_at', '?')[:19]}) — resuming autonomously")
        else:
            log(f"⛔ HALTED — DD {current_dd*100:.1f}% still above {auto_resume_threshold*100:.1f}% auto-resume threshold (halted at {halt_info.get('halted_at', '?')[:19]})")
            log(f"   Monitoring: will auto-resume when DD drops below {auto_resume_threshold*100:.0f}%")
            return

    # === Load runtime tunables from params file (v6 Stage 1) ===
    # If qf_strategy_params.json exists, override constants for this cycle.
    # Only TUNABLE_KEYS can be set — safety constants are unreachable here.
    applied_params = load_runtime_params()
    if applied_params:
        log(f"  ⚙️ Runtime params loaded: {applied_params}")

    price = get_btc_price()
    log(f"BTC price: ${price:,.2f}")

    candles = get_btc_klines_1h(REGIME_LOOKBACK_HOURS)
    regime, signals = detect_regime(candles)

    # === Adversarial Debate (v13) — PRIMARY regime detector when active ===
    # When active (level >= 2 via auto-promotion gate), runs Bull vs Bear vs Judge
    # debate as the FIRST regime detector. Combines ALL signal sources into
    # structured arguments, weighs by historical accuracy, and produces a verdict.
    # Shadow mode (level 1): runs but only logs, doesn't affect decisions.
    # Inactive (level 0): skipped entirely.
    # This runs BEFORE micro because when promoted, the debate is the most
    # comprehensive signal — it already incorporates micro, Kronos, Polymarket,
    # swarm, TA, and sentiment into its structured reasoning.
    DEBATE_ACTIVE = True           # Gate-controlled — set False to disable entirely
    DEBATE_SHADOW_ENABLED = True   # Allow shadow mode logging
    DEBATE_MIN_LEVEL_FOR_ACTION = 2  # Level where debate can override regime

    debate_used = False
    if DEBATE_ACTIVE:
        try:
            from quantforge_debate_gate import is_debate_active, is_debate_shadow, get_debate_level

            def _debate_to_regime(debate_regime: str) -> str:
                """Map debate verdict regimes to agent's regime labels."""
                mapping = {
                    "STRONG_BULL": "STRONG_BULL", "BULL": "BULL",
                    "NEUTRAL": "NEUTRAL", "CHOP": "CHOP",
                    "BEAR": "BEAR", "STRONG_BEAR": "STRONG_BEAR",
                }
                return mapping.get(debate_regime, "NEUTRAL")

            debate_level = get_debate_level()
            if debate_level >= DEBATE_MIN_LEVEL_FOR_ACTION or (DEBATE_SHADOW_ENABLED and debate_level >= 1):
                try:
                    from quantforge_debate import adjudicate
                    # Gather ALL detector data for the debate
                    debate_micro = None
                    try:
                        from quantforge_micro_regime import micro_detect_regime
                        micro_r, micro_c, micro_s = micro_detect_regime()
                        debate_micro = {
                            "cvd_1h": micro_s.cvd_1h, "cvd_4h": micro_s.cvd_4h,
                            "cvd_trend": micro_s.cvd_trend,
                            "pressure_imbalance": micro_s.pressure_imbalance,
                            "pressure_imbalance_trend": micro_s.pressure_imbalance_trend,
                            "depth_imbalance": micro_s.depth_imbalance,
                            "depth_imbalance_trend": micro_s.depth_imbalance_trend,
                            "micro_return_drift": micro_s.micro_return_drift,
                            "vol_buy_ratio": micro_s.vol_buy_ratio,
                            "spread_widening": micro_s.spread_widening,
                        }
                    except Exception:
                        pass

                    debate_kronos = None
                    try:
                        from quantforge_kronos_regime import kronos_detect_regime, kronos_is_available
                        if kronos_is_available():
                            k_regime, k_conf, k_pct = kronos_detect_regime(candles)
                            debate_kronos = {
                                "regime": k_regime, "confidence": k_conf,
                                "forecast_pct": k_pct,
                            }
                    except Exception:
                        pass

                    debate_polymarket = None
                    try:
                        from quantforge_polymarket import fetch_polymarket_sentiment
                        pm_signal = fetch_polymarket_sentiment()
                        debate_polymarket = {
                            "sentiment": pm_signal.sentiment,
                            "bull_prob": pm_signal.bull_prob,
                            "conviction": pm_signal.conviction,
                            "trend_24h": pm_signal.trend_24h,
                        }
                    except Exception:
                        pass

                    debate_swarm = None
                    try:
                        from quantforge_swarm_regime import swarm_detect_regime
                        s_regime, s_conf, s_votes = swarm_detect_regime(
                            candles, signals, min_agreement=0.30, min_voters=2
                        )
                        debate_swarm = {
                            "regime": s_regime, "confidence": s_conf,
                            "votes": [
                                {"voter": v.voter, "regime": v.regime, "confidence": v.confidence}
                                for v in s_votes
                            ],
                        }
                    except Exception:
                        pass

                    verdict = adjudicate(
                        micro_data=debate_micro,
                        kronos_data=debate_kronos,
                        polymarket_data=debate_polymarket,
                        ta_signals=signals,
                        swarm_data=debate_swarm,
                        price=price,
                    )

                    debate_regime = _debate_to_regime(verdict.regime)

                    if debate_level >= DEBATE_MIN_LEVEL_FOR_ACTION and verdict.confidence >= 0.25:
                        regime = debate_regime
                        debate_used = True
                        log(f"  ⚖️  Debate verdict: {verdict.regime} (conf={verdict.confidence:.3f}, {verdict.resolution})")
                        log(f"  ⚖️  Bull {verdict.bull_strength:.3f} vs Bear {verdict.bear_strength:.3f} — {verdict.winning_theme}")
                        if verdict.contrarian_flag:
                            log(f"  ⚖️  ⚠️ Contrarian flag: unanimous agreement — reduced confidence")
                        if verdict.confidence < 0.35:
                            log(f"  ⚖️  Low debate confidence ({verdict.confidence:.3f}) — will fall through to micro if available")
                            debate_used = False  # Allow micro to override if debate isn't confident
                    elif debate_level >= 1:
                        # Shadow mode — log the verdict without acting on it
                        log(f"  ⚖️  [SHADOW] Debate verdict: {verdict.regime} (conf={verdict.confidence:.3f})")
                        log(f"  ⚖️  [SHADOW] Bull {verdict.bull_strength:.3f} vs Bear {verdict.bear_strength:.3f} — {verdict.winning_theme}")
                except Exception as e:
                    log(f"  ⚠️ Debate unavailable: {e} — falling through cascade")
        except ImportError:
            pass  # Debate module not installed — skip silently


    # === Microstructure-first regime detection (v12) ===
    # This is the forward-looking classifier. If enabled and confident,
    # it overrides BOTH single-model and swarm. Falls back to Kronos if
    # micro confidence is too low, then to Polymarket, then to swarm.
    # SKIPPED if debate already produced a confident verdict.
    micro_used = False
    if MICRO_REGIME and not debate_used:
        try:
            from quantforge_micro_regime import micro_detect_regime
            micro_regime, micro_confidence, micro_signals = micro_detect_regime()
            if micro_confidence >= MICRO_REGIME_MIN_CONFIDENCE:
                regime = micro_regime
                micro_used = True
                log(f"  🔬 Micro regime: {regime} (conf={micro_confidence:.3f}, score={micro_signals.regime_score:+.2f})")
                log(f"  🔬 CVD 4h: {micro_signals.cvd_4h:+,.0f}  pressure: {micro_signals.pressure_imbalance:+.2f}  depth: {micro_signals.depth_imbalance:+.3f}")
                if micro_signals.pressure_extreme:
                    log(f"  🔬 Pressure extreme ({micro_signals.pressure_extreme_direction:+d}) — potential reversal zone")
                # Log prediction for self-tuning
                try:
                    from quantforge_self_tune import log_prediction
                    log_prediction(
                        signals={
                            "cvd_1h": micro_signals.cvd_1h,
                            "cvd_4h": micro_signals.cvd_4h,
                            "cvd_accel": micro_signals.cvd_trend,
                            "pressure_imbalance": micro_signals.pressure_imbalance,
                            "pressure_imbalance_trend": micro_signals.pressure_imbalance_trend,
                            "depth_imbalance": micro_signals.depth_imbalance,
                            "depth_imbalance_trend": micro_signals.depth_imbalance_trend,
                            "micro_return_drift": micro_signals.micro_return_drift,
                            "vol_buy_ratio": micro_signals.vol_buy_ratio,
                            "spread_widening": 1.0 if micro_signals.spread_widening else 0.0,
                        },
                        regime=regime,
                        confidence=micro_confidence,
                        regime_score=micro_signals.regime_score,
                        btc_price=price,
                    )
                except Exception:
                    pass  # Non-critical — self-tuning will catch up later
            else:
                log(f"  🔬 Micro regime confidence too low ({micro_confidence:.3f} < {MICRO_REGIME_MIN_CONFIDENCE}) — falling back to Kronos/swarm/TA")
        except Exception as e:
            log(f"  ⚠️ Micro regime unavailable: {e} — falling back to Kronos/swarm/TA")

    # === Kronos foundation model regime (v12) ===
    # Forward-looking: generates 24h price forecast from 200 candles.
    # Used when micro classifier isn't confident enough AND debate didn't override.
    if not micro_used and not debate_used:
        try:
            from quantforge_kronos_regime import kronos_detect_regime, kronos_is_available
            if kronos_is_available():
                kronos_regime, kronos_conf, kronos_pct = kronos_detect_regime(candles)
                if kronos_conf > 0.3:  # Kronos has meaningful conviction
                    regime = kronos_regime
                    micro_used = True
                    log(f"  🧠 Kronos regime: {regime} (conf={kronos_conf:.3f}, forecast {kronos_pct:+.2f}% 24h)")
                else:
                    log(f"  🧠 Kronos forecast {kronos_pct:+.2f}% — too flat, falling back to swarm")
            else:
                log(f"  🧠 Kronos not available — falling back to swarm")
        except Exception as e:
            log(f"  ⚠️ Kronos unavailable: {e} — falling back to swarm")

    # === Polymarket prediction market signal (v12)
    # Real-money crowd sentiment — what does the market think will happen?
    # Used when both micro and Kronos aren't confident.
    if not micro_used and not debate_used:
        try:
            from quantforge_polymarket import polymarket_regime_signal
            pm_regime, pm_conf = polymarket_regime_signal()
            if pm_conf > 0.3 and pm_regime != "NEUTRAL":
                regime = pm_regime
                micro_used = True
                log(f"  💰 Polymarket regime: {regime} (conf={pm_conf:.3f})")
            elif pm_conf > 0.0:
                log(f"  💰 Polymarket: {pm_regime} (conf={pm_conf:.3f}) — low conviction, falling back")
        except Exception as e:
            pass  # Non-critical — skip if unavailable

    # === Swarm consensus regime override (v11) ===
    if SWARM_REGIME and not micro_used and not debate_used:
        try:
            from quantforge_swarm_regime import swarm_detect_regime, _regime_distance as _rd
            swarm_regime, swarm_confidence, swarm_votes = swarm_detect_regime(
                candles, signals, min_agreement=SWARM_MIN_AGREEMENT,
                min_voters=SWARM_MIN_VOTERS
            )
            if swarm_confidence >= SWARM_CONFIDENCE_THRESHOLD:
                regime = swarm_regime
                active_n = len([v for v in swarm_votes if v.confidence >= 0.30])
                log(f"  Swarm consensus: {regime} (conf={swarm_confidence:.3f}, {active_n}/{len(swarm_votes)} voters)")
                for v in swarm_votes:
                    if v.confidence >= 0.30:
                        log(f"    {v.voter:<12} -> {v.regime:<14} conf={v.confidence:.3f}")
                dissent = [v for v in swarm_votes if v.confidence >= 0.30 and _rd(v.regime, swarm_regime) > 1]
                if dissent:
                    log(f"    Dissent: {', '.join(f'{d.voter}={d.regime}' for d in dissent)}")
            else:
                log(f"  Swarm confidence too low ({swarm_confidence:.3f}) - using single-model {regime}")
        except Exception as e:
            log(f"  Swarm regime unavailable: {e} - using single-model {regime}")

    log(f"Regime: {regime}  (price ${signals['price']:,.0f}, MA20 ${signals['ma20']:,.0f}, MA50 ${signals['ma50']:,.0f}, RSI {signals['rsi14']:.1f}, 7d {signals['change_7d']*100:+.2f}%)")
    if HODL_MODE:
        log(f"  Mode: HODL_MODE — fixed {TARGET_ALLOC[regime]*100:.0f}% BTC across all regimes (regime is observed, not acted on)")

    # Persist regime for inspection
    try:
        with open(REGIME_FILE, "w") as f:
            json.dump({
                "ts": datetime.now(timezone.utc).isoformat(),
                "regime": regime,
                "signals": signals,
                "target_alloc": TARGET_ALLOC.get(regime, 0.45),
            }, f, indent=2)
    except Exception:
        pass

    port = load_portfolio()
    if port is None:
        log("First run — initializing portfolio")
        port = init_portfolio(price, regime)
        save_portfolio(port)
        equity = port["cash"] + port["btc_qty"] * price
        log(f"Initial position: {port['btc_qty']:.6f} BTC @ ${port['btc_avg_cost']:,.2f}, target alloc {TARGET_ALLOC[regime]*100:.0f}%")
        log(f"Equity: ${equity:,.2f}  (cash ${port['cash']:.2f}, btc ${port['btc_qty']*price:.2f})")
        return

    # === v29 one-time migration ===
    # prev_cycle_equity, peak_equity, and regime_perf were all recorded under
    # the pre-v29 formulas (margin excluded / leverage double-counted), so the
    # first v29 cycle would otherwise book the entire deployed margin as a
    # phantom regime_perf gain. Rebase to true equity and clear the corrupted
    # learning record. Self-applying so a cron tick can't race the deploy.
    if port.get("_equity_v", 0) < 29:
        _true_now = _true_equity(port, price)
        port["prev_cycle_equity"] = _true_now
        port["prev_cycle_price"] = price
        port["prev_cycle_ts"] = datetime.now(timezone.utc).isoformat()
        port["regime_perf"] = {}
        port["peak_equity"] = max(port.get("starting_balance", STARTING_BALANCE), _true_now)
        port["_equity_v"] = 29
        save_portfolio(port)
        log(f"  🔧 v29 migration: rebased attribution + peak to true equity ${_true_now:,.2f}; cleared corrupted regime_perf")

    # === Per-regime performance attribution (v3) ===
    # Compute equity & price deltas since last cycle, attribute to current
    # active_regime. This builds a learning record of WHICH regimes our
    # strategy actually adds value in (alpha = our_pnl - passive_hodl_pnl).
    now_iso_attr = datetime.now(timezone.utc).isoformat()
    prev_equity = port.get("prev_cycle_equity")
    prev_price = port.get("prev_cycle_price")
    prev_ts = port.get("prev_cycle_ts")
    # v29: was `cash + btc_qty * price`, which booked every margin transfer
    # (futures open, prehedge open) as a phantom loss in regime_perf — the
    # learning record the self-reflection daemon trains on. True equity is
    # margin-neutral: opening a position no longer "loses" the margin.
    cur_equity_attr = _true_equity(port, price)
    if prev_equity is not None and prev_price and prev_price > 0 and prev_ts:
        equity_delta = cur_equity_attr - prev_equity
        # What 100% BTC HODL would have made on prev_equity dollars
        hodl_delta = prev_equity * (price / prev_price - 1.0)
        alpha_delta = equity_delta - hodl_delta
        try:
            prev_dt = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
            hours_delta = (datetime.now(timezone.utc) - prev_dt).total_seconds() / 3600.0
        except Exception:
            hours_delta = 1.0
        # Attribute to the regime we were ACTING on, not the one just detected
        prev_active = port.get("active_regime", regime)
        rp = port.setdefault("regime_perf", {})
        bucket = rp.setdefault(prev_active, {
            "visits": 0, "hours": 0.0, "our_pnl": 0.0,
            "hodl_pnl": 0.0, "alpha": 0.0,
        })
        bucket["visits"] += 1
        bucket["hours"] += hours_delta
        bucket["our_pnl"] += equity_delta
        bucket["hodl_pnl"] += hodl_delta
        bucket["alpha"] += alpha_delta

    # Persist current cycle snapshot for next cycle's attribution
    port["prev_cycle_equity"] = cur_equity_attr
    port["prev_cycle_price"] = price
    port["prev_cycle_ts"] = now_iso_attr

    # === Regime hysteresis: only accept detected regime as "active" if it
    # persists for N consecutive cycles. This stops the agent from churning
    # on every minor regime flip when BTC oscillates in a tight band.
    history = list(port.get("regime_history", [port.get("current_regime", regime)]))
    history.append(regime)
    history = history[-(REGIME_HYSTERESIS_CYCLES * 2):]  # keep some context
    port["regime_history"] = history
    # Determine active_regime: only change if last N cycles all agree
    active_regime = port.get("active_regime", regime)
    recent = history[-REGIME_HYSTERESIS_CYCLES:]
    if len(recent) >= REGIME_HYSTERESIS_CYCLES and len(set(recent)) == 1:
        if recent[0] != active_regime:
            log(f"📊 Active regime change: {active_regime} → {recent[0]} (confirmed {REGIME_HYSTERESIS_CYCLES}× consecutive)")
            active_regime = recent[0]
            port["active_regime"] = active_regime
    else:
        log(f"  (detected {regime}, holding active {active_regime} — hysteresis: {recent})")

    # Compute current state using ACTIVE regime (smoothed), not raw detected
    # v29: route through _true_equity — the old inline copy double-counted
    # leverage (notional already = margin × leverage) and treated SHORT
    # positions as LONGs (no direction sign), so in BEAR regimes equity moved
    # the wrong way. Also adds prehedge margin + PnL, which every formula missed.
    equity = _true_equity(port, price)
    current_alloc = (port["btc_qty"] * price) / equity if equity > 0 else 0

    # Update peak equity (needed before drawdown calc)
    if equity > port.get("peak_equity", STARTING_BALANCE):
        port["peak_equity"] = equity
    drawdown = (port["peak_equity"] - equity) / port["peak_equity"] if port["peak_equity"] > 0 else 0
    pnl_pct_now = (equity - STARTING_BALANCE) / STARTING_BALANCE

    # === Regime-adaptive weight overrides (v10) ===
    # Auto-swap strategy weights based on active regime.
    # In STRONG_BEAR: futures short gets heavier, spot HODL lighter.
    # In STRONG_BULL: spot HODL heavier, futures long heavier.
    # In CHOP: MR dominates. BEAR: funding arb ramps up.
    weight_changes = _apply_regime_weights(active_regime)
    if weight_changes:
        log(f"  🔄 Regime-adaptive ({active_regime}): {weight_changes}")
        _rebuild_strategy_registry()

    # === Param memory: best-param recall on regime transitions (v22) ===
    # After the regime switch, check if we have historical evidence that
    # different parameters performed better in this regime. If so, load them
    # (backtest-gated via qf_validate_tune.py).
    global _LAST_PARAM_MEMORY_REGIME
    if _LAST_PARAM_MEMORY_REGIME != active_regime:
        _LAST_PARAM_MEMORY_REGIME = active_regime
        try:
            from quantforge_param_memory import apply_best_if_known
            if apply_best_if_known(active_regime):
                log(f"  🧠 Param memory: loaded best historical params for {active_regime}")
        except ImportError:
            pass  # param memory module not available — not a hard failure
        except Exception as e:
            log(f"  ⚠️ Param memory check failed: {e}")

    # === Strategy registry (v6 Stage 2) ===
    # Build a read-only CycleContext and ask each strategy what allocation
    # it wants for its slice. combine_decisions reduces these into a single
    # total_target_btc_value the rebalancer drives toward.
    ctx = CycleContext(
        price=price,
        regime=regime,
        active_regime=active_regime,
        signals=signals,
        total_equity=equity,
        cash=port["cash"],
        btc_qty=port["btc_qty"],
        drawdown_from_peak=drawdown,
        pnl_pct=pnl_pct_now,
        portfolio=port,
    )
    strategy_decisions = [(s, s.evaluate(ctx)) for s in STRATEGY_REGISTRY]
    target_btc_value_global, target_alloc, futures_dir, consensus = combine_decisions(strategy_decisions, equity)
    drift = current_alloc - target_alloc

    # === Learned rules from self-evolution engine (v15) ===
    # Apply rules the system learned from past bleeding events.
    # These override strategy decisions with learned corrective actions.
    try:
        from quantforge_evolve import apply_learned_rules
        evolve_mods = apply_learned_rules(
            port, active_regime, price, signals.get("MA20", price),
            drawdown, [s.name for s in STRATEGY_REGISTRY]
        )
        if evolve_mods.get("alloc_cap", 1.0) < target_alloc:
            old = target_alloc
            target_alloc = evolve_mods["alloc_cap"]
            target_btc_value_global = target_alloc * equity
            log(f"  🧬 Evolved rule: alloc cap {evolve_mods['alloc_cap']*100:.0f}% (from {old*100:.0f}%)")
    except ImportError:
        evolve_mods = {}

    # === Risk layer evaluation (v10) ===
    # Compute correlation, whale score, liquidation zones, position sizing
    try:
        from quantforge_risk import RiskContext
        ml_picks = list(port.get("alt_positions", {}).keys())
        risk = RiskContext().evaluate(
            picks=ml_picks, equity=equity, regime=active_regime, btc_price=price
        )
        log(f"  🛡️ Risk: whale={risk.whale_score:+.1f} corr_penalty={risk.correlation_penalty:.2f} "
            f"max_pos={risk.max_position_pct*100:.2f}% regime_mult={risk.regime_risk_mult:.1f}x")
        for w in risk.warnings[:3]:  # Top 3 warnings
            log(f"  ⚠️  {w}")
    except Exception as e:
        risk = None
        log(f"  ⚠️ Risk layer unavailable: {e}")

    # === Whale signal allocation modifier (v14) ===
    # Scale target allocation based on whale accumulation/distribution.
    # Accumulation (>0.3): add up to +15% allocation. Distribution (<-0.3): cut up to -15%.
    if risk is not None:
        whale = risk.whale_score
        if abs(whale) > 0.2:  # Only apply when signal is material
            whale_mod = 1.0 + whale * 0.15  # whale=-1.0 → 0.85x, whale=+1.0 → 1.15x
            old_target = target_alloc
            target_alloc = min(0.85, target_alloc * whale_mod)  # Never exceed 85% BTC
            target_alloc = max(0.25, target_alloc)  # Never below 25%
            target_btc_value_global = target_alloc * equity
            if abs(target_alloc - old_target) > 0.005:
                log(f"  🐋 Whale modifier: score={whale:+.2f} → alloc {old_target*100:.0f}% → {target_alloc*100:.0f}%")

    # === ML BTC directional signal (v31) ===
    # Run the XGBoost BTC direction predictor. Only act when confidence > 0.55
    # (thin edge — CV win rate 52.1% vs base rate 50.6%).
    ml_btc_pred = _run_ml_btc_predictor()
    if ml_btc_pred is not None:
        ml_conf = ml_btc_pred.get("confidence", 0.5)
        ml_dir = ml_btc_pred.get("direction", "up")
        ml_prob_up = ml_btc_pred.get("prob_up", 0.5)
        if ml_conf > 0.55:
            old_target = target_alloc
            old_fw = FUTURES_WEIGHT
            adjustment = ML_BTC_WEIGHT * equity  # e.g. 5% of equity
            if ml_dir == "up":
                # Buy bias: increase spot target
                target_btc_value_global += adjustment
                target_alloc = target_btc_value_global / equity
                target_alloc = min(0.85, target_alloc)
            else:
                # Sell bias / short conviction: reduce spot, boost futures short
                target_btc_value_global -= adjustment
                target_alloc = target_btc_value_global / equity
                target_alloc = max(0.25, target_alloc)
                FUTURES_WEIGHT = min(0.30, FUTURES_WEIGHT + 0.03)
            log(f"  🤖 ML BTC: {ml_dir} (conf={ml_conf:.0%}, prob_up={ml_prob_up:.0%}) → adjust ${adjustment:+,.0f}")
            if ml_dir == "down" and abs(FUTURES_WEIGHT - old_fw) > 0.001:
                log(f"  🤖 ML BTC short conviction: futures weight {old_fw:.0%} → {FUTURES_WEIGHT:.0%}")

    # Log strategy roster (one line for Stage 2's single strategy; will fan
    # out cleanly when Stage 3 adds more)
    for strategy, decision in strategy_decisions:
        log(f"  Strategy '{strategy.name}' (weight {strategy.weight*100:.0f}%): {decision.notes}")

    log(f"Equity ${equity:,.2f}  |  Cash ${port['cash']:.2f}  |  BTC ${port['btc_qty']*price:.2f} ({current_alloc*100:.1f}%)  |  target {target_alloc*100:.0f}%  |  drift {drift*100:+.1f}%  |  DD {drawdown*100:.2f}%")

    # === Protective DD velocity short (v15) ===
    # Regime-agnostic circuit breaker: when DD > 8% and price is clearly
    # trending below MA20, open a small protective short immediately —
    # don't wait for the cascade to reach consensus.
    if drawdown >= 0.08 and not port.get("futures_position", {}).get("direction"):
        ma20 = signals.get("MA20", price)
        if price < ma20:
            prot_margin = equity * 0.02  # 2% — small insurance bet
            if prot_margin <= port["cash"]:
                port["cash"] -= prot_margin
                port["futures_position"] = {
                    "direction": "SHORT",
                    "margin": prot_margin,
                    "notional": prot_margin,  # 1x — conservative
                    "entry_price": price,
                    "leverage": 1,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "protective": True,  # marked as protection, not directional bet
                }
                log(f"  🛡️ PROTECTIVE SHORT | DD {drawdown*100:.1f}% > 8%, price ${price:,.0f} < MA20 ${ma20:,.0f} — hedging {prot_margin:.0f}")

    # === Accelerated BEAR sell-down (v15) ===
    # When in BEAR and price < MA20, preemptively reduce spot allocation
    # to 35% instead of waiting for regime-adaptive to do it slowly.
    if active_regime == "BEAR" and price < signals.get("MA20", price) and target_alloc > 0.35:
        old_target = target_alloc
        target_alloc = max(0.35, target_alloc * 0.85)  # aggressive cut
        target_btc_value_global = target_alloc * equity
        if abs(target_alloc - old_target) > 0.01:
            log(f"  ⚡ BEAR sell-down: price ${price:,.0f} < MA20 ${signals.get('MA20', price):,.0f} → alloc {old_target*100:.0f}% → {target_alloc*100:.0f}%")

    # === Panic halt circuit breaker (checked BEFORE soft trim) ===
    # Two independent triggers — whichever fires first wins:
    #   1. Drawdown from peak >= PANIC_HALT_PCT
    #   2. Absolute PnL from starting balance <= -PANIC_HALT_ABS_PCT
    # The absolute trigger handles the case where the bot never had a peak
    # (e.g., started losing from cycle 1) — peak-based DD would never fire.
    #
    # v30: Auto-sync stale starting_balance. Topup events inflate the
    # portfolio's starting_balance (e.g. $5,000 → $7,848), causing false
    # absolute-loss panics. If inflated and we're profitable vs the code
    # constant, snap it back before computing abs_pnl_pct.
    _sb = port.get("starting_balance", STARTING_BALANCE)
    if _sb > STARTING_BALANCE * 1.2:
        _pnl_vs_code = (equity - STARTING_BALANCE) / STARTING_BALANCE
        if _pnl_vs_code > -0.05:
            port["starting_balance"] = STARTING_BALANCE
            log(f"  🔧 Auto-synced starting_balance ${_sb:,.0f}→${STARTING_BALANCE:,.0f} (P&L vs code {_pnl_vs_code:+.1%})")
    abs_pnl_pct = (equity - port.get("starting_balance", STARTING_BALANCE)) / port.get("starting_balance", STARTING_BALANCE)
    halt_reason = None
    if drawdown >= PANIC_HALT_PCT:
        halt_reason = f"drawdown_{drawdown*100:.1f}%_from_peak"
    elif abs_pnl_pct <= -PANIC_HALT_ABS_PCT:
        halt_reason = f"absolute_loss_{abs_pnl_pct*100:.1f}%_from_start"
    if halt_reason:
        trip_panic_halt(port, price, halt_reason, drawdown)
        return

    # === DD-velocity circuit breaker (fast leverage flatten, below panic) ===
    # Single-cycle equity drop >= DD_VELOCITY_TRIP_PCT -> flatten leveraged lanes +
    # start a leverage cooldown, keep spot. Faster/lighter than the 15%-from-peak
    # panic above; catches the fast bleeds the panic halt is too high to stop.
    if check_dd_velocity_breaker(port, price):
        save_portfolio(port)
        return

    # === Drawdown circuit breaker (soft, -8%: trims half) ===
    if drawdown >= DRAWDOWN_TRIM_PCT and port["btc_qty"] > 0:
        log(f"⚠️ Drawdown {drawdown*100:.2f}% >= {DRAWDOWN_TRIM_PCT*100:.0f}%. Trimming {DRAWDOWN_TRIM_FACTOR*100:.0f}% of BTC.")
        sell_qty = port["btc_qty"] * DRAWDOWN_TRIM_FACTOR
        if sell_btc(port, price, sell_qty, reason="drawdown_circuit_breaker"):
            port["n_drawdown_trims"] += 1
            # Reset peak so we don't keep trimming on continuing drawdown
            # (v29: true equity — naive reset set the peak ~margin too low,
            # delaying every later breaker)
            port["peak_equity"] = _true_equity(port, price)
            # Record trim time so the rebalancer suppresses buybacks for a
            # window — otherwise it would immediately re-buy what we just
            # defensively sold (the DD-trim/rebalancer conflict).
            port["last_drawdown_trim_ts"] = datetime.now(timezone.utc).isoformat()
            save_portfolio(port)
            return

    # === Trailing stop for spot BTC (v12) ===
    # Protects profits: if BTC rises above our cost basis and then pulls back
    # more than TRAIL_STOP_PCT from its peak, sell to lock in gains.
    # Only activates when position is in profit (price > avg cost + activation buffer).
    if port["btc_qty"] > 0:
        highest_since_entry = port.get("highest_price_since_entry", 0)
        if price > highest_since_entry:
            highest_since_entry = price
            port["highest_price_since_entry"] = highest_since_entry
        btc_cost = port.get("btc_avg_cost", 0)
        if btc_cost > 0 and price > btc_cost * (1 + TRAIL_STOP_ACTIVATE_PCT) and highest_since_entry > 0:
            # Trail is active — check if we've pulled back from peak
            pullback = (highest_since_entry - price) / highest_since_entry
            if pullback >= TRAIL_STOP_PCT:
                log(f"🎯 TRAIL STOP: -{pullback*100:.1f}% from peak ${highest_since_entry:,.0f} → closing position")
                if sell_btc(port, price, port["btc_qty"], reason="trailing_stop"):
                    port.pop("highest_price_since_entry", None)  # Reset trail after exit
                    port["n_trail_stops"] = port.get("n_trail_stops", 0) + 1
                    save_portfolio(port)
                    return

    # === Profit ladder ===
    pnl_pct = (equity - port.get("starting_balance", STARTING_BALANCE)) / port.get("starting_balance", STARTING_BALANCE)
    next_take_threshold = port.get("last_profit_take_pct", 0) + PROFIT_TAKE_INCREMENT
    if pnl_pct >= max(PROFIT_TAKE_PCT, next_take_threshold):
        sell_pct_of_btc = 0.05
        sell_qty = port["btc_qty"] * sell_pct_of_btc
        log(f"🎯 Profit milestone {pnl_pct*100:+.1f}% — taking {sell_pct_of_btc*100:.0f}% off the table")
        if sell_btc(port, price, sell_qty, reason="profit_take"):
            port["n_profit_takes"] += 1
            port["last_profit_take_pct"] = pnl_pct
            save_portfolio(port)
            return

    # === Regime rebalance with cooldown + daily cap ===
    if abs(drift) >= REBALANCE_THRESHOLD:
        # Enforce cooldown: at least N hours since last rebalance
        now_dt = datetime.now(timezone.utc)
        last_rebalance_iso = port.get("last_rebalance_ts")
        if last_rebalance_iso:
            try:
                last_dt = datetime.fromisoformat(last_rebalance_iso.replace("Z", "+00:00"))
                hours_since = (now_dt - last_dt).total_seconds() / 3600.0
            except Exception:
                hours_since = 999
        else:
            hours_since = 999
        
        # v28: Emergency bypass for post-halt cash-lock
        # When agent wakes with 0% BTC in a trending regime, bypass ALL gates
        current_btc_pct = port["btc_qty"] * price / max(equity, 1)
        target_btc_pct = TARGET_ALLOC.get(active_regime, 0.50)
        trending_regimes = ("BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR")
        post_halt_emergency = (
            current_btc_pct < 0.05 and active_regime in trending_regimes
            and abs(current_btc_pct - target_btc_pct) > 0.30
        )
        if post_halt_emergency:
            log(f"  🚨 EMERGENCY BYPASS: {current_btc_pct:.0%} BTC in {active_regime} "
                f"(target {target_btc_pct:.0%}) — skipping cooldown+cap")
            # Fall through — don't return, proceed directly to rebalance
        
        elif hours_since < REBALANCE_COOLDOWN_HOURS:
            log(f"  ⏸️ Rebalance blocked — only {hours_since:.1f}h since last (min {REBALANCE_COOLDOWN_HOURS}h)")
            port["current_regime"] = active_regime
            save_portfolio(port)
            return
        # Enforce daily cap: max N rebalances per 24h
        if not post_halt_emergency:
            rebal_log = port.get("rebalance_log", [])
            cutoff = (now_dt - timedelta(hours=24)).isoformat()
            recent_rebal = [r for r in rebal_log if r > cutoff]
            if len(recent_rebal) >= MAX_REBALANCES_PER_DAY:
                log(f"  ⛔ Rebalance blocked — {len(recent_rebal)} rebalances in last 24h (cap {MAX_REBALANCES_PER_DAY})")
                port["current_regime"] = active_regime
                save_portfolio(port)
                return
        # Cleared all guards — proceed. Use the combined target from the
        # strategy registry (computed above as target_btc_value_global).
        target_btc_value = target_btc_value_global
        
        # v29: Confidence-gated Kelly governor — prevents nuclear bets
        # Maximum tilt from baseline = half_kelly × debate_confidence
        try:
            from quantforge_gated_kelly import gated_position_size
            debate_conf = 0.30  # default
            try:
                from quantforge_debate_gate import get_debate_level
                from quantforge_debate import adjudicate as _debate_adjudicate
                if get_debate_level() >= 1:
                    verd = _debate_adjudicate(ta_signals=signals)
                    debate_conf = verd.confidence
            except Exception:
                pass
            
            sizing = gated_position_size(debate_conf, equity)
            tilt = sizing["bet_pct"]
            baseline_pct = TARGET_ALLOC.get(active_regime, 0.50)
            
            # In BEAR/STRONG_BEAR: tilt negative (reduce spot)
            if active_regime in ("BEAR", "STRONG_BEAR"):
                tilt = -tilt
            
            max_target = min((baseline_pct + tilt) * equity, equity * 0.85)
            min_target = max((baseline_pct + tilt) * equity, equity * 0.25)
            target_btc_value = max(min_target, min(max_target, target_btc_value))
            
            if abs(tilt) > 0.01:
                log(f"  🎯 Kelly governor: baseline {baseline_pct:.0%} ± {tilt:.1%} "
                    f"(half-Kelly {sizing['kelly']['half_kelly']:.1%} × conf {debate_conf:.2f}) "
                    f"→ target capped {target_btc_value/equity*100:.0f}%")
        except ImportError:
            pass
        current_btc_value = port["btc_qty"] * price
        delta_usd = target_btc_value - current_btc_value
        executed = False
        # Post-trim buyback suppression: if we recently fired a drawdown trim,
        # block rebalance BUYS (but allow sells) so we don't undo the defensive
        # de-risk by buying the falling knife.
        buyback_suppressed = False
        last_trim_iso = port.get("last_drawdown_trim_ts")
        if last_trim_iso:
            try:
                last_trim_dt = datetime.fromisoformat(last_trim_iso.replace("Z", "+00:00"))
                hours_since_trim = (now_dt - last_trim_dt).total_seconds() / 3600.0
                buyback_suppressed = hours_since_trim < DRAWDOWN_TRIM_BUYBACK_SUPPRESS_HOURS
            except Exception:
                buyback_suppressed = False
        # Emergency override: if drift is extreme (>20% of equity), allow
        # buyback after 6h — the trim already served its purpose.
        drift_pct = abs(delta_usd) / equity if equity > 0 else 0
        emergency_override = (
            buyback_suppressed and drift_pct > 0.20 and hours_since_trim >= 6
        )
        if emergency_override:
            log(f"  🚨 Emergency drift override: drift {drift_pct*100:.1f}% > 20%, "
                f"trim was {hours_since_trim:.1f}h ago — allowing buyback")
        if delta_usd > 0 and buyback_suppressed and not emergency_override:
            log(f"  🛡️ Buyback suppressed — drawdown trim {hours_since_trim:.1f}h ago "
                f"(< {DRAWDOWN_TRIM_BUYBACK_SUPPRESS_HOURS}h). Drift {drift_pct*100:.1f}%. "
                f"Holding defensive posture, no rebuy.")
            port["current_regime"] = active_regime
            save_portfolio(port)
            return
        if delta_usd > 0 and port["cash"] >= 10:
            log(f"📈 Increase BTC allocation by ${delta_usd:.2f} (regime {active_regime}, target {target_alloc*100:.0f}%)")
            buy_amt = min(delta_usd, port["cash"] * 0.95)
            if buy_btc(port, price, buy_amt, reason=f"rebalance_to_{active_regime}"):
                port["n_rebalances"] += 1
                executed = True
        elif delta_usd < 0 and port["btc_qty"] > 0:
            log(f"📉 Reduce BTC allocation by ${abs(delta_usd):.2f} (regime {active_regime}, target {target_alloc*100:.0f}%)")
            sell_qty = abs(delta_usd) / price
            if sell_btc(port, price, sell_qty, reason=f"rebalance_to_{active_regime}"):
                port["n_rebalances"] += 1
                executed = True
        if executed:
            now_iso = datetime.now(timezone.utc).isoformat()
            port["last_rebalance_ts"] = now_iso
            rebal_log = port.get("rebalance_log", [])
            rebal_log.append(now_iso)
            # Trim to last 50 entries (plenty for 24h window)
            port["rebalance_log"] = rebal_log[-50:]

    # === Pre-hedge evaluation (v23) ===
    # Runs BEFORE main futures lane. Checks microstructure signals + debate
    # verdict to open a 1% equity SHORT insurance bet when the tape is turning
    # bearish but the cascade hasn't confirmed BEAR yet.
    try:
        from quantforge_prehedge import run_prehedge_cycle
        prehedge_result = run_prehedge_cycle(port, price, equity)
        if prehedge_result["action"] in ("open", "close"):
            log(f"  🛡️ Pre-hedge: {prehedge_result['detail']}")
            save_portfolio(port)  # persist immediately so close/open is durable
        elif prehedge_result["action"] != "idle":
            log(f"  🛡️ Pre-hedge: {prehedge_result['detail']}")
    except ImportError:
        pass  # pre-hedge module not installed — skip silently
    except Exception as e:
        log(f"  ⚠️ Pre-hedge error: {e} — skipping, fall through to futures")

    # === Futures lane execution (v7) ===
    _execute_futures(port, price, futures_dir, active_regime, equity, signals, consensus)

    # === ML Scanner lane execution (v8) ===
    _execute_ml_positions(port, price, equity)

    # === Liquidation dip execution (v14) ===
    _execute_liquidation_dip(port, price, equity)

    # === Funding rate arbitrage execution (v9) ===
    _execute_funding_arb(port, equity, active_regime)

    port["current_regime"] = regime
    save_portfolio(port)
    # Include alt positions in final equity
    alt_value = sum(
        p.get("qty", 0) * price for p in port.get("alt_positions", {}).values()
    )
    final_equity = _true_equity(port, price)
    pnl_base = port.get("starting_balance", STARTING_BALANCE)
    pnl = final_equity - pnl_base
    log(f"=== Cycle end. Equity ${final_equity:,.2f}  PnL ${pnl:+.2f} ({pnl/pnl_base*100:+.2f}%)  Regime {regime}  Trades total {port['n_trades']} ===")

    # Per-component breakdown for charting
    spot_val = port.get("cash", 0) + port.get("btc_qty", 0) * price
    fpos = port.get("futures_position") or {}
    f_margin = fpos.get("margin", 0) or 0
    f_upnl = 0.0
    if fpos.get("direction") and fpos.get("entry_price", 0) > 0:
        entry = fpos["entry_price"]
        notional = fpos.get("notional", 0) or 0
        pct_change = (price - entry) / entry
        if fpos["direction"] == "SHORT":
            pct_change = -pct_change
        f_upnl = pct_change * notional
    ph = port.get("prehedge") or {}
    ph_val = 0.0
    if ph.get("open"):
        ph_margin = ph.get("margin", 0) or 0
        ph_upnl = 0.0
        if ph.get("entry_price", 0) > 0:
            entry_ph = ph["entry_price"]
            notional_ph = ph.get("notional", ph_margin) or 0
            pct_ph = (price - entry_ph) / entry_ph
            if ph.get("direction", "SHORT") == "SHORT":
                pct_ph = -pct_ph
            ph_upnl = pct_ph * notional_ph
        ph_val = ph_margin + ph_upnl
    futures_pnl = port.get('futures_pnl', 0) + f_upnl  # realized + unrealized PnL only, no margin
    log(f"  📊 Components: spot=${spot_val:,.2f} futures=${futures_pnl:+.2f} prehedge=${ph_val:+.2f} alts=${alt_value:,.2f} cash=${port.get('cash',0):,.2f} btc_qty={port.get('btc_qty',0):.6f} btc_price=${price:,.2f}")

    # === Self-detection: money-conservation invariants (fail-safe; never breaks the cycle) ===
    # The layer that was missing when the margin-orphan bug bled silently for weeks. Runs
    # every cycle against the just-saved portfolio + ledger: logs any violation, persists
    # state for the report/self-heal, and on a FRESH unexplained money drop THIS cycle it
    # auto-halts the futures lane (reversible futures_kill) so a live leak stops ITSELF
    # instead of bleeding. Detect -> de-risk -> escalate; the code fix stays human-gated.
    try:
        from quantforge_invariants import evaluate as _inv_evaluate
        _inv_vios, _ = _inv_evaluate(port, price)
        for _v in _inv_vios:
            log(f"  🔎 INVARIANT [{_v.severity}] {_v.name}: {_v.detail}")
        _critical_inv = [v for v in _inv_vios if v.severity == "critical"]
        if _critical_inv and not port.get("futures_kill"):
            _close_futures_position(port, price, "critical_invariant")
            port["futures_kill"] = True
            port["futures_kill_reason"] = "invariant:" + _critical_inv[0].name
            log("  🛡️ SELF-HEAL: critical invariant breach -> futures lane HALTED "
                f"(futures_kill, trigger={_critical_inv[0].name}). Reversible via panic-reset; escalated to report/self-heal.")
            save_portfolio(port)
    except Exception as _inv_e:
        log(f"  ⚠️ invariant self-check skipped: {str(_inv_e)[:120]}")

    # === Self-evolution: bleeding detection + fix generation (v15) ===
    # If this cycle triggered a bleed (rapid DD increase or PnL drop),
    # the evolution engine diagnoses the cause, generates a learned rule,
    # and applies it — all without human intervention.
    try:
        from quantforge_evolve import detect_bleeding, diagnose, generate_fix, add_rules
        bleed = detect_bleeding(
            equity=final_equity,
            peak_equity=port.get("peak_equity", final_equity),
            prev_equity=port.get("prev_cycle_equity", final_equity),
            prev_dd=0.0,
            regime=regime,
            price=price,
            ma20=signals.get("MA20", price),
            strategies=[s.name for s in STRATEGY_REGISTRY],
            futures_dir=port.get("futures_position", {}).get("direction", "FLAT") or "FLAT",
        )
        if bleed:
            bleed.diagnosis = diagnose(bleed)
            fixes = generate_fix(bleed, bleed.diagnosis)
            if fixes:
                add_rules(fixes)
                log(f"  🧬 Evolution: detected bleed ({bleed.trigger}), diagnosed: {bleed.diagnosis[:80]}, generated {len(fixes)} fix(es)")
    except ImportError:
        pass

    # === Strategy Engine: strategy factory + self-patch + infra (v16 → v23) ===
    # v23: Expanded trigger flags — degradation AND opportunity detection.
    # DEGRADATION triggers → force a tuner cycle + alert agent LLM analysis
    # OPPORTUNITY triggers → alert agent LLM analysis only (no auto force needed)
    try:
        from quantforge_agi import run_agi_cycle, get_agi_report
        
        # Track equity over cycles to detect stagnation
        if '_qf_perf_history' not in _PERF_HISTORY_CACHE:
            _PERF_HISTORY_CACHE['_qf_perf_history'] = []
        hist = _PERF_HISTORY_CACHE['_qf_perf_history']
        hist.append(final_equity)
        if len(hist) > 24:
            hist.pop(0)
        
        force_agi = False
        reasons = []
        alert_reasons = []  # triggers alert agent but not necessarily auto force

        # ── EXISTING DEGRADATION TRIGGERS ────────────────────────

        # Check 12h PnL (last 12 cycles, roughly)
        if len(hist) >= 12:
            pnl_12h = final_equity - hist[-12]
            if pnl_12h < 0:
                reasons.append(f"12h PnL ${pnl_12h:+.0f} (LOSING)")
                force_agi = True
            elif pnl_12h < (STARTING_BALANCE * 0.002):
                reasons.append(f"12h PnL ${pnl_12h:+.0f} (STAGNANT)")
                force_agi = True

        # Check DD from peak
        dd_ratio = 1.0 - final_equity / max(port.get("peak_equity", final_equity), 1)
        if dd_ratio > 0.10:
            reasons.append(f"DD {dd_ratio:.1%} (HIGH)")
            force_agi = True

        # Check cumulative alpha
        try:
            total_alpha = sum(v.get('alpha', 0) for v in port.get('regime_perf', {}).values())
            if total_alpha < 0:
                reasons.append(f"alpha ${total_alpha:+.0f} (NEGATIVE)")
                force_agi = True
        except Exception:
            pass

        # ── NEW DEGRADATION TRIGGERS ─────────────────────────────

        # Consecutive losing cycles (equity dropped vs previous cycle)
        prev_eq = port.get("prev_cycle_equity", final_equity)
        if final_equity < prev_eq:
            _PERF_HISTORY_CACHE['_qf_losing_streak'] = _PERF_HISTORY_CACHE.get('_qf_losing_streak', 0) + 1
        else:
            _PERF_HISTORY_CACHE['_qf_losing_streak'] = 0
        losing_streak = _PERF_HISTORY_CACHE.get('_qf_losing_streak', 0)
        if losing_streak >= 3:
            reasons.append(f"losing streak {losing_streak}")
            force_agi = True

        # Cash-lock: drift > 3× threshold AND cap hit
        # (agent can't rebalance even though it should)
        try:
            # Current drift = |current_btc_pct - target_btc_pct|
            current_btc_pct = port.get("btc_qty", 0) * price / max(final_equity, 1)
            target_btc_pct = TARGET_ALLOC.get(regime, 0.50)
            drift = abs(current_btc_pct - target_btc_pct)
            rebal_thresh = applied_params.get("rebalance_threshold", 0.05)
            max_reb = applied_params.get("max_rebalances_per_day", 2)
            # Count rebalances in last 24h
            from datetime import datetime as _dt4, timezone as _tz4
            cutoff = (_dt4.now(_tz4.utc).timestamp() - 86400)
            recent_rebs = 0
            for r in port.get("rebalance_log", []):
                try:
                    rt = _dt4.fromisoformat(r).timestamp()
                    if rt > cutoff:
                        recent_rebs += 1
                except Exception:
                    pass
            if drift > 3 * rebal_thresh and recent_rebs >= max_reb:
                reasons.append(f"cash-lock drift {drift:.1%} > {3*rebal_thresh:.1%}")
                force_agi = True
        except Exception:
            pass

        # Collector data staleness (>6h = flying blind)
        try:
            import os as _os2
            from datetime import datetime as _dt2, timezone as _tz2
            stale_count = 0
            check_files = [
                _os2.path.expanduser("~/quantforge/data/quantforge/derivatives/derivatives_state_latest.parquet"),
                _os2.path.expanduser("~/quantforge/data/quantforge/onchain/btc_onchain.json"),
                _os2.path.expanduser("~/quantforge/data/quantforge/sentiment/latest.json"),
            ]
            now_ts = _dt2.now(_tz2.utc).timestamp()
            for cf in check_files:
                if _os2.path.exists(cf):
                    age_h = (now_ts - _os2.path.getmtime(cf)) / 3600
                    if age_h > 6:
                        stale_count += 1
            if stale_count >= 2:
                reasons.append(f"collectors {stale_count}/3 stale >6h")
                force_agi = True
        except Exception:
            pass

        # BTC crossed below MA200 (bear trend structural break)
        if signals.get("ma200") and price < signals["ma200"]:
            prev_price = port.get("prev_cycle_price", price)
            prev_ma200 = signals.get("ma200")  # approximate — MA200 doesn't change fast
            if prev_price >= prev_ma200:
                reasons.append("BTC crossed below MA200")
                force_agi = True

        # ── OPPORTUNITY / INFO TRIGGERS (alert agent only) ────────────

        # Regime transition (different from last cycle)
        prev_regime = _PERF_HISTORY_CACHE.get('_qf_last_regime', '')
        if prev_regime and prev_regime != regime:
            alert_reasons.append(f"regime {prev_regime}→{regime}")
        _PERF_HISTORY_CACHE['_qf_last_regime'] = regime

        # Large BTC price move (>3% in 1h from prev cycle)
        prev_cycle_price = port.get("prev_cycle_price", price)
        if prev_cycle_price > 0:
            pct_move = abs(price - prev_cycle_price) / prev_cycle_price
            if pct_move > 0.03:
                direction = "UP" if price > prev_cycle_price else "DOWN"
                alert_reasons.append(f"BTC {direction} {pct_move:.1%} in 1h")

        # Volatility spike (ATR > 2.5× baseline)
        atr = signals.get("atr_pct", 0.02)
        if atr > 0.08:  # 2.5× ~0.035 baseline = 0.0875
            alert_reasons.append(f"vol spike ATR={atr:.3f}")

        # Fear & Greed extreme
        fng = signals.get("fear_greed", 50)
        if fng < 15:
            alert_reasons.append(f"EXTREME FEAR F&G={fng}")
        elif fng > 85:
            alert_reasons.append(f"EXTREME GREED F&G={fng}")

        # Profit take or drawdown trim fired this cycle
        # (detected by comparing n_profit_takes / n_drawdown_trims to cached counts)
        prev_profit_takes = _PERF_HISTORY_CACHE.get('_qf_n_profit_takes', port.get("n_profit_takes", 0))
        prev_dd_trims = _PERF_HISTORY_CACHE.get('_qf_n_drawdown_trims', port.get("n_drawdown_trims", 0))
        if port.get("n_profit_takes", 0) > prev_profit_takes:
            alert_reasons.append("profit take fired")
        if port.get("n_drawdown_trims", 0) > prev_dd_trims:
            alert_reasons.append("drawdown trim fired")
        _PERF_HISTORY_CACHE['_qf_n_profit_takes'] = port.get("n_profit_takes", 0)
        _PERF_HISTORY_CACHE['_qf_n_drawdown_trims'] = port.get("n_drawdown_trims", 0)

        # ML scanner went from having picks to zero (went blind)
        prev_picks = _PERF_HISTORY_CACHE.get('_qf_ml_pick_count', -1)
        curr_picks = len(port.get("alt_positions", {}))
        if prev_picks > 0 and curr_picks == 0:
            alert_reasons.append("ML picks 0 (was blind)")
        _PERF_HISTORY_CACHE['_qf_ml_pick_count'] = curr_picks

        # Rebalance after long quiet period (>12h since last)
        rebalance_log = port.get("rebalance_log", [])
        if rebalance_log:
            last_reb = rebalance_log[-1]
            try:
                from datetime import datetime as _dt3, timezone as _tz3
                last_reb_dt = _dt3.fromisoformat(last_reb)
                hours_since = (_dt3.now(_tz3.utc) - last_reb_dt).total_seconds() / 3600
                if hours_since > 12:
                    # Only flag if this cycle just rebalanced (last entry is from this cycle)
                    # We can't easily tell, so just flag if quiet period is extreme (>24h)
                    if hours_since > 24:
                        alert_reasons.append(f"no rebalance {hours_since:.0f}h")
            except Exception:
                pass

        # Capital efficiency check (v26) — flag idle cash
        try:
            from quantforge_capital_efficiency import check as _cap_check
            cap_result = _cap_check()
            if cap_result.get("flag") and cap_result.get("urgency") in ("medium", "high"):
                idle_pct = cap_result.get("idle_pct", 0)
                alert_reasons.append(f"cash idle {idle_pct:.0f}%")
        except Exception:
            pass

        # ── Kelly Criterion check (v27) ────────────────────────────
        # On regime transition or high drift, compute optimal position size
        try:
            regime_transitioned = bool(prev_regime and prev_regime != regime)
            rebal_thresh = applied_params.get("rebalance_threshold", 0.05)
            if regime_transitioned or drift > 2 * rebal_thresh:
                from quantforge_kelly import compute_kelly
                kelly = compute_kelly(regime)
                if kelly and kelly["n_trades"] >= 5:
                    alert_reasons.append(
                        f"Kelly f*={kelly['kelly_full']:.1%} WR={kelly['win_rate']:.0%} "
                        f"R={kelly['avg_win']/max(kelly['avg_loss'],1):.1f} "
                        f"→{kelly['verdict']}"
                    )
        except Exception:
            pass

        # ── Execution quality audit (v27) ──────────────────────────
        # On degradation, check if slippage is eating profits
        try:
            if force_agi or losing_streak >= 2:
                from quantforge_exec_tracker import get_slippage_stats, calibrate_model
                slip = get_slippage_stats(168)
                if slip["n_trades"] >= 5 and slip["avg_slippage_bps"] > 15:
                    reasons.append(f"slippage {slip['avg_slippage_bps']:.0f}bps avg (HIGH)")
                    force_agi = True
                    calibrate_model()
        except Exception:
            pass

        # ── Capital efficiency check (v26) — flag idle cash

        # Adaptive cooldown: compress during crisis, relax during calm
        _cd = 3.0  # default
        if dd_ratio > 0.10:
            _cd = 0.0   # EMERGENCY — no cooldown, fire every cycle
        elif signals.get("atr_pct", 0.02) > 0.08:
            _cd = 0.5   # crisis mode
        elif signals.get("atr_pct", 0.02) > 0.05:
            _cd = 1.0   # elevated

        # ── Strategy auto-retirement check (v24) ──────────────────
        # When DD > 10% or losing streak >= 3, audit strategies for
        # underperformers and auto-retire if criteria met.
        retire_names = []
        if dd_ratio > 0.10 or losing_streak >= 3:
            try:
                from quantforge_strategy_retire import audit as retire_audit
                retire_results = retire_audit()
                retiring = [r for r in retire_results if r.get("status") == "RETIRE"]
                if retiring:
                    retire_names = [r["name"] for r in retiring]
                    from quantforge_strategy_retire import trigger_alert as retire_alert
                    retire_alert(retire_names)
                    alert_reasons.append(f"strategy_retire:{','.join(retire_names)}")
            except ImportError:
                pass

        if force_agi:
            log(f"  ⚡ Perf watchdog: {'; '.join(reasons)} → forcing tuner cycle")
            if retire_names:
                log(f"  🚫 Strategy retire flagged: {', '.join(retire_names)}")
            run_agi_cycle(force=True)
            # Trigger alert agent with BOTH degradation + opportunity reasons
            _trigger_alert_monitor(reasons + alert_reasons, _cd)
        elif alert_reasons:
            log(f"  🔔 alert agent trigger: {'; '.join(alert_reasons)}")
            _trigger_alert_monitor(alert_reasons, _cd)
        else:
            run_agi_cycle()
    except ImportError:
        pass

    # === Self-healing check (v12) ===
    # Runs at end of every cycle. Detects degradation and auto-recovers.
    try:
        from quantforge_self_heal import check_health, apply_recovery, _save_health
        import json, os as _os

        # Load recent trades
        trades = []
        trades_file = _os.path.join(_os.path.expanduser("~/quantforge/data/quantforge"), "agent_trades.jsonl")
        if _os.path.exists(trades_file):
            with open(trades_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        health = check_health(
            portfolio=port,
            trades=trades[-30:],
            regime_history=port.get("regime_history", []),
            equity=final_equity,
            peak_equity=port.get("peak_equity", STARTING_BALANCE),
            current_regime=regime,
        )

        if health.status != "healthy":
            log(f"  🩺 Health: {health.status.upper()}")
            for alert in health.alerts:
                log(f"    {'🚨' if 'CRITICAL' in alert else '⚠️'} {alert}")

        if health.recovery_level > 0:
            log(f"  🩺 Auto-recovery L{health.recovery_level}: {health.recovery_reason}")
            params_file = _os.path.join(_os.path.expanduser("~/quantforge/data/quantforge"), "qf_strategy_params.json")
            apply_recovery(health.recovery_level, health.recovery_reason, port, params_file)
            if health.recovery_level >= 3:
                log(f"  🩺 CRITICAL: halt triggered by self-healing. Review required.")
                return  # Don't save portfolio — halt takes precedence

        # Save health state for next cycle
        _save_health({
            "status": health.status,
            "alerts": health.alerts,
            "metrics": health.metrics,
            "last_recovery_ts": health.metrics.get("checked_at"),
        })
    except Exception as e:
        pass  # Self-healing is non-critical — agent continues without it

    # === Debate gate evaluation (v13) ===
    # Check if debate should be promoted or demoted based on system health.
    # Runs at end of every cycle after self-healing has diagnosed the system.
    try:
        from quantforge_debate_gate import evaluate_promotion, get_debate_level

        # Determine which detectors are currently active/available
        detectors_active = {
            "micro": False,
            "kronos": False,
            "polymarket": False,
            "swarm": False,
        }
        try:
            from quantforge_micro_regime import micro_detect_regime
            micro_detect_regime()  # Quick test
            detectors_active["micro"] = True
        except Exception:
            pass
        try:
            from quantforge_kronos_regime import kronos_is_available
            detectors_active["kronos"] = kronos_is_available()
        except Exception:
            pass
        try:
            from quantforge_polymarket import polymarket_regime_signal
            detectors_active["polymarket"] = True
        except Exception:
            pass
        try:
            from quantforge_swarm_regime import swarm_detect_regime
            detectors_active["swarm"] = True
        except Exception:
            pass

        # Load health history for gate evaluation
        health_file = _os.path.join(_os.path.expanduser("~/quantforge/data/quantforge"), "health.json")
        health_history = []
        if _os.path.exists(health_file):
            try:
                with open(health_file) as f:
                    h = json.load(f)
                health_history = [h]
            except Exception:
                pass

        new_level, action, reason = evaluate_promotion(
            portfolio=port,
            trades=trades[-30:],
            health_history=health_history,
            detectors=detectors_active,
            cycle_count=port.get("n_trades", 0),
        )

        if action != "no_change":
            level_names = {0: "INACTIVE", 1: "SHADOW", 2: "ACTIVE", 3: "DOMINANT"}
            icons = {0: "🔴", 1: "🟡", 2: "🟢", 3: "🟣"}
            log(f"  Debate gate: {icons.get(new_level, '❓')} {action.upper()} → {level_names.get(new_level, '?')} (L{new_level})")
            log(f"    Reason: {reason}")
    except ImportError:
        pass  # Gate module not available — skip silently
    except Exception as e:
        pass  # Gate evaluation is non-critical


def cmd_status():
    # Load runtime params so TARGET_ALLOC + HODL_MODE reflect the live state
    # (Stage 1 introduced the params file; the daemon may have updated it.)
    load_runtime_params()
    port = load_portfolio()
    if port is None:
        print("No agent portfolio yet. Run 'run' to initialize.")
        return
    price = get_btc_price()
    # v29: status uses the same single-source equity as the trading loop —
    # the old copy excluded futures + prehedge margin, so dashboards showed
    # a fake drawdown the size of whatever margin happened to be deployed.
    equity = _true_equity(port, price)
    pnl = equity - port["starting_balance"]
    pct = pnl / port["starting_balance"] * 100 if port["starting_balance"] > 0 else 0
    drawdown = (port["peak_equity"] - equity) / port["peak_equity"] * 100 if port["peak_equity"] > 0 else 0
    current_alloc = port["btc_qty"] * price / equity if equity > 0 else 0
    # Compute the REGISTRY-COMBINED target (Stage 2+) rather than just HODL's
    # per-regime alloc — otherwise status misreports when MR is overweighting
    # or underweighting its slice.
    try:
        candles = get_btc_klines_1h(REGIME_LOOKBACK_HOURS)
        regime, signals = detect_regime(candles)
        active_regime = port.get("active_regime", regime)
        pnl_pct_calc = (equity - port.get("starting_balance", STARTING_BALANCE)) / port.get("starting_balance", STARTING_BALANCE)
        ctx = CycleContext(
            price=price, regime=regime, active_regime=active_regime, signals=signals,
            total_equity=equity, cash=port["cash"], btc_qty=port["btc_qty"],
            drawdown_from_peak=drawdown / 100, pnl_pct=pnl_pct_calc, portfolio=port,
        )
        decisions = [(s, s.evaluate(ctx)) for s in STRATEGY_REGISTRY]
        _, target, _, _ = combine_decisions(decisions, equity)
    except Exception:
        # Fallback to legacy behavior if we can't fetch market state
        target = TARGET_ALLOC.get(port.get("current_regime", "NEUTRAL"), 0.45)
    print("=" * 64)
    print(f"  QuantForge Agent — Adaptive BTC Allocator")
    print("=" * 64)
    if is_halted():
        try:
            with open(HALT_FILE) as f:
                halt_info = json.load(f)
            print(f"  🚨 STATUS:          HALTED ({halt_info.get('reason', '?')})")
            print(f"  Halted at:         {halt_info.get('halted_at', '?')}")
            print(f"  Equity at halt:    ${halt_info.get('equity', 0):,.2f}  (DD {halt_info.get('drawdown_pct', 0)}%)")
            print(f"  Resume with:       python3 quantforge_agent.py panic-reset")
            print("-" * 64)
        except Exception:
            print(f"  🚨 STATUS:          HALTED (marker unreadable)")
            print("-" * 64)
    print(f"  Mode:              {'HODL_MODE (regime observed, not acted on)' if HODL_MODE else 'REGIME_ACTIVE'}")
    print(f"  Regime:            {port.get('current_regime', '?')}")
    print(f"  Strategies:        " + ", ".join(f"{s.name}({s.weight*100:.0f}%)" for s in STRATEGY_REGISTRY))
    print(f"  Target allocation: {target*100:.0f}% BTC")
    print(f"  Current alloc:     {current_alloc*100:.1f}% BTC")
    print(f"  Drift from target: {(current_alloc - target)*100:+.1f}%")
    print()
    print(f"  Starting balance:  ${port['starting_balance']:,.2f}")
    print(f"  Current equity:    ${equity:,.2f}")
    print(f"  PnL:               ${pnl:+,.2f} ({pct:+.2f}%)")
    print(f"  Peak equity:       ${port['peak_equity']:,.2f}")
    print(f"  Drawdown from peak:{drawdown:.2f}%")
    print()
    print(f"  Cash:              ${port['cash']:,.2f}")
    print(f"  BTC held:          {port['btc_qty']:.6f} BTC (${port['btc_qty']*price:,.2f})")
    print(f"  BTC avg cost:      ${port['btc_avg_cost']:,.2f}")
    print(f"  BTC current:       ${price:,.2f}")
    print(f"  Unrealized PnL:    ${(price - port['btc_avg_cost']) * port['btc_qty']:+,.2f}")
    print()
    print(f"  Total trades:      {port['n_trades']}")
    print(f"  Rebalances:        {port['n_rebalances']}")
    print(f"  Drawdown trims:    {port['n_drawdown_trims']}")
    print(f"  Profit takes:      {port['n_profit_takes']}")
    print(f"  Total fees:        ${port['total_fees_paid']:.2f}")
    # Futures lane status (v7)
    fp = port.get("futures_position") or {}
    if fp.get("direction"):
        fpnl_unrealized = 0.0
        if fp.get("notional", 0) > 0:
            entry = fp.get("entry_price", price)
            if fp["direction"] == "LONG":
                fpnl_unrealized = fp["notional"] * (price / entry - 1.0)
            else:
                fpnl_unrealized = fp["notional"] * (1.0 - price / entry)
        actual_lev = fp.get("notional", 0) / fp.get("margin", 1) if fp.get("margin", 0) > 0 else FUTURES_LEVERAGE
        print(f"  Futures position:  {fp['direction']} | margin ${fp.get('margin', 0):,.2f} | notional ${fp.get('notional', 0):,.2f} ({actual_lev:.0f}x)")
        print(f"  Futures unreal PnL:${fpnl_unrealized:+,.2f}")
    if port.get("futures_pnl", 0) != 0:
        print(f"  Futures realized:  ${port['futures_pnl']:+,.2f}")
    # ML Scanner lane status (v8)
    alt_positions = port.get("alt_positions", {})
    if alt_positions:
        alt_total = sum(p.get("qty", 0) * price for p in alt_positions.values())
        print(f"  ML Scanner:        {len(alt_positions)} coins, ${alt_total:,.2f} value")
        for sym, pos in alt_positions.items():
            pos_val = pos.get("qty", 0) * price
            print(f"    {sym}: {pos['qty']:.4f} @ ~${price:,.2f} = ${pos_val:,.2f}")
    start_dt, start_source = infer_portfolio_start_anchor(port)
    raw_created = port.get("created_at", "?")
    if start_dt is not None and start_source and start_source != "created_at":
        print(f"  Portfolio start:   {start_dt.isoformat()} ({start_source})")
        print(f"  Created:           {raw_created} [raw]")
    else:
        print(f"  Created:           {raw_created}")
    print(f"  Updated:           {port.get('updated_at', '?')}")
    print("=" * 64)


def cmd_perf():
    """Per-regime performance attribution — our PnL vs passive HODL by regime."""
    port = load_portfolio()
    if port is None:
        print("No agent portfolio yet. Run 'run' to initialize.")
        return
    rp = port.get("regime_perf", {})
    if not rp:
        print("No regime performance data yet. Need at least 2 cycles to attribute PnL.")
        return
    print("=" * 78)
    print(f"  QuantForge Agent — Cycle-Local Performance Attribution")
    print("=" * 78)
    print(f"  {'Regime':<14} {'Visits':>7} {'Hours':>7} {'Our $':>12} {'HODL $':>12} {'Alpha $':>12}")
    print("  " + "-" * 74)
    total_our = total_hodl = total_alpha = 0.0
    # Sort by regime to keep stable order
    order = ["STRONG_BULL", "BULL", "NEUTRAL", "CHOP", "BEAR", "STRONG_BEAR"]
    keys = [k for k in order if k in rp] + [k for k in rp if k not in order]
    for regime in keys:
        b = rp[regime]
        our = b.get("our_pnl", 0.0)
        hodl = b.get("hodl_pnl", 0.0)
        alpha = b.get("alpha", 0.0)
        total_our += our
        total_hodl += hodl
        total_alpha += alpha
        marker = "✅" if alpha > 0 else ("⚠️" if alpha < -2 else "  ")
        print(f"  {regime:<14} {b['visits']:>7d} {b['hours']:>6.1f}h {our:>+12.2f} {hodl:>+12.2f} {alpha:>+12.2f} {marker}")
    print("  " + "-" * 74)
    print(f"  {'TOTAL':<14} {'':>7} {'':>7} {total_our:>+12.2f} {total_hodl:>+12.2f} {total_alpha:>+12.2f}")
    print()
    print(f"  Interpretation:")
    print(f"    This view is cycle-local alpha vs a dynamic BTC overlay, not the")
    print(f"    benchmark gate's reset-to-now whole-window return.")
    if total_alpha > 0:
        print(f"    Strategy adding +${total_alpha:.2f} of alpha vs passive BTC HODL — keep going.")
    elif total_alpha < -5:
        print(f"    Strategy losing ${-total_alpha:.2f} vs passive HODL — consider tuning or HODL-only.")
    else:
        print(f"    Roughly tied with passive HODL (alpha ${total_alpha:+.2f}). Need more data.")
    print()
    print(f"  Best regime for our strategy:  {max(rp.items(), key=lambda kv: kv[1].get('alpha', 0))[0]}")
    print(f"  Worst regime for our strategy: {min(rp.items(), key=lambda kv: kv[1].get('alpha', 0))[0]}")
    print("=" * 78)


def cmd_panic_reset():
    """Clear the panic halt marker and reset the bot's panic flags.
    BTC position stays whatever it is (probably $0 after liquidation) — a fresh
    run cycle will resize per regime."""
    if not is_halted():
        print("No panic halt is active. Nothing to reset.")
        port = load_portfolio()
        if port and port.get("panic_halted"):
            port["panic_halted"] = False
            save_portfolio(port)
            print("Cleared stale panic flag in portfolio.")
        return
    try:
        with open(HALT_FILE) as f:
            halt_info = json.load(f)
        print("=" * 64)
        print("  Panic Halt Status")
        print("=" * 64)
        print(f"  Halted at:    {halt_info.get('halted_at', '?')}")
        print(f"  Reason:       {halt_info.get('reason', '?')}")
        print(f"  Equity:       ${halt_info.get('equity', 0):,.2f}")
        print(f"  Peak equity:  ${halt_info.get('peak_equity', 0):,.2f}")
        print(f"  Drawdown:     {halt_info.get('drawdown_pct', 0)}%")
        print("=" * 64)
    except Exception as e:
        print(f"(could not read halt marker: {e})")
    clear_halt_marker()
    port = load_portfolio()
    if port:
        port["panic_halted"] = False
        port["last_panic_reset_at"] = datetime.now(timezone.utc).isoformat()
        # v29: actually reset the peak (the old line computed a placeholder and
        # never stored it, so a stale pre-halt peak could re-trip the breakers
        # on the very next cycle)
        try:
            port["peak_equity"] = _true_equity(port, get_btc_price())
        except Exception as e:
            print(f"⚠️ Could not refresh peak_equity ({e}) — reset it manually if breakers re-trip")
        save_portfolio(port)
    print("✅ Panic halt cleared. Next cron tick will resume normal trading.")


def cmd_topup():
    """Add capital to the portfolio and reset the panic halt.

    Usage: quantforge_agent.py topup <amount>
    Example: quantforge_agent.py topup 5000

    Adds the specified amount to cash, resets starting_balance to the new total,
    clears panic halt, and resets peak_equity. Closes any open futures position
    at current mark price before adding capital (to avoid stale PnL)."""
    if len(sys.argv) < 3:
        print("Usage: quantforge_agent.py topup <amount>")
        print("Example: quantforge_agent.py topup 5000")
        return
    try:
        topup_amount = float(sys.argv[2])
    except ValueError:
        print(f"Invalid amount: {sys.argv[2]}")
        return
    if topup_amount <= 0:
        print("Topup amount must be positive.")
        return

    price = get_btc_price()
    port = load_portfolio()
    if port is None:
        print("No portfolio found. Run 'run' first to initialize.")
        return

    print("=" * 64)
    print("  QuantForge Topup")
    print("=" * 64)
    old_cash = port["cash"]
    old_futures_pnl = 0.0

    # Close any open futures position first
    fp = port.get("futures_position") or {}
    if fp.get("direction") and fp.get("notional", 0) > 0:
        entry = fp.get("entry_price", price)
        if fp["direction"] == "LONG":
            pnl = fp["notional"] * (price / entry - 1.0)
        else:
            pnl = fp["notional"] * (1.0 - price / entry)
        port["cash"] += fp["margin"] + pnl
        port["futures_pnl"] = port.get("futures_pnl", 0.0) + pnl
        old_futures_pnl = pnl
        print(f"  Closed {fp['direction']} futures: PnL ${pnl:+.2f} | "
              f"entry ${entry:,.2f} → now ${price:,.2f}")
        port["futures_position"] = {"direction": None, "margin": 0,
                                    "notional": 0, "entry_price": 0, "opened_at": None}

    old_equity = _true_equity(port, price)
    # Add the topup to cash
    port["cash"] += topup_amount
    new_equity = _true_equity(port, price)

    # Reset starting balance to the new total
    port["starting_balance"] = new_equity
    # Reset peak_equity to current
    port["peak_equity"] = new_equity
    # Reset attribution state so future alpha is measured from the post-topup
    # capital base instead of carrying pre-topup deltas forward.
    port["regime_perf"] = {}
    port["prev_cycle_equity"] = new_equity
    port["prev_cycle_price"] = price
    port["prev_cycle_ts"] = datetime.now(timezone.utc).isoformat()
    # Clear panic halt
    port["panic_halted"] = False
    port["last_panic_reset_at"] = datetime.now(timezone.utc).isoformat()
    port["futures_kill"] = False  # clear futures kill switch too
    port["futures_pnl"] = 0.0
    # Reset regime history for clean hysteresis
    port["regime_history"] = []
    port["updated_at"] = datetime.now(timezone.utc).isoformat()

    save_portfolio(port)
    clear_halt_marker()

    print(f"  Old equity:    ${old_equity:,.2f}")
    print(f"  Topup:        +${topup_amount:,.2f}")
    print(f"  Futures PnL:   ${old_futures_pnl:+.2f}")
    print(f"  New equity:    ${new_equity:,.2f}")
    print(f"  New balance:   ${new_equity:,.2f}")
    print("=" * 64)
    print("✅ Topup complete. Panic halt cleared. Next cron tick resumes trading.")
    print(f"   Regime-adaptive weights will auto-swap for current regime.")


def cmd_strategies():
    """Inspect the strategy registry: list active strategies, their weights,
    and what each one is currently saying given live market state.

    This is the v6 Stage 2 introspection tool — useful both for humans
    auditing the system and for the reflect daemon to see what each strategy
    contributes when (Stage 3+) multiple strategies coexist.
    """
    load_runtime_params()  # so weights reflect current params if overridden
    print("=" * 64)
    print(f"  QuantForge Strategy Registry")
    print("=" * 64)
    weight_total = sum(s.weight for s in STRATEGY_REGISTRY)
    print(f"  Active strategies: {len(STRATEGY_REGISTRY)}  |  spot weight: {(sum(s.weight for s in STRATEGY_REGISTRY if s.name != 'futures_lane'))*100:.0f}%  |  futures margin: {FUTURES_WEIGHT*100:.0f}%")
    spot_weights = sum(s.weight for s in STRATEGY_REGISTRY if s.name != "futures_lane")
    if abs(spot_weights - 1.0) > 1e-6:
        print(f"  ⚠️  WARNING: spot weights sum to {spot_weights*100:.0f}% (should be 100%)")
    print()
    # Live evaluation — pull market state and ask each strategy
    try:
        price = get_btc_price()
        candles = get_btc_klines_1h(REGIME_LOOKBACK_HOURS)
        regime, signals = detect_regime(candles)
        port = load_portfolio() or {
            "cash": STARTING_BALANCE, "btc_qty": 0.0,
            "starting_balance": STARTING_BALANCE, "peak_equity": STARTING_BALANCE,
        }
        equity = _true_equity(port, price)
        drawdown = (port["peak_equity"] - equity) / port["peak_equity"] if port.get("peak_equity", 0) > 0 else 0
        pnl_pct = (equity - port["starting_balance"]) / port["starting_balance"]
        active_regime = port.get("active_regime", regime)
        ctx = CycleContext(
            price=price, regime=regime, active_regime=active_regime, signals=signals,
            total_equity=equity, cash=port["cash"], btc_qty=port["btc_qty"],
            drawdown_from_peak=drawdown, pnl_pct=pnl_pct, portfolio=port,
        )
        print(f"  Live context: price ${price:,.2f} | regime {regime} | active {active_regime} | equity ${equity:,.2f}")
        print("-" * 64)
        total_btc_target = 0.0
        for s in STRATEGY_REGISTRY:
            d = s.evaluate(ctx)
            slice_capital = equity * s.weight
            slice_btc_target = slice_capital * d.target_alloc_pct
            total_btc_target += slice_btc_target
            print(f"  • {s.name:<12} weight={s.weight*100:>4.0f}%  "
                  f"target_alloc={d.target_alloc_pct*100:>5.1f}%  "
                  f"slice=${slice_capital:>9.2f}  "
                  f"→ BTC target ${slice_btc_target:>9.2f}")
            if d.notes:
                print(f"      notes: {d.notes}")
        effective = total_btc_target / equity if equity > 0 else 0
        print("-" * 64)
        print(f"  COMBINED:  target BTC value ${total_btc_target:,.2f}  ({effective*100:.1f}% of equity)")
    except Exception as e:
        print(f"  (could not pull live state: {e})")
    print("=" * 64)


def _trigger_alert_monitor(reasons, cooldown_h=3.0):
    """Write a trigger file so an external alert agent can run deep LLM analysis.

    An out-of-process watcher (e.g. a cron job) polls this file; when it finds a
    fresh timestamp it fires a deeper analysis pass — event-driven, not clock-driven.
    The path is configurable via QF_ALERT_TRIGGER_FILE.

    Cooldown is ADAPTIVE:
      - DD > 10%       → NO cooldown (emergency — fire every cycle)
      - ATR > 0.08     → 0.5h cooldown (crisis mode)
      - ATR > 0.05     → 1.0h cooldown (elevated)
      - Normal         → 3.0h cooldown (don't spam)
    """
    import json as _json, os as _os
    trigger_file = _os.path.expanduser(
        _os.environ.get("QF_ALERT_TRIGGER_FILE", "~/.quantforge/alert_trigger.json")
    )
    
    # Cooldown check — don't spam triggers
    if _os.path.exists(trigger_file):
        try:
            with open(trigger_file) as f:
                prev = _json.load(f)
            prev_ts = prev.get("ts", "")
            if prev_ts:
                from datetime import datetime as _dt, timezone as _tz
                prev_dt = _dt.fromisoformat(prev_ts)
                age_h = (_dt.now(_tz.utc) - prev_dt).total_seconds() / 3600
                if age_h < cooldown_h:
                    return  # still within cooldown
        except Exception:
            pass
    
    _os.makedirs(_os.path.dirname(trigger_file), exist_ok=True)
    from datetime import datetime as _dt, timezone as _tz
    payload = {
        "ts": _dt.now(_tz.utc).isoformat(),
        "reasons": reasons,
        "consumed": False,
        "cooldown_h": cooldown_h,
    }
    with open(trigger_file, "w") as f:
        _json.dump(payload, f)
    log(f"  📡 alert agent monitor trigger written → LLM analysis queued (cooldown={cooldown_h}h)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run_cycle()
    elif cmd == "status":
        cmd_status()
    elif cmd == "perf":
        cmd_perf()
    elif cmd == "strategies":
        cmd_strategies()
    elif cmd == "panic-reset":
        cmd_panic_reset()
    elif cmd == "topup":
        cmd_topup()
    elif cmd == "regime":
        # Quick regime check without trading
        candles = get_btc_klines_1h(REGIME_LOOKBACK_HOURS)
        regime, signals = detect_regime(candles)
        print(f"Regime: {regime}")
        for k, v in signals.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"Target alloc: {TARGET_ALLOC.get(regime, 0.45)*100:.0f}% BTC")
    else:
        print(f"Usage: {sys.argv[0]} [run|status|perf|strategies|regime|panic-reset|topup]")
        sys.exit(1)
