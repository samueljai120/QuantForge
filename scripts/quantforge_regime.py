#!/usr/bin/env python3
"""QuantForge shared macro regime detector.

Builds a lightweight market regime snapshot from public BTC-USDT hourly candles.
The output stays compatible with the existing QuantForge paper engine while also
adding entropy-aware metadata that downstream governance and monitoring can use.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

import requests

from config import cfg

DATA_DIR = os.path.join(cfg.data, "quantforge")
CACHE_FILE = os.path.join(DATA_DIR, "regime-cache.json")
KUCOIN_BASE = "https://api.kucoin.com"
REGIME_SYMBOL = "BTC-USDT"
REGIME_INTERVAL = "1hour"
CACHE_TTL_HOURS = 4
LOOKBACK_BARS = 240
ENTROPY_WINDOW = 72
VOL_WINDOW = 24
TREND_WINDOW = 48


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _read_cache() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(payload: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def _cached_fresh(payload: dict) -> bool:
    ts = _parse_iso(payload.get("computed_at"))
    if not ts:
        return False
    return datetime.now(timezone.utc) - ts < timedelta(hours=CACHE_TTL_HOURS)


def _default_regime(reason: str = "fallback") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "computed_at": now,
        "source": "fallback",
        "source_symbol": REGIME_SYMBOL,
        "lookback_bars": LOOKBACK_BARS,
        "score": 0.0,
        "label": "NEUTRAL",
        "long_adj": 0.04,
        "short_adj": 0.04,
        "size_mult": 0.80,
        "entropy": 0.70,
        "entropy_label": "MIXED",
        "entropy_penalty": 0.02,
        "trend_strength": 0.0,
        "volatility": 0.0,
        "signals": {},
        "notes": [f"Regime fallback engaged: {reason}"],
    }


def _fetch_hourly_closes(symbol: str = REGIME_SYMBOL, interval: str = REGIME_INTERVAL, limit: int = LOOKBACK_BARS) -> list[float]:
    response = requests.get(
        f"{KUCOIN_BASE}/api/v1/market/candles",
        params={"type": interval, "symbol": symbol},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "200000":
        raise RuntimeError(f"KuCoin candle fetch failed: {payload.get('msg') or payload.get('code')}")
    rows = payload.get("data") or []
    closes = []
    for row in reversed(rows[:limit]):
        try:
            closes.append(float(row[2]))
        except Exception:
            continue
    if len(closes) < max(VOL_WINDOW + 2, ENTROPY_WINDOW // 2):
        raise RuntimeError(f"Not enough candle data for regime detection ({len(closes)} bars)")
    return closes


def _log_returns(closes: list[float]) -> list[float]:
    rets = []
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        cur = closes[idx]
        if prev > 0 and cur > 0:
            rets.append(math.log(cur / prev))
    return rets


def _normalized_entropy(values: list[float], bins: int = 7) -> float:
    if len(values) < 8:
        return 0.0
    sigma = math.sqrt(sum(v * v for v in values) / max(len(values), 1))
    if sigma <= 1e-9:
        return 0.0
    low = -3.0 * sigma
    high = 3.0 * sigma
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        clipped = max(low, min(high, value))
        idx = min(bins - 1, max(0, int((clipped - low) / width)))
        counts[idx] += 1
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(bins)


def _classify_entropy(entropy_score: float) -> tuple[str, float]:
    if entropy_score >= 0.82:
        return "CHAOTIC", 0.08
    if entropy_score >= 0.68:
        return "MIXED", 0.04
    return "ORDERLY", 0.0


def _trend_strength(closes: list[float], returns: list[float]) -> float:
    recent_rets = returns[-TREND_WINDOW:] if len(returns) >= TREND_WINDOW else returns
    if len(recent_rets) < 8:
        return 0.0
    avg = sum(recent_rets) / len(recent_rets)
    vol = math.sqrt(sum((r - avg) ** 2 for r in recent_rets) / len(recent_rets))
    if vol <= 1e-9:
        vol = 1e-9
    momentum = (closes[-1] / closes[-(min(TREND_WINDOW, len(closes) - 1) + 1)] - 1.0) if len(closes) > TREND_WINDOW else (closes[-1] / closes[0] - 1.0)
    score = (avg / vol) * 0.6 + momentum * 6.0
    return max(-1.0, min(1.0, score))


def _volatility_regime(returns: list[float]) -> float:
    recent = returns[-VOL_WINDOW:] if len(returns) >= VOL_WINDOW else returns
    if len(recent) < 8:
        return 0.0
    mean = sum(recent) / len(recent)
    return math.sqrt(sum((r - mean) ** 2 for r in recent) / len(recent)) * math.sqrt(len(recent))


def build_regime() -> dict:
    closes = _fetch_hourly_closes()
    returns = _log_returns(closes)
    recent_entropy = _normalized_entropy(returns[-ENTROPY_WINDOW:] if len(returns) >= ENTROPY_WINDOW else returns)
    entropy_label, entropy_penalty = _classify_entropy(recent_entropy)
    trend_score = _trend_strength(closes, returns)
    realized_vol = _volatility_regime(returns)

    label = "NEUTRAL"
    long_adj = 0.04
    short_adj = 0.04
    size_mult = 0.85
    notes = []

    if trend_score >= 0.22:
        label = "BULL"
        long_adj = -0.01
        short_adj = 0.06
        size_mult = 1.0
        notes.append("Positive trend strength supports longs more than shorts.")
    elif trend_score <= -0.22:
        label = "BEAR"
        long_adj = 0.08
        short_adj = -0.02
        size_mult = 0.72
        notes.append("Negative trend strength makes long entries more selective.")
    else:
        notes.append("Trend strength is mixed, so thresholds stay defensive.")

    if entropy_label == "CHAOTIC":
        long_adj += 0.08
        short_adj += 0.06
        size_mult *= 0.70
        notes.append("High entropy indicates chaotic price action; shrink size and raise thresholds.")
    elif entropy_label == "MIXED":
        long_adj += 0.03
        short_adj += 0.03
        size_mult *= 0.88
        notes.append("Mixed entropy keeps entries selective.")
    else:
        notes.append("Orderly entropy supports steadier execution.")

    if realized_vol >= 0.045:
        long_adj += 0.03
        short_adj += 0.03
        size_mult *= 0.85
        notes.append("Elevated realized volatility keeps exposure smaller.")

    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source": "kucoin_btc_entropy",
        "source_symbol": REGIME_SYMBOL,
        "lookback_bars": LOOKBACK_BARS,
        "score": round(trend_score, 4),
        "label": label,
        "long_adj": round(min(0.20, max(-0.05, long_adj)), 4),
        "short_adj": round(min(0.20, max(-0.10, short_adj)), 4),
        "size_mult": round(min(1.05, max(0.40, size_mult)), 4),
        "entropy": round(recent_entropy, 4),
        "entropy_label": entropy_label,
        "entropy_penalty": round(entropy_penalty, 4),
        "trend_strength": round(trend_score, 4),
        "volatility": round(realized_vol, 4),
        "signals": {
            "entropy_window": ENTROPY_WINDOW,
            "trend_window": TREND_WINDOW,
            "vol_window": VOL_WINDOW,
        },
        "notes": notes,
    }
    return payload


def get_regime(force_refresh: bool = False) -> dict:
    if not force_refresh:
        cached = _read_cache()
        if cached and _cached_fresh(cached):
            return cached
    try:
        regime = build_regime()
        _write_cache(regime)
        return regime
    except Exception as exc:
        cached = _read_cache()
        if cached:
            cached = dict(cached)
            cached.setdefault("notes", []).append(f"Using cached regime after refresh failure: {exc}")
            return cached
        return _default_regime(str(exc))


if __name__ == "__main__":
    print(json.dumps(get_regime(force_refresh=True), indent=2))
