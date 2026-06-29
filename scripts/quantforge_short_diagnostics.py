#!/usr/bin/env python3
"""QuantForge short-side diagnostics.

Produces a lightweight artifact explaining whether the short model is inactive
because of model readiness, thresholds, or simply low bearish confidence in the
current liquid universe.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_ml import generate_signal
from quantforge_paper import SHORT_LIVE_THRESHOLD_CAP, _get_regime, get_klines, screen_coins

BASE_DIR = os.path.join(cfg.data, "quantforge")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_FILE = os.path.join(BASE_DIR, "short-diagnostics.json")


def read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    cfg.require_production_runtime("quantforge_short_diagnostics.py")
    short_meta = read_json(os.path.join(MODEL_DIR, "model_meta_short.json"))
    regime = _get_regime()
    universe = screen_coins(20)
    rows = []

    short_threshold = min(
        float(short_meta.get("optimal_threshold", 0.60) or 0.60) + float(regime.get("short_adj", 0.0) or 0.0),
        SHORT_LIVE_THRESHOLD_CAP,
    )

    for coin in universe[:12]:
        try:
            candles = get_klines(coin["symbol"], "1hour", 300)
            import pandas as pd

            df = pd.DataFrame(
                candles,
                columns=["ts", "open", "close", "high", "low", "volume", "turnover"],
            )
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
            result = generate_signal(
                coin["symbol"],
                candles_df=df,
                short_threshold_override=short_threshold,
            )
        except Exception as exc:
            rows.append({
                "symbol": coin["symbol"],
                "status": "error",
                "reason": str(exc),
            })
            continue

        reasons = result.get("reason") or []
        row = {
            "symbol": coin["symbol"],
            "signal": result.get("signal"),
            "confidence": float(result.get("short_confidence", 0.0) or 0.0),
            "long_confidence": float(result.get("long_confidence", 0.0) or 0.0),
            "threshold": short_threshold,
            "setup_tag": result.get("setup_tag"),
            "setup_score": result.get("setup_score"),
            "reason": reasons[0] if reasons else "n/a",
        }
        if result.get("signal") == "SELL":
            row["status"] = "short_signal"
        else:
            row["status"] = "below_threshold"
        rows.append(row)

    ranked = sorted(rows, key=lambda r: float(r.get("confidence", 0.0) or 0.0), reverse=True)
    top_short_confidence = max([float(r.get("confidence", 0.0) or 0.0) for r in ranked] or [0.0])
    paper_probe_threshold = max(0.35, min(0.45, short_threshold - 0.10))
    near_miss_count = sum(1 for r in ranked if float(r.get("confidence", 0.0) or 0.0) >= max(0.0, short_threshold - 0.10))
    if any(r.get("signal") == "SELL" for r in ranked):
        readiness = "live_ready_candidates_present"
        recommended_action = "keep_live_short_gate_and monitor active candidates"
    elif top_short_confidence >= paper_probe_threshold:
        readiness = "paper_probe_watch"
        recommended_action = "consider bounded paper-only short probe before lowering any live gate"
    else:
        readiness = "not_ready"
        recommended_action = "do not lower the live short gate; improve the short model or wait for a different regime"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "short_model": {
            "gate_pass": bool(short_meta.get("gate_pass", False)),
            "overall_auc": short_meta.get("overall_auc"),
            "holdout_auc": short_meta.get("holdout_auc"),
            "holdout_trades": short_meta.get("holdout_trades"),
            "optimal_threshold": short_meta.get("optimal_threshold"),
            "applied_live_threshold": short_threshold,
        },
        "summary": {
            "universe_size": len(universe[:12]),
            "short_signal_count": sum(1 for r in ranked if r.get("signal") == "SELL"),
            "top_short_confidence": top_short_confidence,
            "diagnosis": (
                "no_live_short_edge_detected"
                if not any(r.get("signal") == "SELL" for r in ranked)
                else "live_short_candidates_present"
            ),
            "readiness": readiness,
            "near_miss_count": near_miss_count,
            "paper_probe_threshold": paper_probe_threshold,
            "recommended_action": recommended_action,
        },
        "rows": ranked[:12],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("QuantForge short diagnostics")
    print(f"Top short confidence: {payload['summary']['top_short_confidence']:.3f}")
    print(f"Short signals:        {payload['summary']['short_signal_count']}")
    print(f"Saved:                {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
