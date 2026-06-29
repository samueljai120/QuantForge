#!/usr/bin/env python3
"""QuantForge — ML Model Training + Signal Generation

Trains XGBoost + LightGBM ensemble on historical features.
Uses walk-forward validation to prevent lookahead bias.

Pipeline:
  1. Load features from quantforge_features.py output
  2. Walk-forward cross-validation (5 folds, chronological)
  3. Find optimal confidence threshold (maximize expected value after fees)
  4. Save model + threshold to data/quantforge/model/
  5. Expose generate_signal(candles_df) for live use

Usage:
    python3 quantforge_ml.py train          # Train on all pairs
    python3 quantforge_ml.py train BTC-USDT # Train on single pair
    python3 quantforge_ml.py eval           # Evaluate saved model
    python3 quantforge_ml.py signal BTC-USDT # Generate live signal
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_target_profiles import (
    apply_research_hold_target_profile,
    apply_setup_quality_target_profile,
)

import xgboost as xgb
import lightgbm as lgb
try:
    import mlflow
except Exception:
    mlflow = None

FEATURES_DIR = os.path.join(cfg.data, "quantforge", "features")
MODEL_DIR = os.path.join(cfg.data, "quantforge", "model")
HISTORICAL_DIR = os.path.join(cfg.data, "quantforge", "historical")
OPTIMIZATION_DIR = os.path.join(cfg.data, "quantforge", "optimization")
BEST_PARAMS_FILE = os.path.join(OPTIMIZATION_DIR, "best-params.json")
MLFLOW_DIR = os.path.join(OPTIMIZATION_DIR, "mlruns")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "target_4h"   # Predict: will price be higher in 4 hours?
FWD_RET_COL = "fwd_ret_4h"
TARGET_SHORT_COL = "target_4h_short"  # Predict: will price drop >= 2.5% in 4 hours?

MAKER_FEE = 0.001
TAKER_FEE = 0.001
ROUND_TRIP_COST = MAKER_FEE + TAKER_FEE  # 0.2%
SHORT_MIN_LIVE_AUC = 0.55
SHORT_PAPER_MIN_HOLDOUT_TRADES = 25

THRESHOLD_MIN = 0.50
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.02

EXCLUDE_COLS = {
    "ts", "symbol", "open", "high", "low", "close", "volume", "turnover",
    "target_1h", "target_2h", "target_4h", "target_8h",
    "target_1h_short", "target_2h_short", "target_4h_short", "target_8h_short",
    "fwd_ret_1h", "fwd_ret_2h", "fwd_ret_4h", "fwd_ret_8h",
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

DEFAULT_LGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

FINAL_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

FINAL_LGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

CANDIDATE_RECOVERY_FILE = os.path.join(cfg.data, "quantforge", "candidate-recovery.json")
EXPERIMENT_LANES_FILE = os.path.join(cfg.data, "quantforge", "experiment-lanes.json")
ACTIVE_TRIAL_STATUSES = {"queued", "active"}
RESEARCH_SURFACE_CANDIDATE_TYPES = {
    "quantforge_research_hold",
    "competitiveness_gap_rebuild",
}

LONG_SETUP_SCORE_COLS = (
    "setup_trend_long_score",
    "setup_breakout_long_score",
    "setup_rebound_long_score",
)
SHORT_SETUP_SCORE_COLS = (
    "setup_trend_short_score",
    "setup_exhaustion_short_score",
)
LONG_SPECIALIST_TARGETS = (
    ("trend_long", "target_4h_research_hold_trend_long"),
    ("breakout_long", "target_4h_research_hold_breakout_long"),
)
LONG_SPECIALIST_SETUP_NAMES = {name for name, _ in LONG_SPECIALIST_TARGETS}
LONG_SPECIALIST_MIN_READY_AUC = 0.60
LONG_SPECIALIST_MIN_READY_POSITIVE_ROWS = 1000
LONG_SPECIALIST_MIN_READY_HOLDOUT_TRADES = 25


def _read_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _research_surface_active(candidate_type: str, trial_type: str, trial_status: str) -> bool:
    return candidate_type in RESEARCH_SURFACE_CANDIDATE_TYPES or (
        trial_type in RESEARCH_SURFACE_CANDIDATE_TYPES
        and trial_status in ACTIVE_TRIAL_STATUSES
    )


def _surface_mode_from_context(redesign_context: dict | None) -> str:
    redesign_context = redesign_context or {}
    if redesign_context.get("research_hold_active"):
        return "research_hold_setup_composite"
    if redesign_context.get("setup_quality_active"):
        return "setup_quality_labeled_directional"
    if redesign_context.get("redesign_active"):
        return "redesign_weighted_directional"
    return "standard_directional"


def _surface_mode_from_training_profile(training_profile: dict | None) -> str:
    training_profile = training_profile or {}
    surface_mode = str(training_profile.get("surface_mode", "") or "").strip()
    if surface_mode:
        return surface_mode
    if training_profile.get("research_hold_active"):
        return "research_hold_setup_composite"
    if training_profile.get("setup_quality_active"):
        return "setup_quality_labeled_directional"
    if training_profile.get("redesign_active"):
        return "redesign_weighted_directional"
    return "standard_directional"


def _signal_gate_status(meta: dict | None, *, allow_gate_bypass: bool = False) -> tuple[bool, bool]:
    meta = meta or {}
    gate_pass = bool(meta.get("gate_pass", False))
    gate_bypassed = bool(allow_gate_bypass and not gate_pass)
    return gate_pass, gate_bypassed


def _short_model_ready_for_signal(
    meta_short: dict | None,
    *,
    has_model: bool,
    allow_gate_bypass: bool = False,
) -> tuple[bool, bool]:
    meta_short = meta_short or {}
    short_holdout_trades = int(meta_short.get("holdout_trades", 0) or 0)
    short_gate_pass = (
        bool(meta_short.get("gate_pass", False))
        or (
            float(meta_short.get("overall_auc", 0.0) or 0.0) >= SHORT_MIN_LIVE_AUC
            and short_holdout_trades >= SHORT_PAPER_MIN_HOLDOUT_TRADES
        )
    )
    short_gate_bypassed = bool(allow_gate_bypass and has_model and not short_gate_pass)
    return bool(has_model and (short_gate_pass or short_gate_bypassed)), short_gate_bypassed


def load_redesign_context() -> dict:
    recovery = _read_json(CANDIDATE_RECOVERY_FILE)
    lanes = _read_json(EXPERIMENT_LANES_FILE)
    trial = (lanes.get("candidate_trial") or {}) if isinstance(lanes, dict) else {}
    candidate_type = str(recovery.get("type", "") or "")
    trial_type = str(trial.get("type", "") or "")
    trial_status = str(trial.get("status", "") or "").lower()
    redesign_active = candidate_type == "quantforge_redesign" or (
        trial_type == "quantforge_redesign" and trial_status in ACTIVE_TRIAL_STATUSES
    )
    research_hold_active = _research_surface_active(candidate_type, trial_type, trial_status)
    setup_quality_active = candidate_type == "setup_quality_recovery" or (
        trial_type == "setup_quality_recovery" and trial_status in ACTIVE_TRIAL_STATUSES
    )
    context = {
        "candidate_type": candidate_type,
        "trial_type": trial_type,
        "trial_status": trial_status,
        "redesign_active": bool(redesign_active),
        "research_hold_active": bool(research_hold_active),
        "setup_quality_active": bool(setup_quality_active),
    }
    context["surface_mode"] = _surface_mode_from_context(context)
    return context


def apply_training_target_profile(df, redesign_context: dict | None = None) -> tuple[pd.DataFrame, dict]:
    redesign_context = redesign_context or load_redesign_context()
    if redesign_context.get("research_hold_active"):
        df_profiled, profile = apply_research_hold_target_profile(df, horizon=4)
        return df_profiled, profile
    if redesign_context.get("setup_quality_active"):
        df_profiled, profile = apply_setup_quality_target_profile(df, horizon=4)
        return df_profiled, profile
    return df, {
        "profile": "standard_directional",
        "horizon_hours": 4,
        "long_target_col": TARGET_COL,
        "short_target_col": TARGET_SHORT_COL,
        "base_target_col": TARGET_COL,
        "base_short_target_col": TARGET_SHORT_COL,
        "fwd_ret_col": FWD_RET_COL,
        "class_balance": {
            "base_long_positive_rate": round(float(df[TARGET_COL].mean()), 6) if TARGET_COL in df.columns else None,
            "research_long_positive_rate": round(float(df[TARGET_COL].mean()), 6) if TARGET_COL in df.columns else None,
            "base_short_positive_rate": round(float(df[TARGET_SHORT_COL].mean()), 6) if TARGET_SHORT_COL in df.columns else None,
            "research_short_positive_rate": round(float(df[TARGET_SHORT_COL].mean()), 6) if TARGET_SHORT_COL in df.columns else None,
        },
    }


def _max_available_cols(df, cols):
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index)
    return df[available].fillna(0.0).max(axis=1)


def _setup_weight_profile(df, target_col: str) -> np.ndarray:
    """Bias training toward explicit setups and away from generic/unlabeled rows."""
    long_setup_strength = _max_available_cols(df, LONG_SETUP_SCORE_COLS)
    short_setup_strength = _max_available_cols(df, SHORT_SETUP_SCORE_COLS)
    setup_strength = long_setup_strength if target_col == TARGET_COL else short_setup_strength

    generic_penalty = np.where(setup_strength < 0.55, 0.70, 1.0)
    labeled_boost = 1.0 + np.clip(setup_strength.values, 0.0, 1.0) * 0.65
    return generic_penalty * labeled_boost


def _setup_context_from_row(row: "pd.Series", *, side: str) -> tuple[str, float]:
    score_cols = LONG_SETUP_SCORE_COLS if side == "BUY" else SHORT_SETUP_SCORE_COLS
    best_tag = "unknown"
    best_score = 0.0
    for col in score_cols:
        if col not in row.index:
            continue
        score = float(row.get(col, 0.0) or 0.0)
        if score >= best_score:
            best_score = score
            best_tag = col.replace("setup_", "").replace("_score", "")
    if best_score < 0.55:
        fallback = "generic_long" if side == "BUY" else "generic_short"
        return fallback, round(best_score, 4)
    return best_tag, round(best_score, 4)


def _start_mlflow_run(run_name: str):
    if mlflow is None:
        return None
    try:
        mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
        mlflow.set_experiment("quantforge-training")
        return mlflow.start_run(run_name=run_name)
    except Exception:
        return None


def get_feature_cols(df):
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and not c.startswith("target_")
        and not c.startswith("fwd_ret_")
        and not c.startswith("rebuild_net_ret_")
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]


def _build_xgb_classifier(params=None, *, scale_pos_weight=None):
    model_params = dict(DEFAULT_XGB_PARAMS)
    if params:
        model_params.update(params)
    if scale_pos_weight is not None:
        model_params["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(**model_params)


def _build_lgb_classifier(params=None, *, scale_pos_weight=None):
    model_params = dict(DEFAULT_LGB_PARAMS)
    if params:
        model_params.update(params)
    if scale_pos_weight is not None:
        model_params["scale_pos_weight"] = scale_pos_weight
    return lgb.LGBMClassifier(**model_params)


def load_optimized_params():
    if not os.path.exists(BEST_PARAMS_FILE):
        return None
    try:
        with open(BEST_PARAMS_FILE) as f:
            payload = json.load(f)
        return {
            "xgb_params": payload.get("xgb_params", {}),
            "lgb_params": payload.get("lgb_params", {}),
            "n_splits": int(payload.get("n_splits", 5)),
            "optimized_threshold": payload.get("threshold"),
            "optimized_at": payload.get("optimized_at"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward_eval(df, n_splits=5, target_col=None, fwd_ret_col=None, xgb_params=None, lgb_params=None):
    """
    Chronological walk-forward validation.
    Train on past data, test on future unseen data — no lookahead.
    """
    if target_col is None: target_col = TARGET_COL
    if fwd_ret_col is None: fwd_ret_col = FWD_RET_COL
    df = df.sort_values("ts").reset_index(drop=True)
    feature_cols = get_feature_cols(df)
    df_clean = df.dropna(subset=feature_cols + [target_col, fwd_ret_col])

    n = len(df_clean)
    fold_size = n // (n_splits + 1)

    all_probs = []
    all_labels = []
    all_fwd_rets = []
    fold_results = []

    print(f"  Walk-forward: {n:,} rows, {n_splits} folds, fold_size={fold_size:,}")

    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)

        if test_end <= test_start:
            break

        train_df = df_clean.iloc[:train_end]
        test_df = df_clean.iloc[test_start:test_end]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values.astype(int)
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values.astype(int)
        fwd_rets_test = test_df[fwd_ret_col].values

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        # XGBoost
        xgb_model = _build_xgb_classifier(xgb_params, scale_pos_weight=scale_pos_weight)
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # LightGBM
        lgb_model = _build_lgb_classifier(lgb_params, scale_pos_weight=scale_pos_weight)
        lgb_model.fit(X_train, y_train)

        # Ensemble: average probabilities
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
        ensemble_probs = (xgb_probs + lgb_probs) / 2

        all_probs.extend(ensemble_probs)
        all_labels.extend(y_test)
        all_fwd_rets.extend(fwd_rets_test)

        auc = roc_auc_score(y_test, ensemble_probs)
        fold_results.append({"fold": fold + 1, "test_size": len(test_df), "auc": round(float(auc), 4)})
        print(f"  Fold {fold+1}: test={len(test_df):,} | AUC={auc:.4f}")

    return np.array(all_probs), np.array(all_labels), np.array(all_fwd_rets), fold_results


def find_optimal_threshold(probs, labels, fwd_rets, *, direction="long"):
    """
    Sweep thresholds — find the one maximizing expected value after fees.
    EV = win_rate * avg_win - loss_rate * avg_loss - round_trip_cost
    """
    best = {"threshold": 0.5, "ev": -999.0, "win_rate": 0.0, "trade_pct": 0.0, "sharpe": 0.0, "trades": 0}
    results = []

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP)
    for threshold in thresholds:
        mask = probs >= threshold
        n_trades = int(mask.sum())
        if n_trades < 30:
            continue

        selected_rets = fwd_rets[mask]
        pnl_rets = selected_rets if direction == "long" else -selected_rets
        wins = pnl_rets > ROUND_TRIP_COST
        win_rate = float(wins.mean())
        avg_win = float(pnl_rets[wins].mean()) if wins.sum() > 0 else 0.0
        avg_loss = float(abs(pnl_rets[~wins].mean())) if (~wins).sum() > 0 else 0.0

        ev = win_rate * avg_win - (1 - win_rate) * avg_loss - ROUND_TRIP_COST
        trade_pct = float(mask.mean())

        ret_std = float(pnl_rets.std())
        # Annualize by ACTUAL trade frequency: 2190 4h-slots/year scaled by the
        # fraction of slots actually traded. The old flat sqrt(2190) credited a
        # 2%-selectivity strategy as if it traded every bar (~7x Sharpe inflation).
        # Still optimistic for pooled multi-symbol data (assumes parallel capacity).
        trades_per_year = max(1.0, 2190.0 * trade_pct)
        sharpe = float(pnl_rets.mean() / ret_std * trades_per_year ** 0.5) if ret_std > 0 else 0.0

        row = {
            "threshold": round(float(threshold), 3),
            "trades": n_trades,
            "trade_pct": round(trade_pct, 4),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 5),
            "avg_loss": round(avg_loss, 5),
            "ev": round(ev, 6),
            "sharpe": round(sharpe, 4),
        }
        results.append(row)

        if ev > best["ev"]:
            best = {
                "threshold": float(threshold), "ev": float(ev),
                "win_rate": float(win_rate), "trade_pct": float(trade_pct),
                "sharpe": float(sharpe), "trades": n_trades,
            }

    return best, results


def evaluate_gate(
    best_threshold,
    *,
    label="Model",
    min_win_rate=0.55,
    min_sharpe=1.0,
    min_auc=None,
    min_holdout_trades=None,
):
    """Return readiness decision and log the gate outcome."""
    ho_win_rate = best_threshold.get("holdout_win_rate")
    ho_sharpe = best_threshold.get("holdout_sharpe")
    ho_auc = best_threshold.get("holdout_auc")
    ho_trades = int(best_threshold.get("holdout_trades", 0) or 0)
    holdout_ready = ho_win_rate is not None and ho_sharpe is not None
    ready = holdout_ready and ho_win_rate >= min_win_rate and ho_sharpe >= min_sharpe
    reasons = []
    if min_auc is not None:
        ready = ready and ho_auc is not None and ho_auc >= min_auc
    if min_holdout_trades is not None:
        ready = ready and ho_trades >= min_holdout_trades

    print("\n" + "=" * 50)
    if ready:
        print(f"GATE: *** PASS *** — {label} ready for live trading")
        print(f"  Hold-out win rate {ho_win_rate:.1%} >= {min_win_rate:.0%}")
        print(f"  Hold-out Sharpe   {ho_sharpe:.2f} >= {min_sharpe}")
        if min_auc is not None and ho_auc is not None:
            print(f"  Hold-out AUC      {ho_auc:.4f} >= {min_auc:.2f}")
        if min_holdout_trades is not None:
            print(f"  Hold-out trades   {ho_trades:,} >= {min_holdout_trades:,}")
    else:
        if not holdout_ready:
            reasons.append("Hold-out metrics unavailable")
        elif ho_win_rate < min_win_rate:
            reasons.append(f"Hold-out win rate {ho_win_rate:.1%} < {min_win_rate:.0%}")
        if holdout_ready and ho_sharpe < min_sharpe:
            reasons.append(f"Hold-out Sharpe {ho_sharpe:.2f} < {min_sharpe}")
        if min_auc is not None:
            if ho_auc is None:
                reasons.append("Hold-out AUC unavailable")
            elif ho_auc < min_auc:
                reasons.append(f"Hold-out AUC {ho_auc:.4f} < {min_auc:.2f}")
        if min_holdout_trades is not None and ho_trades < min_holdout_trades:
            reasons.append(f"Hold-out trades {ho_trades:,} < {min_holdout_trades:,}")
        print(f"GATE: FAIL — {'; '.join(reasons)}")
    print(f"  (CV win rate: {best_threshold['win_rate']:.1%}, CV Sharpe: {best_threshold['sharpe']:.2f} — for reference)")
    print("=" * 50)
    return ready, {
        "label": label,
        "ready": bool(ready),
        "criteria": {
            "min_win_rate": float(min_win_rate),
            "min_sharpe": float(min_sharpe),
            "min_auc": float(min_auc) if min_auc is not None else None,
            "min_holdout_trades": int(min_holdout_trades) if min_holdout_trades is not None else None,
        },
        "measured": {
            "cv_win_rate": float(best_threshold["win_rate"]),
            "cv_sharpe": float(best_threshold["sharpe"]),
            "cv_trades": int(best_threshold.get("trades", 0) or 0),
            "holdout_win_rate": float(ho_win_rate) if ho_win_rate is not None else None,
            "holdout_sharpe": float(ho_sharpe) if ho_sharpe is not None else None,
            "holdout_auc": float(ho_auc) if ho_auc is not None else None,
            "holdout_trades": int(ho_trades),
        },
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Train final model
# ---------------------------------------------------------------------------

RECENCY_LAMBDA = 2.0   # Exponential decay strength — higher = newer data matters more
                        # λ=2 → newest row is ~7x more influential than oldest

def _recency_weights(ts_series: "pd.Series") -> "np.ndarray":
    """Exponential recency weights: w_i = exp(λ * (t_i - t_min) / (t_max - t_min)).
    Newest row → e^λ ≈ 7.4x. Oldest row → 1.0. All middle rows interpolated.
    """
    t = ts_series.values.astype(float)
    span = t.max() - t.min()
    if span == 0:
        return np.ones(len(t))
    return np.exp(RECENCY_LAMBDA * (t - t.min()) / span)


def train_final_model(
    df,
    target_col=None,
    xgb_params=None,
    lgb_params=None,
    redesign_context=None,
    target_profile=None,
):
    """Train final ensemble on full dataset with exponential recency weighting.
    Recent candles are weighted up to 7x more than candles from 2+ years ago,
    so the model adapts to current market conditions without forgetting history.
    """
    if target_col is None: target_col = TARGET_COL
    feature_cols = get_feature_cols(df)
    df_clean = df.dropna(subset=feature_cols + [target_col]).sort_values("ts")
    redesign_context = redesign_context or load_redesign_context()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values.astype(int)
    w = _recency_weights(df_clean["ts"])  # exponential recency weights

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Class-balance weights multiplied by recency weights
    class_w = np.where(y == 1, scale_pos_weight, 1.0)
    sample_w = w * class_w
    if redesign_context.get("redesign_active") or redesign_context.get("setup_quality_active"):
        sample_w = sample_w * _setup_weight_profile(df_clean, target_col)

    final_xgb_params = dict(FINAL_XGB_PARAMS)
    if xgb_params:
        final_xgb_params.update(xgb_params)
    xgb_model = xgb.XGBClassifier(**final_xgb_params)
    xgb_model.fit(X, y, sample_weight=sample_w, verbose=False)

    final_lgb_params = dict(FINAL_LGB_PARAMS)
    if lgb_params:
        final_lgb_params.update(lgb_params)
    lgb_model = lgb.LGBMClassifier(**final_lgb_params)
    lgb_model.fit(X, y, sample_weight=sample_w)

    importances = dict(zip(feature_cols, xgb_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]

    training_profile = {
        "redesign_active": bool(redesign_context.get("redesign_active")),
        "research_hold_active": bool(redesign_context.get("research_hold_active")),
        "setup_quality_active": bool(redesign_context.get("setup_quality_active")),
        "candidate_type": redesign_context.get("candidate_type"),
        "trial_type": redesign_context.get("trial_type"),
        "trial_status": redesign_context.get("trial_status"),
        "surface_mode": _surface_mode_from_context(redesign_context),
        "setup_weighting": bool(redesign_context.get("redesign_active") or redesign_context.get("setup_quality_active")),
        "target_col": target_col,
        "target_profile": target_profile or {},
    }

    return xgb_model, lgb_model, feature_cols, top_features, training_profile


def train_setup_specialist_models(
    df,
    *,
    df_cv,
    df_holdout,
    target_profile,
    fwd_ret_col,
    xgb_params=None,
    lgb_params=None,
    redesign_context=None,
    min_positive_rows=250,
):
    results = []
    support_counts = target_profile.get("support_counts") or {}
    setup_target_cols = target_profile.get("setup_target_cols") or {}
    for setup_name, fallback_col in LONG_SPECIALIST_TARGETS:
        target_col = str(setup_target_cols.get(setup_name) or fallback_col)
        positive_rows = int(support_counts.get(f"long_{setup_name.split('_')[0]}_positive", 0) or 0)
        if target_col not in df.columns or positive_rows < min_positive_rows:
            results.append({
                "setup": setup_name,
                "status": "skipped",
                "target_col": target_col,
                "positive_rows": positive_rows,
                "reason": f"insufficient positives (< {min_positive_rows}) or target missing",
            })
            continue

        probs, labels, fwd_rets, fold_results = walk_forward_eval(
            df_cv,
            n_splits=5,
            target_col=target_col,
            fwd_ret_col=fwd_ret_col,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
        )
        if len(labels) < 50 or len(np.unique(labels)) < 2:
            results.append({
                "setup": setup_name,
                "status": "skipped",
                "target_col": target_col,
                "positive_rows": positive_rows,
                "reason": "insufficient evaluation labels after walk-forward split",
            })
            continue

        overall_auc = roc_auc_score(labels, probs)
        best_threshold, threshold_results = find_optimal_threshold(probs, labels, fwd_rets)

        holdout_summary = {}
        try:
            feature_cols_tmp = get_feature_cols(df_holdout)
            df_cv_clean = df_cv.dropna(subset=feature_cols_tmp + [target_col, fwd_ret_col]).sort_values("ts")
            df_ho_clean = df_holdout.dropna(subset=feature_cols_tmp + [target_col, fwd_ret_col]).sort_values("ts")
            if len(df_cv_clean) and len(df_ho_clean):
                _xgb_tmp = _build_xgb_classifier(xgb_params)
                _xgb_tmp.fit(df_cv_clean[feature_cols_tmp].values, df_cv_clean[target_col].values.astype(int))
                ho_probs = _xgb_tmp.predict_proba(df_ho_clean[feature_cols_tmp].values)[:, 1]
                ho_labels = df_ho_clean[target_col].values.astype(int)
                ho_fwd = df_ho_clean[fwd_ret_col].values
                ho_mask = ho_probs >= best_threshold["threshold"]
                ho_trades = int(ho_mask.sum())
                if ho_trades > 10 and len(np.unique(ho_labels)) > 1:
                    ho_wins = (ho_fwd[ho_mask] > ROUND_TRIP_COST).sum()
                    ho_win_rate = ho_wins / ho_trades
                    ho_auc = roc_auc_score(ho_labels, ho_probs)
                    ho_std = ho_fwd[ho_mask].std()
                    ho_sharpe = (ho_fwd[ho_mask].mean() / ho_std * (2190) ** 0.5) if ho_std > 0 else 0
                    holdout_summary = {
                        "holdout_auc": round(float(ho_auc), 4),
                        "holdout_win_rate": round(float(ho_win_rate), 4),
                        "holdout_sharpe": round(float(ho_sharpe), 4),
                        "holdout_trades": ho_trades,
                    }
        except Exception as exc:
            holdout_summary = {"holdout_error": str(exc)}

        xgb_model, lgb_model, feature_cols, top_features, training_profile = train_final_model(
            df,
            target_col=target_col,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
            redesign_context=redesign_context,
            target_profile=target_profile,
        )
        specialist_model_path = os.path.join(MODEL_DIR, f"ensemble_{setup_name}.pkl")
        with open(specialist_model_path, "wb") as f:
            pickle.dump((xgb_model, lgb_model, feature_cols), f)

        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "setup": setup_name,
            "target_col": target_col,
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "positive_rows": positive_rows,
            "overall_auc": round(float(overall_auc), 4),
            "optimal_threshold": float(best_threshold["threshold"]),
            "win_rate_at_threshold": float(best_threshold["win_rate"]),
            "ev_at_threshold": float(best_threshold["ev"]),
            "sharpe_at_threshold": float(best_threshold["sharpe"]),
            "fold_results": fold_results,
            "threshold_sweep": threshold_results,
            "top_features": [{"feature": f, "importance": round(float(i), 4)} for f, i in top_features],
            "training_profile": training_profile,
            **holdout_summary,
        }
        specialist_meta_path = os.path.join(MODEL_DIR, f"model_meta_{setup_name}.json")
        with open(specialist_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        results.append({
            "setup": setup_name,
            "status": "trained",
            "target_col": target_col,
            "positive_rows": positive_rows,
            "overall_auc": round(float(overall_auc), 4),
            "model_path": specialist_model_path,
            "meta_path": specialist_meta_path,
            **holdout_summary,
        })
    return results


# ---------------------------------------------------------------------------
# Live signal generation
# ---------------------------------------------------------------------------

def load_model(short=False):
    """Load saved model + metadata from disk. short=True loads the SHORT ensemble."""
    suffix = "_short" if short else ""
    model_path = os.path.join(MODEL_DIR, f"ensemble{suffix}.pkl")
    meta_path  = os.path.join(MODEL_DIR, f"model_meta{suffix}.json")
    if not os.path.exists(model_path):
        return None, None, None, None
    with open(model_path, "rb") as f:
        xgb_model, lgb_model, feature_cols = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    return xgb_model, lgb_model, feature_cols, meta


def load_setup_specialist_model(setup_name: str):
    """Load a trained setup-specific long specialist if present."""
    model_path = os.path.join(MODEL_DIR, f"ensemble_{setup_name}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"model_meta_{setup_name}.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None, None, None
    with open(model_path, "rb") as f:
        xgb_model, lgb_model, feature_cols = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    return xgb_model, lgb_model, feature_cols, meta


def _specialist_surface_matches(meta: dict | None, redesign_context: dict | None) -> bool:
    meta = meta or {}
    redesign_context = redesign_context or {}
    active_surface_mode = _surface_mode_from_context(redesign_context)
    model_surface_mode = _surface_mode_from_training_profile(meta.get("training_profile"))
    return active_surface_mode == model_surface_mode


def _specialist_model_ready(
    meta: dict | None,
    summary_row: dict | None,
    redesign_context: dict | None,
    *,
    allow_gate_bypass: bool = False,
) -> tuple[bool, bool]:
    meta = meta or {}
    summary_row = summary_row or {}
    redesign_context = redesign_context or {}
    gate_pass, gate_bypassed = _signal_gate_status(meta, allow_gate_bypass=allow_gate_bypass)
    if gate_pass or gate_bypassed:
        return True, gate_bypassed
    if not redesign_context.get("research_hold_active"):
        return False, False
    positive_rows = int(meta.get("positive_rows", summary_row.get("positive_rows", 0)) or 0)
    overall_auc = float(meta.get("overall_auc", summary_row.get("overall_auc", 0.0)) or 0.0)
    holdout_trades = int(meta.get("holdout_trades", summary_row.get("holdout_trades", 0)) or 0)
    if positive_rows < LONG_SPECIALIST_MIN_READY_POSITIVE_ROWS or overall_auc < LONG_SPECIALIST_MIN_READY_AUC:
        return False, False
    if "holdout_trades" in meta or "holdout_trades" in summary_row:
        return holdout_trades >= LONG_SPECIALIST_MIN_READY_HOLDOUT_TRADES, False
    return True, False


def _score_long_setup_specialist(
    setup_tag: str,
    last: "pd.Series",
    redesign_context: dict | None,
    *,
    allow_gate_bypass: bool = False,
) -> dict | None:
    redesign_context = redesign_context or {}
    if not redesign_context.get("research_hold_active"):
        return None
    if setup_tag not in LONG_SPECIALIST_SETUP_NAMES:
        return None
    registry = _read_json(os.path.join(MODEL_DIR, "setup_specialists.json"))
    specialists = registry.get("specialists") or []
    summary_row = next(
        (row for row in specialists if str(row.get("setup", "") or "") == setup_tag and str(row.get("status", "") or "") == "trained"),
        None,
    )
    xgb_model, lgb_model, feature_cols, meta = load_setup_specialist_model(setup_tag)
    if xgb_model is None or meta is None or not feature_cols:
        return None
    if not _specialist_surface_matches(meta, redesign_context):
        return None
    ready, gate_bypassed = _specialist_model_ready(
        meta,
        summary_row,
        redesign_context,
        allow_gate_bypass=allow_gate_bypass,
    )
    if not ready:
        return None
    row_values = [float(last[col]) if col in last.index and pd.notna(last[col]) else 0.0 for col in feature_cols]
    X = np.array([row_values])
    xgb_prob = float(xgb_model.predict_proba(X)[0][1])
    lgb_prob = float(lgb_model.predict_proba(X)[0][1])
    confidence = (xgb_prob + lgb_prob) / 2
    return {
        "setup": setup_tag,
        "confidence": confidence,
        "threshold": float(meta.get("optimal_threshold", 0.60)),
        "gate_bypassed": bool(gate_bypassed),
        "surface_mode": _surface_mode_from_training_profile(meta.get("training_profile")),
    }


def _apply_long_specialist_confirmation(
    long_confidence: float,
    specialist_signal: dict | None,
    *,
    base_gate_ready: bool = True,
) -> tuple[float, list[str]]:
    if not specialist_signal:
        return (float(long_confidence) if base_gate_ready else 0.0), []
    confidence = float(specialist_signal.get("confidence", 0.0) or 0.0)
    threshold = float(specialist_signal.get("threshold", 1.0) or 1.0)
    setup = str(specialist_signal.get("setup", "unknown") or "unknown")
    if confidence >= threshold:
        if not base_gate_ready:
            return confidence, [f"{setup} specialist reopened long while the base long gate was closed ({confidence:.3f} >= {threshold:.3f})"]
        blended = max(float(long_confidence), (float(long_confidence) + confidence) / 2.0)
        return blended, [f"{setup} specialist confirmed long ({confidence:.3f} >= {threshold:.3f})"]
    if not base_gate_ready:
        return 0.0, [f"{setup} specialist kept long gate closed ({confidence:.3f} < {threshold:.3f})"]
    return min(float(long_confidence), confidence), [f"{setup} specialist vetoed long ({confidence:.3f} < {threshold:.3f})"]


def _dominant_setup_payload(
    *,
    long_setup_tag: str,
    long_setup_score: float,
    short_setup_tag: str,
    short_setup_score: float,
    long_confidence: float,
    short_confidence: float,
) -> tuple[str, float, str]:
    if float(long_confidence) >= float(short_confidence):
        return str(long_setup_tag), float(long_setup_score), "LONG"
    return str(short_setup_tag), float(short_setup_score), "SHORT"


def generate_signal(symbol, candles_df=None, long_threshold_override=None, short_threshold_override=None, allow_gate_bypass=False):
    """
    Generate ML-based trading signal for a symbol.
    candles_df: DataFrame with OHLCV columns, oldest-first, at least 250 rows.
    Returns dict: signal (BUY/SELL/HOLD), confidence, reason
    """
    redesign_context = load_redesign_context()
    xgb_model, lgb_model, feature_cols, meta = load_model(short=False)
    xgb_short, lgb_short, feature_cols_short, meta_short = load_model(short=True)
    training_profile = (meta or {}).get("training_profile") or {}
    short_training_profile = (meta_short or {}).get("training_profile") or {}
    if xgb_model is None:
        return {"signal": "HOLD", "confidence": 0.0, "reason": ["Model not trained"]}
    long_gate_pass, long_gate_bypassed = _signal_gate_status(meta, allow_gate_bypass=allow_gate_bypass)
    long_gate_blocked = not long_gate_pass and not long_gate_bypassed
    if long_gate_blocked and not redesign_context.get("research_hold_active"):
        return {"signal": "HOLD", "confidence": 0.0, "reason": ["Model gate failed; trading disabled until retraining passes"]}

    if candles_df is None or len(candles_df) < 250:
        return {"signal": "HOLD", "confidence": 0.0, "reason": ["Insufficient candle data"]}

    # Import feature builders
    from quantforge_features import (
        add_returns, add_rsi, add_macd, add_bollinger, add_emas,
        add_atr, add_volume_features, add_stochastic, add_adx, add_candle_patterns,
        add_behavior_profile_features, add_flow_and_move_features, add_setup_context_features,
        merge_context, build_benchmark_context, merge_benchmark_context
    )

    df = candles_df.copy()
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
    df = add_setup_context_features(df)

    # Load 4h and 1d data for multi-timeframe context (critical for top features)
    safe = symbol.replace("-", "_")
    csv_4h = os.path.join(HISTORICAL_DIR, f"{safe}_4h.csv")
    csv_1d = os.path.join(HISTORICAL_DIR, f"{safe}_1d.csv")
    if os.path.exists(csv_4h):
        df_4h = pd.read_csv(csv_4h)
        df_1d = pd.read_csv(csv_1d) if os.path.exists(csv_1d) else df_4h
        df = merge_context(df, df_4h, df_1d)
    benchmark_ctx = build_benchmark_context()
    df = merge_benchmark_context(df, benchmark_ctx)

    last = df.iloc[-1]
    row_values = [float(last[col]) if col in last.index and pd.notna(last[col]) else 0.0 for col in feature_cols]
    X = np.array([row_values])

    xgb_prob = float(xgb_model.predict_proba(X)[0][1])
    lgb_prob = float(lgb_model.predict_proba(X)[0][1])
    confidence = (xgb_prob + lgb_prob) / 2

    # SHORT model confidence (if trained)
    short_confidence = 0.0
    short_threshold  = 1.0  # default: never fires if no short model
    short_meta_ready, short_gate_bypassed = _short_model_ready_for_signal(
        meta_short,
        has_model=xgb_short is not None,
        allow_gate_bypass=allow_gate_bypass,
    )
    if short_meta_ready:
        row_short = [float(last[c]) if c in last.index and pd.notna(last[c]) else 0.0
                     for c in feature_cols_short]
        xgb_sp = float(xgb_short.predict_proba(np.array([row_short]))[0][1])
        lgb_sp = float(lgb_short.predict_proba(np.array([row_short]))[0][1])
        short_confidence = (xgb_sp + lgb_sp) / 2
        short_threshold  = float(short_threshold_override) if short_threshold_override is not None else float(meta_short.get("optimal_threshold", 0.60))

    threshold = float(long_threshold_override) if long_threshold_override is not None else float(meta.get("optimal_threshold", 0.60))
    win_rate = float(meta.get("win_rate_at_threshold", 0.5))
    expected_value = float(meta.get("ev_at_threshold", 0.0))

    long_confidence = confidence
    short_conf_for_signal = short_confidence
    long_setup_tag, long_setup_score = _setup_context_from_row(last, side="BUY")
    short_setup_tag, short_setup_score = _setup_context_from_row(last, side="SELL")
    risk_filter_profile = "legacy"
    redesign_notes = []
    gate_notes = []
    specialist_notes = []
    if long_gate_bypassed:
        gate_notes.append("Paper-only trial bypassed long model gate")
    if short_gate_bypassed:
        gate_notes.append("Paper-only trial bypassed short model gate")
    if long_gate_blocked:
        gate_notes.append("Long model gate failed; only research-hold specialists may authorize longs")
    specialist_signal = _score_long_setup_specialist(
        long_setup_tag,
        last,
        redesign_context,
        allow_gate_bypass=allow_gate_bypass,
    )
    long_confidence, specialist_notes = _apply_long_specialist_confirmation(
        long_confidence,
        specialist_signal,
        base_gate_ready=not long_gate_blocked,
    )
    if specialist_signal:
        risk_filter_profile = "research_hold_specialist_confirmation"
    if redesign_context.get("redesign_active"):
        risk_filter_profile = "prediction_risk_execution_split"
        if long_setup_score < 0.55:
            long_confidence *= 0.82
            redesign_notes.append(f"Redesign penalized weak unlabeled long setup ({long_setup_tag}, {long_setup_score:.2f})")
        else:
            long_confidence = 0.85 * long_confidence + 0.15 * long_setup_score
            redesign_notes.append(f"Redesign blended long setup coherence ({long_setup_tag}, {long_setup_score:.2f})")
        if short_setup_score < 0.55:
            short_conf_for_signal *= 0.88
        else:
            short_conf_for_signal = 0.90 * short_conf_for_signal + 0.10 * short_setup_score

    long_ok  = long_confidence >= threshold
    short_ok = short_conf_for_signal >= short_threshold

    if long_ok and short_ok:
        # Both fire — pick stronger signal
        long_ok  = long_confidence >= short_conf_for_signal
        short_ok = not long_ok

    dominant_setup_tag, dominant_setup_score, dominant_setup_direction = _dominant_setup_payload(
        long_setup_tag=long_setup_tag,
        long_setup_score=long_setup_score,
        short_setup_tag=short_setup_tag,
        short_setup_score=short_setup_score,
        long_confidence=long_confidence,
        short_confidence=short_conf_for_signal,
    )
    setup_tag = dominant_setup_tag
    setup_score = dominant_setup_score
    policy_score = 0.0
    if long_ok:
        signal = "BUY"
        setup_tag = long_setup_tag
        setup_score = long_setup_score
        policy_score = long_confidence
        reason = [
            f"ML LONG confidence {long_confidence:.3f} >= threshold {threshold:.3f}",
            f"Expected win rate: {win_rate:.1%}",
            f"Expected value after fees: {expected_value:.4f}",
        ] + gate_notes + specialist_notes + redesign_notes
        confidence = long_confidence
    elif short_ok:
        signal = "SELL"
        setup_tag = short_setup_tag
        setup_score = short_setup_score
        policy_score = short_conf_for_signal
        reason = [
            f"ML SHORT confidence {short_conf_for_signal:.3f} >= threshold {short_threshold:.3f}",
            f"Short model: expected down move >= 2.5%",
        ] + gate_notes
        confidence = short_conf_for_signal
    else:
        signal = "HOLD"
        reason = [f"LONG conf {long_confidence:.3f} < {threshold:.3f}, SHORT conf {short_conf_for_signal:.3f} < {short_threshold:.3f}"] + gate_notes + specialist_notes + redesign_notes

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "long_confidence": round(long_confidence, 4),
        "short_confidence": round(short_conf_for_signal, 4),
        "reason": reason,
        "xgb_prob": round(xgb_prob, 4),
        "lgb_prob": round(lgb_prob, 4),
        "threshold": threshold,
        "setup_tag": setup_tag,
        "setup_score": round(setup_score, 4),
        "dominant_setup_tag": dominant_setup_tag,
        "dominant_setup_score": round(float(dominant_setup_score), 4),
        "dominant_setup_direction": dominant_setup_direction,
        "dominant_long_setup_tag": long_setup_tag,
        "dominant_long_setup_score": round(float(long_setup_score), 4),
        "dominant_short_setup_tag": short_setup_tag,
        "dominant_short_setup_score": round(float(short_setup_score), 4),
        "fakeout_risk": round(float(last.get("fakeout_risk", 0.0) or 0.0), 4),
        "policy_score": round(policy_score, 4),
        "risk_filter_profile": risk_filter_profile,
        "redesign_active": bool(redesign_context.get("redesign_active")),
        "research_hold_active": bool(redesign_context.get("research_hold_active")),
        "gate_bypassed": bool(long_gate_bypassed or short_gate_bypassed),
        "long_gate_pass": bool(long_gate_pass),
        "short_gate_pass": bool(meta_short.get("gate_pass", False) if meta_short else False),
        "specialist_setup": specialist_signal.get("setup") if specialist_signal else None,
        "specialist_confidence": round(float(specialist_signal.get("confidence", 0.0) or 0.0), 4) if specialist_signal else 0.0,
        "specialist_threshold": float(specialist_signal.get("threshold", 0.0) or 0.0) if specialist_signal else 0.0,
        "specialist_gate_bypassed": bool(specialist_signal.get("gate_bypassed", False)) if specialist_signal else False,
        "active_surface_mode": _surface_mode_from_context(redesign_context),
        "model_surface_mode": _surface_mode_from_training_profile(training_profile),
        "short_model_surface_mode": _surface_mode_from_training_profile(short_training_profile),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(pairs=None):
    all_path = os.path.join(FEATURES_DIR, "features_all.parquet")
    if not os.path.exists(all_path):
        print("No features found. Run: python3 quantforge_features.py build")
        return

    df = pd.read_parquet(all_path)
    if pairs:
        df = df[df["symbol"].isin(pairs)]
    mlflow_run = _start_mlflow_run(f"train-{datetime.now(timezone.utc).isoformat()}")
    redesign_context = load_redesign_context()
    df, target_profile = apply_training_target_profile(df, redesign_context)
    long_target_col = target_profile.get("long_target_col", TARGET_COL)
    short_target_col = target_profile.get("short_target_col", TARGET_SHORT_COL)
    fwd_ret_col = target_profile.get("fwd_ret_col", FWD_RET_COL)

    print(f"Training on {len(df):,} rows across {df['symbol'].nunique()} pairs")
    print(f"Features: {len(get_feature_cols(df))}")
    print(f"Target profile: {target_profile.get('profile', 'standard_directional')}")
    print(f"Long target: {long_target_col}")
    print(f"Short target: {short_target_col}")
    print(f"Class balance: {df[long_target_col].mean():.1%} positive")
    print()
    if redesign_context.get("redesign_active"):
        print("Redesign mode: active")
        print("  Setup-aware weighting and split-risk metadata will be embedded in the model.")
        print()
    if redesign_context.get("setup_quality_active"):
        active_surface_owner = redesign_context.get("candidate_type") or redesign_context.get("trial_type") or "setup_quality_recovery"
        print(f"Setup-quality surface: active ({active_surface_owner})")
        print("  Training will bias toward labeled long recovery targets and reuse setup-aware weighting.")
        print()
    if redesign_context.get("research_hold_active"):
        active_surface_owner = redesign_context.get("candidate_type") or redesign_context.get("trial_type") or "quantforge_research_hold"
        print(f"Research surface: active ({active_surface_owner})")
        print("  Training will use setup-conditioned composite targets instead of raw 4h direction labels.")
        print()

    optimized = load_optimized_params()
    xgb_params = optimized.get("xgb_params", {}) if optimized else {}
    lgb_params = optimized.get("lgb_params", {}) if optimized else {}
    n_splits = optimized.get("n_splits", 5) if optimized else 5
    if optimized:
        print(f"Using optimized params from {BEST_PARAMS_FILE}")
        print(f"Optimized at: {optimized.get('optimized_at', 'unknown')}")
        print(f"CV folds:     {n_splits}")
        print()

    # Reserve last 20% of EACH symbol as hold-out — chronologically unseen data.
    # EMBARGO_BARS rows at the boundary are dropped entirely so no forward-looking
    # target window (or multi-timeframe context) from the CV side overlaps the
    # hold-out side. 48 hourly bars covers the longest target horizon plus the
    # 4h/1d context forward-fill with room to spare.
    EMBARGO_BARS = 48
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    holdout_rows = []
    cv_rows = []
    for sym, grp in df.groupby("symbol"):
        n = len(grp)
        split = int(n * 0.80)
        cv_end = max(0, split - EMBARGO_BARS)
        cv_rows.append(grp.iloc[:cv_end])
        holdout_rows.append(grp.iloc[split:])
    df_cv = pd.concat(cv_rows).sort_values("ts").reset_index(drop=True)
    df_holdout = pd.concat(holdout_rows).sort_values("ts").reset_index(drop=True)
    print(f"CV set:      {len(df_cv):,} rows  (80% minus {EMBARGO_BARS}-bar embargo)")
    print(f"Hold-out:    {len(df_holdout):,} rows  (last 20% — never seen during training)")
    print()

    print("Running walk-forward validation on CV set...")
    probs, labels, fwd_rets, fold_results = walk_forward_eval(
        df_cv,
        n_splits=n_splits,
        target_col=long_target_col,
        fwd_ret_col=fwd_ret_col,
        xgb_params=xgb_params,
        lgb_params=lgb_params,
    )
    print()

    overall_auc = roc_auc_score(labels, probs)
    print(f"CV AUC: {overall_auc:.4f}")
    print()

    print("Finding optimal confidence threshold on CV data...")
    best_threshold, threshold_results = find_optimal_threshold(probs, labels, fwd_rets)

    # Validate threshold on hold-out set (true out-of-sample performance)
    print(f"\nValidating threshold {best_threshold['threshold']:.3f} on HOLD-OUT set...")
    try:
        feature_cols_tmp = get_feature_cols(df_holdout)
        # Use a quick single model trained on CV set to score hold-out
        df_cv_clean = df_cv.dropna(subset=feature_cols_tmp + [long_target_col, fwd_ret_col]).sort_values("ts")
        df_ho_clean = df_holdout.dropna(subset=feature_cols_tmp + [long_target_col, fwd_ret_col]).sort_values("ts")
        _xgb_tmp = _build_xgb_classifier(xgb_params)
        _xgb_tmp.fit(df_cv_clean[feature_cols_tmp].values, df_cv_clean[long_target_col].values.astype(int))
        ho_probs = _xgb_tmp.predict_proba(df_ho_clean[feature_cols_tmp].values)[:, 1]
        ho_labels = df_ho_clean[long_target_col].values.astype(int)
        ho_fwd = df_ho_clean[fwd_ret_col].values
        ho_mask = ho_probs >= best_threshold["threshold"]
        ho_trades = ho_mask.sum()
        if ho_trades > 10:
            ho_wins = (ho_fwd[ho_mask] > 0.002).sum()  # > round-trip cost
            ho_win_rate = ho_wins / ho_trades
            ho_auc = roc_auc_score(ho_labels, ho_probs)
            ho_ret_std = ho_fwd[ho_mask].std()
            ho_sharpe = (ho_fwd[ho_mask].mean() / ho_ret_std * (2190) ** 0.5) if ho_ret_std > 0 else 0
            print(f"  Hold-out AUC:      {ho_auc:.4f}")
            print(f"  Hold-out trades:   {ho_trades:,}  ({ho_trades/len(ho_fwd):.1%} of signals)")
            print(f"  Hold-out win rate: {ho_win_rate:.1%}")
            print(f"  Hold-out Sharpe:   {ho_sharpe:.2f}")
            best_threshold["holdout_win_rate"] = round(float(ho_win_rate), 4)
            best_threshold["holdout_auc"] = round(float(ho_auc), 4)
            best_threshold["holdout_sharpe"] = round(float(ho_sharpe), 4)
            best_threshold["holdout_trades"] = int(ho_trades)
    except Exception as e:
        print(f"  Hold-out validation failed: {e}")

    print(f"\nOptimal threshold: {best_threshold['threshold']:.3f}")
    print(f"  Win rate:   {best_threshold['win_rate']:.1%}")
    print(f"  Trades:     {best_threshold['trades']:,} ({best_threshold['trade_pct']:.1%} of signals)")
    print(f"  Exp value:  {best_threshold['ev']:.4f} per trade")
    print(f"  Sharpe:     {best_threshold['sharpe']:.2f} (annualized)")

    print(f"\nThreshold sweep (top 10 by EV):")
    print(f"  {'Threshold':>10} {'Win Rate':>10} {'Trades':>8} {'EV':>10} {'Sharpe':>8}")
    sorted_results = sorted(threshold_results, key=lambda x: x["ev"], reverse=True)[:10]
    for r in sorted_results:
        print(f"  {r['threshold']:>10.3f} {r['win_rate']:>10.1%} {r['trades']:>8,} {r['ev']:>10.5f} {r['sharpe']:>8.2f}")

    ready, gate_eval = evaluate_gate(best_threshold, label="Long model")

    print("\nTraining final model on full dataset...")
    xgb_model, lgb_model, feature_cols, top_features, training_profile = train_final_model(
        df,
        target_col=long_target_col,
        xgb_params=xgb_params,
        lgb_params=lgb_params,
        redesign_context=redesign_context,
        target_profile=target_profile,
    )

    model_path = os.path.join(MODEL_DIR, "ensemble.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((xgb_model, lgb_model, feature_cols), f)

    # ── Train SHORT model (price drops >= 2.5% in 4h) ──────────────────
    print("\nTraining SHORT model (target_4h_short)...")
    if short_target_col in df.columns:
        try:
            short_probs, short_labels, short_fwd, short_folds = walk_forward_eval(
                df,
                n_splits=n_splits,
                target_col=short_target_col,
                fwd_ret_col=fwd_ret_col,
                xgb_params=xgb_params,
                lgb_params=lgb_params,
            )
            short_auc = roc_auc_score(short_labels, short_probs)
            print(f"  SHORT model AUC: {short_auc:.4f}")
            best_short, short_thresh_results = find_optimal_threshold(
                short_probs, short_labels, short_fwd, direction="short")
            print(f"\nValidating SHORT threshold {best_short['threshold']:.3f} on HOLD-OUT set...")
            try:
                feature_cols_tmp = get_feature_cols(df_holdout)
                df_cv_short = df_cv.dropna(subset=feature_cols_tmp + [short_target_col, fwd_ret_col]).sort_values("ts")
                df_ho_short = df_holdout.dropna(subset=feature_cols_tmp + [short_target_col, fwd_ret_col]).sort_values("ts")
                _xgb_short = _build_xgb_classifier(xgb_params)
                _xgb_short.fit(df_cv_short[feature_cols_tmp].values, df_cv_short[short_target_col].values.astype(int))
                ho_short_probs = _xgb_short.predict_proba(df_ho_short[feature_cols_tmp].values)[:, 1]
                ho_short_labels = df_ho_short[short_target_col].values.astype(int)
                ho_short_fwd = df_ho_short[fwd_ret_col].values
                ho_short_mask = ho_short_probs >= best_short["threshold"]
                ho_short_trades = ho_short_mask.sum()
                if ho_short_trades > 10:
                    ho_short_pnl = -ho_short_fwd[ho_short_mask]
                    ho_short_wins = (ho_short_pnl > ROUND_TRIP_COST).sum()
                    ho_short_win_rate = ho_short_wins / ho_short_trades
                    ho_short_auc = roc_auc_score(ho_short_labels, ho_short_probs)
                    ho_short_std = ho_short_pnl.std()
                    ho_short_sharpe = (ho_short_pnl.mean() / ho_short_std * (2190) ** 0.5) if ho_short_std > 0 else 0
                    print(f"  Hold-out AUC:      {ho_short_auc:.4f}")
                    print(f"  Hold-out trades:   {ho_short_trades:,}  ({ho_short_trades/len(ho_short_fwd):.1%} of signals)")
                    print(f"  Hold-out win rate: {ho_short_win_rate:.1%}")
                    print(f"  Hold-out Sharpe:   {ho_short_sharpe:.2f}")
                    best_short["holdout_win_rate"] = round(float(ho_short_win_rate), 4)
                    best_short["holdout_auc"] = round(float(ho_short_auc), 4)
                    best_short["holdout_sharpe"] = round(float(ho_short_sharpe), 4)
                    best_short["holdout_trades"] = int(ho_short_trades)
            except Exception as e:
                print(f"  SHORT hold-out validation failed: {e}")
            short_ready, short_gate_eval = evaluate_gate(
                best_short,
                label="Short model",
                min_auc=SHORT_MIN_LIVE_AUC,
                min_holdout_trades=25,
            )
            xgb_s, lgb_s, fc_s, top_s, short_training_profile = train_final_model(
                df,
                target_col=short_target_col,
                xgb_params=xgb_params,
                lgb_params=lgb_params,
                redesign_context=redesign_context,
                target_profile=target_profile,
            )
            short_model_path = os.path.join(MODEL_DIR, "ensemble_short.pkl")
            with open(short_model_path, "wb") as f:
                pickle.dump((xgb_s, lgb_s, fc_s), f)
            short_meta = {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "pairs": list(df["symbol"].unique()),
                "n_rows": int(len(df)),
                "n_features": int(len(fc_s)),
                "overall_auc": round(float(short_auc), 4),
                "optimal_threshold": float(best_short["threshold"]),
                "win_rate_at_threshold": float(best_short["win_rate"]),
                "ev_at_threshold": float(best_short["ev"]),
                "sharpe_at_threshold": float(best_short["sharpe"]),
                "holdout_win_rate": float(best_short["holdout_win_rate"]) if best_short.get("holdout_win_rate") is not None else None,
                "holdout_auc": float(best_short["holdout_auc"]) if best_short.get("holdout_auc") is not None else None,
                "holdout_sharpe": float(best_short["holdout_sharpe"]) if best_short.get("holdout_sharpe") is not None else None,
                "holdout_trades": int(best_short.get("holdout_trades", 0) or 0),
                "gate_pass": bool(short_ready),
                "gate_evaluation": short_gate_eval,
                "target": short_target_col,
                "fold_results": short_folds,
                "threshold_sweep": short_thresh_results,
                "top_features": [{"feature": f, "importance": round(float(i), 4)}
                                 for f, i in top_s],
                "training_profile": short_training_profile,
            }
            short_meta_path = os.path.join(MODEL_DIR, "model_meta_short.json")
            with open(short_meta_path, "w") as f:
                json.dump(short_meta, f, indent=2)
            print(f"  SHORT model saved: {short_model_path}")
            print(f"  SHORT threshold: {best_short['threshold']:.3f}, win rate: {best_short['win_rate']:.1%}")
        except Exception as _se:
            print(f"  [WARN] SHORT model training failed: {_se}")
    else:
        print(f"  [WARN] {short_target_col} not in data — rebuild features first")

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pairs": list(df["symbol"].unique()),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "overall_auc": round(float(overall_auc), 4),
        "optimal_threshold": float(best_threshold["threshold"]),
        "win_rate_at_threshold": float(best_threshold["win_rate"]),
        "ev_at_threshold": float(best_threshold["ev"]),
        "sharpe_at_threshold": float(best_threshold["sharpe"]),
        "holdout_win_rate": float(best_threshold["holdout_win_rate"]) if best_threshold.get("holdout_win_rate") is not None else None,
        "holdout_auc": float(best_threshold["holdout_auc"]) if best_threshold.get("holdout_auc") is not None else None,
        "holdout_sharpe": float(best_threshold["holdout_sharpe"]) if best_threshold.get("holdout_sharpe") is not None else None,
        "holdout_trades": int(best_threshold.get("holdout_trades", 0) or 0),
        "gate_pass": bool(ready),
        "gate_evaluation": gate_eval,
        "fold_results": fold_results,
        "threshold_sweep": threshold_results,
        "top_features": [{"feature": f, "importance": round(float(i), 4)} for f, i in top_features],
        "optimized_params_source": BEST_PARAMS_FILE if optimized else None,
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
        "training_profile": training_profile,
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if mlflow_run is not None:
        try:
            mlflow.log_params({
                "pairs_count": int(df["symbol"].nunique()),
                "rows": int(len(df)),
                "features": int(len(feature_cols)),
                "optimized": bool(optimized),
                "n_splits": int(n_splits),
            })
            mlflow.log_metric("cv_auc", float(overall_auc))
            mlflow.log_metric("threshold", float(best_threshold["threshold"]))
            mlflow.log_metric("win_rate", float(best_threshold["win_rate"]))
            mlflow.log_metric("ev_at_threshold", float(best_threshold["ev"]))
            mlflow.log_metric("sharpe_at_threshold", float(best_threshold["sharpe"]))
            mlflow.log_metric("holdout_auc", float(meta["holdout_auc"]) if meta.get("holdout_auc") is not None else 0.0)
            mlflow.log_metric("holdout_win_rate", float(meta.get("holdout_win_rate", 0.0)))
            mlflow.log_metric("holdout_sharpe", float(meta.get("holdout_sharpe", 0.0)))
            mlflow.log_metric("holdout_trades", float(meta.get("holdout_trades", 0)))
            mlflow.log_metric("gate_pass", 1.0 if meta.get("gate_pass") else 0.0)
            if os.path.exists(meta_path):
                mlflow.log_artifact(meta_path)
            short_meta_path = os.path.join(MODEL_DIR, "model_meta_short.json")
            if os.path.exists(short_meta_path):
                mlflow.log_artifact(short_meta_path)
        except Exception:
            pass
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print(f"\nModel saved: {model_path}")
    print(f"Metadata:    {meta_path}")
    print(f"\nTop 10 features by importance:")
    for feat, imp in top_features[:10]:
        print(f"  {feat:<35} {imp:.4f}")

    if redesign_context.get("research_hold_active"):
        print("\nTraining research-hold long setup specialists...")
        specialists = train_setup_specialist_models(
            df,
            df_cv=df_cv,
            df_holdout=df_holdout,
            target_profile=target_profile,
            fwd_ret_col=fwd_ret_col,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
            redesign_context=redesign_context,
        )
        specialist_summary_path = os.path.join(MODEL_DIR, "setup_specialists.json")
        with open(specialist_summary_path, "w") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "ready",
                "specialists": specialists,
            }, f, indent=2)
        for row in specialists:
            print(f"  {row['setup']}: {row['status']} ({row.get('positive_rows', 0)} positives)")
        print(f"Specialist summary: {specialist_summary_path}")


def cmd_eval():
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    if not os.path.exists(meta_path):
        print("No model found. Run: python3 quantforge_ml.py train")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Model trained:   {meta['trained_at']}")
    print(f"Pairs:           {', '.join(meta['pairs'])}")
    print(f"Rows / Features: {meta['n_rows']:,} / {meta['n_features']}")
    print(f"AUC:             {meta['overall_auc']:.4f}")
    print(f"Threshold:       {meta['optimal_threshold']:.3f}")
    print(f"Win rate:        {meta['win_rate_at_threshold']:.1%}")
    print(f"Expected value:  {meta['ev_at_threshold']:.5f} per trade")
    print(f"Sharpe:          {meta['sharpe_at_threshold']:.2f}")
    if meta.get("holdout_auc") is not None:
        print(f"Hold-out AUC:    {meta['holdout_auc']:.4f}")
    if meta.get("holdout_win_rate") is not None:
        print(f"Hold-out WR:     {meta['holdout_win_rate']:.1%}")
    if meta.get("holdout_sharpe") is not None:
        print(f"Hold-out Sharpe: {meta['holdout_sharpe']:.2f}")
    if meta.get("holdout_trades") is not None:
        print(f"Hold-out trades: {int(meta['holdout_trades']):,}")
    print(f"Gate:            {'PASS' if meta['gate_pass'] else 'FAIL'}")
    training_profile = meta.get("training_profile") or {}
    if training_profile:
        print(f"Training mode:   {_surface_mode_from_training_profile(training_profile).upper()}")
        if training_profile.get("candidate_type"):
            print(f"Candidate type:  {training_profile.get('candidate_type')}")
    gate_eval = meta.get("gate_evaluation") or {}
    if gate_eval.get("reasons"):
        print("Gate reasons:")
        for reason in gate_eval["reasons"]:
            print(f"  - {reason}")
    print(f"\nTop features:")
    for item in meta.get("top_features", [])[:10]:
        print(f"  {item['feature']:<35} {item['importance']:.4f}")


def cmd_signal(symbol):
    safe = symbol.replace("-", "_")
    csv_path = os.path.join(HISTORICAL_DIR, f"{safe}_1h.csv")
    if not os.path.exists(csv_path):
        print(f"No data for {symbol}. Run: python3 quantforge_data.py fetch {symbol}")
        return
    df = pd.read_csv(csv_path).tail(300)
    result = generate_signal(symbol, df)
    print(f"{symbol}: {result['signal']} (confidence={result['confidence']:.4f})")
    for r in result["reason"]:
        print(f"  {r}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "train":
        pairs_arg = sys.argv[2:] if len(sys.argv) > 2 else None
        cmd_train(pairs_arg)
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "signal":
        symbol_arg = sys.argv[2] if len(sys.argv) > 2 else "BTC-USDT"
        cmd_signal(symbol_arg)
    else:
        print(__doc__)
        sys.exit(1)
