#!/usr/bin/env python3
"""QuantForge — Optuna/MLflow optimizer for ML model tuning.

Usage:
    python3 quantforge_optimize.py optimize
    python3 quantforge_optimize.py optimize 25
    python3 quantforge_optimize.py best
"""

import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_ml import (
    FEATURES_DIR,
    MODEL_DIR,
    TARGET_COL,
    FWD_RET_COL,
    ROUND_TRIP_COST,
    find_optimal_threshold,
    get_feature_cols,
    train_final_model,
    walk_forward_eval,
)

try:
    import optuna
except Exception:
    optuna = None

try:
    import mlflow
except Exception:
    mlflow = None


OPT_DIR = os.path.join(cfg.data, "quantforge", "optimization")
BEST_FILE = os.path.join(OPT_DIR, "best-params.json")
STUDY_FILE = os.path.join(OPT_DIR, "study-summary.json")
MLRUNS_DIR = os.path.join(OPT_DIR, "mlruns")
os.makedirs(OPT_DIR, exist_ok=True)

MIN_HOLDOUT_TRADES_TARGET = 25
MIN_CV_TRADES_TARGET = 80


def _require_production_runtime(script_name):
    if hasattr(cfg, "assert_production_runtime"):
        cfg.assert_production_runtime(script_name)
        return
    if hasattr(cfg, "require_production_runtime"):
        cfg.require_production_runtime(script_name)


_require_production_runtime("quantforge_optimize.py")


def load_features():
    all_path = os.path.join(FEATURES_DIR, "features_all.parquet")
    if not os.path.exists(all_path):
        raise FileNotFoundError("No features found. Run: python3 quantforge_features.py build")
    return pd.read_parquet(all_path)


def split_cv_holdout(df):
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    holdout_rows = []
    cv_rows = []
    for _, grp in df.groupby("symbol"):
        n = len(grp)
        split = int(n * 0.80)
        cv_rows.append(grp.iloc[:split])
        holdout_rows.append(grp.iloc[split:])
    df_cv = pd.concat(cv_rows).sort_values("ts").reset_index(drop=True)
    df_holdout = pd.concat(holdout_rows).sort_values("ts").reset_index(drop=True)
    return df_cv, df_holdout


def evaluate_holdout(df_cv, df_holdout, threshold, *, xgb_params=None, lgb_params=None):
    feature_cols = get_feature_cols(df_holdout)
    df_cv_clean = df_cv.dropna(subset=feature_cols + [TARGET_COL]).sort_values("ts")
    df_ho_clean = df_holdout.dropna(subset=feature_cols + [TARGET_COL, FWD_RET_COL]).sort_values("ts")
    if len(df_cv_clean) == 0 or len(df_ho_clean) == 0:
        return {
            "holdout_trades": 0,
            "holdout_win_rate": 0.0,
            "holdout_sharpe": 0.0,
            "holdout_avg_net_edge_bps": -999.0,
        }

    xgb_model, lgb_model, used_cols, _ = train_final_model(
        df_cv_clean, target_col=TARGET_COL, xgb_params=xgb_params, lgb_params=lgb_params
    )
    X_holdout = df_ho_clean[used_cols].fillna(0.0).values
    probs = (xgb_model.predict_proba(X_holdout)[:, 1] + lgb_model.predict_proba(X_holdout)[:, 1]) / 2
    mask = probs >= threshold
    holdout_trades = int(mask.sum())
    if holdout_trades == 0:
        return {
            "holdout_trades": 0,
            "holdout_win_rate": 0.0,
            "holdout_sharpe": 0.0,
            "holdout_avg_net_edge_bps": -999.0,
        }

    selected = df_ho_clean.loc[mask, FWD_RET_COL].values
    net_rets = selected - ROUND_TRIP_COST
    win_rate = float((net_rets > 0).mean())
    std = float(net_rets.std())
    # Annualize by actual trade frequency (see quantforge_ml.py) — flat
    # sqrt(2190) assumed a trade every 4h slot and inflated Sharpe.
    trade_frac = holdout_trades / max(1, len(df_ho_clean))
    trades_per_year = max(1.0, 2190.0 * trade_frac)
    sharpe = float(net_rets.mean() / std * math.sqrt(trades_per_year)) if std > 0 else 0.0
    avg_net_edge_bps = float(net_rets.mean() * 10000)
    return {
        "holdout_trades": holdout_trades,
        "holdout_win_rate": win_rate,
        "holdout_sharpe": sharpe,
        "holdout_avg_net_edge_bps": avg_net_edge_bps,
    }


def make_trial_params(trial):
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 450, step=50),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 7),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.02, 0.10, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.7, 1.0),
    }
    lgb_params = {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 200, 450, step=50),
        "max_depth": trial.suggest_int("lgb_max_depth", 3, 7),
        "learning_rate": trial.suggest_float("lgb_learning_rate", 0.02, 0.10, log=True),
        "subsample": trial.suggest_float("lgb_subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.7, 1.0),
    }
    n_splits = trial.suggest_int("n_splits", 4, 5)
    return xgb_params, lgb_params, n_splits


def objective_factory(df_cv, df_holdout):
    def objective(trial):
        xgb_params, lgb_params, n_splits = make_trial_params(trial)
        probs, labels, fwd_rets, _ = walk_forward_eval(
            df_cv,
            n_splits=n_splits,
            target_col=TARGET_COL,
            fwd_ret_col=FWD_RET_COL,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
        )
        best_threshold, _ = find_optimal_threshold(probs, labels, fwd_rets)
        cv_trade_count = int(best_threshold.get("trades", 0))
        holdout = evaluate_holdout(
            df_cv,
            df_holdout,
            best_threshold["threshold"],
            xgb_params=xgb_params,
            lgb_params=lgb_params,
        )
        holdout_trade_count = int(holdout["holdout_trades"])
        cv_trade_penalty = max(0, MIN_CV_TRADES_TARGET - cv_trade_count) * 2.0
        holdout_trade_penalty = max(0, MIN_HOLDOUT_TRADES_TARGET - holdout_trade_count) * 8.0
        score = (
            holdout["holdout_avg_net_edge_bps"]
            + (holdout["holdout_win_rate"] * 100.0)
            + (holdout["holdout_sharpe"] * 5.0)
            - cv_trade_penalty
            - holdout_trade_penalty
        )

        trial.set_user_attr("threshold", float(best_threshold["threshold"]))
        trial.set_user_attr("cv_trades", cv_trade_count)
        trial.set_user_attr("holdout_trades", int(holdout["holdout_trades"]))
        trial.set_user_attr("holdout_win_rate", float(holdout["holdout_win_rate"]))
        trial.set_user_attr("holdout_sharpe", float(holdout["holdout_sharpe"]))
        trial.set_user_attr("holdout_avg_net_edge_bps", float(holdout["holdout_avg_net_edge_bps"]))
        trial.set_user_attr("cv_trade_penalty", float(cv_trade_penalty))
        trial.set_user_attr("holdout_trade_penalty", float(holdout_trade_penalty))

        if mlflow is not None:
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params({**xgb_params, **lgb_params, "n_splits": n_splits})
                mlflow.log_metric("score", score)
                mlflow.log_metric("threshold", float(best_threshold["threshold"]))
                mlflow.log_metric("cv_trades", cv_trade_count)
                mlflow.log_metric("holdout_trades", int(holdout["holdout_trades"]))
                mlflow.log_metric("holdout_win_rate", float(holdout["holdout_win_rate"]))
                mlflow.log_metric("holdout_sharpe", float(holdout["holdout_sharpe"]))
                mlflow.log_metric("holdout_avg_net_edge_bps", float(holdout["holdout_avg_net_edge_bps"]))
                mlflow.log_metric("cv_trade_penalty", float(cv_trade_penalty))
                mlflow.log_metric("holdout_trade_penalty", float(holdout_trade_penalty))
        return score

    return objective


def cmd_optimize(n_trials=20):
    if optuna is None:
        print("Optuna is not installed on this runtime.")
        return

    if mlflow is not None:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment("quantforge-optimization")

    df = load_features()
    df_cv, df_holdout = split_cv_holdout(df)
    print(f"Optimizing on {len(df_cv):,} CV rows with {len(df_holdout):,} holdout rows")
    print(f"Pairs: {df['symbol'].nunique()} | Features: {len(get_feature_cols(df))}")

    study = optuna.create_study(direction="maximize", study_name="quantforge-optimization")
    objective = objective_factory(df_cv, df_holdout)

    if mlflow is not None:
        with mlflow.start_run(run_name=f"study-{datetime.utcnow().isoformat()}"):
            mlflow.log_params({
                "n_trials": int(n_trials),
                "cv_rows": int(len(df_cv)),
                "holdout_rows": int(len(df_holdout)),
                "pairs": int(df["symbol"].nunique()),
            })
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            mlflow.log_metric("best_score", float(study.best_value))
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    xgb_params, lgb_params, n_splits = make_trial_params(best)
    payload = {
        "optimized_at": datetime.utcnow().isoformat(),
        "best_score": float(best.value),
        "threshold": float(best.user_attrs.get("threshold", 0.0)),
        "cv_trades": int(best.user_attrs.get("cv_trades", 0)),
        "holdout_trades": int(best.user_attrs.get("holdout_trades", 0)),
        "holdout_win_rate": float(best.user_attrs.get("holdout_win_rate", 0.0)),
        "holdout_sharpe": float(best.user_attrs.get("holdout_sharpe", 0.0)),
        "holdout_avg_net_edge_bps": float(best.user_attrs.get("holdout_avg_net_edge_bps", 0.0)),
        "cv_trade_penalty": float(best.user_attrs.get("cv_trade_penalty", 0.0)),
        "holdout_trade_penalty": float(best.user_attrs.get("holdout_trade_penalty", 0.0)),
        "min_cv_trades_target": MIN_CV_TRADES_TARGET,
        "min_holdout_trades_target": MIN_HOLDOUT_TRADES_TARGET,
        "n_splits": int(n_splits),
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
    }
    with open(BEST_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    with open(STUDY_FILE, "w") as f:
        json.dump({
            "optimized_at": payload["optimized_at"],
            "n_trials": int(n_trials),
            "best_value": float(study.best_value),
            "best_params": best.params,
        }, f, indent=2)

    print("QuantForge optimization complete")
    print(f"  best score:         {payload['best_score']:.2f}")
    print(f"  threshold:          {payload['threshold']:.3f}")
    print(f"  cv trades:          {payload['cv_trades']}")
    print(f"  holdout trades:     {payload['holdout_trades']}")
    print(f"  holdout win rate:   {payload['holdout_win_rate']:.1%}")
    print(f"  holdout sharpe:     {payload['holdout_sharpe']:.2f}")
    print(f"  avg net edge:       {payload['holdout_avg_net_edge_bps']:+.2f} bps")
    print(f"  best params file:   {BEST_FILE}")
    print(f"  study summary file: {STUDY_FILE}")


def cmd_best():
    if not os.path.exists(BEST_FILE):
        print("No optimization results found. Run: python3 quantforge_optimize.py optimize")
        return
    with open(BEST_FILE) as f:
        payload = json.load(f)
    print("QuantForge best optimization result")
    print(f"  optimized at:       {payload['optimized_at']}")
    print(f"  best score:         {payload['best_score']:.2f}")
    print(f"  threshold:          {payload['threshold']:.3f}")
    print(f"  cv trades:          {payload.get('cv_trades', 0)}")
    print(f"  holdout trades:     {payload['holdout_trades']}")
    print(f"  holdout win rate:   {payload['holdout_win_rate']:.1%}")
    print(f"  holdout sharpe:     {payload['holdout_sharpe']:.2f}")
    print(f"  avg net edge:       {payload['holdout_avg_net_edge_bps']:+.2f} bps")
    print(f"  xgb params:         {payload['xgb_params']}")
    print(f"  lgb params:         {payload['lgb_params']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "best"
    if cmd == "optimize":
        trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        cmd_optimize(trials)
    elif cmd == "best":
        cmd_best()
    else:
        print(__doc__)
        sys.exit(1)
