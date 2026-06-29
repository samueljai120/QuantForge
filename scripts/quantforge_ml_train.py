#!/usr/bin/env python3
"""QuantForge ML Model Trainer — BTC-Relative Target

Builds a clean XGBoost + LightGBM ensemble that predicts whether a coin
will outperform BTC over the next 4 hours.

Target: target_btc_rel_4h = (coin_fwd_ret_4h - btc_fwd_ret_4h) > 0

Excludes:
  - All setup_* features (lookahead bias)
  - All fwd_ret_*, target_* columns (future data)
  - symbol, ts (non-predictive)

Output:
  - ensemble.pkl  → (xgb_model, lgb_model, feature_cols)
  - model_meta.json → {gate_pass, trained_at, overall_auc, cv_win_rate, ...}
  - features_all.parquet → combined features for fast scanning
"""

import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.path.join(DATA_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ensemble.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
ALL_FEATURES_PATH = os.path.join(FEATURES_DIR, "features_all.parquet")

# ── Columns to EXCLUDE ─────────────────────────────────────────────
SETUP_COLS = [
    "setup_trend_long_score", "setup_breakout_long_score",
    "setup_rebound_long_score", "setup_trend_short_score",
    "setup_exhaustion_short_score", "setup_trend_long",
    "setup_breakout_long", "setup_rebound_long",
    "setup_trend_short", "setup_exhaustion_short",
]

# All forward-return and target columns (anything that looks ahead)
TARGET_PATTERNS = ["fwd_ret_", "target_", "research_hold"]

NON_FEATURE_COLS = ["symbol", "ts"]

# ── Training parameters ────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,  # Constrained server
    "eval_metric": "logloss",
}

LGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}

CV_SPLITS = 5
MIN_SAMPLES_PER_COIN = 100  # Skip coins with too little data


def load_all_features():
    """Load all coin feature files and BTC features in batches to stay under 12GB RAM."""
    print("[1/5] Loading feature files (batch mode — memory-safe)...")

    # Load BTC first
    btc_path = os.path.join(FEATURES_DIR, "BTC_USDT_features.parquet")
    if not os.path.exists(btc_path):
        print("ERROR: BTC_USDT_features.parquet not found!")
        sys.exit(1)

    btc_df = pd.read_parquet(btc_path)
    btc_df = btc_df.sort_values("ts").reset_index(drop=True)
    print(f"  BTC: {len(btc_df)} rows")

    # Build BTC forward return lookup
    btc_fwd = btc_df[["ts", "fwd_ret_4h"]].copy()
    btc_fwd = btc_fwd.rename(columns={"fwd_ret_4h": "btc_fwd_ret_4h"})

    # Load altcoin files in batches to control memory
    feature_files = sorted([
        f for f in os.listdir(FEATURES_DIR)
        if f.endswith("_features.parquet") and f != "BTC_USDT_features.parquet"
    ])

    BATCH_SIZE = 30  # Process 30 coins at a time
    MAX_ROWS_PER_COIN = 3000  # Only keep last N rows per coin (enough for 4h prediction)
    total_coins = len(feature_files)
    batches_processed = 0
    batches = []

    for start in range(0, total_coins, BATCH_SIZE):
        batch_files = feature_files[start:start + BATCH_SIZE]
        batch_dfs = []
        for fname in batch_files:
            path = os.path.join(FEATURES_DIR, fname)
            try:
                df = pd.read_parquet(path)
                if len(df) >= MIN_SAMPLES_PER_COIN:
                    # Keep only the most recent rows
                    df = df.tail(MAX_ROWS_PER_COIN).copy()
                    batch_dfs.append(df)
            except Exception:
                continue

        if not batch_dfs:
            continue

        batch = pd.concat(batch_dfs, ignore_index=True)
        batch = batch.sort_values(["symbol", "ts"]).reset_index(drop=True)

        # Merge BTC forward return
        batch = batch.merge(btc_fwd, on="ts", how="left")

        # Drop rows with NaN forward returns to save memory
        valid = batch["fwd_ret_4h"].notna() & batch["btc_fwd_ret_4h"].notna()
        batch = batch[valid].copy()

        batches.append(batch)
        batches_processed += 1
        print(f"  Batch {batches_processed}: {start+1}-{min(start+BATCH_SIZE, total_coins)}/{total_coins} "
              f"({len(batch)} rows, RAM ~{batch.memory_usage(deep=True).sum()/1e6:.0f}MB)")

        # Free memory aggressively
        del batch_dfs

    # Combine all batches
    print("  Concatenating all batches...")
    combined = pd.concat(batches, ignore_index=True)
    del batches  # Free memory

    # CRITICAL: TimeSeriesSplit assumes rows are in chronological order. The
    # per-batch sort above is [symbol, ts] (symbol-major), which would make
    # "time-series" folds split by SYMBOL, training on some symbols' future
    # while validating other symbols over the same calendar window — leaking
    # future market state through cross-asset correlation. Sort time-major.
    combined = combined.sort_values(["ts", "symbol"]).reset_index(drop=True)

    print(f"  Combined: {combined.shape[0]} rows × {combined.shape[1]} cols (time-major order)")
    return combined


def build_target(df):
    """Build BTC-relative target: did coin outperform BTC over next 4h?"""
    print("[2/5] Building BTC-relative target...")

    # NaN filtering already done in load_all_features
    # Target: coin return minus BTC return > 0
    df["target_btc_rel_4h"] = (
        (df["fwd_ret_4h"] - df["btc_fwd_ret_4h"]) > 0
    ).astype(int)

    win_rate = df["target_btc_rel_4h"].mean()
    print(f"  {len(df)} rows, target distribution: {win_rate:.1%} outperform BTC")
    return df


def build_features(df):
    """Select clean feature columns (no leaks)."""
    print("[3/5] Selecting clean features...")

    # Identify columns to exclude
    exclude = set(SETUP_COLS + NON_FEATURE_COLS)

    # Exclude anything matching target patterns
    for col in df.columns:
        for pattern in TARGET_PATTERNS:
            if pattern in col.lower():
                exclude.add(col)
                break

    # Also exclude the BTC forward return (it's not a feature for the coin)
    exclude.add("btc_fwd_ret_4h")

    feature_cols = [c for c in df.columns if c not in exclude]

    # Keep only numeric columns
    numeric_cols = []
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            numeric_cols.append(col)

    print(f"  Feature columns: {len(numeric_cols)} (excluded {len(exclude)})")
    return numeric_cols


def train_and_evaluate(df, feature_cols):
    """Train XGBoost + LightGBM with time-series CV and evaluate."""
    print("[4/5] Training models with time-series cross-validation...")

    target_col = "target_btc_rel_4h"
    X = np.nan_to_num(df[feature_cols].fillna(0).values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[target_col].values

    # Embargo between train and validation: the 4h forward-return target means
    # the last rows of each train fold overlap the first validation hours.
    # With time-major ordering there are ~n_symbols rows per hourly timestamp,
    # so a 48-hour purge is 48 * n_symbols rows. Capped so small datasets
    # still split cleanly.
    n_symbols = int(df["symbol"].nunique()) if "symbol" in df.columns else 1
    embargo_rows = min(48 * max(1, n_symbols), max(0, len(X) // (CV_SPLITS + 2)))
    print(f"  Embargo: {embargo_rows} rows (~48h x {n_symbols} symbols)")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=embargo_rows)

    xgb_wins = []
    lgb_wins = []
    ensemble_wins = []
    xgb_aucs = []
    lgb_aucs = []
    ensemble_aucs = []

    fold = 0
    for train_idx, val_idx in tscv.split(X):
        fold += 1
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train XGBoost
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
        xgb_pred_class = (xgb_preds >= 0.5).astype(int)

        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        lgb_pred_class = (lgb_preds >= 0.5).astype(int)

        # Ensemble: average probability
        ensemble_preds = (xgb_preds + lgb_preds) / 2
        ensemble_pred_class = (ensemble_preds >= 0.5).astype(int)

        xgb_wr = accuracy_score(y_val, xgb_pred_class)
        lgb_wr = accuracy_score(y_val, lgb_pred_class)
        ens_wr = accuracy_score(y_val, ensemble_pred_class)

        xgb_auc = roc_auc_score(y_val, xgb_preds)
        lgb_auc = roc_auc_score(y_val, lgb_preds)
        ens_auc = roc_auc_score(y_val, ensemble_preds)

        xgb_wins.append(xgb_wr)
        lgb_wins.append(lgb_wr)
        ensemble_wins.append(ens_wr)
        xgb_aucs.append(xgb_auc)
        lgb_aucs.append(lgb_auc)
        ensemble_aucs.append(ens_auc)

        print(f"  Fold {fold}: XGB WR={xgb_wr:.3f} AUC={xgb_auc:.3f}  "
              f"LGB WR={lgb_wr:.3f} AUC={lgb_auc:.3f}  "
              f"Ensemble WR={ens_wr:.3f} AUC={ens_auc:.3f}")

    print(f"\n  ── CV Summary ──")
    print(f"  XGBoost:      WR={np.mean(xgb_wins):.3f}±{np.std(xgb_wins):.3f}  "
          f"AUC={np.mean(xgb_aucs):.3f}±{np.std(xgb_aucs):.3f}")
    print(f"  LightGBM:     WR={np.mean(lgb_wins):.3f}±{np.std(lgb_wins):.3f}  "
          f"AUC={np.mean(lgb_aucs):.3f}±{np.std(lgb_aucs):.3f}")
    print(f"  Ensemble:     WR={np.mean(ensemble_wins):.3f}±{np.std(ensemble_wins):.3f}  "
          f"AUC={np.mean(ensemble_aucs):.3f}±{np.std(ensemble_aucs):.3f}")

    # Calculate EV (expected value per trade)
    # For each CV fold, compute: win_rate * avg_win - loss_rate * avg_loss
    evs = []
    for fold_idx in range(CV_SPLITS):
        _, val_idx = list(tscv.split(X))[fold_idx]
        y_val = y[val_idx]
        # Get ensemble predictions for this fold
        fold_start = sum(len(list(tscv.split(X))[i][1]) for i in range(fold_idx))
        fold_end = fold_start + len(val_idx)
        # Re-train on full to get final ensemble preds for each sample
        # For simplicity, compute EV using overall distribution
    avg_win_pct = df[df[target_col] == 1]["fwd_ret_4h"].mean()
    avg_loss_pct = abs(df[df[target_col] == 0]["fwd_ret_4h"].mean())
    base_rate = y.mean()
    ev_base = base_rate * avg_win_pct - (1 - base_rate) * avg_loss_pct

    # Model EV estimate: at 50% threshold
    model_wr = np.mean(ensemble_wins)
    model_ev = model_wr * avg_win_pct - (1 - model_wr) * avg_loss_pct

    print(f"  Base rate: {base_rate:.3f}, Avg win: {avg_win_pct:.4%}, "
          f"Avg loss: {avg_loss_pct:.4%}")
    print(f"  EV base: {ev_base:.4%}, EV model: {model_ev:.4%}")

    return {
        "cv_win_rate": float(np.mean(ensemble_wins)),
        "cv_win_rate_std": float(np.std(ensemble_wins)),
        "cv_auc": float(np.mean(ensemble_aucs)),
        "cv_auc_std": float(np.std(ensemble_aucs)),
        "xgb_win_rate": float(np.mean(xgb_wins)),
        "lgb_win_rate": float(np.mean(lgb_wins)),
        "base_rate": float(base_rate),
        "avg_win_pct": float(avg_win_pct),
        "avg_loss_pct": float(avg_loss_pct),
        "ev_model": float(model_ev),
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
    }


def train_final_model(df, feature_cols):
    """Train final ensemble on all data."""
    print("[5/5] Training final ensemble on all data...")

    target_col = "target_btc_rel_4h"
    X = np.nan_to_num(df[feature_cols].fillna(0).values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[target_col].values

    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X, y)

    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(X, y)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((xgb_model, lgb_model, feature_cols), f)

    print(f"  Model saved: {MODEL_PATH}")
    return xgb_model, lgb_model


def save_meta(metrics, feature_cols):
    """Save model metadata."""
    meta = {
        "gate_pass": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": "target_btc_rel_4h",
        "overall_auc": metrics["cv_auc"],
        "optimal_threshold": 0.50,
        "cv_win_rate": metrics["cv_win_rate"],
        "cv_win_rate_std": metrics["cv_win_rate_std"],
        "base_rate": metrics["base_rate"],
        "ev_model": metrics["ev_model"],
        "n_samples": metrics["n_samples"],
        "n_features": metrics["n_features"],
        "excluded_setup_cols": len(SETUP_COLS),
        "excluded_target_cols": "all fwd_ret_*, target_* patterns",
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Meta saved: {META_PATH}")
    print(f"\n  ══ Model Ready ══")
    print(f"  CV Win Rate:  {metrics['cv_win_rate']:.1%} ±{metrics['cv_win_rate_std']:.1%}")
    print(f"  CV AUC:       {metrics['cv_auc']:.3f} ±{metrics['cv_auc_std']:.3f}")
    print(f"  EV per trade: {metrics['ev_model']:.4%}")
    gate = "PASS" if meta["gate_pass"] else "FAIL"
    print(f"  Gate:         {gate}")


def build_combined_features(df):
    """Save combined features_all.parquet for fast scanning."""
    print("  Building features_all.parquet...")
    df.to_parquet(ALL_FEATURES_PATH, index=False)
    print(f"  Saved: {ALL_FEATURES_PATH} ({len(df)} rows)")


def main():
    start = time.time()

    # Step 1: Load all features
    df = load_all_features()

    # Step 2: Build BTC-relative target
    df = build_target(df)

    # Step 3: Select clean features
    feature_cols = build_features(df)

    # Step 4: Cross-validate
    metrics = train_and_evaluate(df, feature_cols)

    # Gate check: CV win rate must be > base rate
    if metrics["cv_win_rate"] <= metrics["base_rate"]:
        print("\n   GATE FAILED: CV win rate not better than random")
        meta = {
            "gate_pass": False,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "reason": f"CV WR {metrics['cv_win_rate']:.3f} <= base {metrics['base_rate']:.3f}",
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        sys.exit(1)

    # Step 5: Train final model + save
    train_final_model(df, feature_cols)
    save_meta(metrics, feature_cols)

    # Build combined features for fast scanning
    build_combined_features(df)

    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
