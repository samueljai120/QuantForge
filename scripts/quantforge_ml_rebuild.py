#!/usr/bin/env python3
"""QuantForge ML Deep Rebuild — research-only trainer (2026-06-12).

Rebuild scope per governance research_hold candidate: labels, targets,
features, evaluation. The old pipeline (quantforge_ml_train.py) produced
AUC 0.532 — no edge — because of three design flaws this rebuild fixes:

1. LABELS: old target was sign(coin_4h_ret - btc_4h_ret). A +0.01% and a
   +8% outperformance got the same label, and ~half of all labels sat
   within fee-noise of zero. Rebuild: fee-hurdled dead-zone label —
   POS only if relative return clears max(round-trip fees, 0.25 x 4h ATR),
   NEG only if it loses by the same hurdle, ambiguous middle DROPPED.
   The model now learns only moves that are tradeable after costs.

2. FEATURES: the scanner trades the cross-section (top-N coins at each
   scan), but the old features were purely per-coin time-series. Rebuild
   adds per-timestamp cross-sectional percentile ranks of the key signals,
   so the model sees "how does this coin rank vs the others right now".

3. EVALUATION: old gate was "CV accuracy > base rate" (trivially passable)
   and EV was computed over ALL rows, not the rows the scanner would pick.
   Rebuild: selection EV — mean fee-adjusted relative return of the
   top-decile predictions and of the top-3-per-timestamp picks (mirrors the
   live scanner). Hard gates on AUC consistency AND selection EV, verified
   on an untouched final time holdout.

OUTPUTS (research-only — NEVER touches the live model files):
  model/rebuild_ensemble.pkl   (xgb, lgb, feature_cols)
  model/rebuild_verdict.json   gate_pass + all metrics

(pickle is required here: the live scanner already loads ensemble.pkl via
pickle and xgb/lgb model objects are not JSON-serializable; files are
locally generated, never loaded from untrusted sources.)

Promotion to live (ensemble.pkl) remains a separate, explicit operator /
governance action gated on this verdict.
"""

import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from quantforge_target_profiles import apply_research_rebuild_slice

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.path.join(DATA_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Overridable so the research director can run experiment arms without
# clobbering each other's outputs.
REBUILD_MODEL_PATH = os.environ.get(
    "QF_MODEL_PATH", os.path.join(MODEL_DIR, "rebuild_ensemble.pkl"))
VERDICT_PATH = os.environ.get(
    "QF_VERDICT_PATH", os.path.join(MODEL_DIR, "rebuild_verdict.json"))

# ── Label parameters (env-overridable experiment knobs) ────────────
FEE_ROUND_TRIP = 0.002          # KuCoin spot taker 0.1% x 2 sides
VOL_HURDLE_MULT = float(os.environ.get("QF_HURDLE_MULT", "0.25"))
HORIZON = os.environ.get("QF_HORIZON", "4h")        # "4h" or "8h"
if HORIZON not in ("4h", "8h"):
    HORIZON = "4h"
OBJECTIVE_MODE = str(os.environ.get("QF_OBJECTIVE_MODE", "btc_relative") or "").strip().lower()
if OBJECTIVE_MODE not in {"btc_relative", "absolute_edge"}:
    OBJECTIVE_MODE = "btc_relative"
FWD_COL = f"fwd_ret_{HORIZON}"
BTC_FWD_COL = f"btc_{FWD_COL}"
# hourly atr_norm -> horizon ATR proxy (sqrt of horizon hours)
ATR_4H_SCALE = 2.0 if HORIZON == "4h" else 2.83

# ── Data limits (tune for your host RAM) ──────────────────────────
MAX_ROWS_PER_COIN = int(os.environ.get("QF_MAX_ROWS", "3000"))
MIN_SAMPLES_PER_COIN = 100
BATCH_SIZE = 30

# History-mode knobs (for the "does old data help?" experiment arm):
# QF_NAN_PREFUNDING=1 -> deriv_* values before the collector epoch become NaN
#   (they are stored as 0.0, which the model would misread as "funding is zero")
# QF_KEEP_NAN=1       -> skip fillna(0); XGB/LGB route NaN natively
# QF_HALFLIFE_DAYS=N  -> exponential recency sample weights (0 = off)
NAN_PREFUNDING = os.environ.get("QF_NAN_PREFUNDING") == "1"
KEEP_NAN = os.environ.get("QF_KEEP_NAN") == "1"
HALFLIFE_DAYS = float(os.environ.get("QF_HALFLIFE_DAYS", "0"))
FUNDING_EPOCH = "2026-04-05"    # derivatives collector came online
# QF_MACRO=1 -> point-in-time join TradFi macro features (VIX/SPX/DXY/US10Y/
# gold/oil + 1d/5d changes) from the macro collector. Tests whether cross-asset
# risk appetite adds edge beyond the crypto-internal features.
USE_MACRO = os.environ.get("QF_MACRO") == "1"
SYMBOL_ALLOWLIST_RAW = os.environ.get("QF_SYMBOL_ALLOWLIST", "").strip()
SLICE_PROFILE_RAW = os.environ.get("QF_SLICE_PROFILE", "").strip()

# ── Exclusions (leakage + non-features) ────────────────────────────
TARGET_PATTERNS = ["fwd_ret_", "target_", "setup_", "research_hold"]
NON_FEATURE_COLS = ["symbol", "ts"]

# Cross-sectional rank features: per-timestamp percentile rank of these
RANK_COLS = [
    "ret_4h", "ret_24h", "vol_ratio", "rsi_14_norm",
    "rel_btc_ret_4h", "deriv_funding_rate", "turnover_ratio", "atr_norm",
]

# ── Model parameters (n_jobs=1: constrained server) ────────────────
XGB_PARAMS = {
    "n_estimators": 150, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
    "reg_alpha": 0.5, "reg_lambda": 1.0, "random_state": 42,
    "n_jobs": 1, "eval_metric": "logloss", "tree_method": "hist",
}
LGB_PARAMS = {
    "n_estimators": 150, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 30,
    "reg_alpha": 0.5, "reg_lambda": 1.0, "random_state": 42,
    "n_jobs": 1, "verbose": -1,
}

CV_SPLITS = 5
HOLDOUT_FRAC = 0.15             # final time slice, untouched until the end
EMBARGO_HOURS = 48

# ── Hard gates ─────────────────────────────────────────────────────
GATE_MEAN_AUC = 0.55
GATE_MIN_FOLD_AUC = 0.52
GATE_EV_FOLDS_POSITIVE = 4      # of 5: top-decile EV after fees > 0
GATE_HOLDOUT_AUC = 0.53
GATE_MIN_LABELED_ROWS = int(os.environ.get("QF_MIN_LABELED_ROWS", "50000"))


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def _normalize_symbol_allowlist(raw: str) -> set[str]:
    allow = set()
    for part in str(raw or "").split(","):
        token = part.strip().upper()
        if not token:
            continue
        if "-" not in token:
            token = f"{token}-USDT"
        allow.add(token)
    return allow


def _is_excluded_training_col(col: str) -> bool:
    lower = str(col or "").lower()
    if lower.startswith("setup_") and lower.endswith("_score"):
        return False
    return any(p in lower for p in TARGET_PATTERNS)


def _objective_return_arrays(
    fwd_ret,
    btc_fwd_ret,
    *,
    objective_mode: str,
):
    if str(objective_mode or "").strip().lower() == "absolute_edge":
        return np.asarray(fwd_ret, dtype=np.float32)
    return np.asarray(fwd_ret, dtype=np.float32) - np.asarray(btc_fwd_ret, dtype=np.float32)


def load_features():
    """Batch-load coin features; drop excluded cols early; downcast to f32."""
    log("[1/6] Loading features (memory-safe batches)...")
    btc_path = os.path.join(FEATURES_DIR, "BTC_USDT_features.parquet")
    btc_df = pd.read_parquet(btc_path).sort_values("ts")
    btc_fwd = btc_df[["ts", FWD_COL]].rename(columns={FWD_COL: BTC_FWD_COL})

    files = sorted(f for f in os.listdir(FEATURES_DIR)
                   if f.endswith("_features.parquet") and f != "BTC_USDT_features.parquet")

    def slim(df):
        # keep the label-input forward return + atr_norm + all clean features
        drop = [c for c in df.columns if _is_excluded_training_col(c) and c != FWD_COL]
        df = df.drop(columns=drop, errors="ignore")
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        return df

    batches = []
    for start in range(0, len(files), BATCH_SIZE):
        dfs = []
        for fname in files[start:start + BATCH_SIZE]:
            try:
                df = pd.read_parquet(os.path.join(FEATURES_DIR, fname))
                if len(df) >= MIN_SAMPLES_PER_COIN:
                    dfs.append(slim(df.tail(MAX_ROWS_PER_COIN)))
            except Exception:
                continue
        if not dfs:
            continue
        batch = pd.concat(dfs, ignore_index=True).merge(btc_fwd, on="ts", how="left")
        batch = batch[batch[FWD_COL].notna() & batch[BTC_FWD_COL].notna()]
        batches.append(batch)
        log(f"  batch {start // BATCH_SIZE + 1}: rows={len(batch)}")
        del dfs

    combined = pd.concat(batches, ignore_index=True)
    del batches
    # time-major: TimeSeriesSplit folds must split by calendar time
    combined = combined.sort_values(["ts", "symbol"]).reset_index(drop=True)

    if NAN_PREFUNDING:
        epoch = int(pd.Timestamp(FUNDING_EPOCH, tz="UTC").timestamp())
        deriv_cols = [c for c in combined.columns if c.startswith("deriv_")]
        mask = combined["ts"] < epoch
        combined.loc[mask, deriv_cols] = np.nan
        log(f"  NaN'd {len(deriv_cols)} deriv_* cols on {int(mask.sum())} "
            f"pre-{FUNDING_EPOCH} rows (0.0 there means 'missing', not 'zero funding')")

    log(f"  combined: {combined.shape[0]} rows x {combined.shape[1]} cols")
    return combined


def build_labels(df):
    """Fee-hurdled dead-zone label on the configured training objective."""
    log(f"[2/6] Building fee-hurdled dead-zone labels (objective={OBJECTIVE_MODE}, "
        f"horizon={HORIZON}, hurdle={VOL_HURDLE_MULT} x ATR)...")
    objective_ret = _objective_return_arrays(
        df[FWD_COL].values,
        df[BTC_FWD_COL].values,
        objective_mode=OBJECTIVE_MODE,
    )
    atr4h = (df.get("atr_norm", pd.Series(0.01, index=df.index))
             .fillna(0.01).clip(lower=0.001) * ATR_4H_SCALE)
    hurdle = np.maximum(FEE_ROUND_TRIP, VOL_HURDLE_MULT * atr4h)

    df["objective_ret"] = objective_ret.astype(np.float32)
    df["label"] = np.where(objective_ret > hurdle, 1, np.where(objective_ret < -hurdle, 0, -1)).astype(np.int8)

    n_total = len(df)
    df = df[df["label"] >= 0].reset_index(drop=True)
    kept = len(df) / max(n_total, 1)
    log(f"  kept {len(df)}/{n_total} rows ({kept:.1%}) — dead zone dropped "
        f"{1 - kept:.1%} of noise labels; base rate {df['label'].mean():.3f}")
    return df


def add_rank_features(df):
    """Per-timestamp cross-sectional percentile ranks of key signals."""
    log("[3/6] Adding cross-sectional rank features...")
    added = []
    for col in RANK_COLS:
        if col in df.columns:
            rc = f"xrank_{col}"
            df[rc] = (df.groupby("ts")[col]
                        .rank(pct=True, na_option="keep")
                        .astype(np.float32))
            added.append(rc)
    log(f"  added {len(added)}: {added}")
    return df


def select_features(df):
    exclude = set(NON_FEATURE_COLS) | {BTC_FWD_COL, FWD_COL, "objective_ret", "label"}
    for c in df.columns:
        if _is_excluded_training_col(c):
            exclude.add(c)
    cols = [c for c in df.columns
            if c not in exclude and df[c].dtype in (np.float32, np.float64, np.int64, np.int32, np.int8)]
    log(f"  feature columns: {len(cols)}")
    return cols


def selection_ev(val_df, preds, fee=FEE_ROUND_TRIP):
    """EV of what the scanner would actually trade.

    Returns (top-decile EV, top-3-per-timestamp EV), both fee-adjusted
    BTC-relative mean returns.
    """
    v = val_df.copy()
    v["pred"] = preds
    # global top decile within the fold
    k = max(1, int(len(v) * 0.10))
    top = v.nlargest(k, "pred")
    ev_decile = float((top["objective_ret"] - fee).mean())
    # top-3 per timestamp (mirrors live scanner picks)
    top3 = (v.sort_values("pred", ascending=False)
              .groupby("ts").head(3))
    ev_top3 = float((top3["objective_ret"] - fee).mean())
    return ev_decile, ev_top3


def to_matrix(frame, cols):
    X = frame[cols].values.astype(np.float32)
    if KEEP_NAN:
        X[np.isinf(X)] = np.nan   # NaN routed natively by XGB/LGB
        return X
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit_ensemble(X_tr, y_tr, w=None):
    xm = xgb.XGBClassifier(**XGB_PARAMS)
    xm.fit(X_tr, y_tr, sample_weight=w)
    lm = lgb.LGBMClassifier(**LGB_PARAMS)
    lm.fit(X_tr, y_tr, sample_weight=w)
    return xm, lm


def predict_ensemble(xm, lm, X):
    return (xm.predict_proba(X)[:, 1] + lm.predict_proba(X)[:, 1]) / 2


def main():
    t0 = time.time()
    df = load_features()
    slice_summary = {}

    symbol_allowlist = _normalize_symbol_allowlist(SYMBOL_ALLOWLIST_RAW)
    if symbol_allowlist:
        before = len(df)
        df = df[df["symbol"].isin(symbol_allowlist)].reset_index(drop=True)
        log(f"  symbol allowlist {sorted(symbol_allowlist)}: {before} -> {len(df)} rows")

    # Optional: restrict to data since a cutoff date (e.g. when the
    # derivatives collector came online — funding/OI features are zero
    # before ~2026-04-05, and the model only shows edge where they exist).
    if len(sys.argv) > 1:
        cutoff = int(pd.Timestamp(sys.argv[1], tz="UTC").timestamp())
        before = len(df)
        df = df[df["ts"] >= cutoff].reset_index(drop=True)
        log(f"  --since {sys.argv[1]}: {before} -> {len(df)} rows")

    if SLICE_PROFILE_RAW:
        before = len(df)
        df, slice_summary = apply_research_rebuild_slice(
            df,
            slice_profile=SLICE_PROFILE_RAW,
        )
        log(f"  slice profile {SLICE_PROFILE_RAW}: {before} -> {len(df)} rows")

    df = build_labels(df)

    if len(df) < GATE_MIN_LABELED_ROWS:
        verdict = {
            "gate_pass": False,
            "reason": f"only {len(df)} labeled rows (< {GATE_MIN_LABELED_ROWS})",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_labeled_rows": int(len(df)),
            "symbol_allowlist": sorted(symbol_allowlist),
            "slice_profile": str(SLICE_PROFILE_RAW or "").strip().lower(),
            "slice_profile_summary": slice_summary,
        }
        json.dump(verdict, open(VERDICT_PATH, "w"), indent=2)
        log(f" {verdict['reason']}")
        sys.exit(1)

    df = add_rank_features(df)

    if USE_MACRO:
        try:
            from quantforge_macro_collector import merge_macro_features
            df, macro_added = merge_macro_features(df)
            cov = df[macro_added[0]].notna().mean() if macro_added else 0.0
            log(f"  macro features: +{len(macro_added)} cols, "
                f"{cov:.1%} of rows have macro (rest NaN, routed natively)")
        except Exception as e:
            log(f"   macro merge failed ({e}) — continuing without macro")

    feature_cols = select_features(df)

    # ── time-based split: CV window vs untouched holdout ───────────
    ts_sorted = np.sort(df["ts"].unique())
    cut_ts = ts_sorted[int(len(ts_sorted) * (1 - HOLDOUT_FRAC))]
    embargo_s = EMBARGO_HOURS * 3600
    cv_df = df[df["ts"] < cut_ts].reset_index(drop=True)
    ho_df = df[df["ts"] >= cut_ts + embargo_s].reset_index(drop=True)
    log(f"[4/6] CV window: {len(cv_df)} rows | holdout: {len(ho_df)} rows "
        f"(embargo {EMBARGO_HOURS}h at boundary)")

    X = to_matrix(cv_df, feature_cols)
    y = cv_df["label"].values

    weights = None
    if HALFLIFE_DAYS > 0:
        age_days = (cv_df["ts"].max() - cv_df["ts"]).values / 86400.0
        weights = np.power(0.5, age_days / HALFLIFE_DAYS).astype(np.float32)
        log(f"  recency weights: half-life {HALFLIFE_DAYS:.0f}d "
            f"(oldest row weight {weights.min():.4f})")

    n_symbols = int(cv_df["symbol"].nunique())
    gap = min(EMBARGO_HOURS * max(1, n_symbols), max(0, len(X) // (CV_SPLITS + 2)))
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=gap)
    log(f"  embargo gap: {gap} rows (~{EMBARGO_HOURS}h x {n_symbols} symbols)")

    fold_aucs, fold_ev_dec, fold_ev_top3 = [], [], []
    for i, (tr, va) in enumerate(tscv.split(X), 1):
        xm, lm = fit_ensemble(X[tr], y[tr],
                              weights[tr] if weights is not None else None)
        preds = predict_ensemble(xm, lm, X[va])
        auc = roc_auc_score(y[va], preds)
        evd, ev3 = selection_ev(cv_df.iloc[va], preds)
        fold_aucs.append(auc)
        fold_ev_dec.append(evd)
        fold_ev_top3.append(ev3)
        log(f"  fold {i}: AUC={auc:.4f}  EV(top-decile)={evd:+.4%}  EV(top3/ts)={ev3:+.4%}")

    mean_auc = float(np.mean(fold_aucs))
    min_auc = float(np.min(fold_aucs))
    n_ev_pos = int(sum(1 for e in fold_ev_dec if e > 0))

    # ── final holdout: train on full CV window, test once ──────────
    log("[5/6] Holdout evaluation (train on CV window, single shot)...")
    xm, lm = fit_ensemble(X, y, weights)
    Xh = to_matrix(ho_df, feature_cols)
    ho_preds = predict_ensemble(xm, lm, Xh)
    ho_auc = float(roc_auc_score(ho_df["label"].values, ho_preds))
    ho_ev_dec, ho_ev_top3 = selection_ev(ho_df, ho_preds)
    log(f"  holdout: AUC={ho_auc:.4f}  EV(top-decile)={ho_ev_dec:+.4%}  "
        f"EV(top3/ts)={ho_ev_top3:+.4%}")

    gates = {
        "mean_auc": mean_auc >= GATE_MEAN_AUC,
        "min_fold_auc": min_auc >= GATE_MIN_FOLD_AUC,
        "ev_folds_positive": n_ev_pos >= GATE_EV_FOLDS_POSITIVE,
        "holdout_auc": ho_auc >= GATE_HOLDOUT_AUC,
        "holdout_ev_positive": ho_ev_dec > 0,
    }
    gate_pass = all(gates.values())

    verdict = {
        "gate_pass": gate_pass,
        "gates": gates,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "label_design": (
            f"fee_hurdled_dead_zone_abs_{HORIZON}"
            if OBJECTIVE_MODE == "absolute_edge"
            else f"fee_hurdled_dead_zone_btc_rel_{HORIZON}"
        ),
        "objective_mode": OBJECTIVE_MODE,
        "horizon": HORIZON,
        "fee_round_trip": FEE_ROUND_TRIP,
        "vol_hurdle_mult": VOL_HURDLE_MULT,
        "max_rows_per_coin": MAX_ROWS_PER_COIN,
        "halflife_days": HALFLIFE_DAYS,
        "keep_nan": KEEP_NAN,
        "nan_prefunding": NAN_PREFUNDING,
        "use_macro": USE_MACRO,
        "symbol_allowlist": sorted(symbol_allowlist),
        "slice_profile": str(SLICE_PROFILE_RAW or "").strip().lower(),
        "slice_profile_summary": slice_summary,
        "cv": {
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "mean_auc": round(mean_auc, 4),
            "min_auc": round(min_auc, 4),
            "fold_ev_top_decile": [round(e, 6) for e in fold_ev_dec],
            "fold_ev_top3_per_ts": [round(e, 6) for e in fold_ev_top3],
            "n_ev_positive_folds": n_ev_pos,
        },
        "holdout": {
            "auc": round(ho_auc, 4),
            "ev_top_decile": round(ho_ev_dec, 6),
            "ev_top3_per_ts": round(ho_ev_top3, 6),
            "n_rows": len(ho_df),
        },
        "n_labeled_rows": int(len(df)),
        "n_features": len(feature_cols),
        "base_rate": round(float(y.mean()), 4),
        "old_pipeline_baseline_auc": 0.532,
        "note": ("research-only: live ensemble.pkl untouched; promotion is a "
                 "separate governance action gated on this verdict"),
    }
    json.dump(verdict, open(VERDICT_PATH, "w"), indent=2)

    log("[6/6] Saving research model + verdict...")
    with open(REBUILD_MODEL_PATH, "wb") as f:
        pickle.dump((xm, lm, feature_cols), f)

    log(f"  verdict: {VERDICT_PATH}")
    log(f"\n  ══ REBUILD VERDICT: {'PASS — promotion candidate' if gate_pass else 'FAIL — lane stays frozen'} ══")
    for k, v in gates.items():
        log(f"    {'' if v else ''} {k}")
    log(f"  done in {(time.time() - t0) / 60:.1f} min")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
