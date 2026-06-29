#!/usr/bin/env python3
"""QuantForge Research Director — autonomous edge-research loop .

This is the closed loop the ML lane was missing. Once a week it:

  1. Runs the LOCKED baseline experiment (4h horizon, 0.25 hurdle, funding-era
     data) so the primary evidence stream accumulates uncontaminated.
  2. Picks ONE experiment arm from a small bounded grid (round-robin until all
     have ≥2 runs, then exploits: re-runs the arm with the best mean holdout EV)
     and runs it.
  3. Appends every verdict to an append-only research ledger
     (research_ledger.jsonl) — config + full metrics, nothing overwritten.
  4. If any arm's last two runs BOTH pass the hard gates, writes
     model/promotion_candidate.json for governance review.

What it deliberately CANNOT do:
  - touch the live model (ensemble.pkl), live trading, strategy params,
    or the portfolio — outputs are research files only
  - invent new experiment knobs — the grid is fixed in this file
  - promote anything — promotion stays an explicit governance/operator action

Production host safety: skips entirely if 1-min load > 4.0; flock-guarded; each run
is niced by the caller (cron) and takes ~1.5-3 min.

Usage: quantforge_research_director.py [--once-arm ARM_ID]
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
MODEL_DIR = os.path.join(DATA_DIR, "model")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.path.join(DATA_DIR, "research_ledger.jsonl")
PROMOTION_PATH = os.path.join(MODEL_DIR, "promotion_candidate.json")
REBUILD_SCRIPT = os.path.join(SCRIPTS_DIR, "quantforge_ml_rebuild.py")
PYTHON_BIN = os.path.expanduser("~/.venvs/quant-ops/bin/python")
SINCE_DATE = "2026-04-05"   # derivatives collector online — locked epoch start

MAX_LOAD = 4.0
RUN_TIMEOUT_S = 1800

# ── Bounded experiment grid ─────────────────────────────────────────
# baseline_4h_h25 is the LOCKED primary stream — it runs EVERY week.
# history_8m_decay tests "does pre-funding-era data help?": ~8 months of
# rows, deriv_* NaN'd before the collector epoch (0.0 there means missing),
# NaN routed natively by the trees, 60-day recency half-life.
ARMS = {
    "baseline_4h_h25": {"horizon": "4h", "hurdle": 0.25},
    "explore_4h_h15":  {"horizon": "4h", "hurdle": 0.15},
    "explore_4h_h35":  {"horizon": "4h", "hurdle": 0.35},
    "explore_8h_h25":  {"horizon": "8h", "hurdle": 0.25},
    "explore_8h_h35":  {"horizon": "8h", "hurdle": 0.35},
    "majors_focus_4h_h25": {
        "horizon": "4h",
        "hurdle": 0.25,
        "symbol_allowlist": "ETH,SOL,XRP,BCH,TRX",
        "min_labeled_rows": 3000,
        "note": "Research-hold majors-first arm based on segmented holdout showing alt slices and short slices as the main drag.",
    },
    "majors_non_fragile_4h_h25": {
        "horizon": "4h",
        "hurdle": 0.25,
        "symbol_allowlist": "ETH,SOL,XRP,BCH,TRX",
        "slice_profile": "majors_non_fragile",
        "min_labeled_rows": 2500,
        "note": "Research-hold arm that removes fragile major rows before rebuild evaluation.",
    },
    "majors_non_fragile_absolute_edge_4h_h25": {
        "horizon": "4h",
        "hurdle": 0.25,
        "objective_mode": "absolute_edge",
        "symbol_allowlist": "ETH,SOL,XRP,BCH,TRX",
        "slice_profile": "majors_non_fragile",
        "min_labeled_rows": 2500,
        "note": "Research-hold arm that tests absolute post-cost edge instead of BTC-relative outperformance within the non-fragile major slice.",
    },
    "majors_positive_longs_4h_h25": {
        "horizon": "4h",
        "hurdle": 0.25,
        "symbol_allowlist": "ETH,SOL,XRP,BCH,TRX",
        "slice_profile": "majors_positive_long_slices",
        "min_labeled_rows": 750,
        "note": "Research-hold arm focused on major, non-fragile trend/breakout-long contexts that survived segmented holdout review.",
    },
    "history_8m_decay": {"horizon": "4h", "hurdle": 0.25, "max_rows": 6000,
                         "nan_prefunding": 1, "keep_nan": 1, "halflife": 60,
                         "since": None, "timeout": 14400},
}
BASELINE_ARM = "baseline_4h_h25"
MIN_RUNS_BEFORE_EXPLOIT = 2


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def load_ok():
    try:
        with open("/proc/loadavg") as f:
            load1 = float(f.read().split()[0])
        if load1 > MAX_LOAD:
            log(f"SKIP: load {load1} > {MAX_LOAD}")
            return False
    except Exception:
        pass
    return True


def read_ledger():
    entries = []
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        continue
    return entries


def append_ledger(entry):
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_arm(arm_id):
    """Run one rebuild experiment; return the ledger entry (or None)."""
    cfg = ARMS[arm_id]
    verdict_path = os.path.join(MODEL_DIR, f"verdict_{arm_id}.json")
    model_path = os.path.join(MODEL_DIR, f"model_{arm_id}.pkl")
    env = {
        **os.environ,
        "QF_HORIZON": cfg["horizon"],
        "QF_HURDLE_MULT": str(cfg["hurdle"]),
        "QF_VERDICT_PATH": verdict_path,
        "QF_MODEL_PATH": model_path,
        "PYTHONWARNINGS": "ignore",
    }
    if cfg.get("max_rows"):
        env["QF_MAX_ROWS"] = str(cfg["max_rows"])
    if cfg.get("nan_prefunding"):
        env["QF_NAN_PREFUNDING"] = "1"
    if cfg.get("keep_nan"):
        env["QF_KEEP_NAN"] = "1"
    if cfg.get("halflife"):
        env["QF_HALFLIFE_DAYS"] = str(cfg["halflife"])
    if cfg.get("symbol_allowlist"):
        env["QF_SYMBOL_ALLOWLIST"] = str(cfg["symbol_allowlist"])
    if cfg.get("slice_profile"):
        env["QF_SLICE_PROFILE"] = str(cfg["slice_profile"])
    if cfg.get("objective_mode"):
        env["QF_OBJECTIVE_MODE"] = str(cfg["objective_mode"])
    if cfg.get("min_labeled_rows"):
        env["QF_MIN_LABELED_ROWS"] = str(cfg["min_labeled_rows"])
    # since=None -> full loaded history (bounded by max_rows per coin)
    since = cfg.get("since", SINCE_DATE)
    argv = [PYTHON_BIN, "-u", REBUILD_SCRIPT] + ([since] if since else [])
    timeout = cfg.get("timeout", RUN_TIMEOUT_S)
    log(f"running arm {arm_id} ({cfg})...")
    try:
        # gate_pass=False exits 1 by design — that is a successful run
        subprocess.run(
            argv, env=env, timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        log(f"  arm {arm_id} TIMED OUT after {timeout}s")
        return None
    except Exception as e:
        log(f"  arm {arm_id} failed to launch: {e}")
        return None

    if not os.path.exists(verdict_path):
        log(f"  arm {arm_id} produced no verdict — see rebuild logs")
        return None
    try:
        with open(verdict_path) as f:
            verdict = json.load(f)
    except Exception as e:
        log(f"  arm {arm_id} verdict unreadable: {e}")
        return None

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arm": arm_id,
        "config": cfg,
        "since": since,
        "gate_pass": verdict.get("gate_pass", False),
        "gates": verdict.get("gates", {}),
        "cv_mean_auc": verdict.get("cv", {}).get("mean_auc"),
        "cv_min_auc": verdict.get("cv", {}).get("min_auc"),
        "holdout_auc": verdict.get("holdout", {}).get("auc"),
        "holdout_ev_top_decile": verdict.get("holdout", {}).get("ev_top_decile"),
        "holdout_ev_top3": verdict.get("holdout", {}).get("ev_top3_per_ts"),
        "n_labeled_rows": verdict.get("n_labeled_rows"),
    }
    append_ledger(entry)
    log(f"  arm {arm_id}: gate_pass={entry['gate_pass']} "
        f"mean_auc={entry['cv_mean_auc']} holdout_ev={entry['holdout_ev_top_decile']}")
    return entry


def pick_explore_arm(ledger):
    """Round-robin under-tested arms first; then exploit best holdout EV."""
    counts = {a: 0 for a in ARMS if a != BASELINE_ARM}
    ev_sums = {a: [] for a in counts}
    for e in ledger:
        a = e.get("arm")
        if a in counts:
            counts[a] += 1
            ev = e.get("holdout_ev_top_decile")
            if ev is not None:
                ev_sums[a].append(ev)

    under = [a for a, c in sorted(counts.items()) if c < MIN_RUNS_BEFORE_EXPLOIT]
    if under:
        return under[0]
    # exploit: best mean holdout EV
    best = max(ev_sums, key=lambda a: (sum(ev_sums[a]) / len(ev_sums[a])) if ev_sums[a] else -9)
    return best


def check_promotion(ledger):
    """Two consecutive gate-passes on the same arm -> promotion candidate."""
    by_arm = {}
    for e in ledger:
        by_arm.setdefault(e["arm"], []).append(e)
    for arm, runs in by_arm.items():
        if len(runs) >= 2 and runs[-1]["gate_pass"] and runs[-2]["gate_pass"]:
            candidate = {
                "arm": arm,
                "config": ARMS.get(arm, {}),
                "consecutive_passes": 2,
                "runs": runs[-2:],
                "flagged_at": datetime.now(timezone.utc).isoformat(),
                "action_required": ("governance review: arm passed hard gates twice "
                                    "consecutively — eligible for paper-trial promotion. "
                                    "Promotion is NOT automatic."),
            }
            with open(PROMOTION_PATH, "w") as f:
                json.dump(candidate, f, indent=2)
            log(f" PROMOTION CANDIDATE: {arm} passed gates twice consecutively "
                f"-> {PROMOTION_PATH}")
            return candidate
    return None


def main():
    if not load_ok():
        sys.exit(0)

    ledger = read_ledger()
    log(f"research director: {len(ledger)} prior runs in ledger")

    if "--once-arm" in sys.argv:
        arm = sys.argv[sys.argv.index("--once-arm") + 1]
        run_arm(arm)
        check_promotion(read_ledger())
        return

    # 1. locked baseline stream — every week
    run_arm(BASELINE_ARM)

    # 2. one exploration/exploitation arm — if load still ok
    if load_ok():
        arm = pick_explore_arm(read_ledger())
        run_arm(arm)

    # 3. promotion check
    check_promotion(read_ledger())
    log("research director: cycle complete")


if __name__ == "__main__":
    main()
