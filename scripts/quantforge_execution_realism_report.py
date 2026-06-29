#!/usr/bin/env python3
"""QuantForge - execution realism report."""

from __future__ import annotations

import json
import os
import sys
import ast
import importlib.util
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from quantforge_target_profiles import EXECUTION_REALISM_HAIRCUT, ROUND_TRIP_COST_PROXY

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTPUT_FILE = os.path.join(BASE_DIR, "execution-realism-report.json")
TARGET_REBUILD_FILE = os.path.join(BASE_DIR, "target-rebuild-report.json")
PAPER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantforge_paper.py")

REALISM_CONSTANTS = [
    "PAPER_ENTRY_SLIPPAGE_BPS",
    "PAPER_EXIT_SLIPPAGE_BPS",
    "PAPER_SPREAD_BPS",
    "REBUILD_TRIAL_ENTRY_SLIPPAGE_BPS",
    "REBUILD_TRIAL_EXIT_SLIPPAGE_BPS",
    "REBUILD_TRIAL_SPREAD_BPS",
    "REBUILD_TRIAL_MARK_HAIRCUT_BPS",
    "RESEARCH_HOLD_ENTRY_SLIPPAGE_BPS",
    "RESEARCH_HOLD_EXIT_SLIPPAGE_BPS",
    "RESEARCH_HOLD_SPREAD_BPS",
    "RESEARCH_HOLD_STOP_GAP_BPS",
    "RESEARCH_HOLD_MARK_HAIRCUT_BPS",
]


def read_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_constants() -> dict:
    if not os.path.exists(PAPER_SCRIPT):
        return {}
    try:
        spec = importlib.util.spec_from_file_location("quantforge_paper_runtime", PAPER_SCRIPT)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            constants = {}
            for name in REALISM_CONSTANTS:
                value = getattr(module, name, None)
                if value is not None:
                    constants[name] = float(value)
            if constants:
                return constants
    except Exception:
        pass

    text = open(PAPER_SCRIPT).read()
    tree = ast.parse(text, filename=PAPER_SCRIPT)
    constants = {}
    names = set(REALISM_CONSTANTS)

    def _literal_number(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        if name not in names:
            continue
        value = stmt.value
        default = None
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "float":
            if value.args:
                arg = value.args[0]
                default = _literal_number(arg)
                if default is None and isinstance(arg, ast.Call):
                    for candidate in list(arg.args)[::-1]:
                        default = _literal_number(candidate)
                        if default is not None:
                            break
        else:
            default = _literal_number(value)
        if default is not None:
            constants[name] = float(default)
    return constants


def _mode_profile(constants: dict, mode: str) -> dict:
    if mode == "standard":
        entry_slip = constants.get("PAPER_ENTRY_SLIPPAGE_BPS", 10.0)
        exit_slip = constants.get("PAPER_EXIT_SLIPPAGE_BPS", 10.0)
        spread = constants.get("PAPER_SPREAD_BPS", 4.0)
        mark = 0.0
        stop_gap = 0.0
    elif mode == "rebuild_trial":
        entry_slip = max(constants.get("PAPER_ENTRY_SLIPPAGE_BPS", 10.0), constants.get("REBUILD_TRIAL_ENTRY_SLIPPAGE_BPS", 35.0))
        exit_slip = max(constants.get("PAPER_EXIT_SLIPPAGE_BPS", 10.0), constants.get("REBUILD_TRIAL_EXIT_SLIPPAGE_BPS", 30.0))
        spread = max(constants.get("PAPER_SPREAD_BPS", 4.0), constants.get("REBUILD_TRIAL_SPREAD_BPS", 12.0))
        mark = constants.get("REBUILD_TRIAL_MARK_HAIRCUT_BPS", 18.0)
        stop_gap = 0.0
    else:
        entry_slip = max(constants.get("PAPER_ENTRY_SLIPPAGE_BPS", 10.0), constants.get("RESEARCH_HOLD_ENTRY_SLIPPAGE_BPS", 30.0))
        exit_slip = max(constants.get("PAPER_EXIT_SLIPPAGE_BPS", 10.0), constants.get("RESEARCH_HOLD_EXIT_SLIPPAGE_BPS", 35.0))
        spread = max(constants.get("PAPER_SPREAD_BPS", 4.0), constants.get("RESEARCH_HOLD_SPREAD_BPS", 14.0))
        mark = constants.get("RESEARCH_HOLD_MARK_HAIRCUT_BPS", 28.0)
        stop_gap = constants.get("RESEARCH_HOLD_STOP_GAP_BPS", 18.0)
    return {
        "mode": mode,
        "entry_slippage_bps": round(entry_slip, 2),
        "exit_slippage_bps": round(exit_slip, 2),
        "spread_bps": round(spread, 2),
        "mark_haircut_bps": round(mark, 2),
        "stop_gap_bps": round(stop_gap, 2),
        "round_trip_bps": round(entry_slip + exit_slip + spread, 2),
    }


def build_report() -> dict:
    constants = _extract_constants()
    target_rebuild = read_json(TARGET_REBUILD_FILE)
    target_profile = (target_rebuild.get("profile") or {}) if target_rebuild else {}
    target_rules = target_profile.get("rules") or {}

    standard = _mode_profile(constants, "standard")
    rebuild = _mode_profile(constants, "rebuild_trial")
    research = _mode_profile(constants, "research_hold")

    target_cost_bps = (
        float(target_rules.get("round_trip_cost_proxy", ROUND_TRIP_COST_PROXY)) +
        float(target_rules.get("execution_realism_haircut", EXECUTION_REALISM_HAIRCUT))
    ) * 10000.0
    rebuild_gap_bps = rebuild["round_trip_bps"] - target_cost_bps
    research_gap_bps = research["round_trip_bps"] - target_cost_bps

    if rebuild_gap_bps > 10 or research_gap_bps > 10:
        verdict = "too_optimistic"
        next_actions = ["raise target haircuts", "tighten paper cost model", "collect book snapshots"]
    elif rebuild_gap_bps < -10 and research_gap_bps < -10:
        verdict = "too_conservative"
        next_actions = ["confirm target realism", "ready_for_replay"]
    else:
        verdict = "aligned"
        next_actions = ["ready_for_replay", "collect book snapshots"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if constants else "missing_paper_profile",
        "verdict": verdict,
        "target_cost_bps": round(target_cost_bps, 2),
        "modes": {
            "standard": standard,
            "rebuild_trial": rebuild,
            "research_hold": research,
        },
        "comparison": {
            "rebuild_gap_vs_target_bps": round(rebuild_gap_bps, 2),
            "research_gap_vs_target_bps": round(research_gap_bps, 2),
        },
        "next_actions": next_actions,
    }


def main() -> None:
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge execution realism report")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
