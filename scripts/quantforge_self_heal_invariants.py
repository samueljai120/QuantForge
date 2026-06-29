#!/usr/bin/env python3
"""Loop closure: detection -> auto-opened, GATED fix candidate.

When the money-conservation invariants (quantforge_invariants.py) report a CRITICAL
breach, this bridge opens a fix candidate in the existing qf_safety.candidate_pipeline —
with the invariant encoded as the hard acceptance bar (full test suite green AND the
invariant clears). A coding agent then attempts a patch; SelfImprovementLoop.run drives
it through sandbox -> the full test gate -> independent review, and ONLY a passing patch
reaches APPROVED-pending-human. The deploy stays human-gated (the user's non-negotiable
rule + the code-mutation guard). So a wrong fix can never reach you; a tested one becomes
a one-click approval.

This module only OPENS gated candidates (idempotently) and surfaces them. It does NOT
write code or deploy — that is the coding agent (patch) and the human (approve) steps.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qf_safety.candidate_pipeline import CandidatePipeline

DATA = os.path.expanduser("~/quantforge/data/quantforge")
INVARIANTS_STATE = os.path.join(DATA, "qf_invariants_state.json")
STORE = os.path.join(DATA, "qf_fix_candidates.json")   # CASStore for the pipeline
INDEX = os.path.join(DATA, "qf_fix_candidate_index.json")  # violation-signature -> candidate id

# Best-effort map from an invariant hint to the source file the fix lives in.
_DEFAULT_TARGET = "scripts/quantforge_agent.py"


def _files_from_hint(hint: str):
    """Parse 'quantforge_agent._execute_futures: ...' -> ['scripts/quantforge_agent.py'].
    Falls back to the trading core when no module is named."""
    head = str(hint or "").split(":", 1)[0].strip()
    mod = head.split(".", 1)[0].strip()
    if mod and mod.replace("_", "").isalnum() and mod.startswith("quantforge"):
        return [f"scripts/{mod}.py"]
    return [_DEFAULT_TARGET]


def build_fix_candidate(violation: dict, *, patch: str = "") -> dict:
    """Build a candidate-pipeline proposal (all required fields) from an invariant
    violation. ``patch`` is supplied by the coding agent; until then it is a placeholder
    so the candidate can sit in PROPOSED awaiting a fix attempt."""
    name = violation.get("name", "unknown")
    detail = violation.get("detail", "")
    hint = violation.get("hint", "")
    return {
        "author": "invariant_detector",
        "problem_statement": f"Money-conservation invariant breached: {name} — {detail}",
        "evidence": f"qf_invariants_state.json critical: {name}: {detail}",
        "hypothesis": hint or "see invariant detail; trace the money flow that fails to conserve",
        "patch": patch or "AWAITING_FIX: coding agent must produce a patch that makes the invariant clear",
        "files_changed": _files_from_hint(hint),
        "risk_classification": "money_conservation",
        "test_plan": f"full pytest suite green AND quantforge_invariants.py reports no '{name}' critical",
        "expected_improvement": f"invariant '{name}' clears; no orphaned/leaked money",
        "possible_regressions": "futures/spot lifecycle, equity accounting, ledger parity",
        "rollback_procedure": "git revert the patch; the invariant + full test suite gate every deploy",
    }


def _signature(violation: dict) -> str:
    raw = f"{violation.get('name','')}|{violation.get('detail','')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def open_for_state(state: dict) -> list:
    """For each CRITICAL violation in the invariant state, open a gated fix candidate —
    idempotent: a violation already opened (by signature) is skipped. Returns the list of
    (candidate_id, violation_name) newly opened."""
    crits = [v for v in (state.get("violations") or []) if v.get("severity") == "critical"]
    index = _read_json(INDEX, {})
    pipeline = CandidatePipeline(STORE)
    opened = []
    for v in crits:
        sig = _signature(v)
        if sig in index:
            continue
        cid = pipeline.submit(build_fix_candidate(v))
        index[sig] = {"cid": cid, "name": v.get("name"), "opened_at": datetime.now(timezone.utc).isoformat()}
        opened.append((cid, v.get("name")))
    if opened:
        try:
            with open(INDEX, "w") as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass
    return opened


def main():
    state = _read_json(INVARIANTS_STATE, {})
    crits = [v for v in (state.get("violations") or []) if v.get("severity") == "critical"]
    if not crits:
        print("Self-heal/invariants: no critical money-conservation breaches — nothing to fix.")
        return 0
    opened = open_for_state(state)
    index = _read_json(INDEX, {})
    print(f"Self-heal/invariants: {len(crits)} critical breach(es); {len(opened)} new gated fix candidate(s) opened.")
    for cid, name in opened:
        print(f"  [proposed] fix candidate {cid} for '{name}' — awaiting coding-agent patch -> gated tests -> human approval")
    already = len(index) - len(opened)
    if already > 0:
        print(f"  ({already} breach(es) already have an open fix candidate)")
    # Surface autofix-dispatch verdicts (filled on cron): a passing patch is one click away.
    for info in index.values():
        v = str(info.get("attempt_verdict", "") or "")
        if "APPROVED" in v:
            print(f"  [READY] fix for '{info.get('name')}' PASSED the gate -> APPROVE to deploy "
                  f"(attempt candidate {info.get('attempt_cid')})")
        elif info.get("attempted") and v:
            print(f"  [info] fix attempt for '{info.get('name')}': {v[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
