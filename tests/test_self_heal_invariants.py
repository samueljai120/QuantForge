"""Loop closure: a detected money-conservation breach -> an auto-opened, GATED fix
candidate. The trustworthy property: a wrong patch is REJECTED at the test gate; only a
passing patch reaches APPROVED-pending-human (deploy stays human-gated).
"""
import importlib

import pytest

shi = importlib.import_module("quantforge_self_heal_invariants")
from qf_safety.candidate_pipeline import CandidatePipeline
from qf_safety.self_improvement import SelfImprovementLoop

REQUIRED = {"problem_statement", "evidence", "hypothesis", "patch", "files_changed",
            "risk_classification", "test_plan", "expected_improvement",
            "possible_regressions", "rollback_procedure", "author"}


def _vio(name="futures_open_close_parity", detail="4 opened, 0 closed -> 4 orphaned",
         hint="quantforge_agent._execute_futures: close must credit margin + ledger"):
    return {"name": name, "severity": "critical", "detail": detail, "hint": hint}


def test_candidate_has_all_required_fields_and_targets_the_code():
    c = shi.build_fix_candidate(_vio())
    assert REQUIRED <= set(c), f"missing: {REQUIRED - set(c)}"
    assert "scripts/quantforge_agent.py" in c["files_changed"]   # parsed from the hint
    assert c["author"] == "invariant_detector"
    assert "futures_open_close_parity" in c["problem_statement"]


def test_run_opens_one_candidate_per_critical_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(shi, "STORE", str(tmp_path / "cands.json"))
    monkeypatch.setattr(shi, "INDEX", str(tmp_path / "index.json"))
    state = {"violations": [_vio(), _vio(name="cash_negative", detail="cash -$3"),
                            {"name": "x", "severity": "warning", "detail": "w", "hint": ""}]}
    opened1 = shi.open_for_state(state)
    assert len(opened1) == 2                      # 2 criticals, warning ignored
    opened2 = shi.open_for_state(state)           # same state again
    assert opened2 == []                          # idempotent — no duplicate candidates


def test_gated_loop_approves_good_patch_rejects_bad(tmp_path):
    pipe = CandidatePipeline(str(tmp_path / "store.json"))
    loop = SelfImprovementLoop(
        pipe,
        sandbox_runner=lambda cand, path: None,                       # no-op sandbox
        test_runner=lambda cand, path: {"passed": "GOOD" in cand["patch"],
                                        "report": "ok" if "GOOD" in cand["patch"] else "FAIL"},
        evidence_runner=lambda cand: {"regression": False},
        reviewer=lambda cand: True,
    )
    good = shi.build_fix_candidate(_vio(), patch="GOOD: close+ledger on flip")
    res_g = loop.run(good)
    assert res_g["approved_pending_deploy"] and not res_g["rejected"]  # passes -> human-approvable

    bad = shi.build_fix_candidate(_vio(name="other"), patch="BAD: breaks tests")
    res_b = loop.run(bad)
    assert res_b["rejected"]                                           # fails the gate -> never reaches you
