"""Phase 8 — Parameter-proposal gate (the trustworthy road for L2 changes).

A parameter change is L2 (config proposal), not L3 (code). Its trustworthy gate
is: schema check -> fail-closed backtest -> atomic apply + decision log, with
risk/kill-switch params ESCALATED to a human instead of auto-applied. This is
what the reflect daemon proposes through instead of writing the params file
directly.
"""

import json

import pytest

from qf_safety.param_schema import ParamRegistry
from qf_safety.param_proposal import ParameterProposalGate, atomic_param_applier


@pytest.fixture
def registry():
    return ParamRegistry.from_dict({
        "fixed_alloc_pct": {
            "type": "float", "min": 0.0, "max": 1.0, "default": 0.55,
            "owner": "allocator", "risk_class": "low",
            "autonomous_allowed": True, "rollback_value": 0.55,
            "required_validation_tests": ["backtest_gate"], "approval_required": False,
        },
        "panic_halt_pct": {
            "type": "float", "min": 0.05, "max": 0.5, "default": 0.15,
            "owner": "risk", "risk_class": "kill_switch",
            "autonomous_allowed": False, "rollback_value": 0.15,
            "required_validation_tests": ["panic_halt_test"], "approval_required": True,
        },
    })


def _gate(registry, *, gate_runner, applier, current=None, log=None):
    return ParameterProposalGate(
        registry=registry,
        current_params_loader=lambda: dict(current or {"fixed_alloc_pct": 0.55}),
        gate_runner=gate_runner,
        applier=applier,
        decision_log=log,
    )


def test_autonomous_param_applied_when_gate_approves(registry):
    applied = []
    g = _gate(registry, gate_runner=lambda p, c: {"approved": True, "reason": "ok"},
              applier=lambda k, v: applied.append((k, v)))
    r = g.propose("fixed_alloc_pct", 0.6, author="reflect")
    assert r.status == "applied"
    assert applied == [("fixed_alloc_pct", 0.6)]


def test_risk_param_escalated_not_applied(registry):
    applied = []
    g = _gate(registry, gate_runner=lambda p, c: {"approved": True, "reason": "ok"},
              applier=lambda k, v: applied.append((k, v)))
    r = g.propose("panic_halt_pct", 0.2, author="reflect")
    assert r.status == "escalated"
    assert applied == []  # a kill-switch is never applied autonomously


def test_unregistered_param_rejected_before_gate(registry):
    called = []
    g = _gate(registry, gate_runner=lambda p, c: called.append(1) or {"approved": True},
              applier=lambda k, v: None)
    r = g.propose("mystery_param", 1.0, author="reflect")
    assert r.status == "rejected"
    assert r.stage == "schema"
    assert called == []  # schema short-circuits before the backtest


def test_out_of_range_rejected(registry):
    g = _gate(registry, gate_runner=lambda p, c: {"approved": True},
              applier=lambda k, v: None)
    assert g.propose("fixed_alloc_pct", 5.0, author="reflect").status == "rejected"


def test_gate_rejection_not_applied(registry):
    applied = []
    g = _gate(registry, gate_runner=lambda p, c: {"approved": False, "reason": "DD too high"},
              applier=lambda k, v: applied.append((k, v)))
    r = g.propose("fixed_alloc_pct", 0.6, author="reflect")
    assert r.status == "rejected"
    assert r.stage == "gate"
    assert applied == []


def test_gate_exception_fails_closed(registry):
    applied = []

    def boom(p, c):
        raise RuntimeError("gate down")

    g = _gate(registry, gate_runner=boom, applier=lambda k, v: applied.append((k, v)))
    r = g.propose("fixed_alloc_pct", 0.6, author="reflect")
    assert r.status == "rejected"
    assert applied == []


def test_applier_exception_reported(registry):
    def bad_apply(k, v):
        raise IOError("disk full")

    g = _gate(registry, gate_runner=lambda p, c: {"approved": True}, applier=bad_apply)
    r = g.propose("fixed_alloc_pct", 0.6, author="reflect")
    assert r.status == "rejected"
    assert r.stage == "apply"


def test_decision_log_records_outcome(registry):
    class FakeLog:
        def __init__(self):
            self.entries = []

        def append(self, payload):
            self.entries.append(payload)

    log = FakeLog()
    g = _gate(registry, gate_runner=lambda p, c: {"approved": True},
              applier=lambda k, v: None, log=log)
    g.propose("fixed_alloc_pct", 0.6, author="reflect")
    assert any(e.get("event") == "param_proposal" for e in log.entries)


def test_atomic_applier_sets_value_and_preserves_others(tmp_path):
    p = tmp_path / "params.json"
    p.write_text(json.dumps({"fixed_alloc_pct": 0.55, "other": 7}))
    apply = atomic_param_applier(str(p))
    apply("fixed_alloc_pct", 0.6)
    result = json.loads(p.read_text())
    assert result["fixed_alloc_pct"] == 0.6
    assert result["other"] == 7  # untouched
