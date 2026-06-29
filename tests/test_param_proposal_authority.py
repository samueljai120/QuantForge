"""EWAA P1 — single-write-authority at the proposal gate.

Only author='ewaa_proposer' may write regime_weight_table (and the active weight
keys). Other authors (reflect, self_heal, weight_learner) ESCALATE on those keys.
Exemptions (Option B, verified live): fixed_alloc_pct (reflect owns it) and
ml_btc_weight (agent/ml owns it) are NOT blocked — blocking them would regress
reflect's nightly re-risk and freeze ml_btc_weight forever.
"""
import pytest

from qf_safety.param_schema import ParamRegistry
from qf_safety.param_proposal import ParameterProposalGate


@pytest.fixture
def registry():
    base = {"type": "float", "min": 0.0, "max": 1.0, "default": 0.1, "owner": "x",
            "risk_class": "low", "autonomous_allowed": True, "rollback_value": 0.1,
            "required_validation_tests": [], "approval_required": False}
    return ParamRegistry.from_dict({
        "regime_weight_table": {**base, "type": "object", "default": {}},
        "mr_weight": dict(base), "ml_scanner_weight": dict(base),
        "futures_weight": dict(base), "funding_arb_weight": dict(base),
        "fixed_alloc_pct": {**base, "default": 0.55},
        "ml_btc_weight": dict(base),
    })


TABLE = {r: {"spot_alloc_pct": 0.85, "futures_weight": 0.0, "mr_weight": 0.0,
             "ml_scanner_weight": 0.0, "funding_arb_weight": 0.0}
         for r in ["STRONG_BEAR", "BEAR", "CHOP", "NEUTRAL", "BULL", "STRONG_BULL"]}


def _gate(registry, applied):
    return ParameterProposalGate(
        registry=registry, current_params_loader=lambda: {},
        gate_runner=lambda p, c: {"approved": True, "reason": "ok"},
        applier=lambda k, v: applied.append((k, v)),
    )


def test_regime_weight_table_blocked_for_non_ewaa(registry):
    applied = []
    r = _gate(registry, applied).propose("regime_weight_table", TABLE, author="quantforge_reflect")
    assert r.status == "escalated" and r.stage == "authority"
    assert applied == []


def test_regime_weight_table_applied_for_ewaa(registry):
    applied = []
    r = _gate(registry, applied).propose("regime_weight_table", TABLE, author="ewaa_proposer")
    assert r.status == "applied" and applied == [("regime_weight_table", TABLE)]


def test_active_weight_keys_blocked_for_non_ewaa(registry):
    for author in ("self_heal_llm", "quantforge_weight_learner"):
        for key in ("mr_weight", "ml_scanner_weight", "futures_weight", "funding_arb_weight"):
            applied = []
            r = _gate(registry, applied).propose(key, 0.2, author=author)
            assert r.status == "escalated" and r.stage == "authority", f"{author}/{key}"
            assert applied == []


def test_fixed_alloc_pct_still_applies_for_reflect(registry):
    applied = []
    r = _gate(registry, applied).propose("fixed_alloc_pct", 0.6, author="quantforge_reflect")
    assert r.status == "applied" and applied == [("fixed_alloc_pct", 0.6)]


def test_ml_btc_weight_exempt_applies_for_reflect(registry):
    applied = []
    r = _gate(registry, applied).propose("ml_btc_weight", 0.1, author="quantforge_reflect")
    assert r.status == "applied" and applied == [("ml_btc_weight", 0.1)]
