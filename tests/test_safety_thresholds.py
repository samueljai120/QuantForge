"""Phase 0.5 — Safety-control characterization (threshold tripwires).

The audit found ~30 test assertions for a 7,156-line subsystem and ZERO tests on
the deterministic safety controls. A full behavioural harness for the 4,000-line
agent is Phase 1+ work (it pulls in pandas/xgboost). As a first, robust layer we
pin the agent's risk constants by AST-parsing the source — no heavy import — so
any silent or unauthorized change to a kill switch, leverage limit, or drawdown
breaker fails immediately, and we verify the parameter registry's rollback values
match the live agent (so a rollback restores the real default).
"""

import ast
import os

import pytest

from qf_safety.param_schema import ParamRegistry

_HERE = os.path.dirname(__file__)
AGENT = os.path.join(_HERE, "..", "scripts", "quantforge_agent.py")
REGISTRY = os.path.join(_HERE, "..", "config", "quantforge", "param_registry.json")


def _module_constants(path):
    """Top-level NAME = <literal> assignments only (deterministic, no execution)."""
    with open(path) as f:
        tree = ast.parse(f.read())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    consts[t.id] = node.value.value
    return consts


@pytest.fixture(scope="module")
def consts():
    if not os.path.exists(AGENT):
        pytest.skip("quantforge_agent.py not present in this checkout")
    return _module_constants(AGENT)


def test_panic_halt_threshold_pinned(consts):
    assert consts.get("PANIC_HALT_PCT") == 0.15


def test_drawdown_trim_threshold_pinned(consts):
    assert consts.get("DRAWDOWN_TRIM_PCT") == 0.08


def test_spot_leverage_is_one(consts):
    assert consts.get("LEVERAGE") == 1


def test_max_rebalances_per_day_pinned(consts):
    assert consts.get("MAX_REBALANCES_PER_DAY") == 2


def test_futures_leverage_within_safe_bounds(consts):
    fl = consts.get("FUTURES_LEVERAGE")
    assert fl is not None and 1 <= fl <= 5


def test_registry_rollback_values_match_agent(consts):
    reg = ParamRegistry.from_file(REGISTRY)
    assert reg.spec("panic_halt_pct").rollback_value == consts.get("PANIC_HALT_PCT")
    assert reg.spec("drawdown_trim_pct").rollback_value == consts.get("DRAWDOWN_TRIM_PCT")
    assert reg.spec("futures_leverage").rollback_value == consts.get("FUTURES_LEVERAGE")
    assert reg.spec("max_rebalances_per_day").rollback_value == consts.get(
        "MAX_REBALANCES_PER_DAY"
    )


def test_kill_switch_params_are_not_autonomously_modifiable():
    reg = ParamRegistry.from_file(REGISTRY)
    for key in ("panic_halt_pct", "drawdown_trim_pct"):
        spec = reg.spec(key)
        assert spec is not None
        assert spec.autonomous_allowed is False, f"{key} must not be autonomous"
