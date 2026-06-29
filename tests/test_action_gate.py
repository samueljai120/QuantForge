"""Phase 8 — Action gate for self-healing.

Self-heal takes actions, not parameter edits. Each action is classified L0-L4 and
routed: reversible operational fixes execute autonomously; config goes to the
param gate; code/model/package changes go to the candidate pipeline (blocked
autonomous); and anything that clears a kill switch / halt (re-enabling risk) is
ESCALATED — never autonomous. Unknown actions fail closed.
"""

import pytest

from qf_safety.permissions import PermissionLevel
from qf_safety.action_gate import ActionGate, SELF_HEAL_ACTION_LEVELS, LLM_ACTION_LEVELS


LEVELS = {
    "restart_collector": PermissionLevel.REVERSIBLE_OP,
    "retune_param": PermissionLevel.CONFIG_PROPOSAL,
    "rebuild_model": PermissionLevel.CODE_MODEL_CHANGE,
    "install_package": PermissionLevel.CODE_MODEL_CHANGE,
    "clear_futures_kill": PermissionLevel.FINANCIAL_SECURITY,
}


@pytest.fixture
def gate():
    return ActionGate(LEVELS)


def test_reversible_op_executes_with_rollback(gate):
    v = gate.evaluate("restart_collector", autonomous=True, has_rollback=True)
    assert v.route == "execute"
    assert v.allowed is True


def test_reversible_op_blocked_without_rollback(gate):
    v = gate.evaluate("restart_collector", autonomous=True, has_rollback=False)
    assert v.route == "block"
    assert v.allowed is False


def test_config_routes_to_param_gate(gate):
    v = gate.evaluate("retune_param", autonomous=True)
    assert v.route == "param_gate"
    assert v.allowed is False  # not directly executed


def test_code_change_routes_to_candidate_pipeline(gate):
    v = gate.evaluate("rebuild_model", autonomous=True)
    assert v.route == "candidate_pipeline"
    assert v.allowed is False
    assert v.level == PermissionLevel.CODE_MODEL_CHANGE


def test_kill_switch_action_escalates(gate):
    v = gate.evaluate("clear_futures_kill", autonomous=True)
    assert v.route == "escalate"
    assert v.allowed is False


def test_unknown_action_fails_closed(gate):
    v = gate.evaluate("rm_rf_slash", autonomous=True)
    assert v.route == "block"
    assert v.allowed is False
    assert v.level == PermissionLevel.UNKNOWN


def test_self_heal_classification_safety_critical():
    g = ActionGate(SELF_HEAL_ACTION_LEVELS)
    # Clearing a kill switch / halt re-enables risk -> must escalate, never run.
    for action in ("_fix_futures_kill", "_fix_agent_halt", "_fix_system_blocked"):
        v = g.evaluate(action, autonomous=True)
        assert v.route == "escalate", action
        assert v.allowed is False, action
    # Operational restarts run autonomously.
    assert g.evaluate("_fix_stale_collectors", autonomous=True).route == "execute"
    # Model rebuild / package install -> candidate pipeline, not direct.
    for action in ("_fix_venv", "_fix_rebuild_labels", "_fix_ml_stale"):
        assert g.evaluate(action, autonomous=True).route == "candidate_pipeline", action
    # An unmapped fix fails closed.
    assert g.evaluate("_fix_generic", autonomous=True).route == "block"


def test_llm_actions_never_execute_autonomously():
    g = ActionGate(LLM_ACTION_LEVELS)
    # The arbitrary-shell path and code mutations must not run autonomously.
    for action in ("run_command", "write_script", "install_package", "fetch_tool"):
        v = g.evaluate(action, autonomous=True, has_rollback=False)
        assert v.route == "candidate_pipeline", action
        assert v.allowed is False, action
    # Portfolio edits / flag clears escalate (could touch a kill switch).
    for action in ("edit_portfolio", "clear_flag"):
        assert g.evaluate(action, autonomous=True).route == "escalate", action
    # Param changes go through the param gate, not direct write.
    assert g.evaluate("set_param", autonomous=True).route == "param_gate"
    # No LLM act_type is allowed to execute directly.
    for action in LLM_ACTION_LEVELS:
        assert g.evaluate(action, autonomous=True, has_rollback=False).allowed is False, action


def test_decision_log_records(gate):
    class FakeLog:
        def __init__(self):
            self.entries = []

        def append(self, payload):
            self.entries.append(payload)

    log = FakeLog()
    g = ActionGate(LEVELS, decision_log=log)
    g.evaluate("rebuild_model", autonomous=True)
    assert any(e.get("event") == "action_gate" for e in log.entries)
