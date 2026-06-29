"""Action gate for self-healing — Phase 8.

Self-heal takes *actions* (restart a collector, rebuild a model, clear a kill
switch), not parameter edits. This gate classifies each action L0-L4 (reusing
``qf_safety.permissions``) and routes it:

* OBSERVE / REVERSIBLE_OP  -> ``execute``  (autonomous; L1 needs a rollback)
* CONFIG_PROPOSAL          -> ``param_gate`` (goes through ParameterProposalGate)
* CODE_MODEL_CHANGE        -> ``candidate_pipeline`` (blocked autonomous)
* FINANCIAL_SECURITY       -> ``escalate`` (clearing a kill switch / halt =
                              re-enabling risk; never autonomous)
* UNKNOWN                  -> ``block`` (fail closed)

``allowed`` is True only for the ``execute`` route — every other action must be
routed, escalated, or blocked, never run directly by the autonomous loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from qf_safety.permissions import PermissionEnforcer, PermissionLevel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Classification of quantforge_self_heal_actions.py auto-fix functions. This is a
# safety decision: anything that clears a kill switch / halt re-enables risk and
# must escalate; model/package changes go through the candidate pipeline; only
# reversible operational fixes run autonomously. Unmapped actions fail closed.
SELF_HEAL_ACTION_LEVELS = {
    # L1 — reversible operational (autonomous OK)
    "_fix_stale_collectors": PermissionLevel.REVERSIBLE_OP,
    "_fix_stale_portfolio": PermissionLevel.REVERSIBLE_OP,
    "_fix_stale_scans": PermissionLevel.REVERSIBLE_OP,
    "_fix_stale_monitor": PermissionLevel.REVERSIBLE_OP,
    "_fix_self_tune": PermissionLevel.REVERSIBLE_OP,
    "_fix_reflect_gate": PermissionLevel.REVERSIBLE_OP,
    "_apply_autopilot_freeze": PermissionLevel.REVERSIBLE_OP,  # freezing is conservative
    # L2 — config proposal
    "_fix_execution_realism": PermissionLevel.CONFIG_PROPOSAL,
    "_fix_narrow_scope": PermissionLevel.CONFIG_PROPOSAL,
    "_fix_market_data": PermissionLevel.CONFIG_PROPOSAL,
    "_fix_research_data": PermissionLevel.CONFIG_PROPOSAL,
    # L3 — code / model / package change (candidate pipeline; blocked autonomous)
    "_fix_venv": PermissionLevel.CODE_MODEL_CHANGE,
    "_fix_ml_stale": PermissionLevel.CODE_MODEL_CHANGE,
    "_fix_rebuild_labels": PermissionLevel.CODE_MODEL_CHANGE,
    "_fix_layer_split": PermissionLevel.CODE_MODEL_CHANGE,
    # L4 — clears a risk control / re-enables risk (escalate; NEVER autonomous)
    "_fix_futures_kill": PermissionLevel.FINANCIAL_SECURITY,
    "_fix_agent_halt": PermissionLevel.FINANCIAL_SECURITY,
    "_fix_system_blocked": PermissionLevel.FINANCIAL_SECURITY,
    "_fix_trim_buyback": PermissionLevel.FINANCIAL_SECURITY,
    # _fix_generic and any unmapped action -> UNKNOWN -> block (fail closed).
}


# Classification of LLM-proposed self-heal act_types (_execute_llm_fixes). Per
# the mandate, LLM self-healing paths must NOT autonomously mutate the system:
# arbitrary shell, script writes, package installs, tool fetches -> candidate
# pipeline; portfolio edits / flag clears (could touch a kill switch) -> escalate;
# parameter changes -> the param gate. None execute directly.
LLM_ACTION_LEVELS = {
    "set_param": PermissionLevel.CONFIG_PROPOSAL,
    "run_command": PermissionLevel.CODE_MODEL_CHANGE,    # arbitrary shell — the worst one
    "write_script": PermissionLevel.CODE_MODEL_CHANGE,
    "install_package": PermissionLevel.CODE_MODEL_CHANGE,
    "fetch_tool": PermissionLevel.CODE_MODEL_CHANGE,
    "edit_portfolio": PermissionLevel.FINANCIAL_SECURITY,
    "clear_flag": PermissionLevel.FINANCIAL_SECURITY,    # may clear a kill switch / halt
}


@dataclass(frozen=True)
class ActionVerdict:
    action: str
    level: PermissionLevel
    allowed: bool   # may execute autonomously right now (route == "execute")
    route: str      # execute | param_gate | candidate_pipeline | escalate | block
    reason: str


class ActionGate:
    def __init__(self, action_levels: Dict[str, PermissionLevel], *, decision_log=None):
        # An explicit registry pins the allowlist; unknown actions fail closed.
        self.enforcer = PermissionEnforcer(registry=dict(action_levels))
        self.decision_log = decision_log

    def evaluate(self, action: str, *, autonomous: bool = True, has_rollback: bool = True) -> ActionVerdict:
        decision = self.enforcer.check(action, autonomous=autonomous, has_rollback=has_rollback)
        level = decision.level

        if level == PermissionLevel.OBSERVE:
            route = "execute"
        elif level == PermissionLevel.REVERSIBLE_OP:
            route = "execute" if decision.allowed else "block"
        elif level == PermissionLevel.CONFIG_PROPOSAL:
            route = "param_gate"
        elif level == PermissionLevel.CODE_MODEL_CHANGE:
            route = "candidate_pipeline"
        elif level == PermissionLevel.FINANCIAL_SECURITY:
            route = "escalate"
        else:  # UNKNOWN
            route = "block"

        verdict = ActionVerdict(
            action=action,
            level=level,
            allowed=(route == "execute"),
            route=route,
            reason=decision.reason,
        )
        if self.decision_log is not None:
            self.decision_log.append({
                "event": "action_gate",
                "action": action,
                "level": int(level),
                "route": route,
                "allowed": verdict.allowed,
                "reason": verdict.reason,
                "ts": _now_iso(),
            })
        return verdict
