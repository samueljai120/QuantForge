"""Code-enforced action permissions — Phase 0.4.

The audit found the self-heal and code-evolver LLM paths are bounded only by
*prompt text*, not code. An LLM that ignores its instructions (or is steered by
prompt injection) could attempt high-risk actions. This module makes the bounds
deterministic: a small, auditable function decides whether an action may run,
independent of any model output.

Levels (from the mandate)
-------------------------
* L0 OBSERVE            — read logs/metrics/docs, generate reports. Autonomous.
* L1 REVERSIBLE_OP      — restart a collector, retry idempotent task, switch to
                          an approved public fallback, clear caches. Autonomous
                          *iff* a rollback is defined.
* L2 CONFIG_PROPOSAL    — propose a param change / disable a strategy / retrain.
                          May be proposed autonomously but must pass validation.
* L3 CODE_MODEL_CHANGE  — modify source, add deps, replace a model, change an
                          action-controlling prompt, add a strategy. Requires
                          isolated testing + INDEPENDENT human approval.
* L4 FINANCIAL_SECURITY — enable real trading, change credentials, raise risk
                          limits, move funds, change withdrawals, disable
                          monitoring or a kill switch. NEVER autonomous.

Unknown actions fail closed (treated as deny).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional


class PermissionLevel(IntEnum):
    OBSERVE = 0
    REVERSIBLE_OP = 1
    CONFIG_PROPOSAL = 2
    CODE_MODEL_CHANGE = 3
    FINANCIAL_SECURITY = 4
    UNKNOWN = 99


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    level: PermissionLevel
    action: str
    reason: str
    requires_validation: bool = False
    requires_human_approval: bool = False


# Default action -> level registry. Names mirror the mandate's examples.
DEFAULT_REGISTRY: Dict[str, PermissionLevel] = {
    # L0 — observe
    "read_logs": PermissionLevel.OBSERVE,
    "read_metrics": PermissionLevel.OBSERVE,
    "read_docs": PermissionLevel.OBSERVE,
    "generate_report": PermissionLevel.OBSERVE,
    # L1 — reversible operational
    "restart_collector": PermissionLevel.REVERSIBLE_OP,
    "retry_idempotent_task": PermissionLevel.REVERSIBLE_OP,
    "switch_backup_public_source": PermissionLevel.REVERSIBLE_OP,
    "clear_temp_cache": PermissionLevel.REVERSIBLE_OP,
    # L2 — configuration proposal
    "propose_param_change": PermissionLevel.CONFIG_PROPOSAL,
    "propose_disable_strategy": PermissionLevel.CONFIG_PROPOSAL,
    "propose_retrain_model": PermissionLevel.CONFIG_PROPOSAL,
    # L3 — code or model change
    "modify_source_code": PermissionLevel.CODE_MODEL_CHANGE,
    "add_dependency": PermissionLevel.CODE_MODEL_CHANGE,
    "replace_model": PermissionLevel.CODE_MODEL_CHANGE,
    "change_action_prompt": PermissionLevel.CODE_MODEL_CHANGE,
    "add_strategy": PermissionLevel.CODE_MODEL_CHANGE,
    "install_package": PermissionLevel.CODE_MODEL_CHANGE,
    # L4 — financial or security (never autonomous)
    "enable_real_trading": PermissionLevel.FINANCIAL_SECURITY,
    "change_credentials": PermissionLevel.FINANCIAL_SECURITY,
    "increase_risk_limit": PermissionLevel.FINANCIAL_SECURITY,
    "transfer_funds": PermissionLevel.FINANCIAL_SECURITY,
    "change_withdrawal": PermissionLevel.FINANCIAL_SECURITY,
    "disable_monitoring": PermissionLevel.FINANCIAL_SECURITY,
    "disable_kill_switch": PermissionLevel.FINANCIAL_SECURITY,
}


class PermissionEnforcer:
    """Deterministic gate over actions. No model output influences the verdict."""

    def __init__(self, registry: Optional[Dict[str, PermissionLevel]] = None):
        # An explicit registry fully replaces the defaults (so callers can pin an
        # allowlist); otherwise the mandate's default registry is used.
        self._registry = dict(registry) if registry is not None else dict(DEFAULT_REGISTRY)

    def level_of(self, action: str) -> PermissionLevel:
        return self._registry.get(action, PermissionLevel.UNKNOWN)

    def check(
        self,
        action: str,
        *,
        autonomous: bool = True,
        has_rollback: bool = False,
        human_approved: bool = False,
    ) -> PermissionDecision:
        level = self.level_of(action)

        if level == PermissionLevel.UNKNOWN:
            return PermissionDecision(
                allowed=False,
                level=level,
                action=action,
                reason="unregistered action — fail closed",
            )

        if level == PermissionLevel.FINANCIAL_SECURITY:
            allowed = (not autonomous) and human_approved
            return PermissionDecision(
                allowed=allowed,
                level=level,
                action=action,
                reason=(
                    "financial/security action: requires human execution with approval"
                    if not allowed
                    else "approved human-executed financial/security action"
                ),
                requires_human_approval=True,
            )

        if level == PermissionLevel.CODE_MODEL_CHANGE:
            allowed = (not autonomous) and human_approved
            return PermissionDecision(
                allowed=allowed,
                level=level,
                action=action,
                reason=(
                    "code/model change: requires isolated testing + independent approval"
                    if not allowed
                    else "approved code/model change"
                ),
                requires_human_approval=True,
            )

        if level == PermissionLevel.CONFIG_PROPOSAL:
            return PermissionDecision(
                allowed=True,
                level=level,
                action=action,
                reason="proposal allowed; apply must pass fail-closed validation",
                requires_validation=True,
            )

        if level == PermissionLevel.REVERSIBLE_OP:
            return PermissionDecision(
                allowed=bool(has_rollback),
                level=level,
                action=action,
                reason=(
                    "reversible op with defined rollback"
                    if has_rollback
                    else "reversible op rejected: no rollback defined"
                ),
            )

        # OBSERVE
        return PermissionDecision(
            allowed=True,
            level=level,
            action=action,
            reason="observe-only action",
        )
