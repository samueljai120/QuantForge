"""Parameter-proposal gate — the trustworthy road for L2 parameter changes.

The reflect daemon (and any LLM loop that proposes a parameter change) routes
through this instead of writing ``qf_strategy_params.json`` directly. Stages:

    schema check  ->  fail-closed backtest gate  ->  atomic apply + decision log

Outcomes:
* **applied**   — autonomous-allowed param, schema + backtest passed, written atomically.
* **escalated** — risk limit / kill switch (``autonomous_allowed = False``): never
  applied autonomously; flagged for human approval.
* **rejected**  — unregistered / wrong-type / out-of-range, or the backtest gate
  blocked it (including any gate exception — fail closed).

Reuses ``qf_safety.param_schema`` (schema), the fail-closed backtest gate
(injected), ``qf_safety.atomic_json`` (atomic write), and the decision log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from qf_safety.atomic_json import file_lock, read_json, atomic_write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# EWAA single-write authority: only ewaa_proposer may write the regime_weight_table
# and the active capital-lane weight keys. Other authors (reflect, self_heal,
# weight_learner) escalate on these. EXEMPT (Option B, verified live): fixed_alloc_pct
# stays reflect-owned; ml_btc_weight stays agent/ml-owned — blocking them would
# regress reflect's nightly re-risk and freeze ml_btc_weight forever.
WEIGHT_AUTHORITY = {"regime_weight_table": "ewaa_proposer"}
WEIGHT_KEYS_BLOCKED = {"mr_weight", "ml_scanner_weight", "futures_weight", "funding_arb_weight"}
WEIGHT_AUTHORITY_EXEMPT = {"fixed_alloc_pct", "ml_btc_weight"}


@dataclass(frozen=True)
class ProposalResult:
    status: str  # "applied" | "escalated" | "rejected"
    stage: str   # "schema" | "gate" | "apply" | "escalate"
    key: str
    value: Any
    reason: str


def atomic_param_applier(params_path: str, *, modified_by: str = "quantforge_reflect"):
    """Production applier: atomically set one key in the flat params file under a
    lock, preserving all other keys."""
    def _apply(key: str, value: Any) -> None:
        with file_lock(params_path + ".lock"):
            current = read_json(params_path, default={})
            if not isinstance(current, dict):
                current = {}
            current[key] = value
            current["_last_modified_at"] = _now_iso()
            current["_last_modified_by"] = modified_by
            atomic_write_json(params_path, current)

    return _apply


class ParameterProposalGate:
    def __init__(
        self,
        *,
        registry,
        current_params_loader: Callable[[], Dict[str, Any]],
        gate_runner: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        applier: Callable[[str, Any], None],
        decision_log=None,
    ):
        self.registry = registry
        self.current_params_loader = current_params_loader
        self.gate_runner = gate_runner
        self.applier = applier
        self.decision_log = decision_log

    def _log(self, result: ProposalResult, author: str, reasoning: str) -> None:
        if self.decision_log is not None:
            self.decision_log.append({
                "event": "param_proposal",
                "author": author,
                "key": result.key,
                "value": result.value,
                "status": result.status,
                "stage": result.stage,
                "reason": result.reason,
                "reasoning": reasoning[:500],
                "ts": _now_iso(),
            })

    def propose(self, key: str, value: Any, *, author: str, reasoning: str = "") -> ProposalResult:
        # 0. Single-write authority (EWAA). regime_weight_table + active weight keys
        #    may only be written by ewaa_proposer; other authors escalate. Exempt
        #    keys (fixed_alloc_pct, ml_btc_weight) pass through to normal handling.
        if key not in WEIGHT_AUTHORITY_EXEMPT:
            owner = WEIGHT_AUTHORITY.get(key)
            if (owner is not None and author != owner) or (key in WEIGHT_KEYS_BLOCKED and author != "ewaa_proposer"):
                result = ProposalResult(
                    "escalated", "authority", key, value,
                    f"{key} is single-write (ewaa_proposer only); author '{author}' escalated",
                )
                self._log(result, author, reasoning)
                return result

        # 1. Schema. Distinguish a non-autonomous (risk/kill-switch) param — which
        #    should ESCALATE — from an outright invalid one — which is REJECTED.
        schema = self.registry.validate_change(key, value, autonomous=True)
        if not schema.approved:
            spec = self.registry.spec(key)
            if spec is not None and not spec.autonomous_allowed:
                result = ProposalResult(
                    "escalated", "escalate", key, value,
                    f"{key} (risk_class={spec.risk_class}) requires human approval",
                )
            else:
                result = ProposalResult("rejected", "schema", key, value, schema.reason)
            self._log(result, author, reasoning)
            return result

        # 2. Fail-closed backtest gate.
        current = self.current_params_loader()
        proposed_copy = dict(current)
        proposed_copy[key] = value
        try:
            gate = self.gate_runner(proposed_copy, current)
        except Exception as e:  # noqa: BLE001 — any gate failure rejects
            result = ProposalResult("rejected", "gate", key, value,
                                    f"gate error ({type(e).__name__}): {str(e)[:100]}")
            self._log(result, author, reasoning)
            return result
        if not isinstance(gate, dict) or gate.get("approved") is not True:
            reason = gate.get("reason", "gate blocked") if isinstance(gate, dict) else "malformed gate output"
            result = ProposalResult("rejected", "gate", key, value, reason)
            self._log(result, author, reasoning)
            return result

        # 3. Atomic apply.
        try:
            self.applier(key, value)
        except Exception as e:  # noqa: BLE001
            result = ProposalResult("rejected", "apply", key, value,
                                    f"apply failed: {str(e)[:100]}")
            self._log(result, author, reasoning)
            return result

        result = ProposalResult("applied", "apply", key, value,
                                "passed schema + backtest, applied atomically")
        self._log(result, author, reasoning)
        return result
