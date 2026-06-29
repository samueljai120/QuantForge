"""Model registry + promotion state machine (Phase 1).

State machine::

    RESEARCH -> CANDIDATE -> VALIDATED -> SHADOW -> CANARY -> PAPER_PRODUCTION
                                                                  -> RETIRED
    (any active state) -> ROLLED_BACK

Promotion into VALIDATED or beyond requires **economic** evidence (out-of-sample
value / risk-adjusted return) AND statistical significance — a model may NOT be
promoted on AUC alone. Persistence is atomic and versioned via qf_safety.CASStore
so concurrent writers cannot corrupt or lose the registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from qf_safety.atomic_json import CASStore
from qf_mlops.model_card import ModelCard


class PromotionState(str, Enum):
    RESEARCH = "research"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    SHADOW = "shadow"
    CANARY = "canary"
    PAPER_PRODUCTION = "paper_production"
    RETIRED = "retired"
    ROLLED_BACK = "rolled_back"


LEGAL_TRANSITIONS: Dict[PromotionState, set] = {
    PromotionState.RESEARCH: {PromotionState.CANDIDATE, PromotionState.ROLLED_BACK},
    PromotionState.CANDIDATE: {PromotionState.VALIDATED, PromotionState.ROLLED_BACK},
    PromotionState.VALIDATED: {PromotionState.SHADOW, PromotionState.ROLLED_BACK},
    PromotionState.SHADOW: {PromotionState.CANARY, PromotionState.ROLLED_BACK},
    PromotionState.CANARY: {PromotionState.PAPER_PRODUCTION, PromotionState.ROLLED_BACK},
    PromotionState.PAPER_PRODUCTION: {PromotionState.RETIRED, PromotionState.ROLLED_BACK},
    PromotionState.RETIRED: set(),
    PromotionState.ROLLED_BACK: set(),
}

# Promotions INTO these states require economic + statistical evidence.
EVIDENCE_REQUIRED = {
    PromotionState.VALIDATED,
    PromotionState.SHADOW,
    PromotionState.CANARY,
    PromotionState.PAPER_PRODUCTION,
}

# Any one of these counts as economic (not classification-only) evidence.
ECONOMIC_KEYS = {
    "incremental_value",
    "net_return_after_costs",
    "sharpe",
    "sortino",
    "risk_adjusted_return",
}


class IllegalTransition(Exception):
    pass


class PromotionEvidenceError(Exception):
    pass


class DuplicateModel(Exception):
    pass


class ProductionSlotOccupied(Exception):
    pass


class ModelNotFound(KeyError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_economic_evidence(evidence: Any) -> bool:
    if not isinstance(evidence, dict):
        return False
    has_econ = any(k in evidence for k in ECONOMIC_KEYS)
    sig = evidence.get("statistical_significance")
    if isinstance(sig, dict):
        has_sig = sig.get("significant") is True
    else:
        has_sig = bool(sig)
    return has_econ and has_sig


class ModelRegistry:
    def __init__(self, path: str):
        self._store = CASStore(path)

    def _models(self) -> Dict[str, Any]:
        return self._store.load()["data"]

    def register(self, card: ModelCard) -> None:
        # The duplicate check runs INSIDE the locked mutator so two concurrent
        # registrations of the same id cannot both succeed (atomic, no TOCTOU).
        def mutator(data):
            if card.model_id in data:
                raise DuplicateModel(f"model already registered: {card.model_id}")
            data[card.model_id] = {
                "card": card.to_dict(),
                "state": PromotionState.RESEARCH.value,
                "history": [
                    {"from": None, "to": PromotionState.RESEARCH.value,
                     "actor": "register", "ts": _now_iso(), "evidence": {}}
                ],
            }
            return data

        self._store.update(mutator, actor="register")

    def get(self, model_id: str) -> Dict[str, Any]:
        models = self._models()
        if model_id not in models:
            raise ModelNotFound(model_id)
        entry = models[model_id]
        return {
            "card": entry["card"],
            "state": PromotionState(entry["state"]),
            "history": entry["history"],
        }

    def promote(
        self,
        model_id: str,
        to_state: PromotionState,
        *,
        evidence: Dict[str, Any],
        actor: str,
    ) -> None:
        to_state = PromotionState(to_state)

        # All validation runs INSIDE the locked mutator so the transition check,
        # evidence check, and single-production invariant are evaluated against
        # the committed state under the lock (atomic; no TOCTOU race).
        def mutator(data):
            if model_id not in data:
                raise ModelNotFound(model_id)
            from_state = PromotionState(data[model_id]["state"])

            if to_state not in LEGAL_TRANSITIONS[from_state]:
                raise IllegalTransition(
                    f"{from_state.value} -> {to_state.value} is not allowed"
                )

            if to_state in EVIDENCE_REQUIRED and not _has_economic_evidence(evidence):
                raise PromotionEvidenceError(
                    f"cannot promote {model_id} to {to_state.value} on AUC alone: "
                    f"need economic value ({sorted(ECONOMIC_KEYS)}) AND statistical significance"
                )

            if to_state == PromotionState.PAPER_PRODUCTION:
                for other_id, entry in data.items():
                    if other_id != model_id and entry["state"] == PromotionState.PAPER_PRODUCTION.value:
                        raise ProductionSlotOccupied(
                            f"{other_id} already in paper_production; retire or roll it back first"
                        )

            data[model_id]["state"] = to_state.value
            data[model_id]["history"].append({
                "from": from_state.value,
                "to": to_state.value,
                "actor": actor,
                "ts": _now_iso(),
                "evidence": evidence,
            })
            return data

        self._store.update(mutator, actor=actor)

    def rollback(self, model_id: str, *, actor: str, reason: str) -> None:
        self.promote(
            model_id,
            PromotionState.ROLLED_BACK,
            evidence={"reason": reason},
            actor=actor,
        )

    def current_production(self) -> Optional[str]:
        for model_id, entry in self._models().items():
            if entry["state"] == PromotionState.PAPER_PRODUCTION.value:
                return model_id
        return None
