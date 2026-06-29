"""Model card — full provenance for every model result (Phase 1).

A model result you cannot fully describe is a result you cannot reproduce or
trust. A card must declare its data lineage (dataset/feature/label version), the
chronological train/val/test split, the exact code commit, hyperparameters, RNG
seeds, the costs assumed in evaluation, the metrics, and benchmark results.
Anything missing → the card is rejected and the model cannot be registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

# Fields that must be present AND non-empty for a card to be valid.
REQUIRED_FIELDS = {
    "model_id",
    "model_type",
    "artifact_hash",
    "dataset_version",
    "feature_version",
    "label_version",
    "train_period",
    "val_period",
    "test_period",
    "code_commit",
    "hyperparameters",
    "random_seeds",
    "costs_assumed",
    "metrics",
    "benchmark_results",
}


class IncompleteModelCard(ValueError):
    """Raised when a model card is missing required provenance."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    model_type: str
    artifact_hash: str
    dataset_version: str
    feature_version: str
    label_version: str
    train_period: str
    val_period: str
    test_period: str
    code_commit: str
    hyperparameters: Dict[str, Any]
    random_seeds: Dict[str, Any]
    costs_assumed: Dict[str, Any]
    metrics: Dict[str, Any]
    benchmark_results: Dict[str, Any]
    created_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        missing = []
        for fname in REQUIRED_FIELDS:
            if fname not in data:
                missing.append(fname)
                continue
            value = data[fname]
            # Empty string / empty dict / None all count as missing — a declared
            # but empty costs/metrics block is not acceptable provenance.
            if value is None or (isinstance(value, (str, dict, list)) and len(value) == 0):
                missing.append(fname)
        if missing:
            raise IncompleteModelCard(
                f"model card missing/empty required fields: {sorted(missing)}"
            )
        return cls(
            model_id=data["model_id"],
            model_type=data["model_type"],
            artifact_hash=data["artifact_hash"],
            dataset_version=data["dataset_version"],
            feature_version=data["feature_version"],
            label_version=data["label_version"],
            train_period=data["train_period"],
            val_period=data["val_period"],
            test_period=data["test_period"],
            code_commit=data["code_commit"],
            hyperparameters=dict(data["hyperparameters"]),
            random_seeds=dict(data["random_seeds"]),
            costs_assumed=dict(data["costs_assumed"]),
            metrics=dict(data["metrics"]),
            benchmark_results=dict(data["benchmark_results"]),
            created_at=data.get("created_at") or _now_iso(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "artifact_hash": self.artifact_hash,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "train_period": self.train_period,
            "val_period": self.val_period,
            "test_period": self.test_period,
            "code_commit": self.code_commit,
            "hyperparameters": dict(self.hyperparameters),
            "random_seeds": dict(self.random_seeds),
            "costs_assumed": dict(self.costs_assumed),
            "metrics": dict(self.metrics),
            "benchmark_results": dict(self.benchmark_results),
            "created_at": self.created_at,
        }
