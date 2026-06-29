"""Centralized typed parameter schema — Phase 0.3.

Every tunable parameter must be registered with a type, bounds, owner, risk
class, and whether autonomous modification is permitted. Changes are validated
against this schema *before* any backtest gate runs. Unregistered parameters and
out-of-bounds / wrong-type values are rejected (fail closed). Risk limits and
kill switches are flagged ``autonomous_allowed=False`` and can only change with a
human in the loop.

This is the schema layer. It is necessary but not sufficient: an in-range,
autonomous-allowed change still has to pass the fail-closed backtest gate
(``qf_validate_tune``) before being applied.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # "float" | "int" | "bool" | "str"
    default: Any
    owner: str
    risk_class: str  # "low" | "medium" | "high" | "kill_switch"
    autonomous_allowed: bool
    rollback_value: Any
    approval_required: bool
    min: Optional[float] = None
    max: Optional[float] = None
    required_validation_tests: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    approved: bool
    reason: str
    key: str


def _type_ok(spec_type: str, value: Any) -> bool:
    # bool is a subclass of int — reject it for numeric/int params explicitly.
    if isinstance(value, bool):
        return spec_type == "bool"
    if spec_type == "int":
        return isinstance(value, int)
    if spec_type == "float":
        if not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
    if spec_type == "bool":
        return isinstance(value, bool)
    if spec_type == "str":
        return isinstance(value, str)
    if spec_type == "object":
        # Structured params (e.g. regime_weight_table). Accept a dict; range is
        # not applicable. Deeper structural validation is the agent's job.
        return isinstance(value, dict)
    return False


class ParamRegistry:
    def __init__(self, specs: Dict[str, ParamSpec]):
        self._specs = dict(specs)

    @classmethod
    def from_dict(cls, raw: Dict[str, Dict[str, Any]]) -> "ParamRegistry":
        specs = {}
        for name, d in raw.items():
            specs[name] = ParamSpec(
                name=name,
                type=d["type"],
                default=d.get("default"),
                owner=d.get("owner", "unknown"),
                risk_class=d.get("risk_class", "high"),
                autonomous_allowed=bool(d.get("autonomous_allowed", False)),
                rollback_value=d.get("rollback_value", d.get("default")),
                approval_required=bool(d.get("approval_required", True)),
                min=d.get("min"),
                max=d.get("max"),
                required_validation_tests=list(d.get("required_validation_tests", [])),
            )
        return cls(specs)

    @classmethod
    def from_file(cls, path: str) -> "ParamRegistry":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def spec(self, key: str) -> Optional[ParamSpec]:
        return self._specs.get(key)

    def keys(self) -> List[str]:
        return list(self._specs.keys())

    def validate_change(
        self, key: str, value: Any, *, autonomous: bool = True
    ) -> ValidationResult:
        spec = self._specs.get(key)
        if spec is None:
            return ValidationResult(False, f"unregistered parameter: {key}", key)

        if not _type_ok(spec.type, value):
            return ValidationResult(
                False, f"type mismatch: expected {spec.type}, got {type(value).__name__}", key
            )

        if autonomous and not spec.autonomous_allowed:
            return ValidationResult(
                False,
                f"parameter '{key}' (risk_class={spec.risk_class}) is not "
                f"autonomously modifiable; requires human approval",
                key,
            )

        if spec.type in ("int", "float"):
            try:
                numeric = float(value)
            except (OverflowError, ValueError):
                return ValidationResult(False, f"value {value!r} not representable as float", key)
            if spec.min is not None and numeric < spec.min:
                return ValidationResult(
                    False, f"value {value} below minimum {spec.min}", key
                )
            if spec.max is not None and numeric > spec.max:
                return ValidationResult(
                    False, f"value {value} above maximum {spec.max}", key
                )

        return ValidationResult(True, "ok", key)
