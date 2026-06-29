"""Baseline portfolio decomposition — Phase 1 honest attribution.

Run these arms under identical inputs (same data, same costs) and compare:

    A HODL · B RULES_ONLY · C RULES_ML · D ML_ONLY · E CURRENT ·
    F EQUAL_WEIGHT · G RANDOM_ML · H CASH

The single most important number is the **incremental ML value**, C - B (the
system with ML minus the identical system without it). ML only "earns its place"
if it beats both rules-only AND a randomized-prediction control (G) — otherwise
the apparent edge is noise. And buy-and-hold (A) gains are NOT intelligence:
alpha is measured *over* HODL, not in absolute return.

All arm returns passed in must already be net of realistic costs (fees, spread,
slippage, funding). This module does the accounting, not the simulation.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class Arm(str, Enum):
    HODL = "HODL"
    RULES_ONLY = "RULES_ONLY"
    RULES_ML = "RULES_ML"
    ML_ONLY = "ML_ONLY"
    CURRENT = "CURRENT"
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    RANDOM_ML = "RANDOM_ML"
    CASH = "CASH"


# The minimum arms needed to make any honest ML-value statement.
REQUIRED_ARMS = {Arm.HODL, Arm.RULES_ONLY, Arm.RULES_ML, Arm.RANDOM_ML}


def incremental_ml_value(with_ml_return: float, without_ml_return: float) -> float:
    """The core metric: system *with* ML minus the identical system *without* it."""
    return with_ml_return - without_ml_return


def decompose(arms: Dict[Arm, float], *, min_margin: float = 0.0) -> Dict:
    """Decompose arm returns into honest attribution.

    ``min_margin`` is the edge (in the same units as the returns) that ML must
    clear over *both* rules-only and the randomized control to count as adding
    value. Raises ``ValueError`` if a required arm is missing (fail closed — you
    cannot make a claim without the baselines).
    """
    missing = [a for a in REQUIRED_ARMS if a not in arms]
    if missing:
        raise ValueError(f"missing required arms: {sorted(a.value for a in missing)}")

    hodl = arms[Arm.HODL]
    rules = arms[Arm.RULES_ONLY]
    rules_ml = arms[Arm.RULES_ML]
    random_ml = arms[Arm.RANDOM_ML]

    inc = incremental_ml_value(rules_ml, rules)
    ml_vs_random = rules_ml - random_ml

    warnings: List[str] = []
    if ml_vs_random <= min_margin:
        warnings.append(
            "ML does not beat randomized predictions beyond the margin — likely no real signal"
        )
    if inc <= min_margin:
        warnings.append("ML does not improve return vs rules-only beyond the margin")

    ml_adds_value = (inc > min_margin) and (ml_vs_random > min_margin)

    return {
        "incremental_ml_value": inc,
        "ml_vs_random": ml_vs_random,
        "rules_alpha_vs_hodl": rules - hodl,
        "system_alpha_vs_hodl": rules_ml - hodl,
        "beats_hodl": rules_ml > hodl,
        "ml_adds_value": ml_adds_value,
        "warnings": warnings,
        "arms": {a.value: v for a, v in arms.items()},
    }
