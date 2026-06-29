#!/usr/bin/env python3
"""Shared QuantForge signal ranking helpers."""

from __future__ import annotations


def signal_rank_value(row: dict) -> float:
    execution_score = float(
        row.get("execution_score", row.get("score", row.get("ml_confidence", 0.0))) or 0.0
    )
    threshold = float(row.get("entry_threshold", 0.0) or 0.0)
    if threshold > 0:
        return execution_score - threshold
    return execution_score
