#!/usr/bin/env python3
"""Shared QuantForge parameter loading helpers.

Legacy paper/research tools still read `strategy-params.json`, while the live
BTC agent and newer control loops read `qf_strategy_params.json`. This helper
merges both so paper-side consumers can see governed runtime params without
dropping legacy paper-only knobs.
"""

from __future__ import annotations

import json
import os

from config import cfg

DATA_DIR = os.path.join(cfg.data, "quantforge")
LEGACY_PARAMS_FILE = os.path.join(DATA_DIR, "strategy-params.json")
QF_PARAMS_FILE = os.path.join(DATA_DIR, "qf_strategy_params.json")


def _read_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def load_merged_quantforge_params() -> dict:
    """Return legacy paper params overlaid with non-null qf runtime params.

    `strategy-params.json` remains the source for older paper-specific keys.
    `qf_strategy_params.json` overlays it only when a key is explicitly set.
    """

    merged = _read_json(LEGACY_PARAMS_FILE)
    qf_params = _read_json(QF_PARAMS_FILE)
    for key, value in qf_params.items():
        if value is not None:
            merged[key] = value
    return merged
