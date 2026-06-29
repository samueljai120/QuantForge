#!/usr/bin/env python3
"""Regression tests for QuantForge runtime param compatibility."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quantforge_agent as qa


class TestRuntimeParams(unittest.TestCase):
    def setUp(self):
        self._orig_regime_adaptive = qa.REGIME_ADAPTIVE
        self._orig_timesfm_weight = qa.TIMESFM_SIGNAL_WEIGHT

    def tearDown(self):
        qa.REGIME_ADAPTIVE = self._orig_regime_adaptive
        qa.TIMESFM_SIGNAL_WEIGHT = self._orig_timesfm_weight
        qa._rebuild_strategy_registry()

    def test_uppercase_regime_adaptive_alias_is_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = os.path.join(tmpdir, "qf_strategy_params.json")
            with open(params_path, "w") as f:
                f.write('{"REGIME_ADAPTIVE": false}')

            with mock.patch.object(qa, "QF_PARAMS_FILE", params_path), \
                 mock.patch.object(qa, "log"):
                applied = qa.load_runtime_params()

        self.assertFalse(qa.REGIME_ADAPTIVE)
        self.assertIn("regime_adaptive", applied)
        self.assertFalse(applied["regime_adaptive"])

    def test_zero_timesfm_weight_removes_strategy_from_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = os.path.join(tmpdir, "qf_strategy_params.json")
            with open(params_path, "w") as f:
                f.write('{"timesfm_signal_weight": 0.0}')

            with mock.patch.object(qa, "QF_PARAMS_FILE", params_path), \
                 mock.patch.object(qa, "log"):
                applied = qa.load_runtime_params()

        self.assertEqual(qa.TIMESFM_SIGNAL_WEIGHT, 0.0)
        self.assertIn("timesfm_signal_weight", applied)
        self.assertNotIn("timesfm_signal", [s.name for s in qa.STRATEGY_REGISTRY])


if __name__ == "__main__":
    unittest.main(verbosity=2)
