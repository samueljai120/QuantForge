#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import quantforge_params as qp


class QuantforgeParamsMergeTests(unittest.TestCase):
    def test_qf_overlays_non_null_values_without_erasing_legacy_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = os.path.join(tmpdir, "strategy-params.json")
            qf = os.path.join(tmpdir, "qf_strategy_params.json")
            with open(legacy, "w") as f:
                json.dump(
                    {
                        "signal_confidence_threshold": 0.7,
                        "scan_top_n": 48,
                        "paper_only_knob": "legacy",
                        "max_open_positions": 0,
                    },
                    f,
                )
            with open(qf, "w") as f:
                json.dump(
                    {
                        "ml_scanner_weight": 0.03,
                        "max_open_positions": None,
                        "timesfm_signal_weight": 0.0,
                    },
                    f,
                )

            old_legacy = qp.LEGACY_PARAMS_FILE
            old_qf = qp.QF_PARAMS_FILE
            try:
                qp.LEGACY_PARAMS_FILE = legacy
                qp.QF_PARAMS_FILE = qf
                merged = qp.load_merged_quantforge_params()
            finally:
                qp.LEGACY_PARAMS_FILE = old_legacy
                qp.QF_PARAMS_FILE = old_qf

        self.assertEqual(merged["signal_confidence_threshold"], 0.7)
        self.assertEqual(merged["scan_top_n"], 48)
        self.assertEqual(merged["paper_only_knob"], "legacy")
        self.assertEqual(merged["max_open_positions"], 0)
        self.assertEqual(merged["ml_scanner_weight"], 0.03)
        self.assertEqual(merged["timesfm_signal_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
