#!/usr/bin/env python3
import unittest

import quantforge_model_layer_report as qmlr


class QuantforgeModelLayerExecutedSubsetTests(unittest.TestCase):
    def test_prediction_ready_for_positive_executed_subset_with_bounded_failures(self):
        ready, details = qmlr._prediction_layer_ready(
            {"gate_pass": False},
            {
                "status": "ready",
                "analysis_mode": "executed_subset_ranked",
                "trade_count": 8961,
                "summary": {"net_edge_bps_mean": 58.75},
                "failing_segments": [
                    {"dimension": "regime_bucket", "segment": "fragile"},
                    {"dimension": "setup_tag", "segment": "exhaustion_short"},
                    {"dimension": "direction", "segment": "short"},
                ],
            },
        )

        self.assertTrue(ready)
        self.assertEqual(details["ready_basis"], "executed_subset_ranked_positive")

    def test_prediction_not_ready_for_unbounded_failing_slice(self):
        ready, details = qmlr._prediction_layer_ready(
            {"gate_pass": False},
            {
                "status": "ready",
                "analysis_mode": "executed_subset_ranked",
                "trade_count": 8961,
                "summary": {"net_edge_bps_mean": 58.75},
                "failing_segments": [
                    {"dimension": "setup_tag", "segment": "generic_long"},
                ],
            },
        )

        self.assertFalse(ready)
        self.assertEqual(details["ready_basis"], "not_ready")


if __name__ == "__main__":
    unittest.main(verbosity=2)
