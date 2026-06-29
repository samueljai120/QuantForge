#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import quantforge_paper as qfp


class PaperScanArtifactTests(unittest.TestCase):
    def test_save_last_scan_flattens_flow_and_counts_hold_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "last_scan.json")
            old_path = qfp.LAST_SCAN_FILE
            try:
                qfp.LAST_SCAN_FILE = path
                report = {
                    "ts": "2026-06-25T18:00:00+00:00",
                    "signals": [{"symbol": "ADA-USDT", "side": "SELL", "score": 0.61}],
                    "results": [
                        {"symbol": "BTC-USDT", "status": "hold", "reason": "LONG conf 0.456 < 0.570"},
                        {"symbol": "ETH-USDT", "status": "skip", "reason": "quality filter - history too short"},
                        {"symbol": "ADA-USDT", "status": "signal", "reason": "ML SHORT confidence 0.610 >= threshold 0.550"},
                    ],
                    "flow": {
                        "scan_top_n": 48,
                        "screened_universe": 48,
                        "quality_passed": 35,
                        "quality_blocked": 13,
                        "model_no_signal": 34,
                        "threshold_miss": 33,
                        "trained_pair_blocked": 1,
                        "ml_gate_blocked": 0,
                        "selection_blocked": 0,
                        "feedback_blocked": 0,
                        "actionable_signals": 1,
                        "buy_signals": 0,
                        "sell_signals": 1,
                        "open_position_skips": 0,
                        "reentry_cooldown_skips": 0,
                        "loss_lockout_skips": 0,
                        "error_count": 0,
                    },
                }
                qfp.save_last_scan(report)
                with open(path) as f:
                    saved = json.load(f)
            finally:
                qfp.LAST_SCAN_FILE = old_path

        self.assertEqual(saved["scan_top_n"], 48)
        self.assertEqual(saved["model_no_signal"], 34)
        self.assertEqual(saved["actionable"], 1)
        self.assertEqual(saved["signal_count"], 1)
        self.assertEqual(saved["result_count"], 3)
        self.assertEqual(saved["summary"]["counts"]["hold"], 1)
        self.assertEqual(saved["summary"]["counts"]["signal"], 1)
        self.assertIn("LONG conf 0.456 < 0.570", saved["summary"]["blocked_reasons"])

    def test_save_last_execution_backfills_ts_and_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "last_execution.json")
            old_path = qfp.LAST_EXECUTION_FILE
            try:
                qfp.LAST_EXECUTION_FILE = path
                qfp.save_last_execution({
                    "generated_at": "2026-06-27T02:41:17.801312+00:00",
                    "autopilot_mode": "run_candidate_paper_trial",
                    "executed_count": 0,
                })
                with open(path) as f:
                    saved = json.load(f)
            finally:
                qfp.LAST_EXECUTION_FILE = old_path

        self.assertEqual(saved["ts"], saved["generated_at"])
        self.assertEqual(saved["mode"], "run_candidate_paper_trial")


if __name__ == "__main__":
    unittest.main()
