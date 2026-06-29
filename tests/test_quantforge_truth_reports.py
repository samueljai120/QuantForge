#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

import quantforge_diagnose as qd
import quantforge_monitor as qm


class QuantforgeTruthReportTests(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_monitor_and_diagnose_surface_threshold_miss_bottlenecks(self):
        now = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            portfolio_path = os.path.join(tmpdir, "portfolio.json")
            last_scan_path = os.path.join(tmpdir, "last_scan.json")
            governance_path = os.path.join(tmpdir, "governance-report.json")
            diagnosis_path = os.path.join(tmpdir, "diagnosis-report.json")
            monitor_path = os.path.join(tmpdir, "monitor-report.json")
            promotion_path = os.path.join(tmpdir, "promotion_report.json")
            history_path = os.path.join(tmpdir, "history-summary.json")
            trades_path = os.path.join(tmpdir, "paper-trades.jsonl")

            self._write_json(portfolio_path, {"updated": now, "positions": {}})
            self._write_json(
                last_scan_path,
                {
                    "ts": now,
                    "signals": [],
                    "feedback": {"summary": {"risk_mult": 1.0}},
                    "regime": {"label": "BEAR", "score": -0.4, "entropy": 0.6, "entropy_label": "ORDERLY"},
                    "flow": {"model_no_signal": 38, "threshold_miss": 38, "trained_pair_blocked": 0},
                    "results": [
                        {"symbol": "BTC-USDT", "status": "hold", "reason": "LONG conf 0.46 < 0.61", "decision_stage": "threshold_miss"},
                        {"symbol": "PEPE-USDT", "status": "skip", "reason": "quality filter - price too low"},
                    ],
                },
            )
            self._write_json(governance_path, {"recommendation": "REVIEW", "paper": {"total_pnl_pct": 0.0}, "recent_closes": {}})
            self._write_json(diagnosis_path, {"causes": []})
            self._write_json(monitor_path, {"generated_at": now, "health": "DRIFTING"})
            self._write_json(promotion_path, {"overall_decision": "KEEP_IN_PAPER"})
            self._write_json(history_path, {"status": "ok", "posture": "degraded", "cycles_sampled": 96, "ratios": {}, "averages": {}, "counts": {"diagnosis_causes": {}}})
            with open(trades_path, "w") as f:
                f.write("")

            old_monitor = (
                qm.PORTFOLIO_FILE,
                qm.LAST_SCAN_FILE,
                qm.GOVERNANCE_FILE,
                qm.DIAGNOSIS_FILE,
                qm.TRADES_FILE,
            )
            old_diagnose = (
                qd.PORTFOLIO_FILE,
                qd.LAST_SCAN_FILE,
                qd.GOVERNANCE_FILE,
                qd.PROMOTION_FILE,
                qd.HISTORY_FILE,
                qd.TRADES_FILE,
            )
            try:
                qm.PORTFOLIO_FILE = portfolio_path
                qm.LAST_SCAN_FILE = last_scan_path
                qm.GOVERNANCE_FILE = governance_path
                qm.DIAGNOSIS_FILE = diagnosis_path
                qm.TRADES_FILE = trades_path

                qd.PORTFOLIO_FILE = portfolio_path
                qd.LAST_SCAN_FILE = last_scan_path
                qd.GOVERNANCE_FILE = governance_path
                qd.PROMOTION_FILE = promotion_path
                qd.HISTORY_FILE = history_path
                qd.TRADES_FILE = trades_path

                monitor_report = qm.build_report()
                diagnose_report = qd.build_report()
            finally:
                (
                    qm.PORTFOLIO_FILE,
                    qm.LAST_SCAN_FILE,
                    qm.GOVERNANCE_FILE,
                    qm.DIAGNOSIS_FILE,
                    qm.TRADES_FILE,
                ) = old_monitor
                (
                    qd.PORTFOLIO_FILE,
                    qd.LAST_SCAN_FILE,
                    qd.GOVERNANCE_FILE,
                    qd.PROMOTION_FILE,
                    qd.HISTORY_FILE,
                    qd.TRADES_FILE,
                ) = old_diagnose

        self.assertIn("model_no_signal_bottleneck", monitor_report["drift_flags"])
        self.assertIn("threshold_miss_bottleneck", monitor_report["drift_flags"])
        self.assertIn("LONG conf 0.46 < 0.61", monitor_report["blocked_reasons"])
        self.assertEqual(monitor_report["summary"]["model_no_signal"], 38)
        self.assertEqual(monitor_report["summary"]["threshold_miss"], 38)

        self.assertIn("selection_filter_active", diagnose_report["causes"])
        self.assertIn("model_no_signal_bottleneck", diagnose_report["causes"])
        self.assertIn("threshold_miss_bottleneck", diagnose_report["causes"])
        self.assertEqual(diagnose_report["summary"]["model_no_signal"], 38)
        self.assertEqual(diagnose_report["summary"]["threshold_miss"], 38)
        self.assertIn("threshold_miss", diagnose_report["evidence"]["hold_stages"])


if __name__ == "__main__":
    unittest.main()
