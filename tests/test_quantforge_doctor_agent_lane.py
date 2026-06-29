#!/usr/bin/env python3
"""Regression tests for doctor truth around the live agent lane."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

import quantforge_doctor as qd


class QuantforgeDoctorAgentLaneTests(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_agent_lane_failure_blocks_readiness(self):
        now = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "portfolio": os.path.join(tmpdir, "portfolio.json"),
                "last_scan": os.path.join(tmpdir, "last_scan.json"),
                "governance": os.path.join(tmpdir, "governance-report.json"),
                "monitor": os.path.join(tmpdir, "monitor-report.json"),
                "autopilot": os.path.join(tmpdir, "autopilot-report.json"),
                "lanes": os.path.join(tmpdir, "experiment-lanes.json"),
                "harness": os.path.join(tmpdir, "harness-report.json"),
                "review": os.path.join(tmpdir, "candidate-review.json"),
                "agent_portfolio": os.path.join(tmpdir, "agent_portfolio.json"),
                "invariants": os.path.join(tmpdir, "qf_invariants_state.json"),
                "health": os.path.join(tmpdir, "health.json"),
            }
            self._write_json(paths["portfolio"], {"updated": now})
            self._write_json(paths["last_scan"], {"ts": now})
            self._write_json(paths["governance"], {"generated_at": now})
            self._write_json(paths["monitor"], {"generated_at": now, "health": "OK"})
            self._write_json(paths["autopilot"], {"generated_at": now, "mode": "observe"})
            self._write_json(paths["lanes"], {"candidate_trial": {"status": "ready", "assessment": "supportive"}})
            self._write_json(paths["harness"], {"status": "ok"})
            self._write_json(paths["review"], {"recommendation": "observe"})
            self._write_json(paths["agent_portfolio"], {"futures_kill": False})
            self._write_json(paths["invariants"], {"n_critical": 1, "n_warning": 0, "violations": [{"name": "futures_open_close_parity"}]})
            self._write_json(paths["health"], {"status": "critical", "alerts": ["critical futures kill"]})

            old = (
                qd.PORTFOLIO_FILE,
                qd.LAST_SCAN_FILE,
                qd.GOVERNANCE_FILE,
                qd.MONITOR_FILE,
                qd.AUTOPILOT_FILE,
                qd.LANES_FILE,
                qd.HARNESS_FILE,
                qd.REVIEW_FILE,
                qd.AGENT_PORTFOLIO_FILE,
                qd.INVARIANTS_STATE_FILE,
                qd.WATCHDOG_HEALTH_FILE,
                qd.LEGACY_HEALTH_FILE,
            )
            try:
                qd.PORTFOLIO_FILE = paths["portfolio"]
                qd.LAST_SCAN_FILE = paths["last_scan"]
                qd.GOVERNANCE_FILE = paths["governance"]
                qd.MONITOR_FILE = paths["monitor"]
                qd.AUTOPILOT_FILE = paths["autopilot"]
                qd.LANES_FILE = paths["lanes"]
                qd.HARNESS_FILE = paths["harness"]
                qd.REVIEW_FILE = paths["review"]
                qd.AGENT_PORTFOLIO_FILE = paths["agent_portfolio"]
                qd.INVARIANTS_STATE_FILE = paths["invariants"]
                qd.WATCHDOG_HEALTH_FILE = os.path.join(tmpdir, "missing-watchdog.json")
                qd.LEGACY_HEALTH_FILE = paths["health"]

                report = qd.build_report()
            finally:
                (
                    qd.PORTFOLIO_FILE,
                    qd.LAST_SCAN_FILE,
                    qd.GOVERNANCE_FILE,
                    qd.MONITOR_FILE,
                    qd.AUTOPILOT_FILE,
                    qd.LANES_FILE,
                    qd.HARNESS_FILE,
                    qd.REVIEW_FILE,
                    qd.AGENT_PORTFOLIO_FILE,
                    qd.INVARIANTS_STATE_FILE,
                    qd.WATCHDOG_HEALTH_FILE,
                    qd.LEGACY_HEALTH_FILE,
                ) = old

        self.assertEqual(report["readiness"], "BLOCKED")
        self.assertEqual(report["agent_status"], "FAIL")

    def test_control_param_conflict_blocks_readiness(self):
        now = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "portfolio": os.path.join(tmpdir, "portfolio.json"),
                "last_scan": os.path.join(tmpdir, "last_scan.json"),
                "governance": os.path.join(tmpdir, "governance-report.json"),
                "monitor": os.path.join(tmpdir, "monitor-report.json"),
                "autopilot": os.path.join(tmpdir, "autopilot-report.json"),
                "lanes": os.path.join(tmpdir, "experiment-lanes.json"),
                "harness": os.path.join(tmpdir, "harness-report.json"),
                "review": os.path.join(tmpdir, "candidate-review.json"),
            }
            self._write_json(paths["portfolio"], {"updated": now})
            self._write_json(paths["last_scan"], {"ts": now})
            self._write_json(paths["governance"], {"generated_at": now})
            self._write_json(paths["monitor"], {"generated_at": now, "health": "OK"})
            self._write_json(paths["autopilot"], {"generated_at": now, "mode": "pause_new_entries"})
            self._write_json(paths["lanes"], {"candidate_trial": {}})
            self._write_json(paths["harness"], {"status": "ok"})
            self._write_json(paths["review"], {"recommendation": "observe"})

            old = (
                qd.PORTFOLIO_FILE,
                qd.LAST_SCAN_FILE,
                qd.GOVERNANCE_FILE,
                qd.MONITOR_FILE,
                qd.AUTOPILOT_FILE,
                qd.LANES_FILE,
                qd.HARNESS_FILE,
                qd.REVIEW_FILE,
            )
            try:
                qd.PORTFOLIO_FILE = paths["portfolio"]
                qd.LAST_SCAN_FILE = paths["last_scan"]
                qd.GOVERNANCE_FILE = paths["governance"]
                qd.MONITOR_FILE = paths["monitor"]
                qd.AUTOPILOT_FILE = paths["autopilot"]
                qd.LANES_FILE = paths["lanes"]
                qd.HARNESS_FILE = paths["harness"]
                qd.REVIEW_FILE = paths["review"]

                with mock.patch.object(
                    qd,
                    "load_merged_quantforge_params",
                    return_value={"autopilot_override": "allow_entries", "max_open_positions": 0},
                ):
                    report = qd.build_report()
            finally:
                (
                    qd.PORTFOLIO_FILE,
                    qd.LAST_SCAN_FILE,
                    qd.GOVERNANCE_FILE,
                    qd.MONITOR_FILE,
                    qd.AUTOPILOT_FILE,
                    qd.LANES_FILE,
                    qd.HARNESS_FILE,
                    qd.REVIEW_FILE,
                ) = old

        self.assertEqual(report["readiness"], "BLOCKED")
        conflict = next(row for row in report["checks"] if row["name"] == "control_param_conflict")
        self.assertFalse(conflict["ok"])
        self.assertIn("autopilot_override=allow_entries", conflict["detail"])
        self.assertIn("max_open_positions=0", conflict["detail"])

    def test_queued_paper_trial_with_explicit_caps_clears_zero_cap_conflict(self):
        now = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "portfolio": os.path.join(tmpdir, "portfolio.json"),
                "last_scan": os.path.join(tmpdir, "last_scan.json"),
                "governance": os.path.join(tmpdir, "governance-report.json"),
                "monitor": os.path.join(tmpdir, "monitor-report.json"),
                "autopilot": os.path.join(tmpdir, "autopilot-report.json"),
                "lanes": os.path.join(tmpdir, "experiment-lanes.json"),
                "harness": os.path.join(tmpdir, "harness-report.json"),
                "review": os.path.join(tmpdir, "candidate-review.json"),
            }
            self._write_json(paths["portfolio"], {"updated": now})
            self._write_json(paths["last_scan"], {"ts": now})
            self._write_json(paths["governance"], {"generated_at": now})
            self._write_json(paths["monitor"], {"generated_at": now, "health": "DRIFTING"})
            self._write_json(paths["autopilot"], {"generated_at": now, "mode": "pause_new_entries"})
            self._write_json(
                paths["lanes"],
                {
                    "candidate_trial": {
                        "status": "queued",
                        "paper_only": True,
                        "changes": [
                            {"key": "max_long_positions", "value": 2},
                        ],
                    }
                },
            )
            self._write_json(paths["harness"], {"status": "ok"})
            self._write_json(paths["review"], {"recommendation": "queue_candidate_trial"})

            old = (
                qd.PORTFOLIO_FILE,
                qd.LAST_SCAN_FILE,
                qd.GOVERNANCE_FILE,
                qd.MONITOR_FILE,
                qd.AUTOPILOT_FILE,
                qd.LANES_FILE,
                qd.HARNESS_FILE,
                qd.REVIEW_FILE,
            )
            try:
                qd.PORTFOLIO_FILE = paths["portfolio"]
                qd.LAST_SCAN_FILE = paths["last_scan"]
                qd.GOVERNANCE_FILE = paths["governance"]
                qd.MONITOR_FILE = paths["monitor"]
                qd.AUTOPILOT_FILE = paths["autopilot"]
                qd.LANES_FILE = paths["lanes"]
                qd.HARNESS_FILE = paths["harness"]
                qd.REVIEW_FILE = paths["review"]

                with mock.patch.object(
                    qd,
                    "load_merged_quantforge_params",
                    return_value={"max_open_positions": 0, "max_long_positions": 0, "max_short_positions": 0},
                ):
                    report = qd.build_report()
            finally:
                (
                    qd.PORTFOLIO_FILE,
                    qd.LAST_SCAN_FILE,
                    qd.GOVERNANCE_FILE,
                    qd.MONITOR_FILE,
                    qd.AUTOPILOT_FILE,
                    qd.LANES_FILE,
                    qd.HARNESS_FILE,
                    qd.REVIEW_FILE,
                ) = old

        conflict = next(row for row in report["checks"] if row["name"] == "control_param_conflict")
        self.assertTrue(conflict["ok"])

    def test_queued_paper_trial_with_reduce_to_cap_clears_zero_cap_conflict(self):
        now = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "portfolio": os.path.join(tmpdir, "portfolio.json"),
                "last_scan": os.path.join(tmpdir, "last_scan.json"),
                "governance": os.path.join(tmpdir, "governance-report.json"),
                "monitor": os.path.join(tmpdir, "monitor-report.json"),
                "autopilot": os.path.join(tmpdir, "autopilot-report.json"),
                "lanes": os.path.join(tmpdir, "experiment-lanes.json"),
                "harness": os.path.join(tmpdir, "harness-report.json"),
                "review": os.path.join(tmpdir, "candidate-review.json"),
            }
            self._write_json(paths["portfolio"], {"updated": now})
            self._write_json(paths["last_scan"], {"ts": now})
            self._write_json(paths["governance"], {"generated_at": now})
            self._write_json(paths["monitor"], {"generated_at": now, "health": "DRIFTING"})
            self._write_json(paths["autopilot"], {"generated_at": now, "mode": "pause_new_entries"})
            self._write_json(
                paths["lanes"],
                {
                    "candidate_trial": {
                        "status": "queued",
                        "paper_only": True,
                        "changes": [
                            {"key": "max_long_positions", "action": "reduce", "from": 3, "to": 2},
                        ],
                    }
                },
            )
            self._write_json(paths["harness"], {"status": "ok"})
            self._write_json(paths["review"], {"recommendation": "queue_candidate_trial"})

            old = (
                qd.PORTFOLIO_FILE,
                qd.LAST_SCAN_FILE,
                qd.GOVERNANCE_FILE,
                qd.MONITOR_FILE,
                qd.AUTOPILOT_FILE,
                qd.LANES_FILE,
                qd.HARNESS_FILE,
                qd.REVIEW_FILE,
            )
            try:
                qd.PORTFOLIO_FILE = paths["portfolio"]
                qd.LAST_SCAN_FILE = paths["last_scan"]
                qd.GOVERNANCE_FILE = paths["governance"]
                qd.MONITOR_FILE = paths["monitor"]
                qd.AUTOPILOT_FILE = paths["autopilot"]
                qd.LANES_FILE = paths["lanes"]
                qd.HARNESS_FILE = paths["harness"]
                qd.REVIEW_FILE = paths["review"]

                with mock.patch.object(
                    qd,
                    "load_merged_quantforge_params",
                    return_value={"max_open_positions": 0, "max_long_positions": 0, "max_short_positions": 0},
                ):
                    report = qd.build_report()
            finally:
                (
                    qd.PORTFOLIO_FILE,
                    qd.LAST_SCAN_FILE,
                    qd.GOVERNANCE_FILE,
                    qd.MONITOR_FILE,
                    qd.AUTOPILOT_FILE,
                    qd.LANES_FILE,
                    qd.HARNESS_FILE,
                    qd.REVIEW_FILE,
                ) = old

        conflict = next(row for row in report["checks"] if row["name"] == "control_param_conflict")
        self.assertTrue(conflict["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
