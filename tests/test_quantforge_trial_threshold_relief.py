#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import quantforge_paper as qfp


class QuantforgeTrialThresholdReliefTests(unittest.TestCase):
    def test_trial_threshold_relief_reads_active_trial_overrides(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "trial_long_threshold_relief", "value": 0.10},
                {"key": "trial_short_threshold_relief", "value": 0.15},
            ],
        }
        old = qfp.TRIAL_LONG_THRESHOLD_RELIEF
        try:
            qfp.TRIAL_LONG_THRESHOLD_RELIEF = 0.02
            long_relief, short_relief = qfp._trial_threshold_relief(trial)
        finally:
            qfp.TRIAL_LONG_THRESHOLD_RELIEF = old

        self.assertEqual((long_relief, short_relief), (0.10, 0.15))

    def test_trial_threshold_relief_uses_defaults_when_trial_inactive(self):
        trial = {
            "status": "completed",
            "changes": [
                {"key": "trial_long_threshold_relief", "value": 0.10},
                {"key": "trial_short_threshold_relief", "value": 0.15},
            ],
        }
        old = qfp.TRIAL_LONG_THRESHOLD_RELIEF
        try:
            qfp.TRIAL_LONG_THRESHOLD_RELIEF = 0.02
            long_relief, short_relief = qfp._trial_threshold_relief(trial)
        finally:
            qfp.TRIAL_LONG_THRESHOLD_RELIEF = old

        self.assertEqual((long_relief, short_relief), (0.02, 0.0))

    def test_trial_allows_adverse_short_entries_only_when_active(self):
        active = {
            "status": "active",
            "changes": [
                {"key": "allow_short_entries_in_adverse_regime", "value": True},
            ],
        }
        queued = {
            "status": "queued",
            "changes": [
                {"key": "allow_short_entries_in_adverse_regime", "value": True},
            ],
        }
        completed = {
            "status": "completed",
            "changes": [
                {"key": "allow_short_entries_in_adverse_regime", "value": True},
            ],
        }

        self.assertTrue(qfp._trial_allows_adverse_short_entries(active))
        self.assertTrue(qfp._trial_allows_adverse_short_entries(queued))
        self.assertFalse(qfp._trial_allows_adverse_short_entries(completed))

    def test_save_idle_execution_report_writes_fresh_idle_artifact(self):
        trial = {
            "status": "active",
            "changes": [],
        }
        autopilot = {"mode": "run_candidate_paper_trial"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "last_execution.json")
            old = qfp.LAST_EXECUTION_FILE
            try:
                qfp.LAST_EXECUTION_FILE = path
                qfp._save_idle_execution_report(
                    autopilot=autopilot,
                    trial=trial,
                    signal_count=0,
                    execution_permission="allowed",
                    idle_reason="no_actionable_signals",
                    details=["fresh cycle"],
                )
                with open(path) as f:
                    saved = json.load(f)
            finally:
                qfp.LAST_EXECUTION_FILE = old

        self.assertEqual(saved["idle_reason"], "no_actionable_signals")
        self.assertEqual(saved["execution_permission"], "allowed")
        self.assertEqual(saved["signal_count"], 0)
        self.assertTrue(saved["trial_active"])
        self.assertEqual(saved["details"], ["fresh cycle"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
