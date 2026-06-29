#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from unittest import mock

import quantforge_research_director as qrd


class QuantforgeResearchDirectorSliceProfileTests(unittest.TestCase):
    def test_new_arms_are_registered(self):
        self.assertIn("majors_non_fragile_4h_h25", qrd.ARMS)
        self.assertEqual(
            qrd.ARMS["majors_non_fragile_4h_h25"]["slice_profile"],
            "majors_non_fragile",
        )
        self.assertIn("majors_non_fragile_absolute_edge_4h_h25", qrd.ARMS)
        self.assertEqual(
            qrd.ARMS["majors_non_fragile_absolute_edge_4h_h25"]["objective_mode"],
            "absolute_edge",
        )
        self.assertIn("majors_positive_longs_4h_h25", qrd.ARMS)
        self.assertEqual(
            qrd.ARMS["majors_positive_longs_4h_h25"]["slice_profile"],
            "majors_positive_long_slices",
        )

    def test_run_arm_passes_slice_profile_env(self):
        captured = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            verdict_path = os.path.join(tmpdir, "verdict_majors_positive_longs_4h_h25.json")
            model_path = os.path.join(tmpdir, "model_majors_positive_longs_4h_h25.pkl")

            def fake_run(argv, env=None, timeout=None, stdout=None, stderr=None):
                captured["argv"] = argv
                captured["env"] = dict(env or {})
                captured["timeout"] = timeout
                with open(verdict_path, "w") as f:
                    json.dump({
                        "gate_pass": False,
                        "gates": {},
                        "cv": {"mean_auc": 0.0, "min_auc": 0.0},
                        "holdout": {"auc": 0.0, "ev_top_decile": 0.0, "ev_top3_per_ts": 0.0},
                        "n_labeled_rows": 0,
                    }, f)

            with mock.patch.object(qrd, "MODEL_DIR", tmpdir), \
                 mock.patch.object(qrd, "append_ledger"), \
                 mock.patch.object(qrd.subprocess, "run", side_effect=fake_run):
                entry = qrd.run_arm("majors_positive_longs_4h_h25")

        self.assertIsNotNone(entry)
        self.assertEqual(captured["env"]["QF_SLICE_PROFILE"], "majors_positive_long_slices")
        self.assertEqual(captured["env"]["QF_MODEL_PATH"], model_path)
        self.assertEqual(captured["env"]["QF_VERDICT_PATH"], verdict_path)

    def test_run_arm_passes_objective_mode_env(self):
        captured = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            verdict_path = os.path.join(tmpdir, "verdict_majors_non_fragile_absolute_edge_4h_h25.json")

            def fake_run(argv, env=None, timeout=None, stdout=None, stderr=None):
                captured["env"] = dict(env or {})
                with open(verdict_path, "w") as f:
                    json.dump({
                        "gate_pass": False,
                        "gates": {},
                        "cv": {"mean_auc": 0.0, "min_auc": 0.0},
                        "holdout": {"auc": 0.0, "ev_top_decile": 0.0, "ev_top3_per_ts": 0.0},
                        "n_labeled_rows": 0,
                    }, f)

            with mock.patch.object(qrd, "MODEL_DIR", tmpdir), \
                 mock.patch.object(qrd, "append_ledger"), \
                 mock.patch.object(qrd.subprocess, "run", side_effect=fake_run):
                entry = qrd.run_arm("majors_non_fragile_absolute_edge_4h_h25")

        self.assertIsNotNone(entry)
        self.assertEqual(captured["env"]["QF_OBJECTIVE_MODE"], "absolute_edge")


if __name__ == "__main__":
    unittest.main(verbosity=2)
