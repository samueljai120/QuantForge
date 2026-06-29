#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import types
import unittest


def _ensure_stub(module_name, stub_factory):
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
    except Exception:
        sys.modules[module_name] = stub_factory()


def _install_import_stubs():
    _ensure_stub("numpy", lambda: types.SimpleNamespace(ndarray=object))
    _ensure_stub("pandas", lambda: types.SimpleNamespace(DataFrame=object, Series=object))
    _ensure_stub("xgboost", lambda: types.SimpleNamespace(XGBClassifier=object))
    _ensure_stub("lightgbm", lambda: types.SimpleNamespace(LGBMClassifier=object))

    if "sklearn.metrics" not in sys.modules:
        try:
            __import__("sklearn.metrics")
        except Exception:
            sklearn_mod = sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
            metrics_mod = types.ModuleType("sklearn.metrics")
            metrics_mod.roc_auc_score = lambda *args, **kwargs: 0.5
            sys.modules["sklearn.metrics"] = metrics_mod
            sklearn_mod.metrics = metrics_mod

    if "quantforge_target_profiles" not in sys.modules:
        try:
            __import__("quantforge_target_profiles")
        except Exception:
            target_profiles_mod = types.ModuleType("quantforge_target_profiles")
            target_profiles_mod.apply_research_hold_target_profile = (
                lambda df, *, horizon=4: (df, {"profile": "research_hold_setup_composite", "horizon_hours": horizon})
            )
            target_profiles_mod.apply_setup_quality_target_profile = (
                lambda df, *, horizon=4: (df, {"profile": "setup_quality_labeled_directional", "horizon_hours": horizon})
            )
            sys.modules["quantforge_target_profiles"] = target_profiles_mod


_install_import_stubs()

import quantforge_ml as qml


class QuantforgeMlSurfaceModeTests(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_load_redesign_context_treats_competitiveness_gap_rebuild_as_research_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = os.path.join(tmpdir, "candidate-recovery.json")
            lanes = os.path.join(tmpdir, "experiment-lanes.json")
            self._write_json(
                recovery,
                {
                    "candidate_id": "competitiveness_gap_rebuild:20260626T134034Z",
                    "type": "competitiveness_gap_rebuild",
                },
            )
            self._write_json(
                lanes,
                {
                    "candidate_trial": {
                        "candidate_id": "competitiveness_gap_rebuild:20260626T134034Z",
                        "type": "competitiveness_gap_rebuild",
                        "status": "active",
                    }
                },
            )

            old = (qml.CANDIDATE_RECOVERY_FILE, qml.EXPERIMENT_LANES_FILE)
            try:
                qml.CANDIDATE_RECOVERY_FILE = recovery
                qml.EXPERIMENT_LANES_FILE = lanes
                context = qml.load_redesign_context()
            finally:
                qml.CANDIDATE_RECOVERY_FILE, qml.EXPERIMENT_LANES_FILE = old

        self.assertFalse(context["redesign_active"])
        self.assertTrue(context["research_hold_active"])
        self.assertEqual(context["surface_mode"], "research_hold_setup_composite")

    def test_completed_rebuild_trial_does_not_enable_research_surface_without_live_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = os.path.join(tmpdir, "candidate-recovery.json")
            lanes = os.path.join(tmpdir, "experiment-lanes.json")
            self._write_json(recovery, {})
            self._write_json(
                lanes,
                {
                    "candidate_trial": {
                        "candidate_id": "competitiveness_gap_rebuild:20260626T134034Z",
                        "type": "competitiveness_gap_rebuild",
                        "status": "completed",
                    }
                },
            )

            old = (qml.CANDIDATE_RECOVERY_FILE, qml.EXPERIMENT_LANES_FILE)
            try:
                qml.CANDIDATE_RECOVERY_FILE = recovery
                qml.EXPERIMENT_LANES_FILE = lanes
                context = qml.load_redesign_context()
            finally:
                qml.CANDIDATE_RECOVERY_FILE, qml.EXPERIMENT_LANES_FILE = old

        self.assertFalse(context["research_hold_active"])
        self.assertEqual(context["surface_mode"], "standard_directional")

    def test_apply_training_target_profile_dispatches_rebuild_lane_to_research_surface(self):
        calls = []

        def fake_apply(df, *, horizon=4):
            calls.append((df, horizon))
            return {"profiled": True}, {"profile": "research_hold_setup_composite", "horizon_hours": horizon}

        old = qml.apply_research_hold_target_profile
        try:
            qml.apply_research_hold_target_profile = fake_apply
            profiled, profile = qml.apply_training_target_profile(
                {"rows": 2},
                {
                    "candidate_type": "competitiveness_gap_rebuild",
                    "trial_type": "competitiveness_gap_rebuild",
                    "trial_status": "active",
                    "redesign_active": False,
                    "research_hold_active": True,
                    "surface_mode": "research_hold_setup_composite",
                },
            )
        finally:
            qml.apply_research_hold_target_profile = old

        self.assertEqual(calls, [({"rows": 2}, 4)])
        self.assertEqual(profiled, {"profiled": True})
        self.assertEqual(profile["profile"], "research_hold_setup_composite")

    def test_load_redesign_context_treats_setup_quality_trial_as_setup_quality_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = os.path.join(tmpdir, "candidate-recovery.json")
            lanes = os.path.join(tmpdir, "experiment-lanes.json")
            self._write_json(
                recovery,
                {
                    "candidate_id": "setup_quality_recovery:20260628T051658Z",
                    "type": "setup_quality_recovery",
                },
            )
            self._write_json(
                lanes,
                {
                    "candidate_trial": {
                        "candidate_id": "setup_quality_recovery:20260628T051658Z",
                        "type": "setup_quality_recovery",
                        "status": "active",
                    }
                },
            )

            old = (qml.CANDIDATE_RECOVERY_FILE, qml.EXPERIMENT_LANES_FILE)
            try:
                qml.CANDIDATE_RECOVERY_FILE = recovery
                qml.EXPERIMENT_LANES_FILE = lanes
                context = qml.load_redesign_context()
            finally:
                qml.CANDIDATE_RECOVERY_FILE, qml.EXPERIMENT_LANES_FILE = old

        self.assertTrue(context["setup_quality_active"])
        self.assertFalse(context["research_hold_active"])
        self.assertEqual(context["surface_mode"], "setup_quality_labeled_directional")

    def test_apply_training_target_profile_dispatches_setup_quality_surface(self):
        calls = []

        def fake_apply(df, *, horizon=4):
            calls.append((df, horizon))
            return {"profiled": True}, {"profile": "setup_quality_labeled_directional", "horizon_hours": horizon}

        old = qml.apply_setup_quality_target_profile
        try:
            qml.apply_setup_quality_target_profile = fake_apply
            profiled, profile = qml.apply_training_target_profile(
                {"rows": 3},
                {
                    "candidate_type": "setup_quality_recovery",
                    "trial_type": "setup_quality_recovery",
                    "trial_status": "active",
                    "redesign_active": False,
                    "research_hold_active": False,
                    "setup_quality_active": True,
                    "surface_mode": "setup_quality_labeled_directional",
                },
            )
        finally:
            qml.apply_setup_quality_target_profile = old

        self.assertEqual(calls, [({"rows": 3}, 4)])
        self.assertEqual(profiled, {"profiled": True})
        self.assertEqual(profile["profile"], "setup_quality_labeled_directional")


if __name__ == "__main__":
    unittest.main(verbosity=2)
