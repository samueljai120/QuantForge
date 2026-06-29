#!/usr/bin/env python3
import json
import os
import pickle
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
    _ensure_stub("numpy", lambda: types.SimpleNamespace(array=lambda rows: rows, ndarray=object))
    _ensure_stub("pandas", lambda: types.SimpleNamespace(DataFrame=object, Series=object, notna=lambda v: v is not None))
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
            sys.modules["quantforge_target_profiles"] = target_profiles_mod


_install_import_stubs()

import quantforge_ml as qml


class FakeProbModel:
    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, rows):
        return [[1.0 - self.prob, self.prob] for _ in rows]


class FakeRow(dict):
    @property
    def index(self):
        return self.keys()


class QuantforgeMlSpecialistTests(unittest.TestCase):
    def test_dominant_setup_payload_prefers_long_on_tie_or_stronger_long(self):
        tag, score, direction = qml._dominant_setup_payload(
            long_setup_tag="trend_long",
            long_setup_score=0.44,
            short_setup_tag="trend_short",
            short_setup_score=0.63,
            long_confidence=0.12,
            short_confidence=0.12,
        )

        self.assertEqual(tag, "trend_long")
        self.assertAlmostEqual(score, 0.44, places=6)
        self.assertEqual(direction, "LONG")

    def test_dominant_setup_payload_prefers_short_when_short_confidence_is_stronger(self):
        tag, score, direction = qml._dominant_setup_payload(
            long_setup_tag="breakout_long",
            long_setup_score=0.51,
            short_setup_tag="exhaustion_short",
            short_setup_score=0.66,
            long_confidence=0.09,
            short_confidence=0.21,
        )

        self.assertEqual(tag, "exhaustion_short")
        self.assertAlmostEqual(score, 0.66, places=6)
        self.assertEqual(direction, "SHORT")

    def test_specialist_model_ready_allows_research_hold_without_holdout_when_support_is_strong(self):
        ready, gate_bypassed = qml._specialist_model_ready(
            {
                "overall_auc": 0.91,
                "positive_rows": 5000,
                "training_profile": {"surface_mode": "research_hold_setup_composite"},
            },
            {"setup": "trend_long", "status": "trained", "positive_rows": 5000},
            {"research_hold_active": True, "surface_mode": "research_hold_setup_composite"},
        )

        self.assertTrue(ready)
        self.assertFalse(gate_bypassed)

    def test_specialist_model_ready_requires_holdout_when_metadata_explicitly_reports_it(self):
        ready, gate_bypassed = qml._specialist_model_ready(
            {
                "overall_auc": 0.91,
                "positive_rows": 5000,
                "holdout_trades": 12,
                "training_profile": {"surface_mode": "research_hold_setup_composite"},
            },
            {"setup": "breakout_long", "status": "trained", "positive_rows": 5000, "holdout_trades": 12},
            {"research_hold_active": True, "surface_mode": "research_hold_setup_composite"},
        )

        self.assertFalse(ready)
        self.assertFalse(gate_bypassed)

    def test_apply_long_specialist_confirmation_confirms_and_blends_up(self):
        confidence, notes = qml._apply_long_specialist_confirmation(
            0.72,
            {"setup": "trend_long", "confidence": 0.90, "threshold": 0.80},
        )

        self.assertAlmostEqual(confidence, 0.81, places=6)
        self.assertIn("confirmed long", notes[0])

    def test_apply_long_specialist_confirmation_vetoes_and_caps_confidence(self):
        confidence, notes = qml._apply_long_specialist_confirmation(
            0.82,
            {"setup": "breakout_long", "confidence": 0.61, "threshold": 0.70},
        )

        self.assertAlmostEqual(confidence, 0.61, places=6)
        self.assertIn("vetoed long", notes[0])

    def test_apply_long_specialist_confirmation_can_reopen_long_when_base_gate_is_closed(self):
        confidence, notes = qml._apply_long_specialist_confirmation(
            0.0,
            {"setup": "trend_long", "confidence": 0.86, "threshold": 0.80},
            base_gate_ready=False,
        )

        self.assertAlmostEqual(confidence, 0.86, places=6)
        self.assertIn("reopened long", notes[0])

    def test_score_long_setup_specialist_reads_registry_and_scores_supported_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "setup_specialists.json")
            model_path = os.path.join(tmpdir, "ensemble_trend_long.pkl")
            meta_path = os.path.join(tmpdir, "model_meta_trend_long.json")
            with open(registry_path, "w") as f:
                json.dump(
                    {
                        "status": "ready",
                        "specialists": [
                            {"setup": "trend_long", "status": "trained", "positive_rows": 6000}
                        ],
                    },
                    f,
                )
            with open(model_path, "wb") as f:
                pickle.dump((FakeProbModel(0.92), FakeProbModel(0.88), ["feature_a", "feature_b"]), f)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "overall_auc": 0.95,
                        "positive_rows": 6000,
                        "optimal_threshold": 0.80,
                        "training_profile": {"surface_mode": "research_hold_setup_composite"},
                    },
                    f,
                )

            old_model_dir = qml.MODEL_DIR
            old_notna = getattr(qml.pd, "notna", None)
            try:
                qml.MODEL_DIR = tmpdir
                qml.pd.notna = lambda value: value is not None
                specialist = qml._score_long_setup_specialist(
                    "trend_long",
                    FakeRow({"feature_a": 1.0, "feature_b": 2.0}),
                    {"research_hold_active": True, "surface_mode": "research_hold_setup_composite"},
                )
            finally:
                qml.MODEL_DIR = old_model_dir
                if old_notna is not None:
                    qml.pd.notna = old_notna

        self.assertIsNotNone(specialist)
        self.assertEqual(specialist["setup"], "trend_long")
        self.assertAlmostEqual(specialist["confidence"], 0.90, places=6)
        self.assertAlmostEqual(specialist["threshold"], 0.80, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
