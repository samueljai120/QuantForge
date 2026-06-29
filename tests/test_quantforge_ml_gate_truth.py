#!/usr/bin/env python3
import sys
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
            sys.modules["quantforge_target_profiles"] = target_profiles_mod


_install_import_stubs()

import quantforge_ml as qml


class QuantforgeMlGateTruthTests(unittest.TestCase):
    def test_evaluate_gate_fails_closed_when_holdout_metrics_are_missing(self):
        ready, payload = qml.evaluate_gate(
            {
                "win_rate": 0.61,
                "sharpe": 1.4,
                "trades": 120,
            },
            label="Long model",
        )

        self.assertFalse(ready)
        self.assertFalse(payload["ready"])
        self.assertIn("Hold-out metrics unavailable", payload["reasons"])
        self.assertIsNone(payload["measured"]["holdout_win_rate"])
        self.assertEqual(payload["measured"]["cv_win_rate"], 0.61)

    def test_evaluate_gate_uses_real_holdout_metrics_when_present(self):
        ready, payload = qml.evaluate_gate(
            {
                "win_rate": 0.61,
                "sharpe": 1.4,
                "trades": 120,
                "holdout_win_rate": 0.58,
                "holdout_sharpe": 1.2,
                "holdout_trades": 48,
            },
            label="Long model",
        )

        self.assertTrue(ready)
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["measured"]["holdout_trades"], 48)
        self.assertEqual(payload["measured"]["holdout_win_rate"], 0.58)


if __name__ == "__main__":
    unittest.main(verbosity=2)
