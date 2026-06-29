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
import quantforge_paper as qfp


class QuantforgeMlPaperTrialGateBypassTests(unittest.TestCase):
    def test_signal_gate_status_only_bypasses_when_explicitly_allowed(self):
        self.assertEqual(qml._signal_gate_status({"gate_pass": False}, allow_gate_bypass=False), (False, False))
        self.assertEqual(qml._signal_gate_status({"gate_pass": False}, allow_gate_bypass=True), (False, True))
        self.assertEqual(qml._signal_gate_status({"gate_pass": True}, allow_gate_bypass=True), (True, False))

    def test_short_model_ready_allows_paper_trial_bypass(self):
        ready, bypassed = qml._short_model_ready_for_signal(
            {"gate_pass": False, "overall_auc": 0.40, "holdout_trades": 0},
            has_model=True,
            allow_gate_bypass=True,
        )
        self.assertTrue(ready)
        self.assertTrue(bypassed)

    def test_only_active_paper_trials_bypass_model_gate(self):
        active_paper = {"status": "active", "paper_only": True, "changes": []}
        queued_paper = {"status": "queued", "paper_only": True, "changes": []}
        active_non_paper = {"status": "active", "paper_only": False, "changes": []}
        completed_paper = {"status": "completed", "paper_only": True, "changes": []}

        self.assertTrue(qfp._allow_candidate_trial_gate_bypass(active_paper))
        self.assertTrue(qfp._allow_candidate_trial_gate_bypass(queued_paper))
        self.assertFalse(qfp._allow_candidate_trial_gate_bypass(active_non_paper))
        self.assertFalse(qfp._allow_candidate_trial_gate_bypass(completed_paper))


if __name__ == "__main__":
    unittest.main(verbosity=2)
