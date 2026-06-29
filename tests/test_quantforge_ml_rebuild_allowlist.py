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
    def _numpy_stub():
        class _FakeArray(list):
            def __sub__(self, other):
                return _FakeArray([float(a) - float(b) for a, b in zip(self, other)])

            def tolist(self):
                return list(self)

        return types.SimpleNamespace(
            float32=float,
            asarray=lambda values, dtype=None: _FakeArray([float(v) for v in values]),
        )

    _ensure_stub("numpy", _numpy_stub)
    _ensure_stub("pandas", lambda: types.SimpleNamespace())
    _ensure_stub("xgboost", lambda: types.SimpleNamespace(XGBClassifier=object))
    _ensure_stub("lightgbm", lambda: types.SimpleNamespace(LGBMClassifier=object))
    if "sklearn.model_selection" not in sys.modules:
        sklearn_mod = sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
        model_selection_mod = types.ModuleType("sklearn.model_selection")
        model_selection_mod.TimeSeriesSplit = object
        sys.modules["sklearn.model_selection"] = model_selection_mod
        sklearn_mod.model_selection = model_selection_mod
    if "sklearn.metrics" not in sys.modules:
        sklearn_mod = sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
        metrics_mod = types.ModuleType("sklearn.metrics")
        metrics_mod.roc_auc_score = lambda *args, **kwargs: 0.5
        sys.modules["sklearn.metrics"] = metrics_mod
        sklearn_mod.metrics = metrics_mod


_install_import_stubs()

import quantforge_ml_rebuild as qmr


class QuantforgeMlRebuildAllowlistTests(unittest.TestCase):
    def test_normalize_symbol_allowlist_accepts_base_and_full_symbols(self):
        allow = qmr._normalize_symbol_allowlist("eth, SOL-USDT, xrp")
        self.assertEqual(allow, {"ETH-USDT", "SOL-USDT", "XRP-USDT"})

    def test_normalize_symbol_allowlist_ignores_empty_entries(self):
        allow = qmr._normalize_symbol_allowlist(" , ,btc,, ")
        self.assertEqual(allow, {"BTC-USDT"})

    def test_setup_score_columns_are_not_excluded_from_training_features(self):
        self.assertFalse(qmr._is_excluded_training_col("setup_trend_long_score"))
        self.assertFalse(qmr._is_excluded_training_col("setup_exhaustion_short_score"))
        self.assertTrue(qmr._is_excluded_training_col("target_4h"))
        self.assertTrue(qmr._is_excluded_training_col("research_hold_marker"))

    def test_objective_return_arrays_support_absolute_and_relative_modes(self):
        rel = qmr._objective_return_arrays([0.04, -0.01], [0.01, -0.03], objective_mode="btc_relative")
        abs_edge = qmr._objective_return_arrays([0.04, -0.01], [0.01, -0.03], objective_mode="absolute_edge")
        self.assertEqual([round(v, 4) for v in rel.tolist()], [0.03, 0.02])
        self.assertEqual([round(v, 4) for v in abs_edge.tolist()], [0.04, -0.01])


if __name__ == "__main__":
    unittest.main(verbosity=2)
