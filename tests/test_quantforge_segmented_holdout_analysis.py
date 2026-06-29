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
    _ensure_stub("pandas", lambda: types.SimpleNamespace(DataFrame=object, Series=object, read_parquet=lambda *args, **kwargs: None))
    if "quantforge_ml" not in sys.modules:
        qml = types.ModuleType("quantforge_ml")
        qml.ROUND_TRIP_COST = 0.002
        qml.apply_training_target_profile = lambda df, redesign_context=None: (df, {"profile": "research_hold_setup_composite"})
        qml.load_model = lambda short=False: (None, None, None, None)
        qml.load_redesign_context = lambda: {"candidate_type": "quantforge_research_hold"}
        sys.modules["quantforge_ml"] = qml


_install_import_stubs()

import quantforge_segmented_holdout_report as qshr


class QuantforgeSegmentedHoldoutAnalysisTests(unittest.TestCase):
    def test_analysis_gate_bypass_only_for_gate_failed_models(self):
        self.assertTrue(qshr._analysis_gate_bypass({"gate_pass": False}))
        self.assertFalse(qshr._analysis_gate_bypass({"gate_pass": True}))
        self.assertTrue(qshr._analysis_gate_bypass({}))

    def test_symbol_tier_keeps_major_mapping(self):
        self.assertEqual(qshr._symbol_tier("BTC-USDT"), "major")
        self.assertEqual(qshr._symbol_tier("ADA-USDT"), "alt")

    def test_execution_limit_for_side_matches_live_caps(self):
        self.assertEqual(qshr._execution_limit_for_side("long"), 3)
        self.assertEqual(qshr._execution_limit_for_side("short"), 2)
        self.assertEqual(qshr._execution_limit_for_side("other"), 1)


class QuantforgeSignalRankingTests(unittest.TestCase):
    def test_signal_rank_value_uses_margin_over_threshold(self):
        from quantforge_signal_ranking import signal_rank_value

        lower_raw_higher_margin = {"execution_score": 0.91, "entry_threshold": 0.80}
        higher_raw_lower_margin = {"execution_score": 0.94, "entry_threshold": 0.90}

        self.assertGreater(
            signal_rank_value(lower_raw_higher_margin),
            signal_rank_value(higher_raw_lower_margin),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
