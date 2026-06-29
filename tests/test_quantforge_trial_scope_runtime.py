#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import quantforge_paper as qfp


class QuantforgeTrialScopeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._orig_caps = (
            qfp.MAX_POSITIONS,
            qfp.MAX_LONG_POSITIONS,
            qfp.MAX_SHORT_POSITIONS,
        )

    def tearDown(self):
        (
            qfp.MAX_POSITIONS,
            qfp.MAX_LONG_POSITIONS,
            qfp.MAX_SHORT_POSITIONS,
        ) = self._orig_caps

    def test_high_conviction_scope_maps_to_runtime_controls(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "strategy_scope", "value": "slower_high_conviction_majors_only"},
            ],
        }

        self.assertEqual(qfp._trial_runtime_value(trial, "entry_profile", ""), "majors_only")
        self.assertEqual(qfp._trial_runtime_value(trial, "symbol_universe", ""), "major_liquidity_tier")
        self.assertEqual(qfp._trial_runtime_value(trial, "generic_long_policy", ""), "require_top_quality")
        self.assertEqual(qfp._trial_runtime_value(trial, "entry_selection", ""), "require_regime_support_and_labeled_setup_alignment")
        self.assertTrue(qfp._trial_runtime_value(trial, "non_major_entries", False))
        self.assertEqual(qfp._trial_runtime_int(trial, "max_long_positions", 3), 1)
        self.assertEqual(qfp._trial_runtime_int(trial, "max_short_positions", 2), 1)

    def test_selection_adjustments_blocks_non_major_long_for_high_conviction_scope(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "strategy_scope", "value": "slower_high_conviction_majors_only"},
            ],
        }
        sig = {
            "side": "BUY",
            "setup_tag": "trend_long",
            "setup_score": 0.82,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}

        allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("ADA-USDT", sig, quality, trial, regime)

        self.assertFalse(allowed)
        self.assertTrue(any("restricts longs" in reason and "major" in reason for reason in reasons))
        self.assertEqual(score_adj, 0.0)
        self.assertEqual(size_mult, 1.0)

    def test_trend_long_uses_regime_label_for_bull_tailwind(self):
        sig = {
            "side": "BUY",
            "setup_tag": "trend_long",
            "setup_score": 0.82,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}

        allowed, reasons, score_adj, _ = qfp._selection_adjustments("BTC-USDT", sig, quality, None, regime)

        self.assertTrue(allowed)
        self.assertIn("trend-long regime tailwind", reasons)
        self.assertGreater(score_adj, 0.03)

    def test_signal_rank_value_prefers_edge_over_threshold(self):
        stronger_raw = {"score": 0.94, "execution_score": 0.94, "entry_threshold": 0.90}
        stronger_edge = {"score": 0.91, "execution_score": 0.91, "entry_threshold": 0.80}

        self.assertLess(qfp._signal_rank_value(stronger_raw), qfp._signal_rank_value(stronger_edge))

    def test_positive_holdout_scope_blocks_fragile_long_context(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "strategy_scope", "value": "major_symbols_and_positive_holdout_slices"},
            ],
        }
        sig = {
            "side": "BUY",
            "setup_tag": "trend_long",
            "setup_score": 0.82,
            "fakeout_risk": 0.72,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}

        allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("BTC-USDT", sig, quality, trial, regime)

        self.assertFalse(allowed)
        self.assertIn("candidate trial rejects fragile long context", reasons[0])
        self.assertEqual(score_adj, 0.0)
        self.assertEqual(size_mult, 1.0)

    def test_positive_holdout_scope_allows_only_trend_short(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "strategy_scope", "value": "major_symbols_and_positive_holdout_slices"},
            ],
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BEAR", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}

        allowed_exhaustion, reasons_exhaustion, _, _ = qfp._selection_adjustments(
            "BTC-USDT",
            {"side": "SELL", "setup_tag": "exhaustion_short", "setup_score": 0.82, "redesign_active": False},
            quality,
            trial,
            regime,
        )
        allowed_trend, _, _, _ = qfp._selection_adjustments(
            "BTC-USDT",
            {"side": "SELL", "setup_tag": "trend_short", "setup_score": 0.82, "redesign_active": False},
            quality,
            trial,
            regime,
        )

        self.assertFalse(allowed_exhaustion)
        self.assertIn("allowed: trend_short", reasons_exhaustion[0])
        self.assertTrue(allowed_trend)

    def test_positive_holdout_scope_enables_adverse_short_bypass(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "strategy_scope", "value": "major_symbols_and_positive_holdout_slices"},
            ],
        }

        self.assertTrue(qfp._trial_allows_adverse_short_entries(trial))

    def test_positive_holdout_scope_matches_executed_subset_short_cap(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "strategy_scope", "value": "major_symbols_and_positive_holdout_slices"},
            ],
        }

        self.assertEqual(qfp._trial_runtime_int(trial, "max_short_positions", 1), 2)

    def test_positive_holdout_scope_raises_total_cap_for_trial_shorts(self):
        trial = {
            "status": "active",
            "paper_only": True,
            "changes": [
                {"key": "strategy_scope", "value": "major_symbols_and_positive_holdout_slices"},
            ],
        }

        self.assertEqual(qfp._trial_effective_position_cap(trial, "SHORT", 1, 2), 2)
        self.assertEqual(qfp._trial_effective_position_cap(trial, "LONG", 1, 2), 1)

    def test_position_cap_normalization_honors_zero_hard_halt(self):
        self.assertEqual(qfp._normalize_position_caps(0, 3, 2), (0, 0, 0))
        self.assertEqual(qfp._normalize_position_caps(2, 3, 5), (2, 2, 2))

    def test_regime_controls_never_reopen_slots_when_total_cap_is_zero(self):
        qfp.MAX_POSITIONS = 0

        controls = qfp._regime_controls(0.0)

        self.assertEqual(controls["max_positions"], 0)

    def test_non_paper_trial_short_cap_cannot_bypass_global_zero_cap(self):
        qfp.MAX_POSITIONS = 0
        trial = {
            "status": "active",
            "changes": [
                {"key": "strategy_scope", "value": "major_symbols_and_positive_holdout_slices"},
            ],
        }

        self.assertEqual(qfp._trial_effective_position_cap(trial, "SHORT", 0, 2), 0)

    def test_paper_only_trial_can_reopen_bounded_total_cap_when_global_cap_is_zero(self):
        qfp.MAX_POSITIONS = 0
        qfp.MAX_LONG_POSITIONS = 0
        qfp.MAX_SHORT_POSITIONS = 0
        trial = {
            "status": "active",
            "paper_only": True,
            "changes": [
                {"key": "max_long_positions", "value": 2},
            ],
        }

        self.assertEqual(qfp._trial_effective_position_cap(trial, "LONG", 0, 0), 2)

    def test_research_hold_blocks_fragile_short_context(self):
        sig = {
            "side": "SELL",
            "setup_tag": "trend_short",
            "setup_score": 0.82,
            "fakeout_risk": 0.72,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BEAR", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}
        original = qfp.load_autopilot_report
        qfp.load_autopilot_report = lambda: {"actions": ["freeze_for_rebuild"]}
        try:
            allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("BTC-USDT", sig, quality, None, regime)
        finally:
            qfp.load_autopilot_report = original

        self.assertFalse(allowed)
        self.assertIn("research/hold blocks fragile context", reasons[0])
        self.assertEqual(score_adj, 0.0)
        self.assertEqual(size_mult, 1.0)

    def test_research_hold_allows_high_conviction_top_alt_long(self):
        sig = {
            "side": "BUY",
            "setup_tag": "trend_long",
            "setup_score": 0.74,
            "fakeout_risk": 0.20,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.98}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}
        original = qfp.load_autopilot_report
        qfp.load_autopilot_report = lambda: {"actions": ["freeze_for_rebuild"]}
        try:
            allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("AVAX-USDT", sig, quality, None, regime)
        finally:
            qfp.load_autopilot_report = original

        self.assertTrue(allowed)
        self.assertIn("research hold top-alt long size 0.90x", reasons)
        self.assertGreater(score_adj, 0.0)
        self.assertEqual(size_mult, 0.9)

    def test_research_hold_rejects_weak_top_alt_long_even_in_allowed_family(self):
        sig = {
            "side": "BUY",
            "setup_tag": "breakout_long",
            "setup_score": 0.63,
            "fakeout_risk": 0.20,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.98}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}
        original = qfp.load_autopilot_report
        qfp.load_autopilot_report = lambda: {"actions": ["freeze_for_rebuild"]}
        try:
            allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("AVAX-USDT", sig, quality, None, regime)
        finally:
            qfp.load_autopilot_report = original

        self.assertFalse(allowed)
        self.assertIn("research hold non-major long requires setup", reasons[0])
        self.assertEqual(score_adj, 0.0)
        self.assertEqual(size_mult, 1.0)

    def test_candidate_expansion_allowed_long_setups_blocks_generic_long(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "entry_profile", "value": "majors_plus_liquid_alts"},
                {"key": "symbol_universe", "value": "major_and_top_alt_tier"},
                {"key": "allowed_long_setups", "value": ["trend_long", "breakout_long"]},
            ],
        }
        sig = {
            "side": "BUY",
            "setup_tag": "generic_long",
            "setup_score": 0.82,
            "fakeout_risk": 0.20,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}

        allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("DOGE-USDT", sig, quality, trial, regime)

        self.assertFalse(allowed)
        self.assertIn("candidate trial rejects long setup generic_long", reasons[0])
        self.assertEqual(score_adj, 0.0)
        self.assertEqual(size_mult, 1.0)

    def test_major_liquidity_expansion_top_alt_long_bypasses_late_majors_only_gate(self):
        trial = {
            "status": "active",
            "paper_only": True,
            "changes": [
                {"key": "entry_profile", "value": "majors_plus_liquid_alts"},
                {"key": "symbol_universe", "value": "major_and_top_alt_tier"},
            ],
        }

        self.assertTrue(qfp._trial_bypasses_major_only_for_symbol(trial, "DOGE-USDT", "LONG"))
        self.assertFalse(qfp._trial_bypasses_major_only_for_symbol(trial, "DOGE-USDT", "SHORT"))
        self.assertFalse(qfp._trial_bypasses_major_only_for_symbol(trial, "EIGEN-USDT", "LONG"))

    def test_setup_quality_recovery_allows_approved_non_major_labeled_long(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "entry_profile", "value": "majors_only"},
                {"key": "allowed_long_symbols", "value": ["ICNT-USDT"]},
                {"key": "allowed_long_setups", "value": ["trend_long", "breakout_long"]},
                {"key": "entry_selection", "value": "require_regime_support_and_labeled_setup_alignment"},
                {"key": "generic_long_policy", "value": "require_labeled_setup_and_top_quality"},
            ],
        }
        sig = {
            "side": "BUY",
            "setup_tag": "trend_long",
            "setup_score": 0.82,
            "fakeout_risk": 0.20,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}

        allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("ICNT-USDT", sig, quality, trial, regime)

        self.assertTrue(allowed)
        self.assertIn("trend-long regime tailwind", reasons)
        self.assertGreater(score_adj, 0.0)
        self.assertEqual(size_mult, 1.0)

    def test_setup_quality_recovery_blocks_non_allowlisted_or_unlabeled_non_major_long(self):
        trial = {
            "status": "active",
            "changes": [
                {"key": "entry_profile", "value": "majors_only"},
                {"key": "allowed_long_symbols", "value": ["ICNT-USDT"]},
                {"key": "allowed_long_setups", "value": ["trend_long", "breakout_long"]},
                {"key": "entry_selection", "value": "require_regime_support_and_labeled_setup_alignment"},
                {"key": "generic_long_policy", "value": "require_labeled_setup_and_top_quality"},
            ],
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BULL", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}

        blocked_allowlist, reasons_allowlist, _, _ = qfp._selection_adjustments(
            "RAVE-USDT",
            {"side": "BUY", "setup_tag": "trend_long", "setup_score": 0.82, "fakeout_risk": 0.20, "redesign_active": False},
            quality,
            trial,
            regime,
        )
        blocked_generic, reasons_generic, _, _ = qfp._selection_adjustments(
            "ICNT-USDT",
            {"side": "BUY", "setup_tag": "generic_long", "setup_score": 0.82, "fakeout_risk": 0.20, "redesign_active": False},
            quality,
            trial,
            regime,
        )

        self.assertFalse(blocked_allowlist)
        self.assertIn("approved recovery symbols", reasons_allowlist[0])
        self.assertFalse(blocked_generic)
        self.assertIn("blocks unlabeled long setup generic_long", reasons_generic[0])

    def test_active_expansion_trial_retires_after_repeated_missing_long_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lanes_path = os.path.join(tmpdir, "experiment-lanes.json")
            last_scan_path = os.path.join(tmpdir, "last_scan.json")
            with open(lanes_path, "w") as f:
                json.dump(
                    {
                        "candidate_trial": {
                            "candidate_id": "major_liquidity_expansion:20260628T001543Z",
                            "type": "major_liquidity_expansion",
                            "status": "active",
                            "paper_only": True,
                            "cycles_run": 0,
                            "max_cycles": 6,
                        }
                    },
                    f,
                )
            with open(last_scan_path, "w") as f:
                json.dump(
                    {
                        "ts": "2026-06-28T04:45:03.460220+00:00",
                        "flow": {
                            "buy_signals": 0,
                            "sell_signals": 3,
                            "threshold_miss": 28,
                            "selection_blocked": 1,
                        },
                        "results": [
                            {
                                "symbol": "ICNT-USDT",
                                "status": "hold",
                                "reason": "LONG conf 0.004 < 0.740, SHORT conf 0.001 < 0.840",
                                "long_confidence": 0.0042,
                                "short_confidence": 0.0011,
                                "setup_tag": "trend_long",
                            }
                        ],
                    },
                    f,
                )

            old = (qfp.EXPERIMENT_LANES_FILE, qfp.LAST_SCAN_FILE)
            try:
                qfp.EXPERIMENT_LANES_FILE = lanes_path
                qfp.LAST_SCAN_FILE = last_scan_path

                first = qfp.finalize_candidate_trial_cycle({"mode": "run_candidate_paper_trial"})
                second = qfp.finalize_candidate_trial_cycle({"mode": "run_candidate_paper_trial"})
            finally:
                qfp.EXPERIMENT_LANES_FILE, qfp.LAST_SCAN_FILE = old

        self.assertEqual(first["status"], "active")
        self.assertEqual(first["no_target_long_surface_cycles"], 1)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["assessment"], "fail")
        self.assertEqual(second["completion_reason"], "no_target_long_surface")
        self.assertEqual(second["next_candidate_hint"], "setup_quality_recovery")
        self.assertEqual(second["completion_summary"]["strongest_long_hold"]["symbol"], "ICNT-USDT")

    def test_research_hold_blocks_exhaustion_short_slice(self):
        sig = {
            "side": "SELL",
            "setup_tag": "exhaustion_short",
            "setup_score": 0.82,
            "fakeout_risk": 0.20,
            "redesign_active": False,
        }
        quality = {"quality_score": 0.99}
        regime = {"label": "BEAR", "entropy_label": "ORDERLY", "entropy_penalty": 0.0, "size_mult": 1.0}
        original = qfp.load_autopilot_report
        qfp.load_autopilot_report = lambda: {"actions": ["freeze_for_rebuild"]}
        try:
            allowed, reasons, score_adj, size_mult = qfp._selection_adjustments("BTC-USDT", sig, quality, None, regime)
        finally:
            qfp.load_autopilot_report = original

        self.assertFalse(allowed)
        self.assertIn("research/hold blocks exhaustion_short", reasons[0])
        self.assertEqual(score_adj, 0.0)
        self.assertEqual(size_mult, 1.0)

    def test_execute_paper_trades_honors_zero_caps_even_with_override(self):
        qfp.MAX_POSITIONS = 0
        qfp.MAX_LONG_POSITIONS = 1
        qfp.MAX_SHORT_POSITIONS = 1
        port = {"cash": 1000.0, "positions": {}}
        autopilot = {"mode": "pause_new_entries"}
        signal = {
            "symbol": "BTC-USDT",
            "side": "BUY",
            "score": 0.99,
            "execution_score": 0.99,
            "price": 100.0,
            "reasons": ["test"],
            "quality_pass": True,
        }
        saved = {}
        original_load_params = qfp._load_strategy_params
        original_daily = qfp._daily_loss_breaker_active
        original_weekly = qfp._weekly_loss_breaker_active
        original_drawdown = qfp._drawdown_halt_active
        original_gap = qfp._market_gap_halt
        original_event = qfp._event_risk_block
        original_stress = qfp._btc_stress_mode
        original_tickers = qfp.get_futures_tickers
        original_save_last_execution = qfp.save_last_execution
        original_append_trade = qfp.append_trade
        qfp._load_strategy_params = lambda: {"autopilot_override": "allow_entries"}
        qfp._daily_loss_breaker_active = lambda port: False
        qfp._weekly_loss_breaker_active = lambda port: False
        qfp._drawdown_halt_active = lambda port: False
        qfp._market_gap_halt = lambda: None
        qfp._event_risk_block = lambda: None
        qfp._btc_stress_mode = lambda: False
        qfp.get_futures_tickers = lambda: [{"baseCurrency": "BTC", "lastTradePrice": "100.0"}]
        qfp.save_last_execution = lambda payload: saved.update(payload)
        qfp.append_trade = lambda trade: self.fail("zero cap should not append OPEN trades")
        try:
            executed = qfp.execute_paper_trades([signal], port, autopilot=autopilot, regime={"score": 0.0})
        finally:
            qfp._load_strategy_params = original_load_params
            qfp._daily_loss_breaker_active = original_daily
            qfp._weekly_loss_breaker_active = original_weekly
            qfp._drawdown_halt_active = original_drawdown
            qfp._market_gap_halt = original_gap
            qfp._event_risk_block = original_event
            qfp._btc_stress_mode = original_stress
            qfp.get_futures_tickers = original_tickers
            qfp.save_last_execution = original_save_last_execution
            qfp.append_trade = original_append_trade

        self.assertEqual(executed, [])
        self.assertEqual(saved["execution_permission"], "allowed")
        self.assertEqual(saved["executed_count"], 0)
        self.assertEqual(saved["skip_summary"].get("regime adverse — no new entries"), 1)

    def test_save_strategy_params_does_not_persist_autopilot_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = os.path.join(tmpdir, "strategy-params.json")
            original_path = qfp.LEGACY_PARAMS_FILE
            qfp.LEGACY_PARAMS_FILE = params_path
            try:
                qfp._save_strategy_params(
                    {
                        "autopilot_override": "allow_entries",
                        "max_open_positions": 0,
                    }
                )
                with open(params_path) as f:
                    saved = json.load(f)
            finally:
                qfp.LEGACY_PARAMS_FILE = original_path

        self.assertNotIn("autopilot_override", saved)
        self.assertEqual(saved["max_open_positions"], 0)
        self.assertEqual(saved["_last_modified_by"], "quantforge_autotune")

    def test_update_drawdown_refreshes_portfolio_equity_snapshot(self):
        port = {"peak_equity": 100.0, "max_drawdown": 0.0}

        qfp.update_drawdown(port, 91.23456)

        self.assertEqual(port["equity"], 91.2346)
        self.assertAlmostEqual(port["max_drawdown"], 0.087654, places=6)

    def test_load_portfolio_repairs_null_equity_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            portfolio_path = os.path.join(tmpdir, "portfolio.json")
            with open(portfolio_path, "w") as f:
                json.dump({"cash": 123.45, "positions": {}, "equity": None}, f)

            original_path = qfp.PORTFOLIO_FILE
            qfp.PORTFOLIO_FILE = portfolio_path
            try:
                port = qfp.load_portfolio()
            finally:
                qfp.PORTFOLIO_FILE = original_path

        self.assertEqual(port["equity"], 123.45)


if __name__ == "__main__":
    unittest.main(verbosity=2)
