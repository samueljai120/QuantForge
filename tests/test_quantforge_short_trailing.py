#!/usr/bin/env python3
import unittest

import quantforge_paper as qfp


class QuantforgeShortTrailingTests(unittest.TestCase):
    def test_short_trailing_waits_until_min_r_profit(self):
        pos = {
            "direction": "SHORT",
            "entry_price": 100.0,
            "initial_stop_loss": 110.0,
            "initial_risk_per_unit": 10.0,
            "stop_loss": 110.0,
            "best_price": 100.0,
            "trailing_active": False,
            "open_ts": "1970-01-01T00:00:00+00:00",
        }

        update = qfp._update_trailing_exit(pos, 95.0, candles=None)

        self.assertIsNone(update)
        self.assertFalse(pos["trailing_active"])
        self.assertEqual(pos["stop_loss"], 110.0)

    def test_short_trailing_uses_profile_giveback_not_fixed_half_percent(self):
        pos = {
            "direction": "SHORT",
            "entry_price": 100.0,
            "initial_stop_loss": 110.0,
            "initial_risk_per_unit": 10.0,
            "stop_loss": 110.0,
            "best_price": 100.0,
            "trailing_active": False,
            "open_ts": "1970-01-01T00:00:00+00:00",
        }
        candles = [
            [1, 0, 100.0, 101.0, 99.0, 0, 0],
            [2, 0, 95.0, 96.0, 94.0, 0, 0],
            [3, 0, 90.0, 91.0, 89.0, 0, 0],
            [4, 0, 80.0, 81.0, 79.0, 0, 0],
        ]

        update = qfp._update_trailing_exit(pos, 80.0, candles=candles)

        self.assertIsNotNone(update)
        self.assertEqual(update["trailing_tier"], "fade_lock")
        self.assertAlmostEqual(pos["stop_loss"], 80.64, places=2)
        self.assertAlmostEqual(update["stop_loss_after"], 80.64, places=2)
        for key in (
            "giveback_pct",
            "lock_share",
            "best_price_before",
            "best_price_after",
            "trailing_was_active",
            "trailing_is_active",
        ):
            self.assertIn(key, update)
        self.assertEqual(update["best_price_before"], 100.0)
        self.assertEqual(update["best_price_after"], 80.0)
        self.assertFalse(update["trailing_was_active"])
        self.assertTrue(update["trailing_is_active"])

    def test_long_trailing_update_exposes_check_stops_fields(self):
        pos = {
            "direction": "LONG",
            "entry_price": 100.0,
            "initial_stop_loss": 90.0,
            "initial_risk_per_unit": 10.0,
            "stop_loss": 90.0,
            "best_price": 100.0,
            "trailing_active": False,
            "open_ts": "1970-01-01T00:00:00+00:00",
        }

        update = qfp._update_trailing_exit(pos, 120.0, candles=None)

        self.assertIsNotNone(update)
        self.assertEqual(update["trailing_tier"], "r_lock")
        self.assertEqual(update["best_price_before"], 100.0)
        self.assertEqual(update["best_price_after"], 120.0)
        self.assertAlmostEqual(update["giveback_pct"], 0.005, places=6)
        self.assertFalse(update["trailing_was_active"])
        self.assertTrue(update["trailing_is_active"])

    def test_tp2_lock_matches_locked_half_r_label_for_short(self):
        pos = {
            "direction": "SHORT",
            "entry_price": 100.0,
            "initial_risk_per_unit": 10.0,
            "stop_loss": 99.8,
        }

        qfp._apply_partial_profit_stop_lock(pos, "tp2_done")

        self.assertEqual(pos["stop_loss"], 95.0)

    def test_tp1_lock_stays_near_breakeven_buffer_for_long(self):
        pos = {
            "direction": "LONG",
            "entry_price": 100.0,
            "initial_risk_per_unit": 10.0,
            "stop_loss": 95.0,
        }

        qfp._apply_partial_profit_stop_lock(pos, "tp1_done")

        self.assertEqual(pos["stop_loss"], 100.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
