#!/usr/bin/env python3
"""Truth-surface regressions for the live benchmark gate."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from quantforge_equity import compute_spot_equity, compute_true_equity


class BenchmarkTruthTests(unittest.TestCase):
    def test_true_equity_restores_parked_futures_margin(self):
        port = {
            "cash": 4500.0,
            "btc_qty": 0.0,
            "alt_positions": {},
            "futures_position": {
                "direction": "LONG",
                "margin": 500.0,
                "notional": 1000.0,
                "entry_price": 60000.0,
            },
            "prehedge": {"open": False},
            "liq_dip_position": {},
        }

        self.assertEqual(compute_spot_equity(port, 60000.0), 4500.0)
        self.assertEqual(compute_true_equity(port, 60000.0), 5000.0)

    def test_true_equity_keeps_short_direction_signed(self):
        port = {
            "cash": 4500.0,
            "btc_qty": 0.0,
            "alt_positions": {},
            "futures_position": {
                "direction": "SHORT",
                "margin": 500.0,
                "notional": 1000.0,
                "entry_price": 60000.0,
            },
            "prehedge": {"open": False},
            "liq_dip_position": {},
        }

        self.assertAlmostEqual(compute_true_equity(port, 59400.0), 5010.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
