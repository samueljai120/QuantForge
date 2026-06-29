#!/usr/bin/env python3
import unittest

import quantforge_paper as qfp


class QuantforgeRrGateTests(unittest.TestCase):
    def test_rr_gate_allows_effective_equality_with_rounding_noise(self):
        self.assertFalse(qfp._rr_gate_blocks(1.9999999, 2.0))

    def test_rr_gate_blocks_clear_shortfall(self):
        self.assertTrue(qfp._rr_gate_blocks(1.98, 2.0))

    def test_entry_exit_levels_compute_rr_before_rounding_prices(self):
        entry_price = 0.100049
        stop_pct = qfp.TAKE_PROFIT_PCT / 2.0
        sl, tp, rr = qfp._entry_exit_levels(entry_price, stop_pct, "SHORT")

        rounded_rr = abs(tp - entry_price) / max(abs(entry_price - sl), 1e-10)

        self.assertFalse(qfp._rr_gate_blocks(rr, 2.0))
        self.assertTrue(qfp._rr_gate_blocks(rounded_rr, 2.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
