#!/usr/bin/env python3
"""Tests for quantforge_strategy_retire."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import quantforge_strategy_retire as qsr


# Two cycles ~21 days apart in which strategy 'foo_gen1' is the only active
# strategy and equity falls 5000 -> 4900. That yields a negative cumulative
# PnL for a strategy older than the 20-day window, which audit() must flag
# as RETIRE (the "neg PnL for >= 20d" branch).
_LOG_FIXTURE = """\
[2026-01-01T00:00:00.000000+00:00] === Agent cycle start ===
[2026-01-01T00:00:00.000000+00:00]   Strategy 'foo_gen1' (weight 100%): ACTIVE target=100% regime=NEUTRAL
[2026-01-01T00:00:00.000000+00:00] === Cycle end. Equity $5,000.00  PnL $+0.00 (+0.00%)  Regime NEUTRAL  Trades total 0 ===
[2026-01-22T00:00:00.000000+00:00] === Agent cycle start ===
[2026-01-22T00:00:00.000000+00:00]   Strategy 'foo_gen1' (weight 100%): ACTIVE target=100% regime=NEUTRAL
[2026-01-22T00:00:00.000000+00:00] === Cycle end. Equity $4,900.00  PnL $-100.00 (-2.00%)  Regime NEUTRAL  Trades total 0 ===
"""


class TestStrategyRetire(unittest.TestCase):
    def test_audit_flags_aged_losing_strategy_for_retirement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "agent.log")
            trades_path = os.path.join(tmpdir, "agent_trades.jsonl")  # absent on purpose
            with open(log_path, "w") as f:
                f.write(_LOG_FIXTURE)

            with mock.patch.object(qsr, "LOG_FILE", log_path), \
                 mock.patch.object(qsr, "TRADES_FILE", trades_path):
                results = qsr.audit()

        by_name = {r["name"]: r for r in results}
        self.assertIn("foo_gen1", by_name)
        self.assertEqual(by_name["foo_gen1"]["status"], "RETIRE")
        self.assertLess(by_name["foo_gen1"]["cumulative_pnl"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
