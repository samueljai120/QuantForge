#!/usr/bin/env python3
"""Regression tests for self-heal truth checks around futures kill."""

import json
import os
import tempfile
import unittest

import quantforge_self_heal_actions as qsha


class QuantforgeSelfHealTruthTests(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_fix_futures_kill_stays_engaged_while_critical_invariant_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = qsha.DATA_DIR
            old_invariants = qsha.INVARIANTS_STATE_FILE
            old_price_now = qsha._btc_price_now
            try:
                qsha.DATA_DIR = tmpdir
                qsha.INVARIANTS_STATE_FILE = os.path.join(tmpdir, "qf_invariants_state.json")
                qsha._btc_price_now = lambda: 60000.0

                self._write_json(
                    os.path.join(tmpdir, "agent_portfolio.json"),
                    {
                        "futures_kill": True,
                        "starting_balance": 5000.0,
                        "peak_equity": 5000.0,
                        "cash": 5000.0,
                        "btc_qty": 0.0,
                        "futures_position": {"direction": None, "margin": 0.0, "notional": 0.0, "entry_price": 0.0},
                    },
                )
                self._write_json(
                    qsha.INVARIANTS_STATE_FILE,
                    {"n_critical": 1, "violations": [{"name": "futures_open_close_parity"}]},
                )

                result = qsha._fix_futures_kill()
            finally:
                qsha.DATA_DIR = old_data_dir
                qsha.INVARIANTS_STATE_FILE = old_invariants
                qsha._btc_price_now = old_price_now

        self.assertIn("critical invariants still present", result)

    def test_fix_futures_kill_clears_only_when_live_drawdown_is_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = qsha.DATA_DIR
            old_invariants = qsha.INVARIANTS_STATE_FILE
            old_price_now = qsha._btc_price_now
            try:
                qsha.DATA_DIR = tmpdir
                qsha.INVARIANTS_STATE_FILE = os.path.join(tmpdir, "qf_invariants_state.json")
                qsha._btc_price_now = lambda: 60000.0

                portfolio_path = os.path.join(tmpdir, "agent_portfolio.json")
                self._write_json(
                    portfolio_path,
                    {
                        "futures_kill": True,
                        "starting_balance": 5000.0,
                        "peak_equity": 5000.0,
                        "cash": 5000.0,
                        "btc_qty": 0.0,
                        "futures_position": {"direction": None, "margin": 0.0, "notional": 0.0, "entry_price": 0.0},
                    },
                )
                self._write_json(qsha.INVARIANTS_STATE_FILE, {"n_critical": 0, "violations": []})

                result = qsha._fix_futures_kill()
                with open(portfolio_path) as f:
                    updated = json.load(f)
            finally:
                qsha.DATA_DIR = old_data_dir
                qsha.INVARIANTS_STATE_FILE = old_invariants
                qsha._btc_price_now = old_price_now

        self.assertIn("CLEARED", result)
        self.assertFalse(updated["futures_kill"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
