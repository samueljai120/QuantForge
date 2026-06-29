#!/usr/bin/env python3
"""Regression tests for futures_kill flatten behavior."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import quantforge_agent as qa


class FuturesKillFlattenTests(unittest.TestCase):
    def setUp(self):
        self._old_log = qa.log
        self._old_append_trade = qa.append_trade
        self._old_scan = sys.modules.get("quantforge_ml_scanner")
        qa.log = lambda *a, **k: None
        self.trades = []
        qa.append_trade = lambda t: self.trades.append(t)

        class _StubScanner:
            @staticmethod
            def scan_coins(*a, **k):
                return {"model_ok": False, "picks": []}

        sys.modules["quantforge_ml_scanner"] = _StubScanner()

    def tearDown(self):
        qa.log = self._old_log
        qa.append_trade = self._old_append_trade
        if self._old_scan is None:
            sys.modules.pop("quantforge_ml_scanner", None)
        else:
            sys.modules["quantforge_ml_scanner"] = self._old_scan

    def test_existing_open_position_is_flattened_when_kill_is_engaged(self):
        port = {
            "cash": 1000.0,
            "futures_kill": True,
            "futures_pnl": 0.0,
            "starting_balance": 5000.0,
            "futures_position": {
                "direction": "LONG",
                "margin": 100.0,
                "notional": 1000.0,
                "entry_price": 100.0,
                "opened_at": "2026-06-26T09:00:00+00:00",
            },
        }

        qa._execute_futures(port, 90.0, "LONG", "BULL", 2000.0, signals=None, consensus=0)

        self.assertIsNone(port["futures_position"]["direction"])
        self.assertEqual(len(self.trades), 1)
        self.assertEqual(self.trades[0]["type"], "FUTURES_CLOSE")
        self.assertEqual(self.trades[0]["reason"], "futures_kill_engaged")


if __name__ == "__main__":
    unittest.main(verbosity=2)
