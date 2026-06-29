#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

from quantforge_agent import infer_portfolio_start_anchor


class PortfolioStartAnchorTests(unittest.TestCase):
    def _write_trades(self, tmpdir, rows):
        path = os.path.join(tmpdir, "agent_trades.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def test_falls_back_from_stale_created_at_for_thin_sample(self):
        port = {
            "created_at": "2023-01-01T00:00:00+00:00",
            "n_trades": 11,
            "rebalance_log": [
                "2026-06-23T16:13:55.717743+00:00",
                "2026-06-24T03:05:08.219407+00:00",
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trades_path = self._write_trades(
                tmpdir,
                [
                    {"ts": "2026-06-23T12:05:08.570738+00:00", "type": "BUY"},
                    {"ts": "2026-06-24T05:05:07.654900+00:00", "type": "SELL"},
                ],
            )
            start_dt, source = infer_portfolio_start_anchor(port, trades_path=trades_path)
        self.assertEqual(start_dt.isoformat(), "2026-06-23T12:05:08.570738+00:00")
        self.assertEqual(source, "recent_spot_trades_fallback")

    def test_keeps_recent_created_at_when_metadata_is_coherent(self):
        port = {
            "created_at": "2026-06-23T13:05:06.779195+00:00",
            "n_trades": 11,
            "rebalance_log": ["2026-06-23T16:13:55.717743+00:00"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trades_path = self._write_trades(
                tmpdir,
                [{"ts": "2026-06-23T13:45:08.570738+00:00", "type": "BUY"}],
            )
            start_dt, source = infer_portfolio_start_anchor(port, trades_path=trades_path)
        self.assertEqual(start_dt.isoformat(), "2026-06-23T13:05:06.779195+00:00")
        self.assertEqual(source, "created_at")


if __name__ == "__main__":
    unittest.main()
