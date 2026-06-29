#!/usr/bin/env python3
"""Regression tests for benchmark-start inference in quantforge_benchmark_gate."""

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import quantforge_benchmark_gate as qbg


def _write_trades(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(f"{row}\n")


class TestInferStartAnchor(unittest.TestCase):
    def test_falls_back_from_stale_created_at_for_thin_sample(self):
        port = {
            "created_at": "2023-01-01T00:00:00+00:00",
            "n_trades": 11,
            "rebalance_log": [
                "2026-06-23T16:13:55.717743+00:00",
                "2026-06-23T19:05:08.635971+00:00",
            ],
        }
        trades = [
            '{"ts":"2026-06-22T18:05:08.462028+00:00","type":"BUY"}',
            '{"ts":"2026-06-23T11:05:09.056595+00:00","type":"SELL"}',
            '{"ts":"2026-06-23T12:05:08.570738+00:00","type":"BUY"}',
            '{"ts":"2026-06-23T16:13:55.717648+00:00","type":"SELL"}',
            '{"ts":"2026-06-23T19:05:08.635971+00:00","type":"BUY"}',
            '{"ts":"2026-06-24T03:05:08.219306+00:00","type":"SELL"}',
            '{"ts":"2026-06-24T04:05:06.274078+00:00","type":"SELL"}',
            '{"ts":"2026-06-24T05:05:07.654805+00:00","type":"SELL"}',
            '{"ts":"2026-06-24T17:05:08.517315+00:00","type":"BUY"}',
            '{"ts":"2026-06-24T20:05:08.389720+00:00","type":"SELL"}',
            '{"ts":"2026-06-25T03:05:10.246030+00:00","type":"BUY"}',
            '{"ts":"2026-06-25T07:05:08.139787+00:00","type":"SELL"}',
            '{"ts":"2026-06-25T08:05:08.645850+00:00","type":"SELL"}',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            trades_path = os.path.join(tmpdir, "agent_trades.jsonl")
            _write_trades(trades_path, trades)
            start_dt, source = qbg.infer_start_anchor(port, trades_path=trades_path)

        self.assertEqual(start_dt.isoformat(), "2026-06-23T12:05:08.570738+00:00")
        self.assertEqual(source, "recent_spot_trades_fallback")

    def test_keeps_recent_created_at_when_metadata_is_coherent(self):
        port = {
            "created_at": "2026-06-23T13:05:06.779195+00:00",
            "n_trades": 11,
            "rebalance_log": ["2026-06-23T16:13:55.717743+00:00"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trades_path = os.path.join(tmpdir, "agent_trades.jsonl")
            _write_trades(trades_path, ['{"ts":"2026-06-25T08:05:08.645850+00:00","type":"SELL"}'])
            start_dt, source = qbg.infer_start_anchor(port, trades_path=trades_path)

        self.assertEqual(start_dt.isoformat(), "2026-06-23T13:05:06.779195+00:00")
        self.assertEqual(source, "created_at")

    def test_keeps_long_window_for_mature_sample(self):
        port = {
            "created_at": "2026-05-01T00:00:00+00:00",
            "n_trades": 40,
            "rebalance_log": ["2026-06-23T16:13:55.717743+00:00"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trades_path = os.path.join(tmpdir, "agent_trades.jsonl")
            _write_trades(trades_path, ['{"ts":"2026-06-25T08:05:08.645850+00:00","type":"SELL"}'])
            start_dt, source = qbg.infer_start_anchor(port, trades_path=trades_path)

        self.assertEqual(start_dt.isoformat(), "2026-05-01T00:00:00+00:00")
        self.assertEqual(source, "created_at")


if __name__ == "__main__":
    unittest.main(verbosity=2)
