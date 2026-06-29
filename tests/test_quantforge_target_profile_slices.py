#!/usr/bin/env python3
import unittest

import pandas as pd

import quantforge_target_profiles as qtp


class QuantforgeTargetProfileSliceTests(unittest.TestCase):
    def test_majors_non_fragile_slice_keeps_only_major_non_fragile_rows(self):
        df = pd.DataFrame([
            {"symbol": "ETH-USDT", "fakeout_risk": 0.20},
            {"symbol": "ADA-USDT", "fakeout_risk": 0.10},
            {"symbol": "SOL-USDT", "fakeout_risk": 0.72},
        ])

        filtered, summary = qtp.apply_research_rebuild_slice(
            df,
            slice_profile=qtp.RESEARCH_SLICE_PROFILE_MAJORS_NON_FRAGILE,
        )

        self.assertEqual(filtered["symbol"].tolist(), ["ETH-USDT"])
        self.assertEqual(summary["kept_rows"], 1)
        self.assertEqual(summary["support"]["major_rows"], 2)

    def test_majors_positive_long_slices_require_major_non_fragile_long_bias(self):
        df = pd.DataFrame([
            {
                "symbol": "ETH-USDT",
                "fakeout_risk": 0.20,
                "adx": 24.0,
                "setup_trend_long_score": 0.74,
                "setup_breakout_long_score": 0.50,
                "setup_trend_short_score": 0.40,
                "setup_exhaustion_short_score": 0.20,
            },
            {
                "symbol": "SOL-USDT",
                "fakeout_risk": 0.10,
                "adx": 23.0,
                "setup_trend_long_score": 0.61,
                "setup_breakout_long_score": 0.63,
                "setup_trend_short_score": 0.10,
                "setup_exhaustion_short_score": 0.05,
            },
            {
                "symbol": "XRP-USDT",
                "fakeout_risk": 0.10,
                "adx": 22.0,
                "setup_trend_long_score": 0.68,
                "setup_breakout_long_score": 0.30,
                "setup_trend_short_score": 0.71,
                "setup_exhaustion_short_score": 0.15,
            },
            {
                "symbol": "BCH-USDT",
                "fakeout_risk": 0.80,
                "adx": 30.0,
                "setup_trend_long_score": 0.80,
                "setup_breakout_long_score": 0.20,
                "setup_trend_short_score": 0.10,
                "setup_exhaustion_short_score": 0.05,
            },
            {
                "symbol": "ADA-USDT",
                "fakeout_risk": 0.10,
                "adx": 25.0,
                "setup_trend_long_score": 0.80,
                "setup_breakout_long_score": 0.20,
                "setup_trend_short_score": 0.10,
                "setup_exhaustion_short_score": 0.05,
            },
        ])

        filtered, summary = qtp.apply_research_rebuild_slice(
            df,
            slice_profile=qtp.RESEARCH_SLICE_PROFILE_MAJORS_POSITIVE_LONG_SLICES,
        )

        self.assertEqual(filtered["symbol"].tolist(), ["ETH-USDT"])
        self.assertEqual(summary["profile"], "majors_positive_long_slices")
        self.assertTrue(summary["active"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
