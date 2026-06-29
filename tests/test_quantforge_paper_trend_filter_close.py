#!/usr/bin/env python3
import unittest

import quantforge_paper as qfp


class QuantforgePaperTrendFilterCloseTests(unittest.TestCase):
    def test_trend_filter_close_reads_close_from_kucoin_candle_shape(self):
        candle = [1719420000, "100.0", "101.5", "103.0", "99.0", "42.0", "4200.0"]
        self.assertEqual(qfp._trend_filter_close(candle), 101.5)

    def test_trend_filter_close_reads_close_from_dict_without_low_fallback(self):
        candle = {"open": "100.0", "close": "101.5", "high": "103.0", "low": "99.0"}
        self.assertEqual(qfp._trend_filter_close(candle), 101.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
