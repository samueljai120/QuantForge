#!/usr/bin/env python3
import unittest
from datetime import timezone

import quantforge_autopilot as qauto
import quantforge_candidate_recovery as qrec
import quantforge_candidate_review as qrev
import quantforge_harness_report as qhr


class QuantforgeNaiveModelTimestampTests(unittest.TestCase):
    def test_parse_ts_normalizes_naive_model_timestamp_to_utc(self):
        for mod in (qauto, qrec, qrev, qhr):
            dt = mod._parse_ts("2026-06-26T18:30:24.561053")
            self.assertIsNotNone(dt)
            self.assertEqual(dt.tzinfo, timezone.utc)

    def test_stale_outcome_comparison_accepts_naive_model_timestamp(self):
        latest = {"recorded_at": "2026-06-26T13:40:34.506891+00:00"}
        candidate = {"model_trained_at": "2026-06-26T18:30:24.561053"}
        self.assertTrue(qhr._outcome_is_stale_for_candidate(latest, candidate))
        self.assertTrue(qauto._outcome_is_stale_for_candidate(latest, candidate))
        self.assertTrue(qrec._outcome_is_stale_for_candidate(latest, candidate))
        self.assertTrue(qrev._outcome_is_stale_for_candidate(latest, candidate))


if __name__ == "__main__":
    unittest.main(verbosity=2)
