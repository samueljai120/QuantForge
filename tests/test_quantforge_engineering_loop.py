#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import quantforge_engineering_loop as qel


class QuantforgeEngineeringLoopTests(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_missing_heavy_reports_surface_refresh_action(self):
        now = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            last_scan = os.path.join(tmpdir, "last_scan.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")

            self._write_json(diagnosis, {"causes": ["negative_recent_trade_expectancy"]})
            self._write_json(governance, {"recommendation": "REVIEW"})
            self._write_json(promotion, {"overall_decision": "DO_NOT_PROMOTE"})
            self._write_json(last_scan, {"ts": now, "feedback": {"summary": {"risk_mult": 1.0}}})
            self._write_json(outcomes, {"latest": {}, "history": []})

            old = (
                qel.DIAGNOSIS_FILE,
                qel.GOVERNANCE_FILE,
                qel.PROMOTION_FILE,
                qel.LAST_SCAN_FILE,
                qel.CANDIDATE_OUTCOMES_FILE,
                qel.HEAVY_REPORT_SPECS,
            )
            try:
                qel.DIAGNOSIS_FILE = diagnosis
                qel.GOVERNANCE_FILE = governance
                qel.PROMOTION_FILE = promotion
                qel.LAST_SCAN_FILE = last_scan
                qel.CANDIDATE_OUTCOMES_FILE = outcomes
                qel.HEAVY_REPORT_SPECS = {
                    "target_rebuild": os.path.join(tmpdir, "missing-target.json"),
                    "feature_gap": os.path.join(tmpdir, "missing-feature.json"),
                    "segmented_holdout": os.path.join(tmpdir, "missing-segmented.json"),
                    "execution_realism": os.path.join(tmpdir, "missing-execution.json"),
                    "market_data_gap": os.path.join(tmpdir, "missing-market.json"),
                    "data_source_research": os.path.join(tmpdir, "missing-research.json"),
                    "model_layer": os.path.join(tmpdir, "missing-layer.json"),
                }
                payload = qel.build_actions()
            finally:
                (
                    qel.DIAGNOSIS_FILE,
                    qel.GOVERNANCE_FILE,
                    qel.PROMOTION_FILE,
                    qel.LAST_SCAN_FILE,
                    qel.CANDIDATE_OUTCOMES_FILE,
                    qel.HEAVY_REPORT_SPECS,
                ) = old

        refresh = next(row for row in payload["actions"] if row["type"] == "refresh_rebuild_artifacts")
        self.assertEqual(refresh["priority"], "high")
        self.assertIn("stale_rebuild_inputs", payload["summary"])
        self.assertTrue(payload["stale_inputs"])
        self.assertEqual(payload["heavy_inputs"]["model_layer"]["status"], "missing")

    def test_stale_model_layer_blocks_layered_trial_recommendation(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            last_scan = os.path.join(tmpdir, "last_scan.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            stale_model_layer = os.path.join(tmpdir, "model-layer.json")

            self._write_json(diagnosis, {"causes": []})
            self._write_json(governance, {"recommendation": "OBSERVE"})
            self._write_json(promotion, {"overall_decision": "KEEP_IN_PAPER"})
            self._write_json(last_scan, {"ts": now.isoformat(), "feedback": {"summary": {"risk_mult": 1.0}}})
            self._write_json(outcomes, {"latest": {}, "history": []})
            self._write_json(stale_model_layer, {"ready_layers": 3, "total_layers": 4, "next_step": "prepare_layered_trial_candidate"})
            stale_ts = (now - timedelta(hours=qel.HEAVY_REPORT_MAX_AGE_HOURS + 5)).timestamp()
            os.utime(stale_model_layer, (stale_ts, stale_ts))

            old = (
                qel.DIAGNOSIS_FILE,
                qel.GOVERNANCE_FILE,
                qel.PROMOTION_FILE,
                qel.LAST_SCAN_FILE,
                qel.CANDIDATE_OUTCOMES_FILE,
                qel.HEAVY_REPORT_SPECS,
            )
            try:
                qel.DIAGNOSIS_FILE = diagnosis
                qel.GOVERNANCE_FILE = governance
                qel.PROMOTION_FILE = promotion
                qel.LAST_SCAN_FILE = last_scan
                qel.CANDIDATE_OUTCOMES_FILE = outcomes
                qel.HEAVY_REPORT_SPECS = {
                    "target_rebuild": os.path.join(tmpdir, "missing-target.json"),
                    "feature_gap": os.path.join(tmpdir, "missing-feature.json"),
                    "segmented_holdout": os.path.join(tmpdir, "missing-segmented.json"),
                    "execution_realism": os.path.join(tmpdir, "missing-execution.json"),
                    "market_data_gap": os.path.join(tmpdir, "missing-market.json"),
                    "data_source_research": os.path.join(tmpdir, "missing-research.json"),
                    "model_layer": stale_model_layer,
                }
                payload = qel.build_actions()
            finally:
                (
                    qel.DIAGNOSIS_FILE,
                    qel.GOVERNANCE_FILE,
                    qel.PROMOTION_FILE,
                    qel.LAST_SCAN_FILE,
                    qel.CANDIDATE_OUTCOMES_FILE,
                    qel.HEAVY_REPORT_SPECS,
                ) = old

        action_types = {row["type"] for row in payload["actions"]}
        self.assertIn("refresh_rebuild_artifacts", action_types)
        self.assertNotIn("prepare_layered_trial_candidate", action_types)
        self.assertEqual(payload["heavy_inputs"]["model_layer"]["status"], "stale")

    def test_engineering_loop_surfaces_manual_top_alt_expansion_candidate_when_research_hold_support_is_strong(self):
        now = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            last_scan = os.path.join(tmpdir, "last_scan.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            target_rebuild = os.path.join(tmpdir, "target-rebuild.json")
            feature_gap = os.path.join(tmpdir, "feature-gap.json")
            segmented_holdout = os.path.join(tmpdir, "segmented-holdout.json")
            execution_realism = os.path.join(tmpdir, "execution-realism.json")
            market_data_gap = os.path.join(tmpdir, "market-data-gap.json")
            data_source_research = os.path.join(tmpdir, "data-source-research.json")
            model_layer = os.path.join(tmpdir, "model-layer.json")

            self._write_json(diagnosis, {"causes": []})
            self._write_json(governance, {"recommendation": "OBSERVE"})
            self._write_json(promotion, {"overall_decision": "KEEP_IN_PAPER"})
            self._write_json(last_scan, {"ts": now, "feedback": {"summary": {"risk_mult": 1.0}}})
            self._write_json(
                outcomes,
                {
                    "latest": {"type": "competitiveness_gap_rebuild", "assessment": "fail"},
                    "history": [{"recorded_at": now, "type": "competitiveness_gap_rebuild", "assessment": "fail"}],
                },
            )
            self._write_json(
                target_rebuild,
                {
                    "status": "ready",
                    "gates": {"overall_ready": True},
                    "support_counts": {
                        "long_trend_positive": 120,
                        "long_breakout_positive": 70,
                        "long_rebound_positive": 0,
                    },
                    "setup_target_summary": {"trend_long_positive_rate": 0.03, "breakout_long_positive_rate": 0.015},
                },
            )
            for path in (feature_gap, segmented_holdout, execution_realism, market_data_gap, data_source_research, model_layer):
                self._write_json(path, {})

            old = (
                qel.DIAGNOSIS_FILE,
                qel.GOVERNANCE_FILE,
                qel.PROMOTION_FILE,
                qel.LAST_SCAN_FILE,
                qel.CANDIDATE_OUTCOMES_FILE,
                qel.HEAVY_REPORT_SPECS,
                qel.summarize_top_alt_research_hold_support,
            )
            try:
                qel.DIAGNOSIS_FILE = diagnosis
                qel.GOVERNANCE_FILE = governance
                qel.PROMOTION_FILE = promotion
                qel.LAST_SCAN_FILE = last_scan
                qel.CANDIDATE_OUTCOMES_FILE = outcomes
                qel.HEAVY_REPORT_SPECS = {
                    "target_rebuild": target_rebuild,
                    "feature_gap": feature_gap,
                    "segmented_holdout": segmented_holdout,
                    "execution_realism": execution_realism,
                    "market_data_gap": market_data_gap,
                    "data_source_research": data_source_research,
                    "model_layer": model_layer,
                }
                qel.summarize_top_alt_research_hold_support = lambda: {
                    "status": "ready",
                    "expansion_supported": True,
                    "top_non_major_symbols": [
                        {"symbol": "TAO-USDT", "long_positive_total": 131},
                        {"symbol": "DOGE-USDT", "long_positive_total": 45},
                        {"symbol": "LINK-USDT", "long_positive_total": 31},
                    ],
                }
                payload = qel.build_actions()
            finally:
                (
                    qel.DIAGNOSIS_FILE,
                    qel.GOVERNANCE_FILE,
                    qel.PROMOTION_FILE,
                    qel.LAST_SCAN_FILE,
                    qel.CANDIDATE_OUTCOMES_FILE,
                    qel.HEAVY_REPORT_SPECS,
                    qel.summarize_top_alt_research_hold_support,
                ) = old

        action = next(row for row in payload["actions"] if row["type"] == "prepare_major_liquidity_expansion_candidate")
        self.assertIn("top_alt_long_expansion_supported", payload["summary"])
        self.assertEqual(action["execution_policy"], "manual_only")
        self.assertFalse(action["llm_eligible"])
        self.assertIn("TAO-USDT", action["why"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
