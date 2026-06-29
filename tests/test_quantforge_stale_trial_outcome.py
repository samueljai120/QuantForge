#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

import quantforge_autopilot as qa
import quantforge_candidate_recovery as qcr


class QuantforgeStaleTrialOutcomeTests(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_candidate_recovery_ignores_trial_outcome_older_than_current_candidate_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            agi_history = os.path.join(tmpdir, "agi-history.json")
            monitor = os.path.join(tmpdir, "monitor.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            model_layer = os.path.join(tmpdir, "model-layer.json")
            output = os.path.join(tmpdir, "candidate-recovery.json")

            self._write_json(diagnosis, {"causes": ["persistent_underperformance_across_cycles"]})
            self._write_json(governance, {"recommendation": "REVIEW", "paper": {"total_pnl_pct": 0.0}, "recent_closes": {"win_rate": 0.0}})
            self._write_json(promotion, {"overall_decision": "KEEP_IN_PAPER"})
            self._write_json(agi_history, {"posture": "degraded", "persistent_review": True, "persistent_drift": True, "averages": {"adaptive_risk_mult": 1.0}})
            self._write_json(monitor, {"drift_flags": ["low_signal_activity"], "regime": {"label": "BEAR", "entropy_label": "ORDERLY", "entropy": 0.6}})
            self._write_json(
                lanes,
                {
                    "baseline": {"paper_total_pnl_pct": 0.0},
                    "candidate": {"paper_total_pnl_pct": 0.0, "model_trained_at": "2026-06-22T02:12:37.166088+00:00"},
                    "candidate_trial": None,
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-04-20T15:40:34.431098+00:00",
                        "type": "competitiveness_gap_rebuild",
                        "assessment": "fail",
                        "next_candidate_hint": "quantforge_research_hold",
                    }
                },
            )
            self._write_json(model_layer, {"status": "layer_rebuild_in_progress", "next_step": "retrain_prediction_layer_with_rebuilt_context"})
            self._write_json(output, {})

            old = (
                qcr.DIAGNOSIS_FILE,
                qcr.GOVERNANCE_FILE,
                qcr.PROMOTION_FILE,
                qcr.AGI_HISTORY_FILE,
                qcr.MONITOR_FILE,
                qcr.LANES_FILE,
                qcr.CANDIDATE_OUTCOMES_FILE,
                qcr.MODEL_LAYER_FILE,
                qcr.OUTPUT_FILE,
            )
            try:
                qcr.DIAGNOSIS_FILE = diagnosis
                qcr.GOVERNANCE_FILE = governance
                qcr.PROMOTION_FILE = promotion
                qcr.AGI_HISTORY_FILE = agi_history
                qcr.MONITOR_FILE = monitor
                qcr.LANES_FILE = lanes
                qcr.CANDIDATE_OUTCOMES_FILE = outcomes
                qcr.MODEL_LAYER_FILE = model_layer
                qcr.OUTPUT_FILE = output
                payload = qcr.build_candidate()
            finally:
                (
                    qcr.DIAGNOSIS_FILE,
                    qcr.GOVERNANCE_FILE,
                    qcr.PROMOTION_FILE,
                    qcr.AGI_HISTORY_FILE,
                    qcr.MONITOR_FILE,
                    qcr.LANES_FILE,
                    qcr.CANDIDATE_OUTCOMES_FILE,
                    qcr.MODEL_LAYER_FILE,
                    qcr.OUTPUT_FILE,
                ) = old

        self.assertNotEqual(payload["type"], "quantforge_research_hold")
        self.assertTrue(payload["evidence"]["stale_latest_outcome_ignored"])

    def test_autopilot_does_not_freeze_from_stale_trial_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            portfolio = os.path.join(tmpdir, "portfolio.json")
            history = os.path.join(tmpdir, "history.json")
            monitor = os.path.join(tmpdir, "monitor.json")
            agi_history = os.path.join(tmpdir, "agi-history.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            harness = os.path.join(tmpdir, "harness.json")
            last_scan = os.path.join(tmpdir, "last-scan.json")

            now = datetime.now(timezone.utc).isoformat()
            self._write_json(governance, {"generated_at": now, "recommendation": "REVIEW"})
            self._write_json(promotion, {"overall_decision": "KEEP_IN_PAPER"})
            self._write_json(diagnosis, {"causes": ["persistent_underperformance_across_cycles"]})
            self._write_json(
                lanes,
                {
                    "baseline": {"paper_total_pnl_pct": 0.0},
                    "candidate": {"paper_total_pnl_pct": 0.0, "model_trained_at": "2026-06-22T02:12:37.166088+00:00"},
                    "candidate_trial": None,
                },
            )
            self._write_json(portfolio, {"updated": now, "positions": {}})
            self._write_json(history, {"status": "ok", "cycles_sampled": 96, "posture": "degraded", "averages": {}, "ratios": {}})
            self._write_json(monitor, {"generated_at": now, "health": "DRIFTING", "regime": {}, "drift_flags": []})
            self._write_json(agi_history, {"status": "ok", "persistent_drift": True, "persistent_review": True, "averages": {}})
            self._write_json(recovery, {"type": "defensive_recovery", "status": "proposed", "priority": "high"})
            self._write_json(last_scan, {"ts": now})
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-04-20T15:40:34.431098+00:00",
                        "type": "competitiveness_gap_rebuild",
                        "assessment": "fail",
                        "next_candidate_hint": "quantforge_research_hold",
                    }
                },
            )
            self._write_json(harness, {"status": "ok", "recommendation": "allow_progress", "failed_count": 0, "summary": {}})

            old = (
                qa.GOVERNANCE_FILE,
                qa.PROMOTION_FILE,
                qa.DIAGNOSIS_FILE,
                qa.LANES_FILE,
                qa.PORTFOLIO_FILE,
                qa.HISTORY_FILE,
                qa.MONITOR_FILE,
                qa.AGI_OPERATOR_HISTORY_FILE,
                qa.CANDIDATE_RECOVERY_FILE,
                qa.CANDIDATE_OUTCOMES_FILE,
                qa.HARNESS_FILE,
            )
            try:
                qa.GOVERNANCE_FILE = governance
                qa.PROMOTION_FILE = promotion
                qa.DIAGNOSIS_FILE = diagnosis
                qa.LANES_FILE = lanes
                qa.PORTFOLIO_FILE = portfolio
                qa.HISTORY_FILE = history
                qa.MONITOR_FILE = monitor
                qa.AGI_OPERATOR_HISTORY_FILE = agi_history
                qa.CANDIDATE_RECOVERY_FILE = recovery
                qa.CANDIDATE_OUTCOMES_FILE = outcomes
                qa.HARNESS_FILE = harness
                original_read_json = qa.read_json
                qa.read_json = lambda path: original_read_json(last_scan) if path.endswith("last_scan.json") else original_read_json(path)
                report = qa.build_report()
            finally:
                qa.read_json = original_read_json
                (
                    qa.GOVERNANCE_FILE,
                    qa.PROMOTION_FILE,
                    qa.DIAGNOSIS_FILE,
                    qa.LANES_FILE,
                    qa.PORTFOLIO_FILE,
                    qa.HISTORY_FILE,
                    qa.MONITOR_FILE,
                    qa.AGI_OPERATOR_HISTORY_FILE,
                    qa.CANDIDATE_RECOVERY_FILE,
                    qa.CANDIDATE_OUTCOMES_FILE,
                    qa.HARNESS_FILE,
                ) = old

        self.assertEqual(report["mode"], "hold_in_paper")
        self.assertTrue(report["latest_trial_outcome"]["stale_for_candidate"])

    def test_candidate_recovery_promotes_research_hold_to_major_expansion_when_top_alt_support_is_strong(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            agi_history = os.path.join(tmpdir, "agi-history.json")
            monitor = os.path.join(tmpdir, "monitor.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            model_layer = os.path.join(tmpdir, "model-layer.json")
            output = os.path.join(tmpdir, "candidate-recovery.json")

            self._write_json(diagnosis, {"causes": []})
            self._write_json(governance, {"recommendation": "REVIEW", "paper": {"total_pnl_pct": -3.1}, "recent_closes": {"win_rate": 0.33}})
            self._write_json(promotion, {"overall_decision": "DO_NOT_PROMOTE"})
            self._write_json(agi_history, {"posture": "degraded", "persistent_review": True, "persistent_drift": True, "averages": {"adaptive_risk_mult": 0.58}})
            self._write_json(monitor, {"drift_flags": [], "regime": {"label": "BULL", "entropy_label": "ORDERLY", "entropy": 0.35}})
            self._write_json(
                lanes,
                {
                    "baseline": {"paper_total_pnl_pct": -1.2},
                    "candidate": {"paper_total_pnl_pct": -2.7, "model_trained_at": "2026-06-26T02:12:37.166088+00:00"},
                    "candidate_trial": None,
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-06-27T03:40:34.431098+00:00",
                        "type": "competitiveness_gap_rebuild",
                        "assessment": "fail",
                        "next_candidate_hint": "quantforge_research_hold",
                    }
                },
            )
            self._write_json(model_layer, {"status": "layer_rebuild_in_progress", "next_step": "retrain_prediction_layer_with_rebuilt_context"})
            self._write_json(output, {})

            old = (
                qcr.DIAGNOSIS_FILE,
                qcr.GOVERNANCE_FILE,
                qcr.PROMOTION_FILE,
                qcr.AGI_HISTORY_FILE,
                qcr.MONITOR_FILE,
                qcr.LANES_FILE,
                qcr.CANDIDATE_OUTCOMES_FILE,
                qcr.MODEL_LAYER_FILE,
                qcr.OUTPUT_FILE,
                qcr.summarize_top_alt_research_hold_support,
            )
            try:
                qcr.DIAGNOSIS_FILE = diagnosis
                qcr.GOVERNANCE_FILE = governance
                qcr.PROMOTION_FILE = promotion
                qcr.AGI_HISTORY_FILE = agi_history
                qcr.MONITOR_FILE = monitor
                qcr.LANES_FILE = lanes
                qcr.CANDIDATE_OUTCOMES_FILE = outcomes
                qcr.MODEL_LAYER_FILE = model_layer
                qcr.OUTPUT_FILE = output
                qcr.summarize_top_alt_research_hold_support = lambda: {
                    "status": "ready",
                    "expansion_supported": True,
                    "allowed_long_setups": ["trend_long", "breakout_long"],
                    "top_non_major_symbols": [
                        {"symbol": "TAO-USDT", "long_positive_total": 131, "long_positive_rate": 0.043725},
                        {"symbol": "DOGE-USDT", "long_positive_total": 45, "long_positive_rate": 0.0211},
                        {"symbol": "LINK-USDT", "long_positive_total": 31, "long_positive_rate": 0.0184},
                    ],
                }
                payload = qcr.build_candidate()
            finally:
                (
                    qcr.DIAGNOSIS_FILE,
                    qcr.GOVERNANCE_FILE,
                    qcr.PROMOTION_FILE,
                    qcr.AGI_HISTORY_FILE,
                    qcr.MONITOR_FILE,
                    qcr.LANES_FILE,
                    qcr.CANDIDATE_OUTCOMES_FILE,
                    qcr.MODEL_LAYER_FILE,
                    qcr.OUTPUT_FILE,
                    qcr.summarize_top_alt_research_hold_support,
                ) = old

        self.assertEqual(payload["type"], "major_liquidity_expansion")
        self.assertIn("TAO-USDT", payload["rationale"][1])
        self.assertTrue(payload["evidence"]["top_alt_research_hold_support"]["expansion_supported"])
        self.assertIn(
            {"key": "allowed_long_setups", "action": "restrict", "value": ["trend_long", "breakout_long"]},
            payload["changes"],
        )

    def test_autopilot_does_not_reassert_rebuild_hold_when_new_trial_is_already_queued(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            portfolio = os.path.join(tmpdir, "portfolio.json")
            history = os.path.join(tmpdir, "history.json")
            monitor = os.path.join(tmpdir, "monitor.json")
            agi_history = os.path.join(tmpdir, "agi-history.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            harness = os.path.join(tmpdir, "harness.json")
            last_scan = os.path.join(tmpdir, "last-scan.json")

            now = datetime.now(timezone.utc).isoformat()
            self._write_json(governance, {"generated_at": now, "recommendation": "REVIEW"})
            self._write_json(promotion, {"overall_decision": "DO_NOT_PROMOTE"})
            self._write_json(diagnosis, {"causes": ["persistent_underperformance_across_cycles"]})
            self._write_json(
                lanes,
                {
                    "baseline": {"paper_total_pnl_pct": 0.0},
                    "candidate": {"paper_total_pnl_pct": -1.0, "model_trained_at": now},
                    "candidate_trial": {
                        "candidate_id": "major_liquidity_expansion:20260628T001543Z",
                        "type": "major_liquidity_expansion",
                        "status": "queued",
                        "paper_only": True,
                    },
                },
            )
            self._write_json(portfolio, {"updated": now, "positions": {"IP-USDT": {"direction": "SHORT"}}})
            self._write_json(history, {"status": "ok", "cycles_sampled": 96, "posture": "degraded", "averages": {}, "ratios": {}})
            self._write_json(monitor, {"generated_at": now, "health": "DRIFTING", "regime": {}, "drift_flags": []})
            self._write_json(agi_history, {"status": "ok", "persistent_drift": True, "persistent_review": True, "averages": {}})
            self._write_json(recovery, {"candidate_id": "major_liquidity_expansion:20260628T001543Z", "type": "major_liquidity_expansion", "status": "proposed", "priority": "high"})
            self._write_json(last_scan, {"ts": now})
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": now,
                        "type": "quantforge_layered_trial",
                        "assessment": "fail",
                        "next_candidate_hint": "quantforge_research_hold",
                    }
                },
            )
            self._write_json(harness, {"status": "ok", "recommendation": "queue_candidate_trial", "failed_count": 0, "summary": {}})

            old = (
                qa.GOVERNANCE_FILE,
                qa.PROMOTION_FILE,
                qa.DIAGNOSIS_FILE,
                qa.LANES_FILE,
                qa.PORTFOLIO_FILE,
                qa.HISTORY_FILE,
                qa.MONITOR_FILE,
                qa.AGI_OPERATOR_HISTORY_FILE,
                qa.CANDIDATE_RECOVERY_FILE,
                qa.CANDIDATE_OUTCOMES_FILE,
                qa.HARNESS_FILE,
            )
            try:
                qa.GOVERNANCE_FILE = governance
                qa.PROMOTION_FILE = promotion
                qa.DIAGNOSIS_FILE = diagnosis
                qa.LANES_FILE = lanes
                qa.PORTFOLIO_FILE = portfolio
                qa.HISTORY_FILE = history
                qa.MONITOR_FILE = monitor
                qa.AGI_OPERATOR_HISTORY_FILE = agi_history
                qa.CANDIDATE_RECOVERY_FILE = recovery
                qa.CANDIDATE_OUTCOMES_FILE = outcomes
                qa.HARNESS_FILE = harness
                original_read_json = qa.read_json
                qa.read_json = lambda path: original_read_json(last_scan) if path.endswith("last_scan.json") else original_read_json(path)
                report = qa.build_report()
            finally:
                qa.read_json = original_read_json
                (
                    qa.GOVERNANCE_FILE,
                    qa.PROMOTION_FILE,
                    qa.DIAGNOSIS_FILE,
                    qa.LANES_FILE,
                    qa.PORTFOLIO_FILE,
                    qa.HISTORY_FILE,
                    qa.MONITOR_FILE,
                    qa.AGI_OPERATOR_HISTORY_FILE,
                    qa.CANDIDATE_RECOVERY_FILE,
                    qa.CANDIDATE_OUTCOMES_FILE,
                    qa.HARNESS_FILE,
                ) = old

        self.assertEqual(report["mode"], "pause_new_entries")
        self.assertIn("respect_candidate_trial", report["actions"])
        self.assertNotIn("freeze_for_rebuild", report["actions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
