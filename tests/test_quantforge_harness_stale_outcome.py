#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import quantforge_candidate_review as qrev
import quantforge_candidate_recovery as qrec
import quantforge_experiment_lanes as qel
import quantforge_harness_report as qhr


class QuantforgeHarnessStaleOutcomeTests(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_harness_ignores_outcome_older_than_current_candidate_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autopilot = os.path.join(tmpdir, "autopilot.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            review = os.path.join(tmpdir, "review.json")

            self._write_json(autopilot, {"mode": "run_candidate_paper_trial"})
            self._write_json(recovery, {"candidate_id": "capital_preservation:20260626T113813Z", "type": "capital_preservation"})
            self._write_json(
                lanes,
                {
                    "candidate": {"model_trained_at": "2026-06-22T02:12:37.166088+00:00"},
                    "candidate_trial": {
                        "candidate_id": "capital_preservation:20260626T113813Z",
                        "status": "active",
                        "paper_only": True,
                        "queued_at": "2026-06-26T11:40:34.242545+00:00",
                    },
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-04-20T15:40:34.431098+00:00",
                        "assessment": "fail",
                        "next_candidate_hint": "quantforge_research_hold",
                    }
                },
            )
            self._write_json(review, {"recommendation": "hold_active_trial"})

            old = (
                qhr.AUTOPILOT_FILE,
                qhr.RECOVERY_FILE,
                qhr.LANES_FILE,
                qhr.OUTCOMES_FILE,
                qhr.REVIEW_FILE,
            )
            try:
                qhr.AUTOPILOT_FILE = autopilot
                qhr.RECOVERY_FILE = recovery
                qhr.LANES_FILE = lanes
                qhr.OUTCOMES_FILE = outcomes
                qhr.REVIEW_FILE = review
                report = qhr.build_report()
            finally:
                (
                    qhr.AUTOPILOT_FILE,
                    qhr.RECOVERY_FILE,
                    qhr.LANES_FILE,
                    qhr.OUTCOMES_FILE,
                    qhr.REVIEW_FILE,
                ) = old

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failed_count"], 0)

    def test_harness_allows_fresh_requeue_after_blocked_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autopilot = os.path.join(tmpdir, "autopilot.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            review = os.path.join(tmpdir, "review.json")

            self._write_json(autopilot, {"mode": "hold_in_paper"})
            self._write_json(recovery, {"candidate_id": "capital_preservation:20260626T113813Z", "type": "capital_preservation"})
            self._write_json(
                lanes,
                {
                    "candidate": {"model_trained_at": "2026-06-22T02:12:37.166088+00:00"},
                    "candidate_trial": {
                        "candidate_id": "capital_preservation:20260626T113813Z",
                        "status": "queued",
                        "paper_only": True,
                        "queued_at": "2026-06-26T11:40:34.242545+00:00",
                    },
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-06-26T10:40:34.431098+00:00",
                        "assessment": "blocked",
                        "next_candidate_hint": "capital_preservation",
                    }
                },
            )
            self._write_json(review, {"recommendation": "queue_candidate_trial"})

            old = (
                qhr.AUTOPILOT_FILE,
                qhr.RECOVERY_FILE,
                qhr.LANES_FILE,
                qhr.OUTCOMES_FILE,
                qhr.REVIEW_FILE,
            )
            try:
                qhr.AUTOPILOT_FILE = autopilot
                qhr.RECOVERY_FILE = recovery
                qhr.LANES_FILE = lanes
                qhr.OUTCOMES_FILE = outcomes
                qhr.REVIEW_FILE = review
                report = qhr.build_report()
            finally:
                (
                    qhr.AUTOPILOT_FILE,
                    qhr.RECOVERY_FILE,
                    qhr.LANES_FILE,
                    qhr.OUTCOMES_FILE,
                    qhr.REVIEW_FILE,
                ) = old

        blocked_retry = next(row for row in report["checks"] if row["name"] == "blocked_trial_not_left_queued")
        self.assertTrue(blocked_retry["passed"])

    def test_candidate_review_ignores_stale_latest_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autopilot = os.path.join(tmpdir, "autopilot.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            last_scan = os.path.join(tmpdir, "last-scan.json")

            self._write_json(autopilot, {"mode": "run_candidate_paper_trial"})
            self._write_json(
                lanes,
                {
                    "candidate": {"model_trained_at": "2026-06-22T02:12:37.166088+00:00"},
                    "baseline": {"paper_total_pnl_pct": 0.0},
                    "candidate_trial": {
                        "candidate_id": "capital_preservation:20260626T113813Z",
                        "status": "active",
                        "cycles_run": 1,
                        "max_cycles": 6,
                    },
                },
            )
            self._write_json(recovery, {"candidate_id": "capital_preservation:20260626T113813Z", "type": "capital_preservation", "status": "proposed"})
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-04-20T15:40:34.431098+00:00",
                        "candidate_id": "competitiveness_gap_rebuild:20260420T054042Z",
                        "type": "competitiveness_gap_rebuild",
                        "assessment": "fail",
                        "next_candidate_hint": "quantforge_research_hold",
                    }
                },
            )
            self._write_json(last_scan, {"ts": "2026-06-27T11:40:34.431098+00:00", "flow": {}, "results": []})

            old = (
                qrev.AUTOPILOT_FILE,
                qrev.LANES_FILE,
                qrev.CANDIDATE_RECOVERY_FILE,
                qrev.CANDIDATE_OUTCOMES_FILE,
                qrev.LAST_SCAN_FILE,
            )
            try:
                qrev.AUTOPILOT_FILE = autopilot
                qrev.LANES_FILE = lanes
                qrev.CANDIDATE_RECOVERY_FILE = recovery
                qrev.CANDIDATE_OUTCOMES_FILE = outcomes
                qrev.LAST_SCAN_FILE = last_scan
                payload = qrev.build_review()
            finally:
                (
                    qrev.AUTOPILOT_FILE,
                    qrev.LANES_FILE,
                    qrev.CANDIDATE_RECOVERY_FILE,
                    qrev.CANDIDATE_OUTCOMES_FILE,
                    qrev.LAST_SCAN_FILE,
                ) = old

        self.assertEqual(payload["recommendation"], "hold_active_trial")
        self.assertIsNone(payload["latest_outcome"]["next_candidate_hint"])

    def test_candidate_review_surfaces_missing_target_long_surface_for_active_expansion_trial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autopilot = os.path.join(tmpdir, "autopilot.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            last_scan = os.path.join(tmpdir, "last-scan.json")

            self._write_json(autopilot, {"mode": "run_candidate_paper_trial"})
            self._write_json(
                lanes,
                {
                    "candidate": {"model_trained_at": "2026-06-28T01:00:00+00:00", "paper_total_pnl_pct": -1.08},
                    "baseline": {"paper_total_pnl_pct": 0.0},
                    "candidate_trial": {
                        "candidate_id": "major_liquidity_expansion:20260628T001543Z",
                        "type": "major_liquidity_expansion",
                        "status": "active",
                        "cycles_run": 3,
                        "max_cycles": 6,
                        "paper_only": True,
                    },
                },
            )
            self._write_json(recovery, {"candidate_id": "major_liquidity_expansion:20260628T001543Z", "type": "major_liquidity_expansion", "status": "proposed"})
            self._write_json(outcomes, {"latest": {}})
            self._write_json(
                last_scan,
                {
                    "ts": "2026-06-28T04:45:03.460220+00:00",
                    "flow": {
                        "buy_signals": 0,
                        "sell_signals": 3,
                        "threshold_miss": 28,
                        "selection_blocked": 1,
                    },
                    "results": [
                        {
                            "symbol": "EIGEN-USDT",
                            "status": "skip",
                            "reason": "candidate expansion limits longs to majors and top-liquidity alts",
                            "setup_tag": "trend_long",
                        },
                        {
                            "symbol": "ICNT-USDT",
                            "status": "hold",
                            "reason": "LONG conf 0.004 < 0.740, SHORT conf 0.001 < 0.840",
                            "long_confidence": 0.0042,
                            "short_confidence": 0.0011,
                            "setup_tag": "trend_long",
                        },
                    ],
                },
            )

            old = (
                qrev.AUTOPILOT_FILE,
                qrev.LANES_FILE,
                qrev.CANDIDATE_RECOVERY_FILE,
                qrev.CANDIDATE_OUTCOMES_FILE,
                qrev.LAST_SCAN_FILE,
            )
            try:
                qrev.AUTOPILOT_FILE = autopilot
                qrev.LANES_FILE = lanes
                qrev.CANDIDATE_RECOVERY_FILE = recovery
                qrev.CANDIDATE_OUTCOMES_FILE = outcomes
                qrev.LAST_SCAN_FILE = last_scan
                payload = qrev.build_review()
            finally:
                (
                    qrev.AUTOPILOT_FILE,
                    qrev.LANES_FILE,
                    qrev.CANDIDATE_RECOVERY_FILE,
                    qrev.CANDIDATE_OUTCOMES_FILE,
                    qrev.LAST_SCAN_FILE,
                ) = old

        self.assertEqual(payload["recommendation"], "hold_active_trial")
        self.assertTrue(payload["active_trial_surface"]["no_target_long_surface"])
        self.assertIn("0 buy signals", payload["reasons"][-1])

    def test_candidate_review_still_flags_dead_expansion_surface_without_sell_signals(self):
        summary = qrev._active_trial_surface_summary(
            {
                "ts": "2026-06-28T05:12:39.322496+00:00",
                "flow": {
                    "buy_signals": 0,
                    "sell_signals": 0,
                    "threshold_miss": 29,
                    "selection_blocked": 2,
                },
                "results": [
                    {
                        "symbol": "RAVE-USDT",
                        "status": "skip",
                        "reason": "candidate expansion limits longs to majors and top-liquidity alts",
                        "setup_tag": "trend_long",
                    },
                    {
                        "symbol": "ICNT-USDT",
                        "status": "hold",
                        "reason": "LONG conf 0.008 < 0.740, SHORT conf 0.001 < 0.840",
                        "long_confidence": 0.0078,
                        "short_confidence": 0.0009,
                        "setup_tag": "trend_long",
                    },
                ],
            },
            {
                "candidate_id": "major_liquidity_expansion:20260628T001543Z",
                "type": "major_liquidity_expansion",
                "status": "active",
                "cycles_run": 4,
                "max_cycles": 6,
                "paper_only": True,
            },
        )

        self.assertIsNotNone(summary)
        self.assertTrue(summary["no_target_long_surface"])

    def test_candidate_review_surfaces_blocked_setup_quality_recovery_scope(self):
        summary = qrev._active_trial_surface_summary(
            {
                "ts": "2026-06-28T05:20:20.974195+00:00",
                "flow": {
                    "buy_signals": 0,
                    "sell_signals": 4,
                    "threshold_miss": 25,
                    "selection_blocked": 3,
                },
                "results": [
                    {
                        "symbol": "ICNT-USDT",
                        "status": "skip",
                        "reason": "candidate trial restricts longs to major-liquidity symbols",
                        "setup_tag": "trend_long",
                    },
                    {
                        "symbol": "RAVE-USDT",
                        "status": "skip",
                        "reason": "candidate trial restricts longs to major-liquidity symbols",
                        "setup_tag": "trend_long",
                    },
                ],
            },
            {
                "candidate_id": "setup_quality_recovery:20260628T051658Z",
                "type": "setup_quality_recovery",
                "status": "active",
                "cycles_run": 1,
                "max_cycles": 6,
                "paper_only": True,
            },
        )

        self.assertIsNotNone(summary)
        self.assertTrue(summary["setup_quality_scope_blocked"])
        self.assertEqual(summary["blocked_labeled_longs"][0]["symbol"], "ICNT-USDT")

    def test_candidate_recovery_builds_real_setup_quality_lane_after_expansion_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            agi_history = os.path.join(tmpdir, "agi.json")
            monitor = os.path.join(tmpdir, "monitor.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            model_layer = os.path.join(tmpdir, "model-layer.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            last_scan = os.path.join(tmpdir, "last-scan.json")

            self._write_json(diagnosis, {"causes": ["paper_underperformance", "weak_recent_close_quality"]})
            self._write_json(governance, {"recent_closes": {"win_rate": 0.2}, "paper": {"total_pnl_pct": -1.09}})
            self._write_json(promotion, {"overall_decision": "DO_NOT_PROMOTE"})
            self._write_json(agi_history, {"posture": "degraded", "persistent_review": True, "persistent_drift": True, "averages": {"adaptive_risk_mult": 0.45}})
            self._write_json(monitor, {"drift_flags": [], "regime": {"label": "NEUTRAL", "entropy_label": "MIXED", "entropy": 0.4}})
            self._write_json(
                lanes,
                {
                    "baseline": {"paper_total_pnl_pct": 0.0},
                    "candidate": {"paper_total_pnl_pct": -1.09, "model_trained_at": "2026-06-28T01:00:00+00:00"},
                    "candidate_trial": {
                        "candidate_id": "setup_quality_recovery:20260628T051658Z",
                        "type": "setup_quality_recovery",
                        "status": "active",
                        "paper_only": True,
                    },
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-06-28T05:16:58+00:00",
                        "type": "major_liquidity_expansion",
                        "assessment": "fail",
                        "next_candidate_hint": "setup_quality_recovery",
                    }
                },
            )
            self._write_json(model_layer, {})
            self._write_json(
                last_scan,
                {
                    "results": [
                        {
                            "symbol": "ICNT-USDT",
                            "status": "skip",
                            "reason": "candidate trial restricts longs to major-liquidity symbols",
                            "setup_tag": "trend_long",
                        },
                        {
                            "symbol": "RAVE-USDT",
                            "status": "skip",
                            "reason": "candidate trial restricts longs to major-liquidity symbols",
                            "setup_tag": "trend_long",
                        },
                    ]
                },
            )
            self._write_json(recovery, {})

            old = (
                qrec.DIAGNOSIS_FILE,
                qrec.GOVERNANCE_FILE,
                qrec.PROMOTION_FILE,
                qrec.AGI_HISTORY_FILE,
                qrec.MONITOR_FILE,
                qrec.LANES_FILE,
                qrec.CANDIDATE_OUTCOMES_FILE,
                qrec.MODEL_LAYER_FILE,
                qrec.OUTPUT_FILE,
                qrec.LAST_SCAN_FILE,
            )
            try:
                qrec.DIAGNOSIS_FILE = diagnosis
                qrec.GOVERNANCE_FILE = governance
                qrec.PROMOTION_FILE = promotion
                qrec.AGI_HISTORY_FILE = agi_history
                qrec.MONITOR_FILE = monitor
                qrec.LANES_FILE = lanes
                qrec.CANDIDATE_OUTCOMES_FILE = outcomes
                qrec.MODEL_LAYER_FILE = model_layer
                qrec.OUTPUT_FILE = recovery
                qrec.LAST_SCAN_FILE = last_scan
                payload = qrec.build_candidate()
            finally:
                (
                    qrec.DIAGNOSIS_FILE,
                    qrec.GOVERNANCE_FILE,
                    qrec.PROMOTION_FILE,
                    qrec.AGI_HISTORY_FILE,
                    qrec.MONITOR_FILE,
                    qrec.LANES_FILE,
                    qrec.CANDIDATE_OUTCOMES_FILE,
                    qrec.MODEL_LAYER_FILE,
                    qrec.OUTPUT_FILE,
                    qrec.LAST_SCAN_FILE,
                ) = old

        self.assertEqual(payload["type"], "setup_quality_recovery")
        changes = {row["key"]: row for row in payload["changes"]}
        self.assertIn("allowed_long_symbols", changes)
        self.assertEqual(changes["allowed_long_symbols"]["value"][:2], ["ICNT-USDT", "RAVE-USDT"])
        self.assertEqual(changes["allowed_long_setups"]["value"], ["trend_long", "breakout_long"])
        self.assertEqual(changes["entry_selection"]["value"], "require_regime_support_and_labeled_setup_alignment")
        self.assertNotIn("non_major_entries", changes)
        self.assertIn("ICNT-USDT", " ".join(payload["rationale"]))

    def test_candidate_recovery_preserves_active_trial_hint_even_when_candidate_snapshot_is_newer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnosis = os.path.join(tmpdir, "diagnosis.json")
            governance = os.path.join(tmpdir, "governance.json")
            promotion = os.path.join(tmpdir, "promotion.json")
            agi_history = os.path.join(tmpdir, "agi.json")
            monitor = os.path.join(tmpdir, "monitor.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            model_layer = os.path.join(tmpdir, "model-layer.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            last_scan = os.path.join(tmpdir, "last-scan.json")

            self._write_json(diagnosis, {"causes": ["paper_underperformance"]})
            self._write_json(governance, {"recent_closes": {"win_rate": 0.2}, "paper": {"total_pnl_pct": -0.7}})
            self._write_json(promotion, {"overall_decision": "DO_NOT_PROMOTE"})
            self._write_json(agi_history, {"posture": "degraded", "persistent_review": True, "persistent_drift": True, "averages": {"adaptive_risk_mult": 0.45}})
            self._write_json(monitor, {"drift_flags": [], "regime": {"label": "NEUTRAL", "entropy_label": "ORDERLY", "entropy": 0.3}})
            self._write_json(
                lanes,
                {
                    "baseline": {"paper_total_pnl_pct": 0.0},
                    "candidate": {"paper_total_pnl_pct": 0.2, "model_trained_at": "2026-06-28T07:05:54+00:00"},
                    "candidate_trial": {
                        "candidate_id": "setup_quality_recovery:20260628T051658Z",
                        "type": "setup_quality_recovery",
                        "status": "active",
                        "paper_only": True,
                    },
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-06-28T05:16:58+00:00",
                        "type": "major_liquidity_expansion",
                        "assessment": "fail",
                        "next_candidate_hint": "setup_quality_recovery",
                    }
                },
            )
            self._write_json(model_layer, {})
            self._write_json(last_scan, {"results": []})
            self._write_json(recovery, {})

            old = (
                qrec.DIAGNOSIS_FILE,
                qrec.GOVERNANCE_FILE,
                qrec.PROMOTION_FILE,
                qrec.AGI_HISTORY_FILE,
                qrec.MONITOR_FILE,
                qrec.LANES_FILE,
                qrec.CANDIDATE_OUTCOMES_FILE,
                qrec.MODEL_LAYER_FILE,
                qrec.OUTPUT_FILE,
                qrec.LAST_SCAN_FILE,
            )
            try:
                qrec.DIAGNOSIS_FILE = diagnosis
                qrec.GOVERNANCE_FILE = governance
                qrec.PROMOTION_FILE = promotion
                qrec.AGI_HISTORY_FILE = agi_history
                qrec.MONITOR_FILE = monitor
                qrec.LANES_FILE = lanes
                qrec.CANDIDATE_OUTCOMES_FILE = outcomes
                qrec.MODEL_LAYER_FILE = model_layer
                qrec.OUTPUT_FILE = recovery
                qrec.LAST_SCAN_FILE = last_scan
                payload = qrec.build_candidate()
            finally:
                (
                    qrec.DIAGNOSIS_FILE,
                    qrec.GOVERNANCE_FILE,
                    qrec.PROMOTION_FILE,
                    qrec.AGI_HISTORY_FILE,
                    qrec.MONITOR_FILE,
                    qrec.LANES_FILE,
                    qrec.CANDIDATE_OUTCOMES_FILE,
                    qrec.MODEL_LAYER_FILE,
                    qrec.OUTPUT_FILE,
                    qrec.LAST_SCAN_FILE,
                ) = old

        self.assertEqual(payload["type"], "setup_quality_recovery")
        self.assertFalse(payload["evidence"]["stale_latest_outcome_ignored"])

    def test_experiment_lanes_forces_fail_when_expansion_never_surfaces_target_long_edge(self):
        outcome = qel._score_trial_outcome(
            {
                "candidate_id": "major_liquidity_expansion:20260628T001543Z",
                "type": "major_liquidity_expansion",
                "status": "completed",
                "assessment": "fail",
                "completion_reason": "no_target_long_surface",
                "assessment_reason": "Active major-liquidity expansion never surfaced its target long edge for 2 cycles.",
                "next_candidate_hint": "setup_quality_recovery",
                "cycles_run": 2,
                "max_cycles": 6,
                "paper_only": True,
                "completion_summary": {
                    "buy_signals": 0,
                    "sell_signals": 3,
                    "strongest_long_hold": {
                        "symbol": "ICNT-USDT",
                        "long_confidence": 0.0042,
                    },
                },
                "changes": [],
                "baseline_snapshot": {},
            },
            {"paper_total_pnl_pct": 0.0, "governance_recommendation": "OBSERVE", "promotion_decision": "DO_NOT_PROMOTE"},
            {"paper_total_pnl_pct": 1.2, "governance_recommendation": "OBSERVE", "promotion_decision": "HOLD"},
        )

        self.assertEqual(outcome["assessment"], "fail")
        self.assertEqual(outcome["next_candidate_hint"], "setup_quality_recovery")
        self.assertIn("never surfaced its target long edge", outcome["reasons"][0])
        self.assertIn("ICNT-USDT", outcome["reasons"][1])

    def test_harness_accepts_rotate_candidate_class_for_completed_trial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autopilot = os.path.join(tmpdir, "autopilot.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            review = os.path.join(tmpdir, "review.json")

            self._write_json(autopilot, {"mode": "hold_in_paper"})
            self._write_json(recovery, {"candidate_id": "setup_quality_recovery:20260628T061500Z", "type": "setup_quality_recovery", "status": "proposed"})
            self._write_json(
                lanes,
                {
                    "candidate": {"model_trained_at": "2026-06-28T06:00:00+00:00"},
                    "candidate_trial": {
                        "candidate_id": "major_liquidity_expansion:20260628T001543Z",
                        "status": "completed",
                        "type": "major_liquidity_expansion",
                        "assessment": "fail",
                    },
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-06-28T06:10:00+00:00",
                        "assessment": "fail",
                        "next_candidate_hint": "setup_quality_recovery",
                    }
                },
            )
            self._write_json(review, {"recommendation": "rotate_candidate_class"})

            old = (
                qhr.AUTOPILOT_FILE,
                qhr.RECOVERY_FILE,
                qhr.LANES_FILE,
                qhr.OUTCOMES_FILE,
                qhr.REVIEW_FILE,
            )
            try:
                qhr.AUTOPILOT_FILE = autopilot
                qhr.RECOVERY_FILE = recovery
                qhr.LANES_FILE = lanes
                qhr.OUTCOMES_FILE = outcomes
                qhr.REVIEW_FILE = review
                report = qhr.build_report()
            finally:
                (
                    qhr.AUTOPILOT_FILE,
                    qhr.RECOVERY_FILE,
                    qhr.LANES_FILE,
                    qhr.OUTCOMES_FILE,
                    qhr.REVIEW_FILE,
                ) = old

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failed_count"], 0)

    def test_harness_allows_competitiveness_gap_escalation_after_deeper_fail_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autopilot = os.path.join(tmpdir, "autopilot.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            review = os.path.join(tmpdir, "review.json")

            self._write_json(autopilot, {"mode": "run_candidate_paper_trial"})
            self._write_json(recovery, {"candidate_id": "competitiveness_gap_rebuild:20260626T134034Z", "type": "competitiveness_gap_rebuild"})
            self._write_json(
                lanes,
                {
                    "candidate": {"model_trained_at": "2026-06-22T02:12:37.166088+00:00"},
                    "candidate_trial": {
                        "candidate_id": "competitiveness_gap_rebuild:20260626T134034Z",
                        "status": "active",
                        "paper_only": True,
                        "queued_at": "2026-06-26T13:40:34.242545+00:00",
                    },
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-06-26T13:40:34.431098+00:00",
                        "assessment": "fail",
                        "next_candidate_hint": "setup_quality_recovery",
                    },
                    "history": [
                        {"recorded_at": "2026-04-19T13:44:24.920299+00:00", "type": "model_recalibration", "assessment": "fail"},
                        {"recorded_at": "2026-04-20T05:40:42.617191+00:00", "type": "quantforge_redesign", "assessment": "fail"},
                    ],
                },
            )
            self._write_json(review, {"recommendation": "hold_active_trial"})

            old = (
                qhr.AUTOPILOT_FILE,
                qhr.RECOVERY_FILE,
                qhr.LANES_FILE,
                qhr.OUTCOMES_FILE,
                qhr.REVIEW_FILE,
            )
            try:
                qhr.AUTOPILOT_FILE = autopilot
                qhr.RECOVERY_FILE = recovery
                qhr.LANES_FILE = lanes
                qhr.OUTCOMES_FILE = outcomes
                qhr.REVIEW_FILE = review
                report = qhr.build_report()
            finally:
                (
                    qhr.AUTOPILOT_FILE,
                    qhr.RECOVERY_FILE,
                    qhr.LANES_FILE,
                    qhr.OUTCOMES_FILE,
                    qhr.REVIEW_FILE,
                ) = old

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failed_count"], 0)

    def test_harness_allows_major_expansion_rotation_from_research_hold_hint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autopilot = os.path.join(tmpdir, "autopilot.json")
            recovery = os.path.join(tmpdir, "recovery.json")
            lanes = os.path.join(tmpdir, "lanes.json")
            outcomes = os.path.join(tmpdir, "outcomes.json")
            review = os.path.join(tmpdir, "review.json")

            self._write_json(autopilot, {"mode": "hold_in_paper"})
            self._write_json(recovery, {"candidate_id": "major_liquidity_expansion:20260627T134034Z", "type": "major_liquidity_expansion", "status": "proposed"})
            self._write_json(
                lanes,
                {
                    "candidate": {"model_trained_at": "2026-06-27T12:12:37.166088+00:00"},
                    "candidate_trial": None,
                },
            )
            self._write_json(
                outcomes,
                {
                    "latest": {
                        "recorded_at": "2026-06-27T13:40:34.431098+00:00",
                        "assessment": "fail",
                        "next_candidate_hint": "quantforge_research_hold",
                    }
                },
            )
            self._write_json(review, {"recommendation": "queue_candidate_trial"})

            old = (
                qhr.AUTOPILOT_FILE,
                qhr.RECOVERY_FILE,
                qhr.LANES_FILE,
                qhr.OUTCOMES_FILE,
                qhr.REVIEW_FILE,
            )
            try:
                qhr.AUTOPILOT_FILE = autopilot
                qhr.RECOVERY_FILE = recovery
                qhr.LANES_FILE = lanes
                qhr.OUTCOMES_FILE = outcomes
                qhr.REVIEW_FILE = review
                report = qhr.build_report()
            finally:
                (
                    qhr.AUTOPILOT_FILE,
                    qhr.RECOVERY_FILE,
                    qhr.LANES_FILE,
                    qhr.OUTCOMES_FILE,
                    qhr.REVIEW_FILE,
                ) = old

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failed_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
