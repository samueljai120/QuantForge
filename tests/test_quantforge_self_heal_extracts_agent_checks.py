#!/usr/bin/env python3
"""Ensure self-heal actions ingest doctor agent-lane failures."""

import unittest

import quantforge_self_heal_actions as qsha


class QuantforgeSelfHealExtractTests(unittest.TestCase):
    def test_extract_flags_includes_agent_checks(self):
        diagnostics = {
            "doctor-report.json": {
                "checks": [],
                "agent_checks": [
                    {"name": "invariants_state", "ok": False, "detail": "1 critical / 0 warning (futures_open_close_parity)"},
                ],
            }
        }

        flags = qsha.extract_flags(diagnostics)

        self.assertTrue(flags)
        self.assertIn("invariants_state", flags[0].description)
        self.assertEqual(flags[0].priority, "critical")

    def test_extract_flags_keeps_manual_engineering_action_out_of_llm_path(self):
        diagnostics = {
            "engineering-actions.json": {
                "actions": [
                    {
                        "priority": "high",
                        "type": "prepare_major_liquidity_expansion_candidate",
                        "why": "Top-alt support is stronger than the current major-only survivors.",
                        "execution_policy": "manual_only",
                        "llm_eligible": False,
                    }
                ]
            }
        }

        flags = qsha.extract_flags(diagnostics)

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].flag_type, "manual_only")
        self.assertFalse(flags[0].llm_eligible)


if __name__ == "__main__":
    unittest.main(verbosity=2)
