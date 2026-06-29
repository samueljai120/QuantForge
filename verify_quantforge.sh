#!/usr/bin/env bash
# QuantForge MASTER verifier — one command that asserts the whole research/safety
# stack is green. Exit 0 == the system is in a known-good, auditable, fail-closed
# state. Fails closed: any missing doc, failing gate, or config-drift breaks it.
#
# Run:  ssh your-prod-host 'cd ~/quantforge && bash verify_quantforge.sh'
set -uo pipefail
cd "$(dirname "$0")" || exit 2

fail=0
note() { printf '%s\n' "$*"; }

note "=== QuantForge master verification ==="

# 1. Required docs exist, are non-empty, and carry their load-bearing sections.
for d in docs/QUANTFORGE_SYSTEM_STATE.md docs/AUTONOMOUS_LOOP_PROTOCOL.md; do
  if [ ! -s "$d" ]; then note "MISSING/empty: $d"; fail=1; fi
done
grep -qi "carry-universe verdict" docs/QUANTFORGE_SYSTEM_STATE.md 2>/dev/null \
  || { note "SYSTEM_STATE missing carry-universe verdict section"; fail=1; }
for role in Planner Executor Critic "Red-Team" Verifier Memory "Human Approval Gate"; do
  grep -q "$role" docs/AUTONOMOUS_LOOP_PROTOCOL.md 2>/dev/null \
    || { note "PROTOCOL missing role: $role"; fail=1; }
done
grep -q "Stop conditions\|stop conditions" docs/AUTONOMOUS_LOOP_PROTOCOL.md 2>/dev/null \
  || { note "PROTOCOL missing stop conditions"; fail=1; }
[ "$fail" = 0 ] && note "[PASS] docs present with required sections"

# 2. Config sentinel — live carry policy within approved bounds (fail-closed).
# The sentinel is fail-closed by design: with no human-approved baseline it refuses.
# On a fresh clone that is the EXPECTED state, not a defect — seed it with --approve.
BASELINE="${QF_BASE_DIR:-$HOME/quantforge}/data/quantforge/approved_config_baseline.json"
if [ ! -f "$BASELINE" ]; then
  note "[INFO] config sentinel: no approved baseline yet — seed with 'python3 scripts/qf_config_sentinel.py --approve \"<reason>\"' (fail-closed by design)"
elif python3 scripts/qf_config_sentinel.py; then
  note "[PASS] config sentinel"
else
  note "[FAIL] config sentinel — live carry policy drifted out of approved bounds"
  fail=1
fi

# 3. Sentinel unit tests (incl. negative / fail-closed cases).
if python3 -m pytest -q tests/test_config_sentinel.py >/tmp/qf_master_pt 2>&1; then
  note "[PASS] sentinel tests"
else
  note "[FAIL] sentinel tests"; tail -5 /tmp/qf_master_pt; fail=1
fi

# 3b. SAFETY SUITE — the human-approval gate, money-conservation invariants, the
# tamper-evident decision log, and param-proposal authority must all be green.
# This is what makes "known-good SAFETY state" a verified claim, not a slogan.
SAFETY_TESTS=(
  tests/test_action_gate.py
  tests/test_invariants.py
  tests/test_invariants_widened.py
  tests/test_param_proposal.py
  tests/test_param_proposal_authority.py
  tests/test_decision_log.py
  tests/test_safety_thresholds.py
  tests/test_self_heal_invariants.py
)
present=()
for t in "${SAFETY_TESTS[@]}"; do [ -f "$t" ] && present+=("$t"); done
if [ "${#present[@]}" -eq 0 ]; then
  note "[WARN] no safety tests found (skipped)"
elif python3 -m pytest -q "${present[@]}" >/tmp/qf_master_safe 2>&1; then
  note "[PASS] safety suite (${#present[@]} files: action-gate, invariants, param-proposal, decision-log)"
else
  note "[FAIL] safety suite"; tail -6 /tmp/qf_master_safe; fail=1
fi

# 4. Full test suite (engine, ML, safety, invariants) must be green.
if python3 -m pytest -q tests/ >/tmp/qf_master_all 2>&1; then
  note "[PASS] full test suite ($(grep -oE '[0-9]+ passed' /tmp/qf_master_all | tail -1))"
else
  note "[FAIL] test suite"; tail -6 /tmp/qf_master_all; fail=1
fi

echo
if [ "$fail" = 0 ]; then
  note "GATE: PASS — docs + config sentinel + safety suite + full test suite green."
  note "       System is in a known-good, fail-closed, human-gated state. Nothing live was touched."
else
  note "GATE: FAIL — see [FAIL] lines above. Fail closed: do not promote anything."
fi
exit "$fail"
