"""qf_safety — deterministic, code-enforced safety primitives for QuantForge.

Phase 0 of the staged automation program. These modules wrap the *existing* trading
system with fail-closed validation, typed parameter governance, code-enforced
action permissions, atomic/locked state writes, and tamper-evident decision
logging. They add safety; they do not change trading behaviour.

Nothing in this package may enable real-money trading, move funds, hold
credentials, or relax a deterministic risk control. See docs/AUTONOMOUS_LOOP_PROTOCOL.md.
"""

__all__ = [
    "atomic_json",
    "param_schema",
    "permissions",
    "decision_log",
    "code_mutation_guard",
    "candidate_pipeline",
]
