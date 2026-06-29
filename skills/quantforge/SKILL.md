---
name: quantforge
description: Operate, monitor, and safely tune the QuantForge autonomous quant-research platform. Covers the agent's command surface, the two-layer config (code constants vs runtime params), the fail-closed safety model (permission tiers, parameter-proposal gate, candidate pipeline), and the honest-evidence standard for accepting any change.
---

# QuantForge — Agent Skill

This skill lets an LLM agent (Claude Code, the Claude Agent SDK, or any runtime that
loads `SKILL.md` Agent Skills) **run, monitor, and tune QuantForge within hard safety
guardrails**. QuantForge is an autonomous crypto quant-research platform; this skill is
how an agent drives it without being able to do anything dangerous.

The golden rule: **autonomy is earned through verified safety and honest evidence, not
switched on.** Where convenience and safety conflict, safety wins.

## Command surface

All runtime scripts are env-guarded — set `QF_ALLOW_LOCAL_RUNTIME=1` for local runs.

```bash
# Paper-trading engine
python3 scripts/quantforge_paper.py scan      # screen market, generate signals
python3 scripts/quantforge_paper.py status    # portfolio + performance
python3 scripts/quantforge_paper.py run        # full cycle: scan → signal → trade → update
python3 scripts/quantforge_paper.py backtest   # strategy gate (Sharpe, win rate)

# ML
python3 scripts/quantforge_ml.py train         # walk-forward XGBoost + LightGBM ensemble
python3 scripts/quantforge_ml.py eval          # out-of-sample, leak-free evaluation

# Governance / safety (read-only checks are always safe to run)
python3 scripts/qf_config_sentinel.py          # config-drift + approval check (hash-chain log)
python3 scripts/quantforge_governance.py       # model/strategy health snapshot
bash    verify_quantforge.sh                   # master verifier — exit 0 = stack green
```

## Two-layer configuration

1. **Code constants** — hard limits and structural defaults that live in source
   (e.g. tail-risk caps). Changing these is a *code change* (see the candidate pipeline).
2. **Runtime params** — tunables in the strategy-params JSON, governed by a registry.
   Changing these goes through the **parameter-proposal gate**, never a direct write.

Never write the strategy-params file directly. Route every change through the gate.

## Safety model — what you may do autonomously, and what escalates

QuantForge classifies every action by a risk-tiered `PermissionLevel`
(`qf_safety/permissions.py`, `qf_safety/action_gate.py`):

| Tier | Examples | Who decides |
|------|----------|-------------|
| `REVERSIBLE_OP` | restart a collector, clear a cache, restore a known-good config, roll back | **Autonomous** — just do it |
| `CONFIG_PROPOSAL` | tune a non-risk runtime param | Proposal gate (schema → fail-closed backtest → atomic apply) |
| `CODE_MODEL_CHANGE` | code/model/package change | Candidate pipeline → **human-gated deploy** |
| `FINANCIAL_SECURITY` | clear a kill switch, re-enable risk, touch real money or credentials | **Escalate to a human — always** |

Rules for the agent:
- **Operational healing** is autonomous. Keep things running without asking.
- **Param tuning** → construct a proposal and submit it to `qf_safety.param_proposal.ParameterProposalGate`. It auto-applies low-risk params and **escalates risk-exposure params and kill switches to a human.**
- **Code / model / strategy changes** → build a `Candidate` and run `qf_safety`'s
  candidate pipeline: propose → sandbox (outside the live tree) → full test suite →
  independent review → shadow → canary → APPROVED → **human deploys.** The author of a
  change may not be its own reviewer/approver/deployer. Deploy is blocked unless code
  mutation is explicitly enabled by a human.
- **Risk limits, kill switches, real money, credentials, disabling safety** → **never
  autonomous.** Lead with a recommendation and ask for a single human yes/no. This is the
  one place "ask first" is mandatory — it is what makes the autonomy trustworthy.

Hard invariants you must never undo:
- **Fail-closed validation** — gates REJECT on any error, timeout, or malformed output.
  Never restore a `except: approved = True` pattern.
- **No arbitrary execution** — autonomous `shell=True`, package installs, and arbitrary
  file writes are blocked; they route through the candidate pipeline.
- **Tamper-evident log** — every gated decision is appended to a SHA-256 hash-chained
  decision log (`qf_safety/decision_log.py`). Do not bypass it.

## Honest-evidence standard

Judge every ML/strategy change by **cost-adjusted, leakage-free, out-of-sample** value —
not by in-sample fit or confidence. Use `qf_mlops` (`baselines`, `ml_edge`, `cost_floor`,
`benchmark_gate`) to measure it. A signal must clear the cost floor and the benchmark gate
(out-of-sample AUC, calibration, net-of-cost Sharpe, edge-vs-cost margin, stability across
windows, anti-leakage) before promotion. If the evidence says no edge, **report that** —
a rejected idea honestly evaluated is a success, not a failure.

## Operating style

- For operational work: act, don't ask. Keep the system healthy.
- For risk actions, deploys, and the hard gates above: recommend, then ask for one yes/no.
- Verify before trust: run and observe a change before committing it — deploy-time
  verification has caught real bugs that passed tests.
