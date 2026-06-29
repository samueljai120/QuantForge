# QuantForge

**Stack:** Python 3.10+ | numpy / pandas / scikit-learn / XGBoost / LightGBM | KuCoin public API

## What

Autonomous crypto quant-research harness. Built to find a tradeable edge in crypto markets using ML and carry strategies. Rigorously honest: the system itself verified no durable ML edge exists (OOS AUC ~0.50). The only validated edge is pooled funding-rate carry, which is small at retail scale. See `docs/QUANTFORGE_VERDICT.md` for the evidence-based verdict.

## Quick Start

```bash
./setup.sh                          # Create venv, install deps, copy .env
source .venv/bin/activate
python3 -m pytest tests/ -q         # Run all tests
bash verify_quantforge.sh           # Assert full stack is green (requires prod host)
```

## Commands

```bash
# Development
source .venv/bin/activate
pip install -r requirements.txt

# Tests
python3 -m pytest tests/ -q                          # All tests
python3 -m pytest tests/test_config_sentinel.py -v   # Sentinel unit tests
python3 -m pytest tests/test_quantforge_ml_gate_truth.py -v  # ML gate honesty

# Paper trading engine
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_paper.py scan     # Scan market
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_paper.py status   # Portfolio state
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_paper.py run      # Full cycle

# Agent
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_agent.py status
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_agent.py strategies

# ML training
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_ml.py train
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_ml.py eval

# Config sentinel (read-only check)
python3 scripts/qf_config_sentinel.py
python3 scripts/qf_config_sentinel.py --json

# Governance snapshot
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_governance.py

# Master verifier (runs on production host)
bash verify_quantforge.sh
```

## Architecture

```
scripts/
  quantforge_paper.py         Paper-trading engine, portfolio ledger
  quantforge_agent.py         Adaptive allocator with pluggable strategy registry
  quantforge_ml.py            XGBoost + LightGBM ensemble, walk-forward CV
  quantforge_features.py      Feature engineering pipeline
  quantforge_regime.py        Market regime detector
  quantforge_reflect.py       Self-reflection daemon, proposes param changes
  qf_mlops/carry_backtest.py  Funding-rate carry engine + backtest (validated edge)
  qf_config_sentinel.py       Fail-closed config sentinel, hash-chain audit log
  quantforge_governance.py    Governance snapshot for strategy health
  quantforge_invariants.py    Money-conservation invariant checker
  quantforge_self_heal.py     Self-healing loop for invariant violations
  qf_safety/                  Fail-closed safety core (atomic CAS store, decision log,
                              action gate, invariants, candidate pipeline)
  qf_mlops/                   MLOps sub-package (model registry, carry backtest)
  config.py                   Shared config, path resolution, runtime guard

tests/
  32 pytest files — each covering a specific gate, invariant, or failure mode

docs/
  QUANTFORGE_VERDICT.md         Evidence-based verdict for each strategy
  QUANTFORGE_SYSTEM_STATE.md    Live system state: validated / rejected / next steps
  AUTONOMOUS_LOOP_PROTOCOL.md   How the autonomous loop operates safely
  quantforge_roadmap.md         Stage-gated build plan with explicit pass/fail criteria
```

Data flow: KuCoin public API -> feature engineering -> ML training (leak-free TimeSeriesSplit) -> validation gate (6 criteria) -> carry harvester (funding threshold check) -> paper portfolio ledger -> governance/audit reports.

All live-impacting actions are `HUMAN_GATED`: they produce a proposal artifact and stop — never auto-execute.

## Key Files

```
scripts/config.py                   Shared config singleton; env-var driven; runtime guard
scripts/quantforge_paper.py         Core paper-trading engine entry point
scripts/quantforge_agent.py         BTC allocator with strategy registry pattern
scripts/quantforge_ml.py            ML training with walk-forward CV and AUC gate
scripts/qf_config_sentinel.py       Config drift detection + hash-chain audit log
scripts/qf_mlops/carry_backtest.py  The only validated strategy (funding carry)
tests/test_config_sentinel.py       15 sentinel tests including fail-closed cases
tests/test_quantforge_ml_gate_truth.py  Proves look-ahead leakage is caught
docs/QUANTFORGE_VERDICT.md          Honest strategy verdicts with evidence
docs/QUANTFORGE_SYSTEM_STATE.md     Current validated/rejected/next state
verify_quantforge.sh                Master verifier: exit 0 = stack is green
```

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env`.

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | For LLM reflection | Routes to any model |
| `ANTHROPIC_API_KEY` | For research agent | Direct Anthropic access |
| `QF_BASE_DIR` | No | Data/logs root (default: `~/quantforge`) |
| `QF_PRODUCTION_HOST` | For prod deploy | Hostname that unlocks production scripts |
| `QF_ALLOW_LOCAL_RUNTIME` | Dev only | Set `1` to run production scripts locally |
| `QF_HORIZON` | No | ML forward-return horizon (`1h`/`4h`/`24h`, default `4h`) |
| `QF_SYMBOL_ALLOWLIST` | No | Comma-separated symbol filter |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | Optional | History sync |

## Safety rules (non-negotiable)

1. No live trading. No real-money execution.
2. No live parameter mutation without explicit human approval.
3. Fail closed on uncertainty — every gate rejects rather than assumes OK.
4. `HUMAN_GATED` markers on every live-impacting action.
5. Master verifier `verify_quantforge.sh` must exit 0 before promoting any change.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
