# NullEdge

![CI](https://github.com/samueljai120/NullEdge/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Tests](https://img.shields.io/badge/tests-148%20passing-brightgreen.svg)

An autonomous crypto quant-research harness with adversarial, fail-closed honest evaluation — built to find a tradeable edge, and rigorous enough to prove it found none.

> **NullEdge** is the public release of **QuantForge**, the engine and Python package it ships (modules are named `quantforge_*`). The name says the result: the system searched hard for a trading edge and honestly reported a *null* one — and the rigor behind that conclusion is the point.


---

## Does it make money? Honest results first.

No — not currently, and the system has been built to say that clearly.

QuantForge was built to discover a durable, cost-honest crypto trading edge. Across extensive paper-trading and self-directed research, it found:

| Strategy tested | Verdict |
|---|---|
| Directional ML (RSI, momentum, vol, range, order-flow features) | **No edge** — out-of-sample AUC ≈ 0.50 (coin-flip) |
| Cross-venue arbitrage | **No edge** — sub-cost at retail execution |
| Moonshot / high-conviction ML slice | **Negative EV** in backtest |
| Always-on cross-sectional carry factor | **Negative EV** — edge real but ~10–20× too small to clear turnover cost |
| Pooled funding-rate carry | **Validated** — the only real edge found: ~3%/yr, Sharpe ~3.2, ~1.5% maxDD unlevered |

**The validated edge is small.** Pooled carry is market-neutral and genuinely positive after cost — but at retail capital scale (~$5k) it yields roughly $70–$150/yr unlevered. Economically meaningful only at significant capital or with maker-rebate fee tiers.

**This is the point of the project.** A research harness honest enough to reject its own strategies, prove look-ahead cleanliness, and write an evidence-based "no edge" verdict is more valuable than a backtester that overfits to a Sharpe of 4.

The project's own internal verdict memo (`docs/QUANTFORGE_VERDICT.md`) concludes: *"Stop hunting a directional/predictive edge on free data. The evidence is conclusive."*

> **Not financial advice.** This is a paper-trading and research system. It has never been run with real money. Do not deploy it with real funds without extensive independent validation.

---

## Why this is interesting

The engineering story is rigorous, self-skeptical ML infrastructure:

- **Fail-closed safety gates throughout.** Every validator rejects on uncertainty rather than silently passing. A missing config key, a numeric out of bounds, or a missing file returns a non-zero exit code. Nothing defaults to "assume fine."
- **Tamper-evident hash-chained config sentinel** (`scripts/qf_config_sentinel.py`). The live carry policy JSON is validated against an approved baseline on every run. A drift from approved bounds is recorded to a SHA-256 hash-chain decision log. The sentinel also AST-parses the live harvester source and asserts that the hard-coded policy bounds in the Python match the approved bounds — closing the "deployed != live" gap at the code level.
- **Leak-free ML evaluation.** All model evaluation uses `sklearn.model_selection.TimeSeriesSplit` (strictly chronological folds). A dedicated test (`tests/test_quantforge_ml_gate_truth.py`) explicitly verifies that a feature constructed from future returns is caught and rejected by the gate — if that test passes, the leakage detector is working.
- **Multi-criteria validation gate** (`scripts/quantforge_benchmark_gate.py` + `scripts/qf_mlops/benchmark_gate.py`) — a candidate strategy must clear a benchmark bar before promotion: out-of-sample AUC, calibration, net Sharpe, edge-vs-cost margin, stability across multiple time windows, and the anti-leakage check. The gate fails closed if any criterion is unmet or unmeasurable.
- **Money-conservation invariants.** Equity accounting is verified to be internally consistent: every trade round-trips correctly, orphaned margin is detected, and a "compute_true_equity" canonical formula is asserted consistent across the codebase.
- **Master verifier.** `verify_quantforge.sh` asserts the entire stack in one command: required docs, config sentinel, safety test suite (action-gate / invariants / decision-log / param-proposal — 56+ tests), and all per-area sub-verifiers. Exit 0 means the system is in a known-good, auditable, fail-closed state.
- **Human-gated live actions.** Every action that would touch real trading state, live parameters, or production config is marked `HUMAN_GATED` — it generates a proposal artifact and stops, never auto-executes.
- **`qf_safety` package — the fail-closed core.** Atomic, content-addressed JSON persistence (`atomic_json.CASStore`), a tamper-evident `decision_log`, an `action_gate` that classifies every action by risk level, money-conservation `invariants`, a `candidate_pipeline` with explicit promotion stages, and `permissions`/`param_schema` guards. The trading scripts are thin orchestration over this safety substrate.
- **Scale:** ~38,000 lines across 95 Python modules (engine, ML, the `qf_safety` safety package, the `qf_mlops` sub-package, research/governance tooling) + 32 test files (148 tests, all passing). The harness includes: ML trainer, carry backtester, regime detector, feature proposer, edge-discovery cycle, self-evolution loop, audit snapshots, segmented holdout reports, and governance snapshots.

---

## Architecture overview

```
scripts/
  quantforge_paper.py        Paper-trading engine: scan market, generate signals, update portfolio ledger
  quantforge_agent.py        Adaptive BTC allocator with pluggable strategy registry + safety stack
  quantforge_ml.py           ML training: XGBoost + LightGBM ensemble, walk-forward CV
  quantforge_ml_train.py     Extended trainer with TimeSeriesSplit + target-profile slicing
  quantforge_features.py     Feature engineering (technical indicators, momentum, vol, order-flow)
  quantforge_regime.py       Market regime detector (STRONG_BULL/BULL/NEUTRAL/CHOP/BEAR/STRONG_BEAR)
  quantforge_reflect.py      Self-reflection daemon — reasons about own performance, proposes param changes
  qf_config_sentinel.py      Fail-closed config-drift gate with tamper-evident hash-chain log
  quantforge_governance.py   Evaluation + governance snapshot for model/strategy health
  quantforge_invariants.py   Money-conservation invariant checker
  quantforge_self_heal.py    Self-healing loop for known invariant violations
  qf_mlops/carry_backtest.py Validated funding-rate carry engine + backtest (the only confirmed edge)
  quantforge_benchmark_gate.py   Benchmark pass/fail gate for strategy promotion
  quantforge_research_director.py  Orchestrates multi-slice research campaigns
  qf_safety/                 Fail-closed safety core: atomic CAS store, decision log,
                             action gate, money-conservation invariants, candidate pipeline
  qf_mlops/                  MLOps sub-package: model registry, carry backtest, benchmark gate, etc.

docs/
  QUANTFORGE_SYSTEM_STATE.md     Single source of truth: what is validated, rejected, and next
  AUTONOMOUS_LOOP_PROTOCOL.md    How the autonomous research loop operates safely
  QUANTFORGE_VERDICT.md          Evidence-based final verdict on each tested strategy
  quantforge_roadmap.md          Stage-gated build plan with explicit pass/fail criteria

tests/
  32 test files — each covering a specific gate, invariant, or failure mode
```

Data flows: market data (KuCoin public API) → feature engineering → ML training (leak-free) → validation gate → carry harvester (if funding threshold met) → paper portfolio ledger → audit/governance reports → knowledge ledger (so dead ends are never retested).

All live-impacting actions (param changes, universe changes, leverage) are routed through the human-approval gate and become proposal artifacts, never automatic mutations.

---

## Quick start

```bash
git clone https://github.com/samueljai120/NullEdge.git
cd quantforge
./setup.sh
```

Then edit `.env` with your API keys, and run tests to confirm the stack is clean:

```bash
python3 -m pytest tests/ -q
```

Run a research script locally (market data scan, no trading state):

```bash
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_paper.py scan
```

Run the master verifier (requires production host setup):

```bash
bash verify_quantforge.sh
```

### Prerequisites

- Python 3.10+
- pip
- A KuCoin account (optional — public API endpoints are used for market data; no API keys required for read-only operations)
- OpenRouter API key or Anthropic API key (for LLM-assisted reflection and research)

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key settings:

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | For LLM reflection | Routes to any LLM model |
| `ANTHROPIC_API_KEY` | For research agent | Direct Anthropic access |
| `QF_BASE_DIR` | No (defaults to `~/quantforge`) | Where data, logs, and artifacts live |
| `QF_PRODUCTION_HOST` | For production deploy | Hostname that unlocks production-only scripts |
| `QF_ALLOW_LOCAL_RUNTIME` | Dev only | Set to `1` to run production scripts locally |
| `QF_HORIZON` | No (default `4h`) | Forward-return horizon for ML models |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | Optional | For history summary sync |

See `.env.example` for the full list with descriptions.

---

## Repo layout

```
nulledge/
  scripts/               67 modules — engine, ML, carry, research, governance
  scripts/qf_safety/     13 modules — fail-closed safety core (CAS store, gates, invariants)
  scripts/qf_mlops/      15 modules — MLOps (model registry, carry backtest, edge attribution)
  tests/                 32 pytest test files (148 tests)
  docs/                  System state, protocol, verdict, roadmap
  verify_quantforge.sh   Master verifier — asserts entire stack is green
  setup.sh               One-command bootstrap
  .env.example           All configuration with documentation
  CLAUDE.md              Claude Code context (commands, architecture, key files)
  CONTRIBUTING.md        Development and contribution guide
```

---

## Honest limitations

- **No live edge at current scale.** The only validated profit engine (funding carry) requires significant capital or fee-tier advantages to generate meaningful income. At $5k it yields ~$70–$150/yr.
- **Free data only.** The ML signal bottleneck is feature quality, not model architecture. Paid alternative data is the most plausible path to a directional edge.
- **Production runtime guard.** Most cycle scripts check `QF_PRODUCTION_HOST` and will abort locally unless `QF_ALLOW_LOCAL_RUNTIME=1` is set. This is intentional — it prevents accidentally running production loops on a dev machine.
- **Paper only.** The system has never executed a real trade. KuCoin integration exists for market data reads only. Live execution would require exchange API keys and a separate, carefully reviewed execution layer.
- **Not an AGI.** The "autonomous loop" is a bounded research loop with explicit stop conditions and a human-approval gate. It improves the research harness — it does not claim to be a generally intelligent agent.

---

## Using with Claude Code

This project includes a `CLAUDE.md` that gives Claude Code full context on commands, architecture, and key files.

```bash
claude    # Start Claude Code — reads CLAUDE.md automatically
```

---

## License

MIT — see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
