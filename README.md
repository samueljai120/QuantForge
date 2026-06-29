# QuantForge

![CI](https://github.com/samueljai120/QuantForge/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Tests](https://img.shields.io/badge/tests-148%20passing-brightgreen.svg)

**An autonomous crypto quant-research platform** — ML signal ensembles, regime-aware allocation, a funding-carry engine, a self-evolving research loop, and a fail-closed safety/MLOps stack. ~38,000 lines across 95 Python modules and 148 tests.

QuantForge runs the full loop end to end: it ingests market data, engineers features, trains and validates ML models with leak-free cross-validation, allocates capital across regime-aware strategies, and governs its own changes through a tamper-evident, human-gated safety layer — then proposes its next round of research and repeats.

---

## Highlights

**🤖 Machine-learning signal stack**
- XGBoost + LightGBM ensembles with walk-forward, strictly chronological cross-validation (`TimeSeriesSplit`).
- Per-symbol and target-profile model slicing; configurable forward-return horizons (1h/4h/24h).
- Feature engineering across technical, momentum, volatility, range, order-flow, and derivatives-state signals (funding rates, open interest, top-of-book, market breadth, macro).
- Specialist models + a benchmark gate a candidate must clear before it can be promoted.

**📊 Regime-aware allocation**
- A 6-state market-regime classifier (`STRONG_BULL → STRONG_BEAR`) drives a pluggable strategy registry: trend/HODL, mean-reversion in chop, a leveraged futures lane, liquidation-dip, funding-carry, CVD momentum, volatility breakout, cross-asset, OI-divergence, and an ML scanner.
- A regime-weight table rebalances exposure as conditions shift; risk-adjusted position sizing with correlation penalties.

**🔁 Autonomous research loop**
- A self-reflection daemon reasons over realized performance and proposes parameter changes within safety bounds.
- A feature proposer self-engineers candidate signals and an edge-discovery cycle tests each one leak-free on real OHLCV.
- A research director orchestrates multi-slice campaigns; a self-evolution loop tunes the system — every change is gated, never auto-applied to live state.

**🛠️ MLOps + persistence (`qf_mlops`)**
- Model registry with atomic, content-addressed, versioned persistence (promotions are never silent).
- Carry backtester, edge-attribution, baselines, and benchmark gating.

**🔒 Fail-closed safety core (`qf_safety`)**
- Every gate rejects on uncertainty rather than silently passing — a missing key, an out-of-bounds number, or a missing file returns a non-zero exit.
- Tamper-evident **SHA-256 hash-chained decision log**; an action gate that classifies every action by risk level; money-conservation invariants; a candidate-promotion pipeline.
- A **config sentinel** AST-parses the live source and asserts the deployed policy bounds match the approved baseline — closing the "deployed ≠ live" gap at the code level.
- Every live-impacting action (params, leverage, universe) is `HUMAN_GATED`: it emits a proposal artifact and stops.

**✅ Engineering**
- 95 Python modules, **148 tests green on Python 3.10 / 3.11 / 3.12** (CI matrix).
- Leak-free evaluation enforced by a test that *deliberately* feeds a future-return feature and asserts the gate rejects it.
- A master verifier (`verify_quantforge.sh`) asserts the entire stack — docs, config sentinel, safety suite, sub-verifiers — and exits 0 only when the system is in a known-good, auditable state.

---

## Architecture

```
                      ┌─────────────────────────────────────────────┐
   market data  ───▶  │  feature engineering (technical · momentum · │
   (KuCoin API)       │  vol · order-flow · funding/derivatives)     │
                      └───────────────────────┬─────────────────────┘
                                              ▼
                      ┌─────────────────────────────────────────────┐
                      │  ML training: XGBoost+LightGBM ensemble,     │
                      │  walk-forward TimeSeriesSplit, per-symbol    │
                      └───────────────────────┬─────────────────────┘
                                              ▼
                      ┌─────────────────────────────────────────────┐
                      │  benchmark gate (AUC · calibration · Sharpe ·│
                      │  edge-vs-cost · stability · anti-leakage)    │  ◀── fails closed
                      └───────────────────────┬─────────────────────┘
                                              ▼
   regime detector ─▶ ┌─────────────────────────────────────────────┐
   (6-state)          │  regime-aware allocator + strategy registry  │
                      └───────────────────────┬─────────────────────┘
                                              ▼
                      ┌─────────────────────────────────────────────┐
                      │  paper portfolio ledger · governance/audit   │
                      └───────────────────────┬─────────────────────┘
                                              ▼
                      ┌─────────────────────────────────────────────┐
                      │  qf_safety: action gate · invariants ·       │
                      │  hash-chain decision log · HUMAN_GATED        │
                      └─────────────────────────────────────────────┘
                          ▲                                     │
                          └──── self-reflection / feature ◀─────┘
                               proposer / edge-discovery loop
```

Key modules:

```
scripts/
  quantforge_paper.py        Paper-trading engine: scan, signal, ledger
  quantforge_agent.py        Regime-aware allocator + pluggable strategy registry
  quantforge_ml.py           XGBoost + LightGBM ensemble, walk-forward CV
  quantforge_ml_train.py     Extended trainer: TimeSeriesSplit + target-profile slicing
  quantforge_features.py     Feature engineering pipeline
  quantforge_regime.py       6-state market-regime detector
  quantforge_reflect.py      Self-reflection daemon — proposes param changes
  quantforge_research_director.py  Orchestrates multi-slice research campaigns
  quantforge_governance.py   Model/strategy health snapshots
  qf_mlops/                  Model registry, carry backtest, benchmark gate, edge attribution
  qf_safety/                 Fail-closed core: CAS store, decision log, action gate,
                             invariants, candidate pipeline, config schema
  config.py                  Env-driven config + runtime guard

docs/    System state, autonomous-loop protocol, evaluation verdict, roadmap
tests/   32 files, 148 tests — one per gate, invariant, and failure mode
```

---

## Quick start

```bash
git clone https://github.com/samueljai120/QuantForge.git
cd QuantForge
./setup.sh                                    # venv + deps + .env
python3 -m pytest tests/ -q                   # 148 tests
QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_paper.py scan   # run a market scan
```

**Prerequisites:** Python 3.10+, pip. Market data uses KuCoin's public API (no key needed for read-only). An OpenRouter or Anthropic key enables the LLM-assisted reflection/research components. See `.env.example` for all configuration (everything is env-driven).

---

## Rigorous by design

Backtests lie. Most trading code overfits, leaks the future into the past, and ships a beautiful equity curve that evaporates the moment it goes live. QuantForge is engineered so that *can't* happen quietly:

- **Leak-free, or it doesn't ship** — strictly chronological `TimeSeriesSplit`, enforced by a test that deliberately feeds a future-return feature and asserts the gate *rejects* it.
- **A multi-criteria benchmark gate** — a signal must clear out-of-sample AUC, calibration, net-of-cost Sharpe, edge-vs-cost margin, and stability across multiple time windows before it can be promoted. It fails closed.
- **Tamper-evident and human-gated** — every promotion and live-impacting change is recorded to a SHA-256 hash-chain and requires explicit approval; nothing mutates live state on its own.

Point it at a market, bring your own features and ideas, and QuantForge tells you — honestly, with the receipts — whether the edge is real *before* you risk a dollar. That discipline is the product. (Research notes from the bundled crypto study, including which strategies held up under cost, live in [`docs/QUANTFORGE_VERDICT.md`](docs/QUANTFORGE_VERDICT.md).)

> **Not financial advice.** This is a paper-trading and research system; it has never executed a real trade. Most cycle scripts refuse to run outside their configured host unless `QF_ALLOW_LOCAL_RUNTIME=1` is set.

---

## Repo layout

```
QuantForge/
  scripts/               67 modules — engine, ML, allocation, research, governance
  scripts/qf_mlops/      15 modules — MLOps (model registry, carry backtest, attribution)
  scripts/qf_safety/     13 modules — fail-closed safety core
  tests/                 32 test files (148 tests)
  docs/                  system state · loop protocol · evaluation verdict · roadmap
  verify_quantforge.sh   master verifier (exit 0 = stack green)
  setup.sh · .env.example · CLAUDE.md · CONTRIBUTING.md
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New strategies must pass the benchmark gate and the leak-free evaluation before promotion.
