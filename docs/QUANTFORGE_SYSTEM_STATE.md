# QuantForge — System State

> Single source of truth for what QuantForge **is**, what is **validated**, and what is
> **rejected**. Read this first. Maintained alongside `AUTONOMOUS_LOOP_PROTOCOL.md`.

---

## 1. What QuantForge currently does

QuantForge is a **paper-only, human-gated crypto trading-research system**. It does **not**
trade real money and **never** mutates live trading state autonomously.

- **Strategy engines**
  - **Carry (funding-rate) harvester** (`scripts/quantforge_carry_harvester.py`,
    `qf_mlops/carry_backtest.py`) — the *only* engine with a validated, cost-honest edge.
    Enters when |funding| clears a threshold, holds ≥ `min_hold` intervals, exits on decay.
    Pooled, not naive per-coin.
  - **ML directional predictor** (`scripts/quantforge_ml*.py`) — repeatedly measured at OOS
    AUC ≈ 0.50 (coin-flip). **No edge.** Kept as a research substrate, not a profit engine.
  - **Moonshot sleeve** (`scripts/quantforge_moonshot.py`) — negative-EV in backtest; **frozen**.
- **Autonomous loop** — a self-reflection daemon (`scripts/quantforge_reflect.py`) proposes
  param changes within bounds; a self-healing layer (`scripts/quantforge_self_heal_actions.py`)
  auto-fixes operational issues; a self-improvement loop (`scripts/qf_safety/self_improvement.py`)
  carries a candidate through sandbox → tests → review → APPROVED and then **stops for a human**.
- **Paper accounting** — equity / invariants / money-conservation checks (`qf_safety/invariants.py`).

The honest bottleneck (per the system's own evaluation): **a real live edge**, not more loop
machinery. Building more machinery on top of a 0.50-AUC signal is explicitly *not* the high-value move.

## 2. Validators (all fail-closed)

| Validator | File | Guards |
|---|---|---|
| Config sentinel | `scripts/qf_config_sentinel.py` | live policy stays within **approved** bounds; **AST-parses the harvester `_POLICY_BOUNDS` and fails closed if it diverges from the approved bounds** (closes the "deployed≠live" gap); logs drift to a tamper-evident hash chain |
| Benchmark / promotion gate | `scripts/quantforge_benchmark_gate.py`, `qf_mlops/benchmark_gate.py` | OOS AUC, calibration, net Sharpe, edge-vs-cost margin, stability across windows, anti-leakage |
| Cost floor & edge measurement | `qf_mlops/cost_floor.py`, `qf_mlops/ml_edge.py`, `qf_mlops/edge_attribution.py` | net-of-cost edge; a signal must clear the cost floor to count |
| Money-conservation invariants | `scripts/qf_safety/invariants.py` | equity conservation, orphan-margin detection |
| Action gate / param-proposal | `scripts/qf_safety/action_gate.py`, `scripts/qf_safety/param_proposal.py` | risk-tiered permission model; risk params + kill switches escalate to a human |

## 3. Safety gates

- **Risk-tiered permission model** (`qf_safety/permissions.py`, `action_gate.py`): every action is
  `REVERSIBLE_OP` → `CONFIG_PROPOSAL` → `CODE_MODEL_CHANGE` → `FINANCIAL_SECURITY`. Anything above
  routine reversible ops is blocked from autonomous execution and routed to human approval.
- **Tamper-evident decision log** (`qf_safety/decision_log.py`) — SHA-256 hash chain.
- **Atomic, content-addressed JSON writes + file locks** (`qf_safety/atomic_json.py`).
- **Candidate pipeline** (`qf_safety/candidate_pipeline.py`) — propose → sandbox → tests → review →
  APPROVED → human-gated deploy. Code mutation is off unless a human enables it.
- **Master verifier** `verify_quantforge.sh` — asserts docs + config sentinel + the safety test
  suite + the full test suite (204 tests). Exit 0 == verified known-good, fail-closed, human-gated.

## 4. Carry-universe verdict

- Broad widening to a large scan-log list: **rejected** — realistic execution costs destroy most
  apparent edge.
- Individual illiquid coins with apparent edge: **capped-size human proposal only**, never an auto-add.
- **Conclusion:** treat carry as a **pooled** edge, not naive standalone-per-coin; hold the current
  universe unless a *new* validator proves otherwise.

## 5. Accepted findings

- Funding **carry** is the only validated profit engine: pooled ~3%/yr, Sharpe ~3.2, maxDD ~1.5%
  unlevered; ~6%/yr at a safe ~2× leverage ceiling that survives a ~35% squeeze.
- Carry win-rate is **~40% and fat-tailed** — profit comes from winner *size*, not frequency.

## 6. Rejected findings (do not re-test without new data)

- ML directional edge (OOS AUC ≈ 0.50); the high-confidence slice was *anti*-predictive; a microcap
  "+13.9bps" edge was an illusion; moonshot negative-EV; cross-venue arb sub-cost.

## 7. Current risks

- **No live edge** while carry is dormant → no live P&L. This is the real ceiling.
- **"Deployed ≠ live" drift** — mitigated by `verify_quantforge.sh` + the config sentinel, but
  stay disciplined: verify the live files, don't trust "deployed."

## 8. Next best development targets (ranked)

1. **Paid alternative data** for carry — the most plausible path to a live edge (budget/credential-gated).
2. **Walk-forward / Monte-Carlo stress harness** as a reusable library (currently bespoke).
3. **Strategy-registry extraction** — lift the strategy classes out of `quantforge_agent.py` into
   their own module (see README "Known trade-offs").

Do **not**: add a new strategy claiming edge on the 0.50-AUC signal; widen the carry universe;
mutate any live param without going through the gate.
