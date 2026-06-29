# QuantForge — System State

> Single source of truth for what QuantForge **is**, what is **validated**, what is
> **rejected**, and where the next safe leverage is. Read this first in any session.
> Maintained by the autonomous loop (see `AUTONOMOUS_LOOP_PROTOCOL.md`).
> Last updated: 2026-06-28.

---

## 1. What QuantForge currently does

QuantForge is a **paper-only, human-gated crypto trading-research system** deployed on a production host.
It does **not** trade real money and **never** mutates live trading state autonomously.

- **Strategy engines**
  - **Carry (funding-rate) harvester** (`quantforge_carry_harvester.py`) — the *only*
    engine with a validated, cost-honest edge. Enters when |funding| clears a threshold
    on the collecting side, holds ≥ `min_hold` funding intervals, exits on decay.
    Pooled, not naive per-coin. Currently **DORMANT** (live funding below threshold).
  - **ML directional predictor** (`quantforge_ml*.py`) — repeatedly measured at OOS
    AUC ≈ 0.50 (coin-flip). **No edge.** Kept only as a research substrate.
  - **Moonshot sleeve** — negative-EV in backtest; **FROZEN**.
- **Self-research loop** — generators (feature proposer, edge-discovery cycle,
  policy sweep, self-evolve) feed an **honest cost-aware gate**; results are recorded to
  a cumulative ledger so dead ends are never re-tried.
- **Paper accounting** — equity/invariants/watchdog stack with money-conservation checks.

The honest bottleneck (stated by the system's own verifiers): **a real live edge / live
fuel**, not more loop machinery. Building more machinery on top of a 0.50-AUC signal is
explicitly *not* the high-value move.

## 2. Existing validators (all fail-closed, human-gated)

| Validator | File | Guards |
|---|---|---|
| Carry-universe admission | `scripts/qf_carry_universe.py` | liquidity, sample ≥20, clears 30bps stress cost, beats random-entry control, stable across 3 windows |
| Carry scaling / win-rate | `scripts/qf_carry_scale.py`, `qf_carry_winrate.py` | safe leverage ceiling vs squeeze; fat-tailed win-rate (~40%) |
| Carry gate | `scripts/qf_carry_gate.py` | fail-closed accept/reject of tuner output |
| 6-criteria validation gate | `scripts/qf_validation_gate.py` | AUC≥0.57, calibration, net Sharpe>1, edge>3×cost, ≥3 windows, anti-leakage |
| Edge discovery / honesty | `scripts/qf_edge_discovery*.py`, `qf_feature_proposer.py` | leak-free OOF, rejects look-ahead & noise |
| Money-conservation invariants | `scripts/qf_safety/invariants.py` | equity conservation, orphan-margin detection |
| **Config sentinel** | `scripts/qf_config_sentinel.py` | live `carry_policy.json` stays within **approved** bounds; **AST-parses the live harvester `_POLICY_BOUNDS` and fails closed if it diverges from the approved bounds** (closes the "deployed≠live" gap on the clamp itself); logs drift to a tamper-evident hash-chain |

## 3. Existing safety gates

- **Hard policy bounds** in the harvester: `enter∈[0.0010,0.0050]`, `exit∈[0.0004,0.0030]`,
  `min_hold∈[3,12]`. The weekly self-tuner may only move *within* these bounds.
- **HUMAN_GATED** markers on every live-impacting action (universe change, leverage, live
  param apply). Nothing promotes to live without explicit human approval.
- **Tamper-evident decision log** (`scripts/qf_safety/decision_log.py`) — SHA-256 hash chain.
- **Atomic JSON writes + file locks** (`scripts/qf_safety/atomic_json.py`).
- **Action gate** (`scripts/qf_safety/action_gate.py`) — classifies every self-heal action
  L0–L4 and routes risky ones to the param-proposal gate; **none execute live directly.**
- **Master verifier** `verify_quantforge.sh` — one command that asserts the whole stack is green:
  docs, config sentinel (+ code↔approval bounds), the **safety test suite** (action-gate,
  money-conservation invariants, decision-log hash-chain, param-proposal authority — 56 tests),
  and all 6 per-area sub-verifiers. Exit 0 == verified known-good, fail-closed, human-gated.

## 4. Current carry-universe verdict (UNCHANGED)

- Broad widening to the 34-coin scan-log list: **REJECTED** — realistic execution costs
  destroy most apparent edge.
- **TNSR: REJECTED** — apparent edge was one clustered funding regime (bootstrap laundering
  of fat tails; red-team caught it).
- **ALICE:** only real edge-qualified candidate, but **illiquid** → only a tiny capped-size
  human proposal, never an auto-add.
- **Conclusion:** hold the current universe unless a *new* validator proves otherwise.
  Treat carry as a **pooled** edge, not naive standalone-per-coin.

## 5. Known ACCEPTED findings

- Funding **carry** is the only validated profit engine: pooled ~3%/yr Sharpe ~3.2,
  maxDD ~1.5% unlevered; ~6%/yr at a safe ~2× leverage ceiling that survives a 35% squeeze.
- Carry win-rate is **~40% and fat-tailed** — profit comes from winner *size*, not frequency
  (explains live W0/L3 as expected variance, not a broken threshold).
- Self-evolution loop can promote winners and no longer freezes on ties (keystone bug fixed).

## 6. Known REJECTED findings (do not re-test without new data)

- ML directional edge (OOS AUC ≈ 0.50). Cascade-fade. High-confidence slice was
  *anti*-predictive. Microcap "+13.9bps" edge was an illusion. Moonshot negative-EV.
  Cross-venue arb sub-cost. Order-flow individual = noise.

## 7. Current risks

- **No live edge** while carry is dormant → no live P&L. This is the real ceiling.
- **"Deployed ≠ live"** drift: past fixes were not always on the live file. Mitigated now by
  `verify_quantforge.sh` + the config sentinel, but stay disciplined: **verify live files.**
- The production host (hardware constitution): no new cron > 1×/hr, ≤35 cron, ≤4 Docker.

## 8. Next best development targets (safe, ranked)

1. **Paid alt-data / live fuel** for carry — the only honest path to live P&L (needs human
   budget approval; credential-gated).
2. **Human-review proposal writer** — turn any gate "PASS→would-promote" into a structured
   proposal artifact for the human, never an auto-apply. (Highest no-approval next item.)
3. **Walk-forward / Monte-Carlo stress harness** as reusable library (currently bespoke).
4. ~~Extend config sentinel to allocator/risk JSON~~ — investigated: the only live *control*
   file is `carry_policy.json` (`allocator-readout.json` is an output readout, not an input).
   Instead the sentinel now enforces **code↔approval bounds consistency** (done 2026-06-28).

Do **not**: add a new strategy claiming edge on the 0.50-AUC signal; widen the carry universe;
mutate any live param; restart stopped Docker containers.

## 9. How a future session should continue

1. `ssh your-prod-host 'cd ~/quantforge && bash verify_quantforge.sh'` — confirm the stack is green.
2. Read this file + the latest `data/quantforge/config_sentinel.json` and `system_state.json`.
3. Pick the highest item from §8 that needs no human/credential approval; if all do, write a
   proposal artifact and stop at the human gate.
4. Follow `AUTONOMOUS_LOOP_PROTOCOL.md`. Record decisions to the knowledge ledger + decision log.
