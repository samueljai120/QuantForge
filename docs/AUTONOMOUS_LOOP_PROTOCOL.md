# QuantForge — Autonomous Loop Protocol

> How an autonomous session operates QuantForge **safely**. This defines the roles, the
> stop conditions, and the non-negotiable safety rules. Read alongside
> `QUANTFORGE_SYSTEM_STATE.md`. Last updated: 2026-06-28.

The loop is **bounded, self-healing, human-gated**. It improves research, validation,
debugging, and auditability autonomously — it does **not** create unrestricted autonomy,
trade real money, or mutate live state on its own.

---

## Roles

The loop runs these roles each cycle (one agent can play several; the *separation of
concerns* is what matters, not the number of processes).

### 1. Planner
- Reads `QUANTFORGE_SYSTEM_STATE.md` §8 + the prior decision log.
- Converts the goal into the **smallest safe** concrete task with the highest expected
  value / risk ratio. Maintains the backlog. Never picks a task that requires live
  mutation or credentials without routing it through the Human Approval Gate first.

### 2. Executor
- Implements the change as a **small, reversible** edit. Writes code + data **only** for
  research/validation/auditability. **Never** touches live trading state or live params.
- Workflow: edit locally → `rsync to production host → verify on production host (project rule).

### 3. Critic
- Reviews the change's assumptions. Hunts hidden overfitting, weak/circular tests, and
  optimistic conclusions. Asks "what would make this wrong?" before accepting it.

### 4. Red-Team
- Actively tries to **break** every strategy, validator, and conclusion: data leakage,
  look-ahead, survivorship, regime concentration, cost underestimation, liquidity traps,
  bootstrap laundering of fat tails, false robustness. A claim that survives the red-team
  is kept; one that doesn't is rejected and logged.

### 5. Verifier
- Runs the gates: `verify_quantforge.sh` (master) + the per-area `verify_*.sh` + pytest.
- Produces **pass/fail evidence** (real output, never "it works"). **Blocks promotion** on
  any failure. Fails closed on uncertainty.

### 6. Memory
- Records every decision, rejected idea, accepted evidence, and known risk to:
  - `QUANTFORGE_SYSTEM_STATE.md` (§5/§6) — accepted and rejected findings, so dead ends
    are never re-tried;
  - the **tamper-evident decision log** (`scripts/qf_safety/decision_log.py`);
  - `QUANTFORGE_SYSTEM_STATE.md` for the human-readable state.

### 7. Human Approval Gate
- Any **live-risk** action (real money, live trade, live param mutation, production policy
  change, universe change, leverage, credential use, legal/compliance) is converted into a
  **proposal artifact** — never executed. The loop stops at the gate and surfaces the
  proposal with evidence + a recommended conservative default.

### 8. Audit Log
- The decision log records *what* changed, *why*, *what evidence* supported it, and *what
  remains uncertain*. The config sentinel records live-config drift. Together they make any
  decision reconstructable after the fact.

---

## The cycle

1. Inspect repo + state (Planner reads system state + ledger).
2. Identify the highest-leverage **safe** weakness.
3. Decide the next improvement (no human question unless a safety gate is hit).
4. Implement the smallest safe change (Executor).
5. Add/update tests first; every validator gets a **negative** test.
6. Run the verifiers (Verifier) — real output as evidence.
7. Red-team the change; fix any failure (don't stop on first error).
8. Record what changed (Memory + Audit Log).
9. Repeat until a stop condition.

---

## Stop conditions

Stop **only** when:
- The master gate `verify_quantforge.sh` exits 0 **and** the selected improvements are
  complete and verified, **or**
- A **Human Approval Gate** is required (live money / live param / production policy /
  credentials / legal), **or**
- A true blocker only the human can clear (a secret, access grant, paid data budget), **or**
- Required files/data are missing, or tool limits prevent continuing.

Never stop merely on uncertainty or the first error — diagnose, fix, continue.

---

## Non-negotiable safety rules

1. No live trading. 2. No real-money execution. 3. No live parameter mutation without
explicit human approval. 4. No hidden changes to production state. 5. No credential
exposure. 6. No bypassing broker/exchange/platform rules. 7. No pretending weak backtests
are real edge. 8. No AGI claims — build the system, don't market the label. 9. Risky
actions become **proposals**, not automatic execution. 10. **Fail closed** whenever
uncertainty is material.

**Hardware resource guidelines** also binds the loop: ≤35 cron jobs, ≤4 Docker
containers, no script > 1×/hr except the watchdog, load < 6.0 before adding anything.
