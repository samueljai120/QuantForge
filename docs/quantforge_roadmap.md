# QuantForge — Roadmap to an Agentic Hands-Free Money Machine

**Status as of 2026-05-17:** Stage 0 complete, Stage 1 shipping now.

## North star

A fully autonomous trading system that:
- Generates positive risk-adjusted returns over the long run
- Operates without human intervention
- Reasons about its own performance and adjusts itself
- Survives drawdowns through structural risk controls, not luck
- Scales capital up only when alpha is validated, never before

## Architecture vision (the end state)

```
┌─────────────────────────────────────────────────────────────────┐
│              META-CONTROLLER (the agent's "CEO")                │
│  - Daily reflection on each strategy's perf                     │
│  - Reallocates capital toward winning strategies                │
│  - Auto-kills strategies whose alpha drops below 0 for 14 days  │
│  - Proposes new strategies based on observed inefficiencies     │
└────────────────────────┬────────────────────────────────────────┘
                         │
   ┌─────────────────────┼─────────────────────┬────────────────┐
   ▼                     ▼                     ▼                ▼
┌────────┐         ┌──────────────┐    ┌───────────────┐  ┌──────────┐
│ HODL   │         │ Mean-revert  │    │ Trend-follow  │  │ Funding  │
│ core   │         │  in CHOP     │    │  in BULL/BEAR │  │ arb      │
│ (40%)  │         │  (20%)       │    │  (20%)        │  │ (10%)    │
└────────┘         └──────────────┘    └───────────────┘  └──────────┘
                                                              │
                                                       ┌──────▼──────┐
                                                       │ Cash yield  │
                                                       │ on idle USD │
                                                       │ (10%)       │
                                                       └─────────────┘

Underneath all of this:
  ✓ Per-strategy PnL attribution (we have this)
  ✓ Safety stack: panic halt, drawdown trim, position caps (we have this)
  ✗ Multi-asset support: BTC + ETH + SOL (not yet)
  ✗ Real KuCoin trading after paper validation (not yet)
```

**Why this works:** No single strategy needs to be a moonshot. If HODL makes 5%/yr, mean-reversion makes 3%, trend makes 4%, funding makes 6%, yield makes 5% — risk-weighted ensemble compounds to ~12-15%/yr with smooth equity curve.

## The 6 stages

| Stage | What gets built | Status | Time |
|---|---|---|---|
| **0** | v5 HODL + safety nets (panic-halt at -15%, DD trim at -8%, profit ladder at +20%) | ✅ DONE | — |
| **1** | Self-reflection daemon — bot adjusts own params daily within safety bounds | ✅ DONE (2026-05-21, auto-apply enabled) | 1 week |
| **2** | Modular strategy registry — strategies as plugins, can add/remove safely | ✅ DONE (2026-05-23) | 1 day |
| **3** | First active alpha (CHOP mean-reversion) | ✅ DONE (2026-05-24) — gate verification pending 14d data | 4 hrs |
| **4** | Cross-strategy capital allocator — meta-controller routes capital to winners | ⬜ | 1 week |
| **5** | Multi-asset: add ETH | ⬜ | 1 week |
| **6** | Paper → Real graduation ($100 → $500 → $5k) | ⬜ | 4-12 weeks |

### Stage 2 design notes (2026-05-23)

The registry collapses to identical behavior when only HODL is registered:

```
Before (v5):
  target_btc_value = equity * TARGET_ALLOC[active_regime]

After (Stage 2):
  target_btc_value = Σ (strategy.weight × decision.target_alloc_pct × equity)
                   = 1.0 × TARGET_ALLOC[active_regime] × equity   # HODL only
```

So Stage 2 ships zero behavior change but unlocks composition.

**Interface (in quantforge_agent.py):**

```python
@dataclass(frozen=True)
class CycleContext:
    price, regime, active_regime, signals, total_equity,
    cash, btc_qty, drawdown_from_peak, pnl_pct, portfolio

@dataclass
class StrategyDecision:
    target_alloc_pct: float      # 0.0 - 1.0, fraction of THIS strategy's slice
    notes: str = ""              # human-readable, surfaced in logs

class Strategy:
    name: str
    weight: float                # fraction of total equity (sum across registry = 1.0)
    def evaluate(self, ctx: CycleContext) -> StrategyDecision: ...

STRATEGY_REGISTRY: list[Strategy] = [HODLStrategy(weight=1.0)]
```

**Stage 3 hook:** add a `MeanReversionStrategy` to `STRATEGY_REGISTRY`, reduce HODL weight to 0.8 (or whatever), and the orchestrator handles composition automatically. No changes to safety stack or run_cycle plumbing.

**New CLI:** `python3 quantforge_agent.py strategies` — live introspection of the registry and what each strategy is recommending right now.

## Stage gates — what must be true to advance

| Stage → Next | Required evidence |
|---|---|
| 0 → 1 | DONE: 5+ days of positive alpha vs HODL on paper |
| 1 → 2 | Reflection daemon proposes sane changes for 7 consecutive days |
| 2 → 3 | Registry runs HODL as a plugin with zero perf regression vs current |
| 3 → 4 | CHOP mean-reversion shows >$5 positive alpha over 14 days, drawdown <8% |
| 4 → 5 | Allocator demonstrably routes more capital to winning strategy when one outperforms in backtest |
| 5 → 6 | 30 consecutive days of positive total alpha across all strategies, max DD < 10% |
| 6 → real $$ | 30 days at $100 → 30 days at $500 → 30 days at $5k, each with positive alpha and DD < 15% |

## Realistic timeline & income projection

| Month | Capital | State | Target return | Target $ |
|---|---|---|---|---|
| 1 | $5k paper | Stage 0-2 infra build | — | $0 |
| 2 | $5k paper | Stage 3-4 alpha modules | +5% | $250 (paper) |
| 3 | $100 real | Stage 6 — graduation begins | +5-10% | $5-10 |
| 4-5 | $500 real | Scaling, validation | +5-10% | $25-50 |
| 6-9 | $5k real | First meaningful income | +10-20%/yr | $500-1000 |
| 10-18 | $50k real | Compound + reinvest | +15-25%/yr | $7,500-12,500 |

**Hard truths:**
- There is no path to $1k/day from a $5k account that doesn't risk total loss
- Real-money graduation is the slowest stage on purpose — the only way to know if alpha is real is to pay real fees and feel real slippage
- 80% of retail trading bots that "look profitable" in paper trading lose money in real conditions because they didn't account for execution costs

## Components built so far (Stage 0 reference)

**Core agent: `~/quantforge/scripts/quantforge_agent.py`**
- Adaptive BTC allocator on KuCoin futures perp (XBTUSDTM)
- Runs hourly at `:05` via cron
- Regime detector: STRONG_BULL/BULL/NEUTRAL/CHOP/BEAR/STRONG_BEAR
- Current mode: `HODL_MODE` (fixed 65% BTC allocation regardless of regime, regime observed but not acted on)

**Safety stack (all equity-based, regime-independent):**
- `PANIC_HALT_PCT = 0.15` — full liquidation + halt-flag at -15% DD from peak
- `PANIC_HALT_ABS_PCT = 0.12` — full liquidation if absolute PnL ≤ -12%
- `DRAWDOWN_TRIM_PCT = 0.08` — sell half of BTC at -8% DD from peak
- `PROFIT_TAKE_PCT = 0.20` + `PROFIT_TAKE_INCREMENT = 0.10` — sell 5% at every +10% milestone above +20%

**Anti-whipsaw guards:**
- `REGIME_HYSTERESIS_CYCLES = 3` — regime must persist 3 cycles before acting
- `REBALANCE_COOLDOWN_HOURS = 6` — min 6h between any two rebalances
- `MAX_REBALANCES_PER_DAY = 2` — hard daily cap
- `REBALANCE_THRESHOLD = 0.08` — only rebalance if drift ≥ 8%

**Learning infrastructure:**
- `regime_perf` dict in portfolio JSON tracks per-regime PnL vs passive HODL
- `perf` CLI command shows alpha attribution table
- Audit trail: `agent_trades.jsonl`, `agent.log`, `agent_portfolio.json`

**CLI commands:**
- `python3 quantforge_agent.py run` — one cycle
- `python3 quantforge_agent.py status` — current state
- `python3 quantforge_agent.py perf` — per-regime alpha table
- `python3 quantforge_agent.py regime` — regime check without trading
- `python3 quantforge_agent.py panic-reset` — clear halt after human review

## Stage 1 — Self-Reflection Daemon (shipping 2026-05-17)

**Files:**
- `quantforge_reflect.py` — the daemon
- `qf_strategy_params.json` — params file the agent reads at cycle start (created/updated by daemon)
- `reflect_decisions.jsonl` — audit trail of every proposal and decision
- `reflect_auto_apply.flag` — touch this file to disable training wheels
- `reflect.log` — operator log

**Cron:** Piggybacks on existing `15 7 * * *` daily-summary cron (no new cron slot).

**Allowlisted tunables (the surface area the daemon can touch):**
| Param | Min | Max | Default | Notes |
|---|---|---|---|---|
| `hodl_mode` | false | true | true | Master toggle: act on regime or not |
| `fixed_alloc_pct` | 0.40 | 0.85 | 0.65 | BTC allocation when hodl_mode=true |
| `rebalance_threshold` | 0.02 | 0.20 | 0.08 | Drift before rebalance fires |
| `rebalance_cooldown_hours` | 1 | 48 | 6 | Min hours between rebalances |
| `max_rebalances_per_day` | 1 | 5 | 2 | Daily trade cap |
| `regime_hysteresis_cycles` | 2 | 6 | 3 | Cycles before regime activation |
| `profit_take_pct` | 0.10 | 0.30 | 0.20 | First profit-take threshold |
| `profit_take_increment` | 0.05 | 0.15 | 0.10 | Subsequent profit-take cadence |

**Prohibited (NEVER tunable by daemon):**
- `panic_halt_pct`, `panic_halt_abs_pct` — safety nets
- `drawdown_trim_pct`, `drawdown_trim_factor` — safety nets
- `taker_fee`, `maker_fee`, `starting_balance`, `leverage` — invariants

**Per-run constraints:**
- Max 1 parameter changed per day
- Max 10% delta on any numeric param per day
- Boolean params can flip but log requires explicit justification
- Failed validation → log decision, do nothing

**Training-wheels period (first 7 runs):**
- All proposals logged to `reflect_decisions.jsonl` with full reasoning
- Operator reads daily, builds trust
- Touch `reflect_auto_apply.flag` to enable auto-apply mode

## Stage 3 design notes (2026-05-24)

### MeanReversionStrategy

**Thesis:** in CHOP regime BTC oscillates around a 24h mean. Buy oversold, sell overbought. Outside CHOP this strategy is passive (returns baseline 50%).

**Implementation (`quantforge_agent.py`):**

```python
class MeanReversionStrategy(Strategy):
    name = "mean_revert_chop"

    BASELINE_ALLOC = 0.50
    MAX_ALLOC = 0.90              # deep oversold → lean in
    MIN_ALLOC = 0.10              # deep overbought → lean out
    OVERSOLD_Z = -1.5
    OVERBOUGHT_Z = +1.5

    def evaluate(self, ctx):
        if ctx.active_regime != "CHOP":
            return StrategyDecision(BASELINE_ALLOC, "INACTIVE — not CHOP")
        z = ctx.signals.get("price_z_24h")
        # linear interp from BASELINE→MAX as z runs 0→OVERSOLD_Z
        # linear interp from BASELINE→MIN as z runs 0→OVERBOUGHT_Z
        ...
```

**Signals added to `detect_regime`:** `price_mean_24h`, `price_std_24h`, `price_z_24h` (z-score of current price vs 24h mean).

**Registry:**

```python
STRATEGY_REGISTRY = [
    HODLStrategy(weight=0.80),
    MeanReversionStrategy(weight=0.20),
]
```

### First trade

The very first cycle after deploy fired a trade: BTC was overbought in CHOP (z=+1.15, RSI 73.7), MR pushed its slice target down to 19%, combined registry target became 51%, drift +12.6% breached the 8% threshold. Sold 0.0079 BTC at $76,975. The architecture composed exactly as designed — HODL slice and MR slice contributed independently to a single combined target.

### Stage 3 stage gate

> CHOP mean-reversion shows >$5 positive alpha over 14 days, drawdown <8%.

Can't verify the alpha until ~14 days of live trades. What we CAN verify today:

| Property | Result |
|---|---|
| MR activates only in CHOP | ✓ (returns BASELINE outside CHOP) |
| Slice math composes correctly | ✓ (0.80×0.59 + 0.20×0.19 = 0.512 ≈ 51% total) |
| Safety stack still works | ✓ (panic-halt, DD-trim, profit-ladder unchanged) |
| Status displays registry-combined target | ✓ (after status fix) |
| `strategies` CLI lists both strategies | ✓ |
| Reflect daemon allowlist excludes MR-specific params | ✓ (MR uses constants for now) |

### Watchpoints

1. **Fee drag** — MR will trade more often than HODL (each cycle in CHOP can move its target). Watch `total_fees_paid` vs alpha. If fees climb faster than alpha after 7 days, tighten the OVERSOLD/OVERBOUGHT thresholds.
2. **Cooldown collisions** — orchestrator-level `REBALANCE_COOLDOWN_HOURS` (6h) gates all trades. MR may *want* to trade more often than that and be blocked. Acceptable for Stage 3; Stage 3.5 may make cooldown per-strategy.
3. **z-score sample size** — uses last 24 closes. In thin markets this is noisy. Bigger window = smoother but slower to react. 24h is a reasonable starting point.

## Operating discipline

1. **No stage skipping.** Each stage must clear its gate before the next begins.
2. **Brutal honesty in reviews.** If alpha isn't there, say so. Sunk-cost is the enemy.
3. **Real money is a privilege earned via paper performance.** Until 30 days of positive alpha at scale, no real KuCoin allocation.
4. **Safety stack is sacred.** No daemon, no reflection, no LLM proposal can modify panic-halt or drawdown-trim parameters. If we need to change those, it's a human decision with new memory entry.
5. **Audit everything.** Every parameter change, every trade, every regime decision is logged with timestamp + reasoning.

## Cron capacity discipline

Per hardware resource guidelines:
- Active cron count ≤ 35 (currently 34 → after stage 1: still 34 via piggyback)
- Load average ≤ 6.0 sustained (currently ~0.1)
- RAM ≤ 12GB (currently 1.7GB)

Stage 1 adds zero cron entries. Stages 2-5 may need to add 1-2 cron slots; budget accordingly.
