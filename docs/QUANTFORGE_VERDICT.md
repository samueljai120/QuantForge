# QuantForge — Final Honest Verdict & Where the Real Money Is

> One-page decision memo. Date: 2026-06-28. Evidence-based; no marketing.
> Supersedes any "AI that profits from any market state" framing.

## TL;DR
QuantForge is **not a money printer and cannot become one under current constraints**
(free data, ~$5k paper, no budget, retail execution). It correctly converged on the only
edge available to it — **delta-neutral funding carry** — which is real but small
(~+1.4%/yr, Sharpe ~2.9, ~1% maxDD on the pooled sim). Every other avenue has been
honestly tested and rejected. **Run carry as a tiny background yield; move the money-printer
ambition to products.**

## What was tested, and the verdict
| Strategy | Result |
|---|---|
| Directional ML (RSI, momentum, vol, range, accel, orderflow) | NO EDGE — OOS AUC ≈ 0.50 (coin flip) |
| Cross-venue arbitrage | NO EDGE — sub-cost at retail |
| Moonshot sleeve | NEGATIVE EV |
| Always-on cross-sectional carry factor (tested 2026-06-28) | NEGATIVE EV — spread is real (beats shuffle by ~55pp) but ~10–20× too small to cover turnover cost; net −70% to −132%/yr |
| **Pooled funding carry** | ✅ **VALIDATED — the one real edge** (small, market-neutral) |

The carry sleeve's live churn (3 trades, −$0.75) was the **fee problem in miniature**:
funding collected (1–7¢) < round-trip cost (~30¢). The ~95% idle time is not a bug — it is
the cost discipline that keeps carry positive at all.

## Why this matches how real trading firms profit (researched 2026-06-28)
Profitable systematic traders do **not** have a magic predictor. They have a *structural*
edge, and every one costs money/infra/scale that retail-on-free-data does not have:
- **Speed / latency arb** — arb windows ~2.7s, captured by HFT on dedicated nodes. ❌
- **Market making / spread capture** — needs low latency + rebates. ❌
- **Alternative / paid data** — an information edge you'd have to buy. ❌ (no budget)
- **Scale + many uncorrelated signals** — needs capital + research teams. ❌
- **Delta-neutral carry / funding arb** — the one retail-accessible edge. ✅ (what we have)

Peer-reviewed evidence on AI agents specifically (StockBench, arXiv 2510.02209, Oct 2025):
*"most LLM agents struggle to outperform the simple buy-and-hold baseline"* and *"excelling
at static financial knowledge does not translate into successful trading."* The "agent that
profits from any market state" is unsolved even for funded labs — not a gap in our code.

## What would make carry actually pay (all gated on money we don't have)
1. **Capital** — millions, so 1–6%/yr is real income.
2. **Leverage** — delta-neutral, so leverage is comparatively safe (HUMAN-gated; only after live proof).
3. **Cheap execution** — maker rebates / VIP fee tiers / negative-fee venues.
4. **More venues at once** — less idle time (naive cross-sectional fails on turnover cost).

## Decision
- **KEEP:** pooled carry running as a small, low-risk, market-neutral paper yield. Stop
  building validators/sweeps around it — it is done.
- **STOP:** hunting a directional/predictive edge on free data. The evidence is conclusive.
- **PIVOT:** money-printer effort → **products / revenue** (QuantForge has 0 paying customers;
  the honest diagnosis is "strong tech, weak business"). A single $50/mo customer beats a
  $5k market-neutral bot earning ~$70/yr, with far better odds than retail alpha.
- **REVISIT trading** only when real capital exists to deploy carry at scale.

## Sources
Billion Dollar Algorithms (why retail algo traders fail); StockBench arXiv 2510.02209
(LLM agents vs buy-and-hold); QuantVPS (latency arbitrage); Arbitrage Scanner (crypto bot
cost reality); Yahoo Finance (Polymarket arb bots: identical edge, infra decided the winner).
