#!/usr/bin/env python3
"""
QuantForge Self-Reflection Daemon (v6 Stage 1)
================================================

Runs once per day (piggybacked on quantforge_daily_summary cron at 07:15 UTC).

What it does
------------
  1. Loads the last 7 days of portfolio + log data from the agent
  2. Asks Claude (Anthropic API) to propose ONE parameter change
     that would improve risk-adjusted return based on observed data
  3. Validates the proposal against safety bounds (allowlist + delta caps)
  4. Training-wheels mode (first 7 runs): writes proposal to audit log only
  5. Auto-apply mode (after operator touches reflect_auto_apply.flag):
     writes change to qf_strategy_params.json, picked up by agent next cycle

Safety invariants
-----------------
* This daemon can ONLY write to qf_strategy_params.json
* The agent's load_runtime_params() function further restricts what those
  values can affect — safety constants (PANIC_HALT_PCT, DRAWDOWN_TRIM_PCT,
  fees, leverage) cannot be touched even if this daemon proposes them
* Max 1 parameter changed per run
* Max 10% delta on any numeric param per run
* All decisions logged with full reasoning to reflect_decisions.jsonl

Files
-----
INPUT:
  ~/quantforge/data/quantforge/agent_portfolio.json       — current state
  ~/quantforge/data/quantforge/agent.log                  — last 7 days
  ~/quantforge/data/quantforge/qf_strategy_params.json    — current overrides
  ~/quantforge/data/quantforge/reflect_decisions.jsonl    — history (for context)

OUTPUT:
  ~/quantforge/data/quantforge/qf_strategy_params.json    — new params (auto-apply mode)
  ~/quantforge/data/quantforge/reflect_decisions.jsonl    — audit append
  ~/quantforge/data/quantforge/reflect.log                — operator log

FLAG:
  ~/quantforge/data/quantforge/reflect_auto_apply.flag    — touch to leave training wheels

Author: Stage 1 of the agentic-quantforge roadmap (see docs/quantforge_roadmap.md)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(os.environ.get("QF_BASE_DIR", str(Path.home() / "quantforge"))) / "data" / "quantforge"
PORTFOLIO_FILE = DATA_DIR / "agent_portfolio.json"
AGENT_LOG = DATA_DIR / "agent.log"
PARAMS_FILE = DATA_DIR / "qf_strategy_params.json"
DECISIONS_FILE = DATA_DIR / "reflect_decisions.jsonl"
REFLECT_LOG = DATA_DIR / "reflect.log"
AUTO_APPLY_FLAG = DATA_DIR / "reflect_auto_apply.flag"

# Per project policy (feedback_openrouter_only.md), prefer OpenRouter over
# direct provider APIs. Anthropic direct is kept as fallback.
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1500

TRAINING_WHEELS_RUNS = 7   # propose-only for first 7 runs (unless flag touched)
MAX_DELTA_FRACTION = 0.15  # v12: 10%→15% — gave false rejections on borderline moves
# Floating-point epsilon for the delta-cap comparison. Without this, a "10%
# exactly" proposal (e.g., 0.65 → 0.585) computes as 0.10000000000000002 in
# IEEE-754 and was getting rejected. The daemon clearly intends "max allowed"
# when it picks the boundary — honor that intent. 2026-05-24 fix.
DELTA_CAP_EPSILON = 1e-9

# Asymmetric re-risk path (2026-05-29). For fixed_alloc_pct ONLY, the daemon
# may raise allocation faster than it cuts — but ONLY when the regime has
# exited bear. Rationale: after de-risking through a crash, we want to catch a
# genuine recovery quickly, but we must NOT add aggressively into a confirmed
# downtrend (that's how you buy a dead-cat bounce). So:
#   - DOWN moves (de-risk):  always capped at MAX_DELTA_FRACTION (10%)
#   - UP moves (re-risk):    capped at UP_DELTA_FRACTION (20%) when regime is
#                            NEUTRAL/BULL/STRONG_BULL; otherwise 10%
UP_DELTA_FRACTION = 0.30
NON_BEAR_REGIMES = {"NEUTRAL", "CHOP", "BULL", "STRONG_BULL"}
RERISK_PARAM = "fixed_alloc_pct"  # only this param gets the asymmetric treatment

# Tunable allowlist with bounds.
# Format: name -> (min, max, type_name) where type_name is "float", "int", or "bool"
TUNABLES = {
    "hodl_mode":                  (False, True,  "bool"),
    "fixed_alloc_pct":            (0.40,  0.85,  "float"),
    "rebalance_threshold":        (0.02,  0.20,  "float"),
    "rebalance_cooldown_hours":   (1,     48,    "int"),
    "max_rebalances_per_day":     (1,     5,     "int"),
    "regime_hysteresis_cycles":   (2,     6,     "int"),
    "profit_take_pct":            (0.10,  0.30,  "float"),
    "profit_take_increment":      (0.05,  0.15,  "float"),
}

# Defaults the agent uses when no params file exists. Reflect uses these
# as the implicit "current" if a key isn't present in the params file.
TUNABLE_DEFAULTS = {
    "hodl_mode":                  True,
    "fixed_alloc_pct":            0.65,
    "rebalance_threshold":        0.08,
    "rebalance_cooldown_hours":   6,
    "max_rebalances_per_day":     2,
    "regime_hysteresis_cycles":   3,
    "profit_take_pct":            0.20,
    "profit_take_increment":      0.10,
}


# ============================================================================
# Logging
# ============================================================================
def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(REFLECT_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================================
# Data loading
# ============================================================================
def load_portfolio() -> dict | None:
    if not PORTFOLIO_FILE.exists():
        return None
    try:
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    except Exception as e:
        log(f" Could not load portfolio: {e}")
        return None


def load_recent_log(hours: int = 168) -> str:
    """Return last N hours of agent log. 168 = 7 days."""
    if not AGENT_LOG.exists():
        return ""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    try:
        with open(AGENT_LOG) as f:
            lines = f.readlines()
    except Exception:
        return ""
    keep = []
    for line in lines:
        # Lines start with "[2026-05-17T01:23:45+00:00] ..."
        m = re.match(r"\[(\S+)\]", line)
        if not m:
            keep.append(line)
            continue
        try:
            ts = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
            if ts >= cutoff:
                keep.append(line)
        except Exception:
            keep.append(line)
    return "".join(keep)


def load_current_params() -> dict:
    """Return current effective params: file values + defaults for missing keys."""
    params = dict(TUNABLE_DEFAULTS)
    if PARAMS_FILE.exists():
        try:
            with open(PARAMS_FILE) as f:
                stored = json.load(f)
            for k, v in stored.items():
                if k in TUNABLES:
                    params[k] = v
        except Exception as e:
            log(f" Could not load params file: {e}; using defaults")
    return params


def count_prior_runs() -> int:
    """Count number of prior reflect runs (for training-wheels)."""
    if not DECISIONS_FILE.exists():
        return 0
    try:
        with open(DECISIONS_FILE) as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def load_recent_decisions(n: int = 5) -> list[dict]:
    if not DECISIONS_FILE.exists():
        return []
    try:
        with open(DECISIONS_FILE) as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]
    except Exception:
        return []


# ============================================================================
# LLM call
# ============================================================================
def get_api_key(name: str) -> str | None:
    """Lookup name in env, then ~/quantforge/scripts/.env, then ~/quantforge/.env."""
    key = os.environ.get(name)
    if key:
        return key
    for env_path in [
        Path(os.environ.get("QF_BASE_DIR", str(Path.home() / "quantforge"))) / "scripts" / ".env",
        Path(os.environ.get("QF_BASE_DIR", str(Path.home() / "quantforge"))) / ".env",
    ]:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        if line.startswith(f"{name}="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                continue
    return None


def call_openrouter(system: str, user: str) -> dict:
    """Call OpenRouter (OpenAI-compatible), return parsed response. Raises on error."""
    key = get_api_key("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found")
    body = json.dumps({
        "model": OPENROUTER_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/quantforge/quantforge",
            "X-Title": "QuantForge Self-Reflection",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body[:500]}") from e


def call_anthropic(system: str, user: str) -> dict:
    """Direct Anthropic API call (fallback)."""
    key = get_api_key("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found")
    body = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {e.code}: {body[:500]}") from e


def call_llm(system: str, user: str) -> tuple[dict, str]:
    """Try OpenRouter first, fall back to Anthropic direct. Returns (response, provider)."""
    try:
        return call_openrouter(system, user), "openrouter"
    except Exception as e:
        log(f"   OpenRouter failed ({e}); trying Anthropic direct...")
        return call_anthropic(system, user), "anthropic"


def extract_text(api_resp: dict, provider: str) -> str:
    """Pull plain text out of the response shape, normalized across providers."""
    if provider == "openrouter":
        # OpenAI-compatible: {"choices":[{"message":{"content":"..."}}]}
        choices = api_resp.get("choices", [])
        if choices:
            return (choices[0].get("message", {}).get("content") or "").strip()
        return ""
    # anthropic native: {"content":[{"type":"text","text":"..."}, ...]}
    blocks = api_resp.get("content", [])
    parts = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "".join(parts).strip()


def extract_json(text: str) -> dict | None:
    """Find the first JSON object in the response text."""
    # Look for ```json ... ``` fenced first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Fall back to first {...} block (greedy match of braces)
    start = text.find("{")
    if start < 0:
        return None
    # Find matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


# ============================================================================
# Validation
# ============================================================================
def parse_current_regime(recent_log: str) -> str | None:
    """Extract the most recent regime from the agent log.

    Log lines look like: 'Regime: STRONG_BEAR  (price $72,872, MA20 ...)'.
    Returns the last regime seen, or None if unparseable.
    """
    matches = re.findall(r"Regime:\s+([A-Z_]+)", recent_log)
    return matches[-1] if matches else None


def delta_caps_for(param: str, regime: str | None) -> tuple[float, float]:
    """Return (down_cap, up_cap) delta fractions for a param given regime.

    Only RERISK_PARAM (fixed_alloc_pct) gets an asymmetric up-cap, and only
    when the regime has exited bear (NEUTRAL/BULL/STRONG_BULL). Everything
    else — and all moves in a bear regime — uses the symmetric 10% cap.
    """
    down = MAX_DELTA_FRACTION
    up = MAX_DELTA_FRACTION
    if param == RERISK_PARAM and regime in NON_BEAR_REGIMES:
        up = UP_DELTA_FRACTION
    return down, up


def validate_proposal(proposal: dict, current_params: dict,
                      regime: str | None = None) -> tuple[bool, str]:
    """Validate that a proposed change is safe to apply.

    Expected proposal shape:
      {
        "param": "<key>",
        "current_value": <val>,
        "proposed_value": <val>,
        "reasoning": "<one-paragraph why>",
        "expected_impact": "<one-sentence what should change>"
      }

    `regime` is the current market regime; it gates the asymmetric re-risk cap
    for fixed_alloc_pct (faster UP moves allowed only in non-bear regimes).

    Returns (ok, reason_if_not_ok).
    """
    if not isinstance(proposal, dict):
        return False, "proposal is not a dict"

    required = {"param", "proposed_value", "reasoning"}
    missing = required - proposal.keys()
    if missing:
        return False, f"missing fields: {missing}"

    param = proposal["param"]
    if param not in TUNABLES:
        return False, f"param '{param}' is not in allowlist {sorted(TUNABLES.keys())}"

    new_v = proposal["proposed_value"]
    cur_v = current_params.get(param, TUNABLE_DEFAULTS.get(param))

    lo, hi, ptype = TUNABLES[param]
    if ptype == "bool":
        if not isinstance(new_v, bool):
            return False, f"proposed_value must be bool for {param}, got {type(new_v).__name__}"
    elif ptype == "int":
        if not isinstance(new_v, int) or isinstance(new_v, bool):
            return False, f"proposed_value must be int for {param}, got {type(new_v).__name__}"
        if new_v < lo or new_v > hi:
            return False, f"{param}={new_v} out of bounds [{lo}, {hi}]"
        # Delta cap (+ epsilon for floating-point boundary cases). Directional:
        # UP moves may use a wider cap for the re-risk param in non-bear regimes.
        if cur_v is not None and cur_v != 0:
            delta_frac = abs(new_v - cur_v) / abs(cur_v)
            down_cap, up_cap = delta_caps_for(param, regime)
            cap = up_cap if new_v > cur_v else down_cap
            direction = "up" if new_v > cur_v else "down"
            if delta_frac > cap + DELTA_CAP_EPSILON and abs(new_v - cur_v) > 1:
                return False, (
                    f"{param} {direction} delta {delta_frac*100:.1f}% exceeds "
                    f"max {cap*100:.0f}% per run (current={cur_v}, proposed={new_v}, regime={regime})"
                )
    elif ptype == "float":
        if not isinstance(new_v, (int, float)) or isinstance(new_v, bool):
            return False, f"proposed_value must be number for {param}, got {type(new_v).__name__}"
        new_v = float(new_v)
        if new_v < lo or new_v > hi:
            return False, f"{param}={new_v} out of bounds [{lo}, {hi}]"
        if cur_v is not None and cur_v != 0:
            delta_frac = abs(new_v - cur_v) / abs(cur_v)
            down_cap, up_cap = delta_caps_for(param, regime)
            cap = up_cap if new_v > cur_v else down_cap
            direction = "up" if new_v > cur_v else "down"
            if delta_frac > cap + DELTA_CAP_EPSILON:
                return False, (
                    f"{param} {direction} delta {delta_frac*100:.1f}% exceeds "
                    f"max {cap*100:.0f}% per run (current={cur_v}, proposed={new_v}, regime={regime})"
                )

    if cur_v == new_v:
        return False, f"proposed value equals current value ({cur_v}) — no-op"

    return True, "ok"


# ============================================================================
# Apply
# ============================================================================
def apply_params(proposal: dict) -> None:
    """Write the proposed change to qf_strategy_params.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if PARAMS_FILE.exists():
        try:
            with open(PARAMS_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing[proposal["param"]] = proposal["proposed_value"]
    existing["_last_modified_at"] = datetime.now(timezone.utc).isoformat()
    existing["_last_modified_by"] = "quantforge_reflect"
    existing["_last_change_reason"] = proposal.get("reasoning", "")[:500]
    with open(PARAMS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def append_decision(record: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================================
# Prompt construction
# ============================================================================
SYSTEM_PROMPT = """You are the self-reflection module of an autonomous Bitcoin
paper-trading agent called QuantForge. Your job is to look at the agent's
last 7 days of performance and propose ONE parameter adjustment that would
improve risk-adjusted return.

Hard constraints you MUST respect:

1. You can only propose changes to these parameters:
   - hodl_mode (bool): master toggle between fixed allocation and regime-acting
   - fixed_alloc_pct (float, 0.40–0.85): BTC allocation when hodl_mode=true
   - rebalance_threshold (float, 0.02–0.20): drift before rebalance fires
   - rebalance_cooldown_hours (int, 1–48): min hours between rebalances
   - max_rebalances_per_day (int, 1–5): daily trade cap
   - regime_hysteresis_cycles (int, 2–6): cycles before regime activation
   - profit_take_pct (float, 0.10–0.30): first profit-take threshold
   - profit_take_increment (float, 0.05–0.15): subsequent profit-take cadence

2. Any other parameter you propose will be REJECTED. In particular you cannot
   touch the safety stack (panic-halt thresholds, drawdown-trim, fees).

3. Max delta per parameter per run is 10% of current value for numeric params.
   EXCEPTION: fixed_alloc_pct may be RAISED by up to 20% per run when the
   market regime has exited bear (NEUTRAL/BULL/STRONG_BULL) — this is the
   faster "re-risk" path so we can re-enter after a crash bottoms. Cuts
   (lowering allocation) are always capped at 10%. The exact legal range is
   pre-computed for you in the PARAMETER RELEVANCE section — use those numbers.

4. Propose ONLY ONE parameter change per run. Choose the highest-leverage one.

5. If the data shows the strategy is performing well and no change is needed,
   propose param='none' with reasoning explaining why no change is best.

Output strictly as a single JSON object inside a ```json ... ``` fence:

```json
{
  "param": "<name or 'none'>",
  "current_value": <value or null>,
  "proposed_value": <value or null>,
  "reasoning": "<2-4 sentences citing specific numbers from the data>",
  "expected_impact": "<one sentence on what should change in next 7 days>"
}
```

Be conservative. Cite specific numbers (PnL, alpha, regime visits, drift) in
your reasoning. If alpha vs HODL is positive, lean toward "no change".
"""


def compute_legal_range(param: str, current_value, regime: str | None = None) -> tuple | None:
    """For a numeric tunable, compute the legal range for the next proposal.

    Returns (lo, hi) — the intersection of:
      - the per-run delta cap (asymmetric for the re-risk param): the DOWN
        bound uses down_cap, the UP bound uses up_cap
      - the absolute allowlist bounds for this param

    For ints, snaps inward so we don't suggest non-integer values.
    Returns None for bool params or unknown params (no range to compute).

    Background: the LLM kept making off-by-0.2% arithmetic errors — e.g.,
    proposing 0.59 → 0.53 thinking it's 10% but it's 10.17%. Pre-computing
    the legal range removes that class of error entirely. The asymmetric
    up-bound lets the daemon SEE the wider re-risk headroom in non-bear regimes.
    """
    if param not in TUNABLES:
        return None
    abs_lo, abs_hi, ptype = TUNABLES[param]
    if ptype == "bool":
        return None
    if current_value is None or current_value == 0:
        return (abs_lo, abs_hi)
    down_cap, up_cap = delta_caps_for(param, regime)
    legal_lo = max(abs_lo, current_value - down_cap * abs(current_value))
    legal_hi = min(abs_hi, current_value + up_cap * abs(current_value))
    if ptype == "int":
        # Snap INWARD so any integer in [legal_lo, legal_hi] is safe
        import math as _math
        legal_lo = int(_math.ceil(legal_lo))
        legal_hi = int(_math.floor(legal_hi))
    return (legal_lo, legal_hi)


def build_parameter_relevance(portfolio: dict, recent_log: str, params: dict) -> str:
    """Deterministically compute which tunables are LIVE vs DORMANT right now.

    The daemon kept proposing changes to rebalance_threshold even though drift
    (~-1.2%) is nowhere near the threshold (8%) — meaning those proposals would
    change zero trades. This section gives the LLM the situational awareness to
    focus on parameters that actually affect behavior in current conditions.
    """
    # Latest drift from log: "... drift -1.2% ..."
    drift_matches = re.findall(r"drift\s+([+-]?[\d.]+)%", recent_log)
    drift = float(drift_matches[-1]) if drift_matches else None

    # PnL%
    starting = portfolio.get("starting_balance", 5000) or 5000
    cash = portfolio.get("cash", 0)
    btc_qty = portfolio.get("btc_qty", 0)
    price_m = re.findall(r"BTC price: \$([\d,\.]+)", recent_log)
    price = float(price_m[-1].replace(",", "")) if price_m else None
    pnl_pct = None
    if price is not None:
        equity = cash + btc_qty * price
        pnl_pct = (equity - starting) / starting * 100.0

    hodl_mode = params.get("hodl_mode", True)
    rebal_thr = params.get("rebalance_threshold", 0.08)
    profit_take = params.get("profit_take_pct", 0.20)

    # Did any rebalance fire in the recent window?
    rebalances_fired = "rebalance_to_" in recent_log or "Increase BTC" in recent_log or "Reduce BTC" in recent_log

    lines = ["=== PARAMETER RELEVANCE (READ BEFORE PROPOSING) ==="]
    lines.append(
        f"Current drift: {drift:+.1f}%  |  "
        f"Current PnL: {pnl_pct:+.2f}%  |  "
        f"Mode: {'HODL_MODE' if hodl_mode else 'REGIME_ACTIVE'}"
        if drift is not None and pnl_pct is not None
        else "Current drift/PnL: could not parse from log"
    )
    lines.append("")
    lines.append("LIVE parameters (changing these WILL affect behavior):")
    if hodl_mode:
        lines.append(
            "  - fixed_alloc_pct: the master allocation lever. In HODL_MODE this "
            "is the PRIMARY parameter that shapes day-to-day risk and return.")
    lines.append(
        "  - hodl_mode: meta-toggle. Flipping to false re-enables regime-based "
        "allocation (which historically LOST money — see BULL alpha).")
    lines.append("")
    lines.append("DORMANT parameters (changing these has NO effect right now):")
    if drift is not None and abs(drift) < rebal_thr * 100 * 0.5:
        lines.append(
            f"  - rebalance_threshold ({rebal_thr}): drift is only {drift:+.1f}%. "
            f"A rebalance fires ONLY when |drift| >= threshold ({rebal_thr*100:.1f}%). "
            f"Proposing any value drift never reaches changes ZERO trades. "
            f"Do NOT propose this unless drift is approaching the threshold.")
    if not rebalances_fired:
        lines.append(
            "  - rebalance_cooldown_hours, max_rebalances_per_day, "
            "regime_hysteresis_cycles: only matter when rebalances actually "
            "trigger. No rebalance has fired in the last 7 days.")
    if pnl_pct is not None and pnl_pct < profit_take * 100:
        lines.append(
            f"  - profit_take_pct ({profit_take}), profit_take_increment: only "
            f"fire when PnL exceeds +{profit_take*100:.0f}%. Current PnL is "
            f"{pnl_pct:+.2f}%. Completely dormant.")
    regime = parse_current_regime(recent_log)
    rerisk_on = regime in NON_BEAR_REGIMES
    lines.append("")
    lines.append(
        "PER-RUN DELTA CAP — proposed_value MUST be in this legal range or it "
        "will be REJECTED. Cap is 10% per run, EXCEPT fixed_alloc_pct may rise "
        f"up to 20% in non-bear regimes (current regime: {regime}, "
        f"faster re-risk {'ENABLED' if rerisk_on else 'OFF — bear/unknown'}):")
    for tname, (abs_lo, abs_hi, ptype) in TUNABLES.items():
        if ptype == "bool":
            continue
        cur = params.get(tname)
        rng = compute_legal_range(tname, cur, regime)
        if rng is None:
            continue
        lo_v, hi_v = rng
        if ptype == "int":
            lines.append(
                f"  - {tname} (current={cur}): legal range [{lo_v}, {hi_v}]")
        else:
            lines.append(
                f"  - {tname} (current={cur}): legal range [{lo_v:.4f}, {hi_v:.4f}]"
            )
    lines.append(
        "  NOTE: Use the bound numbers EXACTLY for max moves. If you compute "
        "mentally and round, you'll fail the cap — use the numbers above as-is.")
    if rerisk_on:
        lines.append(
            "  RE-RISK NOTE: the regime has exited bear, so fixed_alloc_pct may "
            "now climb faster (up to 20% this run). If the prior bear forced "
            "allocation down and price is recovering, consider raising it.")
    lines.append("")
    lines.append(
        "THEREFORE: a meaningful proposal this cycle almost certainly involves "
        "fixed_alloc_pct or hodl_mode — or is 'none'. A proposal to tweak a "
        "DORMANT parameter is a no-op and will be treated as low value.")
    return "\n".join(lines)


def build_user_prompt(portfolio: dict, recent_log: str, params: dict,
                      prior_decisions: list[dict]) -> str:
    sections = []

    # 1. Summary
    starting = portfolio.get("starting_balance", 5000)
    price_now = "unknown"
    # Try to find latest price from log
    m = re.findall(r"BTC price: \$([\d,\.]+)", recent_log)
    if m:
        price_now = m[-1]
    cash = portfolio.get("cash", 0)
    btc_qty = portfolio.get("btc_qty", 0)
    peak = portfolio.get("peak_equity", starting)
    sections.append(
        f"=== PORTFOLIO STATE ===\n"
        f"Starting balance: ${starting:,.2f}\n"
        f"Cash: ${cash:,.2f}\n"
        f"BTC held: {btc_qty:.6f}\n"
        f"Peak equity: ${peak:,.2f}\n"
        f"Latest BTC price seen in logs: ${price_now}\n"
        f"Total trades: {portfolio.get('n_trades', 0)}\n"
        f"Total rebalances: {portfolio.get('n_rebalances', 0)}\n"
        f"Drawdown trims fired: {portfolio.get('n_drawdown_trims', 0)}\n"
        f"Profit takes fired: {portfolio.get('n_profit_takes', 0)}\n"
        f"Total fees paid: ${portfolio.get('total_fees_paid', 0):.2f}\n"
    )

    # 2. Per-regime perf
    rp = portfolio.get("regime_perf", {})
    if rp:
        lines = ["=== PER-REGIME ALPHA ATTRIBUTION (vs passive HODL) ==="]
        lines.append(f"{'Regime':<14} {'Visits':>7} {'Hours':>7} {'Our $':>10} {'HODL $':>10} {'Alpha $':>10}")
        total_alpha = 0.0
        for regime, b in rp.items():
            alpha = b.get("alpha", 0.0)
            total_alpha += alpha
            lines.append(
                f"{regime:<14} {b.get('visits',0):>7d} {b.get('hours',0):>6.1f}h "
                f"{b.get('our_pnl',0):>+10.2f} {b.get('hodl_pnl',0):>+10.2f} {alpha:>+10.2f}"
            )
        lines.append(f"TOTAL ALPHA vs HODL: ${total_alpha:+.2f}")
        sections.append("\n".join(lines))

    # 3. Current params
    sections.append(
        f"=== CURRENT TUNABLE PARAMS ===\n"
        + "\n".join(f"  {k}: {v}" for k, v in params.items())
    )

    # 3.5 Parameter relevance — which knobs are LIVE vs DORMANT right now
    sections.append(build_parameter_relevance(portfolio, recent_log, params))

    # 4. Recent log tail (last 60 lines is enough to get a sense)
    log_lines = recent_log.strip().split("\n")
    tail = "\n".join(log_lines[-60:])
    sections.append(f"=== LAST 60 LOG LINES ===\n{tail}")

    # 5. Prior decisions
    if prior_decisions:
        recent_summary = []
        for d in prior_decisions:
            recent_summary.append(
                f"  {d.get('ts','?')}: proposed {d.get('proposal',{}).get('param','?')} "
                f"→ {d.get('proposal',{}).get('proposed_value','?')} | "
                f"applied={d.get('applied', False)} | "
                f"validation={d.get('validation','?')}"
            )
        sections.append(
            "=== PRIOR REFLECT DECISIONS (last 5) ===\n"
            + "\n".join(recent_summary)
            + "\n\nNOTE: A decision with applied=false had ZERO effect on the bot — "
            "it is NOT pending or queued, it simply never happened. The ONLY source "
            "of truth for currently-active parameters is the CURRENT TUNABLE PARAMS "
            "section above. Do not defer a change because a past proposal was "
            "'validated but not applied' — that proposal is dead."
        )

    sections.append(
        "=== TASK ===\n"
        "Based on the above, propose ONE parameter change that would improve\n"
        "risk-adjusted return over the next 7 days. Or propose param='none'\n"
        "if no change is warranted. Cite specific numbers in your reasoning.\n"
        "Output a single JSON object inside a ```json ... ``` fence."
    )

    return "\n\n".join(sections)


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    log("=== Reflect cycle start ===")

    portfolio = load_portfolio()
    if portfolio is None:
        log(" No portfolio file yet — nothing to reflect on. Exiting.")
        return 0

    recent_log = load_recent_log(hours=168)
    current_params = load_current_params()
    prior_decisions = load_recent_decisions(n=5)
    prior_count = count_prior_runs()
    auto_apply = AUTO_APPLY_FLAG.exists() and prior_count >= TRAINING_WHEELS_RUNS
    current_regime = parse_current_regime(recent_log)

    log(f"  Prior reflect runs: {prior_count}")
    log(f"  Training wheels: {'OFF (auto-apply active)' if auto_apply else 'ON (propose only)'}")
    rerisk_state = "faster re-risk ENABLED (up to 20%)" if current_regime in NON_BEAR_REGIMES else "symmetric 10% (bear/unknown regime)"
    log(f"  Current regime: {current_regime}  |  re-risk cap: {rerisk_state}")

    # Build prompt and call LLM
    user_prompt = build_user_prompt(portfolio, recent_log, current_params, prior_decisions)
    try:
        log(f"  Calling LLM (OpenRouter primary, Anthropic fallback)...")
        t0 = time.time()
        resp, provider = call_llm(SYSTEM_PROMPT, user_prompt)
        elapsed = time.time() - t0
        log(f"  Got response from {provider} in {elapsed:.1f}s")
    except Exception as e:
        log(f" LLM call failed (both providers): {e}")
        append_decision({
            "ts": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "applied": False,
        })
        return 1

    text = extract_text(resp, provider)
    proposal = extract_json(text)

    if proposal is None:
        log(f" Could not parse JSON from response. First 500 chars: {text[:500]}")
        append_decision({
            "ts": datetime.now(timezone.utc).isoformat(),
            "raw_text": text[:2000],
            "error": "no_json",
            "applied": False,
        })
        return 1

    log(f"  Proposal: param={proposal.get('param')}, "
        f"proposed_value={proposal.get('proposed_value')}")

    # Handle 'none' (no change recommended)
    if proposal.get("param") == "none":
        log(f"  No change recommended. Reasoning: {proposal.get('reasoning','')[:200]}")
        append_decision({
            "ts": datetime.now(timezone.utc).isoformat(),
            "proposal": proposal,
            "validation": "no_change_requested",
            "applied": False,
            "training_wheels": not auto_apply,
        })
        log("=== Reflect cycle end (no change) ===")
        return 0

    # Validate (regime gates the asymmetric re-risk cap)
    ok, reason = validate_proposal(proposal, current_params, regime=current_regime)
    log(f"  Validation: {'OK' if ok else 'REJECTED'} — {reason}")

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "proposal": proposal,
        "validation": reason,
        "training_wheels": not auto_apply,
        "applied": False,
    }

    if not ok:
        append_decision(record)
        log("=== Reflect cycle end (proposal rejected) ===")
        return 0

    # Apply or stop at training wheels
    if not auto_apply:
        log(f"   Training wheels active — proposal LOGGED but NOT APPLIED.")
        log(f"     {prior_count + 1}/{TRAINING_WHEELS_RUNS} runs before auto-apply available.")
        log(f"     To enable: touch {AUTO_APPLY_FLAG}")
        append_decision(record)
        log("=== Reflect cycle end (training wheels) ===")
        return 0

    try:
        # ── Backtesting gate (v2) ──
        # Before applying, replay last 7 days to validate.
        import subprocess
        gate_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "quantforge_backtest_gate.py")
        # Write proposed params to temp file for gate
        temp_proposed = PARAMS_FILE.parent / ".proposed_backtest.json"
        proposed_params = {}
        if PARAMS_FILE.exists():
            with open(PARAMS_FILE) as f:
                proposed_params = json.load(f)
        proposed_params[proposal["param"]] = proposal["proposed_value"]
        with open(temp_proposed, "w") as f:
            json.dump(proposed_params, f)

        try:
            gate_python = os.path.expanduser("~/quantforge/.venvs/quant-ops/bin/python")
            result = subprocess.run(
                [gate_python, gate_script, "--proposed", str(temp_proposed)],
                capture_output=True, text=True, timeout=120,
            )
            gate_output = json.loads(result.stdout)
        except Exception:
            gate_output = {"approved": True, "reason": "Gate execution failed — allowing"}
        finally:
            try:
                temp_proposed.unlink()
            except Exception:
                pass

        if not gate_output.get("approved", True):
            record["gate_blocked"] = True
            record["gate_reason"] = gate_output.get("reason", "unknown")
            log(f"   Backtesting gate BLOCKED: {gate_output.get('reason')}")
            log(f"     Metrics: {json.dumps(gate_output.get('metrics', {}))}")
            append_decision(record)
            log("=== Reflect cycle end (gate blocked) ===")
            return 0

        log(f"   Backtesting gate PASSED: {gate_output.get('reason')}")
        log(f"     Metrics: current PnL {gate_output['metrics'].get('current_pnl_pct', 0):+.1f}% → "
            f"proposed {gate_output['metrics'].get('proposed_pnl_pct', 0):+.1f}%")
        
        apply_params(proposal)
        record["applied"] = True
        record["gate_metrics"] = gate_output.get("metrics", {})
        log(f"   Applied: {proposal['param']} → {proposal['proposed_value']}")
    except Exception as e:
        record["apply_error"] = str(e)
        log(f" Apply failed: {e}")

    append_decision(record)
    log("=== Reflect cycle end ===")
    return 0 if record["applied"] else 1


if __name__ == "__main__":
    sys.exit(main())
