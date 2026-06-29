#!/usr/bin/env python3
"""Stage 1a — cross-venue steady-state carry PROBE (fail-fast characterization).

The ONE carry idea with a bigger ceiling than the live selective harvester is
*continuously-held* cross-venue funding-spread carry (long the persistently-low-funding
venue's perp, short the high one), strongest CEX-vs-Hyperliquid. Before writing the full
~500-LOC integrated backtest (Stage 1b), this probe answers the cheapest decisive question:

    Does an OPTIMISTIC UPPER BOUND of that carry even clear the ~40bps round-trip cost?

Upper bound = collect |spread| EVERY 8h interval (perfect positioning), pay one round-trip
cost PER same-sign hold (the minimum: enter once, exit once, rebalance only on a sign flip).
If even this best case is net-negative, the idea is dead — KILL for ~100 LOC. If positive,
it is NOT a profit claim, only "not definitively dead" -> escalate to the rigorous Stage 1b
(lagged positioning + 30-seed random control + HAC significance).

HL funding is hourly; Binance settles 8h. We sum HL's hourly rates into Binance's 8h
buckets so both express "funding collected per 8h", then take the signed spread HL-Binance.

PROBE ONLY: no live capital, no cron, no agent change. Pure measurement.
"""
import bisect
import json
import os
import sys
import urllib.request

EIGHT_H_MS = 8 * 3_600_000
# ~40 bps: a cross-venue round-trip is 4 legs (open long-low + short-high, then close both)
# AND margin posted on both venues — double the single-venue harvester's ~20 bps.
COST = float(os.environ.get("XVENUE_COST_BPS", "40")) / 1e4
START_MS = int(os.environ.get("XVENUE_START_MS", "1656633600000"))  # 2022-07-01 (~3yr, HL depth)


# ── pure logic (TDD'd) ───────────────────────────────────────────────────────
def align_8h(binance, hl, min_coverage=6):
    """Sum hourly HL funding into Binance's 8h settlement buckets.

    binance: [(ts_ms, rate)] at 8h settlements; hl: [(ts_ms, rate)] hourly.
    For each Binance settlement T, window (T-8h, T] sums the HL hourly rates inside it.
    Buckets with < min_coverage hourly samples (data gaps) are dropped, not guessed.
    Returns [(T, binance_rate, hl_8h_sum)] aligned and chronological.
    """
    hl = sorted(hl)
    hl_ts = [t for t, _ in hl]
    hl_rate = [r for _, r in hl]
    out = []
    for T, b in sorted(binance):
        lo = bisect.bisect_right(hl_ts, T - EIGHT_H_MS)  # first HL ts > T-8h
        hi = bisect.bisect_right(hl_ts, T)               # first HL ts > T
        if hi - lo >= min_coverage:
            out.append((T, b, sum(hl_rate[lo:hi])))
    return out


def spread_of(aligned):
    """Signed cross-venue spread per interval: HL_8h - Binance (what a steady carry collects)."""
    return [hl8 - b for _, b, hl8 in aligned]


def _sign(x):
    return (x > 0) - (x < 0)


def count_sign_flips(spread):
    flips, prev = 0, 0
    for x in spread:
        s = _sign(x)
        if s == 0:
            continue
        if prev != 0 and s != prev:
            flips += 1
        prev = s
    return flips


def sign_run_lengths(seq):
    """Lengths of consecutive same-sign runs (zeros ignored). [1,1,-1,-1,-1,1] -> [2,3,1]."""
    runs, prev = [], None
    for x in seq:
        s = _sign(x)
        if s == 0:
            continue
        if s == prev:
            runs[-1] += 1
        else:
            runs.append(1)
            prev = s
    return runs


def upper_bound_carry_net(spread, cost_frac):
    """Optimistic best case: collect |spread| every interval, pay ONE round-trip per
    same-sign hold (one entry+exit; a flip starts a new hold). gross - n_holds*cost."""
    if not any(_sign(x) for x in spread):
        return 0.0
    gross = sum(abs(x) for x in spread)
    n_holds = len(sign_run_lengths(spread))
    return gross - n_holds * cost_frac


MIN_INTERVALS = 1000   # ~2.2yr of 8h data before a positive upper bound is trustworthy


def summarize(spread, cost_frac, symbol, days=None):
    n = len(spread)
    gross = sum(abs(x) for x in spread)
    runs = sign_run_lengths(spread)
    net = upper_bound_carry_net(spread, cost_frac)
    mean_abs = gross / n if n else 0.0
    # KILL = best case still loses to cost -> structurally dead (sample size irrelevant).
    # GO   = positive upper bound on ENOUGH history -> not dead -> run Stage 1b.
    # THIN = positive but too little history to trust -> not a GO; revisit with more data.
    if net <= 0:
        verdict = "KILL"
    elif n >= MIN_INTERVALS:
        verdict = "GO"
    else:
        verdict = "THIN"
    out = {
        "symbol": symbol, "n": n,
        "mean_abs_bps": mean_abs * 1e4,
        "gross_pct": gross * 100,
        "net_pct": net * 100,
        "flips": count_sign_flips(spread),
        "n_holds": len(runs),
        "median_run": sorted(runs)[len(runs) // 2] if runs else 0,
        "breakeven_run": (cost_frac / mean_abs) if mean_abs else float("inf"),
        "verdict": verdict,
    }
    if days:
        out["ann_net_pct"] = net * 100 * 365 / days
    return out


# ── impure fetch shell (not unit-tested; exercised by the live production run) ──
def _get(url, data=None):
    req = urllib.request.Request(
        url, data=data, headers={"User-Agent": "QF/1", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=25).read())


def fetch_binance_8h(sym, start_ms=START_MS):
    """[(ts_ms, rate)] Binance 8h funding, paginated forward."""
    out, start = {}, start_ms
    for _ in range(80):
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={start}&limit=1000"
        try:
            d = _get(url)
        except Exception:
            break
        if not d:
            break
        for x in d:
            out[int(x["fundingTime"])] = float(x["fundingRate"])
        nt = int(d[-1]["fundingTime"])
        if nt <= start or len(d) < 200:
            break
        start = nt + 1
    return sorted(out.items())


def fetch_hl_hourly(coin, start_ms=START_MS):
    """[(ts_ms, rate)] Hyperliquid hourly funding via POST /info fundingHistory (paginated)."""
    out, start = {}, start_ms
    for _ in range(400):  # 500/page hourly -> ~3yr needs many pages
        body = json.dumps({"type": "fundingHistory", "coin": coin, "startTime": start}).encode()
        try:
            d = _get("https://api.hyperliquid.xyz/info", data=body)
        except Exception:
            break
        if not d:
            break
        for x in d:
            out[int(x["time"])] = float(x["fundingRate"])
        nt = max(int(x["time"]) for x in d)
        if nt <= start or len(d) < 2:
            break
        start = nt + 1
    return sorted(out.items())


def probe(symbol, binance_sym=None, hl_coin=None):
    binance_sym = binance_sym or (symbol + "USDT")
    hl_coin = hl_coin or symbol
    b = fetch_binance_8h(binance_sym)
    hl = fetch_hl_hourly(hl_coin)
    if len(b) < 100 or len(hl) < 800:
        return {"symbol": symbol, "verdict": "NO_DATA",
                "binance_rows": len(b), "hl_rows": len(hl)}
    aligned = align_8h(b, hl)
    spread = spread_of(aligned)
    days = len(spread) * 8 / 24
    return summarize(spread, COST, symbol, days=days)


def main():
    syms = sys.argv[1:] or ["SOL", "ZEC", "AVAX", "DOGE"]
    print(f"X-venue steady-state carry PROBE  (cost={COST*1e4:.0f} bps round-trip, HL hourly vs Binance 8h)")
    print("Upper bound: collect |spread| every 8h, one round-trip per same-sign hold.\n")
    gos, thins = [], []
    for s in syms:
        try:
            r = probe(s)
        except Exception as e:
            print(f"  {s:6s} ERROR {e}"); continue
        if r.get("verdict") == "NO_DATA":
            print(f"  {s:6s} NO_DATA (binance={r['binance_rows']} hl={r['hl_rows']})"); continue
        if r["verdict"] == "GO":
            gos.append(s)
        elif r["verdict"] == "THIN":
            thins.append(s)
        print(f"  {s:6s} n={r['n']:5d}  mean|spread|={r['mean_abs_bps']:.2f}bps/8h  "
              f"holds={r['n_holds']:4d}  median_run={r['median_run']:3d}  breakeven_run={r['breakeven_run']:.0f}  "
              f"UB net={r.get('ann_net_pct', r['net_pct']):+.1f}%/yr  -> {r['verdict']}")
    print()
    print("=" * 72)
    if gos:
        print(f"VERDICT: GO ({', '.join(gos)}) -> run Stage 1b rigorous backtest")
    else:
        msg = "KILL -> even the optimistic upper bound loses to cost on adequate history; " \
              "the cross-venue carry has no bigger ceiling here."
        if thins:
            msg += f" (THIN/insufficient-history: {', '.join(thins)} — revisit only if HL accrues more data.)"
        print("VERDICT:", msg)
    print("=" * 72)


if __name__ == "__main__":
    main()
