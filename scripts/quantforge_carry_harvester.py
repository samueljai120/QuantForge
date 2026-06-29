#!/usr/bin/env python3
"""QuantForge — Funding-Carry Harvester (Phase E).

The one trading edge that survived the honest gauntlet: selectively harvesting
EXTREME funding, delta-neutral, on high-funding alts. This is the live paper
lane built on the TDD'd qf_mlops.carry_backtest.carry_decision (4.5yr-validated:
SOL +10%/yr, PEPE +5%, AVAX +2%, beating a 30-seed random control 30/30).

Design (matches the backtest that proved the edge):
  - universe: 18 cost-honest symbols (edge survives realistic 20bps — see UNIVERSE)
  - enter when |funding| >= 0.05%/8h on the collecting side; exit on
    normalization (<0.02%) or sign flip
  - realistic two-leg round-trip cost (~20 bps) charged once per episode
  - delta-neutral: price P&L assumed hedged; P&L = funding collected - cost
  - isolated paper sleeve (separate from the agent's $5k) — gathers the live
    evidence the benchmark gate needs before any real allocation

Commands:  run (one cycle, default) | status
Cron-safe via internal flock; intended to piggyback an existing cron line.
"""
import fcntl
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qf_mlops.carry_backtest import carry_decision

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
STATE_PATH = os.path.join(DATA_DIR, "carry_harvester_state.json")
LOCK_PATH = "/tmp/qf_carry_harvester.lock"

# COST-HONEST universe: every symbol here has demonstrated edge at the REALISTIC
# 20bps cost (not the optimistic 6bps backtest) — each beats a 30-seed random control
# 30/30, cost-net, on 4.5yr Binance funding (run `CARRY_COST_BPS=20 quantforge_carry_eval.py`).
# Deliberately EXCLUDES thin/new/commodity tokens whose headline funding is unrealizable
# at real cost. BTC + LINK were DROPPED 2026-06-22: no edge at 20bps (funding rarely goes
# extreme enough to clear cost). Best edge concentrates in SOL +9.1%/yr, ZEC +4.8%,
# 1000PEPE +3.7%, TRUMP/BNB +2.5%, WLD +2.3%; the rest are positive but thin (<1.5%/yr).
UNIVERSE = {
    "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT",
    "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT",
    "LTC": "LTCUSDT", "SUI": "SUIUSDT", "NEAR": "NEARUSDT",
    "FIL": "FILUSDT", "ENA": "ENAUSDT", "WLD": "WLDUSDT", "XLM": "XLMUSDT",
    "TAO": "TAOUSDT", "PEPE": "1000PEPEUSDT", "TRUMP": "TRUMPUSDT", "ZEC": "ZECUSDT",
}
ENTER_THRESH = 0.0005     # 0.05%/8h — the proven edge zone (best threshold for the universe)
EXIT_THRESH = 0.0002      # 0.02%/8h — normalized
COST_FRAC = 0.002         # ~20 bps two-leg round-trip (honest, not the 6bps optimistic backtest)
NOTIONAL = 150.0          # paper notional per position (18-symbol cost-honest universe)
MAX_CONCURRENT = 6        # cap simultaneous positions so the sleeve isn't over-allocated
SLEEVE = 1000.0           # nominal sleeve for %-reporting
FUNDING_INTERVAL_H = 8.0


def _now():
    return datetime.now(timezone.utc)


def fetch_funding(binance_sym):
    url = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=%s" % binance_sym
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "QF/1"}), timeout=15).read())
    return float(d["lastFundingRate"])


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"starting_balance": SLEEVE, "realized_pnl": 0.0, "positions": {},
            "history": [], "stats": {"trades": 0, "wins": 0, "losses": 0},
            "created_at": _now().isoformat()}


def save_state(s):
    s["updated_at"] = _now().isoformat()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, STATE_PATH)


def run():
    s = load_state()
    now = _now()
    for name, bsym in UNIVERSE.items():
        try:
            funding = fetch_funding(bsym)
        except Exception as e:
            print("  %s: funding fetch failed (%s) — skip" % (name, e))
            continue
        pos = s["positions"].get(name)
        in_pos = pos is not None
        side = pos["side"] if in_pos else 0
        action, new_side = carry_decision(funding, in_pos=in_pos, side=side,
                                          enter_thresh=ENTER_THRESH, exit_thresh=EXIT_THRESH)

        if action == "ENTER":
            if len(s["positions"]) >= MAX_CONCURRENT:
                print("  %s: ENTER signal but at max %d concurrent — skip" % (name, MAX_CONCURRENT))
                continue
            s["positions"][name] = {"side": new_side, "entry_funding": funding,
                                    "entry_ts": now.isoformat(), "last_accrue_ts": now.isoformat(),
                                    "collected": 0.0, "cost": COST_FRAC * NOTIONAL, "notional": NOTIONAL}
            print("  %s: ENTER side=%+d funding=%+.4f%% cost=$%.2f" % (name, new_side, funding * 100, COST_FRAC * NOTIONAL))

        elif action == "HOLD":
            last = datetime.fromisoformat(pos["last_accrue_ts"])
            hours = (now - last).total_seconds() / 3600.0
            accr = abs(funding) * NOTIONAL * (hours / FUNDING_INTERVAL_H)
            pos["collected"] += accr
            pos["last_accrue_ts"] = now.isoformat()
            print("  %s: HOLD side=%+d funding=%+.4f%% +$%.3f (collected $%.2f)" % (name, side, funding * 100, accr, pos["collected"]))

        elif action == "EXIT":
            last = datetime.fromisoformat(pos["last_accrue_ts"])
            hours = (now - last).total_seconds() / 3600.0
            pos["collected"] += abs(funding) * NOTIONAL * (hours / FUNDING_INTERVAL_H)
            pnl = pos["collected"] - pos["cost"]
            s["realized_pnl"] += pnl
            s["stats"]["trades"] += 1
            s["stats"]["wins" if pnl > 0 else "losses"] += 1
            s["history"].append({"symbol": name, "side": pos["side"], "pnl_usd": round(pnl, 3),
                                 "collected": round(pos["collected"], 3), "entry_ts": pos["entry_ts"],
                                 "exit_ts": now.isoformat(), "exit_funding": funding})
            del s["positions"][name]
            print("  %s: EXIT pnl=$%+.3f (collected $%.2f - cost $%.2f)" % (name, pnl, pos["collected"], pos["cost"]))

        else:
            print("  %s: FLAT funding=%+.4f%% (|f|<%.3f%%)" % (name, funding * 100, ENTER_THRESH * 100))

    save_state(s)
    ret_pct = s["realized_pnl"] / s["starting_balance"] * 100.0
    print("CARRY HARVESTER: realized $%+.2f (%+.2f%% of sleeve) | open=%d | trades=%d (W%d/L%d)"
          % (s["realized_pnl"], ret_pct, len(s["positions"]), s["stats"]["trades"], s["stats"]["wins"], s["stats"]["losses"]))


def status():
    s = load_state()
    print(json.dumps({"realized_pnl": s["realized_pnl"], "open_positions": s["positions"],
                      "stats": s["stats"], "n_history": len(s["history"])}, indent=2))


if __name__ == "__main__":
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("SKIP: carry harvester already running")
        sys.exit(0)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    (status if cmd == "status" else run)()
