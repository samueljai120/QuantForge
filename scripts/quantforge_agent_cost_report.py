#!/usr/bin/env python3
"""QuantForge AGENT lane — cost-inclusive paper-alpha report (READ-ONLY).

Produces the ONE number the ~Jul-10 real-capital decision needs: the agent
lane's realized PnL *after* the fees and funding the paper ledger omits but a
live exchange would actually charge.

Why this exists
---------------
The agent's paper ledger (agent_trades.jsonl) does NOT book exchange costs on
its leveraged (futures) fills — every FUTURES_OPEN / FUTURES_CLOSE record carries
`fee = 0.0`. It also never debits funding on held futures positions. Spot
(BUY/SELL) fills, by contrast, already have a real taker fee booked in their
`fee` field. So a live exchange would charge MORE than the paper ledger shows on
the leveraged side, and the headline paper PnL is optimistic.

This report adds back exactly those two omitted live costs:
  1. Exchange taker fee on each leveraged fill (notional * TAKER_FEE).
  2. Funding paid/received over each futures holding window, priced from an
     INDEPENDENT data source: the derivatives state history collector. We do NOT
     trust the agent's own funding view — see the funding_rate=0.0 default bug at
     quantforge_agent.py:688 (single-shot urllib, no retry; on fetch failure the
     agent's funding signal silently becomes zero).

Everything here is read-only. It never writes to or mutates any agent state file;
it only reads ledgers and writes its own report JSON.

Ledger field shapes (verified against quantforge_agent.py)
----------------------------------------------------------
SPOT    BUY  : usd = dollar_amount, fee = booked taker fee  (do NOT re-charge)
SPOT    SELL : usd = proceeds,      fee = booked taker fee  (do NOT re-charge)
FUTURES_OPEN : qty = NOTIONAL, usd = margin, fee = 0.0, has `leverage`
FUTURES_CLOSE: qty = NOTIONAL, usd = margin+pnl, fee = 0.0, pnl_usd = realized

NOTE on fee base: a real exchange charges taker fee on the *notional* of a
leveraged fill, not on the margin posted. For futures records the notional lives
in the `qty` field (usd holds margin). We therefore charge added_fee on `qty`
(notional) for leveraged fills, falling back to `usd` only if `qty` is absent.

Funding accrual model (calendar-exact)
--------------------------------------
KuCoin perpetuals charge funding every 8h at fixed UTC boundaries (00:00 /
08:00 / 16:00). For each holding window we charge funding once per 8h boundary
that falls inside (t_open, t_close], applying `notional * rate` where `rate` is
the funding rate in effect at that boundary — the most recent derivatives
sample at-or-before the boundary. A window held entirely between two boundaries
incurs zero funding (an exact zero, not an estimate).

No look-ahead: a boundary's rate is the most recent sample whose timestamp is
<= the boundary. Samples dated after a boundary (or after the window's close)
are never used to price it. If the most recent sample is staler than
FUNDING_STALE_SEC, the boundary is priced with a conservative fallback and the
window is flagged estimated.
"""

import bisect
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths — driven by QF_BASE_DIR env var via cfg
# ---------------------------------------------------------------------------
DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
TRADES_FILE = os.path.join(DATA_DIR, "agent_trades.jsonl")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "agent_portfolio.json")
DERIV_FILE = os.path.join(DATA_DIR, "derivatives", "derivatives_state_history.jsonl")
REPORT_FILE = os.path.join(DATA_DIR, "agent-cost-inclusive-report.json")

# Fee constants — mirror quantforge_agent.py (KuCoin futures).
TAKER_FEE = 0.0006

# KuCoin perpetuals charge funding every 8h at 00:00 / 08:00 / 16:00 UTC.
FUNDING_INTERVAL_SEC = 8 * 3600
# A boundary is priced from history only if the most recent derivatives sample
# is within this many seconds of it; otherwise the rate is treated as stale and
# the conservative per-event fallback is used (and the window is flagged estimated).
FUNDING_STALE_SEC = 8 * 3600
# Conservative per-event fallback funding rate (1 bp per 8h funding event).
FALLBACK_FUNDING_PER_8H = 0.0001

# Funding-default bug location, cited in caveats.
AGENT_FUNDING_BUG_REF = "quantforge_agent.py:688"

LEVERAGED_TYPES = {"FUTURES_OPEN", "FUTURES_CLOSE"}
SPOT_TYPES = {"BUY", "SELL"}


# ---------------------------------------------------------------------------
# IO helpers (stdlib only, tolerate missing/garbled lines)
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                continue
    return rows


def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (ValueError, json.JSONDecodeError, OSError):
        return default


def _parse_iso(ts):
    """Parse an ISO8601 timestamp -> unix float seconds. None on failure."""
    if not ts:
        return None
    s = str(ts).strip()
    # Python's fromisoformat handles "+00:00"; normalise a trailing Z.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# Derivatives funding lookup
# ---------------------------------------------------------------------------
def _load_btc_funding_rows(deriv_rows):
    """Return BTC funding samples as sorted [(unix_ts:int, funding_rate:float)].

    A row is the BTC perpetual if its futures_symbol is XBTUSDTM or its symbol
    starts with "BTC".
    """
    out = []
    for r in deriv_rows:
        fsym = str(r.get("futures_symbol", "") or "")
        sym = str(r.get("symbol", "") or "")
        if fsym == "XBTUSDTM" or sym.startswith("BTC"):
            ts = r.get("timestamp")
            fr = r.get("funding_rate")
            if ts is None or fr is None:
                continue
            try:
                out.append((int(ts), float(fr)))
            except (TypeError, ValueError):
                continue
    out.sort(key=lambda x: x[0])
    return out


def _funding_for_window(t_open, t_close, direction, notional, btc_funding, btc_ts):
    """Compute funding cost (USD, positive = cost paid) for one holding window.

    Funding is charged once per 8h UTC boundary (00:00/08:00/16:00) inside
    (t_open, t_close]. Each boundary is priced with `notional * rate`, where
    `rate` is the most recent derivatives sample at-or-before the boundary (NO
    look-ahead). If the most recent sample is staler than FUNDING_STALE_SEC the
    boundary uses the conservative fallback and the window is flagged estimated.
    A window spanning no boundary incurs zero funding (an exact zero).

    LONG pays when funding_rate > 0, receives when < 0; SHORT is the opposite.

    Returns (funding_cost_usd, used_real_history: bool).
    btc_ts must be the ascending list of sample timestamps matching btc_funding.
    """
    if t_open is None or t_close is None or t_close < t_open:
        return 0.0, False

    dir_sign = 1.0 if str(direction).upper() == "LONG" else -1.0

    # Enumerate 8h funding boundaries strictly after the open, up to the close.
    first = (int(t_open) // FUNDING_INTERVAL_SEC + 1) * FUNDING_INTERVAL_SEC
    boundaries = []
    b = first
    while b <= t_close:
        boundaries.append(b)
        b += FUNDING_INTERVAL_SEC

    if not boundaries:
        # Held entirely between two funding events — zero funding, exactly known.
        return 0.0, True

    rate_sum = 0.0
    any_fallback = False
    for bnd in boundaries:
        idx = bisect.bisect_right(btc_ts, bnd) - 1  # most recent sample ts <= bnd
        if idx >= 0 and (bnd - btc_ts[idx]) <= FUNDING_STALE_SEC:
            rate_sum += btc_funding[idx][1]
        else:
            rate_sum += FALLBACK_FUNDING_PER_8H  # one missing 8h event
            any_fallback = True

    cost = notional * rate_sum * dir_sign
    return cost, (not any_fallback)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_cost_report(trades, portfolio, btc_funding):
    """Pure function: given parsed inputs, return the report dict.

    Kept dependency-free (no file IO) so tests can drive it with fixtures.
    """
    starting_balance = 0.0
    if isinstance(portfolio, dict):
        try:
            starting_balance = float(portfolio.get("starting_balance", 0.0) or 0.0)
        except (TypeError, ValueError):
            starting_balance = 0.0

    # Precompute ascending sample timestamps once for boundary rate lookup.
    btc_ts = [t for (t, _) in btc_funding]

    realized_pnl = 0.0
    already_booked_spot_fees = 0.0
    added_exchange_fees = 0.0
    n_leveraged_fills = 0

    # FIFO open queues per direction for funding pairing.
    open_longs = []   # list of dicts: {t_open, notional}
    open_shorts = []

    funding_cost = 0.0
    funding_windows_real = 0
    funding_windows_estimated = 0

    for tr in trades:
        ttype = str(tr.get("type", "") or "")

        # --- realized pnl: sum pnl_usd over all close-bearing trades ---------
        # Spot SELL and FUTURES_CLOSE both carry realized pnl_usd; BUY/OPEN are 0.
        try:
            realized_pnl += float(tr.get("pnl_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass

        if ttype in SPOT_TYPES:
            # Spot fee is ALREADY booked — record it, never re-charge.
            try:
                already_booked_spot_fees += float(tr.get("fee", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            continue

        if ttype not in LEVERAGED_TYPES:
            continue  # unknown type — ignore for cost purposes

        # --- leveraged fill: add exchange taker fee on NOTIONAL -------------
        n_leveraged_fills += 1
        # Notional lives in `qty` for futures records; fall back to `usd`.
        notional = tr.get("qty", None)
        if notional is None:
            notional = tr.get("usd", 0.0)
        try:
            notional = abs(float(notional or 0.0))
        except (TypeError, ValueError):
            notional = 0.0
        added_exchange_fees += notional * TAKER_FEE

        # --- funding pairing (FIFO by direction) ---------------------------
        direction = str(tr.get("direction", "") or "").upper()
        t_evt = _parse_iso(tr.get("ts"))

        if ttype == "FUTURES_OPEN":
            entry = {"t_open": t_evt, "notional": notional}
            if direction == "LONG":
                open_longs.append(entry)
            else:
                open_shorts.append(entry)
        elif ttype == "FUTURES_CLOSE":
            queue = open_longs if direction == "LONG" else open_shorts
            if queue:
                opened = queue.pop(0)  # FIFO
                # Use the OPEN's notional (position size held) for funding base.
                base_notional = opened.get("notional", notional) or notional
                cost, used_real = _funding_for_window(
                    opened.get("t_open"), t_evt, direction, base_notional,
                    btc_funding, btc_ts
                )
                funding_cost += cost
                if used_real:
                    funding_windows_real += 1
                else:
                    funding_windows_estimated += 1
            # A CLOSE with no matching OPEN (e.g. ledger truncation) is ignored
            # for funding — we never fabricate an open time.

    # --- still-open positions: report as caveat, do NOT fabricate funding ---
    open_remaining = len(open_longs) + len(open_shorts)
    open_position_caveat = None
    if open_remaining > 0:
        notional_open = sum(e.get("notional", 0.0) for e in (open_longs + open_shorts))
        open_position_caveat = (
            f"{open_remaining} futures position(s) still OPEN at report time "
            f"(notional ~${notional_open:,.2f}); their unrealized PnL and ongoing "
            f"funding are NOT included — realized figures only."
        )

    net = realized_pnl - added_exchange_fees - funding_cost
    net_pct = (net / starting_balance * 100.0) if starting_balance else 0.0

    # coverage_ok means: we had history AND every closed window priced from it.
    derivatives_coverage_ok = (len(btc_funding) > 0) and (funding_windows_estimated == 0)

    caveats = []
    # ALWAYS cite the funding-default bug.
    caveats.append(
        f"Funding priced from independent derivatives history, NOT the agent's own "
        f"signal: the agent defaults funding_rate=0.0 on fetch failure "
        f"({AGENT_FUNDING_BUG_REF}, single-shot urllib no retry), so its trades may "
        f"have been taken on a zero-funding view. This report does not rely on that."
    )
    caveats.append(
        "Exchange taker fee charged on leveraged-fill NOTIONAL (qty field), "
        f"TAKER_FEE={TAKER_FEE}. Spot (BUY/SELL) fees are already booked in the "
        "ledger and are NOT re-charged."
    )
    caveats.append(
        "Funding charged calendar-exact: once per 8h UTC boundary "
        "(00:00/08:00/16:00) inside each holding window, priced by the most "
        "recent non-stale derivatives sample at-or-before the boundary."
    )
    if len(btc_funding) == 0:
        caveats.append(
            "No BTC derivatives history rows found — ALL funding windows used the "
            f"conservative fallback ({FALLBACK_FUNDING_PER_8H} per 8h, scaled by "
            "hours). Funding figure is an estimate, not measured."
        )
    elif funding_windows_estimated > 0:
        caveats.append(
            f"{funding_windows_estimated} funding window(s) had no covering "
            "derivatives rows and used the conservative fallback estimate."
        )
    if open_position_caveat:
        caveats.append(open_position_caveat)
    if not isinstance(portfolio, dict) or starting_balance <= 0:
        caveats.append(
            "starting_balance missing/zero in agent_portfolio.json — net pct could "
            "not be computed against capital; treated as 0."
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "realized_pnl_usd": round(realized_pnl, 2),
        "already_booked_spot_fees": round(already_booked_spot_fees, 2),
        "added_exchange_fees": round(added_exchange_fees, 2),
        "funding_cost": round(funding_cost, 2),
        "net_cost_inclusive_pnl": round(net, 2),
        "net_cost_inclusive_pct": round(net_pct, 4),
        "n_leveraged_fills": n_leveraged_fills,
        "funding_windows_real": funding_windows_real,
        "funding_windows_estimated": funding_windows_estimated,
        "derivatives_coverage_ok": bool(derivatives_coverage_ok),
        "open_position_caveat": open_position_caveat,
        "caveats": caveats,
        # Extra context (not in the required schema but harmless / auditable):
        "starting_balance": round(starting_balance, 2),
        "btc_funding_rows_available": len(btc_funding),
    }


# ---------------------------------------------------------------------------
# Report driver (does the file IO, then computes)
# ---------------------------------------------------------------------------
def run_report(write=True):
    """Load files, compute the report, optionally write it, return (report, status).

    status: "ok" | "no_trades" | "missing_trades_file"
    Always returns a report dict (may be a minimal one if inputs are missing).
    """
    if not os.path.exists(TRADES_FILE):
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "missing_trades_file",
            "trades_file": TRADES_FILE,
            "caveats": [
                f"agent_trades.jsonl not found at {TRADES_FILE}; nothing to report. "
                f"Funding-default note still applies ({AGENT_FUNDING_BUG_REF})."
            ],
        }
        if write:
            _safe_write(report)
        return report, "missing_trades_file"

    trades = _read_jsonl(TRADES_FILE)
    portfolio = _read_json(PORTFOLIO_FILE, default={})
    deriv_rows = _read_jsonl(DERIV_FILE)
    btc_funding = _load_btc_funding_rows(deriv_rows)

    report = compute_cost_report(trades, portfolio, btc_funding)
    report["status"] = "ok" if trades else "no_trades"
    report["trades_count"] = len(trades)

    if write:
        _safe_write(report)
    return report, report["status"]


def _safe_write(report):
    try:
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        with open(REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
    except OSError as e:
        print(f"WARN: could not write report to {REPORT_FILE}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------
def print_summary(report):
    status = report.get("status", "ok")
    if status == "missing_trades_file":
        print("QuantForge AGENT cost-inclusive report")
        print("=" * 60)
        print(f"  STATUS: agent_trades.jsonl missing ({report.get('trades_file')})")
        print("  Nothing to report. (Run on the production host where the ledger lives.)")
        return

    print("QuantForge AGENT lane — COST-INCLUSIVE paper-alpha report")
    print("=" * 60)
    print(f"  generated_at            : {report.get('generated_at')}")
    print(f"  status                  : {status} ({report.get('trades_count', 0)} ledger lines)")
    print(f"  starting_balance        : ${report.get('starting_balance', 0):,.2f}")
    print("-" * 60)
    print(f"  realized PnL (paper)    : ${report.get('realized_pnl_usd', 0):+,.2f}")
    print(f"  spot fees (already kept): ${report.get('already_booked_spot_fees', 0):,.2f}  (NOT re-charged)")
    print(f"  - added exchange fees   : -${report.get('added_exchange_fees', 0):,.2f}  "
          f"({report.get('n_leveraged_fills', 0)} leveraged fills @ {TAKER_FEE})")
    _fc = report.get('funding_cost', 0.0)
    _fc_note = "credit received, raises net" if _fc < 0 else "drag paid, lowers net"
    print(f"  funding (signed cost)   : ${_fc:+,.2f}  "
          f"({_fc_note}; real:{report.get('funding_windows_real', 0)} "
          f"est:{report.get('funding_windows_estimated', 0)})")
    print(f"  derivatives coverage ok : {report.get('derivatives_coverage_ok')}  "
          f"({report.get('btc_funding_rows_available', 0)} BTC rows)")
    if report.get("open_position_caveat"):
        print(f"  OPEN-POSITION CAVEAT    : {report['open_position_caveat']}")
    print("-" * 60)
    print("  Caveats:")
    for c in report.get("caveats", []):
        print(f"    - {c}")
    print("=" * 60)
    net = report.get("net_cost_inclusive_pnl", 0.0)
    pct = report.get("net_cost_inclusive_pct", 0.0)
    added = report.get("added_exchange_fees", 0.0)
    fund = report.get("funding_cost", 0.0)
    # ONE headline line. `fund` is a signed cost (positive = drag, negative =
    # credit received); render it unambiguously with the right word + sign.
    fund_str = (f"+${-fund:,.2f} funding credit" if fund < 0
                else f"-${fund:,.2f} funding cost")
    print(
        f"COST-INCLUSIVE NET: ${net:+,.2f} ({pct:+.2f}%) "
        f"— vs paper: -${added:,.2f} fees, {fund_str}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv):
    cmd = argv[1] if len(argv) > 1 else "report"
    if cmd not in ("report",):
        print(f"usage: {os.path.basename(argv[0])} [report]")
        print("  report  (default) — compute cost-inclusive agent-lane PnL and write JSON")
        return 0

    report, status = run_report(write=True)
    print_summary(report)
    if status in ("ok", "no_trades", "missing_trades_file"):
        print(f"\n(report written to {REPORT_FILE})" if os.path.exists(REPORT_FILE) else "")
    return 0  # tolerate missing files gracefully — always exit 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
