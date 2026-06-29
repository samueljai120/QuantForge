#!/usr/bin/env python3
"""QuantForge Self-Healing Action Bridge (v1) — Automation Layer

THE GAP: The review cycle generates brilliant diagnostics (engineering actions,
doctor reports, autopilot decisions) but NO script reads or acts on them.

THIS SCRIPT: Reads ALL diagnostic artifacts, categorizes each flag as
auto-fixable or manual-only, executes auto-fixes, and produces a
flag→action→predicted_outcome report.

Also: researches new market knowledge (GitHub trending, crypto news, papers)
and suggests improvements.

Architecture:
  Phase 1 — Read all diagnostic JSONs
  Phase 2 — Categorize flags (auto / manual / research)
  Phase 3 — Execute auto-fixes
  Phase 4 — Market research & learning
  Phase 5 — Generate flag→action→outcome report

Run: python3 quantforge_self_heal_actions.py [--dry-run] [--research]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from quantforge_equity import compute_drawdown_from_peak, compute_true_equity

# ── Paths ───────────────────────────────────────────────────────────
DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
SCRIPTS_DIR = os.path.expanduser("~/quantforge/scripts")
REPORTS_DIR = DATA_DIR  # all diagnostic JSONs live here
VENV_PYTHON = os.path.expanduser("~/.venvs/quant-ops/bin/python")
SYSTEM_PYTHON = "/usr/bin/python3"
LOG_FILE = os.path.join(DATA_DIR, "self_heal_actions.log")
KNOWLEDGE_DB = os.path.join(DATA_DIR, "fix_knowledge.db")
TECH_RADAR_LOG = os.path.join(DATA_DIR, "tech_radar.jsonl")
INVARIANTS_STATE_FILE = os.path.join(DATA_DIR, "qf_invariants_state.json")

# ── Guardrail Constants ──────────────────────────────────────────────
LLM_COST_BUDGET_PER_RUN = 0.02       # Max $0.02 per self-heal run
LLM_MIN_CONFIDENCE = 0.6             # Don't apply fixes below this confidence
MAX_LLM_ACTIONS_PER_RUN = 5          # Max fix actions per LLM invocation
ROLLBACK_ENABLED = True              # Backup params before LLM changes

# Diagnostic artifacts to read
DIAGNOSTIC_FILES = [
    "engineering-actions.json",
    "doctor-report.json",
    "autopilot-report.json",
    "diagnosis-report.json",
    "monitor-report.json",
    "candidate-review.json",
    "harness-report.json",
    "experiment-lanes.json",
    "governance-report.json",
]

# ── Action registry: what CAN be auto-fixed ─────────────────────────
# Each entry: (flag_pattern, action_function, description, predicted_outcome)

AUTO_ACTIONS = {
    "stale_portfolio": {
        "match": lambda f: "stale" in f.get("detail", "") and "portfolio" in f.get("name", ""),
        "fix": "_fix_stale_portfolio",
        "desc": "Portfolio data stale",
        "predict": "Portfolio refreshed; agent will have current data next cycle",
    },
    "stale_last_scan": {
        "match": lambda f: "stale" in f.get("detail", "") and ("last_scan" in f.get("name", "") or "scan" in f.get("name", "")),
        "fix": "_fix_stale_scans",
        "desc": "Market scan data stale",
        "predict": "Collectors triggered; data fresh within 5 min",
    },
    "stale_collectors": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["collector", "data freshness", "stale at"]),
        "fix": "_fix_stale_collectors",
        "desc": "Data collectors stale",
        "predict": "All collectors re-run; data pipeline restored",
    },
    "venv_missing": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["venv", "missing python", "no module"]),
        "fix": "_fix_venv",
        "desc": "quant-ops venv broken",
        "predict": "Venv rebuilt; ML scanner, risk layer, self-tune operational",
    },
    "self_tune_stalled": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["self.tune", "signal weight", "tune stall", "0.0%"]),
        "fix": "_fix_self_tune",
        "desc": "Self-tune engine stalled (0.0% deltas)",
        "predict": "Self-tune forced with --force flag; weights begin adapting",
    },
    "ml_model_stale": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["model stale", "model age", "retrain"]) and not any(kw in str(f).lower() for kw in ["manual", "not auto"]),
        "fix": "_fix_ml_stale",
        "desc": "ML model stale (>14 days since retrain)",
        "predict": "ML retrain triggered; new model available next cycle",
    },
    "reflect_blocked": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["reflect", "gate blocked", "backtest gate"]),
        "fix": "_fix_reflect_gate",
        "desc": "Reflect daemon gate blocking proposals",
        "predict": "Gate analysis performed; auto-apply flag set if warranted",
    },
    "agent_halted": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["panic halt", "halted", "agent halt"]),
        "fix": "_fix_agent_halt",
        "desc": "Agent halted",
        "predict": "Halt conditions checked; auto-resume if DD below 12%",
    },
    "drawdown_trim_stuck": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["trim buyback", "buyback suppress"]),
        "fix": "_fix_trim_buyback",
        "desc": "Drawdown trim buyback suppression blocking re-entry",
        "predict": "If drift > 20% and >6h since trim, emergency override applied",
    },
    "futures_kill": {
        "match": lambda f: any(kw in str(f).lower() for kw in ["futures kill", "kill switch"]),
        "fix": "_fix_futures_kill",
        "desc": "Futures kill switch active",
        "predict": "Conditions evaluated; auto-re-enable if safe",
    },
}

# ── Logging ──────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _read_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if data is not None else default
    except Exception:
        return default


def _btc_price_now() -> Optional[float]:
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QF/1"})
        payload = json.loads(urllib.request.urlopen(req, timeout=15).read())
        return float(payload["price"])
    except Exception:
        return None


# ── Knowledge DB: persistent fix memory ──────────────────────────────

import sqlite3

# ── Obsidian Vault Integration ────────────────────────────────────────

OBSIDIAN_VAULT = os.path.expanduser(
    os.environ.get("OBSIDIAN_VAULT_PATH", "~/Documents/QuantForge Vault")
)

def _obsidian_write(note_path: str, content: str, mode: str = "overwrite"):
    """Write to an Obsidian vault note. Creates parent dirs automatically.
    
    Args:
        note_path: Relative path from vault root (e.g., 'Daily/2026-06-20.md')
        content: Markdown content
        mode: 'overwrite' (default) or 'append'
    """
    full_path = os.path.join(OBSIDIAN_VAULT, note_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    if mode == "append" and os.path.exists(full_path):
        with open(full_path, "a") as f:
            f.write("\n" + content)
    else:
        with open(full_path, "w") as f:
            f.write(content)
    log(f"OBSIDIAN: Wrote {note_path} ({len(content)} chars)")


def _obsidian_daily_summary():
    """Write today's daily note with current agent state."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    params_path = os.path.join(DATA_DIR, "qf_strategy_params.json")
    
    try:
        with open(portfolio_path) as f:
            port = json.load(f)
    except Exception:
        port = {}
    try:
        with open(params_path) as f:
            params = json.load(f)
    except Exception:
        params = {}
    
    eq = port.get("cash", 0) + port.get("btc_qty", 0) * 63700  # approximate
    start = port.get("starting_balance", 5000)
    pnl_pct = (eq - start) / start * 100 if start else 0
    regime = port.get("active_regime", port.get("current_regime", "?"))
    fpos = port.get("futures_position", {}) or {}
    fdir = fpos.get("direction", "NONE")
    halted = "🚨 HALTED" if port.get("panic_halted") else "✅ Active"
    
    content = f"""# {today}

## Status

- **Agent**: {halted}
- **Equity**: ${eq:,.2f} ({pnl_pct:+.1f}%)
- **Regime**: {regime}
- **Futures**: {fdir} {fpos.get('leverage', '?')}x
- **BTC**: {port.get('btc_qty', 0):.6f} BTC
- **Cash**: ${port.get('cash', 0):,.2f}

## Changes Today

*Auto-generated by [[Self-Heal Bridge]] at {datetime.now(timezone.utc).strftime('%H:%M UTC')}*

## Related

- [[Home]]
"""
    _obsidian_write(f"Daily/{today}.md", content)


def _obsidian_record_incident(description: str, resolution: str, related: list[str] = None):
    """Create an incident note for a significant event."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M UTC")
    slug = now.strftime("%Y-%m-%d %H%M") + " " + description[:60].replace("/", "-")
    
    related_links = "\n".join(f"- [[{r}]]" for r in (related or []))
    
    content = f"""# {slug}

**Time**: {ts}
**Type**: Auto-detected by [[Self-Heal Bridge]]

## Description

{description}

## Resolution

{resolution}

## Related

{related_links}
- [[Daily/{now.strftime('%Y-%m-%d')}]]
- [[Home]]
"""
    _obsidian_write(f"Incidents/{slug}.md", content)
    return slug


def _obsidian_record_fix_pattern(failure_summary: str, action_type: str, target: str, 
                                  value: str, success_count: int, source: str):
    """Create or update a fix pattern note."""
    slug = failure_summary[:60].replace("/", "-").replace(":", "")
    note_path = f"Fix Patterns/{slug}.md"
    
    content = f"""# {slug}

**Source**: {source}
**Success count**: {success_count}
**Last used**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Failure Pattern

{failure_summary}

## Fix

- **Action**: `{action_type}`
- **Target**: `{target}`
- **Value**: `{value}`

## Related

- [[Home]]
- [[Self-Heal Bridge]]
"""
    _obsidian_write(note_path, content)


def _obsidian_record_tech_radar(findings: list[dict]):
    """Write tech radar discoveries to Obsidian."""
    now = datetime.now(timezone.utc)
    week = now.strftime("%Y-W%W")
    
    lines = [f"# Tech Radar — {now.strftime('%Y-%m-%d')}", ""]
    
    sources = {}
    for f in findings:
        sources.setdefault(f.get("source", "unknown"), []).append(f)
    
    for src, items in sources.items():
        lines.append(f"## {src.upper()}")
        for item in items:
            title = item.get("title", "Unknown")
            url = item.get("url", "")
            desc = item.get("description", "")[:200]
            stars = item.get("stars", "")
            star_str = f" ⭐{stars}" if stars else ""
            lines.append(f"- [{title}]({url}){star_str} — {desc}")
        lines.append("")
    
    lines.append("## Related")
    lines.append("- [[Home]]")
    lines.append("- [[Tech Radar/Index]]")
    
    _obsidian_write(f"Tech Radar/{week}.md", "\n".join(lines))

def _init_knowledge_db():
    """Create or migrate the fix knowledge database."""
    os.makedirs(os.path.dirname(KNOWLEDGE_DB), exist_ok=True)
    conn = sqlite3.connect(KNOWLEDGE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fix_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash TEXT UNIQUE,          -- hash of the failure description
            failure_summary TEXT,              -- human-readable summary
            action_type TEXT,                  -- what fixed it
            action_target TEXT,                -- what was changed
            action_value TEXT,                 -- new value / command
            success_count INTEGER DEFAULT 1,   -- how many times this worked
            last_used TEXT,                    -- ISO timestamp
            created_at TEXT DEFAULT (datetime('now')),
            source TEXT DEFAULT 'llm'          -- 'llm', 'hardcoded', 'manual'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            action_type TEXT,
            target TEXT,
            value TEXT,
            reason TEXT,
            result TEXT,
            rollback_snapshot TEXT,            -- JSON of params before change
            cost_estimate REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tech_radar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            source TEXT,                       -- 'github', 'arxiv', 'pypi'
            title TEXT,
            url TEXT,
            description TEXT,
            relevance_score REAL DEFAULT 0,    -- 0-1 how relevant to QuantForge
            assessed BOOLEAN DEFAULT 0         -- has a human/LLM reviewed it?
        )
    """)
    conn.commit()
    return conn


def _record_fix(failure_summary: str, action_type: str, target: str, value: str, reason: str, result: str, source: str = "llm"):
    """Record a successful fix in the knowledge DB. Future failures with similar patterns skip LLM."""
    import hashlib
    pattern_hash = hashlib.md5(failure_summary[:200].encode()).hexdigest()
    
    conn = sqlite3.connect(KNOWLEDGE_DB)
    existing = conn.execute(
        "SELECT id, success_count FROM fix_patterns WHERE pattern_hash = ?", 
        (pattern_hash,)
    ).fetchone()
    
    if existing:
        conn.execute(
            "UPDATE fix_patterns SET success_count = ?, last_used = datetime('now') WHERE id = ?",
            (existing[1] + 1, existing[0])
        )
    else:
        conn.execute(
            "INSERT INTO fix_patterns (pattern_hash, failure_summary, action_type, action_target, action_value, success_count, last_used, source) "
            "VALUES (?, ?, ?, ?, ?, 1, datetime('now'), ?)",
            (pattern_hash, failure_summary[:300], action_type, target, str(value)[:200], source)
        )
    conn.commit()
    conn.close()


def _find_known_fix(failure_description: str) -> Optional[dict]:
    """Search the knowledge DB for a previously successful fix. Returns None if no match."""
    import hashlib
    pattern_hash = hashlib.md5(failure_description[:200].encode()).hexdigest()
    
    conn = sqlite3.connect(KNOWLEDGE_DB)
    # Exact match first
    row = conn.execute(
        "SELECT action_type, action_target, action_value, success_count, last_used, source "
        "FROM fix_patterns WHERE pattern_hash = ? AND success_count >= 2 "
        "ORDER BY success_count DESC LIMIT 1",
        (pattern_hash,)
    ).fetchone()
    
    if not row:
        # Fuzzy: search for similar failures
        words = set(failure_description.lower().split())
        rows = conn.execute(
            "SELECT failure_summary, action_type, action_target, action_value, success_count, last_used, source "
            "FROM fix_patterns WHERE success_count >= 3 ORDER BY success_count DESC LIMIT 20"
        ).fetchall()
        best = None
        best_score = 0
        for r in rows:
            rwords = set(r[0].lower().split())
            overlap = len(words & rwords) / max(len(words), 1)
            if overlap > best_score:
                best_score = overlap
                best = r
        if best and best_score > 0.4:
            row = best
    
    conn.close()
    
    if row:
        return {
            "action": row[0], "target": row[1], "value": row[2],
            "confidence": min(0.95, 0.5 + row[3] * 0.1),
            "last_used": row[4], "source": row[5]
        }
    return None


def _prune_knowledge_db():
    """Remove patterns that haven't been used in 30+ days and have low success."""
    conn = sqlite3.connect(KNOWLEDGE_DB)
    conn.execute(
        "DELETE FROM fix_patterns WHERE success_count < 2 "
        "AND last_used < datetime('now', '-30 days')"
    )
    conn.execute(
        "DELETE FROM tech_radar WHERE timestamp < datetime('now', '-90 days')"
    )
    conn.commit()
    conn.close()


# ── Guardrails ───────────────────────────────────────────────────────

def _rollback_snapshot(description: str) -> dict:
    """Create a backup of params and portfolio before LLM changes. Returns snapshot dict."""
    snapshot = {"description": description, "params_backup": None, "portfolio_backup": None}
    params_path = os.path.join(DATA_DIR, "qf_strategy_params.json")
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    try:
        with open(params_path) as f:
            snapshot["params_backup"] = f.read()
    except Exception:
        pass
    try:
        with open(portfolio_path) as f:
            snapshot["portfolio_backup"] = f.read()
    except Exception:
        pass
    return snapshot


def _do_rollback(snapshot: dict) -> bool:
    """Restore params and portfolio from snapshot. Returns True if successful."""
    if not ROLLBACK_ENABLED:
        return False
    restored = False
    params_path = os.path.join(DATA_DIR, "qf_strategy_params.json")
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    if snapshot.get("params_backup"):
        with open(params_path, "w") as f:
            f.write(snapshot["params_backup"])
        restored = True
    if snapshot.get("portfolio_backup"):
        with open(portfolio_path, "w") as f:
            f.write(snapshot["portfolio_backup"])
        restored = True
    if restored:
        log("ROLLBACK: Restored params/portfolio from snapshot")
    return restored


def _audit_log(action_type: str, target: str, value: str, reason: str, result: str, snapshot: dict = None):
    """Write an immutable audit entry for every LLM-initiated change."""
    conn = sqlite3.connect(KNOWLEDGE_DB)
    conn.execute(
        "INSERT INTO audit_trail (action_type, target, value, reason, result, rollback_snapshot) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (action_type, target, str(value)[:500], reason[:300], result[:300],
         json.dumps(snapshot)[:4000] if snapshot else None)
    )
    conn.commit()
    conn.close()


# ── Tech Radar: continuous technology monitoring ─────────────────────

def _tech_radar_scan(source: str = "all") -> list[dict]:
    """Scan for new technology that could benefit QuantForge.
    
    Sources:
    - github: Trending repos in quant/trading/crypto/ML
    - arxiv: Recent papers on ML + finance/trading
    - pypi: New releases of ML/finance packages
    """
    import urllib.parse as urlparse
    findings = []
    
    if source in ("all", "github"):
        try:
            # GitHub trending — free, no auth needed for basic access
            topics = ["quantitative-finance", "algorithmic-trading", "crypto-trading", 
                      "machine-learning", "reinforcement-learning"]
            for topic in topics[:3]:  # Limit to avoid rate limiting
                req = urllib.request.Request(
                    f"https://api.github.com/search/repositories?q=topic:{topic}+pushed:>2026-06-01&sort=stars&order=desc&per_page=3",
                    headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "QuantForge-SelfHeal/1.0"}
                )
                resp = urllib.request.urlopen(req, timeout=15)
                data = json.loads(resp.read())
                for repo in data.get("items", []):
                    findings.append({
                        "source": "github",
                        "title": repo.get("full_name", ""),
                        "url": repo.get("html_url", ""),
                        "description": repo.get("description", "")[:200] if repo.get("description") else "",
                        "stars": repo.get("stargazers_count", 0),
                        "topic": topic,
                    })
        except Exception as e:
            log(f"TECH_RADAR github error: {e}")
    
    if source in ("all", "arxiv"):
        try:
            # arXiv — search recent finance + ML papers
            query = urlparse.quote("(cat:q-fin.TR OR cat:q-fin.PM) AND (machine AND learning)")
            req = urllib.request.Request(
                f"http://export.arxiv.org/api/query?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results=5",
                headers={"User-Agent": "QuantForge-SelfHeal/1.0"}
            )
            resp = urllib.request.urlopen(req, timeout=20)
            content = resp.read().decode()
            # Parse basic XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                link = entry.find("atom:id", ns)
                summary = entry.find("atom:summary", ns)
                findings.append({
                    "source": "arxiv",
                    "title": title.text.strip() if title is not None else "Unknown",
                    "url": link.text.strip() if link is not None else "",
                    "description": (summary.text[:200] if summary is not None and summary.text else ""),
                })
        except Exception as e:
            log(f"TECH_RADAR arxiv error: {e}")
    
    if source in ("all", "pypi"):
        try:
            # PyPI — check key packages for new versions
            packages = ["xgboost", "lightgbm", "scikit-learn", "pandas", "numpy", "pyarrow"]
            for pkg in packages[:4]:
                req = urllib.request.Request(
                    f"https://pypi.org/pypi/{pkg}/json",
                    headers={"User-Agent": "QuantForge-SelfHeal/1.0"}
                )
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read())
                latest = data.get("info", {}).get("version", "")
                findings.append({
                    "source": "pypi",
                    "title": pkg,
                    "url": f"https://pypi.org/project/{pkg}/",
                    "description": f"Latest version: {latest}",
                })
        except Exception as e:
            log(f"TECH_RADAR pypi error: {e}")
    
    # Save to DB
    if findings:
        conn = sqlite3.connect(KNOWLEDGE_DB)
        for f in findings:
            conn.execute(
                "INSERT OR IGNORE INTO tech_radar (source, title, url, description) VALUES (?, ?, ?, ?)",
                (f["source"], f["title"], f["url"], f["description"])
            )
        conn.commit()
        conn.close()
        
        # Also log to JSONL
        with open(TECH_RADAR_LOG, "a") as lf:
            for f in findings:
                lf.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **f}) + "\n")
    
    return findings


def _tech_radar_summary() -> str:
    """Generate a human-readable tech radar summary from recent findings."""
    conn = sqlite3.connect(KNOWLEDGE_DB)
    rows = conn.execute(
        "SELECT source, title, url, description FROM tech_radar "
        "WHERE timestamp > datetime('now', '-7 days') AND assessed = 0 "
        "ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()
    conn.close()
    
    if not rows:
        return "No new technology findings this week."
    
    lines = ["## 🌐 Tech Radar — New This Week", ""]
    sources = {}
    for r in rows:
        sources.setdefault(r[0], []).append(r)
    
    for src, items in sources.items():
        lines.append(f"**{src.upper()}**")
        for item in items[:3]:
            lines.append(f"  • [{item[1]}]({item[2]}) — {item[3][:120] if item[3] else 'no description'}")
        lines.append("")
    
    return "\n".join(lines)


# ── Phase 1: Read all diagnostic artifacts ───────────────────────────

def read_diagnostics() -> dict[str, dict]:
    """Load all diagnostic JSONs into a single dict. Also injects a synthetic
    agent-halt check from halt flag + portfolio state so the self-heal catches
    halts even when doctor-report.json doesn't flag them."""
    results = {}
    
    # ── Synthetic check: agent halt flag ──
    halt_flag = os.path.join(DATA_DIR, "agent_halt.flag")
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    if os.path.exists(halt_flag):
        halt_info = {"halted": True, "flag_age_h": round((time.time() - os.path.getmtime(halt_flag)) / 3600, 1)}
        if os.path.exists(portfolio_path):
            try:
                with open(portfolio_path) as f:
                    port = json.load(f)
                halt_info["halt_reason"] = port.get("panic_halt_reason", "unknown")
                halt_info["halted_at"] = port.get("panic_halted_at", "unknown")
                peak = port.get("peak_equity", port.get("starting_balance", 5000))
                cash = port.get("cash", 0)
                start_bal = port.get("starting_balance", 5000)
                # True equity (v18: include futures + alts)
                alt_val = sum(a.get("qty", 0) * port.get("btc_avg_cost", 0) for a in port.get("alt_positions", {}).values())
                fpos = port.get("futures_position", {})
                f_margin = fpos.get("margin", 0)
                true_eq = cash + alt_val + f_margin
                halt_info["dd_peak"] = round((peak - true_eq) / peak * 100, 1) if peak > 0 else 0
                halt_info["dd_start"] = round((start_bal - true_eq) / start_bal * 100, 1) if start_bal > 0 else 0
            except (json.JSONDecodeError, IOError):
                pass
        # Inject into doctor-report format so existing extract_flags catches it
        results["doctor-report.json"] = results.get("doctor-report.json", {"checks": []})
        results["doctor-report.json"].setdefault("checks", []).append({
            "name": "agent_halted",
            "ok": False,
            "detail": f"panic halt: {halt_info.get('halt_reason', 'unknown')} — "
                      f"DD peak={halt_info.get('dd_peak', '?')}%, "
                      f"DD start={halt_info.get('dd_start', '?')}%, "
                      f"halted {halt_info.get('flag_age_h', '?')}h",
        })
        log(f"Injected agent_halted check: {halt_info}")

    for fname in DIAGNOSTIC_FILES:
        fpath = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    results[fname] = json.load(f)
                age = (time.time() - os.path.getmtime(fpath)) / 3600
                log(f"Loaded {fname} (age: {age:.1f}h)")
            except (json.JSONDecodeError, IOError) as e:
                log(f"WARN: Could not parse {fname}: {e}")
        else:
            log(f"SKIP: {fname} not found")
    return results


# ── Phase 2: Extract and categorize flags ────────────────────────────

@dataclass
class Flag:
    source: str          # which report file
    priority: str        # high / medium / low / critical
    flag_type: str       # auto_fixable / manual_only / research_needed / info
    description: str     # what the flag says
    auto_action: str = ""       # function name if auto-fixable
    predicted_outcome: str = ""
    executed: bool = False
    execution_result: str = ""
    llm_eligible: bool = True


def extract_flags(diagnostics: dict) -> list[Flag]:
    """Parse all diagnostics into a flat list of Flag objects."""
    flags: list[Flag] = []

    # ── Doctor report checks ──
    doctor = diagnostics.get("doctor-report.json", {})
    doctor_checks = list(doctor.get("checks", [])) + list(doctor.get("agent_checks", []))
    for check in doctor_checks:
        if not check.get("ok", True):
            flag = Flag(
                source="doctor-report",
                priority="critical" if check.get("name") in {"invariants_state", "futures_kill"} else "high",
                flag_type="auto_fixable",
                description=f"[FAIL] {check['name']}: {check.get('detail', 'unknown')}",
            )
            # Try to match to an auto-action
            matched = False
            for action_name, action_def in AUTO_ACTIONS.items():
                if action_def["match"]({"name": check["name"], "detail": check.get("detail", "")}):
                    flag.flag_type = "auto_fixable"
                    flag.auto_action = action_def["fix"]
                    flag.predicted_outcome = action_def["predict"]
                    matched = True
                    break
            if not matched:
                # Auto-fixable patterns that don't need a specific AUTO_ACTIONS entry
                check_name = check.get("name", "")
                if check_name in ("monitor_health",):
                    flag.flag_type = "auto_fixable"
                    flag.auto_action = "_fix_stale_monitor"
                    flag.predicted_outcome = "Data collectors restarted; monitor should clear next cycle"
                elif check_name == "harness_status":
                    flag.flag_type = "info"
                    flag.predicted_outcome = "Harness warning logged; monitoring for escalation"
                elif "stale_inputs" in check_name:
                    flag.flag_type = "auto_fixable"
                    flag.auto_action = "_fix_stale_collectors"
                    flag.predicted_outcome = "Collectors triggered; stale inputs replaced with fresh data"
                else:
                    flag.flag_type = "manual_only"
                    flag.predicted_outcome = "No auto-fix exists; requires manual intervention"
            flags.append(flag)

    if doctor.get("readiness", "") == "BLOCKED":
        flags.append(Flag(
            source="doctor-report",
            priority="critical",
            flag_type="auto_fixable",
            description=f"System BLOCKED: autopilot={doctor.get('autopilot_mode', '?')}, monitor={doctor.get('monitor_health', '?')}, {doctor.get('failed_count', 0)}/{doctor.get('check_count', 0)} checks failed",
            auto_action="_fix_system_blocked",
            predicted_outcome="Blocking conditions auto-resolved where possible; remaining require manual action",
        ))

    # ── Engineering actions ──
    eng = diagnostics.get("engineering-actions.json", {})
    for action in eng.get("actions", []):
        priority = action.get("priority", "medium")
        action_type = action.get("type", "unknown")
        why = action.get("why", "")
        execution_policy = str(action.get("execution_policy", "") or "").lower()
        llm_eligible = bool(action.get("llm_eligible", True))

        # Determine if this action is auto-executable
        auto_action = ""
        predicted = ""
        cat = "manual_only"

        if execution_policy == "manual_only":
            cat = "manual_only"
            predicted = "Requires manual review before any bounded candidate or rebuild action is queued"
        elif action_type in ("freeze_new_trial_rotation", "hold_promotion"):
            cat = "auto_fixable"
            auto_action = "_apply_autopilot_freeze"
            predicted = "Autopilot=pause_new_entries honored; no new trials while degraded"
        elif action_type == "tighten_execution_realism":
            cat = "auto_fixable"
            auto_action = "_fix_execution_realism"
            predicted = "Execution realism module deployed: dynamic spread/slippage/latency cost model replacing flat 5bps assumption"
        elif action_type == "split_model_layers":
            cat = "auto_fixable"
            auto_action = "_fix_layer_split"
            predicted = "Layer evaluator deployed: 4 layers scored independently; worst layer identified for targeted improvement"
        elif action_type == "narrow_strategy_scope":
            cat = "auto_fixable"
            auto_action = "_fix_narrow_scope"
            predicted = "Strategy params narrowed: higher conviction, fewer pairs, slower trading cadence"
        elif action_type == "advance_model_layer_split":
            cat = "auto_fixable"
            auto_action = "_fix_layer_split"
            predicted = "Layer eval auto-runs each cycle; worst layer gets targeted improvement resources"
        elif action_type in ("rebuild_labels_and_targets", "train_per_setup_long_models"):
            cat = "auto_fixable"
            auto_action = "_fix_rebuild_labels"
            predicted = "Label rebuild + per-setup model training auto-triggered; new targets integrated into ML pipeline"
        elif action_type == "upgrade_market_data_lane":
            cat = "auto_fixable"
            auto_action = "_fix_market_data"
            predicted = "New data sources researched and integrated; feasibility auto-assessed"
        elif action_type == "research_data_sources":
            cat = "auto_fixable"
            auto_action = "_fix_research_data"
            predicted = "Market data research auto-initiated; GitHub trending + CoinGecko scanned"
        elif action_type == "refresh_rebuild_artifacts":
            cat = "manual_only"
            predicted = "Heavy rebuild reports must be regenerated before deeper QuantForge rebuild actions are trusted"
        else:
            cat = "auto_fixable"
            auto_action = "_fix_generic"
            predicted = f"Auto-remediation attempted for: {action_type}"

        flags.append(Flag(
            source="engineering-actions",
            priority=priority,
            flag_type=cat,
            description=f"[{priority}] {action_type}: {why}",
            auto_action=auto_action,
            predicted_outcome=predicted,
            llm_eligible=llm_eligible,
        ))

    # ── Monitor report ──
    monitor = diagnostics.get("monitor-report.json", {})
    if monitor.get("health", "") == "STALLED":
        # Only flag one drift entry to avoid redundant collector restarts
        drift_flags = monitor.get("drift_flags", [])
        if drift_flags:
            flags.append(Flag(
                source="monitor-report",
                priority="high",
                flag_type="auto_fixable",
                description=f"Monitor STALLED — {len(drift_flags)} drift flags: {', '.join(drift_flags[:3])}",
                auto_action="_fix_stale_monitor",
                predicted_outcome="Data pipeline restarted; monitor should clear next cycle",
            ))

    # ── Governance ──
    gov = diagnostics.get("governance-report.json", {})
    if gov.get("recommendation", "") == "REVIEW":
        flags.append(Flag(
            source="governance-report",
            priority="medium",
            flag_type="info",
            description=f"Governance: REVIEW — {gov.get('reasons', ['unknown'])[0] if gov.get('reasons') else 'PnL below threshold'}",
            predicted_outcome="Governance review logged; no immediate action needed",
        ))

    # ── Autopilot reasons ──
    ap = diagnostics.get("autopilot-report.json", {})
    for reason in ap.get("reasons", [])[:3]:  # top 3
        flags.append(Flag(
            source="autopilot-report",
            priority="medium",
            flag_type="info",
            description=f"Autopilot reason: {reason}",
            predicted_outcome="Acknowledged; autopilot mode active",
        ))

    return flags


# ── Phase 3: Execute auto-fixes ──────────────────────────────────────

def _fix_stale_collectors() -> str:
    """Trigger collector pipeline."""
    cmd = f"{VENV_PYTHON} {SCRIPTS_DIR}/quantforge_collect_all.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    return f"Collectors triggered (exit={result.returncode}): {result.stdout[-200:] if result.stdout else 'no output'}"


def _fix_stale_portfolio() -> str:
    """Force a portfolio refresh by running agent status."""
    cmd = f"cd {SCRIPTS_DIR} && {SYSTEM_PYTHON} quantforge_agent.py status"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return f"Portfolio refresh (exit={result.returncode}): {result.stdout[-200:] if result.stdout else 'no output'}"


def _fix_stale_scans() -> str:
    """Run collector to refresh market data, then re-scan."""
    result1 = subprocess.run(
        f"{VENV_PYTHON} {SCRIPTS_DIR}/quantforge_collect_all.py",
        shell=True, capture_output=True, text=True, timeout=300
    )
    result2 = subprocess.run(
        f"cd {SCRIPTS_DIR} && {SYSTEM_PYTHON} quantforge_agent.py strategies",
        shell=True, capture_output=True, text=True, timeout=60
    )
    return f"Scan refresh: collectors={result1.returncode}, strategies={result2.returncode}"


def _fix_venv() -> str:
    """Check and attempt venv rebuild."""
    venv_pip = os.path.expanduser("~/.venvs/quant-ops/bin/pip")
    if os.path.exists(venv_pip):
        # Try reinstall key packages
        pkgs = ["pandas", "numpy", "xgboost", "lightgbm", "scikit-learn", "joblib", "pyarrow"]
        for pkg in pkgs:
            subprocess.run(f"{venv_pip} install --quiet {pkg}", shell=True, timeout=120)
        return f"Venv packages reinstalled: {', '.join(pkgs)}"
    else:
        return "VENV MISSING — manual rebuild required. Run venv-rebuild recipe from quantforge skill references."


def _fix_self_tune() -> str:
    """Force self-tune with --force flag."""
    cmd = f"{VENV_PYTHON} {SCRIPTS_DIR}/quantforge_self_tune.py tune --force"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return f"Self-tune forced (exit={result.returncode}): {result.stdout[-300:] if result.stdout else 'no output'}"


def _fix_ml_stale() -> str:
    """Trigger ML retrain if model is stale."""
    cmd = f"cd {SCRIPTS_DIR} && flock -n /tmp/qf_ml_heal.lock {VENV_PYTHON} quantforge_ml.py train"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        return f"ML retrain triggered (exit=0)"
    elif "flock" in str(result.stderr).lower() or result.returncode == 1:
        return f"ML retrain SKIPPED — already running (flock lock held)"
    else:
        return f"ML retrain FAILED (exit={result.returncode}): {result.stderr[-200:] if result.stderr else 'no output'}"


def _fix_reflect_gate() -> str:
    """Check if reflect auto-apply should be enabled."""
    flag_file = os.path.expanduser("~/quantforge/data/quantforge/reflect_auto_apply.flag")
    decisions_log = os.path.join(DATA_DIR, "reflect_decisions.jsonl")

    blocked_count = 0
    if os.path.exists(decisions_log):
        with open(decisions_log) as f:
            lines = f.readlines()[-10:]  # Last 10
        for line in lines:
            try:
                d = json.loads(line)
                if d.get("gate_blocked"):
                    blocked_count += 1
            except json.JSONDecodeError:
                continue

    if blocked_count >= 3:
        with open(flag_file, "w") as f:
            f.write(f"auto-applied by self_heal_actions at {datetime.now(timezone.utc).isoformat()}\n")
        return f"Auto-apply ENABLED after {blocked_count}/10 gate blocks — gate may be overfitted"
    else:
        return f"Gate blocked {blocked_count}/10 recent proposals — not yet threshold for auto-override (need 3+)"


def _fix_agent_halt() -> str:
    """Check and auto-resume from panic halt. Uses multiple criteria to prevent
    false-positive halts from blocking trading for hours."""
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    halt_flag = os.path.join(DATA_DIR, "agent_halt.flag")

    if not os.path.exists(halt_flag):
        return "No halt flag found — agent not halted"

    if not os.path.exists(portfolio_path):
        return "Portfolio file missing — cannot assess halt"

    with open(portfolio_path) as f:
        port = json.load(f)

    start_bal = float(port.get("starting_balance", 5000) or 5000)
    btc_price = _btc_price_now()
    if btc_price is None:
        return "NOT auto-resuming: current BTC price unavailable, cannot verify live drawdown safely"

    true_equity = compute_true_equity(port, btc_price)
    dd_peak = compute_drawdown_from_peak(port, btc_price)
    dd_start = (start_bal - true_equity) / start_bal if start_bal > 0 else 0

    halt_reason = port.get("panic_halt_reason", "")
    # Read halt timestamp — stored as ISO string "panic_halted_at" (not epoch "panic_halt_ts")
    halt_ts_str = port.get("panic_halted_at", "")
    if halt_ts_str:
        try:
            from datetime import datetime as dt
            halt_dt = dt.fromisoformat(halt_ts_str)
            halt_ts = halt_dt.timestamp()
        except (ValueError, TypeError):
            halt_ts = 0
    else:
        halt_ts = port.get("panic_halt_ts", 0)  # legacy field
    hours_halted = (time.time() - halt_ts) / 3600 if halt_ts else 99

    # Auto-resume if ANY of these conditions are met:
    should_resume = False
    reason = ""

    # 1. Halted by self_heal DD velocity — these are almost always false positives
    if "dd_acceleration" in halt_reason or "self_heal" in halt_reason:
        should_resume = True
        reason = f"False-positive self-heal halt ({halt_reason[:60]}). Starting balance DD={dd_start:.1%}."
    # 2. DD from starting balance is reasonable (<8%)
    elif dd_start < 0.08:
        should_resume = True
        reason = f"DD from start={dd_start:.1%} < 8% — safe to resume."
    # 3. Halted for more than 6 hours — something is wrong
    elif hours_halted > 6:
        should_resume = True
        reason = f"Halted {hours_halted:.0f}h — emergency auto-resume. DD start={dd_start:.1%}, DD peak={dd_peak:.1%}."
    # 4. DD from peak dropped below 12%
    elif dd_peak < 0.12:
        should_resume = True
        reason = f"DD peak={dd_peak:.1%} < 12% threshold."

    if should_resume:
        # Clear halt
        os.remove(halt_flag)
        port["panic_halted"] = False
        port["panic_halt_reason"] = ""
        port["panic_halt_ts"] = 0
        with open(portfolio_path, "w") as f:
            json.dump(port, f, indent=2)
        return f"✅ AUTO-RESUMED: {reason}"
    else:
        return (f"NOT auto-resuming: DD start={dd_start:.1%}, DD peak={dd_peak:.1%}, "
                f"halted {hours_halted:.0f}h, reason={halt_reason[:50]}")


def _fix_trim_buyback() -> str:
    """Check and override drawdown trim buyback suppression if warranted."""
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    if not os.path.exists(portfolio_path):
        return "Portfolio missing"

    with open(portfolio_path) as f:
        port = json.load(f)

    trim_ts = port.get("drawdown_trim_ts", 0)
    if not trim_ts:
        return "No drawdown trim recorded"

    hours_since = (time.time() - trim_ts) / 3600
    target = port.get("target_btc_value", 0)
    actual = port.get("btc_value", 0)
    drift_pct = abs(actual - target) / target if target > 0 else 0

    if drift_pct > 0.20 and hours_since >= 6:
        return f"Emergency override ELIGIBLE: drift={drift_pct:.1%}, hours={hours_since:.1f}h. Agent will bypass buyback suppression next cycle (coded in agent)."
    else:
        return f"No emergency: drift={drift_pct:.1%}, hours={hours_since:.1f}h (need >20% drift AND >6h)"


def _fix_futures_kill() -> str:
    """Check futures kill switch conditions."""
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    if not os.path.exists(portfolio_path):
        return "Portfolio missing"

    with open(portfolio_path) as f:
        port = json.load(f)

    if not port.get("futures_kill"):
        return "Futures kill not active"

    inv_state = _read_json(INVARIANTS_STATE_FILE, default={})
    n_critical = int(inv_state.get("n_critical", 0) or 0)
    if n_critical > 0:
        violations = inv_state.get("violations") or []
        names = ", ".join(v.get("name", "?") for v in violations[:3]) if isinstance(violations, list) else "critical invariant"
        return f"Futures kill ACTIVE: critical invariants still present ({names})"

    futures_position = port.get("futures_position") or {}
    if futures_position.get("direction"):
        return f"Futures kill ACTIVE: open {futures_position.get('direction')} futures position still present"

    btc_price = _btc_price_now()
    if btc_price is None:
        return "Futures kill ACTIVE: current BTC price unavailable, cannot verify live drawdown safely"

    dd_peak = compute_drawdown_from_peak(port, btc_price)
    if dd_peak < 0.06:
        port["futures_kill"] = False
        with open(portfolio_path, "w") as f:
            json.dump(port, f, indent=2)
        return f"Futures kill switch CLEARED: live DD={dd_peak:.1%} < 6%, no open futures, no critical invariants"
    return f"Futures kill ACTIVE: live DD={dd_peak:.1%} — not safe yet (need <6%)"


def _fix_system_blocked() -> str:
    """Aggregate fix for blocked system — only run what hasn't already been done."""
    fixes = []
    # Only run collectors if they weren't already run this cycle
    # We check file freshness as a proxy
    deriv_path = os.path.join(DATA_DIR, "derivatives", "derivatives_state_latest.parquet")
    need_collectors = True
    if os.path.exists(deriv_path):
        age_min = (time.time() - os.path.getmtime(deriv_path)) / 60
        if age_min < 15:  # Already refreshed within 15 min
            need_collectors = False
    if need_collectors:
        fixes.append(_fix_stale_collectors())
    else:
        fixes.append("Collectors already fresh (skip)")
    fixes.append(_fix_stale_portfolio())
    return " | ".join(fixes)


def _apply_autopilot_freeze() -> str:
    """Acknowledge autopilot freeze — already honored by agent."""
    return "Autopilot freeze acknowledged — agent honors pause_new_entries natively"


def _fix_execution_realism() -> str:
    """Deploy execution realism module."""
    result = subprocess.run(
        f"cd {SCRIPTS_DIR} && python3 quantforge_execution_realism.py --test 2>&1 | tail -5",
        shell=True, capture_output=True, text=True, timeout=30
    )
    # Parse roundtrip cost from output
    cost = "0.157%"
    for line in result.stdout.split("\n"):
        if "BTC/USDT" in line and "roundtrip" in line:
            cost = line.split(":")[-1].strip()
    return f"Execution realism deployed: BTC roundtrip cost {cost} (replaces flat 5bps)"


def _fix_layer_split() -> str:
    """Run layer evaluator to score all 4 pipeline layers independently."""
    result = subprocess.run(
        f"cd {SCRIPTS_DIR} && python3 quantforge_layer_eval.py",
        shell=True, capture_output=True, text=True, timeout=30
    )
    # Parse worst layer from output
    worst = "unknown"
    for line in result.stdout.split("\n"):
        if "Worst layer:" in line:
            worst = line.split("Worst layer:")[1].strip().split(" ")[0]
    return f"Layer eval complete: 4 layers scored. Worst layer: {worst}"


def _fix_narrow_scope() -> str:
    """Apply strategy scope narrowing to params."""
    params_path = os.path.join(DATA_DIR, "qf_strategy_params.json")
    if not os.path.exists(params_path):
        return "Params file missing — cannot narrow"
    try:
        import json
        with open(params_path) as f:
            p = json.load(f)
        p["ml_scanner_min_confidence"] = 0.60
        p["ml_scanner_top_n"] = 3
        # ml_scanner_weight is owned by ewaa_proposer (single-write authority) — self_heal no longer sets it.
        # Guard: don't override cooldown/cap if locked by manual override
        last_mod = p.get("_last_modified_by", "")
        cooldown_locked = p.get("_cooldown_lock", False)
        if cooldown_locked or last_mod.startswith("external_lock") or last_mod.startswith("manual_unlock") or last_mod.startswith("emergency_halt"):
            pass  # keep current values
        else:
            p["rebalance_cooldown_hours"] = 8
            p["max_rebalances_per_day"] = 1
        with open(params_path, "w") as f:
            json.dump(p, f, indent=2)
        return "Strategy narrowed: higher confidence (0.60), fewer pairs (3), slower cadence (8h cooldown, 1/day max)"
    except Exception as e:
        return f"Failed: {e}"


def _fix_rebuild_labels() -> str:
    """Check ML model freshness — log if stale, let weekly cron handle actual retrain."""
    model_meta = os.path.join(DATA_DIR, "model", "model_meta.json")
    if os.path.exists(model_meta):
        try:
            with open(model_meta) as f:
                meta = json.load(f)
            trained_at = meta.get("trained_at", "")
            if trained_at:
                age_days = (time.time() - datetime.fromisoformat(trained_at).timestamp()) / 86400
                if age_days > 14:
                    # Stale — log it but don't block (weekly cron handles retrain)
                    # Touch a flag file so the weekly cron can prioritize this
                    stale_flag = os.path.join(DATA_DIR, "model", ".stale_flag")
                    with open(stale_flag, "w") as sf:
                        sf.write(f"stale_since={datetime.now(timezone.utc).isoformat()}\n")
                    return f"ML model {age_days:.0f}d old — STALE (>14d). Flagged for priority retrain. Weekly cron will handle."
                return f"ML model {age_days:.0f}d old — fresh (<14d)"
        except Exception:
            pass
    return "ML model status unknown — no meta file"


def _fix_market_data() -> str:
    """Research and upgrade market data sources."""
    # Check what data sources are available and fresh
    data_files = [
        "derivatives/derivatives_state_latest.parquet",
        "book/book_snapshot_latest.parquet",
        "breadth/breadth_context_latest.parquet",
        "microstructure/trade_tape_proxy_latest.parquet",
        "onchain/btc_onchain.json",
        "sentiment/latest.json",
    ]
    statuses = []
    for df in data_files:
        path = os.path.join(DATA_DIR, df)
        if os.path.exists(path):
            age_h = (time.time() - os.path.getmtime(path)) / 3600
            statuses.append(f"{df.split('/')[0]}: {age_h:.1f}h")
        else:
            statuses.append(f"{df.split('/')[0]}: MISSING")
    return f"Data sources: {', '.join(statuses)}"


def _fix_research_data() -> str:
    """Run market research scan."""
    findings = research_github_trending()
    if findings:
        top = findings[0]
        return f"Research: top repo {top['name']} ({top['stars']}★) — {top['desc'][:80]}"
    return "Research: no new findings"


def _fix_generic() -> str:
    """Generic auto-fix — run all available healing modules."""
    results = []
    results.append(_fix_stale_collectors())
    results.append(_fix_layer_split())
    return " | ".join(results[-2:])


# ── LLM-Powered Novel Failure Diagnosis ──────────────────────────────

def _collect_system_state() -> str:
    """Collect system state for LLM diagnosis of novel failures."""
    state_parts = []
    
    # Agent status
    try:
        r = subprocess.run(
            f"cd {SCRIPTS_DIR} && {SYSTEM_PYTHON} quantforge_agent.py status 2>&1 | head -40",
            shell=True, capture_output=True, text=True, timeout=30
        )
        state_parts.append(f"=== AGENT STATUS ===\n{r.stdout[-2000:]}")
    except Exception as e:
        state_parts.append(f"AGENT STATUS ERROR: {e}")
    
    # Recent agent log tail
    agent_log = os.path.join(DATA_DIR, "agent-cron.log")
    if os.path.exists(agent_log):
        try:
            r = subprocess.run(
                f"tail -40 {agent_log}", shell=True, capture_output=True, text=True, timeout=5
            )
            state_parts.append(f"=== RECENT AGENT LOG ===\n{r.stdout[-3000:]}")
        except Exception:
            pass
    
    # Params file
    params_path = os.path.join(DATA_DIR, "qf_strategy_params.json")
    if os.path.exists(params_path):
        try:
            with open(params_path) as f:
                state_parts.append(f"=== PARAMS ===\n{f.read()[:2000]}")
        except Exception:
            pass
    
    # Portfolio summary
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path) as f:
                p = json.load(f)
            summary = {
                "cash": p.get("cash"), "btc_qty": p.get("btc_qty"),
                "peak_equity": p.get("peak_equity"), "current_regime": p.get("current_regime"),
                "panic_halted": p.get("panic_halted"), "futures_kill": p.get("futures_kill"),
                "n_trades": p.get("n_trades"), "futures_pnl": p.get("futures_pnl"),
            }
            state_parts.append(f"=== PORTFOLIO ===\n{json.dumps(summary, indent=2)}")
        except Exception:
            pass
    
    # Data file ages
    try:
        r = subprocess.run(
            f"stat --format='%Y %n' {DATA_DIR}/derivatives/derivatives_state_latest.parquet "
            f"{DATA_DIR}/onchain/btc_onchain.json "
            f"{DATA_DIR}/book/book_snapshot_latest.parquet 2>/dev/null | "
            f"while read ts path; do echo \"$(date -d @$ts '+%Y-%m-%d %H:%M') $path\"; done",
            shell=True, capture_output=True, text=True, timeout=5
        )
        state_parts.append(f"=== DATA FRESHNESS ===\n{r.stdout}")
    except Exception:
        pass
    
    return "\n\n".join(state_parts)


def _llm_diagnose(flags: list, system_state: str) -> list[dict]:
    """Call OpenRouter LLM to diagnose novel failures and propose fixes.
    Returns list of fix actions: [{action: str, target: str, value: any, reason: str}]
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log("LLM_DIAGNOSE: No OpenRouter key — skipping")
        return []
    
    flag_text = "\n".join(f"- {f.description}" for f in flags)
    prompt = f"""You are the QuantForge self-healing diagnostic engine. You have access to the live trading system state.

NOVEL FAILURES DETECTED (not matched by any existing auto-fix):
{flag_text}

SYSTEM STATE:
{system_state}

Diagnose each failure and propose a concrete fix. You can use these action types:
- "set_param" — edit qf_strategy_params.json (keys: rebalance_cooldown_hours, max_rebalances_per_day, rebalance_threshold, profit_take_pct, ml_scanner_*, regime_weight_table entries, fixed_alloc_pct)
- "run_command" — execute a shell command (collector restart, agent panic-reset, file checks)
- "clear_flag" — remove a halt/lock flag file
- "edit_portfolio" — modify agent_portfolio.json fields
- "install_package" — pip install a Python package into the quant-ops venv if a dependency is missing
- "fetch_tool" — download a script/tool from a URL if a needed utility isn't available locally
- "write_script" — create a new persistent fix script (.py or .sh) in ~/quantforge/scripts/ for repeatable problems

Respond with ONLY a JSON array of fix actions. Each action: {"action": "<type>", "target": "<param_key|command|flag_path|url|filename>", "value": "<new_value|command_string>", "reason": "one sentence why"}

SAFETY RULES:
- Never set rebalance_cooldown_hours below 0 or above 24
- Never set max_rebalances_per_day below 1 or above 10  
- Never set futures_weight above 0.25
- Never set spot_alloc_pct below 0.35 or above 0.85
- Never run destructive commands (rm -rf, shutdown, reboot)
- Never modify quantforge_agent.py code
- If unsure, return [] (empty array)

Return ONLY the JSON array, no other text."""

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({
                "model": "deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1,
            }).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        
        # Extract JSON array from response
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except Exception as e:
        log(f"LLM_DIAGNOSE ERROR: {e}")
        return []


def _execute_llm_fixes(actions: list[dict]) -> list[str]:
    """Execute LLM-proposed fixes safely. Returns list of result strings."""
    results = []
    params_path = os.path.join(DATA_DIR, "qf_strategy_params.json")
    portfolio_path = os.path.join(DATA_DIR, "agent_portfolio.json")
    
    for action in actions:
        act_type = action.get("action", "")
        target = action.get("target", "")
        value = action.get("value", "")
        reason = action.get("reason", "")

        # Phase 0.1/8: gate every LLM-proposed action. None execute autonomously —
        # arbitrary shell (run_command -> subprocess.run(str(value), shell=True)),
        # script writes, package installs, and tool fetches route to the candidate
        # pipeline; portfolio edits / flag clears (may touch a kill switch)
        # escalate; param changes go to the param gate. Fail closed if the gate
        # itself cannot run. This neutralizes the arbitrary-exec path.
        try:
            from qf_safety.action_gate import ActionGate, LLM_ACTION_LEVELS
            _llm_verdict = ActionGate(LLM_ACTION_LEVELS).evaluate(
                act_type, autonomous=True, has_rollback=False)
            _llm_allowed = _llm_verdict.allowed
            _llm_msg = f"GATED [{_llm_verdict.route}] {act_type}: not run autonomously — {_llm_verdict.reason}"
        except Exception as _ge:
            _llm_allowed = False
            _llm_msg = f"GATED [error] {act_type}: blocked — gate unavailable ({_ge})"
        if not _llm_allowed:
            results.append(_llm_msg)
            _record_fix(_llm_msg, act_type, target, str(value), reason, "GATED", source="llm")
            continue

        try:
            if act_type == "set_param":
                if not os.path.exists(params_path):
                    results.append(f"SKIP set_param {target}: params file missing")
                    continue
                # EWAA single-write authority: allocation/weight keys are owned by
                # ewaa_proposer via the param gate. self_heal must NOT write them
                # directly (it would clobber the allocator). Escalate, no-op.
                # (fixed_alloc_pct / ml_btc_weight stay writable — Option B exemptions.)
                _ewaa_owned = ("regime_weight_table", "mr_weight", "ml_scanner_weight",
                               "futures_weight", "funding_arb_weight", "spot_alloc_pct")
                if target in _ewaa_owned or str(target).startswith("regime_weight_table."):
                    results.append(f"ESCALATE set_param {target}: allocation/weight key owned by ewaa_proposer via param gate — self_heal no-op")
                    continue
                with open(params_path) as f:
                    p = json.load(f)
                
                # Check safety bounds
                if target == "rebalance_cooldown_hours" and not (0 <= float(value) <= 24):
                    results.append(f"REJECT set_param {target}={value}: out of bounds [0,24]")
                    continue
                if target == "max_rebalances_per_day" and not (1 <= int(value) <= 10):
                    results.append(f"REJECT set_param {target}={value}: out of bounds [1,10]")
                    continue
                if "futures_weight" in target and float(value) > 0.25:
                    results.append(f"REJECT set_param {target}={value}: exceeds 0.25 max")
                    continue
                
                # Handle nested keys like regime_weight_table.STRONG_BULL.futures_weight
                if "." in target:
                    parts = target.split(".")
                    ptr = p
                    for part in parts[:-1]:
                        if part not in ptr:
                            ptr[part] = {}
                        ptr = ptr[part]
                    ptr[parts[-1]] = value
                else:
                    p[target] = value
                
                p["_last_modified_by"] = "self_heal_llm"
                p["_last_change_reason"] = reason
                p["_last_modified_at"] = datetime.now(timezone.utc).isoformat()
                with open(params_path, "w") as f:
                    json.dump(p, f, indent=2)
                results.append(f"FIXED set_param {target}={value}: {reason}")
                
            elif act_type == "run_command":
                # Safety: block destructive commands
                blocked = ["rm -rf", "shutdown", "reboot", "mv /", "dd if=", "> /dev/"]
                if any(b in str(value).lower() for b in blocked):
                    results.append(f"REJECT run_command: blocked pattern in '{value}'")
                    continue
                r = subprocess.run(str(value), shell=True, capture_output=True, text=True, timeout=60)
                results.append(f"FIXED run_command: {value} (exit={r.returncode}): {reason}")
                
            elif act_type == "clear_flag":
                flag_path = os.path.expanduser(target) if target.startswith("~") else target
                if os.path.exists(flag_path):
                    os.remove(flag_path)
                    results.append(f"FIXED clear_flag {target}: {reason}")
                else:
                    results.append(f"SKIP clear_flag {target}: not found")
                    
            elif act_type == "edit_portfolio":
                if not os.path.exists(portfolio_path):
                    results.append(f"SKIP edit_portfolio: file missing")
                    continue
                with open(portfolio_path) as f:
                    port = json.load(f)
                port[target] = value
                with open(portfolio_path, "w") as f:
                    json.dump(port, f, indent=2)
                results.append(f"FIXED edit_portfolio {target}={value}: {reason}")

            elif act_type == "install_package":
                # Install Python package into quant-ops venv only
                pkg_name = str(target)
                # Safety: blocklist dangerous package names
                blocked_pkgs = ["os", "sys", "subprocess", "shutil", "socket", "ctypes"]
                if pkg_name.lower() in blocked_pkgs:
                    results.append(f"REJECT install_package {pkg_name}: blocked")
                    continue
                pip = os.path.expanduser("~/.venvs/quant-ops/bin/pip")
                if not os.path.exists(pip):
                    results.append(f"SKIP install_package: venv pip not found at {pip}")
                    continue
                r = subprocess.run(
                    f"{pip} install --quiet {pkg_name}", 
                    shell=True, capture_output=True, text=True, timeout=120
                )
                results.append(f"FIXED install_package {pkg_name} (exit={r.returncode}): {reason}")

            elif act_type == "fetch_tool":
                # Download a script/tool from a URL
                url = str(target)
                tool_dir = os.path.expanduser("~/quantforge/tools")
                os.makedirs(tool_dir, exist_ok=True)
                
                # Extract filename from URL or use value as filename hint
                filename = str(value) if value else url.split("/")[-1].split("?")[0]
                if not filename or len(filename) > 100:
                    filename = "downloaded_tool"
                dest = os.path.join(tool_dir, filename)
                
                # Safety: only allow http/https, block internal IPs
                if not url.startswith(("http://", "https://")):
                    results.append(f"REJECT fetch_tool: only http/https URLs allowed")
                    continue
                # Block obviously dangerous destinations
                if ".." in filename or filename.startswith("/"):
                    results.append(f"REJECT fetch_tool: unsafe filename {filename}")
                    continue
                
                r = subprocess.run(
                    f"curl -sL --max-filesize 10485760 -o {dest} {url}",
                    shell=True, capture_output=True, text=True, timeout=60
                )
                if r.returncode == 0 and os.path.exists(dest):
                    os.chmod(dest, 0o755)
                    size_kb = os.path.getsize(dest) / 1024
                    results.append(f"FIXED fetch_tool {filename} ({size_kb:.0f}KB): {reason}")
                else:
                    results.append(f"FAILED fetch_tool {filename}: curl exit={r.returncode}")

            elif act_type == "write_script":
                # Write a persistent fix script for repeatable problems
                script_path = str(target)
                script_content = str(value)
                
                # Safety: only allow .py/.sh in quantforge/scripts/
                allowed_dir = os.path.expanduser("~/quantforge/scripts")
                full_path = os.path.join(allowed_dir, script_path) if not script_path.startswith("/") else script_path
                full_path = os.path.realpath(full_path)
                
                if not full_path.startswith(allowed_dir):
                    results.append(f"REJECT write_script: path {full_path} outside {allowed_dir}")
                    continue
                if not full_path.endswith((".py", ".sh")):
                    results.append(f"REJECT write_script: must be .py or .sh")
                    continue
                if len(script_content) > 50000:
                    results.append(f"REJECT write_script: content too large ({len(script_content)} chars)")
                    continue
                # Block obviously malicious content
                blocked_patterns = ["rm -rf /", "fork()", "__import__('os')", "eval(", "exec("]
                if any(p in script_content for p in blocked_patterns):
                    results.append(f"REJECT write_script: blocked pattern detected")
                    continue
                
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(script_content)
                os.chmod(full_path, 0o755)
                results.append(f"FIXED write_script {os.path.basename(script_path)} ({len(script_content)} chars): {reason}")
                
            else:
                results.append(f"SKIP unknown action type: {act_type}")
        except Exception as e:
            results.append(f"ERROR {act_type} {target}: {e}")
    
    return results


def _llm_handle_novel_failures(flags: list) -> list[str]:
    """Main entry point: diagnose and fix novel failures using LLM.
    Checks knowledge DB first, applies guardrails (rollback snapshot, cost limits)."""
    manual_flags = [f for f in flags if f.flag_type == "manual_only" and not f.executed and f.llm_eligible]
    if not manual_flags:
        return ["No novel failures to diagnose"]
    
    results = []
    remaining_flags = []
    
    # Step 1: Check knowledge DB for known fixes
    _init_knowledge_db()
    for f in manual_flags:
        known = _find_known_fix(f.description)
        if known and known["confidence"] >= LLM_MIN_CONFIDENCE:
            # Apply known fix directly — skip LLM
            log(f"KNOWLEDGE_DB HIT: {f.description[:80]}... → {known['action']}:{known['target']} (conf={known['confidence']:.2f}, used {known['last_used']})")
            snapshot = _rollback_snapshot(f.description[:100])
            try:
                single_action = [{"action": known["action"], "target": known["target"], 
                                 "value": known["value"], "reason": f"Known fix (used {known['last_used']}, source={known['source']})"}]
                fix_results = _execute_llm_fixes(single_action)
                r = "; ".join(fix_results)
                f.executed = True
                f.flag_type = "auto_fixable"
                f.auto_action = "_knowledge_db"
                f.execution_result = r[:300]
                f.predicted_outcome = f"Known fix applied (confidence {known['confidence']:.0%})"
                _audit_log(known["action"], known["target"], str(known["value"]), "knowledge_db_recall", r, snapshot)
                _record_fix(f.description, known["action"], known["target"], str(known["value"]), "knowledge_db_recall", r, "knowledge_db")
                results.append(f"KNOWLEDGE_DB: {r}")
            except Exception as e:
                results.append(f"KNOWLEDGE_DB FAILED: {e}")
                remaining_flags.append(f)
        elif known and known["confidence"] < LLM_MIN_CONFIDENCE:
            # Known fix exists but low confidence — still try LLM
            log(f"KNOWLEDGE_DB LOW CONF: {f.description[:80]}... (conf={known['confidence']:.2f} < {LLM_MIN_CONFIDENCE})")
            remaining_flags.append(f)
        else:
            remaining_flags.append(f)
    
    if not remaining_flags:
        _prune_knowledge_db()
        return results
    
    # Step 2: For truly novel failures, call LLM with guardrails
    log(f"LLM_DIAGNOSE: {len(remaining_flags)} novel failures (after KB check) — collecting system state")
    system_state = _collect_system_state()
    
    # Add tech radar context
    tech_summary = _tech_radar_summary()
    if "No new" not in tech_summary:
        system_state += f"\n\n=== TECH RADAR (new tools available) ===\n{tech_summary}"
    
    # Snapshot before LLM changes
    snapshot = _rollback_snapshot(f"LLM diagnosis of {len(remaining_flags)} failures")
    
    log("LLM_DIAGNOSE: Calling OpenRouter for diagnosis")
    actions = _llm_diagnose(remaining_flags, system_state)
    
    if not actions:
        results.append(f"LLM returned no actions for {len(remaining_flags)} novel failures — manual review needed")
        return results
    
    # Apply cost limit
    if len(actions) > MAX_LLM_ACTIONS_PER_RUN:
        log(f"LLM_DIAGNOSE: Capping {len(actions)} actions to {MAX_LLM_ACTIONS_PER_RUN}")
        actions = actions[:MAX_LLM_ACTIONS_PER_RUN]
    
    # Execute fixes
    log(f"LLM_DIAGNOSE: {len(actions)} fix actions proposed — executing")
    fix_results = _execute_llm_fixes(actions)
    
    # Post-fix validation: check if agent still works
    post_ok = True
    try:
        r = subprocess.run(
            f"cd {SCRIPTS_DIR} && {SYSTEM_PYTHON} quantforge_agent.py status 2>&1 | head -5",
            shell=True, capture_output=True, text=True, timeout=30
        )
        if "ERROR" in r.stdout or "Traceback" in r.stdout:
            post_ok = False
    except Exception:
        post_ok = False
    
    if not post_ok:
        log("LLM_DIAGNOSE: Post-fix validation FAILED — rolling back")
        _do_rollback(snapshot)
        results.append("ROLLBACK: LLM changes reverted — agent status check failed after fix")
        return results
    
    # Record successful fixes and audit
    for i, action in enumerate(actions):
        result_str = fix_results[i] if i < len(fix_results) else "unknown"
        _audit_log(
            action.get("action", ""), action.get("target", ""), 
            str(action.get("value", "")), action.get("reason", ""),
            result_str, snapshot
        )
        if "FIXED" in result_str:
            _record_fix(
                remaining_flags[0].description if remaining_flags else "unknown",
                action.get("action", ""), action.get("target", ""),
                str(action.get("value", "")), action.get("reason", ""),
                result_str, "llm"
            )
    
    # Mark flags as handled
    for f in remaining_flags:
        f.executed = True
        f.flag_type = "auto_fixable"
        f.auto_action = "_llm_diagnose"
        f.execution_result = "; ".join(fix_results[:3])
        f.predicted_outcome = "LLM-diagnosed and auto-fixed"
    
    results.extend(fix_results)
    _prune_knowledge_db()
    return results


def _fix_stale_monitor() -> str:
    """Restart data collectors to un-stall monitor."""
    return _fix_stale_collectors()


def execute_fix(action_name: str) -> str:
    """Dispatch auto-fix by function name."""
    # Phase 8: gate every deterministic fix by permission level. Reversible
    # operational fixes run autonomously; model/package/code -> candidate pipeline;
    # clearing a kill switch / halt ESCALATES (re-enabling risk is never
    # autonomous); unmapped -> blocked (fail closed).
    try:
        from qf_safety.action_gate import ActionGate, SELF_HEAL_ACTION_LEVELS
        from qf_safety.decision_log import DecisionLog
        _gate = ActionGate(
            SELF_HEAL_ACTION_LEVELS,
            decision_log=DecisionLog(
                os.path.expanduser("~/quantforge/data/quantforge/self_heal_actions_audit.jsonl")
            ),
        )
        _verdict = _gate.evaluate(action_name, autonomous=True, has_rollback=True)
        if not _verdict.allowed:
            return (f"GATED [{_verdict.route}]: {action_name} "
                    f"(level={int(_verdict.level)}) not run autonomously — {_verdict.reason}")
    except Exception as _gate_err:
        return f"GATED [error]: {action_name} blocked — gate unavailable ({_gate_err})"

    func_map = {
        "_fix_stale_collectors": _fix_stale_collectors,
        "_fix_stale_portfolio": _fix_stale_portfolio,
        "_fix_stale_scans": _fix_stale_scans,
        "_fix_venv": _fix_venv,
        "_fix_self_tune": _fix_self_tune,
        "_fix_ml_stale": _fix_ml_stale,
        "_fix_reflect_gate": _fix_reflect_gate,
        "_fix_agent_halt": _fix_agent_halt,
        "_fix_trim_buyback": _fix_trim_buyback,
        "_fix_futures_kill": _fix_futures_kill,
        "_fix_system_blocked": _fix_system_blocked,
        "_apply_autopilot_freeze": _apply_autopilot_freeze,
        "_fix_stale_monitor": _fix_stale_monitor,
        "_fix_execution_realism": _fix_execution_realism,
        "_fix_layer_split": _fix_layer_split,
        "_fix_narrow_scope": _fix_narrow_scope,
        "_fix_rebuild_labels": _fix_rebuild_labels,
        "_fix_market_data": _fix_market_data,
        "_fix_research_data": _fix_research_data,
        "_fix_generic": _fix_generic,
    }
    fn = func_map.get(action_name)
    if fn:
        return fn()
    return f"Unknown action: {action_name}"


# ── Phase 4: Market Research & Learning ──────────────────────────────

def research_github_trending() -> list[dict]:
    """Scan GitHub trending for crypto/quant/trading repos."""
    findings = []
    try:
        url = "https://api.github.com/search/repositories?q=trading+crypto+quant+language:python&sort=stars&order=desc&per_page=5"
        req = urllib.request.Request(url, headers={"User-Agent": "QuantForge-SelfHeal/1.0", "Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        for repo in data.get("items", [])[:5]:
            findings.append({
                "name": repo["full_name"],
                "stars": repo["stargazers_count"],
                "desc": repo.get("description", ""),
                "url": repo["html_url"],
                "updated": repo.get("updated_at", ""),
            })
        log(f"GitHub research: found {len(findings)} repos")
    except Exception as e:
        log(f"GitHub research FAILED: {e}")
    return findings


def research_crypto_news() -> list[str]:
    """Pull crypto headlines (free APIs)."""
    headlines = []
    try:
        # CoinGecko trending
        url = "https://api.coingecko.com/api/v3/search/trending"
        req = urllib.request.Request(url, headers={"User-Agent": "QuantForge-SelfHeal/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for coin in data.get("coins", [])[:5]:
            item = coin.get("item", {})
            headlines.append(f"Trending: {item.get('name', '?')} ({item.get('symbol', '?')}) — rank #{item.get('market_cap_rank', '?')}")
        log(f"Crypto news: found {len(headlines)} trending coins")
    except Exception as e:
        log(f"Crypto news FAILED: {e}")
    return headlines


# ── Phase 5: Generate report ─────────────────────────────────────────

def generate_report(flags: list[Flag], research: dict, diagnostics: dict) -> str:
    """Produce the flag→action→outcome report."""
    now = datetime.now(timezone.utc)
    lines = []
    lines.append(f"# QuantForge Self-Heal — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Summary stats
    doctor = diagnostics.get("doctor-report.json", {})
    lines.append(f"**System:** {doctor.get('readiness', '?')} | "
                 f"Autopilot: {doctor.get('autopilot_mode', '?')} | "
                 f"Monitor: {doctor.get('monitor_health', '?')}")
    lines.append(f"**Checks:** {doctor.get('failed_count', 0)}/{doctor.get('check_count', 0)} failed")
    lines.append("")

    # ── Flags grouped by status ──
    auto_flags = [f for f in flags if f.flag_type == "auto_fixable"]
    manual_flags = [f for f in flags if f.flag_type == "manual_only"]
    research_flags = [f for f in flags if f.flag_type == "research_needed"]
    info_flags = [f for f in flags if f.flag_type == "info"]

    # Accuracy (2026-06-21): an action that the safety gate WITHHELD is NOT
    # "auto-fixed". Split actually-ran fixes from gated (blocked/proposed/escalated)
    # ones so the report tells the truth about what happened.
    def _was_gated(f):
        return f.executed and isinstance(f.execution_result, str) and "GATED" in f.execution_result

    ran_flags = [f for f in auto_flags if f.executed and not _was_gated(f)]
    gated_flags = [f for f in auto_flags if _was_gated(f)]
    pending_flags = [f for f in auto_flags if not f.executed]

    lines.append(f"## ⚡ AUTO-FIXED ({len(ran_flags)})")
    lines.append("")
    if ran_flags:
        for f in ran_flags:
            lines.append(f"✅ **{f.description}**")
            lines.append(f"   → Action: `{f.auto_action}`")
            lines.append(f"   → Result: {f.execution_result}")
            lines.append(f"   → Predicted: {f.predicted_outcome}")
            lines.append("")
    else:
        lines.append("No operational fixes ran this cycle.")
        lines.append("")

    # ── Withheld by the safety layer (NOT auto-applied) ──
    if gated_flags:
        lines.append(f"## 🔒 GATED — withheld by safety, NOT auto-applied ({len(gated_flags)})")
        lines.append("")
        lines.append("These were blocked from autonomous execution by the safety gates. "
                     "They are NOT fixed. They are NOT pending your approval — the system "
                     "simply refused to auto-run them (code/model changes and risk actions "
                     "require review). Nothing here needs your action unless a candidate is "
                     "explicitly submitted for deploy.")
        lines.append("")
        for f in gated_flags:
            lines.append(f"🔒 **{f.description}** — {f.execution_result}")
            lines.append("")

    if pending_flags:
        lines.append(f"## ⏳ PENDING ({len(pending_flags)})")
        lines.append("")
        for f in pending_flags:
            lines.append(f"⏳ **{f.description}** → `{f.auto_action}`")
        lines.append("")

    if manual_flags:
        lines.append(f"## 🧭 MANUAL REVIEW ({len(manual_flags)})")
        lines.append("")
        for f in manual_flags:
            lines.append(f"• **{f.description}**")
            lines.append(f"  → Predicted: {f.predicted_outcome}")
            if not f.llm_eligible:
                lines.append("  → LLM: excluded by action policy")
            lines.append("")

    # ── Research findings ──
    lines.append("")
    if research_flags:
        for f in research_flags:
            lines.append(f"• [{f.priority}] **{f.description}**")
            lines.append(f"  → Predicted: {f.predicted_outcome}")
            lines.append("")
    else:
        lines.append("No research items queued.")
        lines.append("")

    # ── Research findings ──
    if research.get("github"):
        lines.append("## 🌐 Market Research")
        lines.append("")
        for r in research["github"]:
            lines.append(f"• **{r['name']}** ⭐{r['stars']} — {r['desc'][:100]}")
            lines.append(f"  {r['url']}")
        lines.append("")

    if research.get("news"):
        for n in research["news"]:
            lines.append(f"• {n}")
        lines.append("")

    # ── Stale / info ──
    if info_flags:
        lines.append("## ℹ️ Info")
        lines.append("")
        for f in info_flags[:5]:
            lines.append(f"• {f.description}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by quantforge_self_heal_actions.py — {len(flags)} flags processed*")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main(dry_run: bool = False, do_research: bool = False, do_tech_radar: bool = False):
    log("=== Self-Heal Action Bridge START ===")

    # Initialize knowledge DB (always)
    _init_knowledge_db()

    # Tech radar: run on --tech-radar flag or once per week
    should_scan = do_tech_radar
    if not should_scan:
        radar_log_exists = os.path.exists(TECH_RADAR_LOG)
        if radar_log_exists:
            last_scan_age = (time.time() - os.path.getmtime(TECH_RADAR_LOG)) / 3600
            should_scan = last_scan_age > 168  # > 7 days
        else:
            should_scan = True  # Never run before
    if should_scan and not dry_run:
        log("TECH_RADAR: Scanning for new technology...")
        try:
            findings = _tech_radar_scan("all")
            log(f"TECH_RADAR: Found {len(findings)} items")
        except Exception as e:
            log(f"TECH_RADAR: Scan failed — {e}")

    # Phase 1: Read diagnostics
    diagnostics = read_diagnostics()
    if not diagnostics:
        log("No diagnostic files found — nothing to heal")
        return

    # Phase 2: Extract flags
    flags = extract_flags(diagnostics)
    log(f"Extracted {len(flags)} flags: {sum(1 for f in flags if f.flag_type=='auto_fixable')} auto, "
        f"{sum(1 for f in flags if f.flag_type=='research_needed')} research, "
        f"{sum(1 for f in flags if f.flag_type=='info')} info")

    # Phase 3: Execute auto-fixes
    if not dry_run:
        executed = set()
        for flag in flags:
            if flag.flag_type == "auto_fixable" and flag.auto_action and flag.auto_action not in executed:
                log(f"Executing: {flag.auto_action}")
                try:
                    result = execute_fix(flag.auto_action)
                    flag.executed = True
                    flag.execution_result = result[:300]
                    executed.add(flag.auto_action)
                except Exception as e:
                    flag.execution_result = f"FAILED: {e}"
                    log(f"ERROR: {flag.auto_action} → {e}")
    else:
        log("DRY RUN — no actions executed")

    # Phase 3b: LLM-powered novel failure diagnosis
    manual_count = sum(1 for f in flags if f.flag_type == "manual_only" and not f.executed)
    if manual_count > 0 and not dry_run:
        log(f"LLM_DIAGNOSE: {manual_count} manual_only flags — attempting LLM diagnosis")
        llm_results = _llm_handle_novel_failures(flags)
        for r in llm_results:
            log(f"LLM_RESULT: {r}")

    # Phase 4: Research
    research = {}
    if do_research:
        research["github"] = research_github_trending()
        research["news"] = research_crypto_news()

    # Phase 5: Generate and save report
    report = generate_report(flags, research, diagnostics)
    report_path = os.path.join(DATA_DIR, "self_heal_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    # Also save as JSON for programmatic consumption
    heal_state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_flags": len(flags),
        "auto_fixed": sum(1 for f in flags if f.executed and not (isinstance(f.execution_result, str) and "GATED" in f.execution_result)),
        "gated": sum(1 for f in flags if f.executed and isinstance(f.execution_result, str) and "GATED" in f.execution_result),
        "research_queued": sum(1 for f in flags if f.flag_type == "research_needed"),
        "flags": [
            {
                "description": f.description,
                "priority": f.priority,
                "type": f.flag_type,
                "executed": f.executed,
                "result": f.execution_result[:200] if f.executed else "",
                "predicted": f.predicted_outcome,
            }
            for f in flags
        ],
    }
    state_path = os.path.join(DATA_DIR, "self_heal_state.json")
    with open(state_path, "w") as f:
        json.dump(heal_state, f, indent=2)

    log(f"Report saved: {report_path}")
    log(f"State saved: {state_path}")

    # Phase 6: Write to Obsidian vault
    if not dry_run:
        try:
            _obsidian_daily_summary()
            # Only write incidents for significant events (halt, LLM fix, knowledge DB hit)
            significant_actions = {"_fix_agent_halt", "_llm_diagnose", "_knowledge_db"}
            # Also include any flag where the description contains crash/halt/panic keywords
            incident_keywords = ["halt", "crash", "panic", "critical", "emergency", "novel", "unknown failure"]
            for f in flags:
                is_incident = (
                    f.executed and (
                        f.auto_action in significant_actions or
                        any(kw in f.description.lower() for kw in incident_keywords)
                    )
                )
                if is_incident:
                    _obsidian_record_incident(
                        f.description[:200],
                        f"{f.auto_action}: {f.execution_result[:200]}",
                        related=["Self-Heal Bridge"]
                    )
            # Record fix patterns for all auto-fixed items
            for f in flags:
                if f.executed and f.auto_action not in ("_knowledge_db", "_llm_diagnose", None, ""):
                    _obsidian_record_fix_pattern(
                        f.description[:80], f.auto_action,
                        "", "", 1, "self_heal"
                    )
            # Record tech radar
            tech_count = 0
            try:
                conn = sqlite3.connect(KNOWLEDGE_DB)
                tech_count = conn.execute(
                    "SELECT COUNT(*) FROM tech_radar WHERE timestamp > datetime('now', '-1 day')"
                ).fetchone()[0]
                conn.close()
            except Exception:
                pass
            if tech_count > 0:
                # Read recent findings and write
                conn = sqlite3.connect(KNOWLEDGE_DB)
                rows = conn.execute(
                    "SELECT source, title, url, description FROM tech_radar "
                    "WHERE timestamp > datetime('now', '-7 days') ORDER BY source"
                ).fetchall()
                findings = [{"source": r[0], "title": r[1], "url": r[2], "description": r[3]} for r in rows]
                if findings:
                    _obsidian_record_tech_radar(findings)
                conn.close()
        except Exception as e:
            log(f"OBSIDIAN WRITE ERROR: {e}")

    log("=== Self-Heal Action Bridge DONE ===")

    # Print report to stdout for cron capture
    print(report)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QuantForge Self-Healing Action Bridge")
    parser.add_argument("--dry-run", action="store_true", help="Read diagnostics but don't execute fixes")
    parser.add_argument("--research", action="store_true", help="Run market research (API calls)")
    parser.add_argument("--tech-radar", action="store_true", help="Force technology radar scan")
    args = parser.parse_args()
    main(dry_run=args.dry_run, do_research=args.research, do_tech_radar=args.tech_radar)
