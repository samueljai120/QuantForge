#!/usr/bin/env python3
"""Generate a zero-budget off-production-host status site for QuantForge.

This script is intentionally lightweight so it can run in GitHub Actions or on a
small spare machine. It can either:

1. read QuantForge artifacts from a local directory, or
2. fetch a small allowlist of JSON artifacts over SSH from the production host.

Outputs:
  - runtime/offhost/status/status.json
  - runtime/offhost/status/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen


ARTIFACTS = {
    "doctor": "data/quantforge/doctor-report.json",
    "monitor": "data/quantforge/monitor-report.json",
    "autopilot": "data/quantforge/autopilot-report.json",
    "lanes": "data/quantforge/experiment-lanes.json",
    "review": "data/quantforge/candidate-review.json",
    "last_scan": "data/quantforge/last_scan.json",
    "portfolio": "data/quantforge/portfolio.json",
}


@dataclass
class FetchConfig:
    source_dir: str | None
    prod_host: str | None
    prod_user: str | None
    prod_port: int
    prod_base: str
    dashboard_status_url: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=os.getenv("QUANTFORGE_SOURCE_DIR"))
    parser.add_argument("--prod-host", default=os.getenv("QF_PROD_SSH_HOST"))
    parser.add_argument("--prod-user", default=os.getenv("QF_PROD_SSH_USER", "youruser"))
    parser.add_argument("--prod-port", type=int, default=int(os.getenv("QF_PROD_SSH_PORT", "22")))
    parser.add_argument("--prod-base", default=os.getenv("QF_BASE_DIR", "~/quantforge"))
    parser.add_argument(
        "--dashboard-status-url",
        default=os.getenv("QUANTFORGE_DASHBOARD_STATUS_URL"),
        help="HTTP endpoint for production host dashboard status, e.g. http://your-server-ip:8888/api/status",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("QUANTFORGE_STATUS_OUTPUT_DIR", "runtime/offhost/status"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def ssh_fetch_json(host: str, user: str, port: int, base_dir: str, relpath: str) -> Any:
    command = (
        f"python3 - <<'PY'\n"
        f"from pathlib import Path\n"
        f"p = Path({base_dir!r}).expanduser() / {relpath!r}\n"
        f"print(p.read_text())\n"
        f"PY"
    )
    proc = subprocess.run(
        ["ssh", "-p", str(port), f"{user}@{host}", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"ssh failed for {relpath}")
    return json.loads(proc.stdout)


def http_fetch_json(url: str) -> Any:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode())


def collect_artifacts(cfg: FetchConfig) -> tuple[dict[str, Any], list[str]]:
    payload: dict[str, Any] = {}
    issues: list[str] = []

    if cfg.source_dir:
        source = Path(cfg.source_dir).expanduser()
        for name, relpath in ARTIFACTS.items():
            value = load_json(source / relpath)
            if value is None:
                issues.append(f"Missing or unreadable local artifact: {relpath}")
            payload[name] = value
        return payload, issues

    if cfg.prod_host:
        for name, relpath in ARTIFACTS.items():
            try:
                payload[name] = ssh_fetch_json(
                    cfg.prod_host,
                    cfg.prod_user or "youruser",
                    cfg.prod_port,
                    cfg.prod_base,
                    relpath,
                )
            except Exception as exc:
                payload[name] = None
                issues.append(f"Failed to fetch {relpath}: {exc}")
        return payload, issues

    if cfg.dashboard_status_url:
        try:
            payload["dashboard_status"] = http_fetch_json(cfg.dashboard_status_url)
        except Exception as exc:
            issues.append(f"Failed to fetch dashboard status: {exc}")
        for name in ARTIFACTS:
            payload.setdefault(name, None)
        return payload, issues

    issues.append(
        "No data source configured. Set QUANTFORGE_SOURCE_DIR for local reads or "
        "QF_PROD_SSH_HOST/QF_PROD_SSH_USER for remote fetches, or QUANTFORGE_DASHBOARD_STATUS_URL for dashboard-only status."
    )
    for name in ARTIFACTS:
        payload[name] = None
    return payload, issues


def fmt_ts(value: Any) -> str:
    if not value:
        return "missing"
    return str(value)


def build_summary(artifacts: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    dashboard = artifacts.get("dashboard_status") or {}
    dashboard_qf = dashboard.get("quantforge") or {}
    doctor = artifacts.get("doctor") or {}
    monitor = artifacts.get("monitor") or {}
    autopilot = artifacts.get("autopilot") or {}
    lanes = artifacts.get("lanes") or {}
    review = artifacts.get("review") or {}
    scan = artifacts.get("last_scan") or {}
    portfolio = artifacts.get("portfolio") or {}
    trial = (lanes.get("candidate_trial") or {}) if isinstance(lanes, dict) else {}

    if dashboard_qf:
        doctor = {
            **doctor,
            "readiness": doctor.get("readiness") or dashboard_qf.get("readiness"),
            "autopilot_mode": doctor.get("autopilot_mode") or dashboard_qf.get("autopilot_mode"),
            "monitor_health": doctor.get("monitor_health") or dashboard_qf.get("monitor_health"),
            "failed_count": doctor.get("failed_count") if doctor.get("failed_count") is not None else dashboard_qf.get("failed_count"),
            "check_count": doctor.get("check_count") if doctor.get("check_count") is not None else dashboard_qf.get("check_count"),
            "generated_at": doctor.get("generated_at") or dashboard_qf.get("generated_at"),
        }
        autopilot = {
            **autopilot,
            "mode": autopilot.get("mode") or dashboard_qf.get("autopilot_mode"),
        }
        review = {
            **review,
            "recommendation": review.get("recommendation") or dashboard_qf.get("review_recommendation"),
            "current_mode": review.get("current_mode") or dashboard_qf.get("review_mode"),
        }
        portfolio = {
            **portfolio,
            "updated": portfolio.get("updated") or dashboard_qf.get("portfolio_updated"),
            "total_trades": portfolio.get("total_trades") if portfolio.get("total_trades") is not None else dashboard_qf.get("total_trades"),
            "realized_pnl": portfolio.get("realized_pnl") if portfolio.get("realized_pnl") is not None else dashboard_qf.get("realized_pnl"),
            "cash": portfolio.get("cash") if portfolio.get("cash") is not None else dashboard_qf.get("cash"),
            "positions": portfolio.get("positions") or (
                {} if dashboard_qf.get("open_positions") is None else {
                    f"position_{i+1}": {} for i in range(int(dashboard_qf.get("open_positions") or 0))
                }
            ),
        }
        if not trial:
            trial = dashboard_qf.get("trial") or {}
        scan = {
            **scan,
            "generated_at": scan.get("generated_at") or (dashboard_qf.get("scan") or {}).get("generated_at"),
            "pick_count": scan.get("pick_count") if scan.get("pick_count") is not None else (dashboard_qf.get("scan") or {}).get("pick_count"),
            "signal_count": scan.get("signal_count") if scan.get("signal_count") is not None else (dashboard_qf.get("scan") or {}).get("signal_count"),
            "result_count": scan.get("result_count") if scan.get("result_count") is not None else (dashboard_qf.get("scan") or {}).get("result_count"),
            "top_picks": scan.get("top_picks") or (dashboard_qf.get("scan") or {}).get("top_picks") or [],
            "blocked_reasons": scan.get("blocked_reasons") or (dashboard_qf.get("scan") or {}).get("blocked_reasons") or [],
        }

    blocked_reasons = (((scan.get("summary") or {}).get("blocked_reasons")) or [])[:8]
    if isinstance(blocked_reasons, dict):
        blocked_reasons = [
            {"reason": key, "count": value} for key, value in blocked_reasons.items()
        ]
    elif not blocked_reasons:
        blocked_reasons = (scan.get("blocked_reasons") or [])[:8]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ok": not issues,
        "issues": issues,
        "doctor": {
            "readiness": doctor.get("readiness"),
            "autopilot_mode": doctor.get("autopilot_mode"),
            "monitor_health": doctor.get("monitor_health"),
            "failed_count": doctor.get("failed_count"),
            "check_count": doctor.get("check_count"),
        },
        "trial": {
            "candidate_id": trial.get("candidate_id"),
            "type": trial.get("type"),
            "status": trial.get("status"),
            "assessment": trial.get("assessment"),
            "cycles_run": trial.get("cycles_run"),
            "max_cycles": trial.get("max_cycles"),
            "queued_at": trial.get("queued_at"),
            "started_at": trial.get("started_at"),
            "completed_at": trial.get("completed_at"),
            "expires_at": trial.get("expires_at"),
        },
        "review": {
            "recommendation": review.get("recommendation"),
            "current_mode": review.get("current_mode"),
            "reasons": review.get("reasons") or [],
        },
        "autopilot": {
            "mode": autopilot.get("mode"),
            "actions": autopilot.get("actions") or [],
            "reasons": autopilot.get("reasons") or [],
        },
        "scan": {
            "generated_at": scan.get("generated_at") or scan.get("ts"),
            "pick_count": scan.get("pick_count"),
            "signal_count": ((scan.get("summary") or {}).get("counts") or {}).get("signals", scan.get("signal_count")),
            "result_count": ((scan.get("summary") or {}).get("counts") or {}).get("results", scan.get("result_count")),
            "top_picks": scan.get("top_picks") or [],
            "blocked_reasons": blocked_reasons,
        },
        "portfolio": {
            "updated": portfolio.get("updated"),
            "total_trades": portfolio.get("total_trades"),
            "realized_pnl": portfolio.get("realized_pnl"),
            "cash": portfolio.get("cash"),
            "open_positions": len((portfolio.get("positions") or {})),
        },
        "raw_timestamps": {
            "doctor": doctor.get("generated_at"),
            "monitor": monitor.get("generated_at"),
            "autopilot": autopilot.get("generated_at"),
            "review": review.get("generated_at"),
            "scan": scan.get("generated_at") or scan.get("ts"),
            "portfolio": portfolio.get("updated"),
        },
    }


def render_list(items: list[Any], key: str | None = None) -> str:
    if not items:
        return "<li>None</li>"
    rows = []
    for item in items:
        if isinstance(item, dict):
            if key and key in item:
                label = item.get(key)
                count = item.get("count")
                if count is not None:
                    rows.append(f"<li>{html.escape(str(label))} ({count})</li>")
                else:
                    rows.append(f"<li>{html.escape(str(label))}</li>")
            else:
                rows.append(f"<li><code>{html.escape(json.dumps(item, sort_keys=True))}</code></li>")
        else:
            rows.append(f"<li>{html.escape(str(item))}</li>")
    return "\n".join(rows)


def render_html(summary: dict[str, Any]) -> str:
    doctor = summary["doctor"]
    trial = summary["trial"]
    review = summary["review"]
    autopilot = summary["autopilot"]
    scan = summary["scan"]
    portfolio = summary["portfolio"]
    issues = summary["issues"]
    readiness = doctor.get("readiness") or "UNKNOWN"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuantForge Zero-Budget Status</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f3ec;
      --panel: #fffdf7;
      --ink: #18212b;
      --muted: #5a6470;
      --border: #d8d0c1;
      --accent: #0d6b55;
      --warn: #a45900;
      --bad: #a22222;
    }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #f8f6ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p {{
      color: var(--muted);
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 20px 0 28px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(24, 33, 43, 0.05);
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .big {{
      font-size: 30px;
      font-weight: 700;
    }}
    .good {{ color: var(--accent); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      background: #f0ece3;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    ul {{
      padding-left: 18px;
      margin: 10px 0 0;
    }}
    .section {{
      margin-top: 28px;
    }}
  </style>
</head>
<body>
<main>
  <h1>QuantForge Zero-Budget Status</h1>
  <p>Generated at {html.escape(summary["generated_at"])}. This page is designed for low-cost remote visibility without adding load to the production host.</p>

  <div class="grid">
    <div class="card">
      <div class="eyebrow">Doctor</div>
      <div class="big {'good' if readiness == 'READY' else 'warn' if readiness == 'PAUSED' else 'bad'}">{html.escape(readiness)}</div>
      <p>Checks failed: {doctor.get("failed_count", "missing")} / {doctor.get("check_count", "missing")}</p>
    </div>
    <div class="card">
      <div class="eyebrow">Autopilot</div>
      <div class="big">{html.escape(str(doctor.get("autopilot_mode") or autopilot.get("mode") or "missing"))}</div>
      <p>Monitor health: {html.escape(str(doctor.get("monitor_health") or "missing"))}</p>
    </div>
    <div class="card">
      <div class="eyebrow">Trial</div>
      <div class="big">{html.escape(str(trial.get("status") or "missing"))}</div>
      <p>{html.escape(str(trial.get("type") or "missing"))} | {html.escape(str(trial.get("cycles_run") or 0))}/{html.escape(str(trial.get("max_cycles") or 0))}</p>
    </div>
    <div class="card">
      <div class="eyebrow">Portfolio</div>
      <div class="big">{html.escape(str(portfolio.get("total_trades") or 0))} trades</div>
      <p>Open positions: {html.escape(str(portfolio.get("open_positions") or 0))} | Realized PnL: {html.escape(str(portfolio.get("realized_pnl") or 0))}</p>
    </div>
  </div>

  <div class="section card">
    <h2>Setup / Fetch Issues</h2>
    <ul>
      {render_list(issues)}
    </ul>
  </div>

  <div class="section grid">
    <div class="card">
      <h2>Review</h2>
      <p>Recommendation: <code>{html.escape(str(review.get("recommendation") or "missing"))}</code></p>
      <ul>{render_list(review.get("reasons") or [])}</ul>
    </div>
    <div class="card">
      <h2>Autopilot Reasons</h2>
      <ul>{render_list(autopilot.get("reasons") or [])}</ul>
    </div>
  </div>

  <div class="section grid">
    <div class="card">
      <h2>Scan Summary</h2>
      <p>Generated: <code>{html.escape(fmt_ts(scan.get("generated_at")))}</code></p>
      <p>Signals: <code>{html.escape(str(scan.get("signal_count") or 0))}</code> | Results: <code>{html.escape(str(scan.get("result_count") or 0))}</code></p>
      <p>Pick count: <code>{html.escape(str(scan.get("pick_count") or 0))}</code></p>
      <h3>Blocked Reasons</h3>
      <ul>{render_list(scan.get("blocked_reasons") or [], key="reason")}</ul>
    </div>
    <div class="card">
      <h2>Top Picks</h2>
      <ul>{render_list(scan.get("top_picks") or [], key="symbol")}</ul>
    </div>
  </div>

  <div class="section card">
    <h2>Timestamps</h2>
    <ul>
      <li>Doctor: <code>{html.escape(fmt_ts(summary["raw_timestamps"].get("doctor")))}</code></li>
      <li>Monitor: <code>{html.escape(fmt_ts(summary["raw_timestamps"].get("monitor")))}</code></li>
      <li>Autopilot: <code>{html.escape(fmt_ts(summary["raw_timestamps"].get("autopilot")))}</code></li>
      <li>Review: <code>{html.escape(fmt_ts(summary["raw_timestamps"].get("review")))}</code></li>
      <li>Last scan: <code>{html.escape(fmt_ts(summary["raw_timestamps"].get("scan")))}</code></li>
      <li>Portfolio: <code>{html.escape(fmt_ts(summary["raw_timestamps"].get("portfolio")))}</code></li>
    </ul>
  </div>
</main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    cfg = FetchConfig(
        source_dir=args.source_dir,
        prod_host=args.prod_host,
        prod_user=args.prod_user,
        prod_port=args.prod_port,
        prod_base=args.prod_base,
        dashboard_status_url=args.dashboard_status_url,
    )
    artifacts, issues = collect_artifacts(cfg)
    summary = build_summary(artifacts, issues)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "index.html").write_text(render_html(summary))
    print(f"Wrote {output_dir / 'status.json'}")
    print(f"Wrote {output_dir / 'index.html'}")
    if issues:
        print("Completed with issues:")
        for issue in issues:
            print(f"  - {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
