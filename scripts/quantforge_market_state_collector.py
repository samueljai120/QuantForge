#!/usr/bin/env python3
"""QuantForge market-state collector.

Refreshes lightweight breadth and top-of-book context together, then emits a
single compact report that the dashboard or operator tooling can read without
opening multiple artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

BASE_DIR = os.path.join(cfg.data, "quantforge")
OUTPUT_FILE = os.path.join(BASE_DIR, "market-state-report.json")
BREADTH_REPORT_FILE = os.path.join(BASE_DIR, "breadth", "breadth-report.json")
BOOK_REPORT_FILE = os.path.join(BASE_DIR, "book", "spread-depth-report.json")


def read_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_step(script_name: str) -> dict:
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return {
        "script": script_name,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip().splitlines()[-5:],
        "stderr": (proc.stderr or "").strip().splitlines()[-5:],
    }


def build_report() -> dict:
    breadth_step = run_step("quantforge_market_breadth_context_collector.py")
    book_step = run_step("quantforge_top_of_book_snapshot_collector.py")
    breadth = read_json(BREADTH_REPORT_FILE)
    book = read_json(BOOK_REPORT_FILE)
    status = "ready" if breadth.get("status") == "ready" and book.get("status") == "ready" else "partial"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "steps": [breadth_step, book_step],
        "breadth": breadth,
        "top_of_book": book,
    }


def main() -> None:
    cfg.require_production_runtime("quantforge_market_state_collector.py")
    os.makedirs(BASE_DIR, exist_ok=True)
    payload = build_report()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("QuantForge market state collector")
    print(f"Status: {payload['status']}")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
