#!/usr/bin/env python3
"""QuantForge — daily audit snapshot with integrity manifest + verify/restore.

Copies the audit-critical ledgers into data/quantforge/audit-snapshots/<UTC-date>/
with a SHA-256 manifest, and prunes snapshots older than RETAIN_DAYS.

Why plain copies (not tar): the Oracle offload mirror already syncs every
*.json/*.jsonl under data/quantforge/, so dated snapshot dirs ride the existing
off-host sync with zero Oracle-side changes. A tampered or corrupted live file
can be detected against (and restored from) the dated manifests.

Idempotent: re-running on the same UTC day refreshes that day's snapshot.

Subcommands:
  snapshot           write today's dated snapshot + manifest (default; cron path).
  verify <date>      recompute sha256 of each manifest file, print per-file
                     OK/MISMATCH/MISSING and overall PASS/FAIL. Exit nonzero on
                     any mismatch/missing.
  restore <date> [filename] [--yes]
                     restore snapshot file(s) back into DATA_DIR. Without --yes it
                     only PREVIEWS what would be overwritten. With --yes it backs
                     up each current target to <target>.pre-restore-<date> first.

Running with NO args (as cron does) behaves exactly like `snapshot`.
"""

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

DATA_DIR = os.path.expanduser("~/quantforge/data/quantforge")
SNAP_BASE = os.path.join(DATA_DIR, "audit-snapshots")
RETAIN_DAYS = 30

AUDIT_FILES = [
    "paper-trades.jsonl",
    "portfolio.json",
    "agent_trades.jsonl",
    "agent_portfolio.json",
    "reflect_decisions.jsonl",
    "allocator_decisions.jsonl",
    "qf_strategy_params.json",
    "strategy-params.json",
    "governance-report.json",
    "autopilot-report.json",
    # additions : barbell satellite + autonomous research loop
    "moonshot_state.json",
    "research_ledger.jsonl",
    "model/rebuild_verdict.json",
    "model/promotion_candidate.json",
    # S1 : cost-inclusive (time + fee honest) agent report
    "agent-cost-inclusive-report.json",
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_snapshot() -> int:
    """Existing scheduled behavior: write today's dated snapshot + manifest, prune."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap_dir = os.path.join(SNAP_BASE, today)
    os.makedirs(snap_dir, exist_ok=True)

    manifest = {"snapshot_date": today, "generated_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    copied = 0
    for name in AUDIT_FILES:
        src = os.path.join(DATA_DIR, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(snap_dir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        manifest["files"][name] = {
            "sha256": sha256_of(dst),
            "bytes": os.path.getsize(dst),
        }
        copied += 1

    with open(os.path.join(snap_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Prune old snapshots
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETAIN_DAYS)).strftime("%Y-%m-%d")
    pruned = 0
    if os.path.isdir(SNAP_BASE):
        for entry in sorted(os.listdir(SNAP_BASE)):
            full = os.path.join(SNAP_BASE, entry)
            if os.path.isdir(full) and len(entry) == 10 and entry < cutoff:
                shutil.rmtree(full, ignore_errors=True)
                pruned += 1

    print(f"audit snapshot {today}: {copied} files, manifest written, {pruned} old snapshots pruned")
    return 0


def _load_manifest(date: str):
    """Return (snap_dir, manifest_dict) for a date, or (snap_dir, None) if absent/bad."""
    snap_dir = os.path.join(SNAP_BASE, date)
    mpath = os.path.join(snap_dir, "manifest.json")
    if not os.path.isfile(mpath):
        return snap_dir, None
    try:
        with open(mpath) as f:
            return snap_dir, json.load(f)
    except (json.JSONDecodeError, OSError):
        return snap_dir, None


def cmd_verify(date: str) -> int:
    """Recompute sha256 of each manifest file; print per-file status + overall PASS/FAIL."""
    snap_dir, manifest = _load_manifest(date)
    if manifest is None:
        print(f"VERIFY {date}: FAIL — no readable manifest.json at {snap_dir}")
        return 2

    files = manifest.get("files", {})
    if not files:
        print(f"VERIFY {date}: FAIL — manifest lists no files")
        return 2

    ok = mismatch = missing = 0
    for name, meta in sorted(files.items()):
        expected = meta.get("sha256", "")
        path = os.path.join(snap_dir, name)
        if not os.path.isfile(path):
            print(f"  MISSING  {name}")
            missing += 1
            continue
        actual = sha256_of(path)
        if actual == expected:
            print(f"  OK       {name}")
            ok += 1
        else:
            print(f"  MISMATCH {name}")
            print(f"             expected {expected}")
            print(f"             actual   {actual}")
            mismatch += 1

    overall = "PASS" if (mismatch == 0 and missing == 0) else "FAIL"
    print(f"VERIFY {date}: {overall} — {ok} ok, {mismatch} mismatch, {missing} missing")
    return 0 if overall == "PASS" else 1


def cmd_restore(date: str, filename: str, do_it: bool) -> int:
    """Restore snapshot file(s) into DATA_DIR.

    Preview-only unless do_it is True. When restoring, the current live target is
    first backed up to <target>.pre-restore-<date>. A specific filename limits the
    restore to one entry; otherwise every manifest file is restored.
    """
    snap_dir, manifest = _load_manifest(date)
    if manifest is None:
        print(f"RESTORE {date}: ERROR — no readable manifest.json at {snap_dir}")
        return 2

    files = manifest.get("files", {})
    if filename:
        if filename not in files:
            print(f"RESTORE {date}: ERROR — '{filename}' not in manifest. Available:")
            for name in sorted(files):
                print(f"    {name}")
            return 2
        targets = [filename]
    else:
        targets = sorted(files)

    if not targets:
        print(f"RESTORE {date}: nothing to do — manifest lists no files")
        return 0

    mode = "RESTORING" if do_it else "DRY-RUN (no --yes; nothing written)"
    print(f"RESTORE {date}: {mode}")

    restored = skipped = 0
    for name in targets:
        snap_file = os.path.join(snap_dir, name)
        live_target = os.path.join(DATA_DIR, name)
        if not os.path.isfile(snap_file):
            print(f"  SKIP     {name} — snapshot copy missing at {snap_file}")
            skipped += 1
            continue

        exists = os.path.exists(live_target)
        if exists:
            backup = f"{live_target}.pre-restore-{date}"
            print(f"  OVERWRITE {name}")
            print(f"             live   {live_target}")
            print(f"             backup {backup}")
        else:
            print(f"  CREATE   {name}")
            print(f"             live   {live_target} (no existing file)")

        if do_it:
            os.makedirs(os.path.dirname(live_target), exist_ok=True)
            if exists:
                shutil.copy2(live_target, f"{live_target}.pre-restore-{date}")
            shutil.copy2(snap_file, live_target)
            restored += 1

    if do_it:
        print(f"RESTORE {date}: done — {restored} restored, {skipped} skipped")
    else:
        print(f"RESTORE {date}: preview only — re-run with --yes to apply "
              f"({len(targets) - skipped} would change, {skipped} skipped)")
    return 0


def _usage() -> int:
    print(
        "usage:\n"
        "  quantforge_audit_snapshot.py [snapshot]            write today's snapshot (default)\n"
        "  quantforge_audit_snapshot.py verify <date>         verify a snapshot's integrity\n"
        "  quantforge_audit_snapshot.py restore <date> [file] [--yes]\n"
        "                                                     restore file(s); --yes to apply\n"
        "  <date> format: YYYY-MM-DD",
        file=sys.stderr,
    )
    return 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Default / explicit snapshot: cron runs with NO args and must hit this path.
    if not argv or argv[0] == "snapshot":
        return cmd_snapshot()

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "verify":
        if len(rest) != 1:
            print("verify requires exactly one <date> argument", file=sys.stderr)
            return _usage()
        return cmd_verify(rest[0])

    if cmd == "restore":
        do_it = "--yes" in rest
        positionals = [a for a in rest if a != "--yes"]
        if not positionals:
            print("restore requires a <date> argument", file=sys.stderr)
            return _usage()
        date = positionals[0]
        filename = positionals[1] if len(positionals) > 1 else ""
        return cmd_restore(date, filename, do_it)

    print(f"unknown command: {cmd}", file=sys.stderr)
    return _usage()


if __name__ == "__main__":
    sys.exit(main())
