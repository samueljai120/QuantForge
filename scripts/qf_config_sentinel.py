#!/usr/bin/env python3
"""QuantForge CONFIG SENTINEL — fail-closed config-drift / approval gate.

The recurring failure mode in this project is *"deployed != live"*: a fix (or a
threshold) that everyone believes is active turns out not to be on the live file,
or a live param file gets hand-edited to a value the harvester would silently
clamp. The harvester already clamps `carry_policy.json` to hard bounds, but
nothing INDEPENDENTLY asserts that the live file the human approved is still the
live file in force, nor records when it drifts.

This sentinel is that independent assertion. It:
  1. Loads the LIVE carry policy and an APPROVED baseline.
  2. Holds every guarded key (enter / exit / min_hold) to the APPROVED bounds —
     defense in depth: even if the harvester's in-code bounds were widened, the
     live values must still sit inside the bounds the human signed off on here.
  3. RECORDS any drift from the approved snapshot (allowed if still in-bounds,
     because the weekly self-tuner legitimately moves within bounds) so a human
     can see exactly what moved and when.
  4. Writes a machine-readable artifact and appends a tamper-evident decision-log
     entry for auditability.

Fail-closed: a missing/corrupt baseline, a missing key, a non-numeric value, or
ANY out-of-bounds value => FAIL (exit non-zero). When in doubt it REJECTS.

It is **read-only** on trading state: it never changes `carry_policy.json` or any
live param. The only write path that touches the baseline is `--approve`, which is
an explicit, logged, HUMAN-gated action (recording the current live config as the
new approved reference).

CLI:
  qf_config_sentinel.py                 # check; exit 0 PASS / non-zero FAIL
  qf_config_sentinel.py --json          # full verdict as JSON
  qf_config_sentinel.py --approve "<reason>"   # seed/update approved baseline (human-gated)
"""
from __future__ import annotations

import ast
import json
import os
import sys
from datetime import datetime, timezone

DATA = os.path.expanduser("~/quantforge/data/quantforge")
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
LIVE_POLICY = os.path.join(DATA, "carry_policy.json")
BASELINE = os.path.join(DATA, "approved_config_baseline.json")
ARTIFACT = os.path.join(DATA, "config_sentinel.json")
SENTINEL_LOG = os.path.join(DATA, "config_sentinel_log.jsonl")
# The harvester is where the live clamp actually lives. The sentinel asserts the
# bounds in THIS source file still equal the approved bounds — closing the
# "deployed != live" gap for the safety-critical clamp itself.
HARVESTER_SRC = os.path.join(SCRIPTS, "quantforge_carry_harvester.py")
HARVESTER_BOUNDS_NAME = "_POLICY_BOUNDS"

# Guarded keys + their hard, APPROVED safety bounds. These MUST stay in sync with
# the harvester's _POLICY_BOUNDS (quantforge_carry_harvester.py). Two independent
# copies is the point: a widening of one is caught by the other.
GUARDED_BOUNDS = {
    "enter":    (0.0010, 0.0050),
    "exit":     (0.0004, 0.0030),
    "min_hold": (3, 12),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _ast_node_to_bounds(node):
    """Convert an AST dict-of-numeric-pairs node to {key: (lo, hi)} reading ONLY
    constant nodes (no code execution at all). Returns None on any non-literal."""
    if not isinstance(node, ast.Dict):
        return None
    out = {}
    for k, v in zip(node.keys, node.values):
        if not isinstance(k, ast.Constant) or not isinstance(v, (ast.Tuple, ast.List)):
            return None
        nums = []
        for el in v.elts:
            if isinstance(el, ast.Constant) and isinstance(el.value, (int, float)):
                nums.append(el.value)
            elif (isinstance(el, ast.UnaryOp) and isinstance(el.op, ast.USub)
                  and isinstance(el.operand, ast.Constant)):
                nums.append(-el.operand.value)
            else:
                return None
        if len(nums) < 2:
            return None
        out[k.value] = (nums[0], nums[1])
    return out


def live_code_bounds(harvester_path: str = HARVESTER_SRC,
                     name: str = HARVESTER_BOUNDS_NAME):
    """Parse the harvester's `_POLICY_BOUNDS` literal WITHOUT executing the module
    (no import side effects). Returns {key: (lo, hi)} or None if the source is
    unreadable or the literal is absent/non-literal. None => fail-closed upstream."""
    try:
        with open(harvester_path) as f:
            tree = ast.parse(f.read())
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return _ast_node_to_bounds(node.value)
    return None


def _norm_bounds(b):
    """Normalize a bounds mapping to {key: (lo, hi)} for equality comparison."""
    return {k: (v[0], v[1]) for k, v in b.items()} if b else {}


def evaluate(live_policy, baseline, bounds=None, code_bounds="__skip__") -> dict:
    """Pure verdict core. Takes plain dicts, returns a verdict dict. Never raises
    on policy *content* — every problem becomes a failed check (fail-closed).

    Bounds source-of-truth precedence: the baseline's ``approved_bounds`` (what the
    human signed off on) if present, else the module default ``GUARDED_BOUNDS``.

    ``code_bounds`` is the live harvester's parsed ``_POLICY_BOUNDS`` (or None if it
    could not be parsed). Pass the sentinel value ``"__skip__"`` (default) to skip
    the code-consistency check entirely — preserves callers that only check values.
    When a real value (dict or None) is passed, the live code bounds MUST equal the
    approved bounds, closing the "deployed != live" gap on the clamp itself."""
    checks = []
    violations = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            violations.append(name)

    base_pol = baseline.get("carry_policy", {}) if isinstance(baseline, dict) else {}
    # Fail-closed: no approved baseline => no approval => reject.
    add("baseline_present",
        isinstance(baseline, dict) and bool(base_pol),
        "approved baseline missing or has no carry_policy block")

    # Bounds source of truth: approved_bounds from the baseline if present, else
    # the module default. This is what value-in-bounds AND code-consistency use.
    approved_bounds = None
    if isinstance(baseline, dict) and baseline.get("approved_bounds"):
        approved_bounds = _norm_bounds(baseline["approved_bounds"])
    if approved_bounds is None:
        approved_bounds = dict(GUARDED_BOUNDS) if bounds is None else dict(bounds)
    bounds = approved_bounds

    # CODE-CONSISTENCY: the live harvester clamp must equal the approved bounds.
    if code_bounds != "__skip__":
        if code_bounds is None:
            add("code_bounds_parsed", False,
                "could not parse live harvester _POLICY_BOUNDS (source missing/changed)")
        else:
            live_cb = _norm_bounds(code_bounds)
            for k, ab in approved_bounds.items():
                add(f"code_bounds_match:{k}", live_cb.get(k) == ab,
                    f"live code bound for '{k}'={live_cb.get(k)} != approved {ab}")

    drift = {}
    live_ok = isinstance(live_policy, dict)
    add("live_is_object", live_ok, "live policy is not a JSON object")

    for k, (lo, hi) in bounds.items():
        present = live_ok and k in live_policy
        add(f"key_present:{k}", present, f"live policy missing guarded key '{k}'")
        if not present:
            continue
        v = live_policy[k]
        numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
        add(f"numeric:{k}", numeric, f"'{k}'={v!r} is not numeric")
        if not numeric:
            continue
        add(f"in_bounds:{k}", lo <= v <= hi,
            f"'{k}'={v} outside approved bounds [{lo}, {hi}]")
        bv = base_pol.get(k)
        if bv is not None and v != bv:
            drift[k] = {"baseline": bv, "live": v}

    pass_all = not violations
    return {
        "status": "PASS" if pass_all else "FAIL",
        "pass_all": pass_all,
        "checks": checks,
        "violations": violations,
        "drift": drift,
        "drift_detected": bool(drift),
        "guarded_keys": list(bounds),
    }


def _hard_fail(reason: str) -> dict:
    return {
        "status": "FAIL",
        "pass_all": False,
        "checks": [{"name": "load", "ok": False, "detail": reason}],
        "violations": ["load"],
        "drift": {},
        "drift_detected": False,
        "guarded_keys": list(GUARDED_BOUNDS),
    }


def _atomic_write(path: str, obj) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _log_decision(verdict: dict, log_path: str = SENTINEL_LOG) -> bool:
    """Best-effort tamper-evident audit entry. Never fails the gate. Returns True
    iff an entry was actually written (so callers/tests can assert it wired up)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from qf_safety.decision_log import DecisionLog  # type: ignore

        DecisionLog(log_path).append({
            "kind": "config_sentinel",
            "status": verdict.get("status"),
            "violations": verdict.get("violations"),
            "drift": verdict.get("drift"),
        })
        return True
    except Exception:
        return False


def run(live_path=LIVE_POLICY, baseline_path=BASELINE,
        artifact_path=ARTIFACT, write=True, harvester_path=HARVESTER_SRC) -> dict:
    """Load live + baseline (fail-closed on read errors), evaluate, persist."""
    try:
        live = _load_json(live_path)
    except Exception as e:  # missing / corrupt live file => reject
        verdict = _hard_fail(f"live policy unreadable ({live_path}): {e}")
    else:
        try:
            baseline = _load_json(baseline_path)
        except Exception as e:  # no approved baseline => reject (must seed via --approve)
            verdict = _hard_fail(
                f"approved baseline unreadable ({baseline_path}): {e} "
                f"— establish one with: qf_config_sentinel.py --approve '<reason>'")
        else:
            verdict = evaluate(live, baseline,
                               code_bounds=live_code_bounds(harvester_path))

    verdict["ts"] = _now()
    verdict["live_path"] = live_path
    verdict["baseline_path"] = baseline_path
    if write:
        try:
            _atomic_write(artifact_path, verdict)
        except Exception:
            pass
        _log_decision(verdict)
    return verdict


def approve(reason: str, live_path=LIVE_POLICY, baseline_path=BASELINE) -> dict:
    """HUMAN-gated: record the current live guarded keys as the new approved
    baseline. Writes an audit file only — never touches the live policy."""
    live = _load_json(live_path)
    snapshot = {k: live[k] for k in GUARDED_BOUNDS if k in live}
    baseline = {
        "carry_policy": snapshot,
        "approved_bounds": {k: list(v) for k, v in GUARDED_BOUNDS.items()},
        "reason": reason,
        "approved_at": _now(),
        "source_live_path": live_path,
    }
    _atomic_write(baseline_path, baseline)
    _log_decision({"status": "BASELINE_APPROVED", "violations": [],
                   "drift": {"approved": snapshot}})
    return baseline


def _summary(v: dict) -> str:
    head = f"CONFIG SENTINEL: {v['status']}"
    if v["violations"]:
        head += "  violations=" + ",".join(v["violations"])
    if v.get("drift_detected"):
        head += f"  drift={v['drift']}"
    return head


def main(argv) -> int:
    if "--approve" in argv:
        i = argv.index("--approve")
        reason = argv[i + 1] if i + 1 < len(argv) else "manual approval"
        b = approve(reason)
        print(f"BASELINE APPROVED: {b['carry_policy']}  reason={reason!r}")
        return 0
    v = run()
    if "--json" in argv:
        print(json.dumps(v, indent=2, sort_keys=True))
    else:
        print(_summary(v))
    return 0 if v["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
