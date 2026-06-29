"""Tests for qf_config_sentinel — the fail-closed config-drift / approval gate.

Load-bearing negatives: the gate MUST reject out-of-bounds values, a missing
baseline, missing keys, and corrupt files. The positive cases prove it does not
false-fail on the validated live config or on a legitimate in-bounds self-tune.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import qf_config_sentinel as cs  # noqa: E402


# Approved baseline mirrors the validated live config (enter=0.10%, exit=0.05%, hold=12).
APPROVED = {"carry_policy": {"enter": 0.001, "exit": 0.0005, "min_hold": 12}}
LIVE_OK = {"enter": 0.001, "exit": 0.0005, "min_hold": 12, "tuned_from": "sweep"}


# ── positive ────────────────────────────────────────────────────────────────
def test_in_bounds_matching_baseline_passes():
    v = cs.evaluate(LIVE_OK, APPROVED)
    assert v["pass_all"] is True
    assert v["status"] == "PASS"
    assert v["drift_detected"] is False


def test_in_bounds_drift_passes_but_is_recorded():
    """The weekly self-tuner may move within bounds — allowed, but logged."""
    live = {"enter": 0.0015, "exit": 0.0008, "min_hold": 6}  # in-bounds, != baseline
    v = cs.evaluate(live, APPROVED)
    assert v["pass_all"] is True
    assert v["drift_detected"] is True
    assert v["drift"]["enter"] == {"baseline": 0.001, "live": 0.0015}


# ── fail-closed negatives ────────────────────────────────────────────────────
@pytest.mark.parametrize("key,bad", [
    ("enter", 0.0009),    # below 0.0010
    ("enter", 0.0051),    # above 0.0050
    ("exit", 0.0003),     # below 0.0004
    ("min_hold", 2),      # below 3
    ("min_hold", 13),     # above 12
])
def test_out_of_bounds_fails(key, bad):
    live = dict(LIVE_OK, **{key: bad})
    v = cs.evaluate(live, APPROVED)
    assert v["pass_all"] is False
    assert f"in_bounds:{key}" in v["violations"]


def test_missing_baseline_fails_closed():
    v = cs.evaluate(LIVE_OK, {})
    assert v["pass_all"] is False
    assert "baseline_present" in v["violations"]


def test_missing_guarded_key_fails():
    live = {"enter": 0.001, "exit": 0.0005}  # no min_hold
    v = cs.evaluate(live, APPROVED)
    assert v["pass_all"] is False
    assert "key_present:min_hold" in v["violations"]


def test_non_numeric_value_fails():
    live = dict(LIVE_OK, enter="oops")
    v = cs.evaluate(live, APPROVED)
    assert v["pass_all"] is False
    assert "numeric:enter" in v["violations"]


def test_bool_is_not_numeric():
    live = dict(LIVE_OK, min_hold=True)
    v = cs.evaluate(live, APPROVED)
    assert v["pass_all"] is False


# ── run() with files: corrupt + missing => hard fail ─────────────────────────
def test_run_corrupt_live_fails(tmp_path):
    live = tmp_path / "carry_policy.json"
    live.write_text("{ not valid json ")
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps(APPROVED))
    v = cs.run(live_path=str(live), baseline_path=str(base),
               artifact_path=str(tmp_path / "art.json"), write=True)
    assert v["pass_all"] is False
    assert "load" in v["violations"]


def test_run_missing_baseline_fails(tmp_path):
    live = tmp_path / "carry_policy.json"
    live.write_text(json.dumps(LIVE_OK))
    v = cs.run(live_path=str(live), baseline_path=str(tmp_path / "nope.json"),
               artifact_path=str(tmp_path / "art.json"), write=True)
    assert v["pass_all"] is False


def _stub_harvester(tmp_path):
    src = tmp_path / "harv.py"
    src.write_text(HARVESTER_STUB)
    return str(src)


def test_run_happy_path_writes_artifact(tmp_path):
    live = tmp_path / "carry_policy.json"
    live.write_text(json.dumps(LIVE_OK))
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps(APPROVED_FULL))
    art = tmp_path / "art.json"
    v = cs.run(live_path=str(live), baseline_path=str(base), artifact_path=str(art),
               write=True, harvester_path=_stub_harvester(tmp_path))
    assert v["pass_all"] is True
    written = json.loads(art.read_text())
    assert written["status"] == "PASS"
    assert "ts" in written


# ── code↔approval bounds consistency (the "deployed != live" guard) ──────────
APPROVED_FULL = {
    "carry_policy": {"enter": 0.001, "exit": 0.0005, "min_hold": 12},
    "approved_bounds": {"enter": [0.001, 0.005], "exit": [0.0004, 0.003],
                        "min_hold": [3, 12]},
}
HARVESTER_STUB = (
    'X = 1\n'
    '_POLICY_BOUNDS = {"enter": (0.0010, 0.0050), "exit": (0.0004, 0.0030), '
    '"min_hold": (3, 12)}\n'
    'def foo():\n    return _POLICY_BOUNDS\n'
)


def test_live_code_bounds_parses_literal(tmp_path):
    src = tmp_path / "harv.py"
    src.write_text(HARVESTER_STUB)
    b = cs.live_code_bounds(str(src))
    assert b == {"enter": (0.001, 0.005), "exit": (0.0004, 0.003), "min_hold": (3, 12)}


def test_live_code_bounds_missing_literal_returns_none(tmp_path):
    src = tmp_path / "harv.py"
    src.write_text("Y = 2\n")
    assert cs.live_code_bounds(str(src)) is None


def test_live_code_bounds_unreadable_returns_none(tmp_path):
    assert cs.live_code_bounds(str(tmp_path / "nope.py")) is None


def test_code_bounds_match_passes():
    code = {"enter": (0.001, 0.005), "exit": (0.0004, 0.003), "min_hold": (3, 12)}
    v = cs.evaluate(LIVE_OK, APPROVED_FULL, code_bounds=code)
    assert v["pass_all"] is True


def test_code_bounds_widened_fails_closed():
    """Live code clamp widened beyond approved => caught (deployed != live)."""
    code = {"enter": (0.001, 0.01), "exit": (0.0004, 0.003), "min_hold": (3, 12)}
    v = cs.evaluate(LIVE_OK, APPROVED_FULL, code_bounds=code)
    assert v["pass_all"] is False
    assert "code_bounds_match:enter" in v["violations"]


def test_code_bounds_unparseable_fails_closed():
    v = cs.evaluate(LIVE_OK, APPROVED_FULL, code_bounds=None)
    assert v["pass_all"] is False
    assert "code_bounds_parsed" in v["violations"]


def test_approved_bounds_drive_value_check():
    """A tighter approved bound in the baseline overrides the module default."""
    tight = {"carry_policy": {"enter": 0.001},
             "approved_bounds": {"enter": [0.001, 0.0012], "exit": [0.0004, 0.003],
                                 "min_hold": [3, 12]}}
    live = {"enter": 0.002, "exit": 0.0005, "min_hold": 6}  # 0.002 > tight hi 0.0012
    v = cs.evaluate(live, tight)
    assert v["pass_all"] is False
    assert "in_bounds:enter" in v["violations"]


def test_skip_default_preserves_value_only_callers():
    """Default code_bounds='__skip__' must not add code-consistency violations."""
    v = cs.evaluate(LIVE_OK, APPROVED)  # no approved_bounds, no code_bounds
    assert v["pass_all"] is True
    assert not any(c["name"].startswith("code_bounds") for c in v["checks"])


def test_approve_seeds_baseline_from_live(tmp_path):
    live = tmp_path / "carry_policy.json"
    live.write_text(json.dumps(LIVE_OK))
    base = tmp_path / "baseline.json"
    b = cs.approve("test seed", live_path=str(live), baseline_path=str(base))
    assert b["carry_policy"] == {"enter": 0.001, "exit": 0.0005, "min_hold": 12}
    # and the seeded baseline makes the live config pass (approve writes approved_bounds)
    v = cs.run(live_path=str(live), baseline_path=str(base),
               artifact_path=str(tmp_path / "art.json"), write=False,
               harvester_path=_stub_harvester(tmp_path))
    assert v["pass_all"] is True
