"""Autofix orchestrator — drives a fix candidate from bug -> LLM patch -> GATES -> verdict.

The trustworthy property lives here: an LLM-proposed patch is applied in an ISOLATED
sandbox (a lightweight copy of the code dirs, never the live tree), then the FULL test
suite runs against the patched sandbox. Only a patch that keeps every test green and
passes independent review reaches APPROVED-pending-human; anything else is REJECTED.
The patch never touches the live code — deploy stays a separate, human-gated step.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Callable, Optional

CODE_DIRS = ("scripts", "tests", "config")
_ROOT_CONFIG = ("conftest.py", "pytest.ini", "setup.cfg", "pyproject.toml", "tox.ini")


def _coerce_patch(candidate: dict) -> dict:
    p = candidate.get("patch")
    return json.loads(p) if isinstance(p, str) else (p or {})


def apply_edits(edits: list, target_dir: str) -> None:
    """Apply search/replace edits to files under target_dir. Each `find` must match EXACTLY
    ONCE (fail-closed: missing or ambiguous -> raise, which rejects the candidate)."""
    for e in edits:
        path = os.path.join(target_dir, e["file"])
        with open(path) as f:
            src = f.read()
        find = e["find"]
        n = src.count(find)
        if n == 0:
            raise ValueError(f"patch 'find' not present in {e['file']}")
        if n > 1:
            raise ValueError(f"patch 'find' not unique in {e['file']} ({n} matches)")
        with open(path, "w") as f:
            f.write(src.replace(find, e["replace"], 1))


def make_sandbox_runner(repo_dir: str) -> Callable[[dict, str], None]:
    """Returns a sandbox_runner(candidate, sandbox_path): populate the sandbox with a
    lightweight copy of the code dirs (NOT the 20GB repo) and apply the candidate's patch."""
    def _run(candidate: dict, sandbox_path: str) -> None:
        if os.path.exists(sandbox_path):
            shutil.rmtree(sandbox_path)
        os.makedirs(sandbox_path)
        for d in CODE_DIRS:
            src = os.path.join(repo_dir, d)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(sandbox_path, d))
        for f in _ROOT_CONFIG:
            p = os.path.join(repo_dir, f)
            if os.path.exists(p):
                shutil.copy(p, sandbox_path)
        apply_edits(_coerce_patch(candidate).get("edits", []), sandbox_path)
    return _run


def sandbox_pytest_runner(candidate: dict, sandbox_path: str, *, timeout: int = 600) -> dict:
    """The TEST GATE: run the full suite
    against the PATCHED sandbox. passed iff exit 0."""
    # Validate under the SAME interpreter running the gate (sys.executable). On a production host's
    # cron this resolves to /usr/bin/python3 (the live agent's interpreter) — unchanged;
    # locally it is whatever python runs pytest, so the sandbox gate is portable across hosts.
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    proc = subprocess.run(cmd, cwd=sandbox_path, capture_output=True, text=True, timeout=timeout)
    passed = proc.returncode == 0
    tail = (proc.stdout if passed else proc.stdout + proc.stderr)[-1200:]
    return {"passed": passed, "report": tail}


def basic_reviewer(candidate: dict) -> bool:
    """Independent (non-author) review: the patch must only touch production code under
    scripts/ (never tests/ — a fix may not weaken the gate) and have non-empty replacements."""
    patch = _coerce_patch(candidate)
    edits = patch.get("edits") or []
    if not edits:
        return False
    for e in edits:
        f = str(e.get("file", ""))
        if not f.startswith("scripts/") or f.startswith("scripts/../"):
            return False
        if not str(e.get("replace", "")).strip():
            return False
    return True


def attempt_autofix(candidate: dict, *, llm_call: Callable[[str], str], repo_dir: str, pipeline,
                    reviewer: Optional[Callable[[dict], bool]] = None,
                    read_file: Optional[Callable[[str], str]] = None) -> dict:
    """Full wire: generate an LLM patch for `candidate`, then drive it through the candidate
    pipeline (sandbox -> full test gate -> independent review -> ... -> APPROVED-pending-human
    or REJECTED). Returns {disposition, patch, explanation}. Never deploys.

    `read_file` (default: full file from repo_dir) supplies the code shown to the LLM in the
    PROMPT only — pass a bounded/relevant-excerpt reader to keep the prompt small. The sandbox
    always copies + patches the FULL files, so a `find` snippet taken from an excerpt still
    matches the real file."""
    from qf_safety.patch_generator import generate_patch
    from qf_safety.self_improvement import SelfImprovementLoop

    def _full_read(f):
        with open(os.path.join(repo_dir, f)) as fh:
            return fh.read()

    patch = generate_patch(candidate, read_file=read_file or _full_read, llm_call=llm_call)
    cand = dict(candidate)
    cand["patch"] = json.dumps(patch)

    loop = SelfImprovementLoop(
        pipeline,
        sandbox_runner=make_sandbox_runner(repo_dir),
        test_runner=sandbox_pytest_runner,
        evidence_runner=lambda c: {"regression": False},
        reviewer=reviewer or basic_reviewer,
    )
    disposition = loop.run(cand)
    return {"disposition": disposition, "patch": patch, "explanation": patch.get("explanation", "")}
