"""Candidate-change pipeline — Phase 0.1 capstone.

Replaces direct LLM code mutation with a deterministic, fail-closed state machine
that turns autonomous self-improvement into a SAFE continuous-delivery loop:

    PROPOSED -> SANDBOXED -> TESTED -> REVIEWED -> SHADOW -> CANARY -> APPROVED
                                                                    -> DEPLOYED
    (any non-terminal) -> REJECTED

Guarantees
----------
* **Provenance required**: a candidate must carry problem/evidence/hypothesis/
  patch/files/risk/test-plan/expected-improvement/regressions/rollback/author or
  it cannot enter (``IncompleteCandidate``).
* **Fail closed**: a runner that raises, a failed test result, a non-``False``
  regression flag, or a malformed result rejects the candidate.
* **Sandboxed**: generated code is only ever written to a path OUTSIDE the live
  tree (``code_mutation_guard.isolated_sandbox_path``).
* **Separation of duties** (rule 8): the author may not review, approve, or
  deploy its own change.
* **Deploy still gated**: even an APPROVED candidate cannot deploy unless code
  mutation is explicitly human-enabled (the Phase 0.1 guard) — otherwise it stays
  APPROVED, surfaced for a human, never silently applied.
* **Atomic + auditable**: persisted via ``CASStore``; every transition recorded,
  optionally mirrored to a tamper-evident ``DecisionLog``.

Stage runners (sandbox/test/shadow/canary/deploy) are injected, so the pipeline
is testable without touching real worktrees, the real test suite, or live code.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional

from qf_safety.atomic_json import CASStore
from qf_safety import code_mutation_guard

REQUIRED_CANDIDATE_FIELDS = {
    "problem_statement",
    "evidence",
    "hypothesis",
    "patch",
    "files_changed",
    "risk_classification",
    "test_plan",
    "expected_improvement",
    "possible_regressions",
    "rollback_procedure",
    "author",
}


class Stage(str, Enum):
    PROPOSED = "proposed"
    SANDBOXED = "sandboxed"
    TESTED = "tested"
    REVIEWED = "reviewed"
    SHADOW = "shadow"
    CANARY = "canary"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"


LEGAL_TRANSITIONS: Dict[Stage, set] = {
    Stage.PROPOSED: {Stage.SANDBOXED, Stage.REJECTED},
    Stage.SANDBOXED: {Stage.TESTED, Stage.REJECTED},
    Stage.TESTED: {Stage.REVIEWED, Stage.REJECTED},
    Stage.REVIEWED: {Stage.SHADOW, Stage.REJECTED},
    Stage.SHADOW: {Stage.CANARY, Stage.REJECTED},
    Stage.CANARY: {Stage.APPROVED, Stage.REJECTED},
    Stage.APPROVED: {Stage.DEPLOYED, Stage.REJECTED},
    Stage.DEPLOYED: set(),
    Stage.REJECTED: set(),
}


class IncompleteCandidate(ValueError):
    pass


class IllegalStageTransition(Exception):
    pass


class SeparationOfDutiesError(Exception):
    pass


class CandidateNotFound(KeyError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CandidatePipeline:
    def __init__(self, path: str, *, decision_log=None):
        self._store = CASStore(path)
        self._log = decision_log  # optional DecisionLog

    # ── helpers ────────────────────────────────────────────────────────────
    def _all(self) -> Dict[str, Any]:
        return self._store.load()["data"]

    def _record(self, cid: str) -> Dict[str, Any]:
        data = self._all()
        if cid not in data:
            raise CandidateNotFound(cid)
        return data[cid]

    def _candidate(self, cid: str) -> Dict[str, Any]:
        return self._record(cid)["candidate"]

    def _require_stage(self, cid: str, expected: Stage) -> None:
        cur = Stage(self._record(cid)["stage"])
        if cur != expected:
            raise IllegalStageTransition(f"candidate {cid} is {cur.value}, expected {expected.value}")

    def _assert_not_author(self, cid: str, actor: str, role: str) -> None:
        author = self._candidate(cid)["author"]
        if actor == author:
            raise SeparationOfDutiesError(
                f"author '{author}' may not act as {role} for its own candidate"
            )

    def _transition(self, cid: str, expected_prior: Stage, new_stage: Stage,
                    *, actor: str, extra: Optional[dict] = None) -> None:
        def mutator(data):
            if cid not in data:
                raise CandidateNotFound(cid)
            cur = Stage(data[cid]["stage"])
            if cur != expected_prior:
                raise IllegalStageTransition(f"{cur.value} != expected {expected_prior.value}")
            if new_stage not in LEGAL_TRANSITIONS[cur]:
                raise IllegalStageTransition(f"{cur.value} -> {new_stage.value} illegal")
            data[cid]["stage"] = new_stage.value
            entry = {"from": cur.value, "to": new_stage.value, "actor": actor,
                     "ts": _now_iso(), "extra": extra or {}}
            data[cid]["history"].append(entry)
            if extra and "sandbox_path" in extra:
                data[cid]["sandbox_path"] = extra["sandbox_path"]
            return data

        self._store.update(mutator, actor=actor)
        if self._log is not None:
            self._log.append({"event": "candidate_transition", "candidate": cid,
                              "to": new_stage.value, "actor": actor, "extra": extra or {}})

    def _reject(self, cid: str, *, actor: str, reason: str) -> None:
        # Reject is legal from any non-terminal stage.
        def mutator(data):
            if cid not in data:
                raise CandidateNotFound(cid)
            cur = Stage(data[cid]["stage"])
            if cur in (Stage.DEPLOYED, Stage.REJECTED):
                raise IllegalStageTransition(f"cannot reject from terminal {cur.value}")
            data[cid]["stage"] = Stage.REJECTED.value
            data[cid]["history"].append({"from": cur.value, "to": Stage.REJECTED.value,
                                         "actor": actor, "ts": _now_iso(),
                                         "extra": {"reason": reason}})
            return data

        self._store.update(mutator, actor=actor)
        if self._log is not None:
            self._log.append({"event": "candidate_rejected", "candidate": cid,
                              "actor": actor, "reason": reason})

    # ── pipeline API ───────────────────────────────────────────────────────
    def submit(self, candidate: Dict[str, Any]) -> str:
        missing = [f for f in REQUIRED_CANDIDATE_FIELDS
                   if f not in candidate or candidate[f] in (None, "", [], {})]
        if missing:
            raise IncompleteCandidate(f"candidate missing required fields: {sorted(missing)}")

        cid = hashlib.sha256(
            (candidate["author"] + candidate["problem_statement"] + _now_iso()).encode()
        ).hexdigest()[:16]

        def mutator(data):
            data[cid] = {
                "candidate": dict(candidate),
                "stage": Stage.PROPOSED.value,
                "sandbox_path": None,
                "history": [{"from": None, "to": Stage.PROPOSED.value,
                             "actor": candidate["author"], "ts": _now_iso(), "extra": {}}],
            }
            return data

        self._store.update(mutator, actor=candidate["author"])
        return cid

    def sandbox(self, cid: str, *, actor: str,
                sandbox_runner: Callable[[dict, str], dict]) -> None:
        self._require_stage(cid, Stage.PROPOSED)
        path = code_mutation_guard.isolated_sandbox_path(cid)
        try:
            sandbox_runner(self._candidate(cid), path)
        except Exception as e:  # noqa: BLE001 — fail closed
            self._reject(cid, actor=actor, reason=f"sandbox error: {str(e)[:120]}")
            return
        self._transition(cid, Stage.PROPOSED, Stage.SANDBOXED, actor=actor,
                         extra={"sandbox_path": path})

    def test(self, cid: str, *, actor: str,
             test_runner: Callable[[dict, str], dict]) -> None:
        self._require_stage(cid, Stage.SANDBOXED)
        sandbox_path = self._record(cid).get("sandbox_path")
        try:
            result = test_runner(self._candidate(cid), sandbox_path)
        except Exception as e:  # noqa: BLE001 — fail closed
            self._reject(cid, actor=actor, reason=f"test error: {str(e)[:120]}")
            return
        if not isinstance(result, dict) or result.get("passed") is not True:
            self._reject(cid, actor=actor, reason=f"tests not passed: {str(result)[:120]}")
            return
        self._transition(cid, Stage.SANDBOXED, Stage.TESTED, actor=actor,
                         extra={"test_report": result.get("report", "")})

    def review(self, cid: str, *, reviewer: str, approve: bool) -> None:
        self._require_stage(cid, Stage.TESTED)
        self._assert_not_author(cid, reviewer, "reviewer")
        if approve is not True:
            self._reject(cid, actor=reviewer, reason="review rejected the candidate")
            return
        self._transition(cid, Stage.TESTED, Stage.REVIEWED, actor=reviewer)

    def shadow(self, cid: str, *, actor: str, result: dict) -> None:
        self._require_stage(cid, Stage.REVIEWED)
        if not isinstance(result, dict) or result.get("regression") is not False:
            self._reject(cid, actor=actor, reason=f"shadow regression/unknown: {str(result)[:120]}")
            return
        self._transition(cid, Stage.REVIEWED, Stage.SHADOW, actor=actor, extra=result)

    def canary(self, cid: str, *, actor: str, result: dict) -> None:
        self._require_stage(cid, Stage.SHADOW)
        if not isinstance(result, dict) or result.get("regression") is not False:
            self._reject(cid, actor=actor, reason=f"canary regression/unknown: {str(result)[:120]}")
            return
        self._transition(cid, Stage.SHADOW, Stage.CANARY, actor=actor, extra=result)

    def approve(self, cid: str, *, approver: str) -> None:
        self._require_stage(cid, Stage.CANARY)
        self._assert_not_author(cid, approver, "approver")
        self._transition(cid, Stage.CANARY, Stage.APPROVED, actor=approver)

    def deploy(self, cid: str, *, deployer: str,
               deploy_runner: Callable[[dict, str], dict]) -> None:
        self._require_stage(cid, Stage.APPROVED)
        self._assert_not_author(cid, deployer, "deployer")
        # Gated: raises CodeMutationBlocked unless a human has enabled mutation.
        # If blocked, the candidate STAYS approved (not lost) and the block surfaces.
        code_mutation_guard.assert_mutation_allowed(f"deploy_candidate:{cid}")
        try:
            result = deploy_runner(self._candidate(cid), self._record(cid).get("sandbox_path"))
        except Exception as e:  # noqa: BLE001 — fail closed
            self._reject(cid, actor=deployer, reason=f"deploy error: {str(e)[:120]}")
            return
        self._transition(cid, Stage.APPROVED, Stage.DEPLOYED, actor=deployer,
                         extra={"deploy_result": result})

    def reject(self, cid: str, *, actor: str, reason: str) -> None:
        self._reject(cid, actor=actor, reason=reason)

    def get(self, cid: str) -> Dict[str, Any]:
        rec = self._record(cid)
        return {
            "candidate": rec["candidate"],
            "stage": Stage(rec["stage"]),
            "sandbox_path": rec.get("sandbox_path"),
            "history": rec["history"],
        }
