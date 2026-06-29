"""Trustworthy self-improvement loop — Phase 8.

Ties the existing autonomous LLM loops (evolver / self-heal / reflect) to the
safe machinery built in Phase 0/1:

    LLM proposes  ->  candidate pipeline  ->  [sandbox] -> [safety tests] ->
    [independent review] -> [honest cost-adjusted evidence] -> APPROVED -> (STOP)

The loop runs end-to-end **autonomously up to APPROVED**, but it deliberately
NEVER deploys. Deploy stays human-gated (and is blocked by the code-mutation
guard regardless). This is graduated autonomy: the loop earns trust by repeatedly
producing changes that pass honest gates and by correctly rejecting bad ones;
only then does a human widen its authority.

Every gate fails closed: a failed test, an evidence regression, or a reviewer
rejection stops the change at REJECTED. The proposing author can never review or
approve its own change (enforced by the pipeline's separation-of-duties guard).

Stage runners are injected so the loop is testable; production runners
(``pytest_test_runner``, an evidence runner over the qf_mlops harness) are
supplied here for real use.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Callable, Dict, Optional

from qf_safety.candidate_pipeline import CandidatePipeline, Stage


def evidence_gate(baseline: float, candidate: float, *, tolerance: float = 0.0) -> Dict[str, Any]:
    """Honest no-regression check: the candidate's cost-adjusted metric must not
    fall more than ``tolerance`` below the incumbent baseline."""
    regression = candidate < (baseline - tolerance)
    return {"regression": regression, "baseline": baseline, "candidate": candidate}


def pytest_test_runner(repo_dir: str, *, test_path: str = "tests/", ignore=()):
    """Production test runner: runs the safety + evidence suite in ``repo_dir``.

    Returns a runner ``(candidate, sandbox_path) -> {passed, report}`` that the
    pipeline's TEST stage calls. A non-zero pytest exit (any failing test) ->
    ``passed: False`` -> the candidate is rejected.
    """
    def _run(candidate: Dict[str, Any], sandbox_path: Optional[str]) -> Dict[str, Any]:
        cmd = ["python3", "-m", "pytest", test_path, "-q"]
        for ig in ignore:
            cmd += ["--ignore", ig]
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=600)
        tail = (proc.stdout or "")[-400:]
        return {"passed": proc.returncode == 0, "report": tail}

    return _run


class SelfImprovementLoop:
    def __init__(
        self,
        pipeline: CandidatePipeline,
        *,
        sandbox_runner: Callable,
        test_runner: Callable,
        evidence_runner: Callable[[Dict[str, Any]], Dict[str, Any]],
        reviewer: Callable[[Dict[str, Any]], bool],
        ci_actor: str = "ci_runner",
        reviewer_actor: str = "independent_reviewer",
        approver_actor: str = "human_approver",
    ):
        self.pipeline = pipeline
        self.sandbox_runner = sandbox_runner
        self.test_runner = test_runner
        self.evidence_runner = evidence_runner
        self.reviewer = reviewer
        self.ci_actor = ci_actor
        self.reviewer_actor = reviewer_actor
        self.approver_actor = approver_actor

    def _stage(self, cid: str) -> Stage:
        return self.pipeline.get(cid)["stage"]

    def _result(self, cid: str) -> Dict[str, Any]:
        rec = self.pipeline.get(cid)
        return {
            "candidate_id": cid,
            "final_stage": rec["stage"],
            "approved_pending_deploy": rec["stage"] == Stage.APPROVED,
            "rejected": rec["stage"] == Stage.REJECTED,
            "history": rec["history"],
        }

    def run(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Drive a proposal to a verdict. Returns the disposition; never deploys."""
        # submit validates provenance (raises IncompleteCandidate if missing).
        cid = self.pipeline.submit(candidate)

        self.pipeline.sandbox(cid, actor=self.ci_actor, sandbox_runner=self.sandbox_runner)
        if self._stage(cid) == Stage.REJECTED:
            return self._result(cid)

        self.pipeline.test(cid, actor=self.ci_actor, test_runner=self.test_runner)
        if self._stage(cid) == Stage.REJECTED:
            return self._result(cid)

        self.pipeline.review(
            cid, reviewer=self.reviewer_actor, approve=bool(self.reviewer(candidate))
        )
        if self._stage(cid) == Stage.REJECTED:
            return self._result(cid)

        self.pipeline.shadow(cid, actor=self.ci_actor, result=self.evidence_runner(candidate))
        if self._stage(cid) == Stage.REJECTED:
            return self._result(cid)

        self.pipeline.canary(cid, actor=self.ci_actor, result=self.evidence_runner(candidate))
        if self._stage(cid) == Stage.REJECTED:
            return self._result(cid)

        # CANARY passed -> approve. The loop stops here on purpose: DEPLOY is a
        # separate, human-gated step (and the mutation guard blocks it anyway).
        self.pipeline.approve(cid, approver=self.approver_actor)
        return self._result(cid)
