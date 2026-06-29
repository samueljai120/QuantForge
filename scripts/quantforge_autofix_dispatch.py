#!/usr/bin/env python3
"""Autofix dispatcher — the last link: an open fix ticket -> an LLM patch -> the gate.

For the oldest un-attempted fix candidate (opened by quantforge_self_heal_invariants on an
invariant breach), this asks the LLM (OpenRouter via llm_router) for a patch — showing it
only the RELEVANT function named in the breach hint, not the whole file — and drives that
patch through the sandbox + full-suite gate (qf_safety.autofix). A patch that keeps every
test green lands as APPROVED-pending-human (one-click deploy); a failing one is REJECTED.

Calls the LLM at most ONCE per run (budget-conscious) and attempts each ticket once.
Never deploys. Intended to piggyback the 4h self-heal cron line. The verdict is recorded in
the fix-candidate index so the daily report can surface "fix ready to approve" / "no fix yet".
"""
import ast
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qf_safety.candidate_pipeline import CandidatePipeline, Stage
from qf_safety.autofix import attempt_autofix

REPO = os.path.expanduser("~/quantforge")
DATA = os.path.join(REPO, "data/quantforge")
STORE = os.path.join(DATA, "qf_fix_candidates.json")
INDEX = os.path.join(DATA, "qf_fix_candidate_index.json")
ATTEMPTS = os.path.join(DATA, "qf_autofix_attempts.jsonl")


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _hint_function(hint: str):
    head = str(hint or "").split(":", 1)[0].strip()
    return head.split(".")[-1].strip() if "." in head else None


def _extract_function(src: str, funcname: str):
    """Return the source of `funcname` (bounded prompt context) or None."""
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == funcname:
            lines = src.splitlines()
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None) or (start + 1)
            return "\n".join(lines[start:end])
    return None


def _smart_reader(candidate):
    """Reader for the PROMPT only: show the relevant function (from the hint) or a bounded
    slice — never the whole 190KB file. The sandbox still patches the full file."""
    fn = _hint_function(candidate.get("hypothesis", ""))

    def read(f):
        full = open(os.path.join(REPO, f)).read()
        if fn:
            exc = _extract_function(full, fn)
            if exc:
                return f"# (excerpt: function {fn} from {f})\n{exc}"
        return full[:8000]
    return read


def _llm(prompt):
    # Patch generation is a high-stakes code decision -> route to the strong model
    # (DeepSeek V4 via call_brain). call_brain returns a dict {'text', 'usage', ...} —
    # unwrap the 'text' field (the generated content).
    from llm_router import call_brain
    r = call_brain(prompt, high_stakes=True, max_tokens=4096)
    return r["text"] if isinstance(r, dict) else str(r)


def main():
    index = _read_json(INDEX, {})
    if not os.path.exists(STORE):
        print("autofix-dispatch: no fix candidates store yet — nothing to do.")
        return 0
    pipeline = CandidatePipeline(STORE)
    allrecs = pipeline._all()

    # oldest PROPOSED ticket from the detector that we have NOT attempted yet
    tickets = []
    for sig, info in index.items():
        cid = info.get("cid")
        rec = allrecs.get(cid)
        if rec and rec.get("stage") == Stage.PROPOSED.value and not info.get("attempted"):
            tickets.append((sig, info, rec["candidate"]))
    if not tickets:
        print("autofix-dispatch: no un-attempted open fix candidates.")
        return 0
    tickets.sort(key=lambda t: t[1].get("opened_at", ""))
    sig, info, cand = tickets[0]

    print(f"autofix-dispatch: attempting LLM patch for '{cand.get('files_changed')}' — "
          f"{cand.get('problem_statement', '')[:80]}")
    now = datetime.now(timezone.utc).isoformat()
    try:
        res = attempt_autofix(cand, llm_call=_llm, repo_dir=REPO,
                              pipeline=pipeline, read_file=_smart_reader(cand))
        disp = res["disposition"]
        verdict = "APPROVED_PENDING_HUMAN" if disp.get("approved_pending_deploy") else (
            "REJECTED" if disp.get("rejected") else disp.get("final_stage"))
        print(f"  -> {verdict} (attempt candidate {disp.get('candidate_id')}); "
              f"explanation: {res.get('explanation', '')[:120]}")
    except Exception as e:  # noqa: BLE001 — fail closed; the ticket stays open for a human
        verdict = f"ERROR:{type(e).__name__}"
        disp = {"candidate_id": None}
        print(f"  -> patch attempt errored: {str(e)[:150]}")

    # mark the ticket attempted (one shot; a human/next breach drives further work) + log
    info["attempted"] = True
    info["attempt_at"] = now
    info["attempt_verdict"] = verdict
    info["attempt_cid"] = disp.get("candidate_id")
    index[sig] = info
    try:
        with open(INDEX, "w") as f:
            json.dump(index, f, indent=2)
        with open(ATTEMPTS, "a") as f:
            f.write(json.dumps({"ts": now, "ticket": info.get("cid"), "name": info.get("name"),
                                "verdict": verdict, "attempt_cid": disp.get("candidate_id")}) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
