"""LLM patch generator — the one generative step in the self-heal loop.

Given a fix candidate (problem + evidence + the target file), it asks an LLM for a MINIMAL
search/replace patch. The LLM is injected (``llm_call``) so this is pure + testable; in
production it is wired to llm_router.call_llm (OpenRouter). The patch it returns is NOT
trusted — it must still pass the sandbox + full-suite gate in qf_safety.autofix. This module
only turns a bug description into a candidate patch; correctness is the gate's job.

Patch format (robust for LLM output — search/replace, not line-numbered diffs):
    {"edits": [{"file": "scripts/x.py", "find": "<exact unique snippet>", "replace": "<fix>"}],
     "explanation": "<one line>"}
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict


def build_prompt(candidate: dict, file_contents: Dict[str, str]) -> str:
    files_block = "\n\n".join(
        f"### FILE: {f}\n```python\n{c}\n```" for f, c in file_contents.items()
    )
    return (
        "You are fixing a money-conservation bug in a Python paper-trading system. "
        "Make the MINIMAL change that fixes the bug without altering tests.\n\n"
        f"PROBLEM: {candidate.get('problem_statement', '')}\n"
        f"EVIDENCE: {candidate.get('evidence', '')}\n"
        f"HYPOTHESIS / HINT: {candidate.get('hypothesis', '')}\n\n"
        f"{files_block}\n\n"
        "Respond with ONLY a JSON object, no prose:\n"
        '{"edits": [{"file": "<path>", "find": "<exact unique snippet from the file>", '
        '"replace": "<fixed snippet>"}], "explanation": "<one line>"}\n'
        "Rules: `find` must appear EXACTLY ONCE in the named file (include enough surrounding "
        "context to be unique). Keep the edit minimal. Never edit files under tests/. Output JSON only."
    )


def parse_patch(text: str) -> dict:
    """Extract + validate the JSON patch from an LLM response (tolerates ```json fences/prose)."""
    if not text or not str(text).strip():
        raise ValueError("empty LLM response")
    m = re.search(r"\{.*\}", str(text), re.DOTALL)  # first { .. last }
    if not m:
        raise ValueError("no JSON object in LLM response")
    obj = json.loads(m.group(0))
    edits = obj.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("patch has no edits")
    for e in edits:
        if not isinstance(e, dict) or not all(k in e for k in ("file", "find", "replace")):
            raise ValueError("each edit needs file/find/replace")
        if not str(e["find"]).strip():
            raise ValueError("edit has an empty 'find'")
    return {"edits": edits, "explanation": str(obj.get("explanation", ""))[:300]}


def generate_patch(candidate: dict, *, read_file: Callable[[str], str], llm_call: Callable[[str], str]) -> dict:
    """Read the candidate's target files, prompt the LLM, return the parsed patch dict.
    Raises on an unparseable response (fail-closed: the candidate then has no patch to gate)."""
    contents: Dict[str, str] = {}
    for f in candidate.get("files_changed", []):
        try:
            contents[f] = read_file(f)
        except Exception:
            contents[f] = "(file unreadable)"
    resp = llm_call(build_prompt(candidate, contents))
    return parse_patch(resp)
