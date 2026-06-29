"""Code-mutation guard — Phase 0.1.

Closes defects #1 (evolver applies LLM patches to live source, no sandbox) and
#4 (self-heal runs arbitrary LLM-supplied shell). A single deterministic gate
decides whether autonomous code mutation may proceed. It is OFF by default and
fails closed: an LLM cannot turn it on, and a missing/garbage flag means blocked.

When a human explicitly enables mutation (an L3 action with independent approval,
per ``qf_safety.permissions``), candidate code must still be written to an
isolated sandbox path *outside* the live tree — never directly onto the files
that cron executes.

Intended call sites:
* ``quantforge_code_evolver.py`` — before applying any patch.
* ``quantforge_self_heal_actions.py`` — before any ``subprocess.run(..., shell=True)``
  on an LLM-supplied command, and before writing scripts / installing packages.
"""

from __future__ import annotations

import os

CODE_MUTATION_ENV = "QF_ALLOW_CODE_MUTATION"
_TRUTHY = {"1", "true", "yes", "on"}

# Sandbox lives next to (not inside) the live repo so a stray relative write
# cannot land on an executing file.
SANDBOX_ROOT = os.path.expanduser("~/quantforge-sandbox")


class CodeMutationBlocked(PermissionError):
    """Raised when autonomous code mutation is attempted while disabled."""


def mutation_enabled() -> bool:
    """True only if the env flag is set to an explicit truthy value."""
    return os.environ.get(CODE_MUTATION_ENV, "").strip().lower() in _TRUTHY


def assert_mutation_allowed(action: str = "code_mutation") -> None:
    """Raise ``CodeMutationBlocked`` unless mutation is explicitly enabled."""
    if not mutation_enabled():
        raise CodeMutationBlocked(
            f"code mutation '{action}' is blocked: set {CODE_MUTATION_ENV}=1 "
            f"only under human-approved, independently-reviewed conditions "
            f"(Level 3). Autonomous mutation is disabled by default."
        )


def isolated_sandbox_path(name: str) -> str:
    """Return a path under the sandbox root (outside the live tree) for a
    candidate change. Does not create anything."""
    safe = name.replace(os.sep, "_").lstrip(".") or "candidate"
    return os.path.join(SANDBOX_ROOT, safe)
