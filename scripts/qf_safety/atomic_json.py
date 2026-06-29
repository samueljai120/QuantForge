"""Atomic, locked, versioned JSON state — Phase 0.6 concurrent-write protection.

Problem this solves
--------------------
Multiple cron actors (``quantforge_agent``, ``quantforge_self_tune``,
``quantforge_reflect``, ``quantforge_allocator``, ``quantforge_self_heal_actions``)
write the same shared JSON files (e.g. ``qf_strategy_params.json``) with plain
``json.dump`` and no coordination. A write that interleaves with another can
either be lost (last-writer-wins) or leave a half-written, corrupt file.

What this provides
------------------
* ``atomic_write_json`` — crash-safe write via temp-file + ``os.replace`` (atomic
  on POSIX). Never leaves a partially written file at the target path.
* ``read_json`` — tolerant read with a default for missing files.
* ``file_lock`` — advisory exclusive ``flock`` with a bounded, non-blocking
  acquire (fails *closed* with ``TimeoutError`` rather than blocking forever).
* ``CASStore`` — a versioned store with compare-and-swap. Every update runs
  inside the lock as a read-modify-write, bumps a monotonic version, and records
  provenance (who/when/checksum). A stale ``expected_version`` is rejected with
  ``ConcurrentModificationError`` and leaves state untouched.

Design notes
------------
* ``atomic_write_json`` writes *bare* JSON (no envelope) so it is a drop-in
  replacement for existing ``json.dump`` call sites on flat files.
* ``CASStore`` wraps payloads in a ``{"_qf_meta": {...}, "data": {...}}`` envelope
  so version/provenance travels with the file. Use it for *new* safety-managed
  state, not for files other code reads as flat JSON.
* Pure standard library; Python 3.8+ compatible.
"""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

META_KEY = "_qf_meta"


class ConcurrentModificationError(Exception):
    """Raised when a compare-and-swap update sees an unexpected version."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checksum(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: str, obj: Any, *, indent: int = 2) -> None:
    """Atomically write ``obj`` as JSON to ``path``.

    Writes to a temp file in the same directory, fsyncs, then ``os.replace``s it
    onto the target. On any error the temp file is removed and the original
    target is left untouched.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON from ``path``; return ``default`` if the file does not exist."""
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


@contextmanager
def file_lock(lock_path: str, *, timeout: Optional[float] = None, poll: float = 0.02):
    """Exclusive advisory lock via ``flock``.

    Two modes:

    * ``timeout is None`` (default): a **blocking** ``LOCK_EX``. The kernel queues
      waiters fairly, so there is no starvation under contention. Safe against a
      crashed holder — advisory locks release automatically when the holding
      process exits or closes the fd — and our critical sections are short
      read-modify-write blocks that always release.
    * ``timeout`` is a number: a **non-blocking** poll with a deadline, raising
      ``TimeoutError`` if not acquired. Failing closed (raising) is intentional —
      a writer that cannot serialize must not proceed and risk a torn write. Use
      this when a bounded wait is required.
    """
    directory = os.path.dirname(os.path.abspath(lock_path)) or "."
    os.makedirs(directory, exist_ok=True)
    f = open(lock_path, "w")
    try:
        if timeout is None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # blocking, fair
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as e:
                    if e.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"could not acquire lock {lock_path} within {timeout}s"
                        )
                    time.sleep(poll)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


class CASStore:
    """Versioned JSON store with compare-and-swap and provenance.

    File layout::

        {"_qf_meta": {"version": N, "updated_by": ..., "updated_at": ...,
                      "checksum": ..., "prev_checksum": ...},
         "data": { ... payload ... }}
    """

    def __init__(self, path: str, *, lock_path: Optional[str] = None):
        self.path = path
        self.lock_path = lock_path or (path + ".lock")

    def load(self) -> Dict[str, Any]:
        env = read_json(self.path, default=None)
        if env is None or META_KEY not in env:
            return {"version": 0, "data": (env or {}) if env else {}, "meta": None}
        meta = env.get(META_KEY) or {}
        return {
            "version": int(meta.get("version", 0)),
            "data": env.get("data", {}),
            "meta": meta,
        }

    def update(
        self,
        mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        actor: str,
        expected_version: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Atomically read-modify-write under lock.

        ``mutator`` receives a deep copy of the current data and returns the new
        data. If ``expected_version`` is given and does not match the current
        version, raises ``ConcurrentModificationError`` *without writing*.
        """
        with file_lock(self.lock_path, timeout=timeout):
            current = self.load()
            if expected_version is not None and current["version"] != expected_version:
                raise ConcurrentModificationError(
                    f"expected version {expected_version}, found {current['version']}"
                )
            new_data = mutator(copy.deepcopy(current["data"]))
            if not isinstance(new_data, dict):
                raise TypeError("mutator must return a dict")
            new_version = current["version"] + 1
            prev_checksum = (current.get("meta") or {}).get("checksum")
            envelope = {
                META_KEY: {
                    "version": new_version,
                    "updated_by": actor,
                    "updated_at": _now_iso(),
                    "checksum": _checksum(new_data),
                    "prev_checksum": prev_checksum,
                },
                "data": new_data,
            }
            atomic_write_json(self.path, envelope)
            return {"version": new_version, "data": new_data}
