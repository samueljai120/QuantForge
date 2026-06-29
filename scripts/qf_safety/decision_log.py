"""Tamper-evident decision log — Phase 0 observability (#8).

Append-only JSONL where each entry carries a SHA-256 hash over
``(seq, ts, prev_hash, payload)`` and links to the previous entry's hash. Any
later edit to a past entry breaks the chain and is detected by ``verify()``.

Used to record LLM prompts/responses, tool calls, proposed actions, approval
decisions, and outcomes so a decision can be reconstructed after the fact.

Secrets are redacted before write (mandate rule 12): by key name (anything that
looks like a credential) and by value pattern (provider key prefixes). The log
is never the right place for raw secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from qf_safety.atomic_json import file_lock

GENESIS_HASH = "0" * 64

# Key names whose values are redacted (case-insensitive substring match).
_SECRET_KEY_MARKERS = (
    "api_key", "apikey", "secret", "password", "passwd", "token",
    "credential", "bearer", "private_key", "access_key", "authorization",
)

# Value patterns for common provider key formats.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),         # OpenAI / OpenRouter style
    re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{16,}"),    # OpenRouter v1
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),            # GitHub PAT
    re.compile(r"AKIA[0-9A-Z]{16}"),                # AWS access key id
)

_REDACTED = "***REDACTED***"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pat in _SECRET_VALUE_PATTERNS:
            out = pat.sub(_REDACTED, out)
        return out
    return value


def redact(obj: Any) -> Any:
    """Recursively redact secret-looking keys and values."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if any(m in str(k).lower() for m in _SECRET_KEY_MARKERS):
                result[k] = _REDACTED
            else:
                result[k] = redact(v)
        return result
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return _redact_value(obj)


def _entry_hash(seq: int, ts: str, prev_hash: str, payload: Any) -> str:
    canonical = json.dumps(
        {"seq": seq, "ts": ts, "prev_hash": prev_hash, "payload": payload},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DecisionLog:
    def __init__(self, path: str, *, lock_path: Optional[str] = None):
        self.path = path
        self.lock_path = lock_path or (path + ".lock")

    def _last(self) -> Tuple[int, str]:
        """Return (last_seq, last_hash) without locking (caller holds the lock)."""
        if not os.path.exists(self.path):
            return 0, GENESIS_HASH
        last_seq, last_hash = 0, GENESIS_HASH
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                last_seq = obj["seq"]
                last_hash = obj["hash"]
        return last_seq, last_hash

    def append(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        safe_payload = redact(payload)
        with file_lock(self.lock_path):
            last_seq, last_hash = self._last()
            seq = last_seq + 1
            ts = _now_iso()
            h = _entry_hash(seq, ts, last_hash, safe_payload)
            entry = {
                "seq": seq,
                "ts": ts,
                "prev_hash": last_hash,
                "hash": h,
                "payload": safe_payload,
            }
            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return entry

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def verify(self) -> Tuple[bool, Optional[int]]:
        """Return (ok, broken_index). ``broken_index`` is zero-based."""
        prev = GENESIS_HASH
        for idx, entry in enumerate(self.read_all()):
            if entry.get("prev_hash") != prev:
                return False, idx
            recomputed = _entry_hash(
                entry["seq"], entry["ts"], entry["prev_hash"], entry["payload"]
            )
            if recomputed != entry.get("hash"):
                return False, idx
            prev = entry["hash"]
        return True, None
