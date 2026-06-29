"""Phase 0 #8 — Tamper-evident LLM/decision logging.

The audit found LLM prompts, responses, tool calls, and decisions are not fully
recorded or correlated. This is an append-only, hash-chained JSONL log so any
later edit to a past entry is detectable, and a redaction pass so secrets never
land in the log (mandate rule 12).
"""

import json
import threading

from qf_safety.decision_log import DecisionLog, GENESIS_HASH


def test_append_returns_chained_entry(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    e1 = log.append({"event": "llm_call", "model": "claude"})
    e2 = log.append({"event": "decision", "verdict": "reject"})
    assert e1["seq"] == 1
    assert e1["prev_hash"] == GENESIS_HASH
    assert e2["seq"] == 2
    assert e2["prev_hash"] == e1["hash"]


def test_verify_passes_for_untouched_log(tmp_path):
    log = DecisionLog(str(tmp_path / "d.jsonl"))
    for i in range(5):
        log.append({"event": f"e{i}"})
    ok, bad = log.verify()
    assert ok is True
    assert bad is None


def test_verify_detects_tampering(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(str(path))
    log.append({"event": "a", "amount": 1})
    log.append({"event": "b", "amount": 2})
    log.append({"event": "c", "amount": 3})

    # Tamper with the middle line's payload, keeping its stored hash.
    lines = path.read_text().splitlines()
    obj = json.loads(lines[1])
    obj["payload"]["amount"] = 9999
    lines[1] = json.dumps(obj)
    path.write_text("\n".join(lines) + "\n")

    ok, bad = log.verify()
    assert ok is False
    assert bad == 1  # zero-based index of the broken entry


def test_secrets_redacted_by_key_name(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(str(path))
    log.append({"event": "call", "api_key": "super-secret-value", "model": "claude"})
    raw = path.read_text()
    assert "super-secret-value" not in raw
    assert "REDACTED" in raw


def test_secret_value_pattern_redacted(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(str(path))
    log.append({"event": "call", "prompt": "use token sk-or-v1-abcdef0123456789abcdef0123"})
    raw = path.read_text()
    assert "sk-or-v1-abcdef0123456789abcdef0123" not in raw


def test_concurrent_appends_form_valid_chain(tmp_path):
    path = str(tmp_path / "d.jsonl")
    log = DecisionLog(path)

    def worker(n):
        for i in range(n):
            log.append({"event": "x", "thread": threading.get_ident(), "i": i})

    threads = [threading.Thread(target=worker, args=(25,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entries = log.read_all()
    assert len(entries) == 100
    seqs = sorted(e["seq"] for e in entries)
    assert seqs == list(range(1, 101))  # unique, contiguous
    ok, bad = log.verify()
    assert ok is True, f"chain broke at {bad}"
