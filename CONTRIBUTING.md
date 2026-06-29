# Contributing to QuantForge

Thank you for your interest in contributing. QuantForge is a research harness, not a money-printer — contributions that improve rigor, honesty, test coverage, and documentation are the highest-value additions.

## Getting started

```bash
git clone https://github.com/<your-org>/quantforge.git
cd quantforge
./setup.sh
source .venv/bin/activate
python3 -m pytest tests/ -q       # All tests must pass before you start
```

## Development workflow

1. **Fork the repo** and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first.** Every new validator, gate, or invariant must have:
   - A positive test (it accepts a valid input).
   - A negative / fail-closed test (it rejects bad or uncertain input).
   The test in `tests/test_quantforge_ml_gate_truth.py` is a good model — it constructs a deliberately leaky feature and asserts the gate catches it.

3. **Run the full test suite** before opening a PR:
   ```bash
   python3 -m pytest tests/ -q
   ```

4. **Run the master verifier** if you have a production host configured:
   ```bash
   bash verify_quantforge.sh
   ```

5. **Syntax check** all scripts:
   ```bash
   for f in scripts/*.py; do python3 -m py_compile "$f" && echo "OK: $f"; done
   ```

6. **Open a PR** with a clear description of what changed and why.

## Code style

- Python 3.9+. No dependencies outside `requirements.txt` without discussion.
- Prefer explicit over implicit. Fail closed when uncertain.
- Every new script that runs on a schedule must have a flock guard to prevent stacking.
- Do not add `HUMAN_GATED`-bypassing code paths. If a change affects live trading state or live params, it must go through the proposal-artifact pattern.
- No fabricated performance numbers, Sharpe ratios, or invented edge claims — ever.

## What to contribute

Good contributions:
- Additional fail-closed tests for existing gates
- Improvements to the cost-honest carry backtest
- Walk-forward / Monte-Carlo stress harness as a reusable library (currently bespoke per-script)
- Documentation corrections or clarifications
- Bug fixes with a reproducing test

Things to avoid:
- New ML strategies claiming edge on the existing 0.50-AUC free-data signal
- Widening the carry universe without a new validator that survives red-teaming
- Any change that modifies live trading state automatically (must be human-gated)
- Bypassing the 6-criteria validation gate

## Reporting issues

Before opening a bug report, check `docs/QUANTFORGE_SYSTEM_STATE.md` — known issues and rejected strategies are documented there. If you have a genuine new finding, open an issue with:

- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Whether `verify_quantforge.sh` exits 0 or fails

See `.github/ISSUE_TEMPLATE/` for issue templates.

## Using Claude Code

This project includes `CLAUDE.md` which gives Claude Code full context on the codebase, commands, and architecture.

```bash
claude    # Start Claude Code — reads CLAUDE.md automatically
```

Claude Code works well for: exploring the codebase, writing tests, understanding the safety gates, and proposing research improvements. Always run `python3 -m pytest tests/ -q` to verify any Claude-generated changes before committing.

## License

By contributing, you agree your contributions will be licensed under the MIT License.
