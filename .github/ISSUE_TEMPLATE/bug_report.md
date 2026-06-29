---
name: Bug report
about: Report a reproducible defect in a script, test, gate, or validator
title: '[Bug] '
labels: bug
assignees: ''
---

## Describe the bug

A clear description of what the bug is and what you expected to happen instead.

## Steps to reproduce

```bash
# Exact commands that trigger the bug
python3 scripts/quantforge_paper.py scan
```

## Expected behavior

What should have happened.

## Actual behavior

What happened instead. Paste the full error output if applicable:

```
<paste error output here>
```

## Environment

- OS:
- Python version (`python3 --version`):
- Installed package versions (`pip freeze | grep -E 'numpy|pandas|sklearn|xgboost|lightgbm'`):
- `verify_quantforge.sh` exit code (if applicable):

## Additional context

- Is the bug in a gate or validator? Which one?
- Does the bug cause the system to silently pass when it should fail, or silently fail when it should pass?
- Any relevant data or config state (sanitize any credentials before pasting).
