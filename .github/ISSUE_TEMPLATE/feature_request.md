---
name: Feature request
about: Propose a new feature, validator, research direction, or engineering improvement
title: '[Feature] '
labels: enhancement
assignees: ''
---

## Summary

One or two sentences describing the proposed change.

## Motivation

What problem does this solve, or what research question does it answer? If this is a new strategy or signal, what is the proposed edge hypothesis?

## Proposed implementation

Describe the approach at a high level. For a new validator or gate, describe:
- What inputs it takes
- What the pass/fail criteria are
- What negative (fail-closed) test would prove it rejects bad input

## Honesty check

- [ ] This does not claim edge without out-of-sample validation
- [ ] The proposed gate fails closed on missing or uncertain data
- [ ] Any backtest uses `TimeSeriesSplit` or equivalent (no look-ahead)
- [ ] If this would affect live trading state or live params, it goes through the `HUMAN_GATED` proposal pattern

## Alternatives considered

Any other approaches you considered and why you prefer this one.

## Additional context

Links to relevant papers, existing code in the repo it would build on, or related issues.
