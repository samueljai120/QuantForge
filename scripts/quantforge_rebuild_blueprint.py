#!/usr/bin/env python3
"""Shared QuantForge deeper-rebuild blueprint helpers.

This module turns a generic "research hold" state into a concrete rebuild
program that downstream research-core loops can carry across dashboard, memory, sync,
and operator workflows.
"""

from __future__ import annotations


def data_source_specs():
    return [
        {
            "name": "venue_ohlcv_plus",
            "priority": "required",
            "purpose": "Keep the existing candle lane, but normalize it per venue and symbol so it is no longer the only market context.",
            "minimum_fields": [
                "timestamp",
                "symbol",
                "venue",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trade_count",
            ],
            "artifacts": ["raw candles", "normalized bar parquet", "venue-quality report"],
        },
        {
            "name": "top_of_book_snapshots",
            "priority": "required",
            "purpose": "Measure spread, depth imbalance, and local execution quality instead of assuming frictionless fills.",
            "minimum_fields": [
                "timestamp",
                "symbol",
                "best_bid",
                "best_ask",
                "bid_size_1",
                "ask_size_1",
                "bid_size_5",
                "ask_size_5",
            ],
            "artifacts": ["book snapshot jsonl", "spread-depth parquet", "execution realism report"],
        },
        {
            "name": "derivatives_state",
            "priority": "required",
            "purpose": "Add funding, basis, open interest, and liquidation pressure so the model can detect crowded or stressed conditions.",
            "minimum_fields": [
                "timestamp",
                "symbol",
                "funding_rate",
                "open_interest",
                "basis_bps",
                "long_short_ratio",
                "liquidation_long_usd",
                "liquidation_short_usd",
            ],
            "artifacts": ["derivatives parquet", "crowding report"],
        },
        {
            "name": "market_breadth_context",
            "priority": "required",
            "purpose": "Anchor per-symbol decisions inside broader crypto risk-on or risk-off conditions.",
            "minimum_fields": [
                "timestamp",
                "btc_return_1h",
                "eth_return_1h",
                "majors_breadth",
                "alts_breadth",
                "stablecoin_dominance",
                "market_volume_breadth",
            ],
            "artifacts": ["breadth parquet", "regime context parquet"],
        },
        {
            "name": "trade_tape_or_proxy",
            "priority": "stretch",
            "purpose": "Capture aggressive buying or selling pressure when available; otherwise approximate from short-horizon trade counts and quote changes.",
            "minimum_fields": [
                "timestamp",
                "symbol",
                "aggressive_buy_volume",
                "aggressive_sell_volume",
                "micro_return_30s",
                "micro_vol_30s",
            ],
            "artifacts": ["microstructure parquet", "pressure imbalance report"],
        },
        {
            "name": "calendar_and_event_flags",
            "priority": "stretch",
            "purpose": "Mark macro prints, exchange incidents, listing events, and funding windows so false positives are easier to quarantine.",
            "minimum_fields": [
                "timestamp",
                "event_type",
                "event_severity",
                "symbol_scope",
                "hours_to_event",
            ],
            "artifacts": ["event flags parquet", "event overlap report"],
        },
    ]


def feature_family_specs():
    return [
        {
            "name": "execution_quality",
            "goal": "Estimate whether a signal survives spread, depth, and slippage before ranking it.",
            "examples": [
                "quoted spread bps",
                "effective spread proxy",
                "top-5 depth imbalance",
                "expected impact for target notionals",
                "book staleness",
            ],
        },
        {
            "name": "crowding_and_stress",
            "goal": "Detect when continuation signals are really crowded squeeze or flush setups.",
            "examples": [
                "funding z-score",
                "open-interest acceleration",
                "basis divergence",
                "liquidation cluster pressure",
                "long-short skew",
            ],
        },
        {
            "name": "cross_sectional_relative_strength",
            "goal": "Prefer symbols winning against majors and sector peers rather than isolated noisy movers.",
            "examples": [
                "symbol vs BTC beta-adjusted return",
                "symbol vs sector basket",
                "rolling rank vs top-liquidity universe",
                "volume-confirmed relative strength",
            ],
        },
        {
            "name": "regime_and_state",
            "goal": "Make entropy, breadth, volatility, and macro state first-class gating features instead of side metrics.",
            "examples": [
                "entropy regime",
                "breadth regime",
                "volatility shock buckets",
                "trend persistence",
                "market stress composite",
            ],
        },
        {
            "name": "setup_specific_context",
            "goal": "Train continuation, rebound, exhaustion, and breakout setups with setup-local evidence instead of one monolithic score.",
            "examples": [
                "breakout follow-through window",
                "rebound quality after flush",
                "exhaustion probability after vertical move",
                "trend persistence after pullback",
            ],
        },
        {
            "name": "portfolio_and_correlation",
            "goal": "Reduce clustered long exposure by understanding correlation and book overlap before entry.",
            "examples": [
                "symbol correlation to open book",
                "factor overlap with majors",
                "sector concentration score",
                "incremental drawdown contribution",
            ],
        },
    ]


def phase_specs():
    return [
        {
            "phase": "phase_0_contracts",
            "goal": "Lock the rebuild contract before collecting more data.",
            "deliverables": [
                "dataset schema for each required source",
                "feature store directory layout",
                "rebuild scorecard with baseline metrics and exit gates",
            ],
            "acceptance": "the research core can name every required artifact and where it will be written.",
        },
        {
            "phase": "phase_1_ingestion",
            "goal": "Collect richer market context with source-level quality checks.",
            "deliverables": [
                "book snapshot collector",
                "derivatives-state collector",
                "market breadth builder",
                "source freshness and null-rate report",
            ],
            "acceptance": "At least 14 days of continuous required-source coverage for the target major-symbol universe.",
        },
        {
            "phase": "phase_2_labels_and_targets",
            "goal": "Replace blunt forward-return labels with setup-local, execution-aware targets.",
            "deliverables": [
                "setup label definitions",
                "net-of-cost target builder",
                "failure taxonomy for false breakouts, crowded squeezes, and bad liquidity entries",
            ],
            "acceptance": "Each setup class has enough positive and negative samples to support bounded training.",
        },
        {
            "phase": "phase_3_model_split",
            "goal": "Split the single decision path into prediction, regime filter, risk filter, and execution-policy layers.",
            "deliverables": [
                "prediction model candidate",
                "regime gate candidate",
                "risk filter candidate",
                "execution-cost/ranking policy",
            ],
            "acceptance": "Layer outputs can be evaluated independently on holdout and paper replay.",
        },
        {
            "phase": "phase_4_replay_and_paper",
            "goal": "Beat the old baseline under realistic fills before another live-style paper cycle.",
            "deliverables": [
                "replay backtest with spread/slippage assumptions",
                "major-symbol-only bounded paper candidate",
                "subgroup performance report by setup and regime",
            ],
            "acceptance": "Candidate improves expectancy and drawdown versus baseline under execution-realistic assumptions.",
        },
    ]


def build_rebuild_program():
    return {
        "objective": "Rebuild QuantForge around richer market context, execution realism, and setup-specialized decision layers before any new bounded trial.",
        "target_universe": {
            "primary": ["BTC", "ETH", "SOL", "XRP", "BCH", "TRX"],
            "policy": "Majors-first during rebuild. Non-majors re-enter only after the rebuilt lane proves subgroup edge.",
        },
        "data_sources": data_source_specs(),
        "feature_families": feature_family_specs(),
        "phases": phase_specs(),
        "research_hold_exit_criteria": [
            "Required data sources have stable coverage and quality reporting.",
            "Execution-aware labels and feature families are generated for the primary universe.",
            "The split model stack beats baseline in replay before a new paper trial is queued.",
            "Governance no longer recommends REVIEW or DO_NOT_PROMOTE for the rebuilt candidate lane.",
        ],
    }
