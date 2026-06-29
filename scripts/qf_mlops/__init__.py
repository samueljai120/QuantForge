"""qf_mlops — reproducible-research / MLOps foundation (Phase 1).

Model cards with full provenance, a model registry with a promotion state
machine that cannot advance on AUC alone, and baseline-arm decomposition that
measures the *incremental* value of ML against honest benchmarks under realistic
costs. Builds on qf_safety (atomic, versioned persistence).
"""

__all__ = ["model_card", "model_registry", "baselines"]
