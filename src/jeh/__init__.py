"""Jet Engine Hospital -- a leakage-safe early-warning system for NASA C-MAPSS.

Modules
-------
config        paths, seeds, cost policy, run configuration
data          loading, RUL/horizon labels, engine-level split, data audit
features      causal FeaturePipeline (windows, regime-aware residuals)
models        regression/classification zoos, AnomalyBank, PrognosticsSystem
uncertainty   split conformal intervals, calibration diagnostics
evaluation    metrics, engine-level bootstrap, lead time, asymmetric cost
policy        CONTINUE / INSPECT / STOP decision layer
pipeline      run_experiment(): the locked end-to-end protocol
plots         figures used by the notebook
"""
from .config import RunConfig, CostPolicy, set_seed, SUBSETS, HORIZONS  # noqa: F401

__version__ = "1.0.0"
