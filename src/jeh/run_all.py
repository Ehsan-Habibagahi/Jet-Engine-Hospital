"""Driver: run the locked protocol on every subset, plus ablations and the
FD004 bonus analyses, and cache the results.

    python -m jeh.run_all              # everything (~1 hour)
    python -m jeh.run_all FD001        # one subset, no ablations

The notebook imports :func:`load_or_run` so a re-run is fast, and setting
``REUSE_CACHE = False`` there forces a full recomputation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from . import experiments as X
from .config import ARTIFACT_DIR, RunConfig, SUBSETS, TABLE_DIR, set_seed
from .pipeline import ExperimentResult, run_experiment

CACHE = ARTIFACT_DIR / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def load_or_run(subset: str, reuse: bool = True, **cfg_kw) -> ExperimentResult:
    cfg = RunConfig(subset=subset, **cfg_kw)
    path = CACHE / f"result_{cfg.name}.joblib"
    if reuse and path.exists():
        return joblib.load(path)
    res = run_experiment(cfg, export=(cfg.tag == ""))
    joblib.dump(res, path, compress=3)
    return res


def main(argv: list[str]) -> None:
    set_seed()
    subsets = [a for a in argv if a in SUBSETS] or list(SUBSETS)
    do_extra = not [a for a in argv if a in SUBSETS]
    t0 = time.perf_counter()

    results: dict[str, ExperimentResult] = {}
    for s in subsets:
        print(f"\n{'=' * 70}\n  {s}\n{'=' * 70}")
        results[s] = load_or_run(s, reuse=True)

    master = X.master_table(results)
    master.to_csv(TABLE_DIR / "master_comparison.csv", index=False)
    print("\n--- MASTER COMPARISON ---")
    print(master.to_string())

    if do_extra:
        # Ablations on the foundation subset and on the bonus subset.
        for s, which in (
            ("FD001", ("no_op_settings", "no_trend_features", "short_window")),
            ("FD004", ("no_op_settings", "no_trend_features", "short_window",
                       "global_scaling")),
        ):
            print(f"\n{'=' * 70}\n  ABLATIONS -- {s}\n{'=' * 70}")
            runs = {}
            for tag in which:
                kw = {k: v for k, v in X.ABLATIONS[tag].items() if k != "question"}
                runs[tag] = load_or_run(s, reuse=True, tag=tag, **kw)
            rows = [{"variant": "full model (reference)", "question": "-",
                     **_summary_cols(results[s])}]
            for tag, r in runs.items():
                rows.append({"variant": tag, "question": X.ABLATIONS[tag]["question"],
                             **_summary_cols(r)})
            tbl = pd.DataFrame(rows)
            base = tbl.iloc[0]
            tbl["delta_test_MAE"] = tbl["test_MAE"] - base["test_MAE"]
            tbl["delta_PR_AUC_h20"] = tbl["PR_AUC_h20"] - base["PR_AUC_h20"]
            tbl["delta_policy_cost"] = tbl["policy_avg_cost"] - base["policy_avg_cost"]
            tbl.to_csv(TABLE_DIR / f"ablations_{s}.csv", index=False)
            print(tbl.to_string())

        # Bonus analyses on FD004
        print(f"\n{'=' * 70}\n  BONUS -- FD004\n{'=' * 70}")
        r4 = results["FD004"]
        for name, fn in (
            ("fd004_per_regime", lambda: X.per_regime_breakdown(r4)),
            ("fd004_threshold_stability", lambda: X.anomaly_threshold_stability(r4)),
            ("fd004_slices", lambda: X.performance_by_slice(r4)),
            ("fd004_transfer", lambda: X.transfer_matrix(results, target="FD004")),
        ):
            tbl = fn()
            tbl.to_csv(TABLE_DIR / f"{name}.csv", index=False)
            print(f"\n--- {name} ---")
            print(tbl.to_string())

    print(f"\nTOTAL {time.perf_counter() - t0:.0f}s")


def _summary_cols(r: ExperimentResult) -> dict:
    row = r.tables["run_summary"].iloc[0]
    keys = ("n_features", "rul_cap", "test_MAE", "test_RMSE", "test_PHM_mean",
            "interval_coverage", "interval_width", "PR_AUC_h10", "PR_AUC_h20",
            "PR_AUC_h30", "policy_avg_cost", "policy_miss_rate",
            "policy_mean_lead_time")
    return {k: row[k] for k in keys}


if __name__ == "__main__":
    main(sys.argv[1:])
