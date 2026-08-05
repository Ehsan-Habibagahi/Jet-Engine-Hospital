"""Cross-subset comparison, ablations, and the FD004 bonus analyses.

Everything here reuses ``run_experiment`` -- i.e. the identical split policy,
feature pipeline, metric set, uncertainty method and dashboard contract -- so
the Stage 1 / Stage 2 / bonus comparison is meaningful rather than three
notebooks with different conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import evaluation as E
from .config import ID_COL, CYCLE_COL, N_CONDITIONS, RunConfig, SUBSET_ROLE
from .pipeline import ExperimentResult, _view, run_experiment


# ==========================================================================
# Master comparison table
# ==========================================================================
def master_table(results: dict[str, ExperimentResult]) -> pd.DataFrame:
    rows = [r.tables["run_summary"] for r in results.values()]
    out = pd.concat(rows, ignore_index=True)
    out.insert(1, "n_conditions", out["subset"].map(N_CONDITIONS))
    out.insert(2, "n_fault_modes", out["subset"].map(
        {"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2}))
    return out


def difficulty_grid(results: dict[str, ExperimentResult], metric: str = "test_MAE") -> pd.DataFrame:
    """The 2x2 that isolates the combined challenge.

    C-MAPSS varies exactly two factors, so the four subsets form a complete
    factorial design: {1, 6} operating conditions x {1, 2} fault modes. Reading
    the grid separates the *condition* effect from the *fault-mode* effect and
    exposes whether FD004 is merely their sum or genuinely worse -- which is
    what "isolating the combined challenge" means.

        columns = fault modes, rows = operating conditions
    """
    layout = {("1 condition", "1 fault"): "FD001", ("1 condition", "2 faults"): "FD003",
              ("6 conditions", "1 fault"): "FD002", ("6 conditions", "2 faults"): "FD004"}
    grid = pd.DataFrame(index=["1 condition", "6 conditions"],
                        columns=["1 fault", "2 faults"], dtype=float)
    for (cond, fault), sub in layout.items():
        if sub in results:
            grid.loc[cond, fault] = float(results[sub].tables["run_summary"].iloc[0][metric])
    grid.attrs["metric"] = metric
    return grid


def interaction_summary(results: dict[str, ExperimentResult],
                        metric: str = "test_MAE") -> pd.DataFrame:
    """Main effects and the interaction term of the 2x2 above."""
    g = difficulty_grid(results, metric)
    if g.isna().any().any():
        return pd.DataFrame()
    base = g.loc["1 condition", "1 fault"]
    cond_effect = g.loc["6 conditions", "1 fault"] - base
    fault_effect = g.loc["1 condition", "2 faults"] - base
    combined = g.loc["6 conditions", "2 faults"] - base
    return pd.DataFrame([{
        "metric": metric,
        "baseline_FD001": base,
        "effect_of_6_conditions (FD002-FD001)": cond_effect,
        "effect_of_2_fault_modes (FD003-FD001)": fault_effect,
        "additive_prediction": cond_effect + fault_effect,
        "actual_combined (FD004-FD001)": combined,
        "interaction (actual - additive)": combined - (cond_effect + fault_effect),
    }])


def stack_table(results: dict[str, ExperimentResult], name: str) -> pd.DataFrame:
    frames = []
    for key, r in results.items():
        t = r.tables[name].copy()
        t.insert(0, "run", key)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


# ==========================================================================
# Ablations (Section 6.2 requires at least two)
# ==========================================================================
ABLATIONS = {
    "no_op_settings": dict(use_op_settings=False,
                           question="Do the three operating settings carry information "
                                    "beyond what the sensors already show?"),
    "no_trend_features": dict(use_trend_features=False,
                              question="Are slope / EWM / diff / drift features earning "
                                       "their place, or do rolling means suffice?"),
    "short_window": dict(window_lengths=(3, 5, 10), primary_window=10,
                         question="How much does the trailing window length matter?"),
    "global_scaling": dict(condition_aware=False,
                           question="Does regime-aware normalisation beat one global "
                                    "scaler? (only meaningful for 6-condition subsets)"),
}


def run_ablations(subset: str, base: ExperimentResult, which=("no_op_settings",
                  "no_trend_features", "short_window"), n_boot: int = 200,
                  verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    runs: dict[str, ExperimentResult] = {}
    for tag in which:
        kw = {k: v for k, v in ABLATIONS[tag].items() if k != "question"}
        cfg = RunConfig(subset=subset, tag=tag, **kw)
        runs[tag] = run_experiment(cfg, verbose=verbose, n_boot=n_boot, export=False)

    base_row = base.tables["run_summary"].iloc[0]
    rows = [{"variant": "full model (reference)", "question": "-",
             **{k: base_row[k] for k in ("n_features", "test_MAE", "test_RMSE",
                                         "test_PHM_mean", "interval_coverage",
                                         "PR_AUC_h10", "PR_AUC_h20", "PR_AUC_h30",
                                         "policy_avg_cost", "policy_miss_rate")},
             "delta_test_MAE": 0.0, "delta_PR_AUC_h20": 0.0}]
    for tag, r in runs.items():
        row = r.tables["run_summary"].iloc[0]
        rows.append({
            "variant": tag, "question": ABLATIONS[tag]["question"],
            **{k: row[k] for k in ("n_features", "test_MAE", "test_RMSE", "test_PHM_mean",
                                   "interval_coverage", "PR_AUC_h10", "PR_AUC_h20",
                                   "PR_AUC_h30", "policy_avg_cost", "policy_miss_rate")},
            "delta_test_MAE": float(row["test_MAE"] - base_row["test_MAE"]),
            "delta_PR_AUC_h20": float(row["PR_AUC_h20"] - base_row["PR_AUC_h20"]),
        })
    return runs, pd.DataFrame(rows)


# ==========================================================================
# Bonus: FD004 combined-challenge analyses
# ==========================================================================
def per_regime_breakdown(result: ExperimentResult) -> pd.DataFrame:
    """Performance sliced by operating condition -- the first thing that should
    break when six regimes are mixed with two fault modes."""
    cfg = result.cfg
    fp = result.system.feature_pipeline
    train_raw, test_raw, final_rul = D.load_subset(cfg.subset)
    te = D.add_horizon_labels(D.add_test_rul(test_raw, final_rul), cfg.horizons)
    Xte = fp.transform(te)
    y = D.piecewise_rul(te["RUL"], result.system.rul_cap)
    p = result.system.predict_rul(Xte, interval=False)
    risk = result.system.failure_risk(Xte, cfg.horizons)
    anom = result.system.anomaly_score(Xte)
    regime = fp.assign_regime(te)

    rows = []
    for r in np.unique(regime):
        m = regime == r
        rec = {"regime": int(r), "n_rows": int(m.sum()),
               "n_engines": int(te.loc[m, ID_COL].nunique()),
               "share_of_rows": float(m.mean())}
        rec.update({k: v for k, v in E.regression_metrics(y[m], p[m]).items()
                    if k in ("MAE", "RMSE", "PHM_score_mean")})
        for h in cfg.horizons:
            yh = te[f"fail_within_{h}"].to_numpy()[m]
            ph = risk[f"p_fail_{h}"].to_numpy()[m]
            rec[f"PR_AUC_h{h}"] = (float(E.average_precision_score(yh, ph))
                                   if yh.sum() else np.nan)
        rec["mean_anomaly_percentile"] = float(anom.to_numpy()[m].mean())
        rec["anomaly_alert_rate_at_deployed_threshold"] = float(
            (anom.to_numpy()[m] >= result.system.anomaly_threshold).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def anomaly_threshold_stability(result: ExperimentResult) -> pd.DataFrame:
    """Is ONE anomaly threshold stable across regimes, or does each regime need
    its own? Compares the global validation percentile against regime-specific
    percentiles computed on validation engines."""
    cfg = result.cfg
    fp = result.system.feature_pipeline
    train_raw, test_raw, final_rul = D.load_subset(cfg.subset)
    train_all = D.add_train_rul(train_raw)
    va = D.subset_by_engines(train_all, result.split.val_ids)
    te = D.add_test_rul(test_raw, final_rul)

    Xva, Xte = fp.transform(va), fp.transform(te)
    bank = result.system.anomaly
    det = result.system.anomaly_detector
    raw_va = bank.raw_scores(Xva)[det].to_numpy()
    raw_te = bank.raw_scores(Xte)[det].to_numpy()
    reg_va, reg_te = fp.assign_regime(va), fp.assign_regime(te)

    pct = 100.0 * result.system.anomaly_threshold   # the deployed alert percentile
    global_thr = float(np.percentile(raw_va, pct))
    rows = []
    for r in np.unique(reg_te):
        mv, mt = reg_va == r, reg_te == r
        if mv.sum() < 50:
            continue
        local_thr = float(np.percentile(raw_va[mv], pct))
        rows.append({
            "regime": int(r),
            "global_threshold": round(global_thr, 4),
            "regime_specific_threshold": round(local_thr, 4),
            "relative_gap_%": round(100 * (local_thr - global_thr) / abs(global_thr + 1e-9), 1),
            "test_alert_rate_global_thr": float((raw_te[mt] >= global_thr).mean()),
            "test_alert_rate_regime_thr": float((raw_te[mt] >= local_thr).mean()),
            "n_val_rows": int(mv.sum()), "n_test_rows": int(mt.sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        spread = out["test_alert_rate_global_thr"].max() - out["test_alert_rate_global_thr"].min()
        out.attrs["alert_rate_spread_under_global_threshold"] = float(spread)
        out["alert_rate_spread_across_regimes"] = round(spread, 4)
    return out


def performance_by_slice(result: ExperimentResult) -> pd.DataFrame:
    """Sequence length and RUL region slices (bonus checklist)."""
    cfg = result.cfg
    fp = result.system.feature_pipeline
    _, test_raw, final_rul = D.load_subset(cfg.subset)
    te = D.add_horizon_labels(D.add_test_rul(test_raw, final_rul), cfg.horizons)
    Xte = fp.transform(te)
    y = D.piecewise_rul(te["RUL"], result.system.rul_cap)
    p = result.system.predict_rul(Xte, interval=False)

    lengths = te.groupby(ID_COL)[CYCLE_COL].max()
    edges = np.quantile(lengths, [0, 1 / 3, 2 / 3, 1.0])
    bucket = te[ID_COL].map(
        pd.cut(lengths, bins=np.unique(edges), include_lowest=True).astype(str))

    rows = []
    for name, m in [(f"sequence length {b}", (bucket == b).to_numpy())
                    for b in sorted(bucket.dropna().unique())]:
        rows.append({"slice_type": "observed sequence length", "slice": name,
                     "n_rows": int(m.sum()), **E.regression_metrics(y[m], p[m])})
    for name, lo, hi in E.LIFE_REGIONS:
        m = (te["RUL"].to_numpy() >= lo) & (te["RUL"].to_numpy() < hi)
        if m.sum():
            rows.append({"slice_type": "RUL region", "slice": name,
                         "n_rows": int(m.sum()), **E.regression_metrics(y[m], p[m])})
    return pd.DataFrame(rows)


def transfer_matrix(results: dict[str, ExperimentResult],
                    subsets=("FD001", "FD002", "FD003", "FD004"),
                    target: str = "FD004") -> pd.DataFrame:
    """Apply each subset's *locked* system to another subset's test engines.

    This is the "transfer vs dedicated model" comparison. Note the honest
    caveat: a source pipeline was fitted with its own sensor list, regime
    clusters and cap, so transfer is genuinely zero-shot -- no refitting.
    """
    _, test_raw, final_rul = D.load_subset(target)
    te = D.add_horizon_labels(D.add_test_rul(test_raw, final_rul))
    rows = []
    for src in subsets:
        if src not in results:
            continue
        sysm = results[src].system
        try:
            Xte = sysm.feature_pipeline.transform(te)
            y = D.piecewise_rul(te["RUL"], sysm.rul_cap)
            p = sysm.predict_rul(Xte, interval=False)
            risk = sysm.failure_risk(Xte)
            rec = {"source_model": src, "source_role": SUBSET_ROLE[src],
                   "target_test_set": target,
                   "dedicated": src == target, "rul_cap_used": sysm.rul_cap}
            rec.update({k: v for k, v in E.regression_metrics(y, p).items()
                        if k in ("MAE", "RMSE", "PHM_score_mean")})
            for h in (10, 20, 30):
                yh = te[f"fail_within_{h}"].to_numpy()
                rec[f"PR_AUC_h{h}"] = float(
                    E.average_precision_score(yh, risk[f"p_fail_{h}"].to_numpy()))
            rows.append(rec)
        except Exception as exc:  # incompatible sensor set etc. -- report it
            rows.append({"source_model": src, "target_test_set": target,
                         "dedicated": src == target, "error": str(exc)[:120]})
    out = pd.DataFrame(rows)
    if "MAE" in out:
        out = out.sort_values("MAE")
    return out


def condition_normalisation_comparison(subset: str, n_boot: int = 200,
                                       verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Global scaling vs condition-aware normalisation, on the same split.

    This is the central bonus question for FD002/FD004: can the method separate
    operating-context variation from degradation?
    """
    runs = {
        "condition_aware": run_experiment(
            RunConfig(subset=subset, condition_aware=True, tag="condition_aware"),
            verbose=verbose, n_boot=n_boot, export=False),
        "global_scaling": run_experiment(
            RunConfig(subset=subset, condition_aware=False, tag="global_scaling"),
            verbose=verbose, n_boot=n_boot, export=False),
    }
    tbl = pd.concat([r.tables["run_summary"].assign(normalisation=k)
                     for k, r in runs.items()], ignore_index=True)
    cols = ["normalisation", "n_features", "rul_cap", "rul_model", "test_MAE", "test_RMSE",
            "test_PHM_mean", "interval_coverage", "PR_AUC_h10", "PR_AUC_h20", "PR_AUC_h30",
            "anomaly_detector", "policy_avg_cost", "policy_miss_rate", "policy_mean_lead_time"]
    return runs, tbl[cols]
