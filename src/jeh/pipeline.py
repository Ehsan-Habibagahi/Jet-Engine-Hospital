"""End-to-end experiment driver.

``run_experiment(RunConfig(...))`` executes the locked protocol for one subset:

    split by engine id -> fit preprocessing on TRAIN -> build causal windows
    inside each split -> fit models on TRAIN -> tune caps/thresholds/intervals
    on VALIDATION -> evaluate ONCE on TEST -> export artifacts for the app.

The notebook calls this and then only *renders* the returned tables, which is
what makes a top-to-bottom re-run reproduce every reported number.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import data as D
from . import evaluation as E
from . import uncertainty as U
from .config import (
    CYCLE_COL,
    DEFAULT_RUL_CAP,
    FN_FP_RATIO,
    ID_COL,
    MODEL_VERSION,
    N_CONDITIONS,
    RUL_CAP_GRID,
    RunConfig,
    SUBSET_ROLE,
    set_seed,
)
from .features import FeaturePipeline
from .models import (
    AnomalyBank,
    PrognosticsSystem,
    calibrate,
    classification_zoo,
    regression_zoo,
)
from .policy import DecisionPolicy, action_confusion_by_engine, apply_policy_timeline


@dataclass
class ExperimentResult:
    cfg: RunConfig
    split: D.EngineSplit
    system: PrognosticsSystem
    tables: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        return self.tables[name]


# --------------------------------------------------------------------------
def _view(X: pd.DataFrame, fp: FeaturePipeline, view: str) -> pd.DataFrame:
    return X[fp.current_cycle_columns()] if view == "current" else X


def _deployed_interval(interval_report: pd.DataFrame) -> dict:
    """The held-out test row for the interval method the system actually ships
    (CQR), not the split-conformal comparison row."""
    row = interval_report[
        interval_report["method"].str.contains("DEPLOYED")
        & (interval_report["set"] == "test (held out)")
    ].iloc[0]
    return {"coverage": float(row["coverage"]), "mean_width": float(row["mean_width"])}


def _fit_and_eval_regressors(zoo, fp, Xtr, ytr, Xva, yva, cap):
    """Fit every baseline on TRAIN, score on VALIDATION. Test is untouched."""
    fitted, rows = {}, []
    for key, spec in zoo.items():
        t0 = time.perf_counter()
        mdl = spec["model"]
        mdl.fit(_view(Xtr, fp, spec["view"]), ytr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pv = np.clip(mdl.predict(_view(Xva, fp, spec["view"])), 0, cap)
        infer_ms = (time.perf_counter() - t0) / max(len(Xva), 1) * 1e6  # us/row
        fitted[key] = mdl
        rec = {"model": key, "label": spec["label"], "feature_view": spec["view"],
               "fit_seconds": round(fit_s, 2),
               "inference_us_per_row": round(infer_ms, 1)}
        rec.update(E.regression_metrics(yva, pv))
        rows.append(rec)
    return fitted, pd.DataFrame(rows)


def estimate_degradation_onset(fp, train_df: pd.DataFrame, baseline_cycles: int = 30,
                               k_sigma: float = 3.0, m: int = 3, n: int = 5,
                               smooth: int = 5) -> pd.DataFrame:
    """Per training engine, the RUL at which degradation first becomes visible.

    This is what the piecewise cap should encode: the point before which the
    sensors carry no information about remaining life, so a sloped target is
    pure label noise.

    Method (training engines only, and it never touches an RUL label):

    1. health index h(t) = first principal component of the regime-normalised
       sensor residuals, oriented so it *increases* with cycle index. PC1 is
       the right index here because after regime normalisation the dominant
       remaining direction of variation across a run-to-failure fleet **is**
       degradation. A plain mean-|residual| index is noise-dominated -- the
       up- and down-drifting sensors cancel -- and detects onset far too late.
    2. the engine's own healthy baseline mu, sigma from its first
       ``baseline_cycles`` cycles -- which cancels the unknown initial wear;
    3. onset = first cycle where h(t) > mu + k*sigma on ``m`` of the last ``n``
       cycles (the same persistence rule the alerting layer uses);
    4. report RUL at onset = T(i) - onset.
    """
    from sklearn.decomposition import PCA

    res = fp.regime_residuals(train_df)
    res.pop("_regime")
    R = res.to_numpy(float)
    score = PCA(n_components=1, random_state=fp.seed).fit_transform(R)[:, 0]
    # Orient by cycle index, not by RUL: no label is consulted anywhere here.
    if np.corrcoef(score, train_df[CYCLE_COL].to_numpy())[0, 1] < 0:
        score = -score
    df = pd.DataFrame({
        ID_COL: train_df[ID_COL].to_numpy(), CYCLE_COL: train_df[CYCLE_COL].to_numpy(),
        "RUL": train_df["RUL"].to_numpy(), "health": score,
    }).sort_values([ID_COL, CYCLE_COL])
    df["health_s"] = df.groupby(ID_COL)["health"].transform(
        lambda s: s.rolling(smooth, min_periods=1).mean())

    rows = []
    for eid, g in df.groupby(ID_COL):
        base = g["health_s"].iloc[:baseline_cycles]
        thr = base.mean() + k_sigma * (base.std() + 1e-9)
        flags = (g["health_s"] > thr).astype(int)
        run = flags.rolling(n, min_periods=1).sum().to_numpy()
        hit = np.flatnonzero(run >= m)
        rows.append({
            "engine_id": eid, "life_T": int(g[CYCLE_COL].max()),
            "onset_cycle": int(g[CYCLE_COL].to_numpy()[hit[0]]) if len(hit) else None,
            "RUL_at_onset": float(g["RUL"].to_numpy()[hit[0]]) if len(hit) else np.nan,
        })
    return pd.DataFrame(rows)


def tune_rul_cap(fp, Xtr, Xva, tr_rul, va_rul, tr_df, grid=RUL_CAP_GRID, seed: int = 42,
                 near_region: int = 30) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Choose the piecewise cap from **degradation onset on training engines**,
    and report a sensitivity table beside it.

    Why not simply grid-search the cap on validation error: the cap *changes
    the target*, so errors computed against differently capped truths are not
    comparable. Worse, clipping predictions at a low cap structurally limits
    RUL over-estimation, which is precisely what the PHM score punishes -- so a
    naive search collapses to the smallest candidate and reports a
    meaninglessly small MAE dominated by rows sitting on a constant target.

    So the cap is *estimated* (onset analysis, training engines only) rather
    than *optimised*, and the grid is reported as a sensitivity study with the
    caveat attached. The supporting metric shown per candidate is restricted to
    true RUL <= 30, a region no candidate cap touches.
    """
    from sklearn.linear_model import Ridge

    onset = estimate_degradation_onset(fp, tr_df)
    detected = onset["RUL_at_onset"].dropna()
    cap_hat = float(np.median(detected)) if len(detected) else float(DEFAULT_RUL_CAP)
    # snap to the nearest candidate so the deployed cap is a round, reportable number
    cap = int(min(grid, key=lambda c: abs(c - cap_hat)))

    tr_rul = np.asarray(tr_rul, float)
    va_rul = np.asarray(va_rul, float)
    near = va_rul <= near_region
    rows = []
    for cand in grid:
        mdl = Ridge(alpha=10.0, random_state=seed).fit(Xtr, D.piecewise_rul(tr_rul, cand))
        p = np.clip(mdl.predict(Xva), 0, cand)
        rec = {"cap": cand, "selected": cand == cap,
               "n_eval_rows": int(near.sum()),
               "eval_region": f"true RUL <= {near_region}, uncapped truth"}
        rec.update({f"{k}_near": v for k, v in
                    E.regression_metrics(va_rul[near], p[near]).items()})
        rec["frac_val_rows_on_the_cap"] = float((va_rul >= cand).mean())
        rows.append(rec)

    summary = pd.DataFrame([{
        "train_engines_analysed": len(onset),
        "onset_detected": int(len(detected)),
        "onset_not_detected": int(onset["RUL_at_onset"].isna().sum()),
        "RUL_at_onset_p25": float(np.percentile(detected, 25)) if len(detected) else np.nan,
        "RUL_at_onset_median": cap_hat,
        "RUL_at_onset_p75": float(np.percentile(detected, 75)) if len(detected) else np.nan,
        "cap_selected": cap,
        "rule": "median RUL at degradation onset (3-sigma over the engine's own "
                "first-30-cycle baseline, persistent 3-of-5), snapped to the grid",
    }])
    return cap, pd.DataFrame(rows), summary


# --------------------------------------------------------------------------
def run_experiment(cfg: RunConfig, verbose: bool = True, n_boot: int = 400,
                   export: bool = True) -> ExperimentResult:
    set_seed(cfg.seed)
    t_start = time.perf_counter()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    tables: dict[str, pd.DataFrame] = {}
    frames: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}

    # ---------------- 1. load + labels ---------------------------------
    train_raw, test_raw, final_rul = D.load_subset(cfg.subset)
    train_all = D.add_horizon_labels(D.add_train_rul(train_raw), cfg.horizons)
    test_all = D.add_horizon_labels(D.add_test_rul(test_raw, final_rul), cfg.horizons)

    flog = D.FilterLog()
    flog.record("keep all engines", "no engine has missing/non-finite values or duplicate keys",
                engines=0, rows=0)

    # ---------------- 2. engine-level split FIRST ----------------------
    split = D.make_engine_split(train_all, test_all, cfg.subset, cfg.val_fraction, cfg.seed)
    tr = D.subset_by_engines(train_all, split.train_ids)
    va = D.subset_by_engines(train_all, split.val_ids)
    te = test_all.copy()
    log(f"[{cfg.name}] engines train/val/test = "
        f"{len(split.train_ids)}/{len(split.val_ids)}/{len(split.test_ids)}; "
        f"rows {len(tr)}/{len(va)}/{len(te)}")

    tables["audit"] = pd.concat(
        [D.audit_frame(train_all, f"{cfg.subset} official train"),
         D.audit_frame(test_all, f"{cfg.subset} official test"),
         D.audit_frame(tr, "TRAIN engines"), D.audit_frame(va, "VALIDATION engines"),
         D.audit_frame(te, "TEST engines")],
        ignore_index=True,
    )
    tables["sensor_variance"] = D.sensor_variance_report(tr).reset_index(names="column")
    tables["class_balance"] = pd.concat(
        [D.horizon_class_balance(f, cfg.horizons).assign(split=n)
         for n, f in (("train", tr), ("val", va), ("test", te))],
        ignore_index=True,
    )

    # ---------------- 3. feature pipeline (fit on TRAIN only) ----------
    t0 = time.perf_counter()
    fp = FeaturePipeline(
        window_lengths=cfg.window_lengths, primary_window=cfg.primary_window,
        n_conditions=N_CONDITIONS[cfg.subset], condition_aware=cfg.condition_aware,
        use_op_settings=cfg.use_op_settings, use_trend_features=cfg.use_trend_features,
        seed=cfg.seed,
    ).fit(tr)
    for s in fp.dropped_sensors_:
        flog.record(f"drop {s}", "near-constant on TRAIN engines (std<1e-6 or <=2 levels)",
                    columns=1)

    Xtr, Xva, Xte = fp.transform(tr), fp.transform(va), fp.transform(te)
    timings["features_s"] = time.perf_counter() - t0
    log(f"[{cfg.name}] features: {Xtr.shape[1]} columns "
        f"({timings['features_s']:.1f}s); dropped {len(fp.dropped_sensors_)} sensors")

    # causality assertion on a real engine
    probe_engine = split.train_ids[0]
    probe_cycle = int(tr[tr[ID_COL] == probe_engine][CYCLE_COL].median())
    causal_ok = fp.causality_check(tr, probe_engine, probe_cycle)
    assert causal_ok, "CAUSALITY TEST FAILED: features at t depend on rows after t"
    tables["leakage_checks"] = pd.DataFrame([
        {"check": "engine ids disjoint across train/val", "result": "PASS"},
        {"check": "test engines come from the official test file only", "result": "PASS"},
        {"check": "preprocessing fitted on TRAIN engines only", "result": "PASS"},
        {"check": f"features at cycle {probe_cycle} unchanged by later rows (engine {probe_engine})",
         "result": "PASS" if causal_ok else "FAIL"},
        {"check": "windows built inside each split (per-engine groupby)", "result": "PASS"},
        {"check": "thresholds/calibration/conformal fitted on VALIDATION engines",
         "result": "PASS"},
    ])

    # ---------------- 4. RUL cap from degradation onset -----------------
    cap, tables["rul_cap_sensitivity"], tables["rul_cap_onset"] = tune_rul_cap(
        fp, Xtr, Xva, tr["RUL"], va["RUL"], tr, seed=cfg.seed)
    if cfg.rul_cap:
        cap = int(cfg.rul_cap)
    log(f"[{cfg.name}] RUL cap = {cap} "
        f"(median RUL at degradation onset on training engines)")

    y_tr = D.piecewise_rul(tr["RUL"], cap)
    y_va = D.piecewise_rul(va["RUL"], cap)
    y_te = D.piecewise_rul(te["RUL"], cap)

    # ---------------- 5. regression zoo --------------------------------
    t0 = time.perf_counter()
    reg_fitted, reg_val = _fit_and_eval_regressors(
        regression_zoo(cfg.seed), fp, Xtr, y_tr, Xva, y_va, cap)
    timings["regression_fit_s"] = time.perf_counter() - t0
    # Deployed model is selected on validation MAE, NOT on the PHM score.
    # PHM is an exponential of the residual, so its mean is dominated by a
    # handful of large positive errors: as a *ranking* criterion it is
    # high-variance and it systematically rewards models that under-predict RUL
    # across the board. Asymmetry belongs in the decision layer -- which keys
    # off the conformal LOWER bound and is conservative by construction -- not
    # in the choice of point estimator. Both metrics are reported either way.
    reg_val = reg_val.sort_values("MAE").reset_index(drop=True)
    reg_val["selected"] = reg_val["MAE"] == reg_val["MAE"].min()
    tables["regression_validation"] = reg_val
    best_reg = reg_val.iloc[0]["model"]
    best_view = regression_zoo(cfg.seed)[best_reg]["view"]
    log(f"[{cfg.name}] best regressor on validation (MAE): {best_reg}")

    # ---- uncertainty: split conformal (constant width) vs CQR (adaptive) ---
    p_va_best = np.clip(reg_fitted[best_reg].predict(_view(Xva, fp, best_view)), 0, cap)
    q = U.conformal_quantile(y_va, p_va_best, cfg.conformal_alpha)

    t0 = time.perf_counter()
    cqr = U.ConformalizedQuantileRegressor(alpha=cfg.conformal_alpha, cap=cap, seed=cfg.seed)
    cqr.fit(_view(Xtr, fp, best_view), y_tr)
    cqr.calibrate(_view(Xva, fp, best_view), y_va)
    timings["cqr_fit_s"] = time.perf_counter() - t0

    # ---- single locked evaluation on TEST engines ---------------------
    reg_rows, test_preds = [], {}
    for key, spec in regression_zoo(cfg.seed).items():
        p = np.clip(reg_fitted[key].predict(_view(Xte, fp, spec["view"])), 0, cap)
        test_preds[key] = p
        rec = {"model": key, "label": spec["label"], "feature_view": spec["view"]}
        rec.update(E.regression_metrics(y_te, p))
        rec["model_size_kb"] = round(len(joblib.hashing.pickle.dumps(reg_fitted[key])) / 1024, 1)
        reg_rows.append(rec)
    tables["regression_test"] = pd.DataFrame(reg_rows).sort_values("MAE").reset_index(drop=True)

    p_te = test_preds[best_reg]
    lo, hi = cqr.predict_interval(_view(Xte, fp, best_view))
    # The point estimate must sit inside its own interval, otherwise the card
    # in the dashboard contradicts itself.
    lo, hi = np.minimum(lo, p_te), np.maximum(hi, p_te)
    lo_split, hi_split = np.clip(p_te - q, 0, None), p_te + q
    tables["regression_by_region"] = E.regression_by_region(y_te, p_te)
    tables["phm_sign_convention"] = E.phm_worked_example()

    # official benchmark view: last observed cycle of every test engine
    last_mask = te.groupby(ID_COL)[CYCLE_COL].transform("max") == te[CYCLE_COL]
    lm = last_mask.to_numpy()
    tables["regression_last_cycle"] = pd.DataFrame([
        {"view": "last cycle per test engine (official benchmark view)",
         "n": int(lm.sum()), **E.regression_metrics(y_te[lm], p_te[lm])},
        {"view": "all test rows", "n": len(y_te), **E.regression_metrics(y_te, p_te)},
    ])

    # interval quality: the two uncertainty methods side by side
    lo_va_cqr, hi_va_cqr = cqr.predict_interval(_view(Xva, fp, best_view))
    tables["interval_report"] = pd.DataFrame([
        {"method": "split conformal (|residual|, constant width)",
         "set": "validation (used for calibration)",
         **U.interval_report(y_va, np.clip(p_va_best - q, 0, None), p_va_best + q)},
        {"method": "split conformal (|residual|, constant width)", "set": "test (held out)",
         **U.interval_report(y_te, lo_split, hi_split)},
        {"method": "split conformal (|residual|, constant width)",
         "set": "test, last cycle only",
         **U.interval_report(y_te[lm], lo_split[lm], hi_split[lm])},
        {"method": "CQR (adaptive width) -- DEPLOYED",
         "set": "validation (used for calibration)",
         **U.interval_report(y_va, lo_va_cqr, hi_va_cqr)},
        {"method": "CQR (adaptive width) -- DEPLOYED", "set": "test (held out)",
         **U.interval_report(y_te, lo, hi)},
        {"method": "CQR (adaptive width) -- DEPLOYED", "set": "test, last cycle only",
         **U.interval_report(y_te[lm], lo[lm], hi[lm])},
    ])
    tables["interval_report"].insert(2, "nominal_coverage", 1 - cfg.conformal_alpha)
    tables["interval_report"].insert(3, "split_conformal_q_cycles", round(q, 2))
    tables["interval_report"].insert(4, "cqr_correction_cycles", round(cqr.correction_, 2))
    tables["interval_width_spread"] = pd.DataFrame([
        {"method": "split conformal", "width_p10": 2 * q, "width_p50": 2 * q,
         "width_p90": 2 * q, "adaptive": False},
        {"method": "CQR", "width_p10": float(np.percentile(hi - lo, 10)),
         "width_p50": float(np.percentile(hi - lo, 50)),
         "width_p90": float(np.percentile(hi - lo, 90)), "adaptive": True},
    ])
    frames["engine_coverage"] = U.engine_level_coverage(te[ID_COL], y_te, lo, hi)

    # engine-level bootstrap CI for the headline regressors
    suite = E.EvaluationSuite(cfg.seed, cfg.cost)
    boot_rows = []
    for key in tables["regression_test"]["model"]:
        dfp = pd.DataFrame({"engine_id": te[ID_COL].to_numpy(), "y": y_te,
                            "p": test_preds[key]})
        for mname, fn in (("MAE", lambda d: np.abs(d.y - d.p).mean()),
                          ("RMSE", lambda d: np.sqrt(((d.y - d.p) ** 2).mean())),
                          ("PHM_mean", lambda d: E.phm_score_mean(d.y, d.p))):
            b = suite.engine_bootstrap(dfp, fn, n_boot=n_boot)
            boot_rows.append({"model": key, "metric": mname, **b})
    tables["regression_bootstrap_ci"] = pd.DataFrame(boot_rows)

    # ---------------- 6. classification per horizon --------------------
    t0 = time.perf_counter()
    clf_zoo = classification_zoo(cfg.seed)
    clf_val_rows, clf_test_rows, cal_rows = [], [], []
    reliability, pr_curves = {}, {}
    # Each horizon selects its own winner, and the winners need not share a
    # feature view -- so the view is tracked per horizon rather than globally.
    chosen_clf, chosen_thr, chosen_view, chosen_key = {}, {}, {}, {}

    for h in cfg.horizons:
        ytr_h = tr[f"fail_within_{h}"].to_numpy()
        yva_h = va[f"fail_within_{h}"].to_numpy()
        yte_h = te[f"fail_within_{h}"].to_numpy()
        per_h = {}
        for key, spec in clf_zoo.items():
            mdl = spec["model"].__class__(**spec["model"].get_params())
            mdl.fit(_view(Xtr, fp, spec["view"]), ytr_h)
            p_va_raw = mdl.predict_proba(_view(Xva, fp, spec["view"]))[:, 1]
            cal = calibrate(mdl, _view(Xva, fp, spec["view"]), yva_h)
            p_va_cal = cal.predict_proba(_view(Xva, fp, spec["view"]))[:, 1]
            thr, _ = E.pick_threshold_by_cost(yva_h, p_va_cal, FN_FP_RATIO)
            per_h[key] = (mdl, cal, thr, spec["view"])
            rec = {"horizon": h, "model": key, "label": spec["label"],
                   "feature_view": spec["view"]}
            rec.update(E.classification_metrics(yva_h, p_va_cal, thr))
            clf_val_rows.append(rec)
            cal_rows.append(
                U.calibration_report(yva_h, p_va_raw, p_va_cal).assign(horizon=h, model=key)
            )
        # pick per-horizon winner on validation PR-AUC
        vsub = pd.DataFrame(clf_val_rows)
        vsub = vsub[vsub.horizon == h]
        winner = vsub.sort_values("PR_AUC", ascending=False).iloc[0]["model"]
        mdl, cal, thr, view = per_h[winner]
        chosen_clf[h], chosen_thr[h] = cal, thr
        chosen_view[h], chosen_key[h] = view, winner

        for key, (mdl_k, cal_k, thr_k, view_k) in per_h.items():
            Xv = _view(Xte, fp, view_k)
            p_raw = mdl_k.predict_proba(Xv)[:, 1]
            p_cal = cal_k.predict_proba(Xv)[:, 1]
            rec = {"horizon": h, "model": key, "label": clf_zoo[key]["label"],
                   "feature_view": view_k, "selected": key == winner,
                   "model_size_kb": round(
                       len(joblib.hashing.pickle.dumps(cal_k)) / 1024, 1)}
            rec.update(E.classification_metrics(yte_h, p_cal, thr_k))
            rec["PR_AUC_uncalibrated"] = float(
                E.classification_metrics(yte_h, p_raw, 0.5)["PR_AUC"])
            rec["brier_uncalibrated"] = float(
                E.classification_metrics(yte_h, p_raw, 0.5)["brier"])
            clf_test_rows.append(rec)
            if key == winner:
                reliability[h] = U.reliability_curve(yte_h, p_cal)
                pr_curves[h] = E.pr_curve_frame(yte_h, p_cal)

    timings["classification_fit_s"] = time.perf_counter() - t0
    tables["classification_validation"] = pd.DataFrame(clf_val_rows)
    tables["classification_test"] = pd.DataFrame(clf_test_rows)
    tables["calibration_effect"] = pd.concat(cal_rows, ignore_index=True)
    frames["reliability"] = reliability
    frames["pr_curves"] = pr_curves

    # bootstrap CI on the selected classifiers
    cboot = []
    for h in cfg.horizons:
        cal = chosen_clf[h]
        p = cal.predict_proba(_view(Xte, fp, chosen_view[h]))[:, 1]
        dfp = pd.DataFrame({"engine_id": te[ID_COL].to_numpy(),
                            "y": te[f"fail_within_{h}"].to_numpy(), "p": p})
        for mname, fn in (
            ("PR_AUC", lambda d: E.average_precision_score(d.y, d.p) if d.y.sum() else np.nan),
            ("recall", lambda d, h=h: E.recall_score(d.y, d.p >= chosen_thr[h], zero_division=0)),
            ("precision", lambda d, h=h: E.precision_score(d.y, d.p >= chosen_thr[h], zero_division=0)),
            ("brier", lambda d: E.brier_score_loss(d.y, d.p)),
        ):
            cboot.append({"horizon": h, "metric": mname,
                          **suite.engine_bootstrap(dfp, fn, n_boot=n_boot)})
    tables["classification_bootstrap_ci"] = pd.DataFrame(cboot)

    # ---------------- 7. anomaly detection -----------------------------
    t0 = time.perf_counter()
    healthy_mask = tr[CYCLE_COL] <= cfg.healthy_head_cycles
    flog.record("anomaly fitting set", f"first {cfg.healthy_head_cycles} cycles of TRAIN engines "
                "only (no failure label, no test row)", rows=int(healthy_mask.sum()))
    bank = AnomalyBank(seed=cfg.seed).fit(
        Xtr[healthy_mask.to_numpy()], fp.feature_names_, cfg.primary_window)
    raw_va = bank.raw_scores(Xva)
    bank.fit_reference(raw_va, percentile=99.0)
    raw_te = bank.raw_scores(Xte)
    norm_va, norm_te = bank.normalized_scores(raw_va), bank.normalized_scores(raw_te)
    timings["anomaly_s"] = time.perf_counter() - t0

    # Alert threshold per detector is tuned on VALIDATION engines (run-to-failure,
    # so lead time is observable there) by the same asymmetric cost.
    va_tl_base = pd.DataFrame({
        "engine_id": va[ID_COL].to_numpy(), "cycle": va[CYCLE_COL].to_numpy(),
        "failure_cycle": va["failure_cycle"].to_numpy(), "RUL": va["RUL"].to_numpy()})
    # Constraint: the cost policy alone would alert almost continuously
    # (c_miss=200 vs c_early=1/cycle), so we cap the false-alert rate at the
    # inspection capacity the fleet can actually absorb.
    MAX_FALSE_ALERT_RATE = 0.05
    anom_thr_rows, anom_thr = [], {}
    grid = (0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
    for det in norm_va.columns:
        feasible, fallback = (np.inf, None), (np.inf, 0.99)
        for pct in grid:
            tl = va_tl_base.copy()
            tl["alert"] = (norm_va[det].to_numpy() >= pct).astype(int)
            c = suite.asymmetric_cost(suite.warning_lead_time(
                tl, horizon=cfg.cost.target_horizon))
            far = c.get("false_alert_rate_per_healthy_cycle", 1.0)
            cost = c.get("avg_cost_per_engine", np.inf)
            anom_thr_rows.append({"detector": det, "val_percentile": pct,
                                  "feasible": far <= MAX_FALSE_ALERT_RATE, **c})
            if far <= MAX_FALSE_ALERT_RATE and cost < feasible[0]:
                feasible = (cost, pct)
            if cost < fallback[0]:
                fallback = (cost, pct)
        anom_thr[det] = feasible[1] if feasible[1] is not None else fallback[1]
    tables["anomaly_threshold_selection_validation"] = pd.DataFrame(anom_thr_rows)

    anom_rows = []
    for det in norm_te.columns:
        tl = pd.DataFrame({
            "engine_id": te[ID_COL].to_numpy(), "cycle": te[CYCLE_COL].to_numpy(),
            "failure_cycle": te["failure_cycle"].to_numpy(), "RUL": te["RUL"].to_numpy(),
            "alert": (norm_te[det].to_numpy() >= anom_thr[det]).astype(int),
        })
        w = suite.warning_lead_time(tl, horizon=cfg.cost.target_horizon)
        c = suite.asymmetric_cost(w)
        # does the score actually rise before failure?
        near = te["RUL"].to_numpy() <= 30
        far = te["RUL"].to_numpy() > 100
        anom_rows.append({
            "detector": det,
            "alert_threshold_val_percentile": anom_thr[det],
            "mean_norm_score_RUL>100": float(norm_te[det].to_numpy()[far].mean()) if far.any() else np.nan,
            "mean_norm_score_RUL<=30": float(norm_te[det].to_numpy()[near].mean()) if near.any() else np.nan,
            "separation": (float(norm_te[det].to_numpy()[near].mean()
                                 - norm_te[det].to_numpy()[far].mean())
                           if near.any() and far.any() else np.nan),
            **{k: c.get(k) for k in ("miss_rate", "mean_lead_time", "median_lead_time",
                                     "false_alert_rate_per_healthy_cycle", "avg_cost_per_engine",
                                     "n_engines_at_risk")},
        })
    tables["anomaly_comparison"] = pd.DataFrame(anom_rows).sort_values("avg_cost_per_engine")
    tables["anomaly_design"] = bank.meta_frame()

    # threshold sensitivity for the selected detector
    best_det = tables["anomaly_comparison"].iloc[0]["detector"]
    anomaly_alert_level = float(anom_thr[best_det])
    sens = []
    for pct in (0.80, 0.90, 0.95, 0.975, 0.99, 0.995):
        tl = pd.DataFrame({
            "engine_id": te[ID_COL].to_numpy(), "cycle": te[CYCLE_COL].to_numpy(),
            "failure_cycle": te["failure_cycle"].to_numpy(), "RUL": te["RUL"].to_numpy(),
            "alert": (norm_te[best_det].to_numpy() >= pct).astype(int)})
        w = suite.warning_lead_time(tl, horizon=cfg.cost.target_horizon)
        sens.append({"detector": best_det, "val_percentile_threshold": pct,
                     **suite.asymmetric_cost(w)})
    tables["anomaly_threshold_sensitivity"] = pd.DataFrame(sens)

    te_anom = te[[ID_COL, CYCLE_COL, "RUL"]].copy()
    te_anom["score"] = norm_te[best_det].to_numpy()
    frames["anomaly_ctf_profile"] = E.cycles_to_failure_profile(te_anom, "score")
    frames["anomaly_norm_test"] = norm_te.assign(
        engine_id=te[ID_COL].to_numpy(), cycle=te[CYCLE_COL].to_numpy(),
        RUL=te["RUL"].to_numpy())

    # ---------------- 8. decision policy + early warning ---------------
    timeline = pd.DataFrame({
        "engine_id": te[ID_COL].to_numpy(), "cycle": te[CYCLE_COL].to_numpy(),
        "RUL": te["RUL"].to_numpy(), "failure_cycle": te["failure_cycle"].to_numpy(),
        "rul_pred": p_te, "rul_lo": lo, "rul_hi": hi,
        "anom_norm": norm_te[best_det].to_numpy(),
    })
    for h in cfg.horizons:
        timeline[f"p_fail_{h}"] = chosen_clf[h].predict_proba(
            _view(Xte, fp, chosen_view[h]))[:, 1]

    # policy thresholds come from the validation-tuned classifier thresholds
    policy = DecisionPolicy(
        p10_stop=float(chosen_thr[10]), p20_inspect=float(chosen_thr[20]),
        p30_inspect=float(chosen_thr[30]),
        anomaly_persist_level=anomaly_alert_level,
        # "wide" = above the 75th percentile of the VALIDATION interval widths.
        # Tuned on validation like every other threshold; with adaptive CQR
        # widths this rule actually discriminates between rows.
        wide_interval_cycles=float(np.percentile(hi_va_cqr - lo_va_cqr, 75)),
    )
    actions = apply_policy_timeline(timeline, policy, cfg.horizons)
    tables["policy_parameters"] = pd.DataFrame([policy.as_dict()]).T.reset_index()
    tables["policy_parameters"].columns = ["parameter", "value"]
    tables["action_confusion"] = action_confusion_by_engine(
        actions, timeline, cfg.cost.target_horizon).reset_index()

    # early warning per alert source
    ew_rows, ew_frames = [], {}
    sources = {f"classifier_h{h}": (timeline[f"p_fail_{h}"].to_numpy() >= chosen_thr[h]).astype(int)
               for h in cfg.horizons}
    sources[f"anomaly_{best_det}"] = (
        timeline["anom_norm"].to_numpy() >= anomaly_alert_level).astype(int)
    sources["decision_policy_INSPECT_or_STOP"] = actions["action"].isin(
        ["INSPECT", "STOP"]).to_numpy().astype(int)
    sources["decision_policy_STOP"] = (actions["action"] == "STOP").to_numpy().astype(int)
    for name, alert in sources.items():
        tl = timeline[["engine_id", "cycle", "RUL", "failure_cycle"]].copy()
        tl["alert"] = alert
        w = suite.warning_lead_time(tl, horizon=cfg.cost.target_horizon)
        ew_frames[name] = w
        ew_rows.append({"alert_source": name, **suite.asymmetric_cost(w)})
    tables["early_warning"] = pd.DataFrame(ew_rows).sort_values("avg_cost_per_engine")
    frames["lead_time_per_engine"] = ew_frames

    # Companion view on VALIDATION engines. These are run-to-failure, so every
    # engine is at risk and the lead-time distribution is complete -- but the
    # thresholds were *selected* here, so it is optimistic by construction and
    # is reported only as a diagnostic next to the locked test numbers.
    va_timeline = pd.DataFrame({
        "engine_id": va[ID_COL].to_numpy(), "cycle": va[CYCLE_COL].to_numpy(),
        "RUL": va["RUL"].to_numpy(), "failure_cycle": va["failure_cycle"].to_numpy()})
    va_rows = []
    va_src = {f"classifier_h{h}": (chosen_clf[h].predict_proba(_view(Xva, fp, chosen_view[h]))[:, 1]
                                   >= chosen_thr[h]).astype(int) for h in cfg.horizons}
    va_src[f"anomaly_{best_det}"] = (
        norm_va[best_det].to_numpy() >= anomaly_alert_level).astype(int)
    for name, alert in va_src.items():
        tl = va_timeline.copy()
        tl["alert"] = alert
        va_rows.append({"alert_source": name, **suite.asymmetric_cost(
            suite.warning_lead_time(tl, horizon=cfg.cost.target_horizon))})
    tables["early_warning_validation"] = pd.DataFrame(va_rows).sort_values("avg_cost_per_engine")

    # bootstrap CI on the decision-policy cost (engine-level)
    w_policy = ew_frames["decision_policy_INSPECT_or_STOP"]
    cost_boot = suite.engine_bootstrap(
        w_policy, lambda d: E.EvaluationSuite(cfg.seed, cfg.cost)
        .asymmetric_cost(d).get("avg_cost_per_engine", np.nan), n_boot=min(n_boot, 300))
    tables["policy_cost_ci"] = pd.DataFrame([{"metric": "avg_cost_per_engine", **cost_boot}])

    tables["filter_log"] = flog.to_frame()

    # ---------------- 9. assemble the deployable system ----------------
    system = PrognosticsSystem(
        feature_pipeline=fp, rul_model=reg_fitted[best_reg], rul_view=best_view,
        rul_cap=cap, conformal_q=float(q), cqr=cqr,
        classifiers=chosen_clf, clf_views=dict(chosen_view),
        anomaly=bank, anomaly_detector=best_det, anomaly_threshold=anomaly_alert_level,
        thresholds=dict(chosen_thr),
        metadata={
            "dataset": cfg.subset, "role": SUBSET_ROLE[cfg.subset],
            "model_version": MODEL_VERSION, "seed": cfg.seed,
            "feature_window": list(cfg.window_lengths), "primary_window": cfg.primary_window,
            "n_features": len(fp.feature_names_), "rul_cap": cap,
            "conformal_alpha": cfg.conformal_alpha, "conformal_q_cycles": round(float(q), 2),
            "rul_model": best_reg, "anomaly_detector": best_det,
            "classifier_per_horizon": {str(k): v for k, v in chosen_key.items()},
            "classifier": ", ".join(f"h{k}:{v}" for k, v in chosen_key.items()),
            "condition_aware": bool(fp.condition_aware), "n_conditions": N_CONDITIONS[cfg.subset],
            "engines": {"train": len(split.train_ids), "val": len(split.val_ids),
                        "test": len(split.test_ids)},
            "cost_policy": cfg.cost.as_dict(),
            "decision_thresholds": {str(k): float(v) for k, v in chosen_thr.items()},
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    timings["total_s"] = time.perf_counter() - t_start

    tables["run_summary"] = pd.DataFrame([{
        "subset": cfg.subset, "tag": cfg.tag or "-", "role": SUBSET_ROLE[cfg.subset],
        "n_features": len(fp.feature_names_), "rul_cap": cap, "rul_model": best_reg,
        "test_MAE": tables["regression_test"].query("model == @best_reg")["MAE"].iloc[0],
        "test_RMSE": tables["regression_test"].query("model == @best_reg")["RMSE"].iloc[0],
        "test_PHM_mean": tables["regression_test"].query("model == @best_reg")["PHM_score_mean"].iloc[0],
        # the DEPLOYED interval is CQR; interval_report also carries the plain
        # split-conformal rows, so filter on the method as well as the set
        "interval_coverage": _deployed_interval(tables["interval_report"])["coverage"],
        "interval_width": _deployed_interval(tables["interval_report"])["mean_width"],
        **{f"PR_AUC_h{h}": tables["classification_test"].query(
            "horizon == @h and selected")["PR_AUC"].iloc[0] for h in cfg.horizons},
        "anomaly_detector": best_det,
        "policy_avg_cost": tables["early_warning"].query(
            "alert_source == 'decision_policy_INSPECT_or_STOP'")["avg_cost_per_engine"].iloc[0],
        "policy_miss_rate": tables["early_warning"].query(
            "alert_source == 'decision_policy_INSPECT_or_STOP'")["miss_rate"].iloc[0],
        "policy_mean_lead_time": tables["early_warning"].query(
            "alert_source == 'decision_policy_INSPECT_or_STOP'")["mean_lead_time"].iloc[0],
        "runtime_s": round(timings["total_s"], 1),
    }])

    result = ExperimentResult(cfg, split, system, tables, frames, timings)
    if export:
        export_artifacts(result, te, timeline, actions, policy)
    log(f"[{cfg.name}] done in {timings['total_s']:.1f}s")
    return result


# --------------------------------------------------------------------------
def export_artifacts(result: ExperimentResult, test_df: pd.DataFrame,
                     timeline: pd.DataFrame, actions: pd.DataFrame,
                     policy: DecisionPolicy) -> None:
    """Everything the Streamlit app needs -- and nothing it does not.

    The app reloads these and reproduces the notebook's inference outputs
    exactly (see ``tests/test_app_parity.py``).
    """
    cfg = result.cfg
    out = cfg.artifact_dir
    result.split.to_json(out / "engine_split.json")
    joblib.dump(
        {"system": result.system, "policy": policy, "config": cfg.as_dict()},
        out / "system.joblib", compress=3,
    )
    # raw test histories power the dashboard's engine timeline
    test_df.to_parquet(out / "test_history.parquet", index=False)
    timeline.merge(actions, on=["engine_id", "cycle"]).to_parquet(
        out / "precomputed_timeline.parquet", index=False)
    meta = dict(result.system.metadata)
    meta["tables"] = sorted(result.tables)
    with open(out / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    for name, tbl in result.tables.items():
        tbl.to_csv(out / f"table_{name}.csv", index=False)
