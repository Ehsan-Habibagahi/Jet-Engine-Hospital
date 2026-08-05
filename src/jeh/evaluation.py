"""Metrics, engine-level bootstrap, early-warning lead time and asymmetric cost.

Two conventions used throughout:

* **Engine-level bootstrap.** Rows inside one engine are strongly correlated,
  so resampling rows would give absurdly tight intervals. We resample *engine
  ids* with replacement (Section 6.2).
* **PHM/NASA score sign.** ``d = RUL_pred - RUL_true``. ``d > 0`` means the
  model claims more life than the engine has -- a late warning -- and is
  penalised with the faster ``exp(d/10)`` arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .config import COST_POLICY, PERSIST_M, PERSIST_N, CostPolicy


# ==========================================================================
# Regression metrics
# ==========================================================================
def phm_score(y_true, y_pred, late_scale: float = 10.0, early_scale: float = 13.0) -> float:
    """NASA/PHM'08 asymmetric exponential score (lower is better).

    d = pred - true. Overestimating RUL (d > 0, late warning) is penalised
    harder because the maintenance action arrives after it was needed.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s = np.where(d > 0, np.exp(d / late_scale) - 1.0, np.exp(-d / early_scale) - 1.0)
    return float(s.sum())


def phm_score_mean(y_true, y_pred, **kw) -> float:
    n = max(len(np.asarray(y_true)), 1)
    return phm_score(y_true, y_pred, **kw) / n


def phm_worked_example() -> pd.DataFrame:
    """Sign-convention sanity check demanded by the brief."""
    rows = []
    for true, pred, note in [
        (30.0, 40.0, "overestimate by 10 (LATE warning)"),
        (30.0, 20.0, "underestimate by 10 (EARLY warning)"),
        (30.0, 30.0, "exact"),
    ]:
        rows.append({"RUL_true": true, "RUL_pred": pred, "d = pred - true": pred - true,
                     "phm_score": round(phm_score([true], [pred]), 4), "interpretation": note})
    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "PHM_score_total": phm_score(y_true, y_pred),
        "PHM_score_mean": phm_score_mean(y_true, y_pred),
        "bias_mean_error": float(np.mean(y_pred - y_true)),
        "late_fraction": float(np.mean(y_pred > y_true)),
    }


LIFE_REGIONS = [("near-failure", 0, 30), ("mid-life", 30, 80), ("early-life", 80, np.inf)]


def regression_by_region(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rows = []
    for name, lo, hi in LIFE_REGIONS:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() == 0:
            continue
        rec = {"region": name, "true_RUL_range": f"[{lo}, {hi})", "n_rows": int(m.sum())}
        rec.update(regression_metrics(y_true[m], y_pred[m]))
        rows.append(rec)
    return pd.DataFrame(rows)


# ==========================================================================
# Classification metrics
# ==========================================================================
def classification_metrics(y_true, p, threshold: float) -> dict:
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "PR_AUC": float(average_precision_score(y_true, p)) if y_true.sum() else float("nan"),
        "ROC_AUC": float(roc_auc_score(y_true, p)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "F1": float(f1_score(y_true, yhat, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "positive_rate": float(y_true.mean()),
    }


def pick_threshold_by_cost(y_true, p, fn_fp_ratio: float = 20.0) -> tuple[float, float]:
    """Threshold minimising ``fn_fp_ratio * FN + FP`` on VALIDATION rows.

    0.50 is not assumed optimal: a missed near-failure row costs far more than
    an extra inspection.
    """
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197), np.quantile(p, np.linspace(0, 1, 101))]))
    best_t, best_c = 0.5, np.inf
    for t in grid:
        yhat = p >= t
        fn = int(((y_true == 1) & (~yhat)).sum())
        fp = int(((y_true == 0) & (yhat)).sum())
        c = fn_fp_ratio * fn + fp
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t, best_c


def pr_curve_frame(y_true, p) -> pd.DataFrame:
    pr, rc, th = precision_recall_curve(y_true, p)
    return pd.DataFrame({"precision": pr[:-1], "recall": rc[:-1], "threshold": th})


# ==========================================================================
# Engine-level bootstrap
# ==========================================================================
class EvaluationSuite:
    def __init__(self, seed: int = 42, cost: CostPolicy = COST_POLICY):
        self.seed = seed
        self.cost = cost

    def engine_bootstrap(self, predictions: pd.DataFrame, metric_fn, n_boot: int = 1000,
                         engine_col: str = "engine_id") -> dict:
        """95% CI by resampling ENGINES with replacement.

        ``metric_fn(df) -> float`` is evaluated on each resampled fleet.
        """
        rng = np.random.default_rng(self.seed)
        engines = predictions[engine_col].unique()
        groups = {e: g for e, g in predictions.groupby(engine_col)}
        point = float(metric_fn(predictions))
        vals = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(engines, size=len(engines), replace=True)
            samp = pd.concat([groups[e] for e in pick], ignore_index=True)
            vals[b] = metric_fn(samp)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
                "boot_std": float(vals.std()), "n_boot": n_boot}

    # ------------------------------------------------------------------
    # Task 6 -- early-warning analysis
    # ------------------------------------------------------------------
    @staticmethod
    def first_persistent_alert(cycles: np.ndarray, alerts: np.ndarray,
                               m: int = PERSIST_M, n: int = PERSIST_N) -> float | None:
        """First cycle at which >= m of the last n cycles are alerts.

        Documented persistence rule; single-cycle spikes never trigger.
        """
        a = np.asarray(alerts, int)
        run = pd.Series(a).rolling(n, min_periods=1).sum().to_numpy()
        hit = np.flatnonzero(run >= m)
        return float(cycles[hit[0]]) if len(hit) else None

    def warning_lead_time(self, timeline: pd.DataFrame, persistence_rule=(PERSIST_M, PERSIST_N),
                          horizon: int | None = None) -> pd.DataFrame:
        """``timeline`` needs columns engine_id, cycle, alert, failure_cycle, RUL.

        Eligibility: an engine can only "miss" a warning if its observed window
        actually reaches the action region (min observed RUL <= horizon).
        Test engines truncated 150 cycles before failure are reported
        separately as *not at risk* instead of being scored as misses.
        """
        m, n = persistence_rule
        h = horizon if horizon is not None else self.cost.target_horizon
        rows = []
        for eid, g in timeline.groupby("engine_id"):
            g = g.sort_values("cycle")
            T = float(g["failure_cycle"].iloc[0])
            tau = self.first_persistent_alert(g["cycle"].to_numpy(), g["alert"].to_numpy(), m, n)
            at_risk = bool(g["RUL"].min() <= h)
            lead = (T - tau) if tau is not None else np.nan
            rows.append(
                {
                    "engine_id": eid,
                    "failure_cycle_T": T,
                    "first_persistent_alert_tau": tau,
                    "lead_time_L": lead,
                    "alerted": tau is not None,
                    "at_risk": at_risk,
                    "observed_min_RUL": float(g["RUL"].min()),
                    "n_alert_cycles": int(g["alert"].sum()),
                    "n_cycles": len(g),
                    "false_alert_cycles": int(((g["alert"] == 1) & (g["RUL"] > h)).sum()),
                    "healthy_cycles": int((g["RUL"] > h).sum()),
                }
            )
        df = pd.DataFrame(rows)
        df["late_delay"] = np.where(df.alerted, np.maximum(0.0, h - df.lead_time_L), np.nan)
        df["early_burden"] = np.where(df.alerted, np.maximum(0.0, df.lead_time_L - h), np.nan)
        df["missed"] = (~df.alerted) & df.at_risk
        return df

    def asymmetric_cost(self, warnings: pd.DataFrame, cost_policy: CostPolicy | None = None) -> dict:
        """C = c_miss*I(missed) + c_late*late_delay + c_early*early_burden,
        reported *with* its components so misses cannot hide in the average."""
        pol = cost_policy or self.cost
        w = warnings[warnings.at_risk].copy()
        if w.empty:
            return {"n_engines_at_risk": 0}
        miss = w["missed"].to_numpy(float)
        late = np.nan_to_num(w["late_delay"].to_numpy(float))
        early = np.nan_to_num(w["early_burden"].to_numpy(float))
        per_engine = pol.c_miss * miss + pol.c_late * late + pol.c_early * early
        far_denom = warnings["healthy_cycles"].sum()
        # An all-miss configuration has no observed lead time at all; report NaN
        # explicitly instead of letting numpy warn about an empty slice.
        leads = w["lead_time_L"].dropna().to_numpy()
        return {
            "n_engines_at_risk": int(len(w)),
            "miss_rate": float(miss.mean()),
            "mean_lead_time": float(leads.mean()) if len(leads) else float("nan"),
            "median_lead_time": float(np.median(leads)) if len(leads) else float("nan"),
            "mean_late_delay": float(late.mean()),
            "mean_early_burden": float(early.mean()),
            "cost_component_miss": float(pol.c_miss * miss.mean()),
            "cost_component_late": float(pol.c_late * late.mean()),
            "cost_component_early": float(pol.c_early * early.mean()),
            "avg_cost_per_engine": float(per_engine.mean()),
            "false_alert_rate_per_healthy_cycle": float(
                warnings["false_alert_cycles"].sum() / max(far_denom, 1)
            ),
        }


def cycles_to_failure_profile(df: pd.DataFrame, score_col: str, max_ctf: int = 200) -> pd.DataFrame:
    """Mean +/- IQR of a score aligned by cycles-to-failure (Section 6.3
    diagnostic for anomaly detection)."""
    d = df[df["RUL"] <= max_ctf].copy()
    d["ctf"] = d["RUL"].round().astype(int)
    g = d.groupby("ctf")[score_col]
    out = g.agg(mean="mean", median="median", q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75), n="size").reset_index()
    return out.sort_values("ctf")
