"""Task 7 -- uncertainty: split conformal RUL intervals and calibration
diagnostics for the horizon classifiers.

All calibration quantities are estimated on VALIDATION engines. Test engines
only ever supply the *empirical coverage* that we report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


# --------------------------------------------------------------------------
# Split conformal prediction for RUL
# --------------------------------------------------------------------------
def conformal_quantile(y_val: np.ndarray, pred_val: np.ndarray, alpha: float = 0.10) -> float:
    """Absolute-residual conformal quantile with the finite-sample correction.

    Guarantees marginal coverage >= 1 - alpha on exchangeable data. Engine
    trajectories are autocorrelated, so we report empirical coverage on
    held-out engines rather than trusting the guarantee blindly.
    """
    res = np.abs(np.asarray(y_val, float) - np.asarray(pred_val, float))
    n = len(res)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(res, level, method="higher"))


class ConformalizedQuantileRegressor:
    """CQR (Romano et al., 2019): two quantile regressors + a conformal
    correction fitted on VALIDATION engines.

    Split conformal with absolute residuals gives every row the *same* width,
    which makes an "is this interval wide?" decision rule vacuous. CQR keeps
    the finite-sample coverage guarantee while letting the width follow the
    local difficulty of the trajectory -- which is what the dashboard's
    uncertainty rule needs in order to withhold STOP.
    """

    def __init__(self, alpha: float = 0.10, cap: float = 125.0, seed: int = 42,
                 n_estimators: int = 200, learning_rate: float = 0.06,
                 max_depth: int = 3):
        from sklearn.ensemble import GradientBoostingRegressor

        self.alpha = alpha
        self.cap = cap
        kw = dict(loss="quantile", n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, subsample=0.8, max_features="sqrt", random_state=seed)
        self.lo_model = GradientBoostingRegressor(alpha=alpha / 2, **kw)
        self.hi_model = GradientBoostingRegressor(alpha=1 - alpha / 2, **kw)
        self.correction_: float = 0.0

    def fit(self, X_train, y_train) -> "ConformalizedQuantileRegressor":
        self.lo_model.fit(X_train, y_train)
        self.hi_model.fit(X_train, y_train)
        return self

    def calibrate(self, X_val, y_val) -> "ConformalizedQuantileRegressor":
        lo = self.lo_model.predict(X_val)
        hi = self.hi_model.predict(X_val)
        # conformity score: how far outside the raw quantile band the truth fell
        scores = np.maximum(lo - np.asarray(y_val, float), np.asarray(y_val, float) - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.correction_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_interval(self, X):
        lo = np.clip(self.lo_model.predict(X) - self.correction_, 0.0, self.cap)
        hi = np.clip(self.hi_model.predict(X) + self.correction_, 0.0, self.cap)
        return lo, np.maximum(hi, lo)


def interval_report(y_true, lo, hi) -> dict:
    y_true = np.asarray(y_true, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    covered = (y_true >= lo) & (y_true <= hi)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float((hi - lo).mean()),
        "median_width": float(np.median(hi - lo)),
        "lower_violation_rate": float((y_true < lo).mean()),   # optimistic RUL
        "upper_violation_rate": float((y_true > hi).mean()),
    }


def engine_level_coverage(engine_ids, y_true, lo, hi) -> pd.DataFrame:
    df = pd.DataFrame(
        {"engine_id": np.asarray(engine_ids), "y": np.asarray(y_true, float),
         "lo": np.asarray(lo, float), "hi": np.asarray(hi, float)}
    )
    df["covered"] = (df.y >= df.lo) & (df.y <= df.hi)
    return df.groupby("engine_id").agg(
        coverage=("covered", "mean"), width=("hi", "mean"), n=("y", "size")
    ).reset_index()


# --------------------------------------------------------------------------
# Probability calibration diagnostics
# --------------------------------------------------------------------------
def reliability_curve(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability curve (robust to the heavy mass near 0)."""
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()),
             "observed_rate": float(y_true[m].mean())}
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    rc = reliability_curve(y_true, p, n_bins)
    if rc.empty:
        return float("nan")
    w = rc["n"] / rc["n"].sum()
    return float((w * (rc["mean_pred"] - rc["observed_rate"]).abs()).sum())


def calibration_report(y_true, p_before, p_after) -> pd.DataFrame:
    rows = []
    for tag, p in (("uncalibrated", p_before), ("calibrated", p_after)):
        rows.append(
            {
                "variant": tag,
                "brier": float(brier_score_loss(y_true, p)),
                "ECE": expected_calibration_error(y_true, p),
                "mean_pred": float(np.mean(p)),
                "base_rate": float(np.mean(y_true)),
            }
        )
    return pd.DataFrame(rows)
