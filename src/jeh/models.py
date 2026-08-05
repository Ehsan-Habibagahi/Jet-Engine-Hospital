"""Model families: RUL regression, failure-horizon classification, unsupervised
anomaly detection, and the ``PrognosticsSystem`` facade used by the app.

Nothing in this module ever sees a test engine during ``fit``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import OneClassSVM

try:  # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator

    def _freeze(est):
        return FrozenEstimator(est)

    _PREFIT_CV = None
except ImportError:  # pragma: no cover - older sklearn
    FrozenEstimator = None

    def _freeze(est):
        return est

    _PREFIT_CV = "prefit"

from sklearn.calibration import CalibratedClassifierCV

from .config import SEED


# ==========================================================================
# Task 3 -- RUL regression zoo
# ==========================================================================
def regression_zoo(seed: int = SEED) -> dict[str, object]:
    """Required baselines. ``feature_view`` says whether the model sees only
    cycle-t values or the full causal window representation."""
    return {
        "ridge_current": dict(
            model=Ridge(alpha=10.0, random_state=seed),
            view="current",
            label="Ridge (current cycle)",
        ),
        "poly2_ridge_current": dict(
            model=Pipeline(
                [
                    # PCA first keeps the degree-2 expansion tractable and is
                    # fitted inside the pipeline on training rows only.
                    ("pca", PCA(n_components=12, random_state=seed)),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("ridge", Ridge(alpha=10.0, random_state=seed)),
                ]
            ),
            view="current",
            label="Polynomial(2) Ridge (current cycle)",
        ),
        "ridge_window": dict(
            model=Ridge(alpha=10.0, random_state=seed),
            view="window",
            label="Ridge (window features)",
        ),
        "rf_current": dict(
            model=RandomForestRegressor(
                n_estimators=250, min_samples_leaf=5, max_features="sqrt",
                n_jobs=-1, random_state=seed,
            ),
            view="current",
            label="Random Forest (current cycle)",
        ),
        "rf_window": dict(
            model=RandomForestRegressor(
                n_estimators=250, min_samples_leaf=5, max_features="sqrt",
                n_jobs=-1, random_state=seed,
            ),
            view="window",
            label="Random Forest (window features)",
        ),
        "gbr_window": dict(
            model=GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.06, max_depth=3, subsample=0.8,
                max_features="sqrt", random_state=seed,
            ),
            view="window",
            label="Gradient Boosting (window features)",
        ),
        "hgb_window": dict(
            model=HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False, random_state=seed,
            ),
            view="window",
            label="Hist Gradient Boosting (window features)",
        ),
    }


# ==========================================================================
# Task 4 -- failure-horizon classification
# ==========================================================================
def classification_zoo(seed: int = SEED) -> dict[str, object]:
    return {
        "logreg_current": dict(
            model=LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            view="current",
            label="Logistic Regression (current cycle)",
        ),
        "logreg_window": dict(
            model=LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"),
            view="window",
            label="Logistic Regression (window features)",
        ),
        "hgb_window": dict(
            model=HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=seed,
            ),
            view="window",
            label="Hist Gradient Boosting (window features)",
        ),
    }


def calibrate(estimator, X_val, y_val, method: str = "isotonic"):
    """Post-hoc probability calibration fitted on VALIDATION engines only."""
    if len(np.unique(y_val)) < 2:
        return estimator
    if _PREFIT_CV is None:
        cal = CalibratedClassifierCV(_freeze(estimator), method=method)
    else:  # pragma: no cover
        cal = CalibratedClassifierCV(estimator, method=method, cv=_PREFIT_CV)
    cal.fit(X_val, y_val)
    return cal


# ==========================================================================
# Task 5 -- unsupervised anomaly detection
# ==========================================================================
@dataclass
class AnomalyResult:
    name: str
    fit_seconds: float
    score_seconds_per_1k: float
    n_fit_rows: int
    design: str


class AnomalyBank:
    """Four detectors sharing one leakage-safe feature space.

    Fitting set = the *healthy region* of TRAINING engines (their first
    ``healthy_head_cycles`` cycles). No failure label and no test row is used.
    Every score is oriented so that **larger = more abnormal**, then converted
    to a validation percentile so the four are comparable.
    """

    def __init__(self, seed: int = SEED, n_components: int = 10,
                 max_fit_rows: int = 6000, contamination: float = 0.02,
                 lof_neighbors: int = 35, ocsvm_nu: float = 0.05):
        self.seed = seed
        self.n_components = n_components
        self.max_fit_rows = max_fit_rows
        self.contamination = contamination
        self.lof_neighbors = lof_neighbors
        self.ocsvm_nu = ocsvm_nu

        self.cols_: list[str] = []
        self.proj_: PCA | None = None          # shared projection for IF/LOF/OCSVM
        self.recon_pca_: PCA | None = None     # PCA reconstruction detector
        self.recon_k_: int = 0
        self.models_: dict[str, object] = {}
        self.meta_: dict[str, AnomalyResult] = {}
        self.val_scores_: dict[str, np.ndarray] = {}
        self.thresholds_: dict[str, float] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def anomaly_columns(feature_names: list[str], primary_window: int = 30) -> list[str]:
        """Current residuals + their smoothed version: enough signal, low noise,
        and identical across detectors so the comparison is fair."""
        return [c for c in feature_names
                if c.endswith("_res") or c.endswith(f"_res_rmean{primary_window}")]

    def fit(self, X_healthy: pd.DataFrame, feature_names: list[str],
            primary_window: int = 30) -> "AnomalyBank":
        self.cols_ = self.anomaly_columns(feature_names, primary_window)
        A = X_healthy[self.cols_].to_numpy(float)
        rng = np.random.default_rng(self.seed)
        if len(A) > self.max_fit_rows:
            idx = rng.choice(len(A), self.max_fit_rows, replace=False)
            A_fit = A[idx]
        else:
            A_fit = A

        # --- PCA reconstruction error ---------------------------------
        # Component count chosen on the healthy TRAINING rows by the 95%
        # explained-variance rule (validated later against lead time).
        full = PCA(n_components=min(len(self.cols_), A_fit.shape[0] - 1),
                   random_state=self.seed).fit(A_fit)
        cum = np.cumsum(full.explained_variance_ratio_)
        self.recon_k_ = int(np.searchsorted(cum, 0.95) + 1)
        t0 = time.perf_counter()
        self.recon_pca_ = PCA(n_components=self.recon_k_, random_state=self.seed).fit(A_fit)
        t_recon = time.perf_counter() - t0

        # --- shared projection for the three "boundary" detectors -------
        k = min(self.n_components, A_fit.shape[1], A_fit.shape[0] - 1)
        self.proj_ = PCA(n_components=k, whiten=True, random_state=self.seed).fit(A_fit)
        Z = self.proj_.transform(A_fit)

        specs = {
            "isolation_forest": (
                IsolationForest(n_estimators=300, contamination=self.contamination,
                                random_state=self.seed, n_jobs=-1),
                f"contamination={self.contamination}, fit on healthy head rows",
            ),
            "lof": (
                LocalOutlierFactor(n_neighbors=self.lof_neighbors, novelty=True),
                f"novelty=True, n_neighbors={self.lof_neighbors}",
            ),
            "ocsvm": (
                OneClassSVM(kernel="rbf", nu=self.ocsvm_nu, gamma="scale"),
                f"rbf kernel, nu={self.ocsvm_nu}, gamma='scale', whitened PCA input",
            ),
        }
        for name, (mdl, design) in specs.items():
            t0 = time.perf_counter()
            mdl.fit(Z)
            dt = time.perf_counter() - t0
            self.models_[name] = mdl
            self.meta_[name] = AnomalyResult(name, dt, 0.0, len(A_fit), design)

        self.models_["pca_recon"] = self.recon_pca_
        self.meta_["pca_recon"] = AnomalyResult(
            "pca_recon", t_recon, 0.0, len(A_fit),
            f"k={self.recon_k_} components (95% healthy variance)",
        )
        return self

    # ------------------------------------------------------------------
    def raw_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Larger = more abnormal, for every detector."""
        A = X[self.cols_].to_numpy(float)
        Z = self.proj_.transform(A)
        out = {}
        for name in ("isolation_forest", "lof", "ocsvm"):
            t0 = time.perf_counter()
            # decision_function: positive = inlier -> negate.
            out[name] = -self.models_[name].decision_function(Z)
            self.meta_[name].score_seconds_per_1k = (
                (time.perf_counter() - t0) / max(len(A), 1) * 1000
            )
        t0 = time.perf_counter()
        rec = self.recon_pca_.inverse_transform(self.recon_pca_.transform(A))
        out["pca_recon"] = np.sqrt(((A - rec) ** 2).mean(axis=1))
        self.meta_["pca_recon"].score_seconds_per_1k = (
            (time.perf_counter() - t0) / max(len(A), 1) * 1000
        )
        return pd.DataFrame(out, index=X.index)

    def fit_reference(self, val_raw: pd.DataFrame, percentile: float = 99.0) -> None:
        """Store the VALIDATION score distribution: it defines the percentile
        normalisation and the alert threshold. Never fitted on test rows."""
        for name in val_raw.columns:
            s = np.sort(val_raw[name].to_numpy(float))
            self.val_scores_[name] = s
            self.thresholds_[name] = float(np.percentile(s, percentile))

    def normalized_scores(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map raw scores to their validation percentile in [0, 1]. This is a
        rank, explicitly *not* a probability of failure."""
        out = {}
        for name in raw.columns:
            ref = self.val_scores_[name]
            out[name] = np.searchsorted(ref, raw[name].to_numpy(float), side="right") / len(ref)
        return pd.DataFrame(out, index=raw.index)

    def meta_frame(self) -> pd.DataFrame:
        rows = []
        for name, m in self.meta_.items():
            rows.append(
                {
                    "detector": name,
                    "design_choice": m.design,
                    "fit_rows": m.n_fit_rows,
                    "fit_seconds": round(m.fit_seconds, 3),
                    "score_ms_per_1k_rows": round(m.score_seconds_per_1k * 1000, 3),
                    "val_threshold_p99": round(self.thresholds_.get(name, float("nan")), 4),
                }
            )
        return pd.DataFrame(rows)


# ==========================================================================
# Facade used by the notebook and the app (Section 10 interfaces)
# ==========================================================================
@dataclass
class PrognosticsSystem:
    """One locked pipeline: features -> RUL + interval, 3 calibrated risks,
    anomaly score, decision."""

    feature_pipeline: object
    rul_model: object
    rul_view: str
    rul_cap: int
    conformal_q: float                                # split-conformal fallback width
    cqr: object = None                                # ConformalizedQuantileRegressor
    classifiers: dict = field(default_factory=dict)   # horizon -> fitted (calibrated) model
    #: horizon -> feature view. Each horizon selects its own winner on
    #: validation PR-AUC, and those winners need not share a feature view, so
    #: this must be per-horizon rather than one global setting.
    clf_views: dict = field(default_factory=dict)
    anomaly: AnomalyBank | None = None
    anomaly_detector: str = "pca_recon"
    thresholds: dict = field(default_factory=dict)    # horizon -> decision threshold
    anomaly_threshold: float = 0.99
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------
    def _view(self, X: pd.DataFrame, view: str) -> pd.DataFrame:
        if view == "current":
            return X[self.feature_pipeline.current_cycle_columns()]
        return X

    def predict_rul(self, features: pd.DataFrame, interval: bool = True):
        Xv = self._view(features, self.rul_view)
        point = np.clip(self.rul_model.predict(Xv), 0.0, self.rul_cap)
        if not interval:
            return point
        if self.cqr is not None:
            lo, hi = self.cqr.predict_interval(Xv)
            # keep the card self-consistent: the point must lie in its interval
            return point, np.minimum(lo, point), np.maximum(hi, point)
        return point, np.clip(point - self.conformal_q, 0.0, None), point + self.conformal_q

    def failure_risk(self, features: pd.DataFrame, horizons=(10, 20, 30)) -> pd.DataFrame:
        out = {}
        for h in horizons:
            Xv = self._view(features, self.clf_views.get(h, "window"))
            out[f"p_fail_{h}"] = self.classifiers[h].predict_proba(Xv)[:, 1]
        return pd.DataFrame(out, index=features.index)

    def anomaly_score(self, features: pd.DataFrame) -> pd.Series:
        raw = self.anomaly.raw_scores(features)
        norm = self.anomaly.normalized_scores(raw)
        return norm[self.anomaly_detector]
