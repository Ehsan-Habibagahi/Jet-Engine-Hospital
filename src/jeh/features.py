"""Leakage-safe, causal feature engineering (Task 2).

Design rules enforced here:

* every statistic at cycle ``t`` uses only cycles ``<= t`` of the *same* engine
  (trailing ``rolling``/``ewm``/``diff`` -- never ``center=True``);
* nothing is normalised per whole engine using its own final cycles;
* all learned quantities (regime clusters, regime means/stds, the final
  scaler, the near-constant sensor list) are fitted on TRAINING engines only.

``FeaturePipeline.transform_engine`` gives the app the same numbers the
notebook computed, evaluated at a chosen cycle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import CYCLE_COL, ID_COL, OP_COLS, SENSOR_COLS

_EPS = 1e-8


# --------------------------------------------------------------------------
# small causal helpers
# --------------------------------------------------------------------------
def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """OLS slope of the last ``window`` observations, trailing only.

    slope = cov(x, y) / var(x) with x = 0..w-1. Implemented from rolling
    moments so it stays vectorised.
    """
    idx = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    n = s.rolling(window, min_periods=2).count()
    sx = idx.rolling(window, min_periods=2).sum()
    sy = s.rolling(window, min_periods=2).sum()
    sxy = (idx * s).rolling(window, min_periods=2).sum()
    sxx = (idx * idx).rolling(window, min_periods=2).sum()
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom.replace(0.0, np.nan)
    return slope.fillna(0.0)


def _frozen_baseline(s: pd.Series, k: int) -> pd.Series:
    """Expanding mean that stops updating after the first ``k`` cycles."""
    em = s.expanding(min_periods=1).mean()
    if len(em) > k:
        out = em.copy()
        out.iloc[k:] = em.iloc[k - 1]
        return out
    return em


class FeaturePipeline:
    """Fit on training engines, transform any engine history causally."""

    def __init__(
        self,
        window_lengths=(5, 15, 30),
        primary_window: int = 30,
        n_conditions: int = 1,
        condition_aware: bool = True,
        use_op_settings: bool = True,
        use_trend_features: bool = True,
        baseline_cycles: int = 5,
        seed: int = 42,
    ) -> None:
        self.window_lengths = tuple(sorted(window_lengths))
        self.primary_window = primary_window
        self.n_conditions = int(n_conditions)
        self.condition_aware = bool(condition_aware) and self.n_conditions > 1
        self.use_op_settings = use_op_settings
        self.use_trend_features = use_trend_features
        self.baseline_cycles = baseline_cycles
        self.seed = seed

        self.kmeans_: KMeans | None = None
        self.op_mu_: np.ndarray | None = None
        self.op_sd_: np.ndarray | None = None
        self.sensor_cols_: list[str] = []
        self.regime_stats_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.feature_names_: list[str] = []
        self.base_feature_names_: list[str] = []
        self.scaler_mu_: np.ndarray | None = None
        self.scaler_sd_: np.ndarray | None = None
        self.dropped_sensors_: list[str] = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        # 1) near-constant sensors, judged on training engines only.
        sd = train_df[SENSOR_COLS].std()
        nunique = train_df[SENSOR_COLS].nunique()
        keep = [c for c in SENSOR_COLS if sd[c] > 1e-6 and nunique[c] > 2]
        self.dropped_sensors_ = [c for c in SENSOR_COLS if c not in keep]
        self.sensor_cols_ = keep

        # 2) operating-regime discovery on the three settings.
        ops = train_df[OP_COLS].to_numpy(float)
        self.op_mu_, self.op_sd_ = ops.mean(0), ops.std(0) + _EPS
        ops_z = (ops - self.op_mu_) / self.op_sd_
        if self.n_conditions > 1:
            self.kmeans_ = KMeans(self.n_conditions, n_init=10, random_state=self.seed).fit(ops_z)
            labels = self.kmeans_.labels_
        else:
            labels = np.zeros(len(ops_z), dtype=int)

        # 3) regime-wise (or global) sensor location/scale.
        sens = train_df[self.sensor_cols_].to_numpy(float)
        self.regime_stats_ = {}
        if self.condition_aware:
            for r in range(self.n_conditions):
                m = labels == r
                self.regime_stats_[r] = (sens[m].mean(0), sens[m].std(0) + _EPS)
        else:
            self.regime_stats_ = {r: (sens.mean(0), sens.std(0) + _EPS)
                                  for r in range(max(self.n_conditions, 1))}

        # 4) assemble on training rows to learn the final scaler.
        X = self._build(train_df)
        self.feature_names_ = list(X.columns)
        arr = X.to_numpy(float)
        self.scaler_mu_ = np.nanmean(arr, axis=0)
        self.scaler_sd_ = np.nanstd(arr, axis=0) + _EPS
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        X = self._build(df)
        X = X.reindex(columns=self.feature_names_)
        if scale:
            arr = (X.to_numpy(float) - self.scaler_mu_) / self.scaler_sd_
            X = pd.DataFrame(arr, index=X.index, columns=self.feature_names_)
        return X.fillna(0.0)

    def transform_engine(self, engine_history: pd.DataFrame, at_cycle: int | None = None,
                         scale: bool = True) -> pd.DataFrame:
        """Features for one engine. If ``at_cycle`` is given, the history is
        truncated first -- which is exactly what causality means here."""
        h = engine_history.sort_values(CYCLE_COL)
        if at_cycle is not None:
            h = h[h[CYCLE_COL] <= at_cycle]
        X = self.transform(h, scale=scale)
        return X.iloc[[-1]] if at_cycle is not None else X

    # ------------------------------------------------------------------
    # feature construction
    # ------------------------------------------------------------------
    def assign_regime(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans_ is None:
            return np.zeros(len(df), dtype=int)
        ops_z = (df[OP_COLS].to_numpy(float) - self.op_mu_) / self.op_sd_
        return self.kmeans_.predict(ops_z)

    def regime_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sensor values expressed as z-scores *within their operating regime*.
        For single-condition subsets this collapses to a global z-score."""
        regime = self.assign_regime(df)
        sens = df[self.sensor_cols_].to_numpy(float)
        mu = np.empty_like(sens)
        sdv = np.empty_like(sens)
        for r, (m, s) in self.regime_stats_.items():
            mask = regime == r
            if mask.any():
                mu[mask], sdv[mask] = m, s
        z = (sens - mu) / sdv
        out = pd.DataFrame(z, index=df.index, columns=[f"{c}_res" for c in self.sensor_cols_])
        out["_regime"] = regime
        return out

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([ID_COL, CYCLE_COL])
        res = self.regime_residuals(df)
        regime = res.pop("_regime")
        res[ID_COL] = df[ID_COL].to_numpy()

        blocks: list[pd.DataFrame] = []
        cols = list(res.columns[:-1])
        g = res.groupby(ID_COL, sort=False)

        # --- current-cycle block -------------------------------------------
        cur = res[cols].copy()
        cur.columns = [f"{c}" for c in cols]
        blocks.append(cur)

        # --- trailing window statistics ------------------------------------
        for w in self.window_lengths:
            rm = g[cols].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            rm.columns = [f"{c}_rmean{w}" for c in cols]
            rs = g[cols].transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
            rs.columns = [f"{c}_rstd{w}" for c in cols]
            blocks += [rm, rs]

        w = self.primary_window
        rmin = g[cols].transform(lambda s, w=w: s.rolling(w, min_periods=1).min())
        rmin.columns = [f"{c}_rmin{w}" for c in cols]
        rmax = g[cols].transform(lambda s, w=w: s.rolling(w, min_periods=1).max())
        rmax.columns = [f"{c}_rmax{w}" for c in cols]
        blocks += [rmin, rmax]

        # --- trend / dynamics block (ablatable) ----------------------------
        if self.use_trend_features:
            for w in (self.window_lengths[len(self.window_lengths) // 2], self.primary_window):
                sl = g[cols].transform(lambda s, w=w: _rolling_slope(s, w))
                sl.columns = [f"{c}_slope{w}" for c in cols]
                blocks.append(sl)
            ew = g[cols].transform(lambda s: s.ewm(span=self.primary_window, adjust=False).mean())
            ew.columns = [f"{c}_ewm" for c in cols]
            d1 = g[cols].transform(lambda s: s.diff())
            d1.columns = [f"{c}_diff1" for c in cols]
            # drift away from the engine's own first cycles. Causal: before
            # cycle k the baseline is the expanding mean so far, from cycle k
            # onwards it freezes at the mean of the first k cycles.
            base = g[cols].transform(lambda s, k=self.baseline_cycles: _frozen_baseline(s, k))
            dr = res[cols].to_numpy() - base.to_numpy()
            drift = pd.DataFrame(dr, index=res.index, columns=[f"{c}_drift" for c in cols])
            blocks += [ew, d1, drift]

        # --- context block --------------------------------------------------
        ctx = pd.DataFrame(index=df.index)
        ctx["cycle"] = df[CYCLE_COL].to_numpy(float)
        ctx["log_cycle"] = np.log1p(ctx["cycle"])
        if self.use_op_settings:
            for c in OP_COLS:
                ctx[c] = df[c].to_numpy(float)
            if self.n_conditions > 1:
                for r in range(self.n_conditions):
                    ctx[f"regime_{r}"] = (regime.to_numpy() == r).astype(float)
        blocks.append(ctx)

        X = pd.concat(blocks, axis=1)
        return X.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # views used by the ablation "current cycle vs window features"
    # ------------------------------------------------------------------
    def current_cycle_columns(self) -> list[str]:
        """Feature subset that uses only cycle t (no history at all)."""
        keep = [c for c in self.feature_names_ if c.endswith("_res")]
        keep += [c for c in self.feature_names_
                 if c in ("cycle", "log_cycle") or c.startswith("operational_setting")
                 or c.startswith("regime_")]
        return keep

    # ------------------------------------------------------------------
    def causality_check(self, df: pd.DataFrame, engine_id: int, at_cycle: int,
                        tol: float = 1e-8) -> bool:
        """Assert that rows after ``at_cycle`` cannot change the features at
        ``at_cycle`` (the CAUSALITY TEST call-out in the brief)."""
        hist = df[df[ID_COL] == engine_id].sort_values(CYCLE_COL)
        full = self.transform(hist)
        row_full = full[hist[CYCLE_COL].to_numpy() == at_cycle].to_numpy()
        row_trunc = self.transform_engine(hist, at_cycle=at_cycle).to_numpy()
        return bool(np.nanmax(np.abs(row_full - row_trunc)) <= tol)
