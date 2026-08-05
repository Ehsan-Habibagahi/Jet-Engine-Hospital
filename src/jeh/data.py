"""Loading, label construction, engine-level splitting and the data audit.

Split policy (Section 6.1 of the brief):

* The *official* test engines (``test_FD00x.txt`` + ``RUL_FD00x.txt``) are the
  locked TEST set. They are truncated before failure, exactly the situation the
  deployed system faces, and they are touched once for final reporting.
* The official training engines are partitioned **by engine_id** into TRAIN and
  VALIDATION. Validation engines are run-to-failure, which is what threshold
  tuning, conformal calibration and lead-time rules need.

Every engine id therefore belongs to exactly one split, and the split is
decided before any preprocessing is fitted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    CYCLE_COL,
    DATA_DIR,
    HORIZONS,
    ID_COL,
    OP_COLS,
    RAW_COLS,
    SENSOR_COLS,
    SEED,
    VAL_ENGINE_FRACTION,
)


# --------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------
def _read_table(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # The NASA files carry two trailing empty columns in some releases.
    df = df.iloc[:, : len(RAW_COLS)]
    df.columns = RAW_COLS
    df[ID_COL] = df[ID_COL].astype(int)
    df[CYCLE_COL] = df[CYCLE_COL].astype(int)
    return df


def load_subset(subset: str, data_dir=DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return ``(train_df, test_df, test_final_rul)`` for one C-MAPSS subset."""
    train = _read_table(data_dir / f"train_{subset}.txt")
    test = _read_table(data_dir / f"test_{subset}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()

    n_test_engines = test[ID_COL].nunique()
    assert len(rul) == n_test_engines, (
        f"{subset}: RUL file has {len(rul)} rows but test set has {n_test_engines} engines"
    )
    return train, test, rul


# --------------------------------------------------------------------------
# Label construction (Section 3.3)
# --------------------------------------------------------------------------
def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL(i,t) = T(i) - t with T(i) = max cycle. Valid for run-to-failure
    trajectories only (official train engines)."""
    out = df.copy()
    failure_cycle = out.groupby(ID_COL)[CYCLE_COL].transform("max")
    out["failure_cycle"] = failure_cycle
    out["RUL"] = failure_cycle - out[CYCLE_COL]
    return out


def add_test_rul(df: pd.DataFrame, final_rul: np.ndarray) -> pd.DataFrame:
    """Test engines stop early: RUL at the last observed cycle equals the value
    from ``RUL_FD00x.txt``, aligned by the official engine order."""
    out = df.copy()
    engine_order = out[ID_COL].drop_duplicates().to_numpy()
    assert len(engine_order) == len(final_rul)
    rul_map = dict(zip(engine_order, final_rul))

    last_cycle = out.groupby(ID_COL)[CYCLE_COL].transform("max")
    tail_rul = out[ID_COL].map(rul_map).astype(float)
    # T(i) = last observed cycle + remaining cycles after it.
    out["failure_cycle"] = last_cycle + tail_rul
    out["RUL"] = out["failure_cycle"] - out[CYCLE_COL]
    return out


def add_horizon_labels(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Three *separate* binary columns y_h = 1[RUL <= h]."""
    out = df.copy()
    for h in horizons:
        out[f"fail_within_{h}"] = (out["RUL"] <= h).astype(int)
    return out


def piecewise_rul(rul: pd.Series | np.ndarray, cap: int) -> np.ndarray:
    """Piecewise-linear degradation target: constant while the engine is
    healthy, linear once degradation is observable. The cap is a modelling
    choice tuned on train/validation engines only."""
    return np.minimum(np.asarray(rul, dtype=float), float(cap))


# --------------------------------------------------------------------------
# Engine-level split
# --------------------------------------------------------------------------
@dataclass
class EngineSplit:
    subset: str
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    seed: int

    def check_disjoint(self) -> None:
        a, b, c = set(self.train_ids), set(self.val_ids), set(self.test_ids)
        assert not (a & b), "train/val engine ids overlap"
        assert len(a) and len(b) and len(c)
        # train/val come from train_FD00x, test from test_FD00x: different
        # engines by construction, but ids are reused across the two files, so
        # we track provenance rather than the bare integer.

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "subset": self.subset,
                    "seed": self.seed,
                    "train_engine_ids": sorted(map(int, self.train_ids)),
                    "val_engine_ids": sorted(map(int, self.val_ids)),
                    "test_engine_ids": sorted(map(int, self.test_ids)),
                    "note": "train/val engines come from train_FD00x.txt; "
                            "test engines come from test_FD00x.txt",
                },
                fh,
                indent=2,
            )


def make_engine_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subset: str,
    val_fraction: float = VAL_ENGINE_FRACTION,
    seed: int = SEED,
) -> EngineSplit:
    """Partition engine IDs *before* anything is fitted."""
    rng = np.random.default_rng(seed)
    ids = np.sort(train_df[ID_COL].unique())
    perm = rng.permutation(ids)
    n_val = int(round(val_fraction * len(ids)))
    val_ids = sorted(perm[:n_val].tolist())
    train_ids = sorted(perm[n_val:].tolist())
    test_ids = sorted(test_df[ID_COL].unique().tolist())
    split = EngineSplit(subset, train_ids, val_ids, test_ids, seed)
    split.check_disjoint()
    return split


def subset_by_engines(df: pd.DataFrame, ids) -> pd.DataFrame:
    return df[df[ID_COL].isin(list(ids))].copy()


# --------------------------------------------------------------------------
# Task 1 -- EDA / data audit
# --------------------------------------------------------------------------
def audit_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Row-level and engine-level integrity checks."""
    g = df.groupby(ID_COL)[CYCLE_COL]
    lengths = g.max() - g.min() + 1
    counts = g.count()
    # cycle continuity: 1..T with no gaps and no repeats
    continuous = int((lengths == counts).sum())
    starts_at_one = int((g.min() == 1).sum())

    rec = {
        "frame": name,
        "rows": len(df),
        "engines": df[ID_COL].nunique(),
        "min_len": int(counts.min()),
        "median_len": float(counts.median()),
        "max_len": int(counts.max()),
        "duplicate_(engine,cycle)": int(df.duplicated([ID_COL, CYCLE_COL]).sum()),
        "missing_values": int(df.isna().sum().sum()),
        "non_finite_values": int((~np.isfinite(df[OP_COLS + SENSOR_COLS].to_numpy())).sum()),
        "engines_with_continuous_cycles": continuous,
        "engines_starting_at_cycle_1": starts_at_one,
    }
    return pd.DataFrame([rec])


def sensor_variance_report(train_only: pd.DataFrame) -> pd.DataFrame:
    """Variance / near-constant check computed on TRAINING engines only."""
    stats = train_only[SENSOR_COLS + OP_COLS].agg(["std", "min", "max", "nunique"]).T
    stats.columns = ["std", "min", "max", "n_unique"]
    stats["range"] = stats["max"] - stats["min"]
    stats["near_constant"] = (stats["std"] < 1e-6) | (stats["n_unique"] <= 2)
    return stats.sort_values("std")


def horizon_class_balance(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    rows = []
    for h in horizons:
        col = f"fail_within_{h}"
        rows.append(
            {
                "horizon": h,
                "positives": int(df[col].sum()),
                "rows": len(df),
                "positive_rate": float(df[col].mean()),
                "imbalance_ratio": float((1 - df[col].mean()) / max(df[col].mean(), 1e-9)),
            }
        )
    return pd.DataFrame(rows)


class FilterLog:
    """Documented table of every filtering decision (Task 1, last bullet)."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(self, decision: str, rationale: str, engines: int = 0, rows: int = 0,
               columns: int = 0, detail: str = "") -> None:
        self._rows.append(
            {
                "decision": decision,
                "rationale": rationale,
                "engines_affected": engines,
                "rows_affected": rows,
                "columns_affected": columns,
                "detail": detail,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)
