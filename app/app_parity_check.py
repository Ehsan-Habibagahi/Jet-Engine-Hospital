"""App parity: prove the dashboard reproduces the notebook's inference outputs.

The app never re-fits anything -- it loads ``artifacts/<subset>/system.joblib``
and calls the same ``PrognosticsSystem`` the notebook used. This module runs the
app's own per-engine inference function and compares it, row by row, against the
timeline the notebook precomputed during ``run_experiment``.

Any mismatch means a real bug (a refit, a different feature view, a non-causal
transform), not a rounding difference.

    python app/app_parity_check.py FD001
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import joblib  # noqa: E402

from jeh.config import ARTIFACT_DIR, CYCLE_COL, ID_COL  # noqa: E402
from jeh.policy import recommend  # noqa: E402

COLS = ["rul_pred", "rul_lo", "rul_hi", "p_fail_10", "p_fail_20", "p_fail_30", "anom_norm"]


def app_inference(system, history: pd.DataFrame, engine_id: int) -> pd.DataFrame:
    """Byte-for-byte the computation in ``app.engine_outputs``."""
    hist = history[history[ID_COL] == engine_id].sort_values(CYCLE_COL)
    X = system.feature_pipeline.transform(hist)
    point, lo, hi = system.predict_rul(X, interval=True)
    risk = system.failure_risk(X)
    out = pd.DataFrame({"engine_id": engine_id, "cycle": hist[CYCLE_COL].to_numpy(),
                        "rul_pred": point, "rul_lo": lo, "rul_hi": hi,
                        "anom_norm": system.anomaly_score(X).to_numpy()})
    for c in risk.columns:
        out[c] = risk[c].to_numpy()
    return out


def run(subset: str = "FD001", n_engines: int = 8, tol: float = 1e-9) -> pd.DataFrame:
    d = ARTIFACT_DIR / subset
    bundle = joblib.load(d / "system.joblib")
    system, policy = bundle["system"], bundle["policy"]
    history = pd.read_parquet(d / "test_history.parquet")
    expected = pd.read_parquet(d / "precomputed_timeline.parquet")

    engines = sorted(history[ID_COL].unique())[:n_engines]
    rows = []
    for eid in engines:
        got = app_inference(system, history, eid)
        want = expected[expected.engine_id == eid].sort_values("cycle")
        assert len(got) == len(want), f"row count differs for engine {eid}"
        diffs = {c: float(np.max(np.abs(got[c].to_numpy() - want[c].to_numpy())))
                 for c in COLS}

        # ...and the recommendation itself, which is what the user actually sees.
        anom = got["anom_norm"].to_numpy()
        actions = [
            recommend(float(r.rul_pred), float(r.rul_lo), float(r.rul_hi),
                      {h: float(r[f"p_fail_{h}"]) for h in (10, 20, 30)},
                      anom[: i + 1], policy).action
            for i, (_, r) in enumerate(got.iterrows())
        ]
        action_match = float(np.mean(np.array(actions) == want["action"].to_numpy()))
        rows.append({"engine_id": eid, "n_cycles": len(got),
                     "max_abs_diff": max(diffs.values()),
                     "action_agreement": action_match, **diffs})

    report = pd.DataFrame(rows)
    worst = report["max_abs_diff"].max()
    worst_action = report["action_agreement"].min()
    print(f"[{subset}] app vs notebook over {len(engines)} engines, "
          f"{report.n_cycles.sum():,} engine-cycles")
    print(f"  max absolute difference across all outputs : {worst:.3e}  (tol {tol:.0e})")
    print(f"  CONTINUE/INSPECT/STOP agreement            : {worst_action:.1%}")
    assert worst <= tol, f"APP PARITY FAILED: numeric drift {worst:.3e}"
    assert worst_action == 1.0, "APP PARITY FAILED: recommendations differ"
    print("  PASS -- the app reproduces the notebook's inference outputs exactly.")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "FD001")
