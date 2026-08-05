"""Correctness tests for the leakage controls and the evaluation machinery.

    python -m pytest tests -q        (or simply: python tests/test_pipeline.py)

These are the checks that would catch a silently-optimistic result: causality
of every feature, disjointness of the engine splits, the PHM sign convention,
the persistence rule, and app/notebook parity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from jeh import data as D  # noqa: E402
from jeh import evaluation as E  # noqa: E402
from jeh import uncertainty as U  # noqa: E402
from jeh.config import ARTIFACT_DIR, CYCLE_COL, ID_COL, N_CONDITIONS, SEED  # noqa: E402
from jeh.features import FeaturePipeline  # noqa: E402
from jeh.policy import DecisionPolicy, recommend  # noqa: E402

SUBSET = "FD001"


@pytest.fixture(scope="module")
def prepared():
    train_raw, test_raw, rul = D.load_subset(SUBSET)
    train_all = D.add_horizon_labels(D.add_train_rul(train_raw))
    test_all = D.add_horizon_labels(D.add_test_rul(test_raw, rul))
    split = D.make_engine_split(train_all, test_all, SUBSET)
    tr = D.subset_by_engines(train_all, split.train_ids)
    fp = FeaturePipeline(n_conditions=N_CONDITIONS[SUBSET], seed=SEED).fit(tr)
    return train_all, test_all, split, tr, fp


# ---------------------------------------------------------------- labels
def test_train_rul_reaches_zero_at_failure(prepared):
    train_all = prepared[0]
    last = train_all.groupby(ID_COL).tail(1)
    assert (last["RUL"] == 0).all(), "training RUL must be 0 at the failure cycle"


def test_test_rul_matches_official_file(prepared):
    _, test_all, _, _, _ = prepared
    _, _, rul = D.load_subset(SUBSET)
    last = test_all.groupby(ID_COL).tail(1).sort_values(ID_COL)
    assert np.allclose(last["RUL"].to_numpy(), rul), \
        "RUL at the last observed cycle must equal RUL_FD00x.txt"


def test_horizon_labels_are_nested(prepared):
    train_all = prepared[0]
    # RUL<=10 implies RUL<=20 implies RUL<=30
    assert (train_all.fail_within_10 <= train_all.fail_within_20).all()
    assert (train_all.fail_within_20 <= train_all.fail_within_30).all()


# ---------------------------------------------------------------- split
def test_engine_splits_are_disjoint(prepared):
    split = prepared[2]
    assert not (set(split.train_ids) & set(split.val_ids))
    assert len(split.train_ids) + len(split.val_ids) == 100


def test_split_is_deterministic(prepared):
    train_all, test_all, split, _, _ = prepared
    again = D.make_engine_split(train_all, test_all, SUBSET)
    assert again.train_ids == split.train_ids


# ---------------------------------------------------------------- causality
@pytest.mark.parametrize("cycle", [5, 31, 90, 150])
def test_features_are_causal(prepared, cycle):
    """The headline leakage control: truncating the future must not change the
    features at cycle t."""
    _, _, split, tr, fp = prepared
    for eid in split.train_ids[:5]:
        hist = tr[tr[ID_COL] == eid]
        if cycle <= hist[CYCLE_COL].max():
            assert fp.causality_check(tr, eid, cycle), \
                f"leakage detected: engine {eid}, cycle {cycle}"


def test_preprocessing_never_saw_validation_engines(prepared):
    """Regime stats and the scaler must come from TRAIN rows only, so refitting
    on train alone reproduces them exactly."""
    _, _, split, tr, fp = prepared
    refit = FeaturePipeline(n_conditions=N_CONDITIONS[SUBSET], seed=SEED).fit(tr)
    assert np.allclose(refit.scaler_mu_, fp.scaler_mu_)
    assert refit.dropped_sensors_ == fp.dropped_sensors_


def test_drift_baseline_does_not_peek(prepared):
    """The 'drift from first cycles' feature must use an expanding mean before
    cycle 5, not the mean of the first 5 cycles."""
    from jeh.features import _frozen_baseline

    s = pd.Series([0.0, 10.0, 20.0, 30.0, 40.0, 999.0, 999.0])
    b = _frozen_baseline(s, 5)
    assert b.iloc[0] == 0.0, "at t=0 the baseline can only be the first value"
    assert b.iloc[1] == 5.0
    assert b.iloc[4] == b.iloc[5] == b.iloc[6] == 20.0, "baseline must freeze at cycle 5"


# ---------------------------------------------------------------- metrics
def test_phm_penalises_late_more_than_early():
    late = E.phm_score([30.0], [40.0])    # predicted MORE life than there is
    early = E.phm_score([30.0], [20.0])
    assert late > early, "RUL over-estimation (late warning) must cost more"
    assert E.phm_score([30.0], [30.0]) == pytest.approx(0.0)


def test_threshold_selection_prefers_recall_under_asymmetric_cost():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.02).astype(int)
    p = np.clip(0.02 + 0.6 * y + rng.normal(0, 0.15, 2000), 0, 1)
    t_bal, _ = E.pick_threshold_by_cost(y, p, fn_fp_ratio=1.0)
    t_asym, _ = E.pick_threshold_by_cost(y, p, fn_fp_ratio=20.0)
    assert t_asym < t_bal, "a costlier FN must push the threshold down"


def test_persistence_rule_ignores_isolated_spikes():
    cycles = np.arange(1, 21)
    spike = np.zeros(20, int)
    spike[[3, 9, 15]] = 1                       # isolated, never 3-of-5
    assert E.EvaluationSuite.first_persistent_alert(cycles, spike, 3, 5) is None
    burst = np.zeros(20, int)
    burst[10:13] = 1                            # 3 within 5 -> fires at cycle 13
    assert E.EvaluationSuite.first_persistent_alert(cycles, burst, 3, 5) == 13


def test_lead_time_and_miss_accounting():
    suite = E.EvaluationSuite(seed=0)
    tl = pd.DataFrame({
        "engine_id": [1] * 10 + [2] * 10,
        "cycle": list(range(1, 11)) * 2,
        # engine 1 fails at 12 (observed to RUL 2), engine 2 fails at 200 (never at risk)
        "failure_cycle": [12] * 10 + [200] * 10,
        "RUL": list(range(11, 1, -1)) + list(range(199, 189, -1)),
        "alert": [0] * 5 + [1] * 5 + [0] * 10,
    })
    w = suite.warning_lead_time(tl, horizon=20)
    e1 = w[w.engine_id == 1].iloc[0]
    e2 = w[w.engine_id == 2].iloc[0]
    assert e1.first_persistent_alert_tau == 8 and e1.lead_time_L == 4
    assert e2.at_risk is np.False_ or not e2.at_risk, \
        "an engine truncated 190 cycles before failure is not at risk"
    assert not e2.missed, "a not-at-risk engine must never be counted as a miss"


def test_engine_bootstrap_resamples_engines_not_rows():
    suite = E.EvaluationSuite(seed=0)
    df = pd.DataFrame({"engine_id": np.repeat([1, 2, 3, 4], 50),
                       "v": np.repeat([0.0, 0.0, 0.0, 10.0], 50)})
    out = suite.engine_bootstrap(df, lambda d: d.v.mean(), n_boot=300)
    # with only 4 engines and one extreme engine the CI must be wide;
    # a row-level bootstrap would collapse it to nearly zero width.
    assert out["ci_high"] - out["ci_low"] > 3.0


# ---------------------------------------------------------------- uncertainty
def test_conformal_quantile_achieves_nominal_coverage():
    rng = np.random.default_rng(0)
    y_cal, p_cal = rng.normal(50, 10, 4000), np.zeros(4000) + 50
    q = U.conformal_quantile(y_cal, p_cal, alpha=0.10)
    y_new = rng.normal(50, 10, 4000)
    cov = np.mean(np.abs(y_new - 50) <= q)
    assert cov >= 0.88, f"coverage {cov:.3f} far below nominal 0.90"


# ---------------------------------------------------------------- policy
def test_single_critical_signal_plus_wide_interval_is_downgraded_to_inspect():
    """The uncertainty-aware rule: when the evidence rests on ONE signal and the
    interval is wide, risk cannot be separated from model uncertainty, so STOP
    is withheld."""
    pol = DecisionPolicy(p10_stop=0.5, rul_lo_stop=15.0, wide_interval_cycles=40.0)
    calm_risk = {10: 0.0, 20: 0.0, 30: 0.0}
    tight = recommend(12, 8, 20, calm_risk, np.zeros(5), pol)     # bound only, width 12
    wide = recommend(12, 2, 90, calm_risk, np.zeros(5), pol)      # bound only, width 88
    assert tight.action == "STOP"
    assert wide.action == "INSPECT"
    assert "uncertainty" in wide.trigger and wide.confidence.startswith("low")


def test_two_independent_critical_signals_stop_even_when_wide():
    """...but two independent critical signals are the documented justification,
    so they are NOT downgraded. This asymmetry is intentional and auditable."""
    pol = DecisionPolicy(p10_stop=0.5, rul_lo_stop=15.0, wide_interval_cycles=40.0)
    risk = {10: 0.9, 20: 0.9, 30: 0.9}
    wide = recommend(12, 2, 90, risk, np.zeros(5), pol)
    assert wide.action == "STOP"
    assert "independent" in wide.trigger


def test_continue_requires_all_three_signals_calm():
    pol = DecisionPolicy(wide_interval_cycles=1e9)
    calm = recommend(100, 90, 110, {10: 0.0, 20: 0.0, 30: 0.0}, np.zeros(5), pol)
    assert calm.action == "CONTINUE" and calm.next_review_cycles > 0
    anomalous = recommend(100, 90, 110, {10: 0.0, 20: 0.0, 30: 0.0}, np.ones(5), pol)
    assert anomalous.action == "INSPECT", "a persistent anomaly must escalate"


def test_disagreement_is_surfaced_not_averaged():
    pol = DecisionPolicy(p30_inspect=0.4, wide_interval_cycles=1e9)
    d = recommend(100, 90, 110, {10: 0.0, 20: 0.0, 30: 0.01}, np.ones(5), pol)
    assert d.disagreement, "high anomaly + low supervised risk must be flagged"


def test_every_decision_carries_its_trigger():
    pol = DecisionPolicy()
    for risk, anom in (({10: 0.0, 20: 0.0, 30: 0.0}, np.zeros(5)),
                       ({10: 0.9, 20: 0.9, 30: 0.9}, np.ones(5))):
        d = recommend(50, 30, 70, risk, anom, pol)
        assert d.trigger and d.confidence and d.drivers


# ---------------------------------------------------------------- app parity
@pytest.mark.skipif(not (ARTIFACT_DIR / SUBSET / "system.joblib").exists(),
                    reason="run `python -m jeh.run_all` first")
def test_app_reproduces_notebook_outputs():
    import app_parity_check

    report = app_parity_check.run(SUBSET, n_engines=5)
    assert report["max_abs_diff"].max() <= 1e-9
    assert report["action_agreement"].min() == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
