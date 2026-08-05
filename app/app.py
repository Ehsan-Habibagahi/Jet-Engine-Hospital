"""Jet Engine Hospital -- maintenance decision dashboard (Streamlit).

This file contains NO training code. It loads the artifacts exported by
``jeh.pipeline.export_artifacts`` and reproduces the notebook's inference
outputs exactly (verified by ``tests/test_app_parity.py``).

Run:  streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
# The pickled artifacts reference the ``jeh`` package for their class
# definitions, so it must be importable. On Hugging Face Spaces the src/ tree
# is shipped next to this file.
for cand in (ROOT / "src", APP_DIR / "src"):
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from jeh.config import CYCLE_COL, ID_COL, SENSOR_COLS  # noqa: E402
from jeh.policy import recommend  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
ACTION_STYLE = {
    "CONTINUE": ("#0d8a8a", "🟢", "Green -- keep flying"),
    "INSPECT": ("#e0a516", "🟠", "Amber -- schedule an inspection"),
    "STOP": ("#c62828", "🔴", "Red -- remove from service"),
}

st.set_page_config(page_title="Jet Engine Hospital", page_icon="🛩️", layout="wide")


# --------------------------------------------------------------------------
# Artifact loading (cached: startup memory stays modest, nothing is retrained)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def available_datasets() -> list[str]:
    if not ARTIFACTS.exists():
        return []
    return sorted(d.name for d in ARTIFACTS.iterdir()
                  if d.is_dir() and (d / "system.joblib").exists())


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_bundle(dataset: str):
    d = ARTIFACTS / dataset
    bundle = joblib.load(d / "system.joblib")
    history = pd.read_parquet(d / "test_history.parquet")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    return bundle["system"], bundle["policy"], history, meta


@st.cache_data(show_spinner=False)
def load_table(dataset: str, name: str) -> pd.DataFrame | None:
    p = ARTIFACTS / dataset / f"table_{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def engine_outputs(dataset: str, engine_id: int) -> pd.DataFrame:
    """Full causal inference over one engine's history.

    Computed the same way the notebook does it: transform the whole history
    (every feature is trailing, so row t only ever sees cycles <= t), then run
    the three model families over the result.
    """
    system, _, history, _ = load_bundle(dataset)
    hist = history[history[ID_COL] == engine_id].sort_values(CYCLE_COL)
    X = system.feature_pipeline.transform(hist)
    point, lo, hi = system.predict_rul(X, interval=True)
    risk = system.failure_risk(X)
    anom = system.anomaly_score(X)
    out = pd.DataFrame({
        "cycle": hist[CYCLE_COL].to_numpy(),
        "rul_pred": point, "rul_lo": lo, "rul_hi": hi,
        "anom_norm": anom.to_numpy(),
    })
    for c in risk.columns:
        out[c] = risk[c].to_numpy()
    out["true_RUL"] = hist["RUL"].to_numpy() if "RUL" in hist else np.nan
    return out


# --------------------------------------------------------------------------
# Small presentation helpers
# --------------------------------------------------------------------------
def card(title: str, body_html: str, accent: str = "#3b4252") -> None:
    st.markdown(
        f"""<div style="border:1px solid #d9dde3;border-left:6px solid {accent};
        border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.6rem;
        background:rgba(127,127,127,0.05)">
        <div style="font-size:0.78rem;letter-spacing:.06em;text-transform:uppercase;
        opacity:.7;margin-bottom:.35rem">{title}</div>{body_html}</div>""",
        unsafe_allow_html=True,
    )


def risk_bar(p: float, thr: float) -> str:
    pct = min(max(p, 0.0), 1.0) * 100
    tpos = min(max(thr, 0.0), 1.0) * 100
    color = "#c62828" if p >= thr else "#0d8a8a"
    return (
        f"""<div style="position:relative;height:16px;background:#e6e8ec;border-radius:8px;
        margin:2px 0 8px 0">
        <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:8px"></div>
        <div style="position:absolute;left:{tpos:.1f}%;top:-3px;width:2px;height:22px;
        background:#222"></div></div>"""
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
datasets = available_datasets()
if not datasets:
    st.error(
        "No artifacts found. Run the notebook, or `python -m jeh.run_all`, "
        "to populate `artifacts/`."
    )
    st.stop()

with st.sidebar:
    st.title("🛩️ Jet Engine Hospital")
    dataset = st.selectbox("C-MAPSS subset", datasets,
                           index=datasets.index("FD001") if "FD001" in datasets else 0)
    system, policy, history, meta = load_bundle(dataset)
    engines = sorted(history[ID_COL].unique().tolist())
    engine_id = st.selectbox("Test engine", engines)

    eng_hist = history[history[ID_COL] == engine_id]
    max_cycle = int(eng_hist[CYCLE_COL].max())
    cycle = st.slider("Cycle", 1, max_cycle, max_cycle, 1,
                      help="Only cycles up to this point are used -- every feature "
                           "in the pipeline is causal.")
    st.caption(f"Engine {engine_id} was observed for {max_cycle} cycles.")

    st.divider()
    st.caption("**Model metadata**")
    st.markdown(
        f"""
- **Dataset** `{meta['dataset']}`
- **Role** {meta['role']}
- **Model version** `{meta['model_version']}`
- **RUL model** `{meta['rul_model']}` (cap {meta['rul_cap']} cycles)
- **Classifier** `{meta['classifier']}`
- **Anomaly detector** `{meta['anomaly_detector']}`
- **Feature window** {meta['feature_window']} cycles (primary {meta['primary_window']})
- **Features** {meta['n_features']}
- **Condition-aware** {meta['condition_aware']} ({meta['n_conditions']} regimes)
- **Engines** train {meta['engines']['train']} / val {meta['engines']['val']} /
  test {meta['engines']['test']}
- **Last training run** {meta['trained_at']}
""")

outputs = engine_outputs(dataset, engine_id)
row = outputs[outputs.cycle == cycle].iloc[0]
hist_to_now = outputs[outputs.cycle <= cycle]
p_fail = {h: float(row[f"p_fail_{h}"]) for h in (10, 20, 30)}

decision = recommend(
    float(row.rul_pred), float(row.rul_lo), float(row.rul_hi),
    p_fail, hist_to_now["anom_norm"].to_numpy(), policy,
)
accent, emoji, banner = ACTION_STYLE[decision.action]

# --------------------------------------------------------------------------
# Header + action
# --------------------------------------------------------------------------
st.markdown(f"### Engine {engine_id} · cycle {cycle} of {max_cycle} · `{dataset}`")

st.markdown(
    f"""<div style="border-radius:10px;padding:1rem 1.2rem;background:{accent}18;
    border:2px solid {accent}">
    <div style="font-size:1.6rem;font-weight:700;color:{accent}">
    {emoji} {decision.action}</div>
    <div style="opacity:.85;margin-bottom:.4rem">{banner}</div>
    <div><b>Rule that fired:</b> {decision.trigger}</div>
    <div><b>Confidence:</b> {decision.confidence}</div>
    </div>""",
    unsafe_allow_html=True,
)

if decision.disagreement:
    st.warning(
        "**Models disagree.** The unsupervised detector flags persistent abnormal "
        "behaviour while the supervised near-term risk stays low. This can mean a "
        "fault mode or operating regime under-represented in training. The "
        "recommendation is *not* collapsed into a single confident answer -- treat "
        "this engine as a candidate for manual review.",
        icon="⚠️",
    )
st.write("")

# --------------------------------------------------------------------------
# Evidence cards
# --------------------------------------------------------------------------
c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    width = float(row.rul_hi - row.rul_lo)
    card(
        "Remaining Useful Life",
        f"""<div style="font-size:2.1rem;font-weight:700">{row.rul_pred:.0f}
        <span style="font-size:.95rem;font-weight:400;opacity:.7">cycles</span></div>
        <div style="opacity:.85">{100 * (1 - meta['conformal_alpha']):.0f}% prediction
        interval <b>[{row.rul_lo:.0f}, {row.rul_hi:.0f}]</b> cycles</div>
        <div style="opacity:.7;font-size:.85rem">interval width {width:.0f} cycles
        &middot; "wide" above {policy.wide_interval_cycles:.0f}</div>
        <div style="opacity:.7;font-size:.85rem">target saturates at the
        {meta['rul_cap']}-cycle cap</div>""",
        accent,
    )

with c2:
    bars = ""
    for h in (10, 20, 30):
        thr = policy.p10_stop if h == 10 else (policy.p20_inspect if h == 20
                                               else policy.p30_inspect)
        flag = "over threshold" if p_fail[h] >= thr else "below threshold"
        bars += (f"""<div style="display:flex;justify-content:space-between;
        font-size:.85rem"><span>P(fail within <b>{h}</b> cycles)</span>
        <span><b>{p_fail[h]:.3f}</b> &middot; thr {thr:.3f} &middot; {flag}</span></div>"""
                 + risk_bar(p_fail[h], thr))
    card("Calibrated failure risk", bars
         + """<div style="opacity:.65;font-size:.78rem">Isotonic calibration and
         thresholds fitted on validation engines under the declared asymmetric
         cost. The black tick marks the decision threshold.</div>""", accent)

with c3:
    thr = policy.anomaly_persist_level
    recent = hist_to_now["anom_norm"].to_numpy()[-policy.anomaly_persist_n:]
    n_recent = int((recent >= thr).sum())
    card(
        "Anomaly (unsupervised)",
        f"""<div style="font-size:2.1rem;font-weight:700">{row.anom_norm:.3f}</div>
        <div style="opacity:.85">validation percentile &middot; alert at
        <b>{thr:.3f}</b></div>
        <div style="opacity:.85">margin to threshold
        <b>{row.anom_norm - thr:+.3f}</b></div>
        <div style="opacity:.85">persistence <b>{n_recent}/{len(recent)}</b> of the last
        {policy.anomaly_persist_n} cycles (rule: {policy.anomaly_persist_m} of
        {policy.anomaly_persist_n})</div>
        <div style="opacity:.65;font-size:.78rem">Detector
        <code>{meta['anomaly_detector']}</code>, fitted with no failure label. This is a
        rank, <b>not</b> a probability.</div>""",
        accent,
    )

with st.expander("Full evidence chain (all drivers behind this recommendation)"):
    for d in decision.drivers:
        st.markdown(f"- {d}")
    if decision.next_review_cycles:
        st.markdown(f"- **Next review in {decision.next_review_cycles} cycles**")
    st.markdown(
        f"- Policy: STOP needs P(fail≤10) ≥ {policy.p10_stop:.3f} **and/or** a lower RUL "
        f"bound ≤ {policy.rul_lo_stop:.0f} cycles with a non-wide interval; STOP is "
        "withheld and downgraded to INSPECT when the interval is wide, because risk "
        "cannot then be separated from model uncertainty."
    )

# --------------------------------------------------------------------------
# Timelines
# --------------------------------------------------------------------------
st.markdown("#### Engine timeline")
tab1, tab2, tab3 = st.tabs(["RUL & interval", "Failure risk & anomaly", "Sensors"])

with tab1:
    chart = outputs.set_index("cycle")[["rul_pred", "rul_lo", "rul_hi"]]
    if outputs["true_RUL"].notna().any():
        chart["true_RUL"] = np.minimum(outputs["true_RUL"].to_numpy(), meta["rul_cap"])
    st.line_chart(chart, height=280)
    st.caption(f"Current cycle: {cycle}. `true_RUL` is shown for this offline benchmark "
               f"only -- it is never an input to any model.")

with tab2:
    st.line_chart(outputs.set_index("cycle")[
        ["p_fail_10", "p_fail_20", "p_fail_30", "anom_norm"]], height=280)
    st.caption("Longer horizons rise earlier (more action time) but are noisier and "
               "generate more false alarms -- that trade-off is quantified in the "
               "notebook's early-warning table.")

with tab3:
    default = [s for s in ("sensor_4", "sensor_11", "sensor_9") if s in history.columns]
    picks = st.multiselect("Sensors", SENSOR_COLS, default=default)
    if picks:
        eh = history[history[ID_COL] == engine_id].sort_values(CYCLE_COL)
        st.line_chart(eh.set_index(CYCLE_COL)[picks], height=280)

# --------------------------------------------------------------------------
# Fleet context + validated performance
# --------------------------------------------------------------------------
st.markdown("#### How much should you trust this?")
fc1, fc2 = st.columns(2)
with fc1:
    t = load_table(dataset, "interval_report")
    if t is not None:
        st.caption("**RUL interval quality on held-out test engines**")
        st.dataframe(
            t[["method", "set", "coverage", "mean_width"]].round(3),
            use_container_width=True, hide_index=True)
with fc2:
    t = load_table(dataset, "classification_test")
    if t is not None:
        sel = t[t.selected][["horizon", "PR_AUC", "precision", "recall", "brier",
                             "threshold"]]
        st.caption("**Deployed classifiers on held-out test engines**")
        st.dataframe(sel.round(3), use_container_width=True, hide_index=True)

t = load_table(dataset, "early_warning")
if t is not None:
    st.caption("**Early-warning performance by alert source (test engines)** — "
               "lead time, misses and the asymmetric cost decomposition")
    st.dataframe(
        t[["alert_source", "n_engines_at_risk", "miss_rate", "mean_lead_time",
           "mean_late_delay", "mean_early_burden", "avg_cost_per_engine",
           "false_alert_rate_per_healthy_cycle"]].round(3),
        use_container_width=True, hide_index=True)

st.caption(
    f"Cost policy: c_miss={meta['cost_policy']['c_miss']}, "
    f"c_late={meta['cost_policy']['c_late']}/cycle, "
    f"c_early={meta['cost_policy']['c_early']}/cycle, target action window "
    f"h={meta['cost_policy']['target_horizon']} cycles."
)
