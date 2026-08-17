# ✈️ Jet Engine Hospital: Predicting Failure Before It Happens
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-ff4b4b)
![pytest](https://img.shields.io/badge/tests-pytest-0a9edc)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Type](https://img.shields.io/badge/type-ML%20capstone-purple)
[![Live dashboard](https://img.shields.io/badge/dashboard-live-ff4b4b?logo=streamlit&logoColor=white)](https://jet-engine-hospital-jeh.streamlit.app)
[![Technical report](https://img.shields.io/badge/report-PDF-8b1a1a?logo=adobeacrobatreader&logoColor=white)](reports/doc/report.pdf)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/Ehsan-Habibagahi/Jet-Engine-Hospital/actions/workflows/tests.yml/badge.svg)](https://github.com/Ehsan-Habibagahi/Jet-Engine-Hospital/actions/workflows/tests.yml)

<div align="center">
  <img width="100%" alt="jeh" src="https://github.com/user-attachments/assets/89182539-4455-4a97-98f7-36628708a553" />
</div>


A multi-task early-warning system for NASA C-MAPSS turbofan engines. It combines **1.RUL
regression** with a calibrated prediction interval, **2.failure-horizon classification** at
10/20/30 cycles, **3.unsupervised anomaly detection**, and a **4.CONTINUE / INSPECT / STOP**
maintenance policy tuned under an explicit asymmetric cost.

**[Live dashboard](https://jet-engine-hospital-jeh.streamlit.app)** | **[Full technical report](reports/doc/report.pdf)**

| Stage | Subset | Challenge |
|---|---|---|
| 1. Foundation | FD001 | 1 operating condition, 1 fault mode |
| 2. Base grade | FD003 | 1 condition, 2 fault modes |
| 3. Companion | FD002 | 6 conditions, 1 fault mode |
| 4. Advanced | FD004 | 6 conditions + 2 fault modes |

---
## 🗃️ Dataset Description

**What "FDXXX" means:** the benchmark ships as four subsets, each a different difficulty setting
along two independent axes: how many operating conditions the fleet flies under, and how many
fault modes can occur.

| Subset | Engines (train/test) | Conditions | Fault modes | What makes it hard |
|---|---:|:---:|:---:|---|
| FD001 | 100 / 100 | 1 (sea level) | 1 (HPC degradation) | the clean baseline case |
| FD002 | 260 / 259 | 6 | 1 (HPC) | operating context varies a lot, obscuring degradation |
| FD003 | 100 / 100 | 1 | 2 (HPC + fan) | two distinct failure mechanisms mixed together |
| FD004 | 248 / 249 | 6 | 2 (HPC + fan) | both problems at once, the hardest |

"HPC" means high-pressure compressor degradation; "fan" means fan degradation. A model that
only ever saw FD001 has it easy: one context, one failure story. FD002 forces the model to
separate "the engine is at high altitude" from "the engine is dying," since both shift the
sensors similarly. FD004 requires doing both at the same time.

In each subset, training engines run all the way to failure (so you know their true remaining
life at every cycle), while test engines are cut off at some arbitrary point before failure,
with the true remaining cycles given separately in RUL_FDXXX.txt.

---
## 🚀 Quick start

```bash
pip install -r requirements.txt

# 1. Unpack the dataset (already done if data/ contains train_FD001.txt)
#    CMAPSSData.zip goes into data/

# 2. Run the full protocol: 4 subsets, 7 ablations, advanced analyses (about 60-75 min)
python -m jeh.run_all              # PYTHONPATH=src, or run from the repo root

# 3. Correctness checks (causality, split disjointness, cost signs, app parity)
python -m pytest tests -q

# 4. The dashboard
streamlit run app/app.py
```

The notebook `notebooks/JetEngineHospital.ipynb` is the primary deliverable and runs
top-to-bottom. With the artifact cache present it replays in about 2 minutes and produces
identical tables (everything is seeded); set `REUSE_CACHE = False` in the setup cell for a
cold, fully-recomputed run.

---

## 📂 Layout

```
src/jeh/            the method, imported by BOTH the notebook and the app
  config.py         paths, seeds, cost policy, RunConfig
  data.py           loading, RUL/horizon labels, engine-level split, data audit
  features.py       causal FeaturePipeline (trailing windows, regime-aware residuals)
  models.py         regression/classification zoos, AnomalyBank, PrognosticsSystem
  uncertainty.py    split conformal, conformalized quantile regression, calibration
  evaluation.py     metrics, engine-level bootstrap, lead time, asymmetric cost
  policy.py         the auditable CONTINUE / INSPECT / STOP decision layer
  pipeline.py       run_experiment(): the locked end-to-end protocol
  experiments.py    cross-subset comparison, ablations, FD004 advanced analyses
  plots.py          figures
  run_all.py        driver + result cache
notebooks/          JetEngineHospital.ipynb
app/                app.py (Streamlit) + app_parity_check.py
tests/              leakage, metric and policy correctness tests
artifacts/<subset>/ exported models, calibrators, thresholds, metadata, result tables
reports/            figures, tables, run log
tools/              build_notebook.py, regenerates the .ipynb from source
```

The app contains **no training code**. It loads `artifacts/<subset>/system.joblib` and calls
the same `PrognosticsSystem` the notebook used, which is why parity is provable rather than
hoped for:

```bash
python app/app_parity_check.py FD001
```

This compares the app's inference against the notebook's precomputed timeline row by row and
asserts agreement to 1e-9, **and** 100% agreement on the CONTINUE/INSPECT/STOP recommendation.

---

## 🫗 Leakage controls

The core constraint is that the split happens first and everything learned is fitted
downstream of it.

| Step | Allowed data | How it is enforced |
|---|---|---|
| Split IDs | `engine_id` only | `make_engine_split` runs before any fit; written to `engine_split.json` |
| Fit preprocessing | training engines | scaler, regime clusters, PCA, sensor drop list all fitted in `FeaturePipeline.fit(tr)` |
| Create windows | inside each split | per-engine `groupby`; no window crosses an engine or split boundary |
| Tune / calibrate | validation engines | RUL cap, thresholds, isotonic calibration, conformal quantiles |
| Final evaluation | test engines | one locked pipeline, evaluated once |

**Causality** is asserted, not assumed: `FeaturePipeline.causality_check` recomputes the
features from a truncated history and requires the row at cycle *t* to be bit-identical.
`run_experiment` runs this on every subset, so a leaky feature aborts the pipeline instead of
quietly inflating the score. `tests/test_pipeline.py` runs it across four cycles and five
engines, and separately pins the one subtle case: the "drift from first cycles" baseline uses
an *expanding* mean that freezes at cycle 5, so a row at cycle 2 never sees cycle 5.

---

## Two decisions worth reading before the results

**The RUL cap is estimated, not grid-searched.** The cap changes the *target*, so errors
against differently capped truths are not comparable, and clipping predictions at a low cap
structurally limits RUL over-estimation, which is exactly what the PHM score punishes. A naive
validation grid-search therefore collapses to the smallest candidate and reports a meaningless
MAE dominated by rows sitting on a constant. We instead estimate the cap from degradation
onset (PC1 of the regime-normalised residuals, 3 sigma over each engine's own first-30-cycle
baseline, persistent 3-of-5), on training engines and without touching a label. It lands at
around 110-130 cycles on all four subsets, independently reproducing the value most published
C-MAPSS work assumes.

**The deployed RUL interval is CQR, not plain split conformal.** Split conformal on absolute
residuals gives every row the *same* width. That meets the coverage requirement and fails the
*decision* requirement: the policy must withhold STOP when risk cannot be separated from model
uncertainty, and a constant width makes "is this interval wide?" vacuous. Conformalized
quantile regression keeps the coverage guarantee while letting width track local difficulty.
Both are reported side by side.

---

## ⚖️ Cost policy

| Term | Value | Meaning |
|---|---|---|
| `c_miss` | 200 / engine | an engine fails with no prior alert, the safety-critical outcome |
| `c_late` | 5 / cycle | the alert arrived inside the action window; the maintenance slot is compressed |
| `c_early` | 1 / cycle | a serviceable engine is inspected early, wasted capacity, no safety consequence |
| `h` | 20 cycles | the action window the organisation plans against |

`c_miss > c_late > c_early`, as required. Row-level classification thresholds use a matching
`FN : FP = 20 : 1`, which is why the selected thresholds sit far below 0.50.

Costs are always reported with their components; a low average that hides a non-zero miss
rate is not a better system.

---

## Data attribution

A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation*, PHM 2008. Dataset: NASA Ames Prognostics Data Repository,
*Turbofan Engine Degradation Simulation Data Set*.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository, make your changes, and open a
pull request.

