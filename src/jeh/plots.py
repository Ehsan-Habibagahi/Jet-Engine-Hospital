"""Figures for the notebook. Every function returns the Matplotlib figure so
the notebook can display it and the driver can save it to reports/figures/.

Colour convention used throughout the project:
    healthy / CONTINUE  -> teal      elevated / INSPECT -> amber
    critical / STOP     -> red       reference lines    -> grey dashed
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CYCLE_COL, FIGURE_DIR, ID_COL, SENSOR_COLS

OK, WARN, BAD, REF = "#0d8a8a", "#e0a516", "#c62828", "#8a8f98"
plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def save(fig, name: str) -> None:
    fig.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", dpi=130)


# --------------------------------------------------------------------------
# Task 1 -- EDA
# --------------------------------------------------------------------------
def plot_length_distribution(train_df, test_df, subset: str):
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for df, lab, c in ((train_df, "train engines (run to failure)", OK),
                       (test_df, "test engines (truncated)", WARN)):
        lens = df.groupby(ID_COL)[CYCLE_COL].max()
        ax.hist(lens, bins=30, alpha=0.6, label=f"{lab}  (n={len(lens)})", color=c)
    ax.set_xlabel("observed sequence length (cycles)")
    ax.set_ylabel("engines")
    ax.set_title(f"{subset}: sequence-length distribution")
    ax.legend(frameon=False)
    return fig


def plot_sensor_trajectories(df, engine_ids, sensors=("sensor_2", "sensor_4",
                                                      "sensor_7", "sensor_11",
                                                      "sensor_12", "sensor_15"),
                             subset: str = ""):
    """Raw sensor traces: degradation direction and wildly different scales."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for ax, s in zip(axes.ravel(), sensors):
        for eid in engine_ids:
            g = df[df[ID_COL] == eid].sort_values(CYCLE_COL)
            ax.plot(g[CYCLE_COL], g[s], lw=0.9, alpha=0.85, label=f"engine {eid}")
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("cycle")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{subset}: raw sensor trajectories -- note opposite trends and "
                 f"scales spanning orders of magnitude", fontsize=10)
    fig.tight_layout()
    return fig


def plot_operating_settings(df, regime=None, subset: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    pairs = [(0, 1), (0, 2), (1, 2)]
    ops = df[[f"operational_setting_{i}" for i in (1, 2, 3)]].to_numpy()
    for ax, (i, j) in zip(axes, pairs):
        if regime is None:
            ax.scatter(ops[:, i], ops[:, j], s=2, alpha=0.2, color=OK)
        else:
            ax.scatter(ops[:, i], ops[:, j], s=2, alpha=0.35,
                       c=regime, cmap="tab10")
        ax.set_xlabel(f"setting {i + 1}")
        ax.set_ylabel(f"setting {j + 1}")
    fig.suptitle(f"{subset}: operating-setting space"
                 + (" coloured by discovered regime" if regime is not None else ""),
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_condition_effect(df, regime, sensor="sensor_2", subset: str = ""):
    """Why condition-aware normalisation is needed: the same sensor sits at a
    completely different level in each regime, dwarfing the degradation signal."""
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for r in np.unique(regime):
        ax.hist(df.loc[regime == r, sensor], bins=60, alpha=0.6, label=f"regime {r}")
    ax.set_xlabel(sensor)
    ax.set_ylabel("rows")
    ax.set_title(f"{subset}: {sensor} by operating regime")
    ax.legend(frameon=False, fontsize=7)
    return fig


# --------------------------------------------------------------------------
# Task 3 -- regression diagnostics
# --------------------------------------------------------------------------
def plot_residuals(y_true, y_pred, subset: str = "", model: str = ""):
    resid = np.asarray(y_pred, float) - np.asarray(y_true, float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].scatter(y_true, y_pred, s=3, alpha=0.15, color=OK)
    lim = [0, max(np.max(y_true), np.max(y_pred))]
    axes[0].plot(lim, lim, "--", color=REF)
    axes[0].set_xlabel("true RUL (capped)")
    axes[0].set_ylabel("predicted RUL")
    axes[0].set_title("prediction vs truth")

    axes[1].scatter(y_true, resid, s=3, alpha=0.15, color=OK)
    axes[1].axhline(0, ls="--", color=REF)
    axes[1].set_xlabel("true RUL (capped)")
    axes[1].set_ylabel("residual (pred - true)")
    axes[1].set_title("residuals: positive = LATE warning")

    axes[2].hist(resid, bins=60, color=OK, alpha=0.8)
    axes[2].axvline(0, ls="--", color=REF)
    axes[2].set_xlabel("residual (cycles)")
    axes[2].set_title(f"mean {resid.mean():+.2f}, "
                      f"{100 * (resid > 0).mean():.0f}% late")
    fig.suptitle(f"{subset} -- {model}: residual diagnostics", fontsize=10)
    fig.tight_layout()
    return fig


def plot_engine_traces(timeline: pd.DataFrame, engine_ids, rul_cap: int, subset: str = ""):
    """Engine-level prediction traces with the calibrated interval."""
    n = len(engine_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eid in zip(axes, engine_ids):
        g = timeline[timeline.engine_id == eid].sort_values("cycle")
        ax.fill_between(g.cycle, g.rul_lo, g.rul_hi, color=OK, alpha=0.20,
                        label="90% interval (CQR)")
        ax.plot(g.cycle, g.rul_pred, color=OK, lw=1.6, label="predicted RUL")
        ax.plot(g.cycle, np.minimum(g.RUL, rul_cap), color=REF,
                ls="--", lw=1.4, label=f"true RUL (capped at {rul_cap})")
        ax.set_title(f"engine {eid}")
        ax.set_xlabel("cycle")
    axes[0].set_ylabel("RUL (cycles)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{subset}: engine-level prediction traces", fontsize=10)
    fig.tight_layout()
    return fig


def plot_per_engine_distribution(timeline: pd.DataFrame, rul_cap: int, subset: str = ""):
    """A fleet average can hide catastrophic engines -- so show the whole
    per-engine distribution, not just the mean."""
    y = np.minimum(timeline.RUL.to_numpy(), rul_cap)
    per = (timeline.assign(ae=np.abs(timeline.rul_pred.to_numpy() - y))
                   .groupby("engine_id")
                   .agg(MAE=("ae", "mean"), n=("ae", "size")))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(per.MAE, bins=30, color=OK, alpha=0.85)
    axes[0].axvline(per.MAE.mean(), color=BAD, ls="--", lw=1.4,
                    label=f"fleet mean {per.MAE.mean():.2f}")
    axes[0].axvline(per.MAE.median(), color=REF, ls="--", lw=1.4,
                    label=f"median {per.MAE.median():.2f}")
    axes[0].set_xlabel("per-engine MAE (cycles)")
    axes[0].set_ylabel("engines")
    axes[0].set_title("per-engine error distribution")
    axes[0].legend(frameon=False, fontsize=8)

    s = per.MAE.sort_values().to_numpy()
    axes[1].plot(np.arange(1, len(s) + 1) / len(s) * 100, s, color=OK, lw=1.6)
    axes[1].axhline(per.MAE.mean(), color=BAD, ls="--", lw=1.2)
    axes[1].set_xlabel("percentile of engines")
    axes[1].set_ylabel("per-engine MAE (cycles)")
    axes[1].set_title(f"worst engine: {s[-1]:.1f} cycles "
                      f"({s[-1] / max(per.MAE.mean(), 1e-9):.1f}x the fleet mean)")
    fig.suptitle(f"{subset}: the fleet average hides the tail", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Task 4 -- classification diagnostics
# --------------------------------------------------------------------------
def plot_pr_and_calibration(pr_curves: dict, reliability: dict, subset: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = {10: BAD, 20: WARN, 30: OK}
    for h, curve in pr_curves.items():
        axes[0].plot(curve.recall, curve.precision, color=colors.get(h, OK),
                     lw=1.5, label=f"h = {h} cycles")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("precision-recall (test engines)")
    axes[0].legend(frameon=False)

    axes[1].plot([0, 1], [0, 1], "--", color=REF, lw=1)
    for h, rc in reliability.items():
        axes[1].plot(rc.mean_pred, rc.observed_rate, "o-", ms=4,
                     color=colors.get(h, OK), lw=1.4, label=f"h = {h}")
    axes[1].set_xlabel("mean predicted probability")
    axes[1].set_ylabel("observed failure rate")
    axes[1].set_title("calibration (isotonic, fitted on validation engines)")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{subset}: horizon classification diagnostics", fontsize=10)
    fig.tight_layout()
    return fig


def plot_confusion_grid(clf_test: pd.DataFrame, subset: str = ""):
    sel = clf_test[clf_test.selected]
    fig, axes = plt.subplots(1, len(sel), figsize=(3.1 * len(sel), 2.9))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, sel.iterrows()):
        cm = np.array([[row.TN, row.FP], [row.FN, row.TP]], float)
        ax.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                        fontsize=9, color="black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"h = {int(row.horizon)}  (thr {row.threshold:.3f})", fontsize=9)
        ax.grid(False)
    fig.suptitle(f"{subset}: confusion matrices at validation-tuned thresholds", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Task 5 -- anomaly diagnostics
# --------------------------------------------------------------------------
def plot_anomaly_profiles(norm_scores: pd.DataFrame, subset: str = ""):
    """Score trajectories aligned by cycles-to-failure, all four detectors."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [OK, WARN, BAD, "#5b6dc8"]
    dets = [c for c in norm_scores.columns if c not in ("engine_id", "cycle", "RUL")]
    for c, det in zip(colors, dets):
        d = norm_scores[norm_scores.RUL <= 200].copy()
        prof = d.groupby(d.RUL.round().astype(int))[det].mean()
        ax.plot(prof.index, prof.values, color=c, lw=1.5, label=det)
    ax.invert_xaxis()
    ax.set_xlabel("cycles to failure  (RUL) -- failure is at the right")
    ax.set_ylabel("normalised score (validation percentile)")
    ax.set_title(f"{subset}: do the scores rise before failure?")
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_anomaly_engine_traces(norm_scores: pd.DataFrame, detector: str,
                               engine_ids, threshold: float, subset: str = ""):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for eid in engine_ids:
        g = norm_scores[norm_scores.engine_id == eid].sort_values("cycle")
        ax.plot(g.cycle, g[detector], lw=1.1, alpha=0.9, label=f"engine {eid}")
    ax.axhline(threshold, ls="--", color=BAD, lw=1.2,
               label=f"alert threshold (val p{threshold * 100:.1f})")
    ax.set_xlabel("cycle")
    ax.set_ylabel("normalised anomaly score")
    ax.set_title(f"{subset}: {detector} score trajectories")
    ax.legend(frameon=False, fontsize=7)
    return fig


# --------------------------------------------------------------------------
# Task 6 -- early warning
# --------------------------------------------------------------------------
def plot_lead_time_distribution(lead_frames: dict, target_h: int, subset: str = ""):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels, data = [], []
    for name, w in lead_frames.items():
        v = w.loc[w.at_risk, "lead_time_L"].dropna()
        if len(v):
            short = name.replace("decision_policy_", "policy: ").replace("_", " ")
            labels.append(f"{short}\n(n={len(v)}, miss={w.loc[w.at_risk,'missed'].mean():.0%})")
            data.append(v.to_numpy())
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for patch in bp["boxes"]:
        patch.set_facecolor(OK)
        patch.set_alpha(0.5)
    ax.axhline(target_h, ls="--", color=BAD, lw=1.2,
               label=f"target action window h = {target_h}")
    ax.set_ylabel("lead time L = T - tau (cycles)")
    ax.set_title(f"{subset}: warning lead time by alert source (test engines)")
    ax.legend(frameon=False, fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=7)
    fig.tight_layout()
    return fig


def plot_cost_components(early_warning: pd.DataFrame, subset: str = ""):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    df = early_warning.set_index("alert_source")[
        ["cost_component_miss", "cost_component_late", "cost_component_early"]]
    df.plot(kind="barh", stacked=True, ax=ax, color=[BAD, WARN, OK], width=0.7)
    ax.set_xlabel("average cost per at-risk engine")
    ax.set_ylabel("")
    ax.set_title(f"{subset}: cost decomposition -- misses are never hidden in the total")
    ax.legend(frameon=False, fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def plot_master_comparison(master: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    x = np.arange(len(master))
    axes[0].bar(x, master.test_MAE, color=OK)
    axes[0].set_ylabel("test MAE (cycles)")
    axes[0].set_title("RUL error")
    for h, c in zip((10, 20, 30), (BAD, WARN, OK)):
        axes[1].plot(x, master[f"PR_AUC_h{h}"], "o-", color=c, label=f"h={h}")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("failure-horizon classification")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].bar(x, master.policy_avg_cost, color=WARN)
    axes[2].set_ylabel("avg cost / at-risk engine")
    axes[2].set_title("decision-policy cost")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(master.subset, fontsize=8)
    fig.suptitle("Identical protocol across subsets: difficulty rises with "
                 "conditions and fault modes", fontsize=10)
    fig.tight_layout()
    return fig
