"""Section 2.1 -- turn the three model families into one auditable action.

Every rule is (a) traceable to a named quantity, (b) parameterised by a value
tuned on VALIDATION engines, and (c) reported together with the rule that
fired. Nothing here is a black box: ``recommend`` returns the trigger string.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd


@dataclass
class DecisionPolicy:
    """Thresholds are *learned on validation engines*, never on test."""

    p10_stop: float = 0.5            # calibrated 10-cycle risk that justifies STOP
    p20_inspect: float = 0.3         # calibrated 20-cycle risk that justifies INSPECT
    p30_inspect: float = 0.4         # 30-cycle risk (longer horizon, looser)
    rul_lo_stop: float = 15.0        # critical *lower* conformal bound, in cycles
    rul_lo_inspect: float = 45.0
    anomaly_persist_level: float = 0.99   # normalised (validation-percentile) score
    anomaly_persist_m: int = 3
    anomaly_persist_n: int = 5
    wide_interval_cycles: float = 60.0    # interval width flagged as high uncertainty
    review_fraction: float = 0.3          # next review at 30% of the lower RUL bound
    min_review_cycles: int = 5

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    action: str
    trigger: str
    confidence: str
    drivers: list = field(default_factory=list)
    next_review_cycles: int | None = None
    disagreement: bool = False


def _persistent(flags: np.ndarray, m: int, n: int) -> bool:
    """>= m alerts within the last n cycles of the supplied history."""
    tail = np.asarray(flags, int)[-n:]
    return bool(tail.sum() >= m)


def recommend(
    rul_point: float,
    rul_lo: float,
    rul_hi: float,
    p_fail: dict,
    anomaly_norm_history: np.ndarray,
    policy: DecisionPolicy,
) -> Decision:
    """One engine, one cycle -> one recommendation with its evidence chain."""
    width = float(rul_hi - rul_lo)
    anom_now = float(anomaly_norm_history[-1]) if len(anomaly_norm_history) else 0.0
    anom_flags = (np.asarray(anomaly_norm_history, float) >= policy.anomaly_persist_level)
    anom_persistent = _persistent(anom_flags, policy.anomaly_persist_m, policy.anomaly_persist_n)
    wide = width >= policy.wide_interval_cycles

    drivers = [
        f"RUL point {rul_point:.0f} cycles, 90% interval [{rul_lo:.0f}, {rul_hi:.0f}] "
        f"(width {width:.0f})",
        f"P(fail<=10)={p_fail.get(10, float('nan')):.3f} (STOP at {policy.p10_stop:.2f})",
        f"P(fail<=20)={p_fail.get(20, float('nan')):.3f} (INSPECT at {policy.p20_inspect:.2f})",
        f"P(fail<=30)={p_fail.get(30, float('nan')):.3f} (INSPECT at {policy.p30_inspect:.2f})",
        f"anomaly percentile {anom_now:.3f}"
        + (" -- persistent" if anom_persistent else " -- not persistent"),
    ]

    # --- disagreement: unsupervised says abnormal, supervised says calm ----
    disagreement = bool(anom_persistent and p_fail.get(30, 0.0) < policy.p30_inspect / 2)

    # ---------------- STOP -------------------------------------------------
    # The wide-interval downgrade below exists because the brief forbids STOP
    # when risk cannot be separated from model uncertainty *without
    # justification*. Two independent critical signals ARE that justification:
    # the calibrated 10-cycle probability is a separate statement of risk that
    # does not rest on the interval. So the downgrade applies only when the
    # evidence rests on a SINGLE signal.
    high_near_risk = p_fail.get(10, 0.0) >= policy.p10_stop
    critical_bound = rul_lo <= policy.rul_lo_stop
    if high_near_risk and critical_bound:
        return Decision(
            "STOP",
            f"P(fail<=10)={p_fail[10]:.3f} >= {policy.p10_stop:.2f} AND lower RUL bound "
            f"{rul_lo:.0f} <= {policy.rul_lo_stop:.0f} cycles"
            + (f" (interval is wide at {width:.0f} cycles, but the two signals are "
               "independent, which is the documented justification)" if wide else ""),
            "high (two independent critical signals agree)",
            drivers, None, disagreement,
        )
    if critical_bound and not wide:
        return Decision(
            "STOP",
            f"lower RUL bound {rul_lo:.0f} <= {policy.rul_lo_stop:.0f} cycles with a "
            f"tight interval (width {width:.0f} < {policy.wide_interval_cycles:.0f})",
            "high (calibrated interval is narrow)",
            drivers, None, disagreement,
        )
    if high_near_risk and not wide:
        return Decision(
            "STOP",
            f"validated near-term risk P(fail<=10)={p_fail[10]:.3f} >= {policy.p10_stop:.2f}",
            "high (threshold tuned on validation engines)",
            drivers, None, disagreement,
        )

    # ---------------- INSPECT ---------------------------------------------
    # Deliberately reached when the evidence is critical but *uncertain*: the
    # brief forbids STOP when risk cannot be separated from model uncertainty.
    if (critical_bound or high_near_risk) and wide:
        return Decision(
            "INSPECT",
            f"critical evidence but the 90% RUL interval is wide ({width:.0f} cycles) -- "
            "model uncertainty cannot be separated from risk, so STOP is withheld",
            "low (uncertainty-limited)", drivers,
            max(policy.min_review_cycles, int(policy.review_fraction * max(rul_lo, 0))),
            disagreement,
        )
    reasons = []
    if p_fail.get(20, 0.0) >= policy.p20_inspect:
        reasons.append(f"P(fail<=20)={p_fail[20]:.3f} >= {policy.p20_inspect:.2f}")
    if p_fail.get(30, 0.0) >= policy.p30_inspect:
        reasons.append(f"P(fail<=30)={p_fail[30]:.3f} >= {policy.p30_inspect:.2f}")
    if rul_lo <= policy.rul_lo_inspect:
        reasons.append(f"lower RUL bound {rul_lo:.0f} <= {policy.rul_lo_inspect:.0f}")
    if anom_persistent:
        reasons.append(
            f"anomaly score >= p{policy.anomaly_persist_level * 100:.0f} on "
            f"{policy.anomaly_persist_m}/{policy.anomaly_persist_n} recent cycles"
        )
    if wide and p_fail.get(30, 0.0) >= policy.p30_inspect / 2:
        reasons.append(f"wide RUL interval ({width:.0f} cycles) with elevated risk")
    if reasons:
        return Decision(
            "INSPECT", " ; ".join(reasons),
            "medium" if len(reasons) > 1 else "low",
            drivers,
            max(policy.min_review_cycles, int(policy.review_fraction * max(rul_lo, 0))),
            disagreement,
        )

    # ---------------- CONTINUE --------------------------------------------
    return Decision(
        "CONTINUE",
        f"lower RUL bound {rul_lo:.0f} > {policy.rul_lo_inspect:.0f} cycles, all calibrated "
        "risks below threshold, anomaly not persistent",
        "high" if not wide else "medium (wide interval)",
        drivers,
        max(policy.min_review_cycles, int(policy.review_fraction * max(rul_lo, 0))),
        disagreement,
    )


# --------------------------------------------------------------------------
# Vectorised application over a full timeline (used for evaluation)
# --------------------------------------------------------------------------
def apply_policy_timeline(timeline: pd.DataFrame, policy: DecisionPolicy,
                          horizons=(10, 20, 30)) -> pd.DataFrame:
    """``timeline`` needs engine_id, cycle, rul_pred/lo/hi, p_fail_h, anom_norm."""
    out = []
    for eid, g in timeline.groupby("engine_id"):
        g = g.sort_values("cycle")
        hist = g["anom_norm"].to_numpy(float)
        for i, (_, row) in enumerate(g.iterrows()):
            d = recommend(
                row["rul_pred"], row["rul_lo"], row["rul_hi"],
                {h: row[f"p_fail_{h}"] for h in horizons},
                hist[: i + 1],
                policy,
            )
            out.append(
                {"engine_id": eid, "cycle": row["cycle"], "action": d.action,
                 "trigger": d.trigger, "confidence": d.confidence,
                 "disagreement": d.disagreement,
                 "next_review_cycles": d.next_review_cycles}
            )
    return pd.DataFrame(out)


def action_confusion_by_engine(actions: pd.DataFrame, timeline: pd.DataFrame,
                               horizon: int = 20) -> pd.DataFrame:
    """CONTINUE/INSPECT/STOP confusion against the ground-truth risk state of
    each engine-cycle (Section 6.3 diagnostic)."""
    m = actions.merge(timeline[["engine_id", "cycle", "RUL"]], on=["engine_id", "cycle"])
    m["true_state"] = np.where(m.RUL <= 10, "critical (RUL<=10)",
                        np.where(m.RUL <= horizon, f"at risk (RUL<={horizon})", "healthy"))
    tab = pd.crosstab(m["true_state"], m["action"])
    for col in ("CONTINUE", "INSPECT", "STOP"):
        if col not in tab.columns:
            tab[col] = 0
    return tab[["CONTINUE", "INSPECT", "STOP"]]
