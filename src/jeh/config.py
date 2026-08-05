"""Global configuration: paths, seeds, column names and policy constants.

Everything that the notebook, the experiment driver and the Streamlit app need
to agree on lives here so that the three stay in sync by construction.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

for _d in (ARTIFACT_DIR, FIGURE_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix every RNG we rely on. Called at the top of the notebook and by the
    experiment driver so a top-to-bottom run is deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------
# Column schema (Section 3.2 of the brief) -- files carry no header row.
# --------------------------------------------------------------------------
ID_COL = "engine_id"
CYCLE_COL = "cycle"
OP_COLS = [f"operational_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLS = [ID_COL, CYCLE_COL] + OP_COLS + SENSOR_COLS

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Number of true operating conditions per subset (readme.txt).
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SUBSET_ROLE = {
    "FD001": "Stage 1 - foundation (1 condition, 1 fault)",
    "FD002": "Stage 2 alternative - condition shift (6 conditions, 1 fault)",
    "FD003": "Stage 2 - multi-fault (1 condition, 2 faults)",
    "FD004": "Bonus - combined challenge (6 conditions, 2 faults)",
}

# --------------------------------------------------------------------------
# Task definition
# --------------------------------------------------------------------------
HORIZONS = (10, 20, 30)

#: Candidate piecewise RUL caps; the winner is selected on VALIDATION engines.
RUL_CAP_GRID = (60, 70, 80, 90, 110, 125, 130, 150)
DEFAULT_RUL_CAP = 125

#: Trailing causal window lengths used by the window feature block.
WINDOW_LENGTHS = (5, 15, 30)
PRIMARY_WINDOW = 30

#: Fraction of the *official* training engines held out as validation engines.
VAL_ENGINE_FRACTION = 0.30

#: "Healthy" region for unsupervised fitting: the first N cycles of TRAINING
#: engines only. No test information and no failure label is used.
HEALTHY_HEAD_CYCLES = 30


# --------------------------------------------------------------------------
# Cost policy (Task 6). c_miss > c_late > c_early.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CostPolicy:
    """Asymmetric maintenance cost.

    A missed warning grounds an engine unexpectedly (in-flight shutdown risk),
    a late warning compresses the maintenance slot, an early warning wastes
    inspection capacity. Units are arbitrary "maintenance credits" but the
    ordering c_miss >> c_late > c_early is the operational statement.
    """

    c_miss: float = 200.0        # per engine that fails with no prior alert
    c_late: float = 5.0          # per cycle of shortfall inside the action window
    c_early: float = 1.0         # per cycle of premature alerting
    target_horizon: int = 20     # desired action window h (cycles)

    def as_dict(self) -> dict:
        return asdict(self)


COST_POLICY = CostPolicy()

#: Row-level classification cost used to pick decision thresholds on validation
#: engines. A false negative near failure is far worse than a false alarm.
FN_FP_RATIO = 20.0

# --------------------------------------------------------------------------
# Persistence rule for "first persistent alert" (Task 6).
# --------------------------------------------------------------------------
PERSIST_M = 3   # alerts required ...
PERSIST_N = 5   # ... within the last n cycles

# --------------------------------------------------------------------------
# Split-conformal miscoverage level for the RUL prediction interval.
# --------------------------------------------------------------------------
CONFORMAL_ALPHA = 0.10   # -> nominal 90% interval

MODEL_VERSION = "1.0.0"


@dataclass
class RunConfig:
    """Everything that identifies one experiment run."""

    subset: str = "FD001"
    seed: int = SEED
    val_fraction: float = VAL_ENGINE_FRACTION
    window_lengths: tuple = WINDOW_LENGTHS
    primary_window: int = PRIMARY_WINDOW
    rul_cap: int | None = None            # None -> tuned on validation
    condition_aware: bool = True          # regime-wise normalisation
    use_op_settings: bool = True
    use_trend_features: bool = True
    healthy_head_cycles: int = HEALTHY_HEAD_CYCLES
    horizons: tuple = HORIZONS
    conformal_alpha: float = CONFORMAL_ALPHA
    cost: CostPolicy = field(default_factory=lambda: COST_POLICY)
    tag: str = ""                          # suffix for ablation runs

    @property
    def name(self) -> str:
        return f"{self.subset}{('_' + self.tag) if self.tag else ''}"

    @property
    def artifact_dir(self) -> Path:
        d = ARTIFACT_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def as_dict(self) -> dict:
        d = asdict(self)
        d["window_lengths"] = list(self.window_lengths)
        d["horizons"] = list(self.horizons)
        return d
