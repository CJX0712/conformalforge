"""Shared dataclasses / type aliases for ConformalForge.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass
class Dataset:
    """A single split of features ``X`` and labels ``y``."""

    X: Array
    y: Array

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


@dataclass
class ConformalResult:
    """One conformal prediction outcome.

    For classification ``sets`` is a list (per test point) of arrays of predicted
    class labels. For regression ``lower``/``upper`` bound the interval.
    """

    method: str
    alpha: float
    coverage: float
    # classification-only aggregates
    avg_set_size: float | None = None
    worst_class_gap: float | None = None
    min_class_coverage: float | None = None
    under_gap: float | None = None
    class_coverage: dict[int, float] | None = None
    # regression-only aggregates
    mean_interval_width: float | None = None
    # per-point predictions
    sets: list[Array] | None = field(default=None, repr=False)
    lower: Array | None = field(default=None, repr=False)
    upper: Array | None = field(default=None, repr=False)
    # diagnostics
    n_calib: int = 0
    quantile: float = float("nan")
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = {
            "method": self.method,
            "alpha": self.alpha,
            "coverage": self.coverage,
            "avg_set_size": self.avg_set_size,
            "worst_class_gap": self.worst_class_gap,
            "mean_interval_width": self.mean_interval_width,
            "n_calib": self.n_calib,
            "quantile": self.quantile,
            "notes": self.notes,
        }
        return d
