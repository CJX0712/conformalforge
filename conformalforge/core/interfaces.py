"""Protocol (structural typing) interfaces for ConformalForge.

Modules communicate only through these abstract contracts, so any component can be
replaced or tested in isolation. This is the architectural backbone that keeps the
pipeline acyclic: ``cli -> pipeline -> {data, conformal, eval} -> core``.

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Array, ConformalResult, Dataset


@runtime_checkable
class DataGenerator(Protocol):
    """Produces a reproducible train/calib/test partition."""

    def generate(self, seed: int) -> tuple[Dataset, Dataset, Dataset]: ...


@runtime_checkable
class BaseLearner(Protocol):
    """A model fit on train, scoring calibration + test."""

    def fit(self, X: Array, y: Array) -> BaseLearner: ...

    def predict_proba(self, X: Array) -> Array:
        """Soft probabilities, shape (n, n_classes)."""

    def predict(self, X: Array) -> Array: ...


@runtime_checkable
class ConformalPredictor(Protocol):
    """Wraps a base learner + nonconformity score into a calibrated predictor."""

    name: str

    def fit_predict(
        self, train: Dataset, calib: Dataset, test: Dataset, alpha: float
    ) -> ConformalResult: ...

    def available(self) -> bool:
        """False => offline fallback must be used."""
        ...
