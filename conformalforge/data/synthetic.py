"""Synthetic data generators (classification + regression).

Generators are *deterministic given a seed*: a fresh ``np.random.default_rng`` is
seeded from ``seed`` so that two runs with the same configuration produce
identical partitions. Difficulty knobs (noise / hetero) let the demo find the
"sweet spot" where the base model is good-but-imperfect, which is exactly the
regime where conformal efficiency differences matter.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import Dataset


def _split_indices(rng: np.random.Generator, n: int, fracs) -> list[np.ndarray]:
    idx = rng.permutation(n)
    out = []
    start = 0
    for f in fracs:
        end = start + round(n * f)
        out.append(idx[start:end])
        start = end
    return out


class GaussianBlobGenerator:
    """Multi-class Gaussian blobs with controllable separation/noise."""

    def __init__(
        self,
        n_train: int = 4000,
        n_calib: int = 2000,
        n_test: int = 2000,
        n_classes: int = 5,
        d: int = 12,
        noise: float = 0.9,
        separation: float = 2.4,
    ):
        self.n_train = n_train
        self.n_calib = n_calib
        self.n_test = n_test
        self.n_classes = n_classes
        self.d = d
        self.noise = float(noise)
        self.separation = float(separation)

    def _centers(self, rng: np.random.Generator) -> np.ndarray:
        # Build n_classes orthonormal directions in R^d, each scaled for separation.
        raw = rng.standard_normal((self.d, self.n_classes))
        q, _ = np.linalg.qr(raw)  # q: (d, k) orthonormal columns
        scale = self.separation * np.linspace(0.6, 1.4, self.n_classes)
        return (q * scale[None, :]).T  # (k, d): one row per class

    def generate(self, seed: int) -> tuple[Dataset, Dataset, Dataset]:
        rng = np.random.default_rng(np.uint32(seed))
        centers = self._centers(rng)
        total = self.n_train + self.n_calib + self.n_test
        per_class = total // self.n_classes
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for c in range(self.n_classes):
            n_c = per_class + (1 if c < total % self.n_classes else 0)
            xc = centers[c] + rng.standard_normal((n_c, self.d)) * self.noise
            xs.append(xc)
            ys.append(np.full(n_c, c, dtype=np.int64))
        X = np.vstack(xs)
        y = np.concatenate(ys)
        # Shuffle globally but deterministically.
        perm = rng.permutation(X.shape[0])
        X, y = X[perm], y[perm]
        i_tr, i_cal, i_te = _split_indices(
            rng, X.shape[0], [self.n_train / total, self.n_calib / total, self.n_test / total]
        )
        return (
            Dataset(X[i_tr], y[i_tr]),
            Dataset(X[i_cal], y[i_cal]),
            Dataset(X[i_te], y[i_te]),
        )


class HeteroscedasticRegGenerator:
    """Regression with heteroscedastic (input-dependent) noise.

    ``y = f(x) + sigma(x) * eps`` where ``sigma(x)`` grows with ``||x||``. This is
    the regime where a *local-scale* (weighted) conformal predictor beats the
    plain homoscedastic interval.
    """

    def __init__(
        self,
        n_train: int = 4000,
        n_calib: int = 2000,
        n_test: int = 2000,
        d: int = 6,
        hetero: float = 1.5,
        signal: float = 1.0,
    ):
        self.n_train = n_train
        self.n_calib = n_calib
        self.n_test = n_test
        self.d = d
        self.hetero = float(hetero)
        self.signal = signal

    def _f(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Smooth, mildly nonlinear signal. w fixed by seed => deterministic.
        lin = X @ w
        return self.signal * (np.tanh(lin) + 0.3 * np.sin(2.0 * lin)).ravel()

    def _sigma(self, X: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.maximum((X**2).sum(axis=1), 1e-9))
        r = (r - r.min()) / (r.max() - r.min() + 1e-9)
        return 0.15 + self.hetero * r  # bigger radius -> bigger noise

    def sigma_of(self, X: np.ndarray) -> np.ndarray:
        """True local noise scale sigma(x) for a given feature matrix (for metrics)."""
        return self._sigma(np.asarray(X, dtype=np.float64))

    def generate(self, seed: int) -> tuple[Dataset, Dataset, Dataset]:
        rng = np.random.default_rng(np.uint32(seed))
        total = self.n_train + self.n_calib + self.n_test
        w = rng.standard_normal(self.d)
        X = rng.standard_normal((total, self.d))
        y_mean = self._f(X, w)
        sigma = self._sigma(X)
        # Re-seed for noise so signal/centers don't correlate with noise stream.
        rng2 = np.random.default_rng(np.uint32(seed) ^ 0x9E3779B9)
        y = y_mean + sigma * rng2.standard_normal(total)
        perm = rng.permutation(total)
        X, y = X[perm], y[perm]
        i_tr, i_cal, i_te = _split_indices(
            rng, total, [self.n_train / total, self.n_calib / total, self.n_test / total]
        )
        return (
            Dataset(X[i_tr], y[i_tr]),
            Dataset(X[i_cal], y[i_cal]),
            Dataset(X[i_te], y[i_te]),
        )


def make_generator(task: str, **kwargs):
    if task == "classification":
        return GaussianBlobGenerator(**kwargs)
    if task == "regression":
        return HeteroscedasticRegGenerator(**kwargs)
    raise DataError(f"unknown task {task!r}")
