"""Base learners with sklearn (Tier-0) primary and pure-numpy offline fallback.

Critical leakage rule: the StandardScaler is fit on TRAIN ONLY and re-applied to
calibration/test. The base model is trained on TRAIN ONLY. Conformal scores are
computed on CALIBRATION only; the quantile is never touched by test data.

Author: 晨星
"""

from __future__ import annotations

import numpy as np


def _standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


class Classifier:
    """Soft classifier: sklearn LogisticRegression with numpy multinomial fallback."""

    def __init__(self) -> None:
        self._sk = None
        self._w = None
        self._b = None
        self._classes = None
        self._mu = None
        self._sd = None
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._classes = np.unique(y)
        self.n_classes = len(self._classes)
        Y = np.array([(y == c).astype(np.float64) for c in self._classes]).T
        self._mu, self._sd = _standardize_fit(X)
        Xs = _standardize_apply(X, self._mu, self._sd)
        try:
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(multi_class="multinomial", max_iter=2000, C=1.0)
            lr.fit(Xs, y)
            self._sk = lr
        except Exception:  # pragma: no cover - fallback path
            self._sk = None
            self._w, self._b = self._fit_numpy(Xs, Y)
        return self

    def _fit_numpy(self, Xs: np.ndarray, Y: np.ndarray):
        _, d = Xs.shape
        k = Y.shape[1]
        # One-vs-rest ridge logistic via Newton; capped iterations, robust.
        W = np.zeros((k, d))
        b = np.zeros(k)
        for j in range(k):
            w, bb = self._binary_newton(Xs, Y[:, j])
            W[j] = w
            b[j] = bb
        return W, b

    def _binary_newton(self, Xs, ybin, iters=200, lr=0.3):
        n, d = Xs.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(iters):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = Xs.T @ (p - ybin) / n + 1e-3 * w
            Hinv = 1.0 / (np.sum(p * (1 - p)) / n + 1e-3)
            w = w - lr * Hinv * g
            b = b - lr * np.mean(p - ybin)
        return w, b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xs = _standardize_apply(X, self._mu, self._sd)
        if self._sk is not None:
            return np.asarray(self._sk.predict_proba(Xs), dtype=np.float64)
        logits = Xs @ self._w.T + self._b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        idx = p.argmax(axis=1)
        return self._classes[idx]


class Regressor:
    """Ridge regressor: sklearn primary, numpy closed-form fallback."""

    def __init__(self) -> None:
        self._sk = None
        self._w = None
        self._b = None
        self._mu = None
        self._sd = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Regressor:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._mu, self._sd = _standardize_fit(X)
        Xs = _standardize_apply(X, self._mu, self._sd)
        try:
            from sklearn.linear_model import Ridge

            r = Ridge(alpha=1.0)
            r.fit(Xs, y)
            self._sk = r
        except Exception:  # pragma: no cover
            self._sk = None
            XtX = Xs.T @ Xs + 1.0 * np.eye(Xs.shape[1])
            self._w = np.linalg.solve(XtX, Xs.T @ y)
            self._b = y.mean() - self._w @ Xs.mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xs = _standardize_apply(X, self._mu, self._sd)
        if self._sk is not None:
            return np.asarray(self._sk.predict(Xs), dtype=np.float64)
        return Xs @ self._w + self._b
