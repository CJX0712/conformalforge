"""Regression conformal predictors.

  - SplitConformal : Vovk/Lei homoscedastic interval  [ŷ - q, ŷ + q]  (baseline)
  - RegFuse        : FLAGSHIP — locally weighted / normalized conformal (Lei &
                     Wasserman 2014). Local scale σ(x) estimated by kNN of
                     calibration residuals, then conformalised on normalised scores
                     R = |y - ŷ| / σ(x). Under heteroscedastic noise this yields
                     narrower *valid* intervals.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import ConformalResult
from .base import Regressor
from .scores import conformal_quantile


def _local_scales(X: np.ndarray, resids: np.ndarray, k: int) -> np.ndarray:
    """Mean absolute residual of the k nearest calibration neighbours per point."""
    # Pairwise squared distances (n x n). Small enough for demo sizes.
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Nearest neighbours (exclude self).
    order = np.argsort(d2, axis=1)
    knn = order[:, 1 : k + 1]
    return np.mean(resids[knn], axis=1)


class SplitConformal:
    name = "split_residual"

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        reg = Regressor().fit(train.X, train.y)
        yhat_cal = reg.predict(calib.X)
        yhat_test = reg.predict(test.X)
        r = np.abs(calib.y - yhat_cal)
        q = conformal_quantile(r, alpha)
        lower = yhat_test - q
        upper = yhat_test + q
        cover = float(np.mean((test.y >= lower) & (test.y <= upper)))
        width = float(np.mean(upper - lower))
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cover,
            mean_interval_width=width,
            lower=lower,
            upper=upper,
            n_calib=len(calib.y),
            quantile=q,
            notes="homoscedastic split conformal interval",
        )


class RegFuse:
    name = "reg_fuse"

    def __init__(self, k: int = 25):
        self.k = k

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        reg = Regressor().fit(train.X, train.y)
        Xc, yc = calib.X, calib.y
        Xt = test.X
        yhat_cal = reg.predict(Xc)
        yhat_test = reg.predict(Xt)
        resids = np.abs(yc - yhat_cal)
        # Local scales computed per block using ONLY calibration residuals (no leakage).
        sig_cal = _local_scales(Xc, resids, self.k)
        # sigma for test points: kNN within calibration only.
        diff = Xt[:, None, :] - Xc[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        knn = np.argsort(d2, axis=1)[:, : self.k]
        sig_test = np.mean(resids[knn], axis=1)
        # Normalised (exchangeable) scores on calibration.
        R_cal = resids / np.maximum(sig_cal, 1e-9)
        q = conformal_quantile(R_cal, alpha)
        lower = yhat_test - q * sig_test
        upper = yhat_test + q * sig_test
        cover = float(np.mean((test.y >= lower) & (test.y <= upper)))
        width = float(np.mean(upper - lower))
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cover,
            mean_interval_width=width,
            lower=lower,
            upper=upper,
            n_calib=len(calib.y),
            quantile=q,
            notes=f"locally weighted conformal (k={self.k})",
        )
