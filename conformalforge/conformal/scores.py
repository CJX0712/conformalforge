"""Nonconformity score functions for classification conformal prediction.

All scores obey the exchangeability assumption: they are computed identically on
the calibration set (to obtain the quantile) and on test points (to build sets).
References
  - LAC: Sadinle, Lei & Wasserman (2019), "Least Ambiguous Set-Valued Classifiers".
  - APS / RAPS: Romano, Sesia & Candès (2020), "Classification with Valid and
    Adaptive Coverage" (APS) and Romano, Stanton & Candès (2020) "On the use of
    regularization in conformal prediction" (RAPS).
  - Quantile lemma: Vovk, Gammerman & Shafer (2005).

Author: 晨星
"""

from __future__ import annotations

import numpy as np


def quantile_level(alpha: float, n: int, plus: bool = True) -> float:
    """Target level for the empirical quantile (finite-sample "plus" correction)."""
    if plus:
        return min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return 1.0 - alpha


def conformal_quantile(scores: np.ndarray, alpha: float, plus: bool = True) -> float:
    """Empirical (order-statistic) conformal quantile.

    Split conformal's finite-sample validity REQUIRES the discrete order statistic
    q̂ = the ``ceil((n+1)(1-α))``-th smallest score, NOT an interpolated quantile.
    Linear interpolation (np.quantile default) can select a value below the
    required order statistic and silently breaks the coverage guarantee.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    m = int(np.ceil((n + 1) * (1.0 - alpha))) if plus else int(np.ceil(n * (1.0 - alpha)))
    m = min(max(m, 1), n)
    return float(s[m - 1])


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least Ambiguous Set-Valued score: 1 - p[y]."""
    probs = np.asarray(probs, dtype=np.float64)
    return 1.0 - probs[np.arange(len(y)), np.asarray(y, dtype=int)]


def _cumulative_with_randomization(
    probs: np.ndarray, rng: np.random.Generator, penalty: np.ndarray | None = None
) -> np.ndarray:
    """Return per-class cumulative nonconformity ``pi(c)``.

    ``probs`` shape (n, k). Classes are sorted descending by probability; the
    cumulative sum up to and including each class is taken, plus a small uniform
    randomization ``U * p[c]`` (Romano 2020). ``penalty[c]`` (e.g. RAPS) is added
    to the cumulative before randomization.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n, _ = probs.shape
    order = np.argsort(-probs, axis=1)  # descending
    sorted_p = np.take_along_axis(probs, order, axis=1)
    U = rng.random((n, 1))
    cum = np.cumsum(sorted_p, axis=1)  # inclusive cumulative
    # APS: score for class at rank r = (cum up to r-1) + U * p_r  (exclusive cum + U*p).
    # Using inclusive cum: pi = cum - sorted_p + U * sorted_p.
    pi = cum - sorted_p + U * sorted_p
    if penalty is not None:
        penalty_sorted = np.take_along_axis(np.asarray(penalty, dtype=np.float64), order, axis=1)
        pi = pi + penalty_sorted
    # Map back to original class indices.
    out = np.empty_like(pi)
    # out[i, order[i,j]] = pi[i,j]
    rows = np.arange(n)[:, None]
    out[rows, order] = pi
    return out


def aps_scores(probs: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pi = _cumulative_with_randomization(probs, rng, penalty=None)
    return pi[np.arange(len(y)), np.asarray(y, dtype=int)]


def raps_scores(
    probs: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    lam: float,
    k_reg: int,
) -> np.ndarray:
    pi = raps_pi(probs, rng, lam, k_reg)
    return pi[np.arange(len(y)), np.asarray(y, dtype=int)]


def aps_pi(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Full (n, k) per-class APS cumulative nonconformity matrix (no penalty)."""
    return _cumulative_with_randomization(probs, rng, penalty=None)


def raps_pi(probs: np.ndarray, rng: np.random.Generator, lam: float, k_reg: int) -> np.ndarray:
    """Full (n, k) per-class RAPS cumulative nonconformity matrix."""
    probs = np.asarray(probs, dtype=np.float64)
    n, k = probs.shape
    ranks = np.tile(np.arange(1, k + 1), (n, 1))  # 1-indexed position
    penalty = lam * np.maximum(0.0, ranks - k_reg)
    return _cumulative_with_randomization(probs, rng, penalty=penalty)
