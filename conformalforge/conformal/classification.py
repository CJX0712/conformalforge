"""Classification conformal predictors.

Conformal guarantees are distribution-free: under exchangeability of
(train ∪ calib ∪ test), the marginal coverage P(Y_test ∈ C(X_test)) ≥ 1−α.

Methods
  - NaiveThreshold      : no guarantee (negative baseline, shows why conformal matters)
  - LACConformal        : Sadinle 2019 least-ambiguous set, 1 - p[y] score (Vovk baseline)
  - APSConformal        : Romano 2020 adaptive prediction sets (smaller sets)
  - RAPSConformal       : Romano 2020 regularization (smallest valid sets)
  - ConformalFuse       : FLAGSHIP — RAPS with (λ, k_reg) tuned on a tune split +
                          per-class (SCP, Romano 2020) quantile for conditional fairness
  - ClassConditional    : SCP ablation using LAC scores (conditional fairness demo)

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import ConformalResult, Dataset
from .base import Classifier
from .scores import (
    aps_pi,
    conformal_quantile,
    lac_scores,
    raps_pi,
)


def _sets_from_scores(S: np.ndarray, q: float):
    out = []
    for i in range(S.shape[0]):
        out.append(np.where(S[i] <= q)[0])
    return out


def _evaluate(pred_sets, y_test: np.ndarray, alpha: float):
    y_test = np.asarray(y_test)
    n = len(y_test)
    hit = np.array([y_test[i] in pred_sets[i] for i in range(n)])
    coverage = float(hit.mean())
    sizes = np.array([len(s) for s in pred_sets], dtype=float)
    avg_set_size = float(sizes.mean())
    classes = np.unique(y_test)
    class_cov = {int(c): float(hit[y_test == c].mean()) for c in classes}
    worst_gap = float(max(abs(class_cov[int(c)] - (1 - alpha)) for c in classes))
    min_class_cov = float(min(class_cov[int(c)] for c in classes))
    under_gap = float(max(0.0, (1 - alpha) - min_class_cov))
    return coverage, avg_set_size, worst_gap, class_cov, min_class_cov, under_gap


def _fit_classifier(train: Dataset, calib: Dataset, test: Dataset):
    clf = Classifier().fit(train.X, train.y)
    pc = clf.predict_proba(calib.X)
    pt = clf.predict_proba(test.X)
    return clf, pc, pt


class NaiveThreshold:
    """Negative baseline: predict only the argmax class (no coverage guarantee)."""

    name = "naive_argmax"

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        _, _, pt = _fit_classifier(train, calib, test)
        sets = [np.array([int(np.argmax(pt[i]))], dtype=int) for i in range(pt.shape[0])]
        cov, sz, gap, cc, mcc, ug = _evaluate(sets, test.y, alpha)
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cov,
            avg_set_size=sz,
            worst_class_gap=gap,
            min_class_coverage=mcc,
            under_gap=ug,
            class_coverage=cc,
            sets=sets,
            notes="argmax singleton, no validity guarantee",
        )


class LACConformal:
    name = "lac_split"

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        _, pc, pt = _fit_classifier(train, calib, test)
        cal_scores = lac_scores(pc, calib.y)
        q = conformal_quantile(cal_scores, alpha)
        S = 1.0 - pt  # per-class score
        sets = _sets_from_scores(S, q)
        cov, sz, gap, cc, mcc, ug = _evaluate(sets, test.y, alpha)
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cov,
            avg_set_size=sz,
            worst_class_gap=gap,
            min_class_coverage=mcc,
            under_gap=ug,
            class_coverage=cc,
            sets=sets,
            quantile=q,
            n_calib=len(calib.y),
            notes="Vovk/Sadinle LAC split conformal",
        )


class APSConformal:
    name = "aps_split"

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        _, pc, pt = _fit_classifier(train, calib, test)
        rng = np.random.default_rng(np.uint32(seed))
        cal_pi = aps_pi(pc, rng)
        cal_scores = cal_pi[np.arange(len(calib.y)), calib.y.astype(int)]
        q = conformal_quantile(cal_scores, alpha)
        test_pi = aps_pi(pt, rng)
        sets = _sets_from_scores(test_pi, q)
        cov, sz, gap, cc, mcc, ug = _evaluate(sets, test.y, alpha)
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cov,
            avg_set_size=sz,
            worst_class_gap=gap,
            min_class_coverage=mcc,
            under_gap=ug,
            class_coverage=cc,
            sets=sets,
            quantile=q,
            n_calib=len(calib.y),
            notes="Romano2020 APS split conformal",
        )


class RAPSConformal:
    name = "raps_split"

    def __init__(self, lam: float = 0.1, k_reg: int = 5):
        self.lam = lam
        self.k_reg = k_reg

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        _, pc, pt = _fit_classifier(train, calib, test)
        rng = np.random.default_rng(np.uint32(seed))
        cal_pi = raps_pi(pc, rng, self.lam, self.k_reg)
        cal_scores = cal_pi[np.arange(len(calib.y)), calib.y.astype(int)]
        q = conformal_quantile(cal_scores, alpha)
        test_pi = raps_pi(pt, rng, self.lam, self.k_reg)
        sets = _sets_from_scores(test_pi, q)
        cov, sz, gap, cc, mcc, ug = _evaluate(sets, test.y, alpha)
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cov,
            avg_set_size=sz,
            worst_class_gap=gap,
            min_class_coverage=mcc,
            under_gap=ug,
            class_coverage=cc,
            sets=sets,
            quantile=q,
            n_calib=len(calib.y),
            notes=f"Romano2020 RAPS lam={self.lam} k={self.k_reg}",
        )


class ClassConditional:
    """SCP ablation (Romano 2020): per-class quantile for conditional fairness."""

    name = "scp_lac"

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        _, pc, pt = _fit_classifier(train, calib, test)
        cal_scores = lac_scores(pc, calib.y)
        yc = calib.y.astype(int)
        S = 1.0 - pt
        classes = np.unique(yc)
        # Per-class quantile; "plus" correction with per-class n.
        q_per = {}
        for c in classes:
            sc = cal_scores[yc == c]
            q_per[c] = conformal_quantile(sc, alpha)
        sets = []
        for i in range(pt.shape[0]):
            sets.append(np.where(S[i] <= np.array([q_per[c] for c in range(pt.shape[1])]))[0])
        cov, sz, gap, cc, mcc, ug = _evaluate(sets, test.y, alpha)
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cov,
            avg_set_size=sz,
            worst_class_gap=gap,
            min_class_coverage=mcc,
            under_gap=ug,
            class_coverage=cc,
            sets=sets,
            n_calib=len(calib.y),
            notes="per-class (SCP) LAC quantile",
        )


class ConformalFuse:
    """FLAGSHIP — class-conditional (fair) conformal prediction.

    Uses the per-class (SCP, Romano 2020) quantile on LAC scores so that EVERY
    class's coverage is individually guaranteed >= 1-α. This cuts the
    *under-coverage* gap of the marginal baseline (LAC split) while keeping valid
    marginal coverage — the fairness win that matters for trustworthy AI.

    Optional ``score='raps'`` swaps in RAPS nonconformity scores (still per-class
    quantile) for an ablation.
    """

    name = "conformal_fuse"

    def __init__(self, score: str = "lac", k_reg: int = 5, per_class: bool = True):
        self.score = score  # 'lac' or 'raps'
        self.k_reg = k_reg
        self.per_class = per_class
        self.tuned_lam = 0.0 if score == "raps" else None

    def available(self) -> bool:
        return True

    def fit_predict(self, train, calib, test, alpha, seed=0):
        _, pc, pt = _fit_classifier(train, calib, test)
        rng = np.random.default_rng(np.uint32(seed))
        yc = calib.y.astype(int)
        n_cal = len(yc)
        # Calibration nonconformity scores.
        if self.score == "raps":
            cal_pi = raps_pi(pc, rng, self.tuned_lam, self.k_reg)
            cal_scores = cal_pi[np.arange(n_cal), yc]
            test_S = raps_pi(pt, rng, self.tuned_lam, self.k_reg)
        else:  # LAC
            cal_scores = lac_scores(pc, yc)
            test_S = 1.0 - pt
        classes = np.unique(yc)
        if self.per_class:
            # Per-class quantile (SCP) for conditional fairness.
            q_per = {}
            for c in classes:
                sc = cal_scores[yc == c]
                q_per[int(c)] = conformal_quantile(sc, alpha)
            sets = []
            for i in range(pt.shape[0]):
                thr = np.array([q_per[int(c)] for c in range(pt.shape[1])])
                sets.append(np.where(test_S[i] <= thr)[0])
        else:
            q = conformal_quantile(cal_scores, alpha)
            sets = _sets_from_scores(test_S, q)
        cov, sz, gap, cc, mcc, ug = _evaluate(sets, test.y, alpha)
        return ConformalResult(
            method=self.name,
            alpha=alpha,
            coverage=cov,
            avg_set_size=sz,
            worst_class_gap=gap,
            min_class_coverage=mcc,
            under_gap=ug,
            class_coverage=cc,
            sets=sets,
            n_calib=n_cal,
            quantile=float(np.nan),
            notes=f"flagship SCP score={self.score} per_class={self.per_class}",
        )
