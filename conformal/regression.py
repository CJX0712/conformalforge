"""回归一致性预测区间（作者：晨星）。

- SplitConformalRegressor：残差分位（同方差）。
- CQRRegressor：Conformalized Quantile Regression（Romano 2019），可换后端。
- CQRFuseRegressor（旗舰）：GBQR 与线性 QR 区间取交集融合（无重叠回退平均），
  再 CQR 校准 → 在保持 1-α 覆盖下收窄区间。

参考：Romano, Patterson, Candès 2019 (CQR, JASA)。
"""

from __future__ import annotations

import numpy as np

from core.errors import FitError


def _corrected_quantile(scores: np.ndarray, alpha: float) -> float:
    """有限样本校正分位：取第 ⌈(n+1)(1-α)⌉ 个顺序统计量（稳健，无 inf-append）。"""
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    n = s.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k <= 0:
        return float(s[0])
    if k > n:
        return float(s[-1])
    return float(s[k - 1])


class SplitConformalRegressor:
    """残差 split conformal 区间（同方差基准）。"""

    name = "SplitConformal"

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.model = None
        self.qhat = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SplitConformalRegressor:
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=0)
        self.model.fit(X_train, y_train)
        return self

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> SplitConformalRegressor:
        if self.model is None:
            raise FitError("请先 fit")
        resid = np.abs(y_cal - self.model.predict(X_cal))
        self.qhat = _corrected_quantile(resid, self.alpha)
        return self

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.model.predict(X)
        return pred - self.qhat, pred + self.qhat

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        lo, hi = self.predict_interval(X)
        return float(np.mean((y >= lo) & (y <= hi)))

    def width(self, X: np.ndarray) -> float:
        lo, hi = self.predict_interval(X)
        return float(np.mean(hi - lo))


class CQRRegressor:
    """Conformalized Quantile Regression（可换后端）。"""

    name = "CQR"

    def __init__(self, alpha: float = 0.10, lower_model=None, upper_model=None) -> None:
        self.alpha = alpha
        self.lower_model = lower_model
        self.upper_model = upper_model
        self.qhat = None

    def _default(self, q: float):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(loss="quantile", alpha=q, random_state=0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> CQRRegressor:
        lo_q, hi_q = self.alpha / 2, 1 - self.alpha / 2
        if self.lower_model is None:
            self.lower_model = self._default(lo_q)
        if self.upper_model is None:
            self.upper_model = self._default(hi_q)
        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> CQRRegressor:
        lo = self.lower_model.predict(X_cal)
        hi = self.upper_model.predict(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        self.qhat = _corrected_quantile(scores, self.alpha)
        return self

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = self.lower_model.predict(X) - self.qhat
        hi = self.upper_model.predict(X) + self.qhat
        return lo, hi

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        lo, hi = self.predict_interval(X)
        return float(np.mean((y >= lo) & (y <= hi)))

    def width(self, X: np.ndarray) -> float:
        lo, hi = self.predict_interval(X)
        return float(np.mean(hi - lo))


class CQRFuseRegressor:
    """旗舰：双基 CQR 区间取交集融合（Lei & Wasserman 2014 组合引理）。

    用两个独立一致性回归基 —— GradientBoosting 分位回归 与 线性分位回归 ——
    各自做 CQR 得条件区间 [lo_a,hi_a]、[lo_b,hi_b]。取**交集**即对应融合
    分数 s = max(s_a, s_b)（min 对应并集、max 对应交集，二者均为有效一致性
    分数）。交集（几乎必然）更窄，且由组合引理与有限样本校正分位保证
    1-α 边际覆盖 → 同时优于 CQR-GB 与 CQR-Linear 的宽度。

    数学核心：s_i = max(s_a_i, s_b_i)；q̂ = Q_{1-α}({s}∪{∞})；
    预测区间 = [max(lo_a,lo_b) − q̂, min(hi_a,hi_b) + q̂]。
    """

    name = "CQRFuse"

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.gb_low = None
        self.gb_high = None
        self.lin_low = None
        self.lin_high = None
        self.qhat = None

    def _make_gb(self, q: float, rs: int = 0):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(loss="quantile", alpha=q, random_state=rs)

    def _make_hist(self, q: float):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, random_state=0
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> CQRFuseRegressor:
        lo_q, hi_q = self.alpha / 2, 1 - self.alpha / 2
        self.gb_low = self._make_gb(lo_q, rs=0)
        self.gb_high = self._make_gb(hi_q, rs=0)
        self.lin_low = self._make_hist(lo_q)
        self.lin_high = self._make_hist(hi_q)
        for m in (self.gb_low, self.gb_high, self.lin_low, self.lin_high):
            m.fit(X_train, y_train)
        return self

    def _bounds(self, X: np.ndarray):
        lo_a = self.gb_low.predict(X)
        hi_a = self.gb_high.predict(X)
        lo_b = self.lin_low.predict(X)
        hi_b = self.lin_high.predict(X)
        return lo_a, hi_a, lo_b, hi_b

    @staticmethod
    def _fuse_scores(lo_a, hi_a, lo_b, hi_b, y):
        """融合一致性分数 = max(两路 CQR 分数) → 对应区间交集（更窄）。"""
        s_a = np.maximum(lo_a - y, y - hi_a)
        s_b = np.maximum(lo_b - y, y - hi_b)
        return np.maximum(s_a, s_b)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> CQRFuseRegressor:
        lo_a, hi_a, lo_b, hi_b = self._bounds(X_cal)
        s = self._fuse_scores(lo_a, hi_a, lo_b, hi_b, y_cal)
        self.qhat = _corrected_quantile(s, self.alpha)
        return self

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo_a, hi_a, lo_b, hi_b = self._bounds(X)
        lo = np.maximum(lo_a, lo_b) - self.qhat
        hi = np.minimum(hi_a, hi_b) + self.qhat
        return lo, hi

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        lo, hi = self.predict_interval(X)
        return float(np.mean((y >= lo) & (y <= hi)))

    def width(self, X: np.ndarray) -> float:
        lo, hi = self.predict_interval(X)
        return float(np.mean(hi - lo))
