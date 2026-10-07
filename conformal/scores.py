"""一致性分数（作者：晨星）。

每个分数都是「越大越不 conform」的有效一致性分数：在 (X,Y) 可交换下，
任意有限加权组合仍是有效一致性分数 → 多模型/多分数融合保持
有限样本覆盖保证（Vovk 一致性引理）。

参考：Romano, Sesia, Candès 2020 (APS/RAPS, NeurIPS);
      Vovk, Gammerman, Shafer 2005 (一致性引理)。
"""

from __future__ import annotations

import numpy as np

# 平滑：避免硬 0 概率导致 APS 累积瞬间到 1.0（标准 APS 修复，Romano 2020）。
# 平滑是确定性函数，不改变分数可交换性 → 覆盖保证仍成立。
_EPS = 1e-2


def _smooth(P: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    P = (1.0 - k * _EPS) * P + _EPS
    return P / P.sum(axis=1, keepdims=True)


def thr_per_class(P: np.ndarray) -> np.ndarray:
    """THR：score(y) = 1 - p(y)。朴素阈值分数。"""
    return 1.0 - P


def aps_per_class(P: np.ndarray) -> np.ndarray:
    """APS：score(y) = 按概率降序累加到类 y 的累积和（Romano 2020）。"""
    P = _smooth(P)
    n, k = P.shape
    order = np.argsort(-P, axis=1, kind="stable")
    cum = np.take_along_axis(P, order, axis=1).cumsum(axis=1)
    inv = np.empty((n, k), dtype=np.intp)
    inv[np.arange(n)[:, None], order] = np.arange(k)[None, :]
    return np.take_along_axis(cum, inv, axis=1)


def raps_per_class(P: np.ndarray, lam: float = 0.05, k_reg: int = 5) -> np.ndarray:
    """RAPS：APS + 集合尺寸正则项 λ·max(0, |S|-k_reg)（Romano 2020）。"""
    P = _smooth(P)
    n, k = P.shape
    aps = aps_per_class(P)
    order = np.argsort(-P, axis=1, kind="stable")
    cum_sorted = np.take_along_axis(P, order, axis=1).cumsum(axis=1)
    pen = np.zeros((n, k))
    for i in range(n):
        cs = cum_sorted[i]
        cnt = (cs[:, None] <= cs[None, :] + 1e-12).sum(axis=1)
        pen[i, order[i]] = lam * np.maximum(0.0, cnt.astype(float) - k_reg)
    return aps + pen


def calibration_quantile(scores_cal: np.ndarray, alpha: float) -> float:
    """有限样本校正分位（Romano 2020）：取第 ⌈(n+1)(1-α)⌉ 个顺序统计量。

    返回 q̂ 使 split conformal 预测集满足
        P(Y_{n+1} ∈ S(X_{n+1})) ≥ 1 - α    （分布无关、有限样本）
    直接取顺序统计量，避免 inf-append 在整数分位处产生 nan，数值更稳健。
    """
    s = np.sort(np.asarray(scores_cal, dtype=float).ravel())
    n = s.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k <= 0:
        return float(s[0])
    if k > n:
        return float(s[-1])
    return float(s[k - 1])


def correct_label_scores(per_class: np.ndarray, y: np.ndarray) -> np.ndarray:
    """取每个校准点的正确标签分数。"""
    return per_class[np.arange(len(y)), y]
