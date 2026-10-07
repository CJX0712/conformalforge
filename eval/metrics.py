"""评估指标（作者：晨星）。

所有指标从真实运行输出派生；接口语义统一：覆盖越大越安全，集合/区间越小越高效。
"""

from __future__ import annotations

import numpy as np


def coverage_of_sets(sets: list[np.ndarray], y: np.ndarray) -> float:
    return float(np.mean([int(y[i] in sets[i]) for i in range(len(y))]))


def avg_set_size(sets: list[np.ndarray]) -> float:
    return float(np.mean([len(s) for s in sets]))


def max_set_size(sets: list[np.ndarray]) -> int:
    return int(max((len(s) for s in sets), default=0))


def conditional_coverage(sets: list[np.ndarray], y: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        cov = np.mean([int(y[i] in sets[i]) for i in idx])
        out[int(c)] = float(cov)
    return out


def interval_coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def conditional_coverage_gap(
    sets: list[np.ndarray], y: np.ndarray, target: float = 0.90
) -> float:
    """逐类覆盖与目标的绝对偏差最大值（公平性/分组一致性核心指标）。"""
    cond = conditional_coverage(sets, y)
    if not cond:
        return float("nan")
    return float(max(abs(v - target) for v in cond.values()))


def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def median_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.median(hi - lo))
