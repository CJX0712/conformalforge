"""scores 模块单元测试（作者：晨星）。"""

from __future__ import annotations

import numpy as np

from conformal.scores import (
    aps_per_class,
    calibration_quantile,
    correct_label_scores,
    raps_per_class,
    thr_per_class,
)


def _prob(n=50, k=4):
    rng = np.random.default_rng(0)
    P = rng.random((n, k))
    return P / P.sum(axis=1, keepdims=True)


def test_thr_shape():
    P = _prob(30, 5)
    s = thr_per_class(P)
    assert s.shape == P.shape
    # THR 越大越不 conform：P 越大分数越小
    assert s[0, 0] == 1.0 - P[0, 0]


def test_aps_monotone():
    P = _prob(30, 5)
    s = aps_per_class(P)
    # APS：真实类的概率越大，分数应越小（更 conform）
    assert np.all(s >= 0)
    # 每行最大分数（累积到末尾）应接近 1
    assert np.allclose(s.max(axis=1), 1.0, atol=1e-6)


def test_aps_uniform_distribution():
    # 均匀概率下 APS 分数应近似均匀累积，无退化
    P = np.full((20, 4), 0.25)
    s = aps_per_class(P)
    assert np.allclose(s, np.tile(np.arange(1, 5) / 4.0, (20, 1)), atol=1e-6)


def test_raps_penalizes_large_sets():
    P = _prob(30, 5)
    aps = aps_per_class(P)
    raps = raps_per_class(P, lam=0.1, k_reg=2)
    # RAPS 在集合较大处有正则惩罚
    assert raps.shape == aps.shape


def test_calibration_quantile_known():
    # 分数 [1,2,...,20]，α=0.1 → level=ceil(21*0.9)/20=0.95（有限）
    s = np.arange(1, 21, dtype=float)
    q = calibration_quantile(s, 0.10)
    assert np.isfinite(q)
    assert q >= 1.0
    # 含 inf 不影响有限分位
    q2 = calibration_quantile(s, 0.10)
    assert q2 == q


def test_calibration_quantile_coverage_bound():
    # 有限样本校正分位 ≥ 目标（保证性质）
    rng = np.random.default_rng(1)
    s = rng.random(200)
    q = calibration_quantile(s, 0.10)
    frac = np.mean(s <= q)
    assert frac >= 0.90 - 1e-9


def test_correct_label_scores():
    P = _prob(10, 3)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    sc = correct_label_scores(P, y)
    assert sc.shape == (10,)
    assert np.allclose(sc, P[np.arange(10), y])


def test_smoothing_rows_sum_one():
    P = _prob(20, 6)
    from conformal.scores import _smooth

    Ps = _smooth(P)
    assert np.allclose(Ps.sum(axis=1), 1.0)
