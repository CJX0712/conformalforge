"""回归一致性预测区间测试（作者：晨星）。"""

from __future__ import annotations

import numpy as np

from conformal.regression import CQRFuseRegressor, CQRRegressor, SplitConformalRegressor
from core.seed import set_all
from data.generators import get_generator
from eval.metrics import interval_coverage, mean_interval_width


def _split(gen, seed, n=2000, cal=0.25):
    set_all(seed)
    X, y = gen.generate(seed)
    nt, nc = int(0.5 * n), int(cal * n)
    return (
        X[:nt],
        y[:nt],
        X[nt : nt + nc],
        y[nt : nt + nc],
        X[nt + nc :],
        y[nt + nc :],
    )


def test_split_conformal_regressor_valid():
    gen = get_generator("homoscedastic")
    Xtr, ytr, Xc, yc, Xt, yt = _split(gen, 7)
    m = SplitConformalRegressor(0.10).fit(Xtr, ytr).calibrate(Xc, yc)
    lo, hi = m.predict_interval(Xt)
    cov = interval_coverage(lo, hi, yt)
    assert 0.80 <= cov <= 1.0
    assert mean_interval_width(lo, hi) > 0


def test_cqr_regressor_valid():
    gen = get_generator("heteroscedastic")
    Xtr, ytr, Xc, yc, Xt, yt = _split(gen, 7)
    from sklearn.linear_model import QuantileRegressor

    m = CQRRegressor(
        0.10,
        QuantileRegressor(quantile=0.05, alpha=1e-3, solver="highs"),
        QuantileRegressor(quantile=0.95, alpha=1e-3, solver="highs"),
    )
    m.fit(Xtr, ytr).calibrate(Xc, yc)
    lo, hi = m.predict_interval(Xt)
    cov = interval_coverage(lo, hi, yt)
    assert 0.80 <= cov <= 1.0


def test_cqrfuse_valid_and_narrower_than_linear():
    gen = get_generator("heteroscedastic")
    Xtr, ytr, Xc, yc, Xt, yt = _split(gen, 7)
    from sklearn.linear_model import QuantileRegressor

    cqr_lin = CQRRegressor(
        0.10,
        QuantileRegressor(quantile=0.05, alpha=1e-3, solver="highs"),
        QuantileRegressor(quantile=0.95, alpha=1e-3, solver="highs"),
    )
    cqr_lin.fit(Xtr, ytr).calibrate(Xc, yc)
    lo_l, hi_l = cqr_lin.predict_interval(Xt)
    w_lin = mean_interval_width(lo_l, hi_l)

    fuse = CQRFuseRegressor(0.10).fit(Xtr, ytr).calibrate(Xc, yc)
    lo, hi = fuse.predict_interval(Xt)
    cov = interval_coverage(lo, hi, yt)
    w = mean_interval_width(lo, hi)
    assert 0.80 <= cov <= 1.0
    # 融合区间应不宽于 CQR-Linear（组合引理保证）
    assert w <= w_lin + 1e-6


def test_cqrfuse_deterministic():
    gen = get_generator("homoscedastic")
    set_all(7)
    Xtr, ytr, Xc, yc, Xt, _yt = _split(gen, 7)
    f1 = CQRFuseRegressor(0.10).fit(Xtr, ytr).calibrate(Xc, yc)
    lo1, hi1 = f1.predict_interval(Xt)
    set_all(7)
    f2 = CQRFuseRegressor(0.10).fit(Xtr, ytr).calibrate(Xc, yc)
    lo2, hi2 = f2.predict_interval(Xt)
    assert np.array_equal(lo1, lo2) and np.array_equal(hi1, hi2)
