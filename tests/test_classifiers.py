"""分类一致性预测器测试（作者：晨星）。"""

from __future__ import annotations

import numpy as np
import pytest

from conformal.classifiers import (
    ConformalFuseClassifier,
    FitError,
    MapieBaseline,
    SplitConformalClassifier,
    available_mapie,
)
from core.seed import set_all
from data.generators import get_generator
from eval.metrics import avg_set_size, coverage_of_sets


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


def test_split_conformal_thr_valid():
    gen = get_generator("gaussian_blobs", n_classes=5, n_features=12, separation=3.0)
    Xtr, ytr, Xc, yc, Xt, yt = _split(gen, 7)
    c = (
        SplitConformalClassifier("random_forest", "thr", 0.10)
        .fit(Xtr, ytr)
        .calibrate(Xc, yc)
    )
    sets = c.predict_set(Xt)
    assert len(sets) == len(yt)
    cov = coverage_of_sets(sets, yt)
    assert 0.80 <= cov <= 1.0
    # 集合尺寸可包含空集（THR 阈值下无类达标），非负即可
    assert avg_set_size(sets) >= 0.0


def test_split_conformal_aps_raps_run():
    gen = get_generator("gaussian_blobs", n_classes=5, n_features=12)
    Xtr, ytr, Xc, yc, Xt, yt = _split(gen, 7)
    for score in ("aps", "raps"):
        c = (
            SplitConformalClassifier("random_forest", score, 0.10)
            .fit(Xtr, ytr)
            .calibrate(Xc, yc)
        )
        sets = c.predict_set(Xt)
        assert len(sets) == len(yt)
        assert coverage_of_sets(sets, yt) > 0.7


def test_conformal_fuse_global_valid_and_deterministic():
    gen = get_generator("gaussian_blobs", n_classes=5, n_features=12, separation=3.0)
    Xtr, ytr, Xc, yc, Xt, yt = _split(gen, 7)
    set_all(7)
    f1 = ConformalFuseClassifier(alpha=0.10, score="thr", grouped=False, seed=7)
    f1.fit(Xtr, ytr).calibrate(Xc, yc)
    s1 = f1.predict_set(Xt)
    set_all(7)
    f2 = ConformalFuseClassifier(alpha=0.10, score="thr", grouped=False, seed=7)
    f2.fit(Xtr, ytr).calibrate(Xc, yc)
    s2 = f2.predict_set(Xt)
    assert coverage_of_sets(s1, yt) == coverage_of_sets(s2, yt)
    assert avg_set_size(s1) == avg_set_size(s2)
    assert f1.q_global_ is not None


def test_conformal_fuse_grouped_runs():
    gen = get_generator(
        "gaussian_blobs",
        n_classes=5,
        n_features=12,
        imbalance=(4.0, 1.0, 1.0, 1.0, 1.0),
    )
    Xtr, ytr, Xc, yc, Xt, yt = _split(gen, 42)
    f = ConformalFuseClassifier(alpha=0.10, score="thr", grouped=True, seed=42)
    f.fit(Xtr, ytr).calibrate(Xc, yc)
    sets = f.predict_set(Xt)
    assert len(sets) == len(yt)
    cov = coverage_of_sets(sets, yt)
    assert 0.80 <= cov <= 1.0
    # 分组模式应已估计逐类分位
    assert np.any(f.q_per_class_ < np.inf)


def test_fuse_fit_before_calibrate_raises():
    f = ConformalFuseClassifier()
    with pytest.raises(FitError):
        f.calibrate(np.zeros((5, 3)), np.zeros(5, dtype=int))


def test_mapie_baseline_optional():
    if not available_mapie():
        pytest.skip("mapie 未安装（可选依赖）")
    gen = get_generator("gaussian_blobs", n_classes=5, n_features=12)
    Xtr, ytr, _Xc, _yc, Xt, yt = _split(gen, 7)
    mp = MapieBaseline(alpha=0.10, seed=7)
    mp.fit(Xtr, ytr)
    sets = mp.predict_set(Xt)
    assert len(sets) == len(yt)
