"""端到端管线测试（作者：晨星）。"""

from __future__ import annotations

from core.config import Config
from core.seed import set_all
from pipeline.pipeline import run_classification, run_regression


def test_run_classification_single_seed():
    cfg = Config(alpha=0.10)
    rep = run_classification(
        "gaussian_blobs",
        {"n_classes": 5, "n_features": 12, "separation": 3.0},
        cfg,
        seeds=[7],
        use_mapie=False,
    )
    methods = {r.method for r in rep.rows}
    assert "ConformalFuse" in methods
    assert "ConformalFuse-Grp" in methods
    assert "THR-RF" in methods
    # 多数方法应有效；至少旗舰有效
    fuse = [r for r in rep.rows if r.method == "ConformalFuse"]
    assert all(r.valid for r in fuse)
    # 每行覆盖合理
    assert all(0.5 <= r.coverage <= 1.0 for r in rep.rows)


def test_run_regression_single_seed():
    cfg = Config(alpha=0.10)
    rep = run_regression("heteroscedastic", {}, cfg, seeds=[7])
    methods = {r.method for r in rep.rows}
    assert "CQRFuse" in methods
    assert "CQR-Linear" in methods
    cqr = [r for r in rep.rows if r.method == "CQRFuse"]
    assert all(r.valid for r in cqr)
    assert all(r.mean_interval_width is not None for r in cqr)


def test_run_classification_deterministic():
    cfg = Config()
    set_all(7)
    r1 = run_classification(
        "gaussian_blobs", {"n_classes": 5}, cfg, seeds=[7], use_mapie=False
    )
    set_all(7)
    r2 = run_classification(
        "gaussian_blobs", {"n_classes": 5}, cfg, seeds=[7], use_mapie=False
    )
    c1 = next(r for r in r1.rows if r.method == "ConformalFuse")
    c2 = next(r for r in r2.rows if r.method == "ConformalFuse")
    assert c1.coverage == c2.coverage
    assert c1.avg_set_size == c2.avg_set_size
