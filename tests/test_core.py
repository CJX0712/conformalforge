"""core 模块单元测试（作者：晨星）。"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Config
from core.errors import ConfigError, ConformalError, DependencyError, FitError
from core.seed import set_all
from core.types import BenchmarkReport, Result, Split


def test_set_all_deterministic():
    set_all(42)
    a = np.random.default_rng(0).random(5)
    set_all(42)
    b = np.random.default_rng(0).random(5)
    assert np.array_equal(a, b)


def test_set_all_changes_stream():
    set_all(1)
    x = np.random.rand(3)
    set_all(2)
    y = np.random.rand(3)
    assert not np.array_equal(x, y)


def test_config_default_valid():
    cfg = Config()
    cfg.validate()  # 不应抛异常


def test_config_invalid_alpha():
    with pytest.raises(ConfigError):
        Config(alpha=1.5).validate()
    with pytest.raises(ConfigError):
        Config(alpha=-0.1).validate()


def test_config_invalid_tol():
    with pytest.raises(ConfigError):
        Config(coverage_tol=0.5).validate()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ENV_CONFORMAL_ALPHA", "0.05")
    cfg = Config.from_env()
    assert cfg.alpha == 0.05


def test_config_from_env_bad(monkeypatch):
    monkeypatch.setenv("ENV_CONFORMAL_ALPHA", "not_a_float")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_types_aggregate_classification():
    rep = BenchmarkReport(system="X", domain="d", task="classification", seeds=[7])
    for s in (7, 8):
        rep.rows.append(
            Result(
                method="M",
                task="classification",
                seed=s,
                coverage=0.9,
                coverage_target=0.9,
                coverage_gap=0.0,
                valid=True,
                n_test=10,
                avg_set_size=2.0,
            )
        )
    agg = rep.aggregate()
    assert agg["coverage_mean"] == 0.9
    assert agg["avg_set_size_mean"] == 2.0
    assert agg["n_valid"] == 2


def test_types_aggregate_regression():
    rep = BenchmarkReport(system="X", domain="d", task="regression", seeds=[7])
    rep.rows.append(
        Result(
            method="M",
            task="regression",
            seed=7,
            coverage=0.9,
            coverage_target=0.9,
            coverage_gap=0.0,
            valid=True,
            n_test=10,
            mean_interval_width=1.5,
        )
    )
    agg = rep.aggregate()
    assert agg["mean_interval_width_mean"] == 1.5


def test_errors_hierarchy():
    assert issubclass(DependencyError, ConformalError)
    assert issubclass(FitError, ConformalError)


def test_split_construction():
    X = np.zeros((10, 3))
    y = np.arange(10)
    sp = Split(X_train=X, y_train=y, X_cal=X, y_cal=y, X_test=X, y_test=y)
    assert sp.X_train.shape == X.shape
