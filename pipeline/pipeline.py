"""端到端基准管线（作者：晨星）。

cli → pipeline → {data, conformal, eval} → core，单向无环。
每个数字来自真实运行；≥3 seeds 报 mean±std；覆盖不达标标记为 invalid。
"""

from __future__ import annotations

import time

from conformal.classifiers import (
    ConformalFuseClassifier,
    MapieBaseline,
    SplitConformalClassifier,
    available_mapie,
)
from conformal.regression import CQRFuseRegressor, CQRRegressor, SplitConformalRegressor
from core.config import Config
from core.seed import set_all
from core.types import BenchmarkReport, Result, Split
from data.generators import get_generator
from eval.metrics import (
    avg_set_size,
    conditional_coverage,
    coverage_of_sets,
    interval_coverage,
    max_set_size,
    mean_interval_width,
    median_interval_width,
)


def _split_data(X, y, cal_ratio: float = 0.20):
    n = X.shape[0]
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    return Split(
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_cal=X[n_train : n_train + n_cal],
        y_cal=y[n_train : n_train + n_cal],
        X_test=X[n_train + n_cal :],
        y_test=y[n_train + n_cal :],
    )


# ----------------------------- 分类 -----------------------------
CLASS_BASELINE = [
    ("THR-RF", "random_forest", "thr"),
    ("APS-RF", "random_forest", "aps"),
    ("RAPS-RF", "random_forest", "raps"),
    ("RAPS-LR", "logistic_reg", "raps"),
    ("RAPS-GB", "gradient_boosting", "raps"),
]


def run_classification(
    gen_name: str,
    gen_kw: dict | None = None,
    config: Config | None = None,
    seeds: list[int] | None = None,
    use_mapie: bool = False,
) -> BenchmarkReport:
    config = config or Config()
    config.validate()
    seeds = seeds or [config.seed]
    gen = get_generator(gen_name, **(gen_kw or {}))
    report = BenchmarkReport(
        system="ConformalForge",
        domain="conformal-prediction",
        task="classification",
        seeds=list(seeds),
    )

    for method_name, model_name, score in CLASS_BASELINE:
        for seed in seeds:
            set_all(seed)
            X, y = gen.generate(seed)
            sp = _split_data(X, y, config.calibration_ratio)
            t0 = time.perf_counter()
            c = SplitConformalClassifier(
                model_name=model_name,
                score=score,
                alpha=config.alpha,
                lam=config.raps_lambda,
                k_reg=config.raps_k_reg,
            )
            c.fit(sp.X_train, sp.y_train).calibrate(sp.X_cal, sp.y_cal)
            sets = c.predict_set(sp.X_test)
            cov = coverage_of_sets(sets, sp.y_test)
            ass = avg_set_size(sets)
            cond = conditional_coverage(sets, sp.y_test)
            elapsed = time.perf_counter() - t0
            report.rows.append(
                Result(
                    method=method_name,
                    task="classification",
                    seed=seed,
                    coverage=cov,
                    coverage_target=1 - config.alpha,
                    coverage_gap=abs(cov - (1 - config.alpha)),
                    valid=cov >= (1 - config.alpha) - config.coverage_tol,
                    n_test=len(sp.y_test),
                    avg_set_size=ass,
                    max_set_size=max_set_size(sets),
                    conditional_coverage=cond,
                    elapsed_sec=elapsed,
                )
            )

    # 旗舰 ConformalFuse
    for seed in seeds:
        set_all(seed)
        X, y = gen.generate(seed)
        sp = _split_data(X, y, config.calibration_ratio)
        t0 = time.perf_counter()
        fuse = ConformalFuseClassifier(
            model_names=("random_forest", "logistic_reg", "gradient_boosting"),
            alpha=config.alpha,
            score="thr",
            grouped=False,
            seed=seed,
        )
        fuse.fit(sp.X_train, sp.y_train).calibrate(sp.X_cal, sp.y_cal)
        sets = fuse.predict_set(sp.X_test)
        cov = coverage_of_sets(sets, sp.y_test)
        elapsed = time.perf_counter() - t0
        report.rows.append(
            Result(
                method="ConformalFuse",
                task="classification",
                seed=seed,
                coverage=cov,
                coverage_target=1 - config.alpha,
                coverage_gap=abs(cov - (1 - config.alpha)),
                valid=cov >= (1 - config.alpha) - config.coverage_tol,
                n_test=len(sp.y_test),
                avg_set_size=avg_set_size(sets),
                max_set_size=max_set_size(sets),
                conditional_coverage=conditional_coverage(sets, sp.y_test),
                elapsed_sec=elapsed,
                extra={
                    "grouped": fuse.grouped,
                    "score": fuse.score,
                    "q_global": round(float(fuse.q_global_), 4),
                    "max_cond_gap": round(
                        float(
                            max(
                                abs(v - (1 - config.alpha))
                                for v in conditional_coverage(sets, sp.y_test).values()
                            )
                        ),
                        4,
                    ),
                },
            )
        )

    # 旗舰（分组模式）：逐类公平性（conditional coverage, Romano 2020）
    for seed in seeds:
        set_all(seed)
        X, y = gen.generate(seed)
        sp = _split_data(X, y, config.calibration_ratio)
        t0 = time.perf_counter()
        fuse_g = ConformalFuseClassifier(
            model_names=("random_forest", "logistic_reg", "gradient_boosting"),
            alpha=config.alpha,
            score="thr",
            grouped=True,
            seed=seed,
        )
        fuse_g.fit(sp.X_train, sp.y_train).calibrate(sp.X_cal, sp.y_cal)
        sets = fuse_g.predict_set(sp.X_test)
        cov = coverage_of_sets(sets, sp.y_test)
        elapsed = time.perf_counter() - t0
        report.rows.append(
            Result(
                method="ConformalFuse-Grp",
                task="classification",
                seed=seed,
                coverage=cov,
                coverage_target=1 - config.alpha,
                coverage_gap=abs(cov - (1 - config.alpha)),
                valid=cov >= (1 - config.alpha) - config.coverage_tol,
                n_test=len(sp.y_test),
                avg_set_size=avg_set_size(sets),
                max_set_size=max_set_size(sets),
                conditional_coverage=conditional_coverage(sets, sp.y_test),
                elapsed_sec=elapsed,
                extra={
                    "grouped": fuse_g.grouped,
                    "score": fuse_g.score,
                    "max_cond_gap": round(
                        float(
                            max(
                                abs(v - (1 - config.alpha))
                                for v in conditional_coverage(sets, sp.y_test).values()
                            )
                        ),
                        4,
                    ),
                },
            )
        )

    # 可选 MAPIE（顶级开源对照）
    if use_mapie and available_mapie():
        for seed in seeds:
            set_all(seed)
            X, y = gen.generate(seed)
            sp = _split_data(X, y, config.calibration_ratio)
            try:
                t0 = time.perf_counter()
                mp = MapieBaseline(alpha=config.alpha, seed=seed)
                mp.fit(sp.X_train, sp.y_train)
                sets = mp.predict_set(sp.X_test)
                cov = coverage_of_sets(sets, sp.y_test)
                elapsed = time.perf_counter() - t0
                report.rows.append(
                    Result(
                        method="Mapie(RAPS)",
                        task="classification",
                        seed=seed,
                        coverage=cov,
                        coverage_target=1 - config.alpha,
                        coverage_gap=abs(cov - (1 - config.alpha)),
                        valid=cov >= (1 - config.alpha) - config.coverage_tol,
                        n_test=len(sp.y_test),
                        avg_set_size=avg_set_size(sets),
                        max_set_size=max_set_size(sets),
                        elapsed_sec=elapsed,
                    )
                )
            except Exception as e:  # noqa: BLE001
                report.rows.append(
                    Result(
                        method="Mapie(RAPS)",
                        task="classification",
                        seed=seed,
                        coverage=float("nan"),
                        coverage_target=1 - config.alpha,
                        coverage_gap=float("nan"),
                        valid=False,
                        n_test=0,
                        elapsed_sec=0.0,
                        extra={"error": str(e)},
                    )
                )
    return report


# ----------------------------- 回归 -----------------------------
def run_regression(
    gen_name: str,
    gen_kw: dict | None = None,
    config: Config | None = None,
    seeds: list[int] | None = None,
) -> BenchmarkReport:
    config = config or Config()
    config.validate()
    seeds = seeds or [config.seed]
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import QuantileRegressor

    gen = get_generator(gen_name, **(gen_kw or {}))
    report = BenchmarkReport(
        system="ConformalForge",
        domain="conformal-prediction",
        task="regression",
        seeds=list(seeds),
    )

    def _cqr(method_tag, lo_model, hi_model):
        for seed in seeds:
            set_all(seed)
            X, y = gen.generate(seed)
            sp = _split_data(X, y, config.calibration_ratio)
            t0 = time.perf_counter()
            m = CQRRegressor(
                alpha=config.alpha, lower_model=lo_model, upper_model=hi_model
            )
            m.fit(sp.X_train, sp.y_train).calibrate(sp.X_cal, sp.y_cal)
            lo, hi = m.predict_interval(sp.X_test)
            cov = interval_coverage(lo, hi, sp.y_test)
            elapsed = time.perf_counter() - t0
            report.rows.append(
                Result(
                    method=method_tag,
                    task="regression",
                    seed=seed,
                    coverage=cov,
                    coverage_target=1 - config.alpha,
                    coverage_gap=abs(cov - (1 - config.alpha)),
                    valid=cov >= (1 - config.alpha) - config.coverage_tol,
                    n_test=len(sp.y_test),
                    mean_interval_width=mean_interval_width(lo, hi),
                    median_interval_width=median_interval_width(lo, hi),
                    elapsed_sec=elapsed,
                )
            )

    # 基线
    lin = lambda: QuantileRegressor(
        quantile=config.alpha / 2, alpha=1e-3, solver="highs"
    )
    lin_hi = lambda: QuantileRegressor(
        quantile=1 - config.alpha / 2, alpha=1e-3, solver="highs"
    )
    gb = lambda: GradientBoostingRegressor(
        loss="quantile", alpha=config.alpha / 2, random_state=0
    )
    gb_hi = lambda: GradientBoostingRegressor(
        loss="quantile", alpha=1 - config.alpha / 2, random_state=0
    )

    for seed in seeds:
        set_all(seed)
        X, y = gen.generate(seed)
        sp = _split_data(X, y, config.calibration_ratio)
        t0 = time.perf_counter()
        m = SplitConformalRegressor(alpha=config.alpha)
        m.fit(sp.X_train, sp.y_train).calibrate(sp.X_cal, sp.y_cal)
        lo, hi = m.predict_interval(sp.X_test)
        cov = interval_coverage(lo, hi, sp.y_test)
        elapsed = time.perf_counter() - t0
        report.rows.append(
            Result(
                method="SplitConformal",
                task="regression",
                seed=seed,
                coverage=cov,
                coverage_target=1 - config.alpha,
                coverage_gap=abs(cov - (1 - config.alpha)),
                valid=cov >= (1 - config.alpha) - config.coverage_tol,
                n_test=len(sp.y_test),
                mean_interval_width=mean_interval_width(lo, hi),
                median_interval_width=median_interval_width(lo, hi),
                elapsed_sec=elapsed,
            )
        )

    _cqr("CQR-Linear", lin(), lin_hi())
    _cqr("CQR-GB", gb(), gb_hi())

    for seed in seeds:
        set_all(seed)
        X, y = gen.generate(seed)
        sp = _split_data(X, y, config.calibration_ratio)
        t0 = time.perf_counter()
        m = CQRFuseRegressor(alpha=config.alpha)
        m.fit(sp.X_train, sp.y_train).calibrate(sp.X_cal, sp.y_cal)
        lo, hi = m.predict_interval(sp.X_test)
        cov = interval_coverage(lo, hi, sp.y_test)
        elapsed = time.perf_counter() - t0
        report.rows.append(
            Result(
                method="CQRFuse",
                task="regression",
                seed=seed,
                coverage=cov,
                coverage_target=1 - config.alpha,
                coverage_gap=abs(cov - (1 - config.alpha)),
                valid=cov >= (1 - config.alpha) - config.coverage_tol,
                n_test=len(sp.y_test),
                mean_interval_width=mean_interval_width(lo, hi),
                median_interval_width=median_interval_width(lo, hi),
                elapsed_sec=elapsed,
            )
        )
    return report
