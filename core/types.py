"""核心数据类型（作者：晨星）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class Result:
    """单次运行结果。所有指标均来自真实运行。"""

    method: str
    task: str  # "classification" | "regression"
    seed: int
    # 共同
    coverage: float
    coverage_target: float
    coverage_gap: float  # |coverage - target|
    valid: bool  # 覆盖是否达标（>= target - tol）
    n_test: int
    # 分类
    avg_set_size: float | None = None
    max_set_size: int | None = None
    conditional_coverage: dict[int, float] | None = None
    # 回归
    mean_interval_width: float | None = None
    median_interval_width: float | None = None
    # 元
    elapsed_sec: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    system: str
    domain: str
    task: str
    seeds: list[int]
    rows: list[Result] = field(default_factory=list)

    def aggregate(self) -> dict:
        import numpy as _np

        def _meanstd(attr: str) -> tuple[float, float]:
            vals = [
                _np.asarray(getattr(r, attr))
                for r in self.rows
                if getattr(r, attr) is not None
            ]
            if not vals:
                return (float("nan"), float("nan"))
            arr = _np.stack(vals)
            return float(arr.mean()), float(arr.std())

        target = self.rows[0].coverage_target if self.rows else 0.0
        cov_mean, cov_std = _meanstd("coverage")
        if self.task == "classification":
            eff_mean, eff_std = _meanstd("avg_set_size")
            w_mean = w_std = float("nan")
        else:
            eff_mean = eff_std = float("nan")
            w_mean, w_std = _meanstd("mean_interval_width")
        n_valid = sum(1 for r in self.rows if r.valid)
        return {
            "system": self.system,
            "domain": self.domain,
            "task": self.task,
            "n_seeds": len(self.rows),
            "coverage_mean": cov_mean,
            "coverage_std": cov_std,
            "coverage_target": target,
            "n_valid": n_valid,
            "avg_set_size_mean": eff_mean,
            "avg_set_size_std": eff_std,
            "mean_interval_width_mean": w_mean,
            "mean_interval_width_std": w_std,
        }
