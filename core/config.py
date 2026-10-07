"""配置（作者：晨星）。

所有可调项均可用环境变量 ENV_CONFORMAL_* 覆盖，并做 schema 校验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import ConfigError

_ENV_PREFIX = "ENV_CONFORMAL_"


@dataclass
class Config:
    # 默认显著性水平（目标覆盖 = 1 - alpha）
    alpha: float = 0.10
    # 覆盖容差：实测覆盖需 >= 1 - alpha - coverage_tol 才算有效
    coverage_tol: float = 0.02
    # RAPS 正则化
    raps_lambda: float = 0.05
    raps_k_reg: int = 5
    # 默认校准/测试切分
    calibration_ratio: float = 0.33
    # 默认随机种子
    seed: int = 42
    # Optuna 调参预算（融合权重搜索）
    n_fusion_trials: int = 24
    # 确定性校验排除的字段（计时等）
    exclude_from_determinism: tuple[str, ...] = field(
        default=("elapsed_sec", "wall_sec", "timestamp")
    )

    @classmethod
    def from_env(cls) -> Config:
        kw: dict = {}
        casts = {
            "alpha": float,
            "coverage_tol": float,
            "raps_lambda": float,
            "raps_k_reg": int,
            "calibration_ratio": float,
            "seed": int,
            "n_fusion_trials": int,
        }
        for key, caster in casts.items():
            env = os.environ.get(_ENV_PREFIX + key.upper())
            if env is not None:
                try:
                    kw[key] = caster(env)
                except ValueError as e:
                    raise ConfigError(
                        f"{_ENV_PREFIX}{key.upper()} 无法解析为 {caster}: {e}"
                    )
        return cls(**kw)

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ConfigError(f"alpha 必须在 (0,1): {self.alpha}")
        if not 0.0 <= self.coverage_tol < self.alpha:
            raise ConfigError(f"coverage_tol 必须在 [0, alpha): {self.coverage_tol}")
        if not 0.0 < self.calibration_ratio < 1.0:
            raise ConfigError(
                f"calibration_ratio 必须在 (0,1): {self.calibration_ratio}"
            )
        if self.n_fusion_trials < 1:
            raise ConfigError(f"n_fusion_trials 必须 >=1: {self.n_fusion_trials}")
