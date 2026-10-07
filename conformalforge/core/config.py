"""Configuration with environment override and schema validation.

All knobs are read from ``ENV_CF_*`` environment variables when present, then
validated by ``validate()``. This keeps CI / container runs reproducible and
auditable.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import ConfigError


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # Global determinism
    seed: int = 7
    # Data generation
    n_train: int = 4000
    n_calib: int = 2000
    n_test: int = 2000
    n_classes: int = 5
    noise: float = 0.9  # classification Gaussian blob noise (difficulty knob)
    # Regression difficulty
    hetero: float = 1.5  # heteroscedastic scale exponent strength
    # Conformal
    alpha: float = 0.1  # target 1 - alpha miscoverage
    # Task
    task: str = "classification"  # or "regression"
    # HPO for RAPS-style regularization on a tune split
    raps_lambda_grid: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    raps_k_reg: int = 5
    # Determinism / reporting
    seeds: list[int] = field(default_factory=lambda: [7, 11, 23])

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls()
        mapping = {
            "ENV_CF_SEED": ("seed", int),
            "ENV_CF_N_TRAIN": ("n_train", int),
            "ENV_CF_N_CALIB": ("n_calib", int),
            "ENV_CF_N_TEST": ("n_test", int),
            "ENV_CF_N_CLASSES": ("n_classes", int),
            "ENV_CF_NOISE": ("noise", float),
            "ENV_CF_HETERO": ("hetero", float),
            "ENV_CF_ALPHA": ("alpha", float),
            "ENV_CF_TASK": ("task", str),
        }
        for env_key, (attr, caster) in mapping.items():
            if env_key in os.environ:
                try:
                    setattr(cfg, attr, caster(_env(env_key, "")))
                except (TypeError, ValueError) as e:
                    raise ConfigError(f"bad value for {env_key}: {e}") from e
        if "ENV_CF_SEEDS" in os.environ:
            cfg.seeds = [int(s) for s in _env("ENV_CF_SEEDS", "").split(",") if s]
        return cfg

    def validate(self) -> None:
        if not (0 < self.alpha < 1):
            raise ConfigError(f"alpha must be in (0,1), got {self.alpha}")
        if self.n_classes < 2:
            raise ConfigError("n_classes must be >= 2")
        if self.task not in ("classification", "regression"):
            raise ConfigError(f"unknown task {self.task!r}")
        if self.n_train <= 0 or self.n_calib <= 0 or self.n_test <= 0:
            raise ConfigError("splits must be positive")
        if len(self.seeds) < 1:
            raise ConfigError("need at least one seed")
