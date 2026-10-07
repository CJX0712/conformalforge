"""全局确定性入口（作者：晨星）。

ConformalForge 的每一个随机源都必须从这里一次性设齐，保证
「同 seed 两次运行 benchmark 核心指标逐位一致」。
"""

from __future__ import annotations

import os
import random

import numpy as np

_SEEDED = False


def set_all(seed: int = 42) -> None:
    """一次性设齐 numpy / random / 环境变量，保证可复现。"""
    global _SEEDED
    os.environ["PYTHONHASHSEED"] = str(seed % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    try:  # numpy>=2 才有 default_rng 之外的全局便捷接口
        from numpy.random import PCG64, default_rng  # noqa: F401

        _ = default_rng(seed)
    except Exception:  # noqa: BLE001, S110
        pass
    _SEEDED = True


def is_seeded() -> bool:
    return _SEEDED
