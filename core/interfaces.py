"""抽象契约（作者：晨星）。

所有模块只经 Protocol 通信，可独立验证、可插拔后端。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Scorer(Protocol):
    """一致性分数函数：score 越大越「不 conform」。"""

    name: str

    def fit(self, P: np.ndarray, y: np.ndarray) -> Scorer: ...

    def scores(self, P: np.ndarray, y: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class ConformalPredictor(Protocol):
    """一致性预测器接口。"""

    name: str

    def calibrate(self, P: np.ndarray, y: np.ndarray) -> ConformalPredictor: ...

    def predict_set(self, P: np.ndarray) -> list[np.ndarray]: ...

    def coverage(self, P: np.ndarray, y: np.ndarray) -> float: ...


@runtime_checkable
class DataGenerator(Protocol):
    name: str

    def generate(self, seed: int) -> tuple[np.ndarray, np.ndarray]: ...
