"""合成数据生成（作者：晨星）。

生成器固定 seed，可复现；难度旋钮找「甜点」（基线明显低于天花板），
保证 benchmark 有区分度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianBlobs:
    """多类高斯团，separation 控制可分性，imbalance 控制长尾，label_noise 造难度甜点。"""

    name: str = "gaussian_blobs"
    n_classes: int = 5
    n_features: int = 12
    n_samples: int = 3000
    separation: float = 3.2
    imbalance: tuple[float, ...] | None = None  # 各类先验权重（自动归一化）
    label_noise: float = 0.0  # 标签翻转比例（制造不可约错误，使集合尺寸落在 2~4 甜点）

    def generate(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        centers = rng.normal(
            0.0, self.separation, size=(self.n_classes, self.n_features)
        )
        if self.imbalance is None:
            priors = np.ones(self.n_classes)
        else:
            priors = np.asarray(self.imbalance, dtype=float)
        priors = priors / priors.sum()
        counts = np.round(priors * self.n_samples).astype(int)
        # 保证每类至少 20 个样本（统计意义）
        counts = np.maximum(counts, 20)
        Xs, ys = [], []
        for c in range(self.n_classes):
            n_c = int(counts[c])
            X_c = rng.normal(centers[c], 1.0, size=(n_c, self.n_features))
            Xs.append(X_c)
            ys.append(np.full(n_c, c))
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        if self.label_noise > 0.0:
            n_flip = int(self.label_noise * X.shape[0])
            flip_idx = rng.choice(X.shape[0], size=n_flip, replace=False)
            other = (
                y[flip_idx] + rng.integers(1, self.n_classes, size=n_flip)
            ) % self.n_classes
            y[flip_idx] = other
        idx = rng.permutation(X.shape[0])
        return X[idx], y[idx]


@dataclass
class CorrelatedBlobs:
    """相关特征 + 重叠类，制造「预测置信度温和」的甜点区间。"""

    name: str = "correlated_blobs"
    n_classes: int = 6
    n_features: int = 14
    n_samples: int = 3000
    separation: float = 2.4

    def generate(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        centers = rng.normal(
            0.0, self.separation, size=(self.n_classes, self.n_features)
        )
        Xs, ys = [], []
        for c in range(self.n_classes):
            n_c = self.n_samples // self.n_classes
            base = rng.normal(centers[c], 1.0, size=(n_c, self.n_features))
            # 注入相关噪声：与第一个特征强相关
            base[:, 1:] = base[:, 1:] * 0.6 + base[:, :1] * 0.4
            Xs.append(base)
            ys.append(np.full(n_c, c))
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        idx = rng.permutation(X.shape[0])
        return X[idx], y[idx]


@dataclass
class Heteroscedastic:
    """异方差回归：方差随 |x| 增大，考验区间宽度自适应。"""

    name: str = "heteroscedastic"
    n_features: int = 1
    n_samples: int = 3000
    noise_base: float = 0.15
    noise_slope: float = 0.45

    def generate(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = rng.uniform(-3.0, 3.0, size=(self.n_samples, self.n_features))
        f = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 0]
        sigma = self.noise_base + self.noise_slope * np.abs(X[:, 0])
        y = f + rng.normal(0.0, sigma)
        return X, y


@dataclass
class Homoscedastic:
    """同方差回归：基准对照。"""

    name: str = "homoscedastic"
    n_features: int = 1
    n_samples: int = 3000
    noise: float = 0.4

    def generate(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = rng.uniform(-3.0, 3.0, size=(self.n_samples, self.n_features))
        f = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 0]
        y = f + rng.normal(0.0, self.noise, size=self.n_samples)
        return X, y


# 注册表：pipeline 按名取生成器
REGISTRY: dict[str, object] = {
    "gaussian_blobs": GaussianBlobs,
    "correlated_blobs": CorrelatedBlobs,
    "heteroscedastic": Heteroscedastic,
    "homoscedastic": Homoscedastic,
}


def get_generator(name: str, **kw) -> object:
    if name not in REGISTRY:
        raise KeyError(f"未知生成器: {name}")
    return REGISTRY[name](**kw)  # type: ignore[arg-type]
