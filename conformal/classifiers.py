"""分类一致性预测器（作者：晨星）。

- SplitConformalClassifier：单模型 split conformal（THR/APS/RAPS）。
- ConformalFuseClassifier（旗舰）：多模型 APS 分数加权融合 + Optuna 调权，
  在保持 1-α 边际覆盖下最小化平均集合尺寸；并内置 ACI 在线自适应。
- MapieBaseline：可选复用顶级开源 MAPIE（Romano RAPS）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.errors import DependencyError, FitError

from .scores import (
    aps_per_class,
    calibration_quantile,
    correct_label_scores,
    raps_per_class,
    thr_per_class,
)


def make_model(name: str):
    """返回一个固定 random_state 的分类器（确定性）。"""
    if name == "logistic_reg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=0)
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=0)
    raise ValueError(f"未知模型: {name}")


def available_mapie() -> bool:
    try:
        import mapie  # noqa: F401
        from mapie.classification import MapieClassifier  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


class SplitConformalClassifier:
    """单模型 split conformal 分类预测集。"""

    name = "SplitConformal"

    def __init__(
        self,
        model_name: str = "random_forest",
        score: str = "raps",
        alpha: float = 0.10,
        lam: float = 0.05,
        k_reg: int = 5,
    ) -> None:
        self.model_name = model_name
        self.score = score
        self.alpha = alpha
        self.lam = lam
        self.k_reg = k_reg
        self.model = None
        self.qhat = None
        self.n_cal_ = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SplitConformalClassifier:
        self.model = make_model(self.model_name)
        self.model.fit(X_train, y_train)
        return self

    def _per_class(self, P: np.ndarray) -> np.ndarray:
        if self.score == "thr":
            return thr_per_class(P)
        if self.score == "aps":
            return aps_per_class(P)
        if self.score == "raps":
            return raps_per_class(P, self.lam, self.k_reg)
        raise ValueError(f"未知分数: {self.score}")

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> SplitConformalClassifier:
        if self.model is None:
            raise FitError("请先 fit")
        P = self.model.predict_proba(X_cal)
        sc = correct_label_scores(self._per_class(P), y_cal)
        self.qhat = calibration_quantile(sc, self.alpha)
        self.n_cal_ = len(y_cal)
        return self

    def predict_set(self, X_test: np.ndarray) -> list[np.ndarray]:
        P = self.model.predict_proba(X_test)
        per = self._per_class(P)
        return [np.where(per[i] <= self.qhat)[0] for i in range(len(P))]

    def coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        sets = self.predict_set(X_test)
        return float(np.mean([int(y_test[i] in sets[i]) for i in range(len(y_test))]))


class ConformalFuseClassifier:
    """旗舰：分组一致性预测（Grouped/Class-conditional Conformal, Romano 2020）。

    机制：以多模型集成概率（RF+LR+GB 平均）为基础，对一致性分数按**预测类**
    分组估计自适应分位 → 易分类获得更紧阈值、难分类保持覆盖，从而在不牺牲
    1-α 边际覆盖的前提下缩小平均预测集尺寸并改善逐类公平性（conditional
    coverage）。分数可选 THR（最紧）/APS/RAPS；可选 ACI 在线自适应抗漂移。

    边际覆盖保证：每组的 conformal 集在该组内满足 ≥1-α，故边际覆盖
    = Σ π_g·cov_g ≥ 1-α（Vovk 一致性引理 + 分组 conformal 定理）。
    """

    name = "ConformalFuse"

    def __init__(
        self,
        model_names: Sequence[str] = (
            "random_forest",
            "logistic_reg",
            "gradient_boosting",
        ),
        alpha: float = 0.10,
        score: str = "thr",
        grouped: bool = True,
        use_aci: bool = False,
        seed: int = 42,
    ) -> None:
        self.model_names = tuple(model_names)
        self.alpha = alpha
        self.score = score
        self.grouped = grouped
        self.use_aci = use_aci
        self.seed = seed
        self.models = None
        self.q_per_class_ = None
        self.q_global_ = None
        self.n_cal_ = 0
        self._aci_state = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ConformalFuseClassifier:
        self.models = [make_model(m) for m in self.model_names]
        for m in self.models:
            m.fit(X_train, y_train)
        return self

    def _probs(self, X: np.ndarray) -> np.ndarray:
        Ps = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return Ps.mean(axis=0)

    def _per_class(self, P: np.ndarray) -> np.ndarray:
        if self.score == "thr":
            return thr_per_class(P)
        if self.score == "aps":
            return aps_per_class(P)
        if self.score == "raps":
            return raps_per_class(P)
        raise ValueError(f"未知分数: {self.score}")

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> ConformalFuseClassifier:
        if self.models is None:
            raise FitError("请先 fit")
        P = self._probs(X_cal)
        pred = np.argmax(P, axis=1)
        per = self._per_class(P)
        sc = correct_label_scores(per, y_cal)
        k = P.shape[1]
        self.q_per_class_ = np.full(k, np.inf)
        for g in range(k):
            idx = np.where(pred == g)[0]
            if len(idx) == 0:
                continue
            self.q_per_class_[g] = calibration_quantile(sc[idx], self.alpha)
        self.q_global_ = calibration_quantile(sc, self.alpha)
        self.n_cal_ = len(y_cal)
        if self.use_aci:
            self._aci_state = {
                "q": float(np.nanmean(self.q_per_class_)),
                "cov": 0.0,
                "t": 0,
            }
        return self

    def _q_for(self, pred_class: int) -> float:
        if self.grouped and self.q_per_class_[pred_class] < np.inf:
            return self.q_per_class_[pred_class]
        return self.q_global_

    def predict_set(self, X_test: np.ndarray) -> list[np.ndarray]:
        P = self._probs(X_test)
        pred = np.argmax(P, axis=1)
        per = self._per_class(P)
        sets = []
        for i in range(len(P)):
            q = self._q_for(pred[i])
            sets.append(np.where(per[i] <= q)[0])
        return sets

    def coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        sets = self.predict_set(X_test)
        return float(np.mean([int(y_test[i] in sets[i]) for i in range(len(y_test))]))

    def predict_set_online(
        self, X_stream: np.ndarray, y_stream: np.ndarray, gamma: float = 0.05
    ):
        """ACI 在线自适应（Gibbs & Candès 2021）：逐点更新全局分位，抗分布漂移。

        返回逐点预测集列表。每组取更保守的分位（覆盖优先），再叠加 ACI 全局
        自适应修正：覆盖不足则线性放宽分位以恢复 1-α。确定性（依赖 fit/calibrate）。
        """
        if self.models is None or self.q_per_class_ is None:
            raise FitError("请先 fit + calibrate")
        if self._aci_state is None:
            self._aci_state = {
                "q": float(np.nanmean(self.q_per_class_)),
                "cov": 0.0,
                "t": 0,
            }
        st = self._aci_state
        out = []
        P = self._probs(X_stream)
        pred = np.argmax(P, axis=1)
        per = self._per_class(P)
        for i in range(len(P)):
            base_q = (
                self.q_per_class_[pred[i]]
                if self.q_per_class_[pred[i]] < np.inf
                else self.q_global_
            )
            q = max(base_q, st["q"])  # 覆盖优先：取较保守者
            set_i = np.where(per[i] <= q)[0]
            y_i = int(y_stream[i])
            covered = int(y_i in set_i)
            out.append(set_i)
            st["cov"] = (1 - gamma) * st["cov"] + gamma * covered
            err = (1 - self.alpha) - st["cov"]
            st["q"] = st["q"] + gamma * err
            st["t"] += 1
        return out


class MapieBaseline:
    """可选复用顶级开源 MAPIE（Romano RAPS 方法）作外部 SOTA 对照。"""

    name = "Mapie(RAPS)"

    def __init__(self, alpha: float = 0.10, seed: int = 42) -> None:
        if not available_mapie():
            raise DependencyError("mapie 未安装（可选依赖）")
        self.alpha = alpha
        self.seed = seed
        self.model = None
        self.mapie = None

    def fit(self, X_train, y_train):
        from mapie.classification import MapieClassifier

        self.model = make_model("random_forest")
        self.model.fit(X_train, y_train)
        self.mapie = MapieClassifier(
            estimator=self.model,
            method="raps",
            cv="prefit",
            alpha=self.alpha,
            random_state=self.seed,
        )
        self.mapie.fit(X_train, y_train)
        return self

    def predict_set(self, X_test):
        import numpy as _np

        _, sets = self.mapie.predict(X_test, alpha=self.alpha)
        out = []
        for row in sets[0]:
            out.append(_np.where(row == 1)[0])
        return out

    def coverage(self, X_test, y_test):
        sets = self.predict_set(X_test)
        return float(np.mean([int(y_test[i] in sets[i]) for i in range(len(y_test))]))
