"""错误码（作者：晨星）。

E100~E199 配置/输入   E200~E299 数据   E300~E399 评分/一致性
E400~E499 训练/推断    E500~E599 系统/依赖
"""

from __future__ import annotations


class ConformalError(Exception):
    """所有 ConformalForge 错误的基类。"""


class ConfigError(ConformalError):
    """E1xx 配置或输入非法。"""


class DataError(ConformalError):
    """E2xx 数据问题（空、泄漏、维度不匹配）。"""


class ScoreError(ConformalError):
    """E3xx 一致性分数计算错误。"""


class FitError(ConformalError):
    """E4xx 训练/推断阶段错误。"""


class DependencyError(ConformalError):
    """E5xx 可选依赖缺失或不可用。"""
