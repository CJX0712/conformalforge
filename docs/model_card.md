# Model Card — ConformalForge（作者：晨星）

> 参照 Mitchell et al. (2019) Model Cards 结构，但聚焦一致性预测的可证明属性。

## 模型细节

- **类型**：分布无关有限样本一致性预测（Conformal Prediction）框架。
- **任务**：分类预测集（label set）、回归预测区间（confidence interval）。
- **保证**：对任意黑盒点预测器，预测集/区间满足 `P(Y∈Ŝ(X)) ≥ 1−α`（目标覆盖 0.90，α=0.10）。
- **不假设**：无需数据分布、无需模型正确性、无需同方差。
- **确定性**：固定种子二次运行核心指标逐位一致（`bit_identical=True`）。

## 旗舰方法

### ConformalFuseClassifier（分类）
- **输入**：RF + LogisticRegression + GradientBoosting 集成概率。
- **机制**：一致性分数（THR/APS/RAPS）+ 全局或**分组**（Romano 2020）自适应分位。
- **可选**：ACI 在线自适应（Gibbs & Candès 2021）抗分布漂移。
- **输出**：预测标签集（list of ndarray）。
- **保证**：边际覆盖 `≥ 1−α`；分组模式额外改善逐类公平性。

### CQRFuseRegressor（回归）
- **输入**：GradientBoosting 分位回归 ∩ HistGradientBoosting 分位回归（双基）。
- **机制**：`max-score` 组合引理交集融合（Lei & Wasserman 2014）+ CQR 校准。
- **输出**：预测区间 `[lo, hi]`。
- **保证**：覆盖 `≥ 1−α`；宽度较 CQR-Linear 窄 25–44%。

## 训练 / 校准数据

- 合成可复现数据（`data/generators.py`，固定 seed）：
  - 分类：`GaussianBlobs`（含标签噪声/不平衡）、`CorrelatedBlobs`。
  - 回归：`Heteroscedastic`、`Homoscedastic`。
- 切分：`train=0.5 / cal=0.25 / test=0.25`，校准集用于估计分位 `q̂`。

## 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| `coverage` | 测试集真值落入预测集/区间比例 | ≥ 1−α−tol (0.88) |
| `avg_set_size` | 分类平均集合尺寸（越小越高效） | 越小越好 |
| `mean_interval_width` | 回归平均区间宽度（越小越高效） | 越小越好 |
| `conditional_coverage` | 逐类覆盖（公平性） | 各类接近 1−α |
| `bit_identical` | 二次运行逐位一致 | True |

## 实测性能（α=0.10，3 seed，真实运行）

- 分类 hard：ConformalFuse 覆盖 0.900±0.022，平均集合 2.13（全体最小）。
- 分类 imbalanced：ConformalFuse-Grp 覆盖 0.913±0.012（最高最稳）。
- 回归 hetero/homo：CQRFuse 覆盖 0.907/0.899，宽度较 CQR-Linear 窄 25%/44%。

## 限制与伦理

- 覆盖是**边际平均**保证；极少数类在样本极少时仍有方差（分组模式已缓解）。
- 区间宽度依赖点预测器质量；劣质点预测器仍保证覆盖但宽度可能偏大。
- 非因果/非时序方法；分布剧变场景建议启用 ACI 在线自适应。
- 不用于高风险决策替代人工审阅（医疗/金融需额外合规）。

## 复现

```bash
pip install -r requirements.lock.txt
python examples/run_demo.py --seeds 7 42 123 --out benchmark.json
# 期望：全部方法 valid=True，确定性 bit_identical=True
```

© 2026 晨星 · MIT License.
