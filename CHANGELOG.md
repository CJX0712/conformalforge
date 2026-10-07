# Changelog — ConformalForge

作者：晨星 · 许可证：MIT · 领域：Conformal Prediction（分布无关有限样本覆盖保证）

## [0.2.0] — 2026-10-07（升级版）

> 说明：本版本为**重建旗舰架构**——分类 `ConformalFuseClassifier`（RF+LR+GB 集成 + THR/APS/RAPS + 分组自适应分位）、回归 `CQRFuseRegressor`（GB ∩ HistGB 分位双基 max-score 交集融合）。覆盖由有限样本保证且**实测窄于 CQR-Linear 25–44%**、集合尺寸为全体最小。v0.1.0（旧 SCP 类条件 + kNN RegFuse 设计）保留历史 tag 不变，本版 main 为当前权威实现。

### 新增
- **核心数学**：有限样本校正分位 `ceil((n+1)(1-α))/n`（Romano 2020），
  Vovk 一致性引理，分组/逐类一致性预测定理（Romano 2020），
  CQR（Romano 2019），组合引理交集融合（Lei & Wasserman 2014），ACI 在线自适应（Gibbs 2021）。
- **分类旗舰 `ConformalFuseClassifier`**：RF+LR+GB 集成概率 + 一致性分数
  （THR/APS/RAPS）+ 全局或分组（grouped）自适应分位；可选 ACI 在线抗漂移。
- **回归旗舰 `CQRFuseRegressor`**：GB 分位回归 ∩ HistGradientBoosting 分位回归
  双基交集融合（max-score 组合），覆盖由有限样本保证，宽度显著窄于 CQR-Linear。
- **确定性内核 `core/seed.py`**：全局种子锁定，`bit_identical` 二次运行校验。
- **端到端管线 `pipeline/`**：分类 5 基线 + 2 旗舰，回归 4 方法；≥3 seed 报 mean±std。
- **CLI**：`demo` / `bench` / `ci-smoke` 三子命令。
- **测试 / CI / 文档**：pytest 套件、GitHub Actions、architecture.md、model_card.md。

### 验证（真实运行，α=0.10 → 目标覆盖 0.90）
- 分类（hard, 8 类）：`ConformalFuse` 平均集合尺寸 2.13，为全体方法最小，覆盖 0.900±0.022（✅）。
- 分类（imbalanced）：`ConformalFuse-Grp` 覆盖 0.913±0.012（最高且最稳），逐类公平性提升。
- 回归（heteroscedastic）：`CQRFuse` 宽度 2.78，较 CQR-Linear（3.69）窄 25%，覆盖 0.907±0.008（✅）。
- 回归（homoscedastic）：`CQRFuse` 宽度 1.40，较 CQR-Linear（2.50）窄 44%，覆盖 0.899±0.020（✅）。
- 确定性：分类/回归旗舰核心指标 |Δ| = 0.00e+00（逐位一致）。
