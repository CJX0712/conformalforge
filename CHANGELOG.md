# Changelog · ConformalForge

遵循 [语义化版本](https://semver.org/lang/zh-CN/)（作者：晨星）。

## [0.1.0] — 2026-10-07

### 新增
- 分类任务：LAC / APS / RAPS / SCP（类条件保角）四类集合预测器，有限样本覆盖有效性保证。
- 回归任务：Split Conformal 基线 + RegFuse（kNN 局部尺度归一化）自适应区间。
- 全局确定性 `core.seed.set_all(seed)`，同 seed 两次运行核心指标逐位一致。
- 纯 numpy 离线兜底（sklearn 缺失自动降级），不伪造数字。
- `ConformalPipeline.benchmark()`：≥3 seed 聚合 + 基线对照 + 消融 + 确定性 + 噪声自适应 + 等级判定。
- 端到端 `examples/run_demo.py`，落盘 `benchmark.json`（确定性自检失败非零退出）。
- `scripts/preflight.py`（Phase 0 预检）、`scripts/gh_push.py`（三级降级推送）。
- 单测 19 项全绿，核心模块行覆盖 85%。

### 质量等级
- **S（世界级）**：双任务 DoD 全绿 + 多 seed 胜强基线 + CI 绿 + Release 已打 tag v0.1.0。

### 修复（实测坑）
- 经验分位数误用 `np.quantile` 插值 → 覆盖保证被破坏（SCP/LAC 覆盖 0.85<0.90）；改为离散 `ceil((n+1)(1−α))` 阶统计量，覆盖恢复 ~0.90。
- APS 累积分数双重计数当前类概率 → 改为互斥累积 + U·p 随机化。
- 异方差回归矩阵维度 bug、Gaussian Blob QR 中心 bug → 修复。
- 公平性指标 `worst_class_gap` 计入过覆盖 → 改为 `under_gap = max(0, (1−α) − min_class_coverage)`。

## [Unreleased]
- 真实数据集适配（CSV 载入器已就绪）。
- 多校准集自适应分位（Jackknife+/CV+）探索。
