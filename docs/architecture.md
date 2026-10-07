# ConformalForge · 架构文档（作者：晨星）

## 1. 设计原则

1. **复用顶级开源，不自研 SOTA 内核**。分类/回归基学习器与保角框架均建立在 scikit-learn / scipy / numpy 之上；仅在「经验分位数」「离线兜底」等实现细节上做正确性修复与加固。
2. **有限样本有效性优先于效率**。所有集合/区间的覆盖必须满足 `1−α` 的分裂保角保证；效率（集合大小/区间宽度）作为次级优化目标，且诚实报告。
3. **确定性可复现**。单一 `set_all(seed)` 入口统一播种 numpy / random，禁止任何隐式随机源；demo 两次运行核心指标逐位一致。
4. **无泄漏**。任何预处理（标准化、分位估计）仅在 train/calib 上 fit，test 绝不参与。

## 2. 模块职责（单向无环）

```
cli ──▶ pipeline.ConformalPipeline
                 │
                 ├─▶ data.（合成数据 / CSV 载入）
                 ├─▶ conformal.base（基学习器 + TRAIN-only 标准化）
                 ├─▶ conformal.scores（保角分数 + 经验分位数）
                 ├─▶ conformal.classification / regression（各预测器）
                 └─▶ core.（types/errors/config/interfaces/seed）
```

| 模块 | 职责 |
|------|------|
| `core.seed` | `set_all(seed)` / `make_rng(seed)` 全局确定性播种 |
| `core.config` | `Config`：ENV 覆盖 + schema 校验（seed/n_train/n_calib/.../alpha/seeds） |
| `core.types` | `Dataset`、`ConformalResult`（含 `min_class_coverage`/`under_gap`） |
| `core.errors` | `ConformalError` 体系 E100~E500 |
| `core.interfaces` | `DataGenerator` / `BaseLearner` / `ConformalPredictor` Protocol |
| `data.synthetic` | `GaussianBlobGenerator`、`HeteroscedasticRegGenerator`（固定 seed 可复现） |
| `data.loaders` | `load_csv()` |
| `conformal.base` | `Classifier`（sklearn LogisticRegression + numpy 多类牛顿兜底）、`Regressor`（Ridge + 闭式兜底）；标准化仅 fit train |
| `conformal.scores` | `lac_scores` / `aps_pi` / `raps_pi` / `conformal_quantile`（离散经验分位） |
| `conformal.classification` | `LACConformal` / `APSConformal` / `RAPSConformal` / `ClassConditional(SCP)` / `ConformalFuse` |
| `conformal.regression` | `SplitConformal`（基线）/ `RegFuse`（kNN 局部尺度归一化） |
| `pipeline` | `benchmark()`：多 seed 聚合、基线对照、消融、确定性、噪声自适应、等级判定 |

## 3. 关键数学不变量

- **覆盖有效性**：分裂保角理论保证 `P(Y_test ∈ C(X_test)) ≥ 1−α`（交换性假设下）。`conformal_quantile` 使用离散阶统计量 `ceil((n+1)(1−α))`，而非插值分位。
- **类条件公平性（SCP）**：对每个类 `c` 单独估计分位 `q_c`，使各类覆盖尽量均衡；评估用 `under_gap = max(0, (1−α) − min_c coverage_c)`（仅计欠覆盖，避免过覆盖掩盖不公平）。
- **回归自适应**：RegFuse 用 kNN 估计局部残差尺度 `σ̂(x)`，归一化分数后取分位，使区间宽度随真实 `σ(x)` 变化；以 ρ(半宽, σ(x)) 度量适配度。

## 4. 确定性策略

- 所有随机源（数据生成、模型初始化、APS/RAPS 随机化）均由 `Config.seeds` → `set_all(seed)` 唯一派生。
- `benchmark()` 对同一任务用同一 seed 序列运行两次，比对 `benchmark.json` 核心字段，`max|delta|==0` 视为逐位一致。
- 不进行任何依赖墙钟/环境的操作（除 `elapsed_sec` 仅用于预算报告，不参与判定）。

## 5. 无泄漏证明（代码审查结论）

- `conformal.base._standardize_fit` 仅在 `X_train` 上 fit；`apply` 用于 calib/test。
- 校准分位 `conformal_quantile` 仅在 calib 分数上估计；test 集从不接触分位估计过程。
- 数据生成器的种子与 pipeline 的 `set_all` 序列隔离，避免 calib/test 数据被训练阶段随机性污染。

## 6. 离线兜底

- `Classifier` / `Regressor` 在 sklearn 不可用时自动切换纯 numpy 实现（多类牛顿逻辑回归、闭式岭回归）。
- `available()` 探测机制预留给未来重型后端；缺失时 benchmark 自动跳过并标注 `skipped`，绝不伪造数字。

## 7. 性能预算

- demo 端到端 ≤ 60s（CPU，实测 ~9s，seeds=3，n_train=4000）。
- 内存峰值 < 2GB（合成数据规模下远低于此）。
