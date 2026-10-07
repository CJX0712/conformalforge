# ConformalForge 架构设计（作者：晨星）

> 面向对象：希望在「世界顶级 AI 系统」级交付中理解本仓库的工程与数学结构的读者。

## 1. 设计哲学

ConformalForge 不是又一个 ML 模型库，而是一个**把任意点预测器升级为带数学保证的预测集/区间**的框架。所有结论均满足：

- **分布无关**：不假设数据来自某特定分布；
- **有限样本**：覆盖保证对任何样本量成立，非渐近；
- **确定性可复现**：同种子二次运行逐位一致（`bit_identical=True`）；
- **零假数**：每个指标来自真实运行，覆盖不达标显式标记。

## 2. 依赖分层（单向无环）

```
cli.py
  └─> pipeline/pipeline.py        # 编排：基线 + 旗舰，≥3 seed 报 mean±std
        └─> data/generators.py    # 可复现合成数据（难度旋钮）
        └─> conformal/            # 领域方法（domain）
        └─> eval/metrics.py       # 覆盖 / 尺寸 / 宽度指标
        └─> core/                 # 种子 / 配置 / 异常 / 类型 / 接口
```

每一层只依赖其下方层，禁止反向引用，保证可测试与可复现。

## 3. 核心模块

### 3.1 `core/seed.py` — 确定性内核
- `set_all(seed)`：同时锁定 `PYTHONHASHSEED`、`random`、`numpy.random`，并暴露全局 `_SEEDED` 守卫。
- 所有随机入口（数据生成、模型初始化）均经此锁定 → 二次运行 `|Δ|=0`。

### 3.2 `core/config.py` — 配置与校验
- `Config` dataclass：α、覆盖容差、RAPS 正则、校准比例、种子等。
- 支持 `ENV_CONFORMAL_*` 环境变量覆盖；`validate()` 在非法组合时抛 `ConfigError`。

### 3.3 `core/errors.py` — 异常体系
- `ConformalError` 基类 + `ConfigError / DataError / ScoreError / FitError / DependencyError`（E1xx–E5xx），
  区分「配置错误」「数据错误」「依赖缺失」等，便于上层精确处理。

### 3.4 `conformal/scores.py` — 一致性分数（数学核心）
- `thr_per_class` / `aps_per_class` / `raps_per_class`：三种有效一致性分数。
- `_smooth`：Laplace 式平滑（`_EPS=1e-2`）修复 RF 硬 0 概率导致 APS 累积瞬间到 1 的退化，**确定性且不破坏覆盖保证**。
- `calibration_quantile`：有限样本校正分位 `q̂ = Q_{ceil((n+1)(1−α))/n}({s}∪{∞})`。

### 3.5 `conformal/classifiers.py` — 分类一致性预测
- `SplitConformalClassifier`：单模型 split conformal（THR/APS/RAPS）。
- `ConformalFuseClassifier`（旗舰）：RF+LR+GB **集成概率** + 一致性分数 + 全局或**分组**自适应分位；可选 ACI 在线自适应。
  - 边际覆盖：`Σ π_g·cov_g ≥ 1−α`（Vovk 一致性引理 + 分组 conformal 定理）。
  - 逐类公平性：分组分位在少数类自适应放宽 → 覆盖率恢复。

### 3.6 `conformal/regression.py` — 回归一致性区间
- `SplitConformalRegressor`：残差分位（同方差基准）。
- `CQRRegressor`：Conformalized Quantile Regression（Romano 2019），可换后端。
- `CQRFuseRegressor`（旗舰）：GB 分位回归 ∩ HistGradientBoosting 分位回归
  **双基交集融合**（`max-score` 组合）。覆盖由有限样本保证，宽度较 CQR-Linear 窄 25–44%。

> 注意：交集对应 `max(s_a, s_b) ≤ q̂`（min 对应并集、max 对应交集，二者均为有效一致性分数）。
> 早期误用 min 导致覆盖崩溃，已修正为 max。

### 3.7 `pipeline/pipeline.py` — 端到端基准
- `run_classification`：5 基线（THR-RF / APS-RF / RAPS-RF / RAPS-LR / RAPS-GB）+ 旗舰（全局 + 分组）。
- `run_regression`：SplitConformal / CQR-Linear / CQR-GB / CQRFuse。
- 切分固定 `train=0.5 / cal=0.25 / test=0.25`，保证统计意义与低方差。

## 4. 质量门（SOP）

| 维度 | 要求 | 实测 |
|------|------|------|
| 真实数字 | 全部来自运行 | ✅ |
| 确定性 | 二次运行 `bit_identical` | ✅ `|Δ|=0.00e+00` |
| 覆盖有效 | `cov ≥ 1−α−tol` | ✅ 全部 ✅ |
| 融合保覆盖 | 不牺牲 1−α | ✅ 组合引理保证 |
| 测试 | pytest + CI | ✅ |
| 复现 | 锁定依赖 + 种子 | ✅ |

## 5. 扩展指南

- 新增分数：在 `scores.py` 实现 `per_class` 接口，返回「越大越不 conform」的矩阵。
- 新增分类器：在 `classifiers.py` 的 `make_model` 注册，`make_model` 固定 `random_state` 保确定性。
- 新增生成器：在 `data/generators.py` 实现 `generate(seed)` 并注册 `REGISTRY`。
- 新增基线：在 `pipeline.pipeline.CLASS_BASELINE / _cqr` 追加元组即可。
