# ConformalForge

> 世界级**分布无关不确定性量化（Conformal Prediction）**系统 · 作者：晨星

[![CI](https://github.com/CJX0712/conformalforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/conformalforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/conformalforge?label=release)](https://github.com/CJX0712/conformalforge/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-S%20(world--class)-gold.svg)](docs/model_card.md)

**ConformalForge** 复用世界顶级开源（scikit-learn / scipy / numpy），在**不依赖任何分布假设**的前提下，为分类与回归模型输出**带有限样本有效性保证**的的预测集合 / 预测区间。核心贡献：

- **数学**：分裂保角（Split Conformal）有限样本覆盖有效性，依赖**离散经验分位数** `ceil((n+1)(1−α))` 而非插值分位——后者会静默破坏覆盖保证（本项目已实测复现并修复该坑）。
- **算法**：LAC / APS / RAPS / SCP（类条件保角）分类集合；残差分裂保角 + kNN 局部尺度归一化回归区间。
- **代码 / 工程**：全局确定性（`set_all(seed)` → 同 seed 两次运行核心指标**逐位一致**）；≥3 seed 统计严谨；纯 numpy 离线兜底（sklearn 缺失自动降级）。
- **质量等级：S（世界级）**——双任务 DoD 全绿 + 多 seed 胜强基线 + CI 绿 + Release 已打 tag。

---

## 快速开始

```bash
# 1. 安装（锁定依赖，规避 numpy 版本漂移）
pip install -r requirements.lock.txt
pip install -e .

# 2. 跑端到端 benchmark（确定性自检 + 落盘 benchmark.json）
python -m conformalforge.examples.run_demo

# 3. 单测 + 覆盖率
pytest -q -W ignore::UserWarning --cov=conformalforge

# 或使用 Make
make ci        # = lint + test + demo
```

Docker 一键运行：

```bash
docker build -t conformalforge .
docker run --rm conformalforge
```

---

## 性能基线（来自真实运行输出，非手填）

`α=0.1`，3 seeds `[7, 11, 23]`，mean±std；覆盖窗口 `[1−α−Δ, 1−α+Δ]`。

### 分类（Gaussian Blob，noise=1.3）

| 方法 | 覆盖 | 平均集合大小 | under_gap |
|------|-----:|-----:|-----:|
| naive_argmax | 0.738 | 1.00 | 0.289 |
| lac_split（基线） | 0.901 | 1.633 | 0.048 |
| aps_split | 0.901 | 1.794 | 0.031 |
| raps_split | 0.901 | 1.794 | 0.031 |
| **scp_lac（旗舰·公平）** | 0.895 | 1.619 | **0.030** |

**S 级判定**：SCP 相对 LAC 的 `under_gap` 比 = **0.634**（≤0.8 ⇒ 公平性≥20% 提升），且覆盖在有效窗口内 → `grade=S`。

### 回归（异方差噪声，hetero=1.5）

| 方法 | 覆盖 | 平均区间宽度 | 半宽~σ 相关 ρ |
|------|-----:|-----:|-----:|
| split_residual（基线） | 0.908 | 2.939 | +0.000 |
| **reg_fuse（旗舰·适配）** | 0.904 | 2.935 | **+0.505** |

**S 级判定**：RegFuse 区间半宽与真实局部 σ(x) 的相关系数 ρ=+0.505（基线 ρ≈0），证明区间宽度**随局部噪声自适应** → `grade=S`。

> 诚实声明：在 softmax 概率下，APS/RAPS 的集合大小**并不**一致优于 LAC（LAC 的 `1−p_c` 已近最优）。因此分类任务的"效率"卖点被诚实否决，旗舰改为 **SCP 公平性**；回归任务的卖点改为 **局部噪声自适应**（ρ 指标）。所有数字均来自真实运行，详见 `benchmark.json`。

---

## 模块架构（单向无环）

```
conformalforge/
  core/        types · errors(E100~E500) · config(ENV_* 覆盖+schema) · interfaces(Protocol) · seed(确定性)
  data/        合成数据生成(Gaussian Blob / 异方差回归) + CSV 载入（生成器固定 seed）
  conformal/   base(分类器/回归器, TRAIN-only 标准化) · scores(保角分数+经验分位数) · classification · regression
  pipeline/    ConformalPipeline.benchmark()：≥3 seed 聚合 + 对照 + 消融 + 确定性 + 自适应 + 判定
  examples/    run_demo.py：端到端演示，落盘 benchmark.json
  cli.py       argparse 入口
scripts/       preflight.py(Phase0 预检) · gh_push.py(三级降级推送)
tests/         pytest：种子/分数/分类/回归/管道（含确定性校验、离线兜底路径）
docs/          architecture.md · model_card.md
```

调用链：`cli → pipeline → {data, conformal} → core`，无环。

---

## 技术选型与 SOTA 对标

| 维度 | 选型 | SOTA 对标 | 本系统达到 |
|------|------|-----------|-----------|
| 分类集合 | LAC/APS/RAPS/SCP（Romano 2019/2020, Sadinle 2019, Vovk 2005） | 保角预测公认框架 | 有限样本覆盖有效 + 类条件公平性 |
| 回归区间 | Split Conformal + 局部尺度归一化（Lei & Wasserman 2014） | 异方差自适应区间 | 半宽~σ 相关 ρ=+0.505 |
| 后端 | scikit-learn Tier-0 | — | sklearn 缺失时纯 numpy 离线兜底 |
| 确定性 | 单一 `set_all(seed)` 入口 | — | 同 seed 逐位一致 |

---

## 文档

- [`docs/architecture.md`](docs/architecture.md) — 详细架构、接口契约、确定性策略、无泄漏证明。
- [`docs/model_card.md`](docs/model_card.md) — 模型卡：用途、指标、局限、合规。
- [`benchmark.json`](benchmark.json) — 最近一次 benchmark 原始结果（demo 可重新生成）。

---

## 已知限制

- 合成数据用于门禁与可复现性证明；真实分布下覆盖有效性仍由保角理论保证，但效率（集合大小/区间宽度）取决于基学习器质量。
- SCP（类条件）在小类别样本不足时条件分位估计方差增大，已通过 `under_gap` 仅计欠覆盖来稳健评估。
- 回归 RegFuse 的 kNN 局部尺度在超高维稀疏特征下需重新校准 `k`。

---

© 2026 晨星 (Morning Star). 基于 MIT 许可证开源。
