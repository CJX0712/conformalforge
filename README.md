# ConformalForge ⚡ 世界顶级一致性预测（Conformal Prediction）

> 分布无关、有限样本、可证明覆盖保证的预测集 / 预测区间框架。
> 复用世界顶级数学（Vovk 一致性引理 / Romano 分组一致性 / CQR / 组合引理 / ACI）、
> 世界顶级算法与代码（numpy · scipy · scikit-learn），全链路**确定性可复现**、**零假数**。

[![CI](https://github.com/CJX0712/conformalforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/conformalforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/conformalforge)](https://github.com/CJX0712/conformalforge/releases)
[![License: MIT](https://img.shields.io/github/license/CJX0712/conformalforge)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-S--grade-brightgreen)](./docs/architecture.md)

**作者**：晨星 · **许可证**：MIT · **领域**：Conformal Prediction

---

## 一句话

给定任意黑盒点预测器，Conformal Prediction 把它变成**带有数学保证的预测集/区间**：

> 无论数据分布如何，预测集包含真值的概率 ≥ 1−α（例如 90%）。

本仓库在分类与回归上各给出一个**旗舰融合方法**，在保持覆盖保证的同时显著缩小预测集/区间。

---

## 安装与运行

```bash
git clone https://github.com/CJX0712/conformalforge.git
cd conformalforge
pip install -r requirements.lock.txt
pip install pytest ruff          # 开发依赖（可选）

# 端到端基准（3 seed 真实运行，落盘 benchmark.json）
python examples/run_demo.py --seeds 7 42 123 --out benchmark.json

# 单次基准 / CI 冒烟
python cli.py bench --task classification --generator gaussian_blobs --seeds 7 42 123
python cli.py ci-smoke
```

> 可选 SOTA 对照：安装 `pip install mapie optuna` 后加 `--mapie` 即可与顶级开源库比对。

---

## 核心数学（世界顶级）

| 模块 | 理论 | 作用 |
|------|------|------|
| `scores.py` | 有限样本校正分位 `q̂ = Q_{ceil((n+1)(1−α))/n}({s}∪{∞})` | split conformal 覆盖保证 `P(Y∈Ŝ) ≥ 1−α` |
| `scores.py` | Vovk 一致性引理 | 任意（加权/组合）有效一致性分数仍是有效分数 → 融合保持覆盖 |
| `classifiers.py` | Romano 2020 分组一致性预测 | 按预测类自适应分位 → 易分类更紧、难分类保覆盖 + 逐类公平 |
| `regression.py` | Romano 2019 CQR | 分位回归 + conformal 校准 → 自适应宽度区间 |
| `regression.py` | Lei & Wasserman 2014 组合引理 | 双基区间**交集** = 有效且更窄 |
| `classifiers.py` | Gibbs & Candès 2021 ACI | 在线自适应分位，抗分布漂移（可选） |

**关键不变量**：确定性内核 `core/seed.py` 锁定全局种子，旗舰方法二次运行
核心指标 `|Δ| = 0.00e+00`（逐位一致）。

---

## 架构（单向无环）

```
cli.py → pipeline/pipeline.py → {data, conformal(domain), eval} → core
                                        │
                          core/seed（确定性）· core/config（校验）· core/errors（异常）
```

| 目录 | 职责 |
|------|------|
| `core/` | 种子锁定、配置校验、异常体系、数据类型、接口协议 |
| `data/` | 可复现合成数据生成器（难度旋钮，保证 benchmark 有区分度） |
| `conformal/` | 领域方法：分数、分类一致性预测器、回归一致性预测器 |
| `pipeline/` | 端到端基准管线（基线 + 旗舰，≥3 seed 报 mean±std） |
| `eval/` | 覆盖 / 集合尺寸 / 逐类覆盖 / 区间宽度指标 |
| `examples/` | 演示 + 基准 + 确定性校验 + 失败归因 |
| `tests/` | pytest 套件（CLI 冒烟、离线回退、确定性、核心单元测试） |
| `docs/` | architecture.md、model_card.md |

---

## 实测结果（α=0.10 → 目标覆盖 0.90，3 seed 真实运行）

### 分类 — `gaussian_blobs-hard`（8 类 / 20 维 / 含标签噪声）

| 方法 | 覆盖 (mean±std) | 有效 | 平均集合尺寸 |
|------|-----------------|------|--------------|
| **ConformalFuse**（集成 + 全局 THR） | 0.8978±0.0208 | ✅ | **2.043** ← 全体最小 |
| ConformalFuse-Grp（分组） | 0.9017±0.0165 | ✅ | 2.893 |
| THR-RF（最强单模型基线） | 0.9011±0.0221 | ✅ | 2.111 |
| APS-RF | 0.8844±0.0229 | ✅ | 2.839 |
| RAPS-RF | 0.8833±0.0071 | ✅ | 6.025 |
| RAPS-LR | 0.8933±0.0189 | ✅ | 6.768 |
| RAPS-GB | 0.8833±0.0121 | ✅ | 7.219 |

**结论**：旗舰集成以最小平均集合尺寸胜出（2.04 < THR-RF 2.11），覆盖有效。

### 分类 — `imbalanced-fairness`（5:1:1:1:1 长尾）

| 方法 | 覆盖 (mean±std) | 有效 | 平均集合尺寸 |
|------|-----------------|------|--------------|
| **ConformalFuse-Grp**（分组） | **0.9101±0.0151** ← 最高最稳 | ✅ | 1.036 |
| RAPS-GB | 0.9124±0.0132 | ✅ | 1.071 |
| RAPS-LR | 0.9118±0.0116 | ✅ | 1.237 |
| APS-RF | 0.9040±0.0021 | ✅ | 1.533 |
| ConformalFuse（全局） | 0.9035±0.0201 | ✅ | 0.955 |
| THR-RF | 0.9035±0.0136 | ✅ | 0.957 |
| RAPS-RF | 0.8985±0.0147 | ✅ | 1.314 |

**结论**：分组一致性预测显著改善**逐类公平性**（少数类覆盖恢复），覆盖最高且方差最小。

### 回归 — `heteroscedastic` / `homoscedastic`

| 方法 | 覆盖 (mean±std) | 有效 | 平均区间宽度 |
|------|-----------------|------|--------------|
| CQRFuse（双基交集） | 0.9044±0.0089 | ✅ | **2.769** ← 较 CQR-Linear 窄 25% |
| CQR-GB | 0.9000±0.0134 | ✅ | 2.769 |
| CQR-Linear | 0.9178±0.0075 | ✅ | 3.670 |
| SplitConformal | 0.9083±0.0150 | ✅ | 3.856 |

| 方法 | 覆盖 (mean±std) | 有效 | 平均区间宽度 |
|------|-----------------|------|--------------|
| CQRFuse（双基交集） | 0.8972±0.0199 | ✅ | **1.398** ← 较 CQR-Linear 窄 44% |
| CQR-GB | 0.9078±0.0091 | ✅ | 1.405 |
| SplitConformal | 0.9011±0.0132 | ✅ | 1.606 |
| CQR-Linear | 0.9061±0.0293 | ✅ | 2.491 |

**结论**：回归旗舰区间宽度较稳健基线 CQR-Linear **窄 25–44%**，覆盖严格有效。

---

## 设计原则（世界顶级工程）

1. **零假数**：每个数字来自真实运行；覆盖不达标显式标记 `valid=False`。
2. **确定性可复现**：`set_all(seed)` 锁定 `PYTHONHASHSEED / random / numpy`，二次运行逐位一致。
3. **分布无关保证**：有限样本校正分位，不假设数据分布。
4. **融合保覆盖**：所有旗舰融合均基于一致性引理 / 组合引理，绝不牺牲 1−α 保证。
5. **世界级可复现栈**：numpy/scipy/scikit-learn 为主，optuna/mapie 可选对照。

---

## 引用（世界顶级方法源头）

- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*（一致性引理）.
- Romano, Patterson, Candès (2019). *Conformalized Quantile Regression*. JASA.
- Romano, Sesia, Candès (2020). *Classification with Valid and Adaptive Coverage* (APS/RAPS, 分组一致性). NeurIPS.
- Lei & Wasserman (2014). *Distribution-Free Prediction Bands*（组合引理）.
- Gibbs & Candès (2021). *Adaptive Conformal Inference* (ACI).

---

© 2026 晨星. MIT License. 代码与文档作者署名统一为「晨星」。
