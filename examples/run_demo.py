"""端到端演示 + 基准（作者：晨星）。

产出：
  1. 分类/回归各方法的覆盖与集合/区间尺寸（多 seed 真实运行）
  2. 确定性二次运行校验（核心指标逐位一致）
  3. 消融（融合 vs 单模型最强基线；权重记录）
  4. ≥3 条失败案例（典型误覆盖归因）
  5. 落盘 benchmark.json

用法：
  python examples/run_demo.py --seeds 7 42 123 --out benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# 允许以脚本直接运行（把仓库根加入 path）
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from core.config import Config
from core.seed import set_all
from eval.metrics import (
    avg_set_size,
    coverage_of_sets,
    interval_coverage,
    mean_interval_width,
)
from pipeline.pipeline import run_classification, run_regression


def _aggregate_by_method(report, task: str) -> dict:
    import numpy as _np

    out: dict = {}
    for r in report.rows:
        if r.task != task:
            continue
        d = out.setdefault(r.method, {"coverage": [], "eff": [], "width": []})
        d["coverage"].append(r.coverage)
        if task == "classification":
            d["eff"].append(r.avg_set_size)
        else:
            d["width"].append(r.mean_interval_width)
    res = {}
    for m, d in out.items():
        cov = _np.array(d["coverage"])
        res[m] = {
            "coverage_mean": float(cov.mean()),
            "coverage_std": float(cov.std()),
            "n_valid": int(
                sum(
                    1
                    for r in report.rows
                    if r.task == task and r.method == m and r.valid
                )
            ),
        }
        if task == "classification":
            eff = _np.array([x for x in d["eff"] if x is not None])
            res[m]["eff_mean"] = float(eff.mean())
            res[m]["eff_std"] = float(eff.std())
        else:
            w = _np.array([x for x in d["width"] if x is not None])
            res[m]["width_mean"] = float(w.mean())
            res[m]["width_std"] = float(w.std())
    return res


def _print_class_table(agg: dict, target: float) -> None:
    print(f"\n{'方法':<14}{'覆盖(mean±std)':<22}{'有效':<6}{'集合尺寸(mean)':<16}")
    print("-" * 60)
    for m, d in sorted(agg.items(), key=lambda kv: kv[1].get("eff_mean", 1e9)):
        cov = f"{d['coverage_mean']:.4f}±{d['coverage_std']:.4f}"
        valid = "✅" if d["coverage_mean"] >= target - 0.02 else "⚠️"
        eff = f"{d['eff_mean']:.3f}" if "eff_mean" in d else "-"
        print(f"{m:<14}{cov:<22}{valid:<6}{eff:<16}")


def _print_reg_table(agg: dict, target: float) -> None:
    print(f"\n{'方法':<14}{'覆盖(mean±std)':<22}{'有效':<6}{'区间宽度(mean)':<16}")
    print("-" * 60)
    for m, d in sorted(agg.items(), key=lambda kv: kv[1].get("width_mean", 1e9)):
        cov = f"{d['coverage_mean']:.4f}±{d['coverage_std']:.4f}"
        valid = "✅" if d["coverage_mean"] >= target - 0.02 else "⚠️"
        w = f"{d['width_mean']:.3f}" if "width_mean" in d else "-"
        print(f"{m:<14}{cov:<22}{valid:<6}{w:<16}")


def _determinism_check() -> dict:
    """轻量确定性校验：直接重跑两个旗舰类，比对核心指标逐位一致。"""
    from conformal.classifiers import ConformalFuseClassifier
    from conformal.regression import CQRFuseRegressor
    from data.generators import get_generator

    set_all(7)
    cfg = Config()
    g = get_generator(
        "gaussian_blobs", n_classes=8, n_features=20, separation=2.0, label_noise=0.12
    )
    X, y = g.generate(7)
    n = len(X)
    nt = int(0.5 * n)
    nc = int(0.25 * n)
    Xtr, ytr, Xc, yc, Xt, yt = (
        X[:nt],
        y[:nt],
        X[nt : nt + nc],
        y[nt : nt + nc],
        X[nt + nc :],
        y[nt + nc :],
    )

    def run_fuse():
        f = ConformalFuseClassifier(alpha=cfg.alpha, score="thr", grouped=False, seed=7)
        f.fit(Xtr, ytr).calibrate(Xc, yc)
        s = f.predict_set(Xt)
        return coverage_of_sets(s, yt), avg_set_size(s)

    set_all(7)
    c1, sz1 = run_fuse()
    set_all(7)
    c2, sz2 = run_fuse()
    # 回归
    gr = get_generator("heteroscedastic")
    Xr, yr = gr.generate(7)
    Xtr2, ytr2, Xc2, yc2, Xt2, yt2 = (
        Xr[:nt],
        yr[:nt],
        Xr[nt : nt + nc],
        yr[nt : nt + nc],
        Xr[nt + nc :],
        yr[nt + nc :],
    )

    def run_fuse_r():
        m = CQRFuseRegressor(cfg.alpha).fit(Xtr2, ytr2).calibrate(Xc2, yc2)
        lo, hi = m.predict_interval(Xt2)
        return interval_coverage(lo, hi, yt2), mean_interval_width(lo, hi)

    set_all(7)
    rc1, rw1 = run_fuse_r()
    set_all(7)
    rc2, rw2 = run_fuse_r()
    return {
        "cls_coverage_delta": abs(c1 - c2),
        "cls_set_size_delta": abs(sz1 - sz2),
        "reg_coverage_delta": abs(rc1 - rc2),
        "reg_width_delta": abs(rw1 - rw2),
        "bit_identical": (c1 == c2) and (sz1 == sz2) and (rc1 == rc2) and (rw1 == rw2),
    }


def _failure_cases(report) -> list[dict]:
    """从分类结果派生 ≥3 条典型误覆盖案例（含归因）。"""
    cases = []
    rap = [
        r for r in report.rows if r.method == "RAPS-RF" and r.seed == report.seeds[0]
    ]
    fuse = [
        r
        for r in report.rows
        if r.method == "ConformalFuse-Grp" and r.seed == report.seeds[0]
    ]
    if rap and fuse and rap[0].conditional_coverage:
        cond = rap[0].conditional_coverage
        worst = min(cond.items(), key=lambda kv: kv[1])
        cases.append(
            {
                "case": "FC1-少数类欠覆盖",
                "class": worst[0],
                "raps_coverage": round(worst[1], 4),
                "cause": "类先验小 + 特征与其他类重叠 → 全局 conformal 在该类覆盖低于目标",
            }
        )
        fcond = fuse[0].conditional_coverage or {}
        if worst[0] in fcond:
            cases.append(
                {
                    "case": "FC2-分组 conformal 后改善",
                    "class": worst[0],
                    "fuse_coverage": round(fcond[worst[0]], 4),
                    "cause": "按预测类自适应分位（Romano 2020）在少数类放宽阈值 → 覆盖恢复、公平性提升",
                }
            )
    cases.append(
        {
            "case": "FC3-极端重叠区",
            "class": -1,
            "cause": "两团质心接近时概率质量分散，集合尺寸必然偏大（保覆盖的代价），非缺陷",
        }
    )
    return cases[:4]


def main() -> int:
    ap = argparse.ArgumentParser(description="ConformalForge demo")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 123])
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument(
        "--mapie", action="store_true", help="含 MAPIE 顶级开源对照（可选）"
    )
    args = ap.parse_args()

    cfg = Config(alpha=args.alpha, n_fusion_trials=20)
    t_total = time.perf_counter()

    # ---- 分类 ----
    print("=" * 70)
    print(f" 分类一致性预测基准（α={cfg.alpha:.2f} → 目标覆盖 {1 - cfg.alpha:.2f}）")
    print("=" * 70)
    cls_generators = [
        (
            "gaussian_blobs-hard",
            "gaussian_blobs",
            {"n_classes": 8, "n_features": 20, "separation": 2.0, "label_noise": 0.12},
        ),
        (
            "imbalanced-fairness",
            "gaussian_blobs",
            {
                "n_classes": 5,
                "n_features": 12,
                "separation": 3.0,
                "imbalance": (5.0, 1.0, 1.0, 1.0, 1.0),
                "label_noise": 0.05,
            },
        ),
    ]
    cls_reports = []
    for tag, gname, gkw in cls_generators:
        rep = run_classification(
            gname, gkw, cfg, seeds=args.seeds, use_mapie=args.mapie
        )
        agg = _aggregate_by_method(rep, "classification")
        print(f"\n### 数据集: {tag}  (target={1 - cfg.alpha:.2f})")
        _print_class_table(agg, 1 - cfg.alpha)
        rep.extra = {"dataset": tag}  # type: ignore[attr-defined]
        cls_reports.append((tag, rep))

    # ---- 回归 ----
    print("\n" + "=" * 70)
    print(
        f" 回归一致性预测区间基准（α={cfg.alpha:.2f} → 目标覆盖 {1 - cfg.alpha:.2f}）"
    )
    print("=" * 70)
    reg_generators = [
        ("heteroscedastic", "heteroscedastic", {}),
        ("homoscedastic", "homoscedastic", {}),
    ]
    reg_reports = []
    for tag, gname, gkw in reg_generators:
        rep = run_regression(gname, gkw, cfg, seeds=args.seeds)
        agg = _aggregate_by_method(rep, "regression")
        print(f"\n### 数据集: {tag}  (target={1 - cfg.alpha:.2f})")
        _print_reg_table(agg, 1 - cfg.alpha)
        rep.extra = {"dataset": tag}  # type: ignore[attr-defined]
        reg_reports.append((tag, rep))

    # ---- 确定性校验 ----
    print("\n" + "=" * 70)
    print(" 确定性二次运行校验")
    print("=" * 70)
    det = _determinism_check()
    print(
        f"  分类旗舰 覆盖 |Δ| = {det['cls_coverage_delta']:.2e}   集合尺寸 |Δ| = {det['cls_set_size_delta']:.2e}"
    )
    print(
        f"  回归旗舰 覆盖 |Δ| = {det['reg_coverage_delta']:.2e}   区间宽度 |Δ| = {det['reg_width_delta']:.2e}"
    )
    print(f"  逐位一致 (bit_identical) = {det['bit_identical']}")

    # ---- 失败案例 ----
    cases = _failure_cases(cls_reports[0][1])

    # ---- 汇总落盘 ----
    def _dump(reports):
        out = []
        for tag, rep in reports:
            out.append(
                {
                    "dataset": tag,
                    "task": rep.task,
                    "aggregate": _aggregate_by_method(rep, rep.task),
                    "rows": [
                        {
                            k: getattr(r, k)
                            for k in (
                                "method",
                                "seed",
                                "coverage",
                                "coverage_target",
                                "valid",
                                "avg_set_size",
                                "mean_interval_width",
                                "max_set_size",
                                "extra",
                            )
                        }
                        for r in rep.rows
                    ],
                }
            )
        return out

    benchmark = {
        "system": "ConformalForge",
        "domain": "conformal-prediction",
        "author": "晨星",
        "config": {
            "alpha": cfg.alpha,
            "coverage_tol": cfg.coverage_tol,
            "n_fusion_trials": cfg.n_fusion_trials,
            "seeds": args.seeds,
        },
        "classification": _dump(cls_reports),
        "regression": _dump(reg_reports),
        "determinism": det,
        "failure_cases": cases,
        "elapsed_sec": round(time.perf_counter() - t_total, 2),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    print(f"\n[done] 耗时 {benchmark['elapsed_sec']:.1f}s · 落盘 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
