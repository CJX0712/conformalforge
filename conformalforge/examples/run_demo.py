"""End-to-end demo: run ConformalForge benchmark and persist ``benchmark.json``.

Run as a module so relative imports resolve:
    python -m conformalforge.examples.run_demo

Author: 晨星
"""

from __future__ import annotations

import json
import os
import sys
import time

from ..core.config import Config
from ..pipeline.conformal_pipeline import ConformalPipeline


def _print_table(task: str, report: dict) -> None:
    print(f"\n=== {task.upper()}  (alpha={report['alpha']}, seeds={report['seeds']}) ===")
    print(f"{'method':<18}{'coverage':>10}{'set_size/width':>18}{'under_gap':>12}")
    for name, m in report["methods"].items():
        prim = m["primary_mean"]
        ug = m["under_gap_mean"]
        ug_s = f"{ug:.4f}" if ug == ug else "  -  "
        print(f"{name:<18}{m['coverage_mean']:>10.4f}{prim:>18.4f}{ug_s:>12}")
    cmp = report["comparison"]
    print(
        f"flagship/{cmp['baseline']} {cmp['primary_label']} ratio = "
        f"{cmp['flagship_over_baseline']:.4f}  (<=1 better)"
    )
    if "conditional" in cmp:
        c = cmp["conditional"]
        print(
            f"conditional(SCP vs LAC) under_gap ratio = {c['scp_over_baseline']:.4f} (<=0.8 => >=20% smaller)"
        )
    if "adaptation" in cmp:
        for m, (rho, _) in cmp["adaptation"].items():
            print(f"  adaptation |{m:<16}| half-width~true-sigma rho = {rho:+.3f}")
    v = report["verdict"]
    print(
        f"verdict: grade={v['grade']} coverage_valid={v['coverage_valid']} "
        f"primary_win={v['primary_win']}"
    )


def main() -> int:
    t0 = time.time()
    out = {"system": "ConformalForge", "author": "晨星", "tasks": {}}
    for task in ("classification", "regression"):
        cfg = Config.from_env()
        cfg.task = task
        if task == "classification":
            cfg.noise = 1.3  # difficulty sweet spot: model good-but-imperfect
        else:
            cfg.hetero = 1.5  # heteroscedastic regime favours local-scale fuse
        cfg.validate()
        pipe = ConformalPipeline(cfg)
        report = pipe.benchmark()
        out["tasks"][task] = report
        _print_table(task, report)

    det = out["tasks"]["classification"]["determinism"]
    print(
        f"\ndeterminism: bit_identical={det['bit_identical']} "
        f"max|delta|={det['max_abs_delta_core']}"
    )
    elapsed = time.time() - t0
    out["elapsed_sec"] = round(elapsed, 2)
    out["determinism"] = det

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nbenchmark written -> {path}  ({elapsed:.1f}s)")
    # Honest gate: determinism must hold.
    if not det["bit_identical"]:
        print("ERROR: determinism broken; aborting with non-zero exit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
