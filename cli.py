"""ConformalForge CLI（作者：晨星）。

用法：
  python cli.py demo  --seeds 7 42 123 --out benchmark.json
  python cli.py bench --task classification --generator gaussian_blobs --seeds 7 42 123
  python cli.py ci-smoke
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from core.config import Config
from pipeline.pipeline import run_classification, run_regression


def _row(r):
    base = f"{r.method:<14} seed={r.seed:<4} cov={r.coverage:.4f} valid={r.valid}"
    if r.task == "classification":
        return base + f" set={r.avg_set_size:.3f}"
    return base + f" width={r.mean_interval_width:.3f}"


def cmd_demo(args) -> int:
    from examples.run_demo import main as demo_main

    return demo_main()


def cmd_bench(args) -> int:
    cfg = Config(alpha=args.alpha)
    if args.task == "classification":
        rep = run_classification(
            args.generator, None, cfg, seeds=args.seeds, use_mapie=args.mapie
        )
    else:
        rep = run_regression(args.generator, None, cfg, seeds=args.seeds)
    for r in rep.rows:
        print(_row(r))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        k: getattr(r, k)
                        for k in (
                            "method",
                            "seed",
                            "coverage",
                            "valid",
                            "avg_set_size",
                            "mean_interval_width",
                        )
                    }
                    for r in rep.rows
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
    return 0


def cmd_smoke(args) -> int:
    cfg = Config(alpha=0.10)
    rep = run_classification(
        "gaussian_blobs",
        {"imbalance": (3.0, 1.0, 1.0, 1.0, 1.0)},
        cfg,
        seeds=[7],
        use_mapie=False,
    )
    ok = all(r.valid for r in rep.rows)
    print("ci-smoke classification:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="conformalforge", description="ConformalForge CLI"
    )
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="端到端演示 + 基准")
    d.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 123])
    d.add_argument("--out", default="benchmark.json")
    d.add_argument("--alpha", type=float, default=0.10)
    d.add_argument("--mapie", action="store_true")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("bench", help="单次基准")
    b.add_argument(
        "--task", choices=["classification", "regression"], default="classification"
    )
    b.add_argument("--generator", default="gaussian_blobs")
    b.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 123])
    b.add_argument("--alpha", type=float, default=0.10)
    b.add_argument("--mapie", action="store_true")
    b.add_argument("--json", default=None)
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("ci-smoke", help="CI 冒烟")
    s.set_defaults(func=cmd_smoke)

    args = ap.parse_args()
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
