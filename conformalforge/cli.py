"""ConformalForge command-line entry point.

Usage:
    python -m conformalforge.cli --task classification --alpha 0.1 --seed 7
    python -m conformalforge.cli --task regression --seeds 7,11,23

Author: 晨星
"""

from __future__ import annotations

import argparse
import json

from .core.config import Config
from .pipeline.conformal_pipeline import ConformalPipeline


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="conformalforge", description="World-class conformal prediction"
    )
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seeds", type=str, default="7,11,23")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-calib", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.9, help="classification difficulty")
    p.add_argument("--hetero", type=float, default=1.5, help="regression heteroscedastic strength")
    p.add_argument("--out", type=str, default=None, help="write report JSON here")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s]
    cfg = Config(
        task=args.task,
        alpha=args.alpha,
        seeds=seeds,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_test=args.n_test,
        noise=args.noise,
        hetero=args.hetero,
    )
    cfg.validate()
    pipe = ConformalPipeline(cfg)
    report = pipe.benchmark()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
