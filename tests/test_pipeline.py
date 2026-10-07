from conformalforge.core.config import Config
from conformalforge.pipeline.conformal_pipeline import ConformalPipeline


def _cfg(task, n=800, calib=400, test=400, seeds=(7, 11, 23)):
    return Config(
        task=task,
        n_train=n,
        n_calib=calib,
        n_test=test,
        seeds=list(seeds),
        noise=0.9,
        hetero=1.5,
    )


def test_classification_pipeline_runs_and_deterministic():
    cfg = _cfg("classification")
    pipe = ConformalPipeline(cfg)
    report = pipe.benchmark()
    assert report["determinism"]["bit_identical"] is True
    assert "conformal_fuse" in report["methods"]
    assert "lac_split" in report["methods"]


def test_regression_pipeline_runs_and_deterministic():
    cfg = _cfg("regression")
    pipe = ConformalPipeline(cfg)
    report = pipe.benchmark()
    assert report["determinism"]["bit_identical"] is True
    assert "reg_fuse" in report["methods"]


def test_full_benchmark_both_tasks():
    from conformalforge.examples.run_demo import main as demo_main

    code = demo_main()
    assert code == 0
