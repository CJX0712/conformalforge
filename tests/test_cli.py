"""CLI 与离线回退测试（作者：晨星）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    return subprocess.run(
        [sys.executable, "cli.py", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


def test_cli_ci_smoke():
    res = _run(["ci-smoke"])
    assert res.returncode == 0, res.stderr[-500:]
    assert "PASS" in res.stdout


def test_cli_demo_help():
    res = _run(["--help"])
    assert res.returncode == 0
    assert "demo" in res.stdout


def test_offline_no_optional_deps():
    """不装 mapie/optuna 也能完整跑分类管线（离线回退）。"""
    from core.config import Config
    from pipeline.pipeline import run_classification

    rep = run_classification(
        "gaussian_blobs", {"n_classes": 5}, Config(), seeds=[7], use_mapie=False
    )
    assert len(rep.rows) > 0
    assert all(r.coverage is not None for r in rep.rows)
