#!/usr/bin/env python
"""ConformalForge · Phase 0 环境预检（作者：晨星）。

直接执行，输出 ✅/⚠️ 表作为第一个汇报检查点：
  python scripts/preflight.py --pip-pkg numpy scikit-learn scipy

硬门槛：Python 解释器 / gh 认证 / git 署名 / 磁盘≥5GB / PyPI 可达 / GitHub API。
WARN 级：代理/镜像降级提示。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def run(cmd, check=False):
    try:
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return p
    except Exception as e:

        class _P:
            returncode = 1
            stdout = ""
            stderr = str(e)

        return _P()


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else " (需 >=3.11)")


def check_gh_auth() -> tuple[bool, str]:
    p = run("gh auth status")
    if p.returncode == 0:
        return True, "gh 已登录"
    return False, "gh 未登录（需 gh auth login）"


def check_git_identity() -> tuple[bool, str]:
    n = run("git config user.name").stdout.strip()
    e = run("git config user.email").stdout.strip()
    ok = bool(n) and bool(e)
    return ok, f"user.name={n!r} user.email={e!r}" + ("" if ok else "（需配置署名）")


def check_disk() -> tuple[bool, str]:
    _, _, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    ok = free_gb >= 5.0
    return ok, f"可用 {free_gb:.1f}GB" + ("" if ok else " (<5GB 硬门槛)")


def _http_ok(url: str) -> bool:
    p = run(f"curl -sS -m 8 -o NUL -w %{{http_code}} {url}")
    code = (p.stdout or "").strip()
    return code.startswith("2")


def check_pypi() -> tuple[bool, str]:
    sources = (
        "https://pypi.org/simple/numpy/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/numpy/",
    )
    for url in sources:
        if _http_ok(url):
            return True, f"PyPI 可达 ({url.split('/')[2]})"
    return False, "PyPI 双源均不可达（硬阻断）"


def check_github_api() -> tuple[bool, str]:
    p = run("gh api users/CJX0712 --jq .login")
    if p.returncode == 0 and "CJX0712" in (p.stdout or ""):
        return True, "GitHub API 可达 (CJX0712)"
    return False, "GitHub API 不可达（需网络/代理）"


def check_pip_pkgs(pkgs) -> list[tuple[str, bool, str]]:
    res = []
    for pkg in pkgs:
        mod = pkg.split("[")[0]
        p = run(f'python -c "import {mod}"', check=False)
        ok = p.returncode == 0
        res.append((pkg, ok, "已装" if ok else "未装（将尝试 pip install）"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip-pkg", nargs="*", default=[], help="需预检的 tier0 包")
    args = ap.parse_args()

    print("=== ConformalForge Phase 0 环境预检 ===")
    rows = []
    checks = [
        ("Python 解释器", *check_python()),
        ("gh 认证", *check_gh_auth()),
        ("git 署名", *check_git_identity()),
        ("磁盘空间", *check_disk()),
        ("PyPI 可达", *check_pypi()),
        ("GitHub API", *check_github_api()),
    ]
    for name, ok, msg in checks:
        rows.append((name, ok, msg))
    for pkg, ok, msg in check_pip_pkgs(args.pip_pkg):
        rows.append((f"依赖 {pkg}", ok, msg))

    npass = sum(1 for _, ok, _ in rows if ok)
    nwarn = len(rows) - npass
    for name, ok, msg in rows:
        icon = "✅" if ok else "⚠️"
        print(f"  {icon} {name:<16} {msg}")
    print(f"\n预检完成：PASS={npass} WARN={nwarn}")
    # 任一硬门槛 FAIL 视为失败
    hard_fail = not all(ok for _, ok, _ in checks)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
