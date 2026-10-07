#!/usr/bin/env python
"""ConformalForge · 三级降级 GitHub 推送器（作者：晨星）。

推送策略（自动选择通道，全部幂等可重跑）：
  L1  git push（先直连、失败再试代理各一次）
  L2  Git Data API（blobs -> tree -> commit -> PATCH/POST refs，空仓库先 bootstrap）
  L3  Contents API（逐文件 PUT base64，按 blob-sha 断点续传）

commit 署名固定：user.name=晨星 / user.email=CJX0712@users.noreply.github.com

用法：
  python scripts/gh_push.py --repo conformalforge --local . \
      --description "..." --tag v0.1.0 --release "<benchmark 摘要>"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time

AUTHOR_NAME = "晨星"
AUTHOR_EMAIL = "CJX0712@users.noreply.github.com"
GH = "gh"
MAX_RETRIES = 4


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd, check=True, retries=1, mask=False):
    last = None
    for i in range(retries):
        try:
            p = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            last = p
            if p.returncode == 0:
                return p
            last = p
            if i < retries - 1:
                log(f"  retry {i + 1}: {cmd[:60]} (rc={p.returncode})")
                time.sleep(1.5 * (i + 1))
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
    if check:
        rc = last.returncode if hasattr(last, "returncode") else -1
        out = getattr(last, "stderr", "") or getattr(last, "stdout", "")
        raise RuntimeError(f"command failed (rc={rc}): {cmd[:80]}\n{out[:500]}")
    return last


def gh_api(method: str, path: str, data=None, retries=MAX_RETRIES):
    """Call GitHub REST API through the authenticated `gh` CLI."""
    cmd = f'{GH} api {method} "{path}"'
    if data is not None:
        payload = json.dumps(data)
        cmd += " --input -"
    for i in range(retries):
        try:
            p = subprocess.run(
                cmd,
                shell=True,
                input=(payload if data is not None else None),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if p.returncode == 0:
                txt = p.stdout.strip()
                return json.loads(txt) if txt else {}
            log(f"  gh_api {method} {path} rc={p.returncode}: {p.stderr.strip()[:200]}")
        except Exception as e:
            log(f"  gh_api exception: {e}")
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"gh_api failed after {retries}x: {method} {path}")


def repo_exists(repo_full: str) -> bool:
    p = run(f"{GH} repo view {repo_full}", check=False)
    return p.returncode == 0


def ensure_repo(repo: str, description: str, public: bool) -> str:
    repo_full = f"CJX0712/{repo}"
    if repo_exists(repo_full):
        log(f"repo 已存在：{repo_full}")
        return repo_full
    vis = "--public" if public else "--private"
    log(f"创建仓库 {repo_full} ...")
    p = run(
        f'{GH} repo create {repo_full} {vis} --description "{description}"',
        check=False,
        retries=2,
    )
    if p.returncode != 0:
        # 可能并发已创建，或名称冲突；再确认一次
        if repo_exists(repo_full):
            log("repo 在创建后已存在，继续。")
            return repo_full
        raise RuntimeError(f"无法创建仓库：{p.stderr.strip()[:300]}")
    log(f"仓库已创建：{repo_full}")
    return repo_full


def configure_git(local: str) -> None:
    run(f'git -C "{local}" config user.name "{AUTHOR_NAME}"', check=False)
    run(f'git -C "{local}" config user.email "{AUTHOR_EMAIL}"', check=False)


def commit_all(local: str, message: str) -> None:
    run(f'git -C "{local}" add -A')
    # 空提交保护
    p = run(f'git -C "{local}" status --porcelain', check=False)
    if not p.stdout.strip():
        log("没有待提交变更，跳过 commit。")
        return
    run(
        f'git -C "{local}" -c user.name="{AUTHOR_NAME}" '
        f'-c user.email="{AUTHOR_EMAIL}" commit -m "{message}"',
        check=False,
    )


def detect_branch(local: str) -> str:
    p = run(f'git -C "{local}" rev-parse --abbrev-ref HEAD', check=False)
    br = p.stdout.strip()
    return br or "main"


def L1_git_push(local: str, repo_full: str) -> bool:
    log("L1: 尝试 git push")
    remote_url = f"https://github.com/{repo_full}.git"
    # 设置 remote
    run(f'git -C "{local}" remote remove origin', check=False)
    run(f'git -C "{local}" remote add origin "{remote_url}"', check=False)
    branch = detect_branch(local)
    for proxy in (False, True):
        env = ""
        if proxy:
            # 尝试系统中常见的本地代理（失败也无妨，直接回退 L2）
            env = "https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 "
        cmd = f'{env}git -C "{local}" push -u origin {branch}'
        p = run(cmd, check=False, retries=2)
        if p.returncode == 0:
            log(f"L1 成功：{branch} 已推送。")
            return True
        log(f"  L1 push 失败（proxy={proxy}）：{p.stderr.strip()[:160]}")
    return False


def _list_files(local: str):
    out = []
    skip_dirs = {
        ".git", ".ruff_cache", ".pytest_cache", "__pycache__",
        ".venv", "venv", "env", "node_modules",
    }
    for root, dirs, files in os.walk(local):
        # 排除版本控制/缓存/环境/构建产物
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith((".pyc", ".egg-info")):
                continue
            if f in ("benchmark.json", ".coverage"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local).replace(os.sep, "/")
            if rel.startswith(".git/"):
                continue
            out.append((rel, full))
    out.sort()
    return out


def L2_gitdata_push(local: str, repo_full: str, branch: str) -> bool:
    log("L2: 尝试 Git Data API（blobs -> tree -> commit -> ref）")
    try:
        files = _list_files(local)
        # 1) 创建 blobs
        blob_shas = {}
        for rel, full in files:
            with open(full, "rb") as fh:
                content = fh.read()
            b64 = base64.b64encode(content).decode()
            res = gh_api(
                "POST", f"repos/{repo_full}/git/blobs", {"content": b64, "encoding": "base64"}
            )
            blob_shas[rel] = res["sha"]
            log(f"  blob: {rel} -> {res['sha'][:10]}")
        # 2) 创建 tree
        tree = [
            {"path": rel, "mode": "100644", "type": "blob", "sha": sha}
            for rel, sha in blob_shas.items()
        ]
        tree_res = gh_api("POST", f"repos/{repo_full}/git/trees", {"tree": tree})
        tree_sha = tree_res["sha"]
        # 3) 父 commit（仓库可能为空）
        parent_sha = None
        ref_res = gh_api("GET", f"repos/{repo_full}/git/refs/heads/{branch}", retries=1)
        if isinstance(ref_res, dict) and "object" in ref_res:
            parent_sha = ref_res["object"]["sha"]
        commit_data = {
            "message": "ConformalForge initial delivery (author: 晨星)",
            "tree": tree_sha,
            "author": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL, "date": _now_iso()},
            "committer": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL, "date": _now_iso()},
        }
        if parent_sha:
            commit_data["parents"] = [parent_sha]
        commit_res = gh_api("POST", f"repos/{repo_full}/git/commits", commit_data)
        commit_sha = commit_res["sha"]
        # 4) 更新 / 创建 ref
        if parent_sha:
            gh_api(
                "PATCH",
                f"repos/{repo_full}/git/refs/heads/{branch}",
                {"sha": commit_sha, "force": True},
            )
        else:
            gh_api(
                "POST",
                f"repos/{repo_full}/git/refs",
                {"ref": f"refs/heads/{branch}", "sha": commit_sha},
            )
        log(f"L2 成功：commit {commit_sha[:10]}")
        return True
    except Exception as e:
        log(f"L2 失败：{e}")
        return False


def L3_contents_push(local: str, repo_full: str, branch: str) -> bool:
    log("L3: 尝试 Contents API（逐文件 PUT，blob-sha 续传）")
    try:
        files = _list_files(local)
        for rel, full in files:
            with open(full, "rb") as fh:
                content = fh.read()
            b64 = base64.b64encode(content).decode()
            # 查询已有文件，拿到 sha（用于更新）
            existing = gh_api("GET", f"repos/{repo_full}/contents/{rel}?ref={branch}", retries=1)
            sha = existing.get("sha") if isinstance(existing, dict) else None
            data = {
                "message": f"ConformalForge: update {rel} (author: 晨星)",
                "content": b64,
                "branch": branch,
            }
            if sha:
                data["sha"] = sha
            gh_api("PUT", f"repos/{repo_full}/contents/{rel}", data, retries=2)
            log(f"  put: {rel}")
        log("L3 完成。")
        return True
    except Exception as e:
        log(f"L3 失败：{e}")
        return False


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def create_tag_and_release(repo_full: str, tag: str, release_notes: str) -> None:
    commit_sha = ""
    # 取得默认分支最新 commit
    br_res = gh_api("GET", f"repos/{repo_full}", retries=1)
    default_branch = (br_res.get("default_branch") if isinstance(br_res, dict) else None) or "main"
    ref_res = gh_api("GET", f"repos/{repo_full}/git/refs/heads/{default_branch}", retries=1)
    if isinstance(ref_res, dict) and "object" in ref_res:
        commit_sha = ref_res["object"]["sha"]
    if not commit_sha:
        log("无法取得 commit sha，跳过 tag/release。")
        return
    # tag 已存在则删除重建
    tag_get = gh_api("GET", f"repos/{repo_full}/git/refs/tags/{tag}", retries=1)
    if isinstance(tag_get, dict) and "object" in tag_get:
        log(f"tag {tag} 已存在，删除后重建。")
        gh_api("DELETE", f"repos/{repo_full}/git/refs/tags/{tag}", retries=1)
        time.sleep(6)  # ref 删除后需等待，避免 race
    # 轻量 tag 通过 refs 创建
    gh_api(
        "POST",
        f"repos/{repo_full}/git/refs",
        {"ref": f"refs/tags/{tag}", "sha": commit_sha},
        retries=2,
    )
    log(f"tag {tag} -> {commit_sha[:10]}")
    # release
    rel_res = gh_api("GET", f"repos/{repo_full}/releases/tags/{tag}", retries=1)
    if isinstance(rel_res, dict) and "id" in rel_res:
        log("release 已存在，更新。")
        gh_api(
            "PATCH",
            f"repos/{repo_full}/releases/{rel_res['id']}",
            {"body": release_notes},
            retries=1,
        )
    else:
        gh_api(
            "POST",
            f"repos/{repo_full}/releases",
            {
                "tag_name": tag,
                "name": tag,
                "body": release_notes,
                "target_commitish": default_branch,
            },
            retries=2,
        )
        log("Release 已创建。")


def main() -> int:
    ap = argparse.ArgumentParser(description="ConformalForge GitHub 三级降级推送")
    ap.add_argument("--repo", required=True, help="仓库名（全小写，不含 CJX0712/）")
    ap.add_argument("--local", default=".", help="本地工作区路径")
    ap.add_argument(
        "--description", default="ConformalForge · 世界级分布无关不确定性量化系统", help="仓库描述"
    )
    ap.add_argument("--tag", default="v0.1.0", help="semver tag")
    ap.add_argument("--release", default="", help="Release 说明（benchmark 摘要）")
    ap.add_argument("--private", action="store_true", help="创建为私有仓库")
    ap.add_argument("--skip-git-init", action="store_true", help="本地已是 git 仓库时跳过 init")
    args = ap.parse_args()

    local = os.path.abspath(args.local)
    repo_full = f"CJX0712/{args.repo}"

    # Phase 0: gh 认证
    who = run(f"{GH} auth status", check=False)
    if who.returncode != 0:
        log("ERROR: gh 未登录或认证失败。")
        return 2

    # 初始化 git（如需）
    if not args.skip_git_init and not os.path.isdir(os.path.join(local, ".git")):
        run(f'git -C "{local}" init -q')
    configure_git(local)

    # 确保仓库存在
    ensure_repo(args.repo, args.description, public=not args.private)

    # 提交本地变更
    commit_all(local, "ConformalForge delivery (author: 晨星)")

    # 三级降级推送
    pushed = False
    branch = detect_branch(local)
    try:
        pushed = L1_git_push(local, repo_full)
    except Exception as e:
        log(f"L1 异常：{e}")
    if not pushed:
        pushed = L2_gitdata_push(local, repo_full, branch)
    if not pushed:
        pushed = L3_contents_push(local, repo_full, branch)
    if not pushed:
        log("ERROR: 三级推送均失败。")
        return 1

    # tag + release
    create_tag_and_release(repo_full, args.tag, args.release)

    log(f"DONE: https://github.com/{repo_full}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
