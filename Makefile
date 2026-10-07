# ConformalForge · Makefile（作者：晨星）
# 一键复现：安装 -> 自检 -> demo -> CI 收口
PY ?= python

.PHONY: help install dev test demo lint format ci clean

help:
	@echo "ConformalForge 命令:"
	@echo "  make install   安装运行时依赖 + 本包(editable)"
	@echo "  make dev       安装开发依赖(pytest/pytest-cov/ruff)"
	@echo "  make test      运行单测 + 覆盖率"
	@echo "  make demo      运行端到端 benchmark（确定性自检）"
	@echo "  make lint      ruff check + format 双绿门禁"
	@echo "  make ci        本地 CI 等价收口"
	@echo "  make clean     清理缓存"

install:
	$(PY) -m pip install -r requirements.lock.txt
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install pytest pytest-cov ruff

test:
	$(PY) -m pytest -q -W ignore::UserWarning --cov=conformalforge --cov-report=term

demo:
	$(PY) -m conformalforge.examples.run_demo

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

ci: lint test demo

clean:
	rm -rf .ruff_cache .pytest_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
