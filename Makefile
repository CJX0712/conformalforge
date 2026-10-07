# ConformalForge — 开发工作流（作者：晨星）
# 所有命令在仓库根目录执行。

.PHONY: setup test lint format ci demo clean check

setup:
	pip install -r requirements.lock.txt
	pip install pytest ruff

test:
	python -m pytest -q

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff format .

# 本地等价 CI：lint + 测试 + 冒烟 + 确定性 + 密钥检查
check: lint test ci
	python examples/run_demo.py --seeds 7 42 123 --out benchmark.json
	@echo "[check] 完成"

ci:
	python cli.py ci-smoke

demo:
	python examples/run_demo.py --seeds 7 42 123 --out benchmark.json

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
