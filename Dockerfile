FROM python:3.12-slim

WORKDIR /app

# 锁定依赖，离线可复现
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

# 工具链
RUN pip install --no-cache-dir pytest ruff

# 全量代码
COPY . .

# 默认跑端到端基准（3 seed 真实运行，落盘 benchmark.json）
CMD ["python", "examples/run_demo.py", "--seeds", "7", "42", "123", "--out", "benchmark.json"]
