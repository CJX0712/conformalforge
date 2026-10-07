# ConformalForge · 最小可复现运行镜像（作者：晨星）
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 安装锁定的运行时依赖（先于源码，利用层缓存）
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

# 安装项目本体（editable，确保 `import conformalforge` 稳定）
COPY . .
RUN pip install --no-cache-dir -e .

# 默认入口：运行端到端 benchmark（确定性自检 + 落盘 benchmark.json）
CMD ["python", "-m", "conformalforge.examples.run_demo"]
