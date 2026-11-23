# eagle-ocr CPU 版镜像。
#   docker build -t eagle-ocr:cpu .
#   docker run -d --name eagle-ocr -p 127.0.0.1:8866:8866 -v eagle-ocr-models:/root/.paddlex eagle-ocr:cpu
# 首次启动会下载模型（约 130MB）到 /root/.paddlex，挂卷后重启不再下载。
FROM python:3.12-slim

# opencv（PaddleX 依赖）需要 libGL / glib；paddle 需要 libgomp。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[cpu]"

ENV PYTHONUNBUFFERED=1
EXPOSE 8866
# 容器内必须监听 0.0.0.0 才能被端口映射访问；宿主机侧用 127.0.0.1:8866 映射限制为本机。
CMD ["eagle-ocr", "serve", "--host", "0.0.0.0", "--port", "8866"]
