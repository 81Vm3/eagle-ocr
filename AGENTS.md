# AGENTS.md — eagle-ocr 工程约定

## 是什么
常驻 PaddleOCR HTTP 服务（`eagle-ocr serve`）。调用方（eagle_eye）只靠 `docs/api.md` 对接。

## 硬约定
- **接口以 `docs/api.md` 为准**。改字段、错误码、语义之前先找 Commander；只允许加调用方可忽略的扩展字段。
- HTTP 层只用标准库（`http.server`），不引入 Web 框架。核心依赖只有 Pillow；PaddleOCR 只在 `PaddleEngine.load()` 里 import。
- PaddleOCR 推理必须串行（`OcrService._ocr_lock`）；下载走全局线程池，并发 = `download_concurrency`。
- 默认监听 `127.0.0.1`。日志走 stderr，**不记录请求头 / 图片 headers**（可能带 cookie）；跨 host 重定向时去掉 Cookie / Authorization。
- 单张失败只能落到该张的 `error`，整体仍 200；新的异常路径要映射到文档里已有的错误码。

## 结构
- `src/eagle_ocr/config.py` 配置（TOML，未知键报错）
- `src/eagle_ocr/fetch.py` base64 / 下载（scheme、白名单、流式大小上限、超时）
- `src/eagle_ocr/engine.py` `OcrEngine` 协议、`PaddleEngine`、阅读顺序排序
- `src/eagle_ocr/service.py` 就绪状态、校验、调度（与 HTTP 无关，返回 `(status, body)`）
- `src/eagle_ocr/server.py` HTTP 路由；`cli.py` 命令行

## 开发与测试
- 环境：`uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[cpu]"`
- GPU 环境单独一个 venv（`.venv-gpu`，已 gitignore），`uv pip install -p .venv-gpu/bin/python -e ".[gpu]"`；`cpu`/`gpu` extra 互斥。
- 测试：`.venv/bin/python -m unittest discover -s tests -v`。测试**不联网、不依赖 PaddleOCR**：用 `tests/helpers.py` 的 `FakeEngine` 和 127.0.0.1 本地图床。
- 修 bug 先补能复现的测试。
- 提交：conventional commit，中文正文；按功能原子拆分。

## 已知坑
- paddlepaddle 3.3 CPU + oneDNN 推理 PP-OCRv6 报 `ConvertPirAttribute2RuntimeAttribute not support`，所以 `enable_mkldnn` 默认 false。
- PaddleOCR 在 CPU 版 paddle 上收到 `device="gpu"` 不报错、静默按 CPU 跑。所以 `PaddleEngine.load()` 先跑 `check_gpu()`，GPU 不可用就让加载失败，**不许回退 CPU**。
- 首次加载会从 HuggingFace 下载约 130MB 模型到 `~/.paddlex/official_models`。
