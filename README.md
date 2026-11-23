# eagle-ocr

常驻 PaddleOCR HTTP 服务：模型只加载一次，给 [eagle_eye](../eagle_eye) 等调用方识别图片（小红书图文笔记配图等）里的文字。

- 接口：**[docs/api.md](docs/api.md)**（v1，正式接口文档；改接口先找 Commander）
- `GET /health` 看是否就绪；`POST /v1/ocr` 一次传多张图（url 或 base64），单张失败不影响其他张。
- 标准库 HTTP 服务（`ThreadingHTTPServer`），默认只监听 `127.0.0.1:8866`，**没有鉴权**。

## 依赖与版本

| 包 | 版本范围 | 说明 |
|---|---|---|
| Python | ≥ 3.11 | 实测 3.12 |
| `paddleocr` | `>=3.7,<3.8` | PaddleOCR 3.x 推理包，默认模型 PP-OCRv6（det + rec，约 130MB，首次运行自动下载到 `~/.paddlex/official_models`） |
| `paddlepaddle` | `>=3.3,<3.4` | CPU 推理框架（PyPI 上的包名就是 `paddlepaddle`） |
| `paddlepaddle-gpu` | `>=3.3,<3.4` | GPU 版框架，**和 `paddlepaddle` 二选一**，从飞桨官方源装，见下文 |

核心包只依赖 Pillow；PaddleOCR 放在 extra `cpu` 里，并且在模型加载时才 import，所以不装它也能跑测试。

本服务对 PaddleOCR 3.x 的用法（`src/eagle_ocr/engine.py`）：

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="ch", device="cpu",
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, enable_mkldnn=False)
for res in ocr.predict(bgr_ndarray):          # 3.x 用 predict()，不再是 2.x 的 ocr.ocr()
    res["rec_texts"], res["rec_scores"], res["rec_polys"]
```

> **注意**：paddlepaddle 3.3 在 CPU 上开 oneDNN（MKLDNN，默认开）推理 PP-OCRv6 会报
> `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`。
> 所以服务默认 `enable_mkldnn = false`。

## 安装

```bash
# 作为命令行工具安装（CPU）。--python 3.12：paddlepaddle 3.3 只有 cp39–cp313 的 wheel，
# 而 uv 默认可能选到 3.14，不指定会解析失败。GPU 版见下面「GPU」。
uv tool install --python 3.12 "/path/to/eagle-ocr[cpu]"
eagle-ocr version

# 开发
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e ".[cpu]"
```

## 启动

```bash
eagle-ocr serve                       # 127.0.0.1:8866，cpu，lang=ch
eagle-ocr serve --port 9000 --config ~/.config/eagle-ocr/config.toml
curl -s 127.0.0.1:8866/health         # ready=false 表示模型还在加载，此时 /v1/ocr 返回 503
```

启动后立即监听，模型在后台加载（已缓存时 2 秒左右，首次要先下载模型）。日志写 stderr，不记录请求头。

配置文件是 TOML，所有键见 [deploy/config.example.toml](deploy/config.example.toml)：
`host`、`port`、`device`（cpu / gpu / gpu:N）、`gpu_min_free_mb`、`lang`、`enable_mkldnn`、`max_images_per_request`、`max_image_bytes`、
`download_timeout_seconds`、`download_concurrency`、`allowed_hosts`（空 = 不限制，支持 `*.xhscdn.com`）、
`default_request_timeout_seconds`、`max_request_timeout_seconds`。

快速调用：

```bash
python3 -c 'import base64,json,sys;print(json.dumps({"images":[{"base64":base64.b64encode(open(sys.argv[1],"rb").read()).decode()}]}))' a.png \
  | curl -s -XPOST 127.0.0.1:8866/v1/ocr -H 'Content-Type: application/json' --data-binary @-
```

## 运行方式

### systemd 用户服务

```bash
uv tool install --python 3.12 "/path/to/eagle-ocr[cpu]"   # GPU：换成 [gpu]，并在 config.toml 写 device = "gpu"
mkdir -p ~/.config/systemd/user ~/.config/eagle-ocr
cp deploy/eagle-ocr.service ~/.config/systemd/user/
cp deploy/config.example.toml ~/.config/eagle-ocr/config.toml
systemctl --user daemon-reload && systemctl --user enable --now eagle-ocr
journalctl --user -u eagle-ocr -f
```

### Docker（CPU）

```bash
docker build -t eagle-ocr:cpu .
docker run -d --name eagle-ocr -p 127.0.0.1:8866:8866 -v eagle-ocr-models:/root/.paddlex eagle-ocr:cpu
```

容器内监听 `0.0.0.0`，宿主机用 `127.0.0.1:8866` 映射保持只对本机开放；模型缓存挂卷，重启不再下载。

## GPU（实测：RTX 4060 Laptop 8GB，WSL2，驱动 621.01）

GPU 版用 `gpu` extra：`paddlepaddle-gpu` 3.3（CUDA 12.6 构建）+ paddleocr 3.7。`paddlepaddle-gpu` 不在 PyPI，
`pyproject.toml` 里用 `[tool.uv.sources]` 把它（且只有它）指到飞桨官方 cu126 源（`explicit = true`，其他依赖照常走 PyPI），
所以一条命令就能装：

```bash
uv tool install --python 3.12 "/path/to/eagle-ocr[gpu]"     # 下载约 3GB（paddle 1.9GB + cuDNN/cuBLAS wheel）
```

- `cpu` 和 `gpu` extra 互斥（都提供 `paddle` 模块），不要同时装。
- 不需要装 CUDA toolkit / cuDNN：wheel 自带运行库，宿主机只要有驱动（驱动 ≥ 550.54 才能用 cu126 构建；
  WSL2 下装 Windows 侧支持 WSL 的 NVIDIA 驱动，`nvidia-smi` 能看到卡即可）。
- WSL2 下**不需要** `LD_LIBRARY_PATH`：`libcuda.so.1` 在 `/usr/lib/wsl/lib`，已在 ldconfig 缓存里（在不带任何环境变量的最小环境里实测过）。
  `paddle.utils.run_check()` 不设它时会打一条找不到 `libcuda.so` 的警告，但结论仍是 "works well on 1 GPU"。
- 在 venv 里装：`uv pip install -e ".[gpu]"`（uv 同样读 sources）；用普通 `pip` 时它不认 `[tool.uv.sources]`，要加
  `--extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/`。装完
  `python -c "import paddle; print(paddle.device.is_compiled_with_cuda()); paddle.utils.run_check()"`。

配置：

```toml
device = "gpu"          # 或 gpu:N 选卡
gpu_min_free_mb = 2048  # 加载前要求的空闲显存（MiB），0 = 不检查
```

**不静默回退 CPU**：`device` 是 gpu 但下面任一条件不满足时，模型不加载，`/health` 返回 `ready: false` 和
`load_error`，`/v1/ocr` 返回 503 `not_ready`（都已实测）：

| 情况 | `load_error` |
|---|---|
| 装的是 CPU 版 paddle | `GpuUnavailable: device=gpu，但当前 paddlepaddle 是 CPU 版（未编译 CUDA）；…` |
| 看不到 GPU（驱动 / `CUDA_VISIBLE_DEVICES=`） | `GpuUnavailable: device=gpu，但没检测到可用 GPU（device_count=0）：…` |
| `gpu:N` 超出范围 | `GpuUnavailable: device=gpu:1 超出范围：只有 1 块 GPU（gpu:0..gpu:0）` |
| 空闲显存低于 `gpu_min_free_mb` | `GpuUnavailable: gpu:0 显存不足：可用 7096 MiB，低于 gpu_min_free_mb=100000` |
| 加载 / 预热时显存溢出 | `GpuUnavailable: gpu 显存不足，模型加载 / 预热失败：…` |

（PaddleOCR 自己在 CPU 版 paddle 上收到 `device="gpu"` 不报错，会静默按 CPU 跑，所以服务在加载前自己检查。）

实测对比（同一批 PIL 生成的图，base64 走 `/v1/ocr`；单张为预热 1 次后 5 次的中位数 `elapsed_ms`；模型已缓存）：

| 项目 | CPU（paddlepaddle 3.3.1） | GPU（paddlepaddle-gpu 3.3.1，cu126） |
|---|---|---|
| 启动到 ready | 2.9s | 7.7s（含 CUDA 初始化） |
| 720×300，4 行 | 2237 ms | 74 ms |
| 1080×1440 卡片，19 行 | 15335 ms | 457 ms |
| 1080×2400 长图，33 行 | 27126 ms | 717 ms |
| 显存（nvidia-smi，减去空闲 551MiB） | 0 | 加载后 +270 MiB，跑完长图后 +2130 MiB |
| 识别文字 | — | 与 CPU 完全一致 |

GPU 版 Docker 未实测（本机 Docker 没有 nvidia runtime）。做法：基于飞桨官方 `paddlepaddle/paddle:3.x-gpu-cuda12.6-*`
镜像，`pip install ".[gpu]" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/`，
`docker run --gpus all` 启动，需要 nvidia-container-toolkit。

## 开发

```bash
.venv/bin/python -m unittest discover -s tests -v   # 不联网、不需要 PaddleOCR（用假引擎）
```

工程约定见 [AGENTS.md](AGENTS.md)。
