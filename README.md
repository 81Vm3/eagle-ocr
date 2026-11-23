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
# 作为命令行工具安装（CPU）
uv tool install "/path/to/eagle-ocr[cpu]"
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
`host`、`port`、`device`（cpu / gpu / gpu:N）、`lang`、`enable_mkldnn`、`max_images_per_request`、`max_image_bytes`、
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
uv tool install "/path/to/eagle-ocr[cpu]"
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

## GPU（例：RTX 4060）

GPU 版要把 `paddlepaddle` 换成 `paddlepaddle-gpu`（两者不能同时装），从飞桨官方源安装。
不用关心本机 CUDA toolkit 版本，只看驱动：驱动 ≥ 550.54 选 cu126 源（RTX 4060 + 新驱动选这个），否则选 cu118。

```bash
# 不要带 [cpu] extra
uv tool install /path/to/eagle-ocr \
  --with "paddleocr>=3.7,<3.8" \
  --with "paddlepaddle-gpu>=3.3,<3.4" \
  --index https://www.paddlepaddle.org.cn/packages/stable/cu126/ --index-strategy unsafe-best-match

# 或在 venv 里
uv pip install -e . "paddleocr>=3.7,<3.8"
uv pip install "paddlepaddle-gpu>=3.3,<3.4" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -c "import paddle; paddle.utils.run_check()"   # 确认能用 GPU
```

然后配置 `device = "gpu"`（或 `gpu:0`）。GPU 版未在本仓库实测。WSL2 下需要 Windows 侧装好支持 WSL 的 NVIDIA 驱动（`nvidia-smi` 能看到卡）。
GPU Docker 镜像可基于飞桨官方 `paddlepaddle/paddle:3.x-gpu-cuda12.6-*` 镜像，用 `docker run --gpus all` 启动。

## 开发

```bash
.venv/bin/python -m unittest discover -s tests -v   # 不联网、不需要 PaddleOCR（用假引擎）
```

工程约定见 [AGENTS.md](AGENTS.md)。
