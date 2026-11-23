# eagle-ocr HTTP 接口 v1（Commander 定稿，两边按此实现；要改先找 Commander）

默认监听 `127.0.0.1:8866`。请求和响应都是 JSON（UTF-8）。未知字段一律忽略。

## GET /health

200：
```json
{"status": "ok", "service": "eagle-ocr", "api": 1, "version": "0.1.0",
 "engine": "paddleocr", "device": "cpu", "ready": true}
```
- `ready=false` 表示模型还在加载（此时 POST /v1/ocr 返回 503 + 错误体）。

## POST /v1/ocr

请求：
```json
{
  "images": [
    {"url": "https://sns-webpic-qc.xhscdn.com/...", "headers": {"Referer": "https://www.xiaohongshu.com/"}},
    {"base64": "<图片字节的 base64>"}
  ],
  "timeout_seconds": 60
}
```
- 每项二选一：`url`（只允许 http/https）或 `base64`。`headers` 可选，只用于下载该图。
- 单次最多 `max_images_per_request` 张（服务端配置，默认 20），超出 → 400 `bad_request`。

200：
```json
{
  "api": 1,
  "results": [
    {"index": 0, "text": "整张图按阅读顺序拼好的文字，行之间用 \n",
     "lines": [{"text": "一行", "score": 0.98, "box": [[x,y],[x,y],[x,y],[x,y]]}],
     "elapsed_ms": 812, "error": null},
    {"index": 1, "text": "", "lines": [], "elapsed_ms": 0,
     "error": {"code": "fetch_failed", "message": "HTTP 403"}}
  ],
  "elapsed_ms": 1650
}
```
- `results` 与 `images` 一一对应（按 `index`）。**单张失败不影响其他张**，整体仍 200。
- 单张 `error.code`：`fetch_failed`（下载失败 / 非 2xx / 超时）、`too_large`（超过 `max_image_bytes`，默认 20MB）、`bad_image`（解码失败）、`ocr_failed`（引擎异常）、`timeout`（整体时限到了还没轮到这张）。
- 图里没文字：`text: ""`、`lines: []`、`error: null`（不是错误）。

整体错误（非 200）：`{"api": 1, "error": {"code": "...", "message": "..."}}`
- 400 `bad_request`（JSON 不合法、字段不对、超张数）；503 `not_ready`（模型未加载完）；500 `internal`。
