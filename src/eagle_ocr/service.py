"""业务核心：模型后台加载、请求校验、并发下载 + 串行推理、单张失败隔离。

与 HTTP 无关，返回 (status, payload)，便于测试。
"""

from __future__ import annotations

import io
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from PIL import Image, UnidentifiedImageError

from . import API_VERSION, SERVICE_NAME, __version__
from .config import Config
from .engine import OcrEngine, join_text, sort_reading_order
from .fetch import ItemError, decode_base64, download

log = logging.getLogger(__name__)


class BadRequest(Exception):
    pass


def error_body(code: str, message: str) -> dict:
    return {"api": API_VERSION, "error": {"code": code, "message": message}}


class OcrService:
    def __init__(self, config: Config, engine: OcrEngine) -> None:
        self.config = config
        self.engine = engine
        self._ready = threading.Event()
        self._load_error: str | None = None
        self._ocr_lock = threading.Lock()  # PaddleOCR 推理串行
        # 全局共享的下载池：download_concurrency 是整个进程的下载并发上限。
        self._pool = ThreadPoolExecutor(max_workers=config.download_concurrency,
                                        thread_name_prefix="fetch")

    # ---------- 生命周期 ----------

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def start_loading(self) -> threading.Thread:
        thread = threading.Thread(target=self._load, name="engine-load", daemon=True)
        thread.start()
        return thread

    def _load(self) -> None:
        start = time.monotonic()
        log.info("loading engine %s (device=%s, lang=%s)", self.engine.name,
                 self.config.device, self.config.lang)
        try:
            self.engine.load()
        except BaseException as exc:  # noqa: BLE001 - 加载失败要记录，服务保持 ready=false
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.exception("engine load failed")
            return
        log.info("engine ready in %.1fs", time.monotonic() - start)
        self._ready.set()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------- 接口 ----------

    def health(self) -> dict:
        body = {"status": "ok", "service": SERVICE_NAME, "api": API_VERSION,
                "version": __version__, "engine": self.engine.name,
                "device": self.config.device, "ready": self.ready}
        if self._load_error:
            body["load_error"] = self._load_error  # 扩展字段，客户端按约定忽略
        return body

    def ocr(self, body: Any) -> tuple[int, dict]:
        if not self.ready:
            msg = "model is loading" if self._load_error is None else f"model load failed: {self._load_error}"
            return 503, error_body("not_ready", msg)
        try:
            items, timeout = self._validate(body)
        except BadRequest as exc:
            return 400, error_body("bad_request", str(exc))
        return 200, self._run(items, timeout)

    # ---------- 实现 ----------

    def _validate(self, body: Any) -> tuple[list[dict], float]:
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        images = body.get("images")
        if not isinstance(images, list) or not images:
            raise BadRequest("images must be a non-empty array")
        limit = self.config.max_images_per_request
        if len(images) > limit:
            raise BadRequest(f"too many images: {len(images)} > {limit}")
        items = []
        for i, img in enumerate(images):
            if not isinstance(img, dict):
                raise BadRequest(f"images[{i}] must be an object")
            url, b64 = img.get("url"), img.get("base64")
            if (url is None) == (b64 is None):
                raise BadRequest(f"images[{i}] must have exactly one of url / base64")
            if url is not None and (not isinstance(url, str) or not url):
                raise BadRequest(f"images[{i}].url must be a non-empty string")
            if b64 is not None and (not isinstance(b64, str) or not b64):
                raise BadRequest(f"images[{i}].base64 must be a non-empty string")
            headers = img.get("headers") or {}
            if not isinstance(headers, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                raise BadRequest(f"images[{i}].headers must be an object of strings")
            if any(c in k + v for k, v in headers.items() for c in "\r\n"):
                raise BadRequest(f"images[{i}].headers must not contain CR/LF")
            items.append({"url": url, "base64": b64, "headers": headers})
        timeout = body.get("timeout_seconds", self.config.default_request_timeout_seconds)
        if timeout is None:
            timeout = self.config.default_request_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise BadRequest("timeout_seconds must be a positive number")
        return items, min(float(timeout), self.config.max_request_timeout_seconds)

    def _fetch(self, item: dict, deadline: float) -> bytes:
        if item["base64"] is not None:
            return decode_base64(item["base64"], self.config.max_image_bytes)
        if time.monotonic() >= deadline:
            raise ItemError("timeout", "request deadline exceeded before download started")
        return download(item["url"], item["headers"], self.config, deadline)

    def _run(self, items: list[dict], timeout: float) -> dict:
        start = time.monotonic()
        deadline = start + timeout
        results: list[dict | None] = [None] * len(items)
        started: list[float] = [start] * len(items)

        def submit(i: int) -> Future:
            def task() -> bytes:
                started[i] = time.monotonic()
                try:
                    return self._fetch(items[i], deadline)
                except ItemError as exc:
                    # 记下失败时刻：结果里的 elapsed_ms 反映这张自己的耗时，而不是排队等别的图。
                    exc.finished = time.monotonic()
                    raise
            return self._pool.submit(task)

        pending: dict[Future, int] = {submit(i): i for i in range(len(items))}
        # 按下载完成顺序依次推理：下载并发，推理在锁内串行。
        while pending:
            left = deadline - time.monotonic()
            done, _ = wait(pending, timeout=max(left, 0), return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in sorted(done, key=lambda f: pending[f]):
                i = pending.pop(fut)
                results[i] = self._process_one(i, fut, started[i], deadline)
        for fut, i in pending.items():
            fut.cancel()  # 没开始的下载不再做；已在下载的受单张超时约束
            results[i] = _item(i, started[i], ItemError("timeout", "request deadline exceeded"))
        for i, r in enumerate(results):
            if r is None:  # pragma: no cover - 防御
                results[i] = _item(i, start, ItemError("ocr_failed", "not processed"))
        return {"api": API_VERSION, "results": results,
                "elapsed_ms": int((time.monotonic() - start) * 1000)}

    def _process_one(self, i: int, fut: Future, t0: float, deadline: float) -> dict:
        try:
            data = fut.result()
            image = _decode_image(data)
            left = deadline - time.monotonic()
            if left <= 0 or not self._ocr_lock.acquire(timeout=left):
                raise ItemError("timeout", "request deadline exceeded while waiting for OCR")
            try:
                if time.monotonic() >= deadline:
                    raise ItemError("timeout", "request deadline exceeded while waiting for OCR")
                try:
                    lines = self.engine.recognize(image)
                except Exception as exc:  # noqa: BLE001
                    log.warning("ocr failed for image %d: %s", i, exc, exc_info=True)
                    raise ItemError("ocr_failed", f"{type(exc).__name__}: {exc}") from None
            finally:
                self._ocr_lock.release()
        except ItemError as exc:
            log.info("image %d failed: %s %s", i, exc.code, exc.message)
            return _item(i, t0, exc, getattr(exc, "finished", None))
        except Exception as exc:  # noqa: BLE001 - 兜底：单张异常不能拖垮整个请求
            log.exception("unexpected error for image %d", i)
            return _item(i, t0, ItemError("ocr_failed", f"{type(exc).__name__}: {exc}"))
        rows = sort_reading_order(lines)
        return {"index": i, "text": join_text(rows),
                "lines": [ln.to_json() for row in rows for ln in row],
                "elapsed_ms": int((time.monotonic() - t0) * 1000), "error": None}


def _decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return img.convert("RGB")
    except Image.DecompressionBombError as exc:
        # 像素数超过 PIL 上限（解压炸弹）：字节数没超，按解码失败处理。
        raise ItemError("bad_image", f"image has too many pixels: {exc}") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ItemError("bad_image", f"cannot decode image: {exc}") from None


def _item(i: int, t0: float, err: ItemError, end: float | None = None) -> dict:
    end = time.monotonic() if end is None else end
    return {"index": i, "text": "", "lines": [],
            "elapsed_ms": int(max(end - t0, 0) * 1000),
            "error": {"code": err.code, "message": err.message}}
