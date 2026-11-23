"""测试工具：假引擎、图片生成、本地 HTTP 图床、起停服务。"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from eagle_ocr.config import Config
from eagle_ocr.engine import Line
from eagle_ocr.server import make_server
from eagle_ocr.service import OcrService


def png_bytes(width: int = 40, height: int = 20, color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


FAIL_WIDTH = 13  # 假引擎遇到这个宽度的图就抛异常
EMPTY_WIDTH = 17  # 假引擎遇到这个宽度的图返回无文字


class FakeEngine:
    """可注入的假引擎：返回固定的乱序行，记录并发度，可设置加载 / 识别延迟。"""

    name = "fake"

    def __init__(self, load_delay: float = 0.0, recognize_delay: float = 0.0,
                 load_error: Exception | None = None) -> None:
        self.load_delay = load_delay
        self.recognize_delay = recognize_delay
        self.load_error = load_error
        self.load_gate = threading.Event()
        self.load_gate.set()
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        self.load_gate.wait()
        time.sleep(self.load_delay)
        if self.load_error:
            raise self.load_error

    def recognize(self, image: Image.Image) -> list[Line]:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.recognize_delay)
            if image.width == FAIL_WIDTH:
                raise RuntimeError("boom")
            if image.width == EMPTY_WIDTH:
                return []
            # 故意乱序：第二行右、第一行、第二行左
            return [
                Line("世界", 0.9, [[60, 52], [100, 52], [100, 70], [60, 70]]),
                Line("标题", 0.99, [[10, 10], [80, 10], [80, 30], [10, 30]]),
                Line("你好", 0.95, [[10, 50], [50, 50], [50, 72], [10, 72]]),
            ]
        finally:
            with self._lock:
                self.active -= 1


class ImageHost:
    """本地图床：127.0.0.1 随机端口，按路径返回不同响应，并记录收到的请求头。"""

    def __init__(self) -> None:
        self.received_headers: dict[str, dict[str, str]] = {}
        host = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # 静默
                pass

            def do_GET(self) -> None:  # noqa: N802
                host.received_headers[self.path] = dict(self.headers.items())
                path = self.path
                if path == "/ok.png":
                    self._send(200, png_bytes())
                elif path == "/notfound":
                    self._send(404, b"nope")
                elif path == "/notimage":
                    self._send(200, b"this is not an image")
                elif path == "/big-declared":
                    self._send(200, b"x" * 2048)
                elif path == "/big-stream":
                    # 没有 Content-Length，流式发 4KB，靠读取过程中计数截断
                    self.send_response(200)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for _ in range(64):
                        try:
                            self.wfile.write(b"x" * 64)
                            self.wfile.flush()
                        except OSError:
                            return
                    self.close_connection = True
                elif path == "/slow":
                    time.sleep(3)
                    self._send(200, png_bytes())
                elif path == "/need-referer":
                    if self.headers.get("Referer") == "https://example.com/":
                        self._send(200, png_bytes())
                    else:
                        self._send(403, b"forbidden")
                elif path == "/redirect-ftp":
                    self.send_response(302)
                    self.send_header("Location", "ftp://127.0.0.1/x")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif path == "/redirect-ok":
                    self.send_response(302)
                    self.send_header("Location", "/ok.png")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self._send(404, b"")

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}{path}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class RunningService:
    """在随机端口上起 eagle-ocr，测试结束关闭。"""

    def __init__(self, engine: FakeEngine | None = None, wait_ready: bool = True, **cfg) -> None:
        self.engine = engine or FakeEngine()
        self.config = Config(host="127.0.0.1", port=0, **cfg)
        self.service = OcrService(self.config, self.engine)
        self.server = make_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.load_thread = self.service.start_loading()
        if wait_ready:
            self.load_thread.join(5)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def get(self, path: str) -> tuple[int, dict]:
        return _call(urllib.request.Request(self.base + path))

    def post(self, path: str, body: object, raw: bytes | None = None) -> tuple[int, dict]:
        data = raw if raw is not None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        return _call(req)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.service.close()


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _call(req: urllib.request.Request) -> tuple[int, dict]:
    try:
        with _NO_PROXY.open(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.loads(exc.read())
