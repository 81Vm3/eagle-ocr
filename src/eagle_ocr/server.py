"""标准库 HTTP 层：ThreadingHTTPServer + JSON 路由。"""

from __future__ import annotations

import json
import logging
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import OcrService, error_body

log = logging.getLogger(__name__)


def max_body_bytes(service: OcrService) -> int:
    """能容纳 max_images_per_request 张 max_image_bytes 图片（base64 膨胀 4/3）+ 1MB 余量。"""
    cfg = service.config
    return cfg.max_images_per_request * math.ceil(cfg.max_image_bytes * 4 / 3 + 4) + 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "eagle-ocr"
    protocol_version = "HTTP/1.1"
    service: OcrService  # 由 make_server 注入到子类

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # 只记方法 / 路径 / 状态，不记请求头（可能含 cookie）。健康检查太频繁，降到 DEBUG。
        level = logging.DEBUG if self.path.startswith("/health") else logging.INFO
        log.log(level, "%s - %s", self.address_string(), format % args)

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def do_GET(self) -> None:  # noqa: N802
        if self._path() == "/health":
            self._send(200, self.service.health())
        elif self._path() == "/v1/ocr":
            self._send(405, error_body("method_not_allowed", "use POST"))
        else:
            self._send(404, error_body("not_found", f"no route: {self._path()}"))

    def do_POST(self) -> None:  # noqa: N802
        if self._path() != "/v1/ocr":
            code = 405 if self._path() == "/health" else 404
            self._send(code, error_body("method_not_allowed" if code == 405 else "not_found",
                                        f"no route: POST {self._path()}"))
            return
        try:
            body = self._read_json()
        except ValueError as exc:
            self.close_connection = True  # 请求体可能没读完，别复用连接
            self._send(400, error_body("bad_request", str(exc)))
            return
        try:
            status, payload = self.service.ocr(body)
        except Exception:  # noqa: BLE001
            log.exception("internal error")
            status, payload = 500, error_body("internal", "internal server error")
        self._send(status, payload)

    def _read_json(self) -> object:
        if self.headers.get("Transfer-Encoding", "").lower() not in ("", "identity"):
            raise ValueError("chunked request body not supported; send Content-Length")
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            raise ValueError("Content-Length required")
        n = int(length)
        if n > max_body_bytes(self.service):
            raise ValueError(f"request body too large: {n} bytes")
        raw = self.rfile.read(n)
        if len(raw) != n:
            raise ValueError("incomplete request body")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON: {exc}") from None


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(service: OcrService, host: str, port: int) -> Server:
    handler = type("BoundHandler", (Handler,), {"service": service})
    return Server((host, port), handler)
