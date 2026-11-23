"""取图：base64 解码与 http/https 下载（流式读、大小上限、超时、host 白名单）。"""

from __future__ import annotations

import base64
import binascii
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import Config

_CHUNK = 64 * 1024
_MAX_REDIRECTS = 5
# 跨 host 重定向时去掉的敏感头，防止 cookie 被带到别的站。
_SENSITIVE_HEADERS = ("Cookie", "Authorization", "Proxy-authorization")
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; eagle-ocr/0.1)"


class ItemError(Exception):
    """单张图片失败；code 取值见 docs/api.md。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def decode_base64(data: str, max_bytes: int) -> bytes:
    """解码 base64；兼容 `data:image/...;base64,` 前缀和换行。"""
    if data.startswith("data:"):
        data = data.partition(",")[2]
    data = "".join(data.split())
    if len(data) // 4 * 3 > max_bytes + 3:
        raise ItemError("too_large", f"image exceeds {max_bytes} bytes")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ItemError("bad_image", f"invalid base64: {exc}") from exc
    if len(raw) > max_bytes:
        raise ItemError("too_large", f"image exceeds {max_bytes} bytes")
    if not raw:
        raise ItemError("bad_image", "empty image")
    return raw


def host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """allowed 为空表示不限制；支持精确匹配和 `*.example.com`（匹配所有子域，不含 example.com 本身）。"""
    if not allowed:
        return True
    host = host.lower().rstrip(".")
    for pattern in allowed:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


def check_url(url: str, config: Config) -> str:
    """校验 scheme 与 host 白名单，返回小写 host。不合规 → fetch_failed。"""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname or ""
    except ValueError as exc:
        raise ItemError("fetch_failed", f"invalid url: {exc}") from exc
    if parts.scheme.lower() not in ("http", "https"):
        raise ItemError("fetch_failed", f"scheme not allowed: {parts.scheme or '(none)'}")
    if not host:
        raise ItemError("fetch_failed", "url has no host")
    if not host_allowed(host, config.allowed_hosts):
        raise ItemError("fetch_failed", f"host not allowed: {host}")
    return host


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = _MAX_REDIRECTS

    def __init__(self, config: Config) -> None:
        self.config = config

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        newurl = urllib.parse.urljoin(req.full_url, newurl)
        new_host = check_url(newurl, self.config)  # 重定向目标同样受 scheme / 白名单约束
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and new_host != (urllib.parse.urlsplit(req.full_url).hostname or "").lower():
            for name in _SENSITIVE_HEADERS:
                new_req.headers.pop(name, None)
                new_req.unredirected_hdrs.pop(name, None)
        return new_req


def download(url: str, headers: dict[str, str], config: Config, deadline: float) -> bytes:
    """下载一张图。deadline 是整体时限（time.monotonic()），到了 → timeout；单张超时 → fetch_failed。"""
    check_url(url, config)
    start = time.monotonic()
    download_end = start + config.download_timeout_seconds
    end = min(download_end, deadline)

    def remaining() -> float:
        left = end - time.monotonic()
        if left <= 0:
            if deadline <= download_end:
                raise ItemError("timeout", "request deadline exceeded while downloading")
            raise ItemError("fetch_failed", f"download timed out after {config.download_timeout_seconds:g}s")
        return left

    req_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "image/*,*/*;q=0.8"}
    req_headers.update(headers)
    request = urllib.request.Request(url, headers=req_headers, method="GET")
    opener = urllib.request.build_opener(_RedirectHandler(config))
    try:
        resp = opener.open(request, timeout=remaining())
    except ItemError:
        raise
    except urllib.error.HTTPError as exc:
        exc.close()
        raise ItemError("fetch_failed", f"HTTP {exc.code}") from None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
        if time.monotonic() >= end:
            remaining()  # 抛出对应的超时错误
        reason = getattr(exc, "reason", exc)
        raise ItemError("fetch_failed", f"{type(exc).__name__}: {reason}") from None

    with resp:
        status = getattr(resp, "status", 200)
        if not 200 <= status < 300:
            raise ItemError("fetch_failed", f"HTTP {status}")
        length = resp.headers.get("Content-Length")
        if length and length.isdigit() and int(length) > config.max_image_bytes:
            raise ItemError("too_large", f"image exceeds {config.max_image_bytes} bytes (Content-Length {length})")
        buf = bytearray()
        sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
        try:
            while True:
                left = remaining()  # 每块之前检查时限，并收紧 socket 超时
                if resp.isclosed():  # 按 Content-Length 读完后 http.client 会自动关闭
                    break
                if sock is not None:
                    try:
                        sock.settimeout(left)
                    except OSError:  # socket 已被关闭，交给 read1 处理
                        pass
                chunk = resp.read1(_CHUNK)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > config.max_image_bytes:
                    raise ItemError("too_large", f"image exceeds {config.max_image_bytes} bytes")
        except ItemError:
            raise
        except (http.client.HTTPException, OSError, ValueError) as exc:
            if time.monotonic() >= end:
                remaining()
            raise ItemError("fetch_failed", f"{type(exc).__name__}: {exc}") from None
    if not buf:
        raise ItemError("bad_image", "empty response body")
    return bytes(buf)
