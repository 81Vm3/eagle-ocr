"""服务配置：默认值 + TOML 文件 + 命令行覆盖。"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEVICE_RE = re.compile(r"^(cpu|gpu(:\d+)?)$")


class ConfigError(ValueError):
    """配置文件或参数不合法。"""


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8866
    device: str = "cpu"
    lang: str = "ch"
    # CPU 推理是否开 oneDNN（MKLDNN）。paddlepaddle 3.3 下开启会让 PP-OCRv6 推理报错，默认关。
    enable_mkldnn: bool = False
    # device=gpu 时加载前要求的最小空闲显存（MiB），不够就加载失败（不回退 CPU）。0 = 不检查。
    gpu_min_free_mb: int = 2048
    max_images_per_request: int = 20
    max_image_bytes: int = 20 * 1024 * 1024
    download_timeout_seconds: float = 15.0
    download_concurrency: int = 4
    allowed_hosts: tuple[str, ...] = ()
    # 请求里没给 timeout_seconds 时的整体时限，以及允许的上限。
    default_request_timeout_seconds: float = 60.0
    max_request_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        _check(isinstance(self.host, str) and self.host != "", "host 必须是非空字符串")
        _check(_is_int(self.port) and 0 <= self.port <= 65535, "port 必须是 0-65535 的整数")
        _check(isinstance(self.device, str) and bool(_DEVICE_RE.match(self.device)),
               "device 只能是 cpu、gpu 或 gpu:N")
        _check(isinstance(self.lang, str) and self.lang != "", "lang 必须是非空字符串")
        _check(isinstance(self.enable_mkldnn, bool), "enable_mkldnn 必须是 true / false")
        _check(_is_int(self.gpu_min_free_mb) and self.gpu_min_free_mb >= 0, "gpu_min_free_mb 必须是非负整数")
        for name in ("max_images_per_request", "max_image_bytes", "download_concurrency"):
            value = getattr(self, name)
            _check(_is_int(value) and value > 0, f"{name} 必须是正整数")
        for name in ("download_timeout_seconds", "default_request_timeout_seconds",
                     "max_request_timeout_seconds"):
            value = getattr(self, name)
            _check(_is_number(value) and value > 0, f"{name} 必须是正数")
        _check(self.default_request_timeout_seconds <= self.max_request_timeout_seconds,
               "default_request_timeout_seconds 不能大于 max_request_timeout_seconds")
        hosts = self.allowed_hosts
        _check(isinstance(hosts, (list, tuple)) and all(isinstance(h, str) and h for h in hosts),
               "allowed_hosts 必须是非空字符串列表")
        object.__setattr__(self, "allowed_hosts", tuple(h.strip().lower() for h in hosts))

    def replace(self, **changes: Any) -> Config:
        return dataclasses.replace(self, **{k: v for k, v in changes.items() if v is not None})


def load_config(path: str | Path | None) -> Config:
    """读取 TOML 配置（顶层平铺键）；path 为空时返回默认配置。未知键报错，防止拼写错误被静默忽略。"""
    if path is None:
        return Config()
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise ConfigError(f"读取配置文件失败：{path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"配置文件不是合法 TOML：{path}: {exc}") from exc
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"配置文件含未知键：{', '.join(unknown)}")
    if "allowed_hosts" in data and isinstance(data["allowed_hosts"], list):
        data["allowed_hosts"] = tuple(data["allowed_hosts"])
    try:
        return Config(**data)
    except TypeError as exc:  # pragma: no cover - 已由 unknown 检查覆盖
        raise ConfigError(str(exc)) from exc


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)
