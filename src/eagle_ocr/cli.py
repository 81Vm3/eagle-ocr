"""命令行入口：eagle-ocr serve / eagle-ocr version。"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from . import __version__
from .config import ConfigError, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eagle-ocr", description="常驻 PaddleOCR HTTP 服务")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--host", help="监听地址（默认 127.0.0.1）")
    serve.add_argument("--port", type=int, help="监听端口（默认 8866）")
    serve.add_argument("--config", help="TOML 配置文件路径")
    serve.add_argument("--log-level", default="INFO", help="日志级别（默认 INFO）")
    sub.add_parser("version", help="打印版本")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    return _serve(args)


def _serve(args: argparse.Namespace) -> int:
    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # PaddleX 用 httpx 下模型，INFO 级会把每个 HTTP 请求都打出来。
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("eagle_ocr")
    try:
        config = load_config(args.config).replace(host=args.host, port=args.port)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    # 跳过 PaddleX 每次启动时对模型源的联网检查（模型已缓存时可离线启动）。
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from .engine import PaddleEngine
    from .server import make_server
    from .service import OcrService

    service = OcrService(config, PaddleEngine(lang=config.lang, device=config.device,
                                                enable_mkldnn=config.enable_mkldnn))
    try:
        server = make_server(service, config.host, config.port)
    except OSError as exc:
        log.error("cannot listen on %s:%s: %s", config.host, config.port, exc)
        return 1

    stop = threading.Event()

    def _stop(signum, frame):  # noqa: ANN001, ARG001
        stop.set()

    def _install_handlers() -> None:
        # 无条件重装：paddle 在 C 层用 sigaction 改处理器，signal.getsignal() 看不到。
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _stop)

    _install_handlers()
    server_thread = threading.Thread(target=server.serve_forever, name="http", daemon=True)
    server_thread.start()
    service.start_loading()
    log.info("eagle-ocr %s listening on http://%s:%s", __version__, *server.server_address[:2])
    try:
        # import paddle 会把 SIGTERM/SIGINT 换成它自己的 C++ 处理器（打崩溃栈并 abort），
        # 主线程定期把我们的处理器装回去，保证 systemctl stop / docker stop 能优雅退出。
        while not stop.wait(0.5):
            _install_handlers()
    except KeyboardInterrupt:  # pragma: no cover - 处理器装回前的 Ctrl-C
        pass
    log.info("shutting down")
    server.shutdown()
    server.server_close()
    service.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
