"""HTTP 层端到端测试：假引擎 + 本地图床，不访问外网、不需要 PaddleOCR。"""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.request

from helpers import (EMPTY_WIDTH, FAIL_WIDTH, FakeEngine, ImageHost, RunningService, b64,
                     png_bytes)

from eagle_ocr import __version__

_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_saved_env: dict[str, str] = {}


def setUpModule() -> None:  # 服务端下载走 urllib，会读代理环境变量；测试只连 127.0.0.1
    for k in _PROXY_VARS:
        if k in os.environ:
            _saved_env[k] = os.environ.pop(k)


def tearDownModule() -> None:
    os.environ.update(_saved_env)


class ServiceCase(unittest.TestCase):
    cfg: dict = {}

    def setUp(self) -> None:
        self.svc = RunningService(**self.cfg)
        self.addCleanup(self.svc.close)

    def ocr(self, images: list[dict], **extra) -> tuple[int, dict]:
        return self.svc.post("/v1/ocr", {"images": images, **extra})


class HealthAndReadiness(unittest.TestCase):
    def test_not_ready_then_ready(self) -> None:
        engine = FakeEngine()
        engine.load_gate.clear()  # 卡住加载
        svc = RunningService(engine=engine, wait_ready=False)
        self.addCleanup(svc.close)

        status, body = svc.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok", "service": "eagle-ocr", "api": 1,
                                "version": __version__, "engine": "fake",
                                "device": "cpu", "ready": False})
        status, body = svc.post("/v1/ocr", {"images": [{"base64": b64(png_bytes())}]})
        self.assertEqual(status, 503)
        self.assertEqual(body["api"], 1)
        self.assertEqual(body["error"]["code"], "not_ready")

        engine.load_gate.set()
        svc.load_thread.join(5)
        self.assertTrue(svc.get("/health")[1]["ready"])
        self.assertEqual(svc.post("/v1/ocr", {"images": [{"base64": b64(png_bytes())}]})[0], 200)

    def test_load_failure_stays_not_ready(self) -> None:
        svc = RunningService(engine=FakeEngine(load_error=RuntimeError("no model")))
        self.addCleanup(svc.close)
        _, health = svc.get("/health")
        self.assertFalse(health["ready"])
        self.assertIn("no model", health["load_error"])
        status, body = svc.post("/v1/ocr", {"images": [{"base64": b64(png_bytes())}]})
        self.assertEqual((status, body["error"]["code"]), (503, "not_ready"))

    def test_unknown_route_and_method(self) -> None:
        svc = RunningService()
        self.addCleanup(svc.close)
        self.assertEqual(svc.get("/nope")[0], 404)
        self.assertEqual(svc.get("/v1/ocr")[0], 405)
        self.assertEqual(svc.post("/health", {})[0], 405)


class BadRequests(ServiceCase):
    cfg = {"max_images_per_request": 3}

    def assertBad(self, status_body: tuple[int, dict]) -> None:
        status, body = status_body
        self.assertEqual(status, 400, body)
        self.assertEqual(body["api"], 1)
        self.assertEqual(body["error"]["code"], "bad_request")

    def test_invalid_json(self) -> None:
        self.assertBad(self.svc.post("/v1/ocr", None, raw=b"{not json"))

    def test_wrong_shapes(self) -> None:
        for body in ([], {}, {"images": []}, {"images": "x"}, {"images": [1]},
                     {"images": [{}]},
                     {"images": [{"url": "http://a/x", "base64": "AA=="}]},
                     {"images": [{"url": 3}]},
                     {"images": [{"base64": ""}]},
                     {"images": [{"url": "http://a/x", "headers": {"a": 1}}]},
                     {"images": [{"url": "http://a/x", "headers": {"a": "b\r\nX: y"}}]},
                     {"images": [{"base64": "AA=="}], "timeout_seconds": 0},
                     {"images": [{"base64": "AA=="}], "timeout_seconds": "60"}):
            with self.subTest(body=body):
                self.assertBad(self.svc.post("/v1/ocr", body))

    def test_too_many_images(self) -> None:
        status, body = self.ocr([{"base64": b64(png_bytes())}] * 4)
        self.assertBad((status, body))
        self.assertIn("too many images", body["error"]["message"])
        self.assertEqual(self.ocr([{"base64": b64(png_bytes())}] * 3)[0], 200)

    def test_unknown_fields_ignored(self) -> None:
        status, body = self.svc.post("/v1/ocr", {"images": [{"base64": b64(png_bytes()), "x": 1}],
                                                 "foo": "bar"})
        self.assertEqual(status, 200)
        self.assertIsNone(body["results"][0]["error"])


class OcrResults(ServiceCase):
    def test_reading_order_and_shape(self) -> None:
        status, body = self.ocr([{"base64": b64(png_bytes())}], timeout_seconds=30)
        self.assertEqual(status, 200)
        self.assertEqual(body["api"], 1)
        self.assertIsInstance(body["elapsed_ms"], int)
        (r,) = body["results"]
        self.assertEqual(r["index"], 0)
        self.assertIsNone(r["error"])
        self.assertEqual(r["text"], "标题\n你好 世界")
        self.assertEqual([ln["text"] for ln in r["lines"]], ["标题", "你好", "世界"])
        self.assertEqual(r["lines"][0]["box"], [[10, 10], [80, 10], [80, 30], [10, 30]])
        self.assertAlmostEqual(r["lines"][0]["score"], 0.99)
        self.assertIsInstance(r["elapsed_ms"], int)

    def test_data_uri_prefix_accepted(self) -> None:
        status, body = self.ocr([{"base64": "data:image/png;base64," + b64(png_bytes())}])
        self.assertEqual(status, 200)
        self.assertIsNone(body["results"][0]["error"])

    def test_no_text_is_not_error(self) -> None:
        status, body = self.ocr([{"base64": b64(png_bytes(width=EMPTY_WIDTH))}])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0], {"index": 0, "text": "", "lines": [],
                                              "elapsed_ms": body["results"][0]["elapsed_ms"],
                                              "error": None})

    def test_single_failures_isolated(self) -> None:
        images = [
            {"base64": b64(png_bytes())},                    # 0 ok
            {"base64": "!!!not-base64!!!"},                  # 1 bad_image
            {"base64": b64(b"definitely not an image")},     # 2 bad_image
            {"base64": b64(png_bytes(width=FAIL_WIDTH))},    # 3 ocr_failed
            {"url": "ftp://127.0.0.1/x.png"},                # 4 fetch_failed
            {"base64": b64(png_bytes())},                    # 5 ok
        ]
        status, body = self.ocr(images)
        self.assertEqual(status, 200)
        codes = [r["error"] and r["error"]["code"] for r in body["results"]]
        self.assertEqual(codes, [None, "bad_image", "bad_image", "ocr_failed", "fetch_failed", None])
        self.assertEqual([r["index"] for r in body["results"]], list(range(6)))
        for r in body["results"]:
            if r["error"]:
                self.assertEqual((r["text"], r["lines"]), ("", []))
                self.assertTrue(r["error"]["message"])
        self.assertEqual(body["results"][5]["text"], "标题\n你好 世界")


class SizeLimits(ServiceCase):
    cfg = {"max_image_bytes": 1024}

    def test_base64_too_large(self) -> None:
        status, body = self.ocr([{"base64": b64(b"x" * 1025)}, {"base64": b64(png_bytes())}])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["error"]["code"], "too_large")
        self.assertIsNone(body["results"][1]["error"])

    def test_download_too_large(self) -> None:
        host = ImageHost()
        self.addCleanup(host.close)
        status, body = self.ocr([{"url": host.url("/big-declared")},
                                 {"url": host.url("/big-stream")},
                                 {"url": host.url("/ok.png")}])
        self.assertEqual(status, 200)
        codes = [r["error"] and r["error"]["code"] for r in body["results"]]
        self.assertEqual(codes, ["too_large", "too_large", None])


class Downloads(ServiceCase):
    cfg = {"download_timeout_seconds": 1.0}

    def setUp(self) -> None:
        super().setUp()
        self.host = ImageHost()
        self.addCleanup(self.host.close)

    def codes(self, body: dict) -> list:
        return [r["error"] and r["error"]["code"] for r in body["results"]]

    def test_download_ok_and_failures(self) -> None:
        status, body = self.ocr([{"url": self.host.url("/ok.png")},
                                 {"url": self.host.url("/notfound")},
                                 {"url": self.host.url("/notimage")},
                                 {"url": "http://127.0.0.1:1/refused.png"}])
        self.assertEqual(status, 200)
        self.assertEqual(self.codes(body), [None, "fetch_failed", "bad_image", "fetch_failed"])
        self.assertEqual(body["results"][1]["error"]["message"], "HTTP 404")
        self.assertEqual(body["results"][0]["text"], "标题\n你好 世界")

    def test_request_headers_forwarded(self) -> None:
        _, body = self.ocr([{"url": self.host.url("/need-referer"),
                             "headers": {"Referer": "https://example.com/", "Cookie": "a=secret"}},
                            {"url": self.host.url("/need-referer")}])
        self.assertEqual(self.codes(body), [None, "fetch_failed"])
        got = self.host.received_headers["/need-referer"]
        self.assertIn("eagle-ocr", got["User-Agent"])

    def test_download_timeout(self) -> None:
        _, body = self.ocr([{"url": self.host.url("/slow")}, {"base64": b64(png_bytes())}])
        self.assertEqual(self.codes(body), ["fetch_failed", None])
        self.assertIn("timed out", body["results"][0]["error"]["message"])

    def test_redirects(self) -> None:
        _, body = self.ocr([{"url": self.host.url("/redirect-ok")},
                            {"url": self.host.url("/redirect-ftp")}])
        self.assertEqual(self.codes(body), [None, "fetch_failed"])


class OverallTimeout(ServiceCase):
    cfg = {"download_timeout_seconds": 10.0}

    def setUp(self) -> None:
        super().setUp()
        self.host = ImageHost()
        self.addCleanup(self.host.close)

    def test_deadline_marks_timeout_others_succeed(self) -> None:
        status, body = self.ocr([{"base64": b64(png_bytes())}, {"url": self.host.url("/slow")}],
                                timeout_seconds=0.5)
        self.assertEqual(status, 200)
        self.assertIsNone(body["results"][0]["error"])
        self.assertEqual(body["results"][1]["error"]["code"], "timeout")
        self.assertLess(body["elapsed_ms"], 2500)


class AllowedHosts(ServiceCase):
    cfg = {"allowed_hosts": ("localhost", "*.xhscdn.com")}

    def test_host_whitelist(self) -> None:
        host = ImageHost()
        self.addCleanup(host.close)
        port = host.server.server_address[1]
        _, body = self.ocr([{"url": host.url("/ok.png")},  # 127.0.0.1 不在白名单
                            {"url": f"http://localhost:{port}/ok.png"}])
        self.assertEqual(body["results"][0]["error"]["code"], "fetch_failed")
        self.assertIn("host not allowed", body["results"][0]["error"]["message"])
        self.assertIsNone(body["results"][1]["error"])


class Concurrency(ServiceCase):
    cfg = {"download_concurrency": 4}

    def setUp(self) -> None:
        self.svc = RunningService(engine=FakeEngine(recognize_delay=0.05), **self.cfg)
        self.addCleanup(self.svc.close)

    def test_inference_is_serial_across_requests(self) -> None:
        results: list[int] = []

        def worker() -> None:
            results.append(self.ocr([{"base64": b64(png_bytes())}] * 3)[0])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(results, [200] * 4)
        self.assertEqual(self.svc.engine.calls, 12)
        self.assertEqual(self.svc.engine.max_active, 1)

    def test_failed_item_elapsed_is_its_own(self) -> None:
        # 回归：失败项曾把前一张的 OCR 时间算进自己的 elapsed_ms
        self.svc.engine.recognize_delay = 0.5
        _, body = self.ocr([{"base64": b64(png_bytes())}, {"url": "ftp://127.0.0.1/x"}])
        self.assertGreaterEqual(body["results"][0]["elapsed_ms"], 450)
        self.assertEqual(body["results"][1]["error"]["code"], "fetch_failed")
        self.assertLess(body["results"][1]["elapsed_ms"], 200)


class Logging(ServiceCase):
    def test_cookie_not_logged(self) -> None:
        host = ImageHost()
        self.addCleanup(host.close)
        with self.assertLogs("eagle_ocr", level="DEBUG") as cm:
            req = urllib.request.Request(
                self.svc.base + "/v1/ocr", method="POST",
                data=json.dumps({"images": [{"url": host.url("/notfound"),
                                             "headers": {"Cookie": "web_session=SECRET123"}}]}).encode(),
                headers={"Cookie": "web_session=SECRET456"})
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=10).read()
        joined = "\n".join(cm.output)
        self.assertNotIn("SECRET", joined)


if __name__ == "__main__":
    unittest.main()
