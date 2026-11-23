"""device=gpu 的加载前检查：用假 paddle / 假 paddleocr 模块，不需要真 GPU。"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from eagle_ocr.config import ConfigError, Config
from eagle_ocr.engine import GpuUnavailable, PaddleEngine, check_gpu, is_gpu_oom

from helpers import RunningService, b64, png_bytes

MiB = 1024 * 1024


def fake_paddle(cuda: bool = True, count: int = 1, free_mib: int = 7000) -> types.SimpleNamespace:
    """模拟 paddle 里 check_gpu 用到的那几个 API。"""
    calls: list[str] = []
    return types.SimpleNamespace(
        calls=calls,
        set_device=calls.append,
        device=types.SimpleNamespace(
            is_compiled_with_cuda=lambda: cuda,
            cuda=types.SimpleNamespace(device_count=lambda: count),
        ),
        base=types.SimpleNamespace(core=types.SimpleNamespace(gpu_memory_available=lambda: free_mib * MiB)),
    )


class CheckGpu(unittest.TestCase):
    def test_ok_selects_device(self) -> None:
        p = fake_paddle(count=2)
        check_gpu("gpu:1", 1024, paddle=p)
        self.assertEqual(p.calls, ["gpu:1"])

    def test_cpu_build_of_paddle(self) -> None:
        with self.assertRaisesRegex(GpuUnavailable, "CPU 版"):
            check_gpu("gpu", 1024, paddle=fake_paddle(cuda=False))

    def test_no_visible_gpu(self) -> None:
        with self.assertRaisesRegex(GpuUnavailable, "device_count=0"):
            check_gpu("gpu", 1024, paddle=fake_paddle(count=0))

    def test_gpu_id_out_of_range(self) -> None:
        with self.assertRaisesRegex(GpuUnavailable, "超出范围.*只有 1 块"):
            check_gpu("gpu:3", 1024, paddle=fake_paddle(count=1))

    def test_not_enough_free_memory(self) -> None:
        with self.assertRaisesRegex(GpuUnavailable, "可用 500 MiB，低于 gpu_min_free_mb=1024"):
            check_gpu("gpu", 1024, paddle=fake_paddle(free_mib=500))

    def test_zero_min_free_skips_memory_check(self) -> None:
        p = fake_paddle(free_mib=1)
        check_gpu("gpu", 0, paddle=p)
        self.assertEqual(p.calls, [])

    def test_paddle_not_installed(self) -> None:
        with mock.patch.dict(sys.modules, {"paddle": None}), \
                self.assertRaisesRegex(GpuUnavailable, "没装 paddlepaddle-gpu"):
            check_gpu("gpu", 1024)

    def test_oom_detection(self) -> None:
        self.assertTrue(is_gpu_oom(RuntimeError("ResourceExhaustedError: Out of memory error on GPU 0")))
        self.assertTrue(is_gpu_oom(MemoryError("cudaErrorMemoryAllocation")))
        self.assertFalse(is_gpu_oom(RuntimeError("model file missing")))


class _FakePaddleOCR:
    error: Exception | None = None
    created: list[dict] = []

    def __init__(self, **kwargs) -> None:
        _FakePaddleOCR.created.append(kwargs)
        if _FakePaddleOCR.error is not None:
            raise _FakePaddleOCR.error

    def predict(self, arr):  # noqa: ANN001, ANN201
        return [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]


class PaddleEngineGpu(unittest.TestCase):
    def setUp(self) -> None:
        _FakePaddleOCR.error = None
        _FakePaddleOCR.created = []
        paddleocr = types.SimpleNamespace(PaddleOCR=_FakePaddleOCR)
        patcher = mock.patch.dict(sys.modules, {"paddleocr": paddleocr})
        patcher.start()
        self.addCleanup(patcher.stop)

    def use_paddle(self, p: types.SimpleNamespace) -> None:
        patcher = mock.patch.dict(sys.modules, {"paddle": p})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cpu_paddle_fails_before_building_pipeline(self) -> None:
        """不静默回退：CPU 版 paddle 时根本不构造 PaddleOCR。"""
        self.use_paddle(fake_paddle(cuda=False))
        with self.assertRaises(GpuUnavailable):
            PaddleEngine(device="gpu").load()
        self.assertEqual(_FakePaddleOCR.created, [])

    def test_gpu_ok_passes_device_through(self) -> None:
        self.use_paddle(fake_paddle())
        PaddleEngine(device="gpu:0").load()
        self.assertEqual(_FakePaddleOCR.created[0]["device"], "gpu:0")

    def test_oom_during_load_is_reported_as_gpu_error(self) -> None:
        self.use_paddle(fake_paddle())
        _FakePaddleOCR.error = RuntimeError("ResourceExhaustedError: \n\nOut of memory error on GPU 0.")
        with self.assertRaisesRegex(GpuUnavailable, "gpu 显存不足.*Out of memory"):
            PaddleEngine(device="gpu").load()

    def test_cpu_device_skips_gpu_check(self) -> None:
        self.use_paddle(fake_paddle(cuda=False))
        PaddleEngine(device="cpu").load()
        self.assertEqual(_FakePaddleOCR.created[0]["device"], "cpu")

    def test_health_shows_load_error_and_ocr_503(self) -> None:
        self.use_paddle(fake_paddle(cuda=False))
        svc = RunningService(engine=PaddleEngine(device="gpu"), device="gpu")
        self.addCleanup(svc.close)
        _, health = svc.get("/health")
        self.assertEqual((health["ready"], health["device"]), (False, "gpu"))
        self.assertIn("GpuUnavailable", health["load_error"])
        self.assertIn("CPU 版", health["load_error"])
        status, body = svc.post("/v1/ocr", {"images": [{"base64": b64(png_bytes())}]})
        self.assertEqual((status, body["error"]["code"]), (503, "not_ready"))
        self.assertIn("CPU 版", body["error"]["message"])


class GpuConfig(unittest.TestCase):
    def test_gpu_min_free_mb(self) -> None:
        self.assertEqual(Config().gpu_min_free_mb, 2048)
        self.assertEqual(Config(gpu_min_free_mb=0).gpu_min_free_mb, 0)
        for bad in (-1, 1.5, True, "1024"):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                Config(gpu_min_free_mb=bad)


if __name__ == "__main__":
    unittest.main()
