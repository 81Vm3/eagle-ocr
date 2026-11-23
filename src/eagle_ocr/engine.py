"""OCR 引擎接口、PaddleOCR 实现和阅读顺序排序。

服务层只依赖 `OcrEngine` 协议，测试用假引擎注入；PaddleOCR 在 `load()` 里才 import，
所以不装 PaddleOCR 也能跑测试。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

log = logging.getLogger(__name__)

Box = list[list[int]]  # 4 个 [x, y]，顺时针，左上角起


@dataclass(frozen=True)
class Line:
    text: str
    score: float
    box: Box

    def to_json(self) -> dict:
        return {"text": self.text, "score": round(float(self.score), 4), "box": self.box}


@runtime_checkable
class OcrEngine(Protocol):
    name: str

    def load(self) -> None:
        """加载模型（耗时，在后台线程里调用一次）。失败直接抛异常。"""

    def recognize(self, image: Image.Image) -> list[Line]:
        """识别一张已解码的 RGB 图片。调用方保证串行调用。"""


class PaddleEngine:
    """PaddleOCR 3.x 通用 OCR 流水线（PP-OCR 检测 + 识别）。"""

    name = "paddleocr"

    def __init__(self, lang: str = "ch", device: str = "cpu", enable_mkldnn: bool = False,
                 gpu_min_free_mb: int = 2048) -> None:
        self.lang = lang
        self.device = device
        self.gpu_min_free_mb = gpu_min_free_mb
        # paddlepaddle 3.3 CPU + oneDNN(MKLDNN) + PIR 执行器推理 PP-OCRv6 会报
        # NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
        # [pir::ArrayAttribute<pir::DoubleAttribute>]，所以默认关掉。
        self.enable_mkldnn = enable_mkldnn
        self._ocr = None

    def load(self) -> None:
        is_gpu = self.device.startswith("gpu")
        if is_gpu:
            # PaddleOCR 在 CPU 版 paddle 上收到 device="gpu" 不报错、静默按 CPU 跑，
            # 所以先自己检查，不满足就让加载失败（/health 里看到 load_error），绝不回退 CPU。
            check_gpu(self.device, self.gpu_min_free_mb)
        from paddleocr import PaddleOCR  # 延迟导入：重依赖、慢

        try:
            # 关闭文档方向分类 / 去畸变 / 文本行方向分类：笔记配图基本是正向的，省时间。
            self._ocr = PaddleOCR(
                lang=self.lang,
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=self.enable_mkldnn,
            )
            # 预热一次，把首个真实请求的初始化开销挪到加载阶段。
            self.recognize(Image.new("RGB", (64, 32), "white"))
        except Exception as exc:
            if is_gpu and is_gpu_oom(exc):
                raise GpuUnavailable(f"{self.device} 显存不足，模型加载 / 预热失败：{_brief(exc)}") from exc
            raise

    def recognize(self, image: Image.Image) -> list[Line]:
        import numpy as np

        if self._ocr is None:
            raise RuntimeError("engine not loaded")
        # PaddleOCR 的 ndarray 输入约定为 BGR（OpenCV 习惯）。
        arr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        lines: list[Line] = []
        for res in self._ocr.predict(arr):
            texts = res["rec_texts"]
            scores = res["rec_scores"]
            polys = res["rec_polys"]
            for text, score, poly in zip(texts, scores, polys):
                text = str(text).strip()
                if not text:
                    continue
                box = [[int(round(float(x))), int(round(float(y)))] for x, y in poly]
                lines.append(Line(text=text, score=float(score), box=box))
        return lines


class GpuUnavailable(RuntimeError):
    """配置了 device=gpu 但 GPU 不可用。"""


_OOM_RE = re.compile(r"out of memory|ResourceExhausted|CUDNN_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation",
                     re.IGNORECASE)


def is_gpu_oom(exc: BaseException) -> bool:
    return bool(_OOM_RE.search(f"{type(exc).__name__}: {exc}"))


def _brief(exc: BaseException, limit: int = 300) -> str:
    """异常消息压成一行并截断（paddle 的报错常常是几十行的 C++ 栈）。"""
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= limit else text[:limit] + "…"


def check_gpu(device: str, min_free_mb: int, paddle=None) -> None:  # noqa: ANN001
    """device=gpu / gpu:N 的加载前检查；不满足时抛 GpuUnavailable（消息直接进 /health 的 load_error）。

    paddle 可注入，测试用假模块；默认导入真的 paddle。
    """
    gpu_id = int(device.partition(":")[2] or 0)
    if paddle is None:
        try:
            import paddle  # noqa: PLC0415
        except ImportError as exc:
            raise GpuUnavailable(f"device={device}，但没装 paddlepaddle-gpu（import paddle 失败：{exc}）") from exc
    if not paddle.device.is_compiled_with_cuda():
        raise GpuUnavailable(
            f"device={device}，但当前 paddlepaddle 是 CPU 版（未编译 CUDA）；"
            "请改装 paddlepaddle-gpu（见 README「GPU」），或把 device 改成 cpu")
    count = paddle.device.cuda.device_count()
    if count < 1:
        raise GpuUnavailable(
            f"device={device}，但没检测到可用 GPU（device_count=0）：检查 NVIDIA 驱动 / CUDA_VISIBLE_DEVICES，"
            "WSL 下确认 /usr/lib/wsl/lib/libcuda.so.1 存在")
    if gpu_id >= count:
        raise GpuUnavailable(f"device={device} 超出范围：只有 {count} 块 GPU（gpu:0..gpu:{count - 1}）")
    if min_free_mb > 0:
        paddle.set_device(f"gpu:{gpu_id}")
        free_mb = paddle.base.core.gpu_memory_available() // (1024 * 1024)
        if free_mb < min_free_mb:
            raise GpuUnavailable(
                f"gpu:{gpu_id} 显存不足：可用 {free_mb} MiB，低于 gpu_min_free_mb={min_free_mb}")
        log.info("gpu:%d ok (%d GPU(s), %d MiB free)", gpu_id, count, free_mb)


def sort_reading_order(lines: list[Line]) -> list[list[Line]]:
    """按阅读顺序分行：先上后下；纵向重叠超过较矮者一半的框视为同一行，行内左到右。"""

    def bounds(line: Line) -> tuple[float, float, float]:
        xs = [p[0] for p in line.box]
        ys = [p[1] for p in line.box]
        return min(ys), max(ys), min(xs)

    rows: list[tuple[float, float, list[Line]]] = []  # (top, bottom, lines)
    for line in sorted(lines, key=lambda ln: (sum(p[1] for p in ln.box) / 4, bounds(ln)[2])):
        top, bottom, _ = bounds(line)
        height = max(bottom - top, 1.0)
        if rows:
            r_top, r_bottom, r_lines = rows[-1]
            overlap = min(bottom, r_bottom) - max(top, r_top)
            if overlap > 0.5 * min(height, max(r_bottom - r_top, 1.0)):
                r_lines.append(line)
                rows[-1] = (min(top, r_top), max(bottom, r_bottom), r_lines)
                continue
        rows.append((top, bottom, [line]))
    return [sorted(r, key=lambda ln: bounds(ln)[2]) for _, _, r in rows]


def join_text(rows: list[list[Line]]) -> str:
    """行内用空格、行间用 \\n 拼接。"""
    return "\n".join(" ".join(line.text for line in row) for row in rows)
