"""纯函数单测：配置、阅读顺序、base64、host 白名单、CLI。"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from eagle_ocr import __version__
from eagle_ocr.cli import main
from eagle_ocr.config import Config, ConfigError, load_config
from eagle_ocr.engine import Line, join_text, sort_reading_order
from eagle_ocr.fetch import ItemError, decode_base64, host_allowed


def box(x: int, y: int, w: int, h: int) -> list[list[int]]:
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class ReadingOrder(unittest.TestCase):
    def test_rows_top_to_bottom_left_to_right(self) -> None:
        lines = [
            Line("C", 1, box(200, 102, 50, 20)),  # 第二行右，略低一点
            Line("A", 1, box(10, 10, 50, 20)),
            Line("D", 1, box(10, 200, 50, 20)),
            Line("B", 1, box(10, 100, 50, 20)),
        ]
        self.assertEqual(join_text(sort_reading_order(lines)), "A\nB C\nD")

    def test_tall_and_short_boxes_not_merged_when_little_overlap(self) -> None:
        lines = [Line("x", 1, box(0, 0, 10, 20)), Line("y", 1, box(50, 15, 10, 20))]
        self.assertEqual(join_text(sort_reading_order(lines)), "x\ny")

    def test_empty(self) -> None:
        self.assertEqual(join_text(sort_reading_order([])), "")


class Base64(unittest.TestCase):
    def test_ok_and_whitespace(self) -> None:
        self.assertEqual(decode_base64("aGVs\nbG8=", 100), b"hello")

    def test_errors(self) -> None:
        with self.assertRaises(ItemError) as cm:
            decode_base64("@@@", 100)
        self.assertEqual(cm.exception.code, "bad_image")
        with self.assertRaises(ItemError) as cm:
            decode_base64("aGVsbG8=", 4)
        self.assertEqual(cm.exception.code, "too_large")


class Hosts(unittest.TestCase):
    def test_patterns(self) -> None:
        allowed = ("a.com", "*.xhscdn.com")
        self.assertTrue(host_allowed("anything", ()))
        self.assertTrue(host_allowed("A.com", allowed))
        self.assertTrue(host_allowed("sns-webpic-qc.xhscdn.com", allowed))
        self.assertFalse(host_allowed("xhscdn.com", allowed))
        self.assertFalse(host_allowed("evilxhscdn.com", allowed))
        self.assertFalse(host_allowed("b.com", allowed))


class ConfigLoading(unittest.TestCase):
    def write(self, text: str) -> str:
        d = tempfile.mkdtemp()
        p = Path(d) / "c.toml"
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_defaults(self) -> None:
        c = load_config(None)
        self.assertEqual((c.host, c.port, c.device, c.lang), ("127.0.0.1", 8866, "cpu", "ch"))
        self.assertEqual((c.max_images_per_request, c.max_image_bytes), (20, 20 * 1024 * 1024))

    def test_file_and_override(self) -> None:
        c = load_config(self.write('port = 9000\ndevice = "gpu"\nallowed_hosts = ["X.com"]\n'))
        self.assertEqual((c.port, c.device, c.allowed_hosts), (9000, "gpu", ("x.com",)))
        self.assertEqual(c.replace(port=1234, host=None).port, 1234)

    def test_errors(self) -> None:
        for text in ('prot = 1', 'device = "tpu"', 'max_image_bytes = 0', 'port = "x"', "=bad",
                     'enable_mkldnn = "yes"'):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                load_config(self.write(text))
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/eagle.toml")
        with self.assertRaises(ConfigError):
            Config(download_timeout_seconds=True)


class Cli(unittest.TestCase):
    def test_version(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["version"]), 0)
        self.assertEqual(out.getvalue().strip(), __version__)

    def test_bad_config_exit_code(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["serve", "--config", "/nonexistent.toml"]), 2)


if __name__ == "__main__":
    unittest.main()
