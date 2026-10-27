"""GGUF in (2026-10-21): with no original to build from, Sheep converts a GGUF; it picks the file
closest to the original weights, and still prefers the original when a page names one."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import hf_source, model_slots  # noqa: E402


class FileChoice(unittest.TestCase):
    def test_eight_bit_first_then_sixteen(self):
        names = ["m-q4_k_m.gguf", "m-q8_0.gguf", "m-f16.gguf", "README.md"]
        self.assertEqual(hf_source.gguf_file(names), "m-q8_0.gguf")
        self.assertEqual(hf_source.gguf_file(["m-f16.gguf", "m-Q4_K_M.gguf"]), "m-f16.gguf")

    def test_split_files_and_folders_skipped(self):
        self.assertIsNone(hf_source.gguf_file(["m-q8_0-00001-of-00002.gguf", "sub/m.gguf", "config.json"]))


class Info:
    def __init__(self, names):
        self.siblings = [type("S", (), {"rfilename": n, "size": 500_000_000})() for n in names]


class CheckAcceptsGgufOnlyPages(unittest.TestCase):
    def test_gguf_only_page_is_compatible_and_says_so(self):
        with patch.object(hf_source, "resolve", side_effect=LookupError("no original")), \
                patch("huggingface_hub.HfApi.model_info", return_value=Info(["tune-q8_0.gguf", "README.md"])):
            r = model_slots.check("chat", "someone/fine-tune-GGUF")
        self.assertTrue(r.get("compatible"))
        self.assertEqual(r.get("gguf"), "tune-q8_0.gguf")
        self.assertIn("quality is that file's", r.get("reason", ""))

    def test_page_with_no_gguf_still_refused(self):
        with patch.object(hf_source, "resolve", side_effect=LookupError("no original")), \
                patch("huggingface_hub.HfApi.model_info", return_value=Info(["model.onnx"])):
            r = model_slots.check("chat", "someone/onnx-only")
        self.assertFalse(r.get("compatible"))


if __name__ == "__main__":
    unittest.main()
