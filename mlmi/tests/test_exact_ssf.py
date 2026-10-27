"""Exact-weight SSF payloads round-trip through the production file reader."""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tooling"), str(ROOT)]
import sml_format  # noqa: E402
import sml_read  # noqa: E402
from mlmi.backend.quantizer import recipe  # noqa: E402
from mlmi.backend.levels import LEVELS, size  # noqa: E402


class ExactSSFTests(unittest.TestCase):
    def test_exact_build_is_exposed_as_a_supported_level(self):
        self.assertIn("exact", {level["id"] for level in LEVELS})

    def test_as_is_is_a_distinct_no_quantization_build_and_full_size_estimate(self):
        command = recipe("checkpoint", "model.ssf", 2, level="exact")
        self.assertIn("--exact", command)
        self.assertNotIn("--alloc", command)
        self.assertNotIn("--levels", command)
        self.assertEqual(size([("layer.weight", [3, 2])], "exact", 1),
                         sml_format.HEADER_BYTES + 3 * 2 * 4)

    def test_f32_matrix_values_and_shape_survive_ssf_round_trip(self):
        values = np.array([
            [0.0, -0.0],
            [np.finfo(np.float32).tiny, np.finfo(np.float32).max],
            [-1.25, 0.33333334],
        ], dtype=np.float32)
        payload = values.tobytes()
        weight_off = sml_format.HEADER_BYTES
        table_off = weight_off + len(payload)
        meta = {"kind": "f32", "rows": values.shape[0], "cols": values.shape[1],
                "off": weight_off, "nbytes": len(payload)}
        document = {"arch": "llama", "hparams": {"n_layer": 1, "n_embd": 2},
                    "levels": [], "precision": "exact-f32", "tensors": {"test.weight": meta}}
        jblob = json.dumps(document).encode("utf-8")
        json_off = table_off + sml_format.REC_SIZE
        token_blob = b"[]"
        token_off = json_off + len(jblob)
        packed_record = struct.pack(
            sml_format.REC_FMT, b"test.weight", sml_format.KIND_F32,
            values.shape[0], values.shape[1], 0, weight_off, len(payload),
            *([0] * 18), sml_format.LAYOUT_FLAT, 32)
        header = struct.pack(
            sml_format.HEADER_FMT, sml_format.MAGIC, sml_format.VERSION,
            1, 2, 1, 1, 4, 3, 16, 10000.0, 1e-5, sml_format.GROUP, 1,
            4, 0, 0, 0, 1, table_off, json_off, len(jblob), token_off, len(token_blob), 0,
            0, 0, 0.0, 0.0, 0, 0, 0, sml_format.MODEL_EXACT,
            0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0,
            0, 0, 0, 0.0, 0.0, 0.0, 0, 0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact.ssf"
            with path.open("wb") as output:
                output.write(header)
                output.write(b"\0" * (sml_format.HEADER_BYTES - len(header)))
                output.write(payload)
                output.write(packed_record)
                output.write(jblob)
                output.write(token_blob)
            model = sml_read.SmlFile(path)
            try:
                self.assertEqual(model.flags & sml_format.MODEL_EXACT, sml_format.MODEL_EXACT)
                self.assertEqual(model.head["precision"], "exact-f32")
                restored = model.f32("test.weight")
                self.assertEqual(restored.shape, values.shape)
                self.assertEqual(restored.tobytes(), payload)
            finally:
                model.mm._mmap.close()


if __name__ == "__main__":
    unittest.main()
