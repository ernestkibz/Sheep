"""Other formats with no original (Part C6, 2026-10-21): GPTQ and AWQ copies and Meta's raw .pth
are converted when no original exists; EXL2, ONNX and Core ML get a plain refusal that names
the format."""
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mlmi.backend import hf_source as H, model_slots  # noqa: E402


def info(names, sizes=None):
    return NS(siblings=[NS(rfilename=n, size=(sizes or {}).get(n, 100)) for n in names], card_data={}, sha="abc")


GPTQ = {"quantization_config": {"quant_method": "gptq", "bits": 4}}


class Kinds(unittest.TestCase):
    def test_meta_and_coreml(self):
        self.assertEqual(H.classify(["consolidated.00.pth", "params.json", "tokenizer.model"], {}), "meta")
        self.assertEqual(H.classify(["m.mlpackage/Data/weights.bin", "README.md"], {}), "coreml")
        self.assertEqual(H.classify(["output.safetensors", "config.json"], {"quantization_config": {"quant_method": "exl2"}}),
                         "exl2")

    def test_fallback_unpacks_gptq_and_meta(self):
        fb = H.fallback("x/m-GPTQ", model_info=lambda r: info(["config.json", "model.safetensors"], {"model.safetensors": 7}),
                        read_config=lambda *a: GPTQ)
        self.assertEqual(fb, {"how": "unpack", "kind": "gptq", "bytes": 7, "as_is": {"id": "small", "tag": "GPTQ 4-bit"}})
        fb = H.fallback("meta/L", model_info=lambda r: info(["consolidated.00.pth", "consolidated.01.pth", "params.json"]),
                        read_config=lambda *a: {})
        self.assertEqual((fb["how"], fb["kind"], fb["bytes"]), ("unpack", "meta", 200))
        self.assertIn("Meta's raw .pth", H.convert_note(fb, "meta/L"))
        self.assertIsNone(fb["as_is"])                 # Meta's .pth is an original: every level is offered

    def test_quantized_copies_kept_as_is(self):
        # Ernest, 2026-10-29: a quantized copy is packed at its own width, an original gets every level
        self.assertEqual(H.as_is("gguf", "m-Q4_K_M.gguf"), {"id": "balanced", "tag": "GGUF Q4_K_M"})
        self.assertEqual(H.as_is("gguf", "m.Q8_0.gguf")["id"], "q8")
        self.assertEqual(H.as_is("gguf", "m-IQ3_XXS.gguf")["id"], "q3")
        self.assertIsNone(H.as_is("gguf", "m-f16.gguf"))
        self.assertIsNone(H.as_is("gguf", "m-BF16.gguf"))
        self.assertEqual(H.as_is("awq", cfg={"quantization_config": {"w_bit": 4}})["id"], "small")
        self.assertEqual(H.as_is("gptq", cfg={"quantization_config": {"bits": 8}})["id"], "q8")
        self.assertIsNone(H.as_is("original"))

    def test_other_formats_refused_by_name(self):
        for names, cfg, word in ((["model.onnx"], {}, "ONNX"), (["m.mlpackage/Data/w.bin"], {}, "Core ML"),
                                 (["config.json", "output.safetensors"], {"quantization_config": {"quant_method": "exl2"}},
                                  "EXL2")):
            self.assertIsNone(H.fallback("x/y", model_info=lambda r: info(names), read_config=lambda *a: cfg))
            with self.assertRaises(LookupError) as e:
                H.resolve("x/y", model_info=lambda r: info(names), read_config=lambda *a: cfg, search=lambda n: [])
            self.assertIn(word, str(e.exception))
            self.assertIn("cannot read", str(e.exception))
            self.assertIn("original", str(e.exception))


class Check(unittest.TestCase):
    def test_gptq_only_page_is_compatible_and_says_so(self):
        with patch.object(H, "resolve", side_effect=LookupError("no original")), \
                patch.object(H, "fallback", return_value={"how": "unpack", "kind": "awq", "bytes": 5}), \
                patch("huggingface_hub.HfApi.model_info", return_value=info(["config.json", "model.safetensors"])):
            r = model_slots.check("chat", "someone/tune-AWQ")
        self.assertTrue(r.get("compatible"))
        self.assertEqual(r.get("unpack"), "awq")
        self.assertIn("only this AWQ copy", r["reason"])


MATH = r'''
import sys, torch
sys.path.insert(0, sys.argv[1])
import sml_unpack as U
torch.manual_seed(0)
IN, OUT, GS = 64, 32, 16
q = torch.randint(0, 16, (IN, OUT), dtype=torch.int32)
z = torch.randint(1, 16, (IN // GS, OUT), dtype=torch.int32)   # GPTQ v1 stores zero - 1
s = (torch.rand(IN // GS, OUT) + 0.1).half().float()
want = ((q - z[torch.arange(IN) // GS]).float() * s[torch.arange(IN) // GS]).t()
def pack(x, dim, order=None):           # 8 nibbles per int32 along dim
    if dim == 0:
        x = x.view(-1, 8, x.shape[1]); return sum((x[:, i] << (4 * i)) for i in range(8))
    x = x.view(x.shape[0], -1, 8)
    if order: x = x[:, :, list(order)]
    return sum((x[:, :, i] << (4 * i)) for i in range(8))
g = U.dequant_gptq({"qweight": pack(q, 0), "qzeros": pack(z - 1, 1), "scales": s.half()},
                   {"bits": 4, "group_size": GS})
a = U.dequant_awq({"qweight": pack(q, 1, (0, 2, 4, 6, 1, 3, 5, 7)), "qzeros": pack(z, 1, (0, 2, 4, 6, 1, 3, 5, 7)),
                   "scales": s.half()}, {"bits": 4, "group_size": GS})
print(round((g - want).abs().max().item(), 2), round((a - want).abs().max().item(), 2))
'''


class Math(unittest.TestCase):
    def test_gptq_and_awq_unpack_exactly(self):
        from mlmi.backend import gpu_runtime
        kind = next((k for k in ("cpu", "cuda") if gpu_runtime.ready(k)), None)
        if not kind:
            self.skipTest("no PyTorch runtime on this machine")
        r = subprocess.run([str(gpu_runtime.py(kind)), "-I", "-c", MATH, str(ROOT / "tooling")],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.stdout.split(), ["0.0", "0.0"], r.stderr[-500:])


if __name__ == "__main__":
    unittest.main()
