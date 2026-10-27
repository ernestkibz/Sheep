"""Original weights from any Hugging Face page (2026-10-19): a GGUF, GPTQ, AWQ, MLX or
bitsandbytes copy leads to the original it was made from; one file rule for the download,
the size estimate and the header check."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mlmi.backend import hf_source as H  # noqa: E402


def repo(files, base=None, cfg=None):
    return {"files": files, "base": base, "cfg": cfg or {}}


HUB = {
    "bart/Q-7B-Instruct-GGUF": repo(["README.md", "Q-7B-Instruct-Q4_K_M.gguf", "Q-7B-Instruct-Q8_0.gguf"], "Q/Q-7B-Instruct"),
    "Q/Q-7B-Instruct": repo(["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                             "tokenizer.json"]),
    "Q/Q-3B": repo(["config.json", "model.safetensors"]),
    "Q/Q-3B-Instruct": repo(["config.json", "model.safetensors"]),
    "mlx/Q-3B-Instruct-4bit": repo(["config.json", "model.safetensors"], "Q/Q-3B",
                                   {"quantization": {"group_size": 64, "bits": 4}}),
    "tb/M-7B-GPTQ": repo(["config.json", "model.safetensors"], "m/M-7B", {"quantization_config": {"quant_method": "gptq"}}),
    "m/M-7B": repo(["config.json", "model-00001-of-00003.safetensors", "consolidated.safetensors", "params.json"]),
    "x/fp8": repo(["config.json", "model.safetensors"], None, {"quantization_config": {"quant_method": "fp8"}}),
    "x/orphan-GGUF": repo(["a.gguf"]),
    "aq/M-7B-Instruct-AWQ-INT4": repo(["config.json", "model.safetensors"], None,
                                      {"quantization_config": {"quant_method": "awq"}}),
    "m/M-7B-Instruct": repo(["config.json", "model.safetensors"]),
    "tb/L-7B-Chat-GPTQ": repo(["config.json", "model.safetensors"], "ml/L-7b-chat-hf",
                              {"quantization_config": {"quant_method": "gptq"}}),
    "ml/L-7B-Chat": repo(["consolidated.00.pth", "params.json"]),
    "ml/L-7b-chat-hf": repo(["config.json", "model.safetensors"]),
    "x/lora": repo(["adapter_config.json", "adapter_model.safetensors"], "Q/Q-3B"),
    # EXL2 keeps its weights in its own files, no config.json: the name says what it is (2026-10-21)
    "v/Q-3B-Instruct-4.5bpw-exl2": repo(["README.md", "output.safetensors", "measurement.json"]),
    "v/orphan-exl2": repo(["README.md", "output.safetensors"]),
    "x/nothing": repo(["README.md"]),
}


def info(r):
    if r not in HUB:
        raise FileNotFoundError("404 not found")
    h = HUB[r]
    return NS(siblings=[NS(rfilename=f) for f in h["files"]], card_data={"base_model": h["base"]} if h["base"] else {},
              sha="abc")


def cfg(r, _info, _names):
    return HUB[r]["cfg"]


class Resolve(unittest.TestCase):
    def go(self, r):
        return H.resolve(r, model_info=info, read_config=cfg,
                         search=lambda n: [k for k in HUB if k.split("/")[-1].lower() == n.lower()])

    def test_gguf_page_leads_to_its_original(self):
        src, note, kind = self.go("bart/Q-7B-Instruct-GGUF")
        self.assertEqual((src, kind), ("Q/Q-7B-Instruct", "gguf"))
        self.assertIn("a GGUF of Q/Q-7B-Instruct", note)

    def test_packed_copies_lead_to_originals(self):
        self.assertEqual(self.go("tb/M-7B-GPTQ")[:1], ("m/M-7B",))
        # the card names the base model, but the copy's own name exists at that organisation
        self.assertEqual(self.go("mlx/Q-3B-Instruct-4bit")[0], "Q/Q-3B-Instruct")

    def test_name_guess_only_when_it_is_an_original(self):
        # Meta's own .pth repo under the guessed name is not taken over the card's -hf base
        self.assertEqual(self.go("tb/L-7B-Chat-GPTQ")[0], "ml/L-7b-chat-hf")

    def test_no_base_named_found_by_name(self):
        self.assertEqual(self.go("aq/M-7B-Instruct-AWQ-INT4")[0], "m/M-7B-Instruct")

    def test_originals_and_fp8_stay(self):
        self.assertEqual(self.go("Q/Q-7B-Instruct"), ("Q/Q-7B-Instruct", "", "original"))
        self.assertEqual(self.go("x/fp8")[2], "original")

    def test_exl2_page_leads_to_its_original_by_name(self):
        src, note, kind = self.go("v/Q-3B-Instruct-4.5bpw-exl2")
        self.assertEqual((src, kind), ("Q/Q-3B-Instruct", "exl2"))
        self.assertIn("an EXL2 copy", note)
        with self.assertRaisesRegex(LookupError, "an EXL2 copy, which Sheep cannot read, and its page does not name"):
            self.go("v/orphan-exl2")
        with self.assertRaisesRegex(LookupError, "no model weights Sheep can read"):
            self.go("x/nothing")

    def test_refusals_say_why(self):
        with self.assertRaisesRegex(LookupError, "does not name the original"):
            self.go("x/orphan-GGUF")
        with self.assertRaisesRegex(LookupError, "LoRA adapter for Q/Q-3B"):
            self.go("x/lora")


class Files(unittest.TestCase):
    def test_one_format_root_only(self):
        self.assertEqual(H.weight_files(["model-00001-of-00003.safetensors", "consolidated.safetensors",
                                         "original/model.safetensors"]), ["model-00001-of-00003.safetensors"])
        self.assertEqual(H.weight_files(["pytorch_model.bin", "training_args.bin", "optimizer.bin"]),
                         ["pytorch_model.bin"])
        self.assertEqual(H.weight_files(["consolidated.safetensors", "params.json"]), ["consolidated.safetensors"])

    def test_classify(self):
        self.assertEqual(H.classify(["config.json", "model.safetensors"],
                                    {"quantization_config": {"quant_method": "awq"}}), "awq")
        self.assertEqual(H.classify(["x.gguf"], {}), "gguf")
        self.assertEqual(H.classify(["model.safetensors"], {}), "no-config")

    def test_suffixes(self):
        self.assertEqual(H.name_without_suffix("Llama-3.2-1B-Instruct-bnb-4bit"), "Llama-3.2-1B-Instruct")
        self.assertEqual(H.name_without_suffix("gemma-2-2b-it-Q4_K_M-GGUF"), "gemma-2-2b-it")
        self.assertEqual(H.name_without_suffix("Qwen2.5-3B-Instruct"), "")


if __name__ == "__main__":
    unittest.main()
