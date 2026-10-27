"""Publishing Sheep files (Part C5): only shareable licences, notices kept, a page holding a .ssf
is added ready to run."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import model_slots, publish as P  # noqa: E402


class Licences(unittest.TestCase):
    def test_open_licences_pass_others_refused(self):
        for lic in ("apache-2.0", "mit", "llama3.2", "gemma", "nvidia-open-model-license"):
            self.assertTrue(P.can_share(lic)[0], lic)
        for lic in ("", "other", "cc-by-nc-4.0", "qwen-research"):
            self.assertFalse(P.can_share(lic)[0], lic)

    def test_llama_card_keeps_its_notice(self):
        text = P.card("m.ssf", {"built": {"level": "balanced"}}, "llama3.2", "meta-llama/Llama-3.2-3B-Instruct")
        self.assertIn("Built with Llama", text)
        self.assertIn("license: llama3.2", text)

    def test_licence_read_from_the_original_readme(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "README.md").write_text("---\nlicense: apache-2.0\n---\n# x", encoding="utf-8")
            self.assertEqual(P.licence_of({}, d), "apache-2.0")

    def test_refuses_without_a_shareable_licence(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.ssf"
            f.write_bytes(b"x")
            with patch("huggingface_hub.get_token", return_value="hf_x"), \
                    patch("mlmi.backend.hf_source.signed_in", return_value="ernest"):
                with self.assertRaisesRegex(ValueError, "does not clearly allow"):
                    P.publish(str(f), {"licence": "cc-by-nc-4.0"})


class Info:
    def __init__(self, names):
        self.siblings = [type("S", (), {"rfilename": n, "size": 400_000_000})() for n in names]


class ReadyPages(unittest.TestCase):
    def test_page_with_a_sheep_file_needs_no_build(self):
        with patch("huggingface_hub.HfApi.model_info", return_value=Info(["qwen-sheep.ssf", "README.md"])):
            r = model_slots.check("chat", "ernest/qwen-sheep")
        self.assertTrue(r.get("compatible"))
        self.assertEqual(r.get("ssf"), "qwen-sheep.ssf")


if __name__ == "__main__":
    unittest.main()
