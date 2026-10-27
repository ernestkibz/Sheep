"""More picture and speech models (2026-10-27): any Stable Diffusion 1.5 model as the base for pictures
and video, any faster-whisper model for speech; bigger picture families refused with the reason."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import media_styles as MS  # noqa: E402

SD15_FILES = {"model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json"}


def reads(index_cls, dim=768):
    return lambda repo, name: {"_class_name": index_cls} if name == "model_index.json" else {"cross_attention_dim": dim}


class Styles(unittest.TestCase):
    def test_an_sd15_fine_tune_is_accepted(self):
        with patch.object(MS, "_read_json", reads("StableDiffusionPipeline")):
            ok, why, arch = MS.check_style("someone/realistic-sd15", SD15_FILES)
        self.assertTrue(ok)
        self.assertIn("1.5", arch)

    def test_sdxl_is_refused_with_the_reason(self):
        with patch.object(MS, "_read_json", reads("StableDiffusionXLPipeline")):
            ok, why, _ = MS.check_style("stabilityai/sdxl", SD15_FILES)
        self.assertFalse(ok)
        self.assertIn("8 GB", why)

    def test_sd2_shape_is_refused(self):
        with patch.object(MS, "_read_json", reads("StableDiffusionPipeline", dim=1024)):
            ok, why, _ = MS.check_style("sd2/model", SD15_FILES)
        self.assertFalse(ok)
        self.assertIn("SD 2", why)

    def test_no_safetensors_is_refused(self):
        with patch.object(MS, "_read_json", reads("StableDiffusionPipeline")):
            ok, _, _ = MS.check_style("old/model", {"model_index.json", "unet/diffusion_pytorch_model.bin"})
        self.assertFalse(ok)

    def test_the_base_follows_the_choice_only_when_it_is_here(self):
        with tempfile.TemporaryDirectory() as d, patch.object(MS, "STYLES", Path(d)):
            cfg = {"media_models": {"image_style": "someone/style"}}
            self.assertEqual(MS.chosen_base(cfg).name, "sd15")             # not downloaded: Sheep's own
            sd = MS.style_dir("someone/style")
            (sd / "unet").mkdir(parents=True)
            (sd / "model_index.json").write_text("{}")
            (sd / "unet" / "w.safetensors").write_bytes(b"0")
            self.assertEqual(MS.chosen_base(cfg), sd)


class Whisper(unittest.TestCase):
    def test_any_faster_whisper_format_is_accepted(self):
        self.assertTrue(MS.check_whisper({"config.json", "model.bin", "tokenizer.json", "vocabulary.json"})[0])
        self.assertFalse(MS.check_whisper({"config.json", "pytorch_model.bin"})[0])

    def test_the_chosen_snapshot_is_found(self):
        with tempfile.TemporaryDirectory() as d, patch.object(MS, "whisper_cache", lambda: Path(d)):
            snap = Path(d) / "models--Systran--faster-whisper-base" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "model.bin").write_bytes(b"0")
            (snap / "config.json").write_text("{}")
            self.assertEqual(MS.whisper_snapshot_of("Systran/faster-whisper-base"), snap)
            self.assertIsNone(MS.whisper_snapshot_of("Systran/faster-whisper-large-v3"))


class TestRun(unittest.TestCase):
    """A new model is used only after a test run (2026-10-27)."""

    def test_speech_passes_when_the_model_transcribes(self):
        from mlmi.backend.tools import media as tm

        class Ok:
            def transcribe(self, x, **k):
                return iter([]), None
        with patch.object(tm, "_whisper", lambda: Ok()):
            self.assertTrue(MS.test_run("speech_repo", None)[0])

    def test_speech_fails_with_the_reason(self):
        from mlmi.backend.tools import media as tm

        def broken():
            raise RuntimeError("model.bin is truncated")
        with patch.object(tm, "_whisper", broken):
            ok, why = MS.test_run("speech_repo", None)
        self.assertFalse(ok)
        self.assertIn("truncated", why)


if __name__ == "__main__":
    unittest.main()
