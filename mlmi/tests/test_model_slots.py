"""Typed Hugging Face model-slot preflight tests; no model weights or network."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlmi.backend import adapters, levels, model_slots


class ModelSlotTests(unittest.TestCase):
    def fake_api(self, repo, tag, files):
        class Api:
            def model_info(self, requested, files_metadata=False):
                self.requested = requested
                self.files_metadata = files_metadata
                return SimpleNamespace(sha="1234567890abcdef", pipeline_tag=tag,
                                       card_data=SimpleNamespace(license="apache-2.0"),
                                       siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files])
        module = types.ModuleType("huggingface_hub")
        module.HfApi = Api
        module.snapshot_download = Mock(side_effect=AssertionError("check must not download model weights"))
        module.get_safetensors_metadata = Mock()
        module.hf_hub_download = Mock()
        module.model_info = Mock()
        return module

    def test_supported_text_repo_is_accepted_and_check_does_not_download_weights(self):
        hub = self.fake_api("org/model", "text-generation", [("config.json", 2000), ("model.safetensors", 5_000_000)])
        details = {"kind": "text", "builds": True, "model_type": "llama", "n_weights": 2_000_000,
                   "note": "", "n_layer": 2, "tensors": []}
        with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch("mlmi.backend.levels.inspect", return_value=details):
            result = model_slots.check("chat", "org/model")
        self.assertTrue(result["available"])
        self.assertTrue(result["compatible"])
        self.assertEqual(result["architecture"], "llama")
        self.assertEqual(result["license"], "apache-2.0")
        self.assertEqual(result["download_bytes"], 5_002_000)
        self.assertEqual(result["recommendation"]["quant_level"], "balanced")
        hub.snapshot_download.assert_not_called()

    def test_unsupported_text_architecture_is_rejected(self):
        hub = self.fake_api("org/model", "text-generation", [("config.json", 2000), ("model.safetensors", 100)])
        details = {"kind": "text", "builds": False, "model_type": "unknown_arch", "n_weights": 100,
                   "note": "unsupported architecture", "n_layer": 0, "tensors": []}
        with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch("mlmi.backend.levels.inspect", return_value=details):
            result = model_slots.check("checker", "org/model")
        self.assertFalse(result["compatible"])
        self.assertIn("cannot build this kind of model", result["reason"])

    def test_video_model_cannot_enter_speech_slot(self):
        hub = self.fake_api("ByteDance/AnimateDiff-Lightning", None,
                            [("animatediff_lightning_4step_diffusers.safetensors", 1000)])
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            result = model_slots.check("speech_input", "ByteDance/AnimateDiff-Lightning")
        self.assertTrue(result["available"])
        self.assertFalse(result["compatible"])
        self.assertIn("not a faster-whisper model", result["reason"])   # any faster-whisper model is accepted since 2026-10-27

    def test_supported_image_repo_must_have_expected_pipeline_files_and_task(self):
        hub = self.fake_api("stable-diffusion-v1-5/stable-diffusion-v1-5", "text-generation",
                            [("model_index.json", 100), ("unet/config.json", 100), ("text_encoder/config.json", 100)])
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            result = model_slots.check("image", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.assertFalse(result["compatible"])
        self.assertIn("Stable Diffusion pipeline", result["reason"])

    def test_official_faster_whisper_repo_needs_no_transformers_preprocessor_file(self):
        hub = self.fake_api("Systran/faster-whisper-small", "automatic-speech-recognition", [
            ("config.json", 100), ("model.bin", 2_000), ("tokenizer.json", 300), ("vocabulary.txt", 50),
        ])
        with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch("mlmi.backend.model_slots.importlib.util.find_spec", return_value=object()):
            result = model_slots.check("speech_input", "Systran/faster-whisper-small")
        self.assertTrue(result["available"])
        self.assertTrue(result["compatible"])
        self.assertEqual(result["download_bytes"], 2_450)

    def test_whisper_snapshot_accepts_the_real_ct2_file_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models--Systran--faster-whisper-small" / "snapshots" / "rev"
            root.mkdir(parents=True)
            for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
                (root / name).touch()
            with patch.object(adapters, "whisper_cache", return_value=root.parents[2]):
                self.assertEqual(adapters.whisper_snapshot(), root)

    def test_invalid_and_unavailable_ids_are_reported(self):
        self.assertFalse(model_slots.check("chat", "https://huggingface.co/org/model")["available"])
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: SimpleNamespace(model_info=Mock(side_effect=OSError("404 not found")))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            result = model_slots.check("video", "missing/model")
        self.assertFalse(result["available"])
        self.assertIn("no model called missing/model", result["reason"])

    def test_speech_output_rejects_hf_until_a_local_loader_exists(self):
        result = model_slots.check("speech_output", "org/tts")
        self.assertFalse(result["available"])
        self.assertFalse(result["compatible"])
        self.assertIn("no local Hugging Face", result["reason"])


if __name__ == "__main__":
    unittest.main()
