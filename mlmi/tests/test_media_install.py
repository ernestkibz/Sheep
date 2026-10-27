"""On-demand media model installation regression tests (no network or model weights)."""
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import adapters  # noqa: E402
from mlmi.backend.adapters import install  # noqa: E402
from mlmi.backend.workspace import ToolError  # noqa: E402


class MediaInstallTests(unittest.TestCase):
    def test_image_install_fetches_base_then_lcm_and_publishes_terminal_status(self):
        with tempfile.TemporaryDirectory() as temp:
            media_dir = Path(temp)
            events = []
            installer = install.MediaInstaller(events.append)
            calls = []

            def fetch(name, progress):
                calls.append(name)
                progress(2, 4, "Downloading files", "files")
                if name == "sd15":
                    (media_dir / "sd15").mkdir(parents=True)
                    (media_dir / "sd15" / "model_index.json").touch()
                else:
                    (media_dir / "lcm").mkdir(parents=True)
                    (media_dir / "lcm" / "adapter.safetensors").touch()

            with patch.object(adapters, "MEDIA_DIR", media_dir), \
                    patch("mlmi.backend.adapters.fetch.fetch", side_effect=fetch):
                installer.ensure("image")

            self.assertEqual(calls, ["sd15", "lcm"])
            self.assertTrue(any(event.get("progress") == .5 for event in events))
            self.assertEqual(installer.status()["state"], "complete")
            self.assertEqual(events[-1]["phase"], "Ready")

    def test_video_install_fetches_base_then_motion_and_publishes_terminal_status(self):
        with tempfile.TemporaryDirectory() as temp:
            media_dir = Path(temp)
            events = []
            installer = install.MediaInstaller(events.append)
            calls = []

            def fetch(name, progress):
                calls.append(name)
                progress(3, 10, "Downloading files", "files")
                if name == "sd15":
                    base = media_dir / "sd15"
                    for required in (
                        "model_index.json", "scheduler/scheduler_config.json",
                        "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
                        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
                        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
                        "feature_extractor/preprocessor_config.json",
                    ):
                        item = base / required
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.touch()
                else:
                    motion = media_dir / "animatediff" / "animatediff_lightning_4step_diffusers.safetensors"
                    motion.parent.mkdir()
                    motion.touch()

            with patch.object(adapters, "MEDIA_DIR", media_dir), \
                    patch("mlmi.backend.adapters.fetch.fetch", side_effect=fetch):
                installer.ensure("video")

            self.assertEqual(calls, ["sd15", "animatediff"])
            self.assertTrue(any(event.get("progress") == .3 for event in events))
            self.assertEqual(installer.status()["state"], "complete")
            self.assertEqual(events[-1]["phase"], "Ready")

    def test_photo_guidance_installs_lazily_and_checks_encoder_files(self):
        with tempfile.TemporaryDirectory() as temp:
            media_dir = Path(temp)
            installer = install.MediaInstaller()
            calls = []

            def fetch(name, progress):
                calls.append(name)
                root = media_dir / "ip-adapter" / "models"
                (root / "image_encoder").mkdir(parents=True, exist_ok=True)
                (root / "ip-adapter-plus_sd15.bin").touch()
                (root / "image_encoder" / "config.json").touch()
                (root / "image_encoder" / "model.safetensors").touch()

            with patch.object(adapters, "MEDIA_DIR", media_dir), \
                    patch("mlmi.backend.adapters.fetch.fetch", side_effect=fetch):
                self.assertFalse(adapters.video_reference_ready())
                installer.ensure("video-reference")
                self.assertTrue(adapters.video_reference_ready())

            self.assertEqual(calls, ["ip-adapter"])
            self.assertEqual(installer.status()["state"], "complete")
            self.assertEqual(installer.status()["kind"], "video-reference")

    def test_photo_guidance_download_failure_is_reported_without_fallback(self):
        installer = install.MediaInstaller()
        with patch("mlmi.backend.adapters.install.video_reference_ready", return_value=False), \
                patch("mlmi.backend.adapters.fetch.fetch", side_effect=OSError("network unavailable")):
            with self.assertRaisesRegex(ToolError, "Check your internet connection"):
                installer.ensure("video-reference")
        self.assertEqual(installer.status()["kind"], "video-reference")
        self.assertEqual(installer.status()["state"], "failed")

    def test_concurrent_video_requests_share_the_install(self):
        with tempfile.TemporaryDirectory() as temp:
            media_dir = Path(temp)
            installer = install.MediaInstaller()
            entered, release = threading.Event(), threading.Event()
            calls = []

            def fetch(name, progress):
                calls.append(name)
                if name == "sd15":
                    entered.set()
                    release.wait(3)
                    base = media_dir / "sd15"
                    for required in (
                        "model_index.json", "scheduler/scheduler_config.json",
                        "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
                        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
                        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
                        "feature_extractor/preprocessor_config.json",
                    ):
                        item = base / required
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.touch()
                else:
                    motion = media_dir / "animatediff" / "animatediff_lightning_4step_diffusers.safetensors"
                    motion.parent.mkdir()
                    motion.touch()

            errors = []
            def run():
                try:
                    installer.ensure("video")
                except Exception as error:  # surfaced by the calling request
                    errors.append(error)

            with patch.object(adapters, "MEDIA_DIR", media_dir), \
                    patch("mlmi.backend.adapters.fetch.fetch", side_effect=fetch):
                first = threading.Thread(target=run)
                second = threading.Thread(target=run)
                first.start()
                self.assertTrue(entered.wait(2))
                second.start()
                release.set()
                first.join(3)
                second.join(3)

            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(calls, ["sd15", "animatediff"])

    def test_failure_is_preserved_and_reported_to_waiting_request(self):
        with tempfile.TemporaryDirectory() as temp:
            installer = install.MediaInstaller()
            with patch.object(adapters, "MEDIA_DIR", Path(temp)), \
                    patch("mlmi.backend.adapters.fetch.fetch", side_effect=OSError("network unavailable")):
                with self.assertRaisesRegex(ToolError, "Check your internet connection"):
                    installer.ensure("video")
        self.assertEqual(installer.status()["state"], "failed")
        self.assertEqual(installer.status()["error"],
                         "Download failed. Check your internet connection, then retry the request.")
        self.assertIn("network unavailable", installer.status()["error_detail"])

    def test_byte_progress_is_reported_as_downloaded_bytes_not_global_percent(self):
        events = []
        progress = install.MediaInstaller._progress(lambda **event: events.append(event), "Whisper small")
        progress(1_500_000, 4_000_000, "model.bin", "B")
        self.assertIsNone(events[0]["progress"])
        self.assertIn("1.5 MB of 4.0 MB", events[0]["detail"])

    def test_cancel_signals_an_active_media_download_at_progress_boundary(self):
        installer = install.MediaInstaller()
        installer._active = True
        installer._status = {"id": "job", "kind": "video", "state": "installing"}
        result = installer.cancel()
        self.assertTrue(result["cancelled"])
        with self.assertRaisesRegex(RuntimeError, "cancelled by the user"):
            installer._progress(lambda **event: None, "Video", installer._cancel_event)(1, 10, "motion", "B")

    def test_whisper_small_download_uses_hugging_face_and_marks_ready(self):
        installer = install.MediaInstaller()
        captured = {}
        fake_hub = types.ModuleType("huggingface_hub")
        def download(repo, **options):
            captured["repo"] = repo
            captured.update(options)
        fake_hub.snapshot_download = download
        with patch("mlmi.backend.adapters.install._whisper_ready", side_effect=[False, True]), \
                patch("mlmi.backend.adapters.install.legacy_whisper_snapshot", return_value=None), \
                patch("mlmi.backend.adapters.install.importlib.util.find_spec", return_value=object()), \
                patch.object(install.MediaInstaller, "_tqdm_class", return_value=object()), \
                patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            installer.ensure("audio")
        self.assertEqual(captured["repo"], "Systran/faster-whisper-small")
        self.assertEqual(captured["cache_dir"], str(adapters.whisper_cache()))
        self.assertIn("model.bin", captured["allow_patterns"])
        self.assertEqual(installer.status()["state"], "complete")

    def test_existing_shared_whisper_cache_is_copied_into_sheep_before_use(self):
        installer = install.MediaInstaller()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = root / "shared" / "models--Systran--faster-whisper-small" / "snapshots" / "revision"
            revision.mkdir(parents=True)
            for name in ("model.bin", "config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.txt"):
                (revision / name).write_text("weights" if name == "model.bin" else "{}", encoding="utf-8")
            with patch.object(adapters, "MEDIA_DIR", root / "sheep-media"), \
                    patch("mlmi.backend.adapters.install.importlib.util.find_spec", return_value=object()), \
                    patch.dict(os.environ, {"HF_HUB_CACHE": str(revision.parents[2])}):
                installer.ensure("audio")
                local = adapters.whisper_snapshot()
                self.assertIsNotNone(local)
                self.assertEqual((local / "model.bin").read_text(encoding="utf-8"), "weights")
                self.assertEqual(installer.status()["state"], "complete")

    def test_whisper_is_ready_only_when_sheep_owned_small_weights_exist(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshots = Path(temp) / "hf-cache" / "models--Systran--faster-whisper-small" / "snapshots"
            revision = snapshots / "revision"
            revision.mkdir(parents=True)
            with patch.object(adapters, "MEDIA_DIR", Path(temp)), \
                    patch("mlmi.backend.adapters.importlib.util.find_spec", return_value=object()):
                self.assertFalse(adapters._whisper_ready())
                (revision / "model.bin").touch()
                (revision / "config.json").touch()
                (revision / "preprocessor_config.json").touch()
                (revision / "tokenizer.json").touch()
                (revision / "vocabulary.txt").touch()
                self.assertTrue(adapters._whisper_ready())


if __name__ == "__main__":
    unittest.main()
