import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlmi.backend import server
from mlmi.backend import adapters
from mlmi.backend.adapters import install
from mlmi.backend.tools import media


class ModelChoiceTests(unittest.TestCase):
    def test_fit_marks_cuda_good_cpu_only_video_slow_and_insufficient_memory_bad(self):
        video = {"id": "animatediff-lightning", "kind": "video"}
        good = {"ram": {"available": 8 * 2**30}, "cpu": {"cores": 8},
                "gpu": {"name": "RTX", "vram_total": 8 * 2**30, "vram_used": 1 * 2**30}}
        self.assertEqual(adapters.fit(video, good, cuda_ready=True)["level"], "green")
        cpu = {"ram": {"available": 20 * 2**30}, "cpu": {"cores": 8}, "gpu": None}
        self.assertEqual(adapters.fit(video, cpu)["level"], "yellow")
        poor = {"ram": {"available": 4 * 2**30}, "cpu": {"cores": 4}, "gpu": None}
        self.assertEqual(adapters.fit(video, poor)["level"], "red")

    def test_disabled_voice_is_not_selected_as_the_automatic_default(self):
        catalogue = [
            {"id": "whisper-small", "kind": "audio", "does": "speech to text", "ready": True},
            {"id": "windows-voices", "kind": "audio", "does": "text to speech", "ready": True},
        ]
        with patch.object(adapters, "catalogue", return_value=catalogue):
            self.assertNotIn("audio-out", adapters.defaults({"disabled_media_models": ["windows-voices"]}))

    def test_photo_guidance_has_its_own_media_role(self):
        self.assertEqual(adapters.role({"id": "ip-adapter-plus", "kind": "video"}), "video-reference")

    def test_check_and_use_downloads_before_saving_the_user_choice(self):
        item = {"id": "animatediff-lightning", "kind": "video", "install": "video",
                "installable": True, "ready": False, "does": "AI video clips"}
        ready = dict(item, ready=True)
        state = SimpleNamespace(cfg={"media_models": {}, "disabled_media_models": [item["id"]]})
        with patch.object(adapters, "catalogue", side_effect=[[item], [ready], [ready], [ready]]), \
                patch.object(install, "ensure") as ensure:
            result = server.media_model_action(state, item["id"], "use")
        ensure.assert_called_once_with("video")
        self.assertEqual(state.cfg["media_models"]["video"], item["id"])
        self.assertEqual(state.cfg["disabled_media_models"], [])
        self.assertEqual(result["defaults"]["video"], item["id"])

    def test_remove_disables_windows_voices_without_removing_windows_files(self):
        item = {"id": "windows-voices", "kind": "audio", "does": "text to speech", "ready": True}
        state = SimpleNamespace(cfg={"media_models": {"audio-out": item["id"]}})
        with patch.object(adapters, "catalogue", return_value=[item]), \
                patch.object(install, "remove") as remove:
            server.media_model_action(state, item["id"], "remove")
        remove.assert_not_called()
        self.assertIn(item["id"], state.cfg["disabled_media_models"])
        self.assertNotIn("audio-out", state.cfg["media_models"])

    def test_any_valid_ssf_outside_the_models_folder_is_discoverable(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as home:
            model = Path(directory) / "external-model.ssf"
            model.write_bytes(b"SSF")
            isolated_home = Path(home)
            user_home = isolated_home / "user"
            with patch.object(server, "HOME", isolated_home), \
                    patch.object(server.Path, "home", return_value=user_home), \
                    patch.object(server, "load_registry", return_value={}):
                rows = server.find_models(str(model))
            self.assertEqual([row["path"] for row in rows], [str(model.resolve())])

    def test_picked_model_must_be_a_readable_ssf_file(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "model.gguf"
            wrong.touch()
            with self.assertRaisesRegex(ValueError, r"\.ssf"):
                server.validate_ssf_path(wrong)

            path = Path(directory) / "model.ssf"
            path.touch()
            fake_reader = SimpleNamespace(head={"arch": "llama"})
            fake_module = types.ModuleType("sml_read")
            fake_module.SmlFile = Mock(return_value=fake_reader)
            with patch.dict(sys.modules, {"sml_read": fake_module}):
                self.assertEqual(server.validate_ssf_path(path), str(path.resolve()))

    def test_remove_deletes_only_feature_files_and_preserves_shared_base_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lcm").mkdir()
            (root / "lcm" / "adapter.safetensors").touch()
            (root / "sd15").mkdir()
            (root / "sd15" / "model_index.json").touch()
            workers = [SimpleNamespace(lock=threading.Lock(), stop=Mock()) for _ in range(3)]
            with patch.object(install, "MEDIA_DIR", root), \
                    patch.object(adapters, "MEDIA_DIR", root), \
                    patch.object(adapters, "video_ready", return_value=True), \
                    patch.object(adapters, "video_reference_ready", return_value=False), \
                    patch.object(media, "PICTURES", workers[0]), \
                    patch.object(media, "VIDEOS", workers[1]), \
                    patch.object(media, "REFERENCE_VIDEOS", workers[2]):
                self.assertTrue(install.MediaInstaller().remove("sd15-lcm"))
            self.assertFalse((root / "lcm").exists())
            self.assertTrue((root / "sd15" / "model_index.json").exists())
            self.assertTrue(all(worker.stop.called for worker in workers))

    def test_remove_waits_for_media_download_to_finish(self):
        installer = install.MediaInstaller()
        installer._active = True
        with self.assertRaisesRegex(RuntimeError, "downloaded"):
            installer.remove("whisper-small")


if __name__ == "__main__":
    unittest.main()
