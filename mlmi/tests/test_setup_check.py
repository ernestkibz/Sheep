"""What Sheep needs, what is here and what is missing (server.setup_checklist), and
the NVIDIA runtime's guards (gpu_runtime.py) - 2026-10-13."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import adapters, gpu_runtime, hardware, server  # noqa: E402

CAT = [{"id": "sd15-lcm", "ready": True}, {"id": "animatediff-lightning", "ready": False},
       {"id": "whisper-small", "ready": True}, {"id": "windows-voices", "ready": True}]


def state(model="m.ssf"):
    return SimpleNamespace(model_path=model, cfg={}, orch=SimpleNamespace(abilities=lambda: {"reads pictures": False}))


class ChecklistTests(unittest.TestCase):
    def items(self, kind="", nvidia=False, cuda=False, ffmpeg=True):
        runtimes = [{"kind": "cuda", "label": "NVIDIA (CUDA)", "size": "about 2.5 GB", "needed": True,
                     "ready": cuda}] if nvidia else []
        with patch.object(adapters, "catalogue", return_value=CAT), \
                patch.object(gpu_runtime, "nvidia", return_value=nvidia), \
                patch.object(gpu_runtime, "status_all", return_value=runtimes), \
                patch.object(hardware, "media_cuda", return_value=cuda), \
                patch("shutil.which", return_value="ffmpeg" if ffmpeg else None):
            return server.setup_checklist(state(), kind)

    def test_each_missing_row_says_what_it_is_for_and_how_to_fix_it(self):
        got = self.items("video", nvidia=True, cuda=False)
        video = next(i for i in got if i["label"].startswith("Video model"))
        self.assertTrue(video["need"] and video["optional"])
        self.assertEqual(video["action"]["do"], "media")
        self.assertFalse(video["speedup"])
        gpu = next(i for i in got if "graphics card support" in i["label"])
        self.assertTrue(gpu["speedup"])                  # makes it faster; never blocks a request
        self.assertEqual(gpu["action"]["do"], "runtime")
        ready = next(i for i in got if i["ok"])
        self.assertIsNone(ready["action"])

    def test_one_kind_with_the_text_model(self):
        got = self.items("video")
        self.assertEqual({i["kind"] for i in got}, {"text", "video"})
        video = next(i for i in got if "Video model" in i["label"])
        self.assertFalse(video["ok"])
        self.assertIn("Settings", video["fix"])

    def test_nvidia_without_cuda_says_how_to_add_it(self):
        got = self.items("picture", nvidia=True, cuda=False)
        cuda = next(i for i in got if "NVIDIA" in i["label"])
        self.assertFalse(cuda["ok"])
        self.assertIn("2.5 GB", cuda["fix"])
        self.assertTrue(next(i for i in got if "Picture model" in i["label"])["ok"])

    def test_no_nvidia_no_cuda_row_and_everything_listed_once(self):
        got = self.items(nvidia=False)
        self.assertFalse(any("NVIDIA" in i["label"] for i in got))
        self.assertEqual(len({i["label"] for i in got}), len(got))
        self.assertFalse(next(i for i in got if "vision" in i["label"])["ok"])


class RuntimeTests(unittest.TestCase):
    def test_install_refuses_without_nvidia(self):
        with patch.object(gpu_runtime, "nvidia", return_value=False):
            got = gpu_runtime.install()
        self.assertEqual(got["job"]["state"], "failed")
        self.assertIn("NVIDIA", got["job"]["error"])

    def test_status_reports_without_running_anything_heavy(self):
        got = gpu_runtime.status()
        for k in ("nvidia", "installed", "ready", "job", "size"):
            self.assertIn(k, got)


if __name__ == "__main__":
    unittest.main()
