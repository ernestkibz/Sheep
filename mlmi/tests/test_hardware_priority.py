import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlmi.backend import hardware
from mlmi.backend.modes import LLMError, Local
from mlmi.backend.orchestrator import Orchestrator
from mlmi.backend.tools import media


GB = 2**30


class HardwarePriorityTests(unittest.TestCase):
    def setUp(self):
        self.old_config = dict(media.CONFIG)
        media.CONFIG.clear()
        media.CONFIG.update(media_device="auto")

    def tearDown(self):
        media.CONFIG.clear()
        media.CONFIG.update(self.old_config)

    def test_directml_adapters_put_nvidia_before_radeon_and_intel(self):
        ads = [
            {"name": "Radeon RX 7600", "vendor": "amd", "vram": 8 * GB, "integrated": False, "ok": True},
            {"name": "Intel Arc A770", "vendor": "intel", "vram": 16 * GB, "integrated": False, "ok": True},
            {"name": "NVIDIA RTX 3060", "vendor": "nvidia", "vram": 12 * GB, "integrated": False, "ok": True},
        ]
        devs = [
            {"index": 0, "name": "Radeon RX 7600", "ok": True, "gbps": 450},
            {"index": 1, "name": "Intel Arc A770", "ok": True, "gbps": 500},
            {"index": 2, "name": "NVIDIA RTX 3060", "ok": True, "gbps": 300},
        ]
        with patch.object(hardware, "adapters", return_value=ads), \
                patch.object(hardware, "directml", return_value={"devices": devs}):
            rows = hardware.directml_adapters()
            no_nvidia = hardware.directml_adapters(include_nvidia=False)
        self.assertEqual([a["vendor"] for a in rows], ["nvidia", "amd", "intel"])
        self.assertEqual([a["vendor"] for a in no_nvidia], ["amd", "intel"])

    def test_settings_modes_put_auto_and_nvidia_ahead_of_fallbacks(self):
        with patch.object(hardware, "adapters", return_value=[]), \
                patch.object(hardware, "_nv", return_value={"name": "NVIDIA RTX"}), \
                patch.object(hardware, "amd_text", return_value={"id": "amd", "label": "AMD", "ok": True}), \
                patch.object(hardware, "multi_text", return_value={"id": "multi", "label": "Multi", "ok": True}), \
                patch.object(hardware, "directml", return_value={"state": "ok", "devices": []}), \
                patch.object(hardware, "torch_cuda", return_value=True), \
                patch.object(hardware, "avx2", return_value=True), \
                patch.object(hardware, "PTX", SimpleNamespace(exists=lambda: True)):
            result = hardware.modes({"gpu": {"name": "NVIDIA RTX", "vram_total": 8 * GB}})
        self.assertEqual([x["id"] for x in result["text"]], ["auto", "gpu", "hybrid", "multi", "amd", "cpu"])
        self.assertEqual([x["id"] for x in result["media"]], ["auto", "cuda", "directml", "cpu"])
        self.assertEqual(result["text"][1]["label"], "NVIDIA RTX")
        self.assertEqual(result["media"][1]["label"], "NVIDIA RTX")

    def test_media_auto_tries_cuda_before_directml_then_cpu(self):
        ads = [{"name": "NVIDIA RTX", "vendor": "nvidia", "vram": 8 * GB, "ok": True}]
        dml = [
            {"name": "Radeon RX", "vendor": "amd", "index": 1, "vram": 8 * GB},
            {"name": "Intel Arc", "vendor": "intel", "index": 2, "vram": 8 * GB},
        ]
        with patch.object(hardware, "adapters", return_value=ads), \
                patch.object(hardware, "directml_adapters", return_value=dml), \
                patch.object(hardware, "DML_PY", Path(sys.executable)), \
                patch.object(hardware, "torch_cuda", return_value=True):
            candidates = media._device_candidates()
        self.assertEqual([c[1] for c in candidates], ["cuda", "dml:1", "dml:2", "cpu"])

    def test_video_skips_devices_that_cannot_fit_and_manual_choice_stays_pinned(self):
        ads = [{"name": "NVIDIA RTX", "vendor": "nvidia", "vram": 2 * GB, "ok": True}]
        dml = [
            {"name": "NVIDIA RTX", "vendor": "nvidia", "index": 0, "vram": 2 * GB},
            {"name": "Radeon RX", "vendor": "amd", "index": 1, "vram": 8 * GB},
        ]
        with patch.object(hardware, "adapters", return_value=ads), \
                patch.object(hardware, "directml_adapters", return_value=dml), \
                patch.object(hardware, "DML_PY", Path(sys.executable)), \
                patch.object(hardware, "torch_cuda", return_value=True):
            self.assertEqual([c[1] for c in media._device_candidates(video=True)], ["dml:1", "cpu"])
            media.CONFIG["media_device"] = "directml"
            self.assertEqual([c[1] for c in media._device_candidates()], ["dml:1"])
            media.CONFIG["media_device"] = "cpu"
            self.assertEqual([c[1] for c in media._device_candidates()], ["cpu"])

    def test_worker_retries_a_cuda_inference_error_on_directml(self):
        class Stream:
            closed = False

            def write(self, value):
                pass

            def flush(self):
                pass

            def close(self):
                self.closed = True

        class Process:
            def __init__(self):
                self.stdin, self.stdout, self.stderr = Stream(), Stream(), None
                self.code = None

            def poll(self):
                return self.code

            def kill(self):
                self.code = 0

            def wait(self, timeout=None):
                self.code = 0
                return 0

        devices = lambda: [(sys.executable, "cuda", "the NVIDIA GPU"),
                           (sys.executable, "dml:1", "the Radeon (DirectML)"),
                           (sys.executable, "cpu", "the CPU")]
        worker = media._Worker("test image", lambda _: [], devices=devices)
        try:
            with patch.object(media.subprocess, "Popen", side_effect=[Process(), Process()]) as popen, \
                    patch.object(worker, "_line", side_effect=[{"ok": True}, {"ok": True}]), \
                    patch.object(worker, "_send", side_effect=[{"ok": False, "error": "CUDA out of memory"},
                                                                {"ok": True, "out": "result.png"}]), \
                    patch.object(media, "_add_media_fallback") as note:
                result = worker.make({"prompt": "test", "out": "result.png"})
            self.assertTrue(result["ok"])
            self.assertEqual(worker.dev, "dml:1")
            self.assertEqual(popen.call_count, 2)
            note.assert_called_once()
        finally:
            worker.stop()

    def test_worker_keeps_a_manual_directml_choice_pinned_after_failure(self):
        class Stream:
            closed = False
            def write(self, value): pass
            def flush(self): pass
            def close(self): self.closed = True

        class Process:
            def __init__(self):
                self.stdin, self.stdout, self.stderr = Stream(), Stream(), None
                self.code = None
            def poll(self): return self.code
            def kill(self): self.code = 0
            def wait(self, timeout=None): self.code = 0; return 0

        media.CONFIG["media_device"] = "directml"
        devices = lambda: [(sys.executable, "dml:0", "the selected Radeon (DirectML)"),
                           (sys.executable, "dml:1", "another DirectML adapter"),
                           (sys.executable, "cpu", "the CPU")]
        worker = media._Worker("test image", lambda _: [], devices=devices)
        try:
            with patch.object(media.subprocess, "Popen", return_value=Process()) as popen, \
                    patch.object(worker, "_line", return_value={"ok": True}), \
                    patch.object(worker, "_send", return_value={"ok": False, "error": "DirectML out of memory"}), \
                    patch.object(media, "_add_media_fallback") as note:
                result = worker.make({"prompt": "test", "out": "result.png"})
            self.assertFalse(result["ok"])
            self.assertEqual(worker.dev, "dml:0")
            popen.assert_called_once()
            note.assert_not_called()
        finally:
            worker.stop()

    def test_local_chat_replays_failed_inference_and_clears_partial_draft(self):
        local = Local("http://127.0.0.1:1")
        local.retry_device = Mock(return_value="NVIDIA failed; retrying on CPU")
        deltas, retries = [], []

        def stream(body, on_delta):
            if not hasattr(stream, "failed"):
                stream.failed = True
                on_delta("partial")
                raise LLMError("CUDA out of memory")
            on_delta("complete")
            return {"choices": [{"message": {"role": "assistant", "content": "complete"},
                                  "finish_reason": "stop"}]}

        with patch.object(local, "_stream", side_effect=stream):
            msg, info = local.chat([{"role": "user", "content": "hello"}], render=False,
                                   on_delta=deltas.append, on_retry=retries.append)
        self.assertEqual(msg["content"], "complete")
        self.assertEqual(deltas, ["partial", None, "complete"])
        self.assertEqual(retries, ["NVIDIA failed; retrying on CPU"])
        self.assertEqual(info["device_fallback"], ["NVIDIA failed; retrying on CPU"])

    def test_orchestrator_retries_auto_failures_and_leaves_manual_choice_pinned(self):
        engine = SimpleNamespace(alive=lambda: True)
        st = SimpleNamespace(cfg={"device": "auto"}, model_path="model.ssf", engine=engine)
        orch = object.__new__(Orchestrator)
        orch.st, orch.on, orch.lock = st, True, threading.Lock()
        orch.plan = {"device": "gpu", "why": []}
        orch.say, orch.update_entry = Mock(), Mock()

        def start(path, check=False, pref_override=None):
            self.assertEqual(path, "model.ssf")
            self.assertEqual(pref_override, "amd")
            orch.plan = {"device": "amd", "why": []}
            return True

        orch._start = Mock(side_effect=start)
        result = orch.retry_after_device_error("CUDA device error")
        self.assertIn("retried on AMD", result)
        orch._start.assert_called_once()

        st.cfg["device"] = "gpu"
        self.assertIsNone(orch.retry_after_device_error("CUDA device error"))
        orch._start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
