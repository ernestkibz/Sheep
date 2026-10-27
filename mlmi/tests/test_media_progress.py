"""Video worker progress is forwarded without pretending a file is finished."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlmi.backend.adapters import image_worker
from mlmi.backend import agent
from mlmi.backend.tools import media


class WorkerProtocolTests(unittest.TestCase):
    def test_worker_reads_progress_until_final_result(self):
        worker = object.__new__(media._Worker)
        worker.what = "video"
        worker.proc = type("Process", (), {"stdout": io.StringIO(
            '{"type":"progress","stage":"Denoising video frames","step":1,"total":4}\n'
            '{"ok":true,"frames":16}\n')})()
        seen = []
        self.assertEqual(worker._line(2, on_progress=seen.append), {"ok": True, "frames": 16})
        self.assertEqual(seen, [{"type": "progress", "stage": "Denoising video frames", "step": 1, "total": 4}])

    def test_progress_callback_failure_does_not_fail_worker_result(self):
        worker = object.__new__(media._Worker)
        worker.what = "video"
        worker.proc = type("Process", (), {"stdout": io.StringIO(
            '{"type":"progress","step":1,"total":4}\n{"ok":true}\n')})()

        def broken(_event):
            raise RuntimeError("UI disconnected")

        self.assertEqual(worker._line(2, on_progress=broken), {"ok": True})

    def test_animate_diffusion_reports_measured_denoising_steps(self):
        events = []

        class FakeGenerator:
            def manual_seed(self, _seed):
                return self

        class FakeTorch:
            Generator = lambda _device: FakeGenerator()
            cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})

        class Frame:
            def save(self, path):
                Path(path).write_bytes(b"frame")

        class FakePipeline:
            def __call__(self, *, callback_on_step_end, **kwargs):
                for step in range(kwargs["num_inference_steps"]):
                    callback_on_step_end(self, step, None, {})
                return type("Output", (), {"frames": [[Frame()] * kwargs["num_frames"]]})()

        with tempfile.TemporaryDirectory() as directory, patch.object(image_worker.time, "time", return_value=1):
            result = image_worker.video(FakePipeline(), FakeTorch, {
                "prompt": "a kite", "out": directory, "frames": 8, "seed": 1,
            }, progress=events.append)
            saved_frames = len(list(Path(directory).glob("frame_*.png")))

        self.assertEqual(result["frames"], 8)
        self.assertEqual([event["step"] for event in events], [1, 2, 3, 4])
        self.assertTrue(all(event["total"] == 4 for event in events))
        self.assertEqual(saved_frames, 8)

    def test_video_wait_line_is_a_recent_run_comparison_not_a_promise(self):
        with patch.object(media, "_video_device", return_value=("python", "cuda", "NVIDIA GPU")), \
                patch.object(media, "_speeds", return_value={"cuda": 18.75}), \
                patch.object(media.VIDEOS, "dev", "cuda"), patch.object(media.VIDEOS, "proc", object()):
            line = media.wait_words({"seconds": 2})
        self.assertIn("similar clips took about 5 min", line)
        self.assertIn("this run can vary", line)
        self.assertNotIn("5 min left", line)

    def test_video_eta_is_only_returned_from_a_saved_device_measurement(self):
        with patch.object(media, "_video_device", return_value=("python", "cuda", "NVIDIA GPU")), \
                patch.object(media, "_speeds", return_value={"cuda": 18.75}), \
                patch.object(media.VIDEOS, "dev", "cuda"):
            self.assertEqual(media.video_estimate_seconds({"seconds": 2}), 300)
        with patch.object(media, "_video_device", return_value=("python", "cuda", "NVIDIA GPU")), \
                patch.object(media, "_speeds", return_value={}), \
                patch.object(media.VIDEOS, "dev", "cuda"):
            self.assertIsNone(media.video_estimate_seconds({"seconds": 2}))

    def test_long_video_note_carries_measured_estimate_separately(self):
        with patch.object(media, "wait_words", return_value="Making a video"), \
                patch.object(media, "video_estimate_seconds", return_value=300):
            note = agent.long_note("make_video", {"seconds": 2})
        self.assertEqual(note, {"what": "Making a video", "estimate_seconds": 300})

    def test_chat_timer_is_scoped_to_the_active_tool_and_formats_long_runs(self):
        ui = Path(__file__).resolve().parents[1] / "desktop" / "ui" / "app.js"
        source = ui.read_text(encoding="utf-8")
        self.assertIn("bot.activeToolId = ev.id; bot.toolStartedAt = Date.now();", source)
        self.assertIn('data-estimate="${timerEstimate}"', source)
        self.assertIn("taking longer than similar recent clips", source)
        self.assertIn("n.textContent = elapsedText(n.dataset.t0, n.dataset.estimate)", source)


class RequestedNarrationTests(unittest.TestCase):
    def test_explicit_unquoted_narration_is_preserved_before_following_instructions(self):
        prompt = ("Create a 2-second video. Add spoken narration that says: "
                  "A red kite floats above a green meadow. Use the selected Video model and save the MP4.")
        self.assertEqual(agent.extract_requested_narration(prompt),
                         "A red kite floats above a green meadow.")

    def test_quoted_narration_preserves_internal_punctuation(self):
        prompt = 'Add a voice-over saying: “Wait, look! It is moving.” Then render the video.'
        self.assertEqual(agent.extract_requested_narration(prompt), "Wait, look! It is moving.")

    def test_no_exact_speech_request_returns_empty(self):
        self.assertEqual(agent.extract_requested_narration("Make a video with narration."), "")


if __name__ == "__main__":
    unittest.main()
