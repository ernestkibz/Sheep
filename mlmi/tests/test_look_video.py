"""Watching a video (2026-10-27): key frames where the picture changes, kept apart and capped; the
sound heard; a summary saved; attached videos are watched, not only transcribed."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.agent import routing  # noqa: E402
from mlmi.backend.tools import media  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402


class FrameTimes(unittest.TestCase):
    def test_few_cuts_spread_evenly(self):
        self.assertEqual(media.spread_times([], 10), [0.83, 2.5, 4.17, 5.83, 7.5, 9.17])

    def test_many_cuts_are_capped(self):
        got = media.spread_times([float(t) for t in range(1, 60)], 60)
        self.assertLessEqual(len(got), media.LOOK_FRAMES)
        self.assertEqual(got[0], 0.5)

    def test_flicker_does_not_crowd_the_frames(self):
        got = media.spread_times([1.37, 1.5, 1.63, 1.77, 2.0], 3)
        self.assertTrue(all(b - a >= 0.75 for a, b in zip(got, got[1:])))

    def test_video_questions_are_routed_to_it(self):
        self.assertIn("look_video", routing.route_tools(list(T.TOOLS), "", "what happens in this video clip.mp4"))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class Watch(unittest.TestCase):
    def test_a_real_clip_gets_frames_and_a_summary(self):
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "clip.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=10",
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-shortest", "-pix_fmt", "yuv420p",
                            str(clip)], check=True, timeout=120)
            ws = Workspace(d)
            out = T.run(ws, "look_video", {"path": "clip.mp4"}, by="user", ctx=T.Ctx(ws, "user"))
            self.assertIn("clip.summary.txt", out)
            self.assertIn("with sound", (Path(d) / "clip.summary.txt").read_text(encoding="utf-8"))
            self.assertTrue(list((Path(d) / "clip-frames").glob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
