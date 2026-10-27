"""Studio editing basics and the photo ad (2026-10-27): crop inside a source, fill or fit, fade lengths,
split, and an ad from photos with pan and zoom, words, an end card and music."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.agent import media as AM  # noqa: E402
from mlmi.backend.tools import studio as S  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402


class Maths(unittest.TestCase):
    def test_crop_box_is_clamped_and_all_of_it_is_none(self):
        self.assertEqual(S.crop_box({"crop": "10,10,50,50"}), (10.0, 10.0, 50.0, 50.0))
        self.assertIsNone(S.crop_box({"crop": [0, 0, 100, 100]}))
        self.assertEqual(S.crop_box({"crop": [90, 0, 50, 100]}), (90.0, 0.0, 10.0, 100.0))

    def test_sound_fades_follow_the_layer(self):
        self.assertEqual(S._fades({}), (0.3, 0.6))
        self.assertEqual(S._fades({"in": {"move": "fade", "dur": 1.2}, "out": {"move": "fade", "dur": 2}}), (1.2, 2.0))

    def test_ad_pans_never_show_an_edge(self):
        for a, b in S.AD_PAN:
            for k in (a, b):
                self.assertLessEqual(abs(k["x"] - 50), (k["scale"] - 1) * 50 + 1e-9)
                self.assertLessEqual(abs(k["y"] - 50), (k["scale"] - 1) * 50 + 1e-9)

    def test_ad_layout(self):
        layers, total = S.ad_layers(["a.jpg", "b.jpg", "c.jpg"], "Fresh Bread Daily", "Mama Njeri Bakery", music="m.mp3")
        self.assertEqual(total, 3 * 3.5 + 3.0)
        pics = [l for l in layers if l["kind"] == "image"]
        self.assertEqual([p["src"] for p in pics], ["a.jpg", "b.jpg", "c.jpg"])
        self.assertTrue(all(p["fit"] == "cover" and len(p["keys"]) == 2 for p in pics))
        self.assertTrue(any(l["kind"] == "audio" and l["src"] == "m.mp3" for l in layers))

    def test_ad_requests_plan_the_ad(self):
        class Ws:
            def path(self, p, must_exist=False):
                return Path("Z:/no/such") / p
        calls = AM.explicit_media_plan(Ws(), "make a vertical ad from these photos for my bakery", ["a.jpg", "b.jpg"],
                                       ad=("Fresh Bread Daily", "Mama Njeri Bakery"))
        self.assertEqual([n for n, _ in calls], ["studio_ad", "studio_render"])
        self.assertEqual(calls[0][1]["format"], "9:16")
        self.assertEqual(AM.ad_words("Fresh Bread Daily\nMama Njeri Bakery, Nairobi", ""), ("Fresh Bread Daily", "Mama Njeri Bakery, Nairobi"))


class Split(unittest.TestCase):
    def test_split_keeps_the_length_and_plays_on(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(d)
            doc = S.new_project("16:9", 10)
            doc["layers"] = [{"kind": "clip", "src": "c.mp4", "start": 1, "dur": 6, "trim": 2, "speed": 2}]
            (Path(d) / "v.sstudio").write_text(json.dumps(doc), encoding="utf-8")
            T.run(ws, "studio_split", {"path": "v.sstudio", "layer": 1, "at": 3}, by="user", ctx=T.Ctx(ws, "user"))
            a, b = json.loads((Path(d) / "v.sstudio").read_text(encoding="utf-8"))["layers"]
            self.assertEqual((a["start"], a["dur"], b["start"], b["dur"]), (1, 2, 3, 4))
            self.assertEqual(b["trim"], 2 + 2 * 2)             # 2 s of layer at speed 2 = 4 s of source


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class RenderAd(unittest.TestCase):
    def test_four_photos_with_music_render_to_a_video_with_sound(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            for k, c in enumerate(("#c0392b", "#27ae60", "#2980b9", "#f39c12")):
                Image.new("RGB", (800, 600), c).save(Path(d) / f"p{k}.jpg")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=20",
                            str(Path(d) / "music.wav")], check=True, timeout=60)
            ws = Workspace(d)
            ctx = T.Ctx(ws, "user")
            T.run(ws, "studio_ad", {"path": "ad.sstudio", "images": [f"p{k}.jpg" for k in range(4)],
                                    "headline": "Fresh Bread Daily", "tagline": "Mama Njeri Bakery",
                                    "music": "music.wav", "seconds_per_photo": 2}, by="user", ctx=ctx)
            T.run(ws, "studio_render", {"path": "ad.sstudio", "out": "ad.mp4", "preview": True}, by="user", ctx=ctx)
            dur, has_audio = S._probe(Path(d) / "ad.mp4")
            self.assertAlmostEqual(dur, 4 * 2 + 3, delta=0.5)
            self.assertTrue(has_audio)


if __name__ == "__main__":
    unittest.main()
