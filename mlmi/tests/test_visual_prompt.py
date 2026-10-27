"""The scene the video model is given (2026-10-27): written by the chat model as a description of what
is seen; when that fails, the person's words without the asking, the length or the voice-over."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend.agent.media import clean_visual, explicit_media_plan, scene_from_request  # noqa: E402


class Ws:
    def path(self, p, must_exist=False):
        return Path("Z:/no/such") / p


class VisualPrompt(unittest.TestCase):
    def test_the_fallback_drops_the_asking_and_the_meta_words(self):
        self.assertEqual(scene_from_request("make a 2-second text-to-video clip of a red kite drifting across "
                                            "a clear sunset sky, with a short voice-over"),
                         "a red kite drifting across a clear sunset sky")
        self.assertEqual(scene_from_request("Generate a 4 second video clip of a cat playing piano"), "a cat playing piano")
        self.assertEqual(scene_from_request("create a video of waves at night"), "waves at night")

    def test_a_written_scene_is_kept_and_cleaned(self):
        got = clean_visual('"A red kite glides across a clear orange sunset sky, wide shot, soft warm light."', "x")
        self.assertEqual(got, "A red kite glides across a clear orange sunset sky, wide shot, soft warm light")

    def test_a_refusal_or_code_falls_back(self):
        self.assertEqual(clean_visual("I cannot make videos.", "waves at night"), "waves at night")
        self.assertEqual(clean_visual("```\nprompt\n```", "waves at night"), "waves at night")
        self.assertEqual(clean_visual("", "waves at night"), "waves at night")

    def test_the_plan_gives_the_scene_to_the_video_model(self):
        calls = explicit_media_plan(Ws(), "make a 2-second video of waves at night", [], "", "Dark waves under moonlight")
        video = next(a for n, a in calls if n == "make_video")
        self.assertEqual(video["prompt"], "Dark waves under moonlight")


if __name__ == "__main__":
    unittest.main()
