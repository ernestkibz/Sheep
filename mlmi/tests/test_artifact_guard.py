"""Follow-up acknowledgements/questions cannot start another artifact mutation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend.agent import Session, artifact_action_allowed  # noqa: E402


class ArtifactGuardTests(unittest.TestCase):
    def test_stop_prevents_the_next_action_and_reports_the_active_step(self):
        session = Session()
        self.assertTrue(session.begin_action("make_video"))
        self.assertEqual(session.request_stop(), "make_video")
        self.assertFalse(session.begin_action("slides_create"))
        session.end_action("make_video")
        self.assertTrue(session.stop.is_set())

    def test_acknowledgement_does_not_recreate_deck_or_video(self):
        for tool in ("slides_create", "slides_add", "make_video", "studio_render"):
            with self.subTest(tool=tool):
                self.assertFalse(artifact_action_allowed(tool, "Thanks, that looks great."))

    def test_follow_up_question_does_not_create_second_deck(self):
        self.assertFalse(artifact_action_allowed("slides_create", "What was the final slide about?"))
        self.assertFalse(artifact_action_allowed("make_image", "Why did you create another picture?"))
        self.assertFalse(artifact_action_allowed("studio_render", "How did you render the last video?"))

    def test_explicit_create_and_extend_are_allowed(self):
        self.assertTrue(artifact_action_allowed("slides_create", "Create a new deck about cats."))
        self.assertTrue(artifact_action_allowed("slides_add", "Add two more slides to that deck."))
        self.assertTrue(artifact_action_allowed("make_image", "Can you create a new picture of a cat?"))
        self.assertTrue(artifact_action_allowed("slides_create", "Put together a slideshow about job interviews."))
        self.assertTrue(artifact_action_allowed("slides_create", "I need slides on cyber security at home."))

    def test_putting_previous_answer_in_file_allows_only_matching_artifact(self):
        self.assertTrue(artifact_action_allowed("words_create", "Put that in a document."))
        self.assertTrue(artifact_action_allowed("words_create", "Put this into my report."))
        self.assertTrue(artifact_action_allowed("sheets_create", "Put that in a spreadsheet."))
        self.assertFalse(artifact_action_allowed("make_video", "Put that in a document."))
        self.assertFalse(artifact_action_allowed("words_create", "Put that in the video."))

    def test_read_tools_are_not_blocked(self):
        self.assertTrue(artifact_action_allowed("slides_read", "What was the final slide about?"))


if __name__ == "__main__":
    unittest.main()
