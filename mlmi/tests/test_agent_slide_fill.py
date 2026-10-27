import unittest
import threading
from unittest.mock import Mock

from mlmi.backend.agent import (Session, fill_slide_shortfall, preserve_explicit_sheet_row,
                                repair_slide_placeholders, wait_for_model_selection)
from mlmi.backend.workspace import ModelSelectionRequired, ToolError


class SlideShortfallTests(unittest.TestCase):
    def test_partial_model_draft_gets_a_repair_round(self):
        job = {"kind": "slides", "action": "extend", "count": 2, "target": "deck.sdeck"}
        parsed = ("", "", [{"title": "Forest", "bullets": ["Trees", "Shade"], "notes": ""}])
        writer = Mock(return_value="## Wildlife\n- Birds\n- Deer\n")
        result = fill_slide_shortfall(job, parsed, ["Introduction"], "Trip plan", "Existing outline", writer)
        writer.assert_called_once()
        self.assertEqual(writer.call_args.args[0]["count"], 1)
        self.assertEqual([slide["title"] for slide in result[2]], ["Forest", "Wildlife"])

    def test_repair_round_drops_a_duplicate_and_uses_a_distinct_slide(self):
        job = {"kind": "slides", "action": "extend", "count": 2, "target": "deck.sdeck"}
        parsed = ("", "", [{"title": "Forest", "bullets": ["Trees", "Shade"], "notes": ""}])
        writer = Mock(return_value="## Forest\n- same topic\n- repeated\n\n## Rivers\n- Water\n- Wildlife\n")
        result = fill_slide_shortfall(job, parsed, [], "", "", writer)
        self.assertEqual([slide["title"] for slide in result[2]], ["Forest", "Rivers"])

    def test_placeholder_draft_gets_one_bounded_rewrite_before_saving(self):
        job = {"kind": "slides", "action": "extend", "count": 1, "target": "deck.sdeck"}
        parsed = ("", "", [{"title": "Forest", "bullets": ["<point>", "Trees hold soil."]}])
        writer = Mock(return_value="## Forest\n- Tree roots hold soil in place.\n- Forests shelter wildlife.\n")
        result = repair_slide_placeholders(job, parsed, [], "Trip plan", "Existing outline", writer)

        writer.assert_called_once()
        self.assertTrue(writer.call_args.kwargs["again"])
        self.assertTrue(writer.call_args.kwargs["vary"])
        self.assertNotIn("<point>", " ".join(result[2][0]["bullets"]))

    def test_placeholder_that_survives_rewrite_is_rejected_before_file_tools(self):
        job = {"kind": "slides", "action": "extend", "count": 1, "target": "deck.sdeck"}
        parsed = ("", "", [{"title": "Forest", "bullets": ["<point>", "Trees hold soil."]}])
        writer = Mock(return_value="## Forest\n- <point>\n")
        with self.assertRaisesRegex(ToolError, "no incomplete slide was saved"):
            repair_slide_placeholders(job, parsed, [], "Trip plan", "Existing outline", writer)

    def test_complete_draft_does_not_trigger_an_extra_round(self):
        job = {"count": 1}
        parsed = ("", "", [{"title": "Forest", "bullets": ["Trees", "Shade"], "notes": ""}])
        writer = Mock()
        self.assertIs(fill_slide_shortfall(job, parsed, [], "", "", writer), parsed)
        writer.assert_not_called()

    def test_feature_action_waits_for_explicit_model_use_and_then_resumes(self):
        session = Session()
        required = ModelSelectionRequired("video", "ByteDance/AnimateDiff-Lightning")
        sent = threading.Event()
        events = []
        result = {}

        def emit(event):
            events.append(event)
            sent.set()

        def wait_for_choice():
            result["allowed"] = wait_for_model_selection(session, "tool-1", required, emit, timeout=3)

        thread = threading.Thread(target=wait_for_choice)
        thread.start()
        self.assertTrue(sent.wait(1))
        self.assertEqual(events[0]["slot"], "video")
        self.assertEqual(events[0]["repo_id"], "ByteDance/AnimateDiff-Lightning")
        self.assertTrue(session.answer("tool-1", True))
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["allowed"])

    def test_closing_feature_model_dialog_does_not_run_the_pending_action(self):
        session = Session()
        required = ModelSelectionRequired("speech_input", "Systran/faster-whisper-small")
        sent = threading.Event()
        result = {}

        def wait_for_choice():
            result["allowed"] = wait_for_model_selection(session, "tool-2", required,
                                                         lambda _: sent.set(), timeout=3)

        thread = threading.Thread(target=wait_for_choice)
        thread.start()
        self.assertTrue(sent.wait(1))
        self.assertTrue(session.answer("tool-2", False))
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertFalse(result["allowed"])


class ExplicitSheetRowTests(unittest.TestCase):
    def test_requested_row_and_amount_survive_an_incomplete_model_draft(self):
        result = preserve_explicit_sheet_row(
            "add a row for Internet costing 60 to book-sh006.ssheet",
            ["Item", "Cost"],
            ([], [["Fuel", 22]]),
        )
        self.assertEqual(result[1], [["Fuel", 22], ["Internet", 60]])

    def test_existing_explicit_row_is_corrected_without_creating_a_duplicate(self):
        result = preserve_explicit_sheet_row(
            "Add a row for Internet costing 60 to book.ssheet",
            ["Item", "Cost", "Notes"],
            ([], [["Internet", 50, "monthly"]]),
        )
        self.assertEqual(result[1], [["Internet", 60, "monthly"]])


if __name__ == "__main__":
    unittest.main()


class SlideTopUpRounds(unittest.TestCase):
    def test_three_asked_three_added_when_a_round_repeats(self):
        job = {"kind": "slides", "action": "extend", "count": 3, "target": "deck.sdeck"}
        parsed = ("", "", [{"title": "Origins", "bullets": ["a", "b"], "notes": ""}])
        rounds = iter(["## Origins\n- a\n- b\n\n## Uses\n- c\n- d\n", "## Origins\n- x\n- y\n",
                       "## Future\n- e\n- f\n"])
        writer = Mock(side_effect=lambda *a, **k: next(rounds))
        result = fill_slide_shortfall(job, parsed, [], "", "", writer)
        self.assertEqual([s["title"] for s in result[2]], ["Origins", "Uses", "Future"])
