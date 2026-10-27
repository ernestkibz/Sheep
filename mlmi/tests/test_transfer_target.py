import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend.tools.transfer import _export_target  # noqa: E402


class ExportTargetTests(unittest.TestCase):
    def test_none_like_optional_target_uses_the_default_extension(self):
        for value in (None, "None", "null", "undefined", ""):
            with self.subTest(value=value):
                self.assertEqual(_export_target({"path": "deck.sdeck", "to": value}),
                                 ("deck.pptx", "pptx"))

    def test_explicit_export_target_is_preserved(self):
        self.assertEqual(_export_target({"path": "deck.sdeck", "to": "handout.pptx"}),
                         ("handout.pptx", "pptx"))


if __name__ == "__main__":
    unittest.main()
