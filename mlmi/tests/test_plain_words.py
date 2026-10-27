"""Plain words (2026-10-20): a locked file is said as such, job cards never show tool names, and Electron's
missing prompt() is not used."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import agent  # noqa: E402

UI = Path(__file__).resolve().parents[1] / "desktop" / "ui"


class PlainWords(unittest.TestCase):
    def test_locked_file_is_said_plainly(self):
        self.assertEqual(agent.plain_error("PermissionError: [WinError 5] Access is denied: 'a.pptx'"),
                         "the file is open in another program; close it and try again")
        self.assertEqual(agent.plain_error("first line\nsecond"), "first line")

    def test_ui_uses_plain_names_and_no_prompt(self):
        app = (UI / "app.js").read_text(encoding="utf-8")
        work = (UI / "work.js").read_text(encoding="utf-8")
        self.assertIn('bot.workTask = DOING[ev.name] || "Working";', app)
        self.assertNotIn("window.prompt(", app)
        self.assertNotIn(" prompt(", work)
        self.assertIn("modelName: prettyModel", app)


if __name__ == "__main__":
    unittest.main()
