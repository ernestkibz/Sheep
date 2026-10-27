"""One-line commands Sheep runs itself (direct.py): each simulation request the
3B fumbled on 2026-10-13, and requests that must stay with the model."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import direct  # noqa: E402


class DirectTests(unittest.TestCase):
    def check(self, asked, tool, **args):
        got = direct.parse(asked)
        self.assertIsNotNone(got, asked)
        self.assertEqual(got[0], tool, asked)
        for k, v in args.items():
            self.assertEqual(got[1][k], v, asked)

    def test_the_commands_the_simulation_asked(self):
        self.check("Render intro.sstudio to an MP4.", "studio_render", path="intro.sstudio")
        self.check("Export budget.ssheet to Excel.", "export_file", path="budget.ssheet", format="xlsx")
        self.check("Export save-water.sdeck as PowerPoint.", "export_file", format="pptx")
        self.check("Export trip-plan.sdoc as a Word .docx file.", "export_file", format="docx")
        self.check("Save trip-plan.sdoc as a PDF.", "export_file", format="pdf")
        self.check("Copy packing.txt to backup/packing.txt.", "copy_file", src="packing.txt", dst="backup/packing.txt")
        self.check("Copy packing.txt to backup/", "copy_file", dst="backup/packing.txt")
        self.check("Delete old-notes.txt.", "delete_file", path="old-notes.txt")
        self.check("Rename notes.txt to ideas.txt", "move_file", src="notes.txt", dst="ideas.txt")
        self.check("Give save-water.sdeck a dark theme.", "slides_set_theme", path="save-water.sdeck", theme="dark")
        self.check("Add a calendar event called Dentist on 2026-11-03 at 10:00.", "calendar_add",
                   title="Dentist", start="2026-11-03 10:00")
        self.check("Add an event called Standup on 2026-11-04 at 9am for 15 minutes", "calendar_add",
                   start="2026-11-04 09:00", minutes=15)
        self.check("In trip-plan.sdoc replace the word Nairobi with Mombasa everywhere.", "words_replace",
                   path="trip-plan.sdoc", find="Nairobi", replace="Mombasa")
        self.check("Replace Nairobi with Mombasa in trip-plan.sdoc", "words_replace", find="Nairobi")
        self.check("Add speaker notes to slide 2 of save-water.sdeck saying: pause for questions.", "slides_set",
                   path="save-water.sdeck", slide=2, notes="pause for questions")
        self.check("Add a bar chart of the costs in budget.ssheet.", "sheets_add_chart",
                   path="budget.ssheet", kind="bar", title="Costs")
        self.check("Draft an email to ruth@example.com with the subject Lunch saying shall we meet at noon on Friday.",
                   "mail_draft", to="ruth@example.com", subject="Lunch", body="Shall we meet at noon on Friday.")

    def test_what_stays_with_the_model(self):
        for asked in ("Render intro.sstudio and then export it", "Make a deck about water",
                      "Give save-water.sdeck a sparkly theme", "Export budget.ssheet to my boss",
                      "What is the capital of Kenya?", "Copy the best parts of a.txt into b.txt"):
            self.assertIsNone(direct.parse(asked), asked)


if __name__ == "__main__":
    unittest.main()
