"""The tools no test touched, and the gaps closed on 2026-10-12, graded by the file
each leaves behind (as test_tools.py does): the file is the grade."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from mlmi.backend import agent, tools as T  # noqa: E402
from mlmi.backend.tools import web  # noqa: E402
from mlmi.backend.workspace import ToolError, Workspace  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="mlmi-gaps-"))
        self.addCleanup(shutil.rmtree, self.root, True)        # no mlmi-gaps-* folders left in %TEMP%
        self.ws = Workspace(self.root)

    def run_(self, tool, /, **args):           # positional: herder tools take an argument called name
        return T.run(self.ws, tool, args, by="test")

    def js(self, rel):
        return json.loads((self.root / rel).read_text(encoding="utf-8"))

    def picture(self, rel="media/dot.png"):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (200, 30, 30)).save(p)
        return rel

    def pdf_text(self, rel):
        return " ".join(page.extract_text() or "" for page in PdfReader(str(self.root / rel)).pages)


class FilesAndReading(Base):
    def test_list_and_read(self):
        self.run_("write_text_file", path="notes/a.txt", text="hello sheep")
        self.assertIn("notes/a.txt", self.run_("list_files"))
        self.assertIn("hello sheep", self.run_("read_file", path="notes/a.txt"))
        self.run_("words_create", path="d.sdoc", title="Plan")
        self.assertIn("Plan", self.run_("read_file", path="d.sdoc"))

    def test_folder_move_and_delete_with_undo(self):
        self.run_("write_text_file", path="box/one.txt", text="1")
        self.run_("write_text_file", path="box/in/two.txt", text="2")
        self.assertIn("2 files", self.run_("move_file", src="box", dst="crate"))
        self.assertTrue((self.root / "crate/in/two.txt").is_file())
        self.assertFalse((self.root / "box").exists())
        self.assertIn("2 files", self.run_("delete_file", path="crate"))
        self.assertFalse((self.root / "crate").exists())
        self.ws.undo()                                    # the delete comes back, file by file
        self.assertEqual((self.root / "crate/one.txt").read_text(), "1")
        with self.assertRaises(ToolError):
            self.run_("move_file", src="crate", dst="crate/inner")

    def test_misspelt_argument_is_named_not_silent(self):
        out = self.run_("list_files", directory=".")
        self.assertIn("ignored directory", out)
        self.assertIn("folder", out)


class Words(Base):
    def test_set_move_page_break_image(self):
        self.run_("words_create", path="d.sdoc", title="Title")
        self.run_("words_add", path="d.sdoc", text="first")
        self.run_("words_add", path="d.sdoc", text="second")
        self.run_("words_set", path="d.sdoc", index=1, text="FIRST", kind="heading", level=2)
        b = self.js("d.sdoc")["blocks"]
        self.assertEqual((b[1]["text"], b[1]["type"], b[1]["level"]), ("FIRST", "heading", 2))
        self.run_("words_move", path="d.sdoc", index=2, to=0)
        self.assertEqual(self.js("d.sdoc")["blocks"][0]["text"], "second")
        self.run_("words_add_page_break", path="d.sdoc")
        self.run_("words_add_image", path="d.sdoc", image=self.picture())
        kinds = [x["type"] for x in self.js("d.sdoc")["blocks"]]
        self.assertEqual(kinds[-2:], ["pagebreak", "image"])

    def test_format_bold_off_keeps_italic_and_underline(self):
        self.run_("words_create", path="d.sdoc", title="T")
        self.run_("words_add", path="d.sdoc", text="**bold** *slanted* __under__")
        self.run_("words_format", path="d.sdoc", index=1, bold=False)
        t = self.js("d.sdoc")["blocks"][1]["text"]
        self.assertNotIn("**", t)
        self.assertIn("*slanted*", t)
        self.assertIn("__under__", t)
        self.run_("words_format", path="d.sdoc", index=1, italic=True, align="center")
        b = self.js("d.sdoc")["blocks"][1]
        self.assertEqual(b["align"], "center")
        self.assertIn("*", b["text"])

    def test_markdown_export_with_number_cells_and_pipes(self):
        self.run_("words_create", path="d.sdoc", title="T")
        self.run_("words_add_table", path="d.sdoc", rows=[["Item", "Cost"], ["Rent | flat", 800], ["Food", 250.5]])
        self.run_("export_file", path="d.sdoc", format="md")
        md = (self.root / "d.md").read_text(encoding="utf-8")
        self.assertIn("| Rent \\| flat | 800 |", md)
        self.assertIn("250.5", md)

    def test_pdf_export(self):
        self.run_("words_create", path="d.sdoc", title="Quarterly plan")
        self.run_("words_add", path="d.sdoc", text="Three goals for the quarter.")
        self.run_("export_file", path="d.sdoc", format="pdf")
        self.assertIn("Quarterly plan", self.pdf_text("d.pdf"))


class Sheets(Base):
    def test_table_sort_and_pdf_shows_values_not_formulas(self):
        self.run_("sheets_create", path="b.ssheet", headers=["Item", "Cost"],
                  rows=[["Rent", 800], ["Food", 250], ["Bus", 90]], total=True)
        self.run_("sheets_table", path="b.ssheet", sheet="Sheet1", headers=["Item", "Cost"],
                  rows=[["Rent", 800], ["Food", 250], ["Bus", 90]], total=True)
        self.run_("sheets_sort", path="b.ssheet", range="A1:B4", column="B")
        cells = self.js("b.ssheet")["sheets"][0]["cells"]
        self.assertEqual([cells["A2"], cells["A3"], cells["A4"]], ["Bus", "Food", "Rent"])
        self.run_("export_file", path="b.ssheet", format="pdf")
        text = self.pdf_text("b.pdf")
        self.assertIn("1140", text.replace(",", ""))
        self.assertNotIn("=SUM", text)

    def test_chart_range_takes_labels_and_header_and_drops_empty_and_total(self):
        self.run_("sheets_create", path="b.ssheet", headers=["Item", "Cost"],
                  rows=[["Rent", 800], ["Food", 250], ["Bus", 90]], total=True)
        out = self.run_("sheets_add_chart", path="b.ssheet", range="B2:D5", kind="bar")
        self.assertEqual(self.js("b.ssheet")["sheets"][0]["charts"][0]["range"], "A1:B4")
        self.assertIn("A1:B4", out)

    def test_work_check_flags_a_formula_over_empty_cells(self):
        from mlmi.backend import verify
        self.run_("sheets_create", path="b.ssheet", headers=["Item", "Cost"], rows=[["Rent", 800]])
        self.run_("sheets_set_cell", path="b.ssheet", cell="B3", value="=SUM(C2:C2)")
        short = verify.shortfalls(self.ws, "make a budget", {"b.ssheet"})
        self.assertTrue(any("C2:C2 is empty" in s for s in short), short)

    def test_csv_of_a_chosen_sheet(self):
        self.run_("sheets_create", path="b.ssheet", headers=["A"], rows=[["first"]])
        self.run_("sheets_add_sheet", path="b.ssheet", name="Second")
        self.run_("sheets_write", path="b.ssheet", cell="A1", values=[["on the second tab"]], sheet="Second")
        self.run_("export_file", path="b.ssheet", format="csv", sheet="Second", to="second.csv")
        self.assertIn("on the second tab", (self.root / "second.csv").read_text(encoding="utf-8"))

    def test_blank_write_clears_a_range_in_one_change(self):
        self.run_("sheets_create", path="b.ssheet", headers=["A", "B"], rows=[[1, 2], [3, 4]])
        before = len(self.ws.history())
        self.run_("sheets_write", path="b.ssheet", cell="A2", values=[["", ""], ["", ""]])
        cells = self.js("b.ssheet")["sheets"][0]["cells"]
        self.assertFalse(any(k in cells for k in ("A2", "B2", "A3", "B3")))
        self.assertEqual(len(self.ws.history()), before + 1)


class Slides(Base):
    def test_set_text_box_image_and_pdf(self):
        self.run_("slides_create", path="s.sdeck", title="Save Water")
        self.run_("slides_add", path="s.sdeck", title="Why", bullets=["less waste"])
        self.run_("slides_set", path="s.sdeck", slide=2, title="Why it matters", notes="say it slowly")
        self.run_("slides_add_text", path="s.sdeck", slide=2, text="Every drop", x=10, y=70, width=40, size=20)
        self.run_("slides_add_image", path="s.sdeck", slide=2, image=self.picture())
        s = self.js("s.sdeck")["slides"][1]
        self.assertEqual((s["title"], s["notes"]), ("Why it matters", "say it slowly"))
        self.assertEqual(sorted(i["kind"] for i in s["items"]), ["image", "text"])
        self.run_("export_file", path="s.sdeck", format="pdf")
        self.assertIn("Save Water", self.pdf_text("s.pdf"))


class Studio(Base):
    def test_template_delete_and_put(self):
        self.run_("studio_create", path="v.sstudio")
        self.run_("studio_template", path="v.sstudio", template="title-intro", title="Sheep")
        n = len(self.js("v.sstudio")["layers"])
        self.assertGreater(n, 0)
        self.run_("studio_delete", path="v.sstudio", layer=1)
        self.assertEqual(len(self.js("v.sstudio")["layers"]), n - 1)
        d = self.js("v.sstudio")
        d["layers"] = [{"kind": "text", "text": "Replaced", "start": 0, "dur": 2, "x": 50, "y": 50, "w": 80, "h": 16}]
        self.run_("studio_put", path="v.sstudio", data=d)
        self.assertEqual(self.js("v.sstudio")["layers"][0]["text"], "Replaced")


class Transfer(Base):
    def test_import_refuses_a_wrong_target_extension(self):
        self.run_("words_create", path="d.sdoc", title="T")
        self.run_("export_file", path="d.sdoc")
        with self.assertRaises(ToolError):
            self.run_("import_file", path="d.docx", to="notes.txt")
        self.assertIn("in.sdoc", self.run_("import_file", path="d.docx", to="in"))


class Web(Base):
    def test_search_and_fetch_number_their_sources(self):
        hits = [{"title": "Nairobi", "url": "https://example.org/n", "snippet": "capital of Kenya"}]
        ctx = T.Ctx(self.ws, "test")
        with patch.object(web, "search", return_value=hits), \
                patch.object(web, "fetch", return_value=("Nairobi page", "Nairobi is the capital.")):
            out = T.run(self.ws, "web_search", {"query": "capital of Kenya"}, ctx=ctx)
            self.assertIn("[1] Nairobi (https://example.org/n)", out)
            page = T.run(self.ws, "web_fetch", {"url": "https://example.org/n"}, ctx=ctx)
            self.assertTrue(page.startswith("[1] Nairobi page"))


class Herders(Base):
    def test_show_a_builtin(self):
        name = next(n for n in T.herders.library())
        self.assertIn("def ", self.run_("herder_show", name=name))


class ToolCallRepair(unittest.TestCase):
    KNOWN = {"words_create", "slides_add"}

    def name_of(self, text):
        c = agent._repair_call(text, self.KNOWN)
        return c and (c["function"]["name"], c["function"]["arguments"])

    def test_shapes_small_models_write(self):
        want = ("words_create", {"path": "a.sdoc"})
        self.assertEqual(self.name_of('{"name": "words_create", "parameters": {"path": "a.sdoc"}'), want)
        self.assertEqual(self.name_of('```json\n{"name": "words_create", "arguments": {"path": "a.sdoc"}}\n```'), want)
        self.assertEqual(self.name_of('[{"name": "words_create", "parameters": {"path": "a.sdoc"}}]'), want)
        self.assertEqual(self.name_of('Sure, here it is: {"name": "words_create", "parameters": {"path": "a.sdoc"}}'), want)
        self.assertEqual(self.name_of('<function=words_create>{"path": "a.sdoc"}</function>'), want)
        self.assertEqual(self.name_of('{"name": "words_create", "arguments": "{\\"path\\": \\"a.sdoc\\"}"}'), want)

    def test_prose_and_unknown_names_are_not_run(self):
        self.assertIsNone(agent._repair_call('The JSON {"name": "Ada"} is a person.', self.KNOWN))
        self.assertIsNone(agent._repair_call("no call here", self.KNOWN))

    def test_plan_ignores_brackets_in_prose(self):
        text = 'Plan [draft]: [{"tool": "slides_add", "args": {"title": "x"}}] (done [1])'
        self.assertEqual(agent._plan_calls(text, self.KNOWN), [{"tool": "slides_add", "args": {"title": "x"}}])


class Routing(unittest.TestCase):
    def offered(self, text):
        return agent.route_tools(set(T.TOOLS), "", text)

    def test_cues_that_used_to_collide(self):
        self.assertNotIn("h_agenda_slide", self.offered("Put the agenda for Monday's meeting in my calendar"))
        self.assertIn("h_agenda_slide", self.offered("Add an agenda slide to the deck"))
        self.assertNotIn("make_image", self.offered("Add the picture logo.png to slide 2"))
        self.assertIn("make_image", self.offered("Make a picture of a sheep on a hill"))
        self.assertNotIn("studio_render", self.offered("Write a short summary of the report"))
        self.assertNotIn("export_file", self.offered("In trip-plan.sdoc replace the word Nairobi with Mombasa"))
        self.assertIn("export_file", self.offered("Export trip-plan.sdoc as a Word document"))

    def test_claims_of_work_without_a_tool_are_caught(self):
        import re
        for said in ("Added a bar chart to budget.ssheet.", "Updated budget.ssheet: sorted by Cost, largest first.",
                     "The table has been sorted.", "I've added the slide.",
                     "Given save-water.sdeck a dark theme.", "Your save-water.pptx is ready."):
            self.assertTrue(re.search(agent.CLAIM, said, re.I), said)
        self.assertFalse(re.search(agent.CLAIM, "The capital of Kenya is Nairobi.", re.I))

    def test_copy_with_a_path_but_no_word_file(self):
        self.assertIn("copy_file", self.offered("Copy packing.txt to backup/packing.txt."))

    def test_studio_keeps_the_name_the_person_gave(self):
        from mlmi.backend import shepherd
        p = shepherd.studio_plan("Make a 6-second intro video project called intro.sstudio with the title Sheep.")
        self.assertEqual((p["path"], p["title"], p["duration"]), ("intro.sstudio", "Sheep", 6))

    def test_fit_level_3_keeps_whole_call_result_pairs(self):
        convo = [{"role": "system", "content": "s"}, {"role": "user", "content": "do it"}]
        for i in range(6):
            convo += [{"role": "assistant", "content": None, "tool_calls": [{"id": str(i)}]},
                      {"role": "tool", "content": "x" * 5000, "tool_call_id": str(i)}]
        out = agent.fit(convo, 3)
        self.assertEqual([m["role"] for m in out[:2]], ["system", "user"])
        self.assertEqual(out[2]["role"], "assistant")
        self.assertLessEqual(len(out), 6)
        self.assertTrue(all(len(m["content"]) < 400 for m in out if m["role"] == "tool"))

    def test_tools_that_were_never_offered(self):
        self.assertIn("see_picture", self.offered("What is in this photo?"))
        self.assertIn("studio_voice", self.offered("Add a voice over to the video project"))
        self.assertIn("studio_transcribe", self.offered("Put captions on the clip"))


class Restore(Base):
    def test_restore_a_file_to_before_a_change(self):
        self.run_("write_text_file", path="a.txt", text="one")
        self.run_("write_text_file", path="a.txt", text="two")
        t = self.ws.history("a.txt")[-1]["t"]
        self.ws.restore_before(t, "a.txt")
        self.assertEqual((self.root / "a.txt").read_text(), "one")
        self.ws.undo()
        self.assertEqual((self.root / "a.txt").read_text(), "two")


if __name__ == "__main__":
    unittest.main()
