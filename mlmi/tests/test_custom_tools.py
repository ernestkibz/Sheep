"""Custom tools (2026-10-20): add_totals runs from Settings with no inputs, a built-in is offered by its
phrase, and a tool a model wrote is offered only after it ran cleanly once."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import agent, formats as F, tools as T  # noqa: E402
from mlmi.backend.tools import herders as H  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

UPPER = '''def herd(doc, args):
    for b in doc["blocks"]:
        if b.get("type") == "heading":
            b["text"] = b["text"].upper()
    return "shouted"
'''


class CustomTools(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.w = Workspace(self.d)
        H.load_all()

    def test_add_totals_with_no_range_totals_the_filled_block(self):
        doc = F.new("sheets")
        cells = doc["sheets"][0]["cells"]
        for r, row in enumerate([["Item", "Cost", "Qty"], ["Rent", 800, 1], ["Food", 250, 4], ["Bus", 90, 10]], 1):
            for c, v in enumerate(row):
                cells[f"{chr(65 + c)}{r}"] = v
        F.save(Path(self.d) / "b.ssheet", doc)
        out = T.run(self.w, "herder_run", {"name": "add_totals", "path": "b.ssheet"})
        self.assertIn("B, C", out)
        got = F.load(Path(self.d) / "b.ssheet")["sheets"][0]["cells"]
        self.assertEqual(got["A5"], "Total")
        self.assertEqual(got["B5"], "=SUM(B2:B4)")

    def test_builtin_is_offered_by_its_phrase(self):
        allowed = set(T.TOOLS)
        self.assertIn("h_number_headings", agent.route_tools(allowed, "", "number the headings in report.sdoc"))

    def test_model_made_tool_is_offered_after_one_clean_run(self):
        T.run(self.w, "herder_make", {"name": "shout_heads", "app": "words", "description": "Upper-case headings",
                                      "code": UPPER, "asks": ["shout the headings"]})
        allowed = set(T.TOOLS)
        ask = "shout the headings in r.sdoc"
        self.assertNotIn("h_shout_heads", agent.route_tools(allowed, "", ask))
        doc = F.new("words")
        doc["blocks"].append({"type": "heading", "level": 1, "text": "Plan"})
        F.save(Path(self.d) / "r.sdoc", doc)
        T.run(self.w, "herder_run", {"name": "shout_heads", "path": "r.sdoc"})
        self.assertTrue(json.loads((H.USER_DIR / "shout_heads.json").read_text(encoding="utf-8"))["proven"])
        self.assertIn("h_shout_heads", agent.route_tools(allowed, "", ask))
        T.run(self.w, "herder_delete", {"name": "shout_heads"})


if __name__ == "__main__":
    unittest.main()
