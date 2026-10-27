"""What the B10 proof pass found (2026-10-21): the video's "add three slides to the deck" was refused
as out of scope; the 3B titled slides "<title of new slide 1>"; "i want a file" got a pretend file;
with RAM short the plan's 2048 context could not hold Sheep's own job prompt."""
import re
import tempfile
import unittest
from pathlib import Path

from mlmi.backend import agent, orchestrator as o, shepherd
from mlmi.backend.workspace import Workspace


class B10(unittest.TestCase):
    def test_the_jobs_own_target_is_in_scope(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            (Path(d) / "nemotron_presentation.sdeck").write_text("{}", encoding="utf-8")
            args = {"path": "nemotron_presentation.sdeck", "title": "x", "bullets": ["y"]}
            asked = "add three slides to the deck"
            self.assertIn("not part of this request", agent.in_scope(ws, "slides_add", args, asked, "", []))
            self.assertIsNone(agent.in_scope(ws, "slides_add", args, asked + "\nWork on: nemotron_presentation.sdeck",
                                             "", []))

    def test_a_template_title_is_replaced_by_the_first_point(self):
        slides = shepherd.parse_slides("## <title of new slide 1>\n- Nemotron Nano: up to 30% faster.\n- Open weights\n")[2]
        self.assertEqual(slides[0]["title"], "Nemotron Nano")
        self.assertEqual(len(slides[0]["bullets"]), 2)
        # a "<point>" bullet stays, so the placeholder check still rewrites it
        self.assertIn("<point>", shepherd.parse_slides("## Forest\n- <point>\n- Trees\n")[2][0]["bullets"])

    def test_a_bare_ask_for_a_file(self):
        for t in ("i want a file", "I need the file", "give me the file please", "where is my file?"):
            self.assertTrue(re.match(agent.FILE_ASK, t, re.I), t)
        for t in ("i want a file about dogs", "i want a powerpoint about dogs", "a file"):
            self.assertFalse(re.match(agent.FILE_ASK, t, re.I), t)

    def test_context_never_below_the_job_prompt(self):
        dev = {"ram": {"available": 2.5e9, "total": 8e9}, "cpu": {"cores": 4, "name": "CPU"}, "power": {"plugged": True},
               "gpu": None, "amd": None, "pool": []}
        p = o.plan(dev, {"bytes": 1.9e9, "n_weights": 3.1e9, "n_layer": 36, "kv_per_token": 36864, "n_ctx_train": 32768})
        self.assertGreaterEqual(p["ctx"], 4096)
        p = o.plan(dev, {"bytes": 0.3e9, "n_weights": 0.1e9, "n_layer": 12, "kv_per_token": 9216, "n_ctx_train": 2048})
        self.assertEqual(p["ctx"], 2048)          # a model trained on less keeps its own length


if __name__ == "__main__":
    unittest.main()
