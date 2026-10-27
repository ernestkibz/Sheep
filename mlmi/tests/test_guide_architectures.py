"""Keep the Guide's static family list faithful to Sheep's inference roster."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tooling import sml_archs, sml_compat


class GuideArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guide = (ROOT / "mlmi" / "desktop" / "ui" / "guide.js").read_text(encoding="utf-8")

    def test_dropdown_matches_exact_build_gate_roster(self):
        match = re.search(r"const SUPPORTED_ARCHITECTURES\s*=\s*\[(.*?)\];", self.guide, re.S)
        self.assertIsNotNone(match, "Guide must keep the repository-backed architecture dropdown")
        names = re.findall(r'"([^\"]+)"', match.group(1))
        self.assertEqual(len(names), len(set(names)), "dropdown must not duplicate families")
        self.assertEqual(set(names), {name for name, row in sml_archs.ROSTER.items() if row[0] == "ok"})

    def test_counts_are_generated_from_the_evidence(self):
        # every number the guide states comes from tooling/sml_compat.py --emit-guide
        facts = json.loads(re.search(r"const MODEL_FACTS = (\{.*?\});", self.guide).group(1))
        want = sml_compat.facts()
        for k in ("roster", "ok", "gap", "scope", "unknown", "checked", "heads", "headsPass"):
            with self.subTest(fact=k):
                self.assertEqual(facts[k], want[k], "run: python tooling/sml_compat.py --emit-guide")
        checked = re.search(r"const CHECKED_ARCHITECTURES\s*=\s*\[(.*?)\];", self.guide, re.S)
        self.assertEqual(re.findall(r'"([^\"]+)"', checked.group(1)), want["runs"])
        self.assertNotIn("83 of 139", self.guide, "no hand-typed counts")

    def test_guide_separates_gate_grade_and_hardware_fit_evidence(self):
        self.assertIn("<b>Build gate:</b>", self.guide)
        self.assertIn("<b>Checked end to end:</b>", self.guide)
        self.assertIn("<b>Real-checkpoint grades:</b>", self.guide)
        self.assertIn("${MODEL_FACTS.ok} of ${MODEL_FACTS.roster} families", self.guide)
        self.assertIn("9 families", self.guide)
        self.assertIn("<details class=\"g-tech\">", self.guide)
        self.assertIn("depends on its size and your computer’s RAM, CPU, GPU and graphics memory", self.guide)
        self.assertIn("Auto is a good place to start", self.guide)
        self.assertIn("const QUICK_PARTS = [", self.guide)
        self.assertIn("const SLOTS = [", self.guide)


if __name__ == "__main__":
    unittest.main()
