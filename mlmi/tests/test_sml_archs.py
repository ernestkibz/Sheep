"""Architecture roster and quantizer build-gate behavior."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tooling"))
import sml_archs  # noqa: E402


class ArchitectureRosterTests(unittest.TestCase):
    def test_summary_accounts_for_every_rostered_architecture(self):
        counts, needs = sml_archs.summary()
        self.assertEqual(sum(counts.values()), len(sml_archs.ROSTER))
        self.assertEqual(set(counts), {"ok", "gap", "unknown", "scope"})
        self.assertTrue(needs)

    def test_models_doc_roster_counts_match_the_current_table(self):
        counts, _ = sml_archs.summary()
        readme = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")   # the table moved from README.md
        total = re.search(r"rosters \*\*(\d+) architectures\*\*", readme)
        self.assertIsNotNone(total)
        self.assertEqual(int(total.group(1)), len(sml_archs.ROSTER))
        labels = {
            "roster marked `ok`": "ok",
            "gap": "gap",
            "out of scope": "scope",
            "unknown": "unknown",
        }
        rows = {
            match.group(1).replace("*", "").strip(): int(match.group(2))
            for match in re.finditer(r"^\|\s*(.*?)\s*\|\s*\**(\d+)\**\s*\|",
                                     readme, re.MULTILINE)
        }
        for label, status in labels.items():
            with self.subTest(status=status):
                self.assertIn(label, rows, f"docs/MODELS.md is missing {status} count")
                self.assertEqual(rows[label], counts[status])

    def test_aliases_resolve_to_the_canonical_roster_row(self):
        for alias, canonical in sml_archs.ALIASES.items():
            with self.subTest(alias=alias):
                self.assertIsNotNone(sml_archs.check(alias))
                self.assertEqual(sml_archs.check(alias), sml_archs.check(canonical))

    def test_falcon_support_depends_on_checkpoint_rope_config(self):
        alibi_row = sml_archs.check("falcon")
        rope_row = sml_archs.check("falcon", {"rope_theta": 10000})
        self.assertEqual(alibi_row[0], "ok")
        self.assertEqual(rope_row[0], "ok")
        self.assertNotEqual(alibi_row[2], rope_row[2])
        self.assertIn("RoPE", rope_row[2])

    def test_unknown_architecture_fails_closed_with_explanation(self):
        self.assertIsNone(sml_archs.check("not_a_real_architecture"))
        self.assertIn("not in the roster", sml_archs.explain("not_a_real_architecture"))

    def test_explanations_distinguish_supported_gap_unknown_and_scope(self):
        self.assertIn("supported", sml_archs.explain("llama"))
        gap = next(name for name, row in sml_archs.ROSTER.items() if row[0] == "gap")
        unknown = next(name for name, row in sml_archs.ROSTER.items() if row[0] == "unknown")
        scope = next(name for name, row in sml_archs.ROSTER.items() if row[0] == "scope")
        self.assertIn("NOT supported", sml_archs.explain(gap))
        self.assertIn("NOT established", sml_archs.explain(unknown))
        self.assertIn("out of scope", sml_archs.explain(scope))

    def test_build_gate_allows_only_ok_rows_without_exact_override(self):
        for model_type, row in sml_archs.ROSTER.items():
            with self.subTest(model_type=model_type):
                actual, allowed = sml_archs.build_gate(model_type)
                self.assertEqual(actual, row)
                self.assertEqual(allowed, row[0] == "ok")
        self.assertEqual(sml_archs.build_gate("unlisted")[1], False)

    def test_override_is_exact_and_marks_ungraded_builds_allowed(self):
        for model_type in ("unlisted", next(
                name for name, row in sml_archs.ROSTER.items() if row[0] != "ok")):
            with self.subTest(model_type=model_type):
                row, allowed = sml_archs.build_gate(model_type, override=model_type)
                self.assertTrue(allowed)
                self.assertTrue(row is None or row[0] != "ok")
        self.assertFalse(sml_archs.build_gate("unlisted", override="llama")[1])


if __name__ == "__main__":
    unittest.main()
