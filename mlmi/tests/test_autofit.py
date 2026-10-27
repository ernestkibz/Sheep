"""Auto-fit: a saved fit reaches the builder's flags and the roster; the search finds a known family blind."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tooling"))

import sml_archs  # noqa: E402
import sml_hf  # noqa: E402

CFG = {"model_type": "never_listed_lm", "hidden_size": 256, "num_attention_heads": 4, "num_hidden_layers": 2,
       "hidden_act": "silu"}


class AutofitHookTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.fits = os.path.join(self.dir.name, "fits.json")
        env = mock.patch.dict(os.environ, {"MLMI_FITS": self.fits})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SML_FLAGS_OVERRIDE", None)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, fits):
        with open(self.fits, "w", encoding="utf-8") as f:
            json.dump(fits, f)

    def test_unlisted_type_stays_unlisted_without_a_fit(self):
        self.assertIsNone(sml_archs.check("never_listed_lm"))
        self.assertEqual(sml_hf.arch_flags(CFG)["rope_type"], 0)

    def test_saved_fit_sets_the_flags_and_the_roster_row(self):
        self.write({"never_listed_lm": {"pass": True, "rel": 0.042, "t": "2026-10-20T11:00:00",
                                        "flags": {"rope_type": 1, "qknorm_late": True}}})
        f = sml_hf.arch_flags(CFG)
        self.assertEqual((f["rope_type"], f["qknorm_late"]), (1, True))
        row = sml_archs.check("never_listed_lm")
        self.assertEqual(row[0], "ok")
        self.assertIn("auto-fitted on 2026-10-20", row[2])
        self.assertTrue(sml_archs.build_gate("never_listed_lm")[1])

    def test_failed_fit_changes_nothing(self):
        self.write({"never_listed_lm": {"pass": False, "why": "no transformers class", "flags": {"rope_type": 1}}})
        self.assertIsNone(sml_archs.check("never_listed_lm"))
        self.assertEqual(sml_hf.arch_flags(CFG)["rope_type"], 0)

    def test_override_wins_over_the_fit(self):
        self.write({"never_listed_lm": {"pass": True, "rel": 0.04, "flags": {"rope_type": 1}}})
        with mock.patch.dict(os.environ, {"SML_FLAGS_OVERRIDE": json.dumps({"rope_type": 0, "norm_type": 1})}):
            f = sml_hf.arch_flags(CFG)
        self.assertEqual((f["rope_type"], f["norm_type"]), (0, 1))

    def test_listed_ok_row_ignores_fits(self):
        self.write({"llama": {"pass": True, "rel": 0.5, "flags": {}}})
        self.assertNotIn("auto-fitted", sml_archs.check("llama")[2])

    def test_tiny_copy_drops_sizes_and_cuts_per_layer_lists(self):
        import sml_autofit
        over = sml_autofit.tiny_over({"model_type": "x", "hidden_size": 4096, "num_hidden_layers": 4,
                                      "layer_types": ["a", "b", "a", "b"], "mlp_only_layers": [0, 3],
                                      "rope_theta": 1e6, "auto_map": {"AutoModel": "x.X"}})
        self.assertEqual(over, {"layer_types": ["a", "b"], "mlp_only_layers": [0], "rope_theta": 1e6})


CLI = ROOT / "sml" / ("sml-cli.exe" if os.name == "nt" else "sml-cli")
HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


@unittest.skipUnless(CLI.exists() and HAVE_TORCH and os.environ.get("SML_TOK_SRC"),
                     "needs the engine, transformers and a GPT-2 tokenizer (SML_TOK_SRC)")
class BlindFitTests(unittest.TestCase):
    def test_glm_rediscovered_blind(self):
        import sml_autofit
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"MLMI_FITS": os.path.join(d, "f.json")}):
            fit = sml_autofit.blind("glm", log=lambda s: None)
        self.assertTrue(fit["pass"], fit)
        self.assertEqual(fit["flags"].get("rope_type"), 1)


if __name__ == "__main__":
    unittest.main()
