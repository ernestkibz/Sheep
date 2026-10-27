import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CheckerTrainingSetupTests(unittest.TestCase):
    def test_training_device_is_saved_in_settings_and_reused_without_a_modal(self):
        ui = (ROOT / "desktop" / "ui" / "views.js").read_text(encoding="utf-8")
        backend = (ROOT / "backend" / "server.py").read_text(encoding="utf-8")
        self.assertIn('id="hCheckerTraining"', ui)
        self.assertIn('checker_training_mode: training.value', ui)
        self.assertIn('const savedMode = hw.prefs?.checker_training_mode || "auto"', ui)
        start = ui.index("const trainingChoice = async")
        end = ui.index("this.trainingChoice = trainingChoice", start)
        self.assertNotIn('className = "modal"', ui[start:end])
        self.assertIn('"checker_training_mode": self.cfg.get("checker_training_mode", "auto")', backend)
        self.assertIn('training_mode not in ("auto", "cpu", "gpu_bounded")', backend)

    def test_temperature_note_describes_the_heat_guard(self):
        # Until 2026-10-12 only Checker training watched temperature, and the
        # laptop shut down from heat; now Sheep paces everything by it (thermal.py).
        ui = (ROOT / "desktop" / "ui" / "views.js").read_text(encoding="utf-8")
        # the heat guard's full explanation sits beside its switch (2026-10-26: it was
        # stranded in the Advanced fold); the Checker training choice stays in that fold
        self.assertIn('id="hHeat"', ui)
        start = ui.index('id="hHeat"')
        note = ui[start:ui.index('id="hAdv"', start)]
        self.assertIn("background checks run in bursts", note)
        adv = ui[ui.index('id="hAdv"'):ui.index('id="hList"')]
        self.assertIn('id="hCheckerTraining"', adv)
        self.assertNotIn("Chat, image, and video work are never paused", note)


if __name__ == "__main__":
    unittest.main()
