"""Keep long model installs visibly measurable in the desktop chooser."""
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ModelInstallProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ui = ROOT / "mlmi" / "desktop" / "ui"
        cls.app = (ui / "app.js").read_text(encoding="utf-8")
        cls.css = (ui / "app.css").read_text(encoding="utf-8")

    def test_download_bytes_and_build_layers_feed_accessible_progress(self):
        self.assertIn('data-progress max="100" hidden aria-label="Model download and build progress"', self.app)
        self.assertIn(r"/downloaded\s+([\d.]+)\s+of\s+([\d.]+)\s+GB/i", self.app)
        self.assertIn("transferDone / transferTotal * 100", self.app)
        self.assertIn('setInstallProgress(Number(q.progress) * 100', self.app)
        self.assertIn("installProgress.removeAttribute(\"value\")", self.app)
        self.assertIn("accent-color:var(--accent)", self.css)


if __name__ == "__main__":
    unittest.main()
