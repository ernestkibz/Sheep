"""Part C4: a compatibility-mode model is rebuilt natively once Sheep's engine runs its type."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tooling"))
from mlmi.backend import orchestrator  # noqa: E402


def orch(path, quant_state="idle"):
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    o.st = SimpleNamespace(model_path=path, quant=SimpleNamespace(state=quant_state, start=Mock()), save=Mock())
    o.say = Mock()
    o.history = Mock()
    o.lock = threading.Lock()
    return o


class Upgrade(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        src = Path(self.d) / "orig"
        src.mkdir()
        (src / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
        self.path = str(Path(self.d) / "m.compat.json")
        Path(self.path).write_text(json.dumps({"src": str(src)}), encoding="utf-8")

    def test_rebuilds_when_the_engine_runs_the_type(self):
        o = orch(self.path)
        with patch("sml_archs.check", return_value=("ok", (), "")):
            self.assertTrue(o.upgrade_compat(self.path))
        o.st.quant.start.assert_called_once()

    def test_leaves_it_when_the_type_is_still_unsupported(self):
        o = orch(self.path)
        with patch("sml_archs.check", return_value=("gap", (), "")):
            self.assertFalse(o.upgrade_compat(self.path))

    def test_waits_while_another_build_runs(self):
        o = orch(self.path, quant_state="building")
        with patch("sml_archs.check", return_value=("ok", (), "")):
            self.assertFalse(o.upgrade_compat(self.path))


if __name__ == "__main__":
    unittest.main()
