"""Deleted model files are removed from selectable metadata and current state."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import server  # noqa: E402


class ModelCatalogueTests(unittest.TestCase):
    def test_prune_removes_missing_registry_entry_and_clears_stale_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            kept = Path(directory) / "kept.ssf"
            kept.touch()
            missing = Path(directory) / "removed.ssf"
            registry = {str(kept): {"display": "Kept"}, str(missing): {"display": "Removed"}}
            engine = SimpleNamespace(stopped=False, stop=lambda: setattr(engine, "stopped", True))
            orch = SimpleNamespace(on=True, lamp="green", stage="", progress=1.0,
                                   cancel_load=lambda: None, plan={"device": "cpu"}, facts={"arch": "llama"})
            state = SimpleNamespace(model_path=str(missing), engine=engine, orch=orch, saved=0)
            state.save = lambda: setattr(state, "saved", state.saved + 1)
            stored = []

            with patch.object(server, "load_registry", return_value=registry), \
                    patch.object(server, "save_registry", side_effect=stored.append):
                server.prune_model_catalogue(state)

            self.assertEqual(stored, [{str(kept): {"display": "Kept"}}])
            self.assertEqual(state.model_path, "")
            self.assertTrue(engine.stopped)
            self.assertEqual(orch.plan, None)
            self.assertEqual(orch.facts, None)
            self.assertEqual(orch.stage, "selected model was removed from disk")
            self.assertEqual(state.saved, 1)
            self.assertTrue(kept.is_file())


if __name__ == "__main__":
    unittest.main()
