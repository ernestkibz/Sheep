import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlmi.backend import server


class ManagedModelRemovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.originals = self.root / "originals"
        self.models.mkdir()
        self.source = self.originals / "org--model"
        self.source.mkdir(parents=True)
        self.model = self.models / "model.ssf"
        self.model.write_bytes(b"test model placeholder")
        self.entry = {"hf_repo": "org/model", "source_dir": str(self.source), "licence": "apache-2.0"}
        self.state = SimpleNamespace(model_path="", cfg={"checker": {"path": ""}})
        self.home = patch.object(server, "HOME", self.root)
        self.home.start()
        self.addCleanup(self.home.stop)
        self.registry = patch.object(server, "load_registry", return_value={str(self.model.resolve()): self.entry})
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.save = patch.object(server, "save_registry")
        self.save.start()
        self.addCleanup(self.save.stop)
        self.validate = patch.object(server, "validate_ssf_path", side_effect=lambda value: str(Path(value).resolve()))
        self.validate.start()
        self.addCleanup(self.validate.stop)

    def tearDown(self):
        self.temp.cleanup()

    def test_removes_managed_file_and_unused_original_only(self):
        result = server.remove_managed_model(self.state, str(self.model))
        self.assertFalse(self.model.exists())
        self.assertFalse(self.source.exists())
        self.assertTrue(result["original_removed"])
        self.assertTrue(result["shared_hugging_face_cache_preserved"])
        server.save_registry.assert_called_once_with({})

    def test_refuses_active_model_without_changes(self):
        self.state.model_path = str(self.model)
        with self.assertRaisesRegex(RuntimeError, "active in Chat or Checker"):
            server.remove_managed_model(self.state, str(self.model))
        self.assertTrue(self.model.exists())
        self.assertTrue(self.source.exists())

    def test_keeps_original_used_by_another_managed_model(self):
        second = self.models / "other.ssf"
        second.write_bytes(b"another model")
        shared_registry = {
            str(self.model.resolve()): self.entry,
            str(second.resolve()): {"hf_repo": "org/model", "source_dir": str(self.source)},
        }
        server.load_registry.return_value = shared_registry
        result = server.remove_managed_model(self.state, str(self.model))
        self.assertFalse(self.model.exists())
        self.assertTrue(self.source.exists())
        self.assertFalse(result["original_removed"])

    def test_refuses_unregistered_model_and_external_path(self):
        server.load_registry.return_value = {}
        with self.assertRaisesRegex(ValueError, "not installed through Sheep"):
            server.remove_managed_model(self.state, str(self.model))
        external = self.root / "outside.ssf"
        external.write_bytes(b"placeholder")
        server.load_registry.return_value = {str(external.resolve()): self.entry}
        with self.assertRaisesRegex(ValueError, "only Sheep-managed"):
            server.remove_managed_model(self.state, str(external))
        self.assertTrue(external.exists())


if __name__ == "__main__":
    unittest.main()
