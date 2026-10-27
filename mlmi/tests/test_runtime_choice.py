"""A model type only one PyTorch runtime's Transformers knows is fitted and run by that runtime
(Youtu-LLM-2B, 2026-10-20: the CUDA runtime had Transformers 4.57, the CPU one 5.20)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import gpu_runtime as G  # noqa: E402


class RuntimeChoice(unittest.TestCase):
    def test_runtime_that_knows_the_type_wins(self):
        with patch.object(G, "ready", return_value=True), \
                patch.object(G, "knows", side_effect=lambda kind, mt: kind == "cpu"), \
                patch.object(G, "py", side_effect=lambda kind: Path(kind)):
            self.assertEqual(G.torch_python_for("youtu"), (Path("cpu"), "cpu"))

    def test_no_type_or_none_knows_falls_back(self):
        with patch.object(G, "ready", return_value=True), \
                patch.object(G, "knows", return_value=False), \
                patch.object(G, "torch_python", return_value=(Path("cuda"), "cuda")) as tp:
            self.assertEqual(G.torch_python_for("never_heard"), (Path("cuda"), "cuda"))
            self.assertEqual(G.torch_python_for(""), (Path("cuda"), "cuda"))
            self.assertEqual(tp.call_count, 2)


if __name__ == "__main__":
    unittest.main()
