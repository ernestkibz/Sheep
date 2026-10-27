import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tooling import sml_kl_guard as guard


class TestKLMemGuard(unittest.TestCase):
    def test_start_requires_floor_plus_startup_headroom(self):
        required = guard.MEMORY_FLOOR + guard.STARTUP_HEADROOM
        self.assertTrue(guard.can_start_with_memory(required))
        self.assertFalse(guard.can_start_with_memory(required - 1))

    def test_early_stop_threshold_keeps_margin_above_floor(self):
        threshold = guard.MEMORY_FLOOR + guard.STOP_HEADROOM
        self.assertTrue(guard.should_stop_for_memory(threshold - 1))
        self.assertFalse(guard.should_stop_for_memory(threshold))
        self.assertGreater(threshold, guard.MEMORY_FLOOR)

    def test_low_memory_preflight_does_not_launch_or_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit" / "candidate.ssf"
            with patch.object(guard.psutil, "virtual_memory",
                              return_value=SimpleNamespace(available=3 * 1024**3)), \
                 patch.object(guard.subprocess, "Popen") as launch:
                result = guard.build_with_memory_floor("unused", "q2", output)
            self.assertEqual(result["state"], "resource_blocked")
            self.assertIsNone(result["partial_file"])
            self.assertFalse(output.parent.exists())
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
