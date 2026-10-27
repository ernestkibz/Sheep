"""Settings › Basics: "How Sheep runs" is three presets over the Advanced settings; a hand change shows custom."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import server  # noqa: E402


class Presets(unittest.TestCase):
    def test_a_fresh_config_is_balanced(self):
        self.assertEqual(server.preset_of({}), "balanced")

    def test_each_bundle_reads_back_as_itself(self):
        for name, bundle in server.PRESETS.items():
            self.assertEqual(server.preset_of(dict(bundle)), name)

    def test_a_hand_change_is_custom(self):
        self.assertEqual(server.preset_of({"vram_reserve_gb": 1.5}), "custom")
        self.assertEqual(server.preset_of({"thermal_guard": False}), "custom")

    def test_numbers_saved_as_ints_still_match(self):
        self.assertEqual(server.preset_of({"pool_mode": "off", "vram_reserve_gb": 2, "pool_gap": 4}), "quiet")

    def test_presets_only_use_keys_the_settings_endpoint_accepts(self):
        src = (Path(server.__file__)).read_text(encoding="utf-8")
        for bundle in server.PRESETS.values():
            for k in bundle:
                self.assertIn(f'"{k}" in b', src)


if __name__ == "__main__":
    unittest.main()
