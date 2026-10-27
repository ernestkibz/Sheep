import unittest
from types import SimpleNamespace

from mlmi.backend.orchestrator import feature_models_kpi, kpis


class FeatureModelKpiTests(unittest.TestCase):
    def test_reports_ready_slots_and_each_selected_model(self):
        result = feature_models_kpi([
            {"id": "chat", "label": "Chat", "active": "chat.ssf", "active_name": "Chat model", "ready": True},
            {"id": "video", "label": "Video", "active": "motion", "active_name": "Motion model", "ready": False},
            {"id": "speech_output", "label": "Speech output", "active": "", "active_name": "Not selected", "ready": False},
        ])

        self.assertEqual(result["rag"], "amber")
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["unit"], "/ 2")
        self.assertEqual(result["detail"][0], ["Chat", "Chat model · ready"])
        self.assertEqual(result["detail"][1], ["Video", "Motion model · selected · not ready"])
        self.assertEqual(result["detail"][2], ["Speech output", "Not selected · not selected"])

    def test_no_models_is_neutral_and_disabled_slots_are_not_counted_as_failures(self):
        empty = feature_models_kpi([])
        disabled = feature_models_kpi([
            {"id": "video", "label": "Video", "active": "motion", "active_name": "Motion", "ready": False, "disabled": True},
        ])

        self.assertEqual(empty["rag"], "none")
        self.assertIsNone(empty["value"])
        self.assertEqual(disabled["rag"], "none")
        self.assertEqual(disabled["detail"][0][1], "Motion · disabled in Settings")

    def test_hardware_kpis_show_settings_preferences_and_live_multi_device_split(self):
        orch = SimpleNamespace(
            plan={"device": "multi", "gpu_layers": 3, "mid_layers": 2, "device_name": "AMD Radeon",
                  "gpu": {"fit": "split", "layers": 5}, "ram_fit": True, "threads": 8,
                  "ctx": 1024, "slots": 1, "est_tok_s": 4},
            facts={"n_layer": 10, "bytes": 1_000_000_000, "arch": "llama"},
            st=SimpleNamespace(model_path=""), tok_s=lambda: 4,
            turns={"prompt": 0, "prompt_ms": 0, "trips": 0, "n": 0}, frame="own")
        device = {"gpu": {}, "ram": {"available": 4_000_000_000, "total": 8_000_000_000},
                  "cpu": {"name": "CPU", "threads": 8}, "power": {"plugged": True}, "others": []}

        cards = kpis(orch, device, [], {"device": "multi", "media_device": "directml"})
        fit = next(card for card in cards if card["id"] == "fit")
        details = dict(fit["detail"])
        self.assertEqual(details["runs on"], "NVIDIA 3 layers + AMD Radeon 2 + CPU 5")
        self.assertIn("Multi-GPU + CPU", details["Settings preference"])
        self.assertIn("DirectML (AMD / Intel)", details["Settings preference"])


if __name__ == "__main__":
    unittest.main()
