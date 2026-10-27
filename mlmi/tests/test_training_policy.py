import unittest
from unittest.mock import patch

from mlmi.backend.training_policy import checker_training_advice, nvidia_thermal


class TrainingPolicyTests(unittest.TestCase):
    def test_sensorless_fit_recommends_cpu_but_exposes_bounded_gpu_choice(self):
        advice = checker_training_advice(cuda_ready=True, free_vram_gb=8, available_ram_gb=16,
                                         thermal={"available": False, "reason": "no sensor"})
        self.assertEqual(advice["recommended"], "cpu")
        self.assertTrue(advice["gpu_available"])
        self.assertEqual(advice["sensorless_chunk_seconds"], 300)

    def test_gpu_fit_and_readable_temperature_recommend_bounded_training(self):
        advice = checker_training_advice(cuda_ready=True, free_vram_gb=8, available_ram_gb=16,
                                         thermal={"available": True, "current_c": 60,
                                                  "pause_at_c": 87, "resume_below_c": 82})
        self.assertEqual(advice["recommended"], "gpu_bounded")

    def test_fit_scales_for_large_checkpoint_and_calls_out_low_memory(self):
        advice = checker_training_advice(cuda_ready=True, free_vram_gb=3.9, available_ram_gb=8,
                                         source_bytes=12 * 2**30,
                                         thermal={"available": False})
        self.assertEqual(advice["gpu_need_gb"], 4.0)
        self.assertFalse(advice["gpu_available"])
        self.assertFalse(advice["cpu_available"])
        self.assertIn("CPU needs about", advice["reason"])

    def test_unknown_ram_does_not_claim_cpu_training_fits(self):
        advice = checker_training_advice(cuda_ready=False, available_ram_gb=None,
                                         thermal={"available": False})
        self.assertFalse(advice["cpu_available"])

    def test_gpu_fit_can_use_chat_yielded_headroom_with_lower_memory_estimate(self):
        advice = checker_training_advice(cuda_ready=True, free_vram_gb=2.8, available_ram_gb=8,
                                         source_bytes=3 * 2**30,
                                         thermal={"available": True, "current_c": 60, "pause_at_c": 87})
        self.assertEqual(advice["gpu_need_gb"], 2.5)
        self.assertTrue(advice["gpu_available"])

    def test_nvidia_sensor_parses_driver_thermal_limits(self):
        stdout = """GPU Current Temp                       : 77 C
        GPU Target Temperature                : 87 C
        GPU Slowdown Temp                     : 97 C
        GPU Shutdown Temp                     : 100 C
"""
        with patch("mlmi.backend.training_policy.shutil.which", return_value="nvidia-smi"), \
                patch("mlmi.backend.training_policy.subprocess.run", return_value=type("Result", (), {
                    "returncode": 0, "stdout": stdout, "stderr": ""})()):
            result = nvidia_thermal()
        self.assertTrue(result["available"])
        self.assertEqual(result["target_c"], 87)
        self.assertEqual(result["pause_at_c"], 89)
        self.assertEqual(result["resume_below_c"], 87)


if __name__ == "__main__":
    unittest.main()
