"""The heat governor (thermal.py): levels with hysteresis, sustained load, and the
waits it imposes. The laptop shut down from heat on 2026-10-12."""
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import thermal as T  # noqa: E402


def gov(on=True):
    return T.Governor(enabled=lambda: on, read=lambda: (None, None, None))


# temperatures are written against the limits, so the tests follow any retuning
GW, GH, GC = T.GPU_AT
CW, CH, CC = T.CPU_AT
R = T.RELEASE


class LevelTests(unittest.TestCase):
    def test_rises_with_heat_and_falls_only_below_release(self):
        g = gov()
        g.update(GW - 10, 60, 10, now=0)
        self.assertEqual(g.name, "cool")
        g.update(GH + 1, 60, 10, now=1)
        self.assertEqual(g.name, "hot")
        g.update(GH - R + 1, 60, 10, now=2)      # below hot, above its release: stays hot
        self.assertEqual(g.name, "hot")
        g.update(GH - R - 1, 60, 10, now=3)      # below the release: warm
        self.assertEqual(g.name, "warm")
        g.update(GC + 1, 60, 10, now=4)
        self.assertEqual(g.name, "critical")

    def test_cpu_counts_as_much_as_gpu(self):
        g = gov()
        g.update(GW - 10, CH + 1, 10, now=0)
        self.assertEqual(g.name, "hot")

    def test_sustained_load_while_warm_is_hot(self):
        g = gov()
        g.update(GW + 1, 60, 95, now=0)
        self.assertEqual(g.name, "warm")
        g.update(GW + 1, 60, 95, now=T.LOAD_FOR_S + 1)
        self.assertEqual(g.name, "hot")
        g.update(GW + 1, 60, 20, now=T.LOAD_FOR_S + 2)
        self.assertEqual(g.name, "warm")

    def test_switched_off_never_paces(self):
        g = gov(on=False)
        g.update(GC + 5, CC + 5, 100, now=0)
        self.assertEqual(g.name, "cool")
        self.assertIsNone(g.background_wait())


class WaitTests(unittest.TestCase):
    def test_warm_runs_at_full_speed(self):
        g = gov()
        g.update(GW + 1, 60, 10, now=0)
        self.assertEqual(g.name, "warm")
        self.assertIsNone(g.background_wait(now=0))
        self.assertEqual(g.pace(), 0.0)

    def test_hot_runs_background_in_bursts_and_never_starves(self):
        g = gov()
        g.update(GH + 1, 60, 10, now=0)
        self.assertIsNone(g.background_wait(now=0))
        self.assertIsNotNone(g.background_wait(now=T.HOT_RUN_S + 1))
        self.assertIsNone(g.background_wait(now=T.HOT_RUN_S + T.HOT_REST_S + 1))
        self.assertIsNone(g.gpu_wait())
        self.assertEqual(g.pace(), 0.0)

    def test_critical_stops_background_and_new_gpu_work(self):
        g = gov()
        g.update(GC + 1, 60, 10, now=0)
        self.assertIsNotNone(g.background_wait(now=0))
        self.assertIsNotNone(g.gpu_wait())

    def test_critical_below_the_danger_line_costs_the_person_nothing(self):
        # 2026-10-26: chat waited at 93 C, the laptop's normal load heat, and fell behind llama.cpp
        g = gov()
        g.update(T.DANGER_GPU - 1, 60, 10, now=0)
        self.assertEqual(g.name, "critical")
        self.assertEqual(g.pace(), 0.0)
        self.assertIsNone(g.gpu_wait(person=True))
        self.assertIsNotNone(g.gpu_wait())                 # background GPU work still waits
        self.assertIsNotNone(g.background_wait(now=0))

    def test_pace_is_zero_in_normal_heat_and_rises_smoothly_to_the_danger_line(self):
        g = gov()
        for t, want in ((80, 0), (93, 0), (T.PACE_FROM_GPU, 0)):
            g.update(t, 60, 10, now=0)
            self.assertEqual(g.pace_ms(), want, t)
        g.update((T.PACE_FROM_GPU + T.DANGER_GPU) / 2, 60, 10, now=0)
        mid = g.pace_ms()
        self.assertTrue(0 < mid < T.PACE_MAX_MS)
        g.update(T.DANGER_GPU + 1, 60, 10, now=0)
        self.assertEqual(g.pace_ms(), T.PACE_MAX_MS)

    def test_pace_off_with_the_guard_off(self):
        g = T.Governor(enabled=lambda: False, read=lambda: (None, None, None), read_power=lambda: None)
        g.update(T.DANGER_GPU + 1, 60, 10, now=0)
        self.assertEqual(g.pace_ms(), 0)

    def test_chat_step_waits_at_the_danger_line_until_it_falls(self):
        g = gov()
        g.update(T.DANGER_GPU + 1, 60, 10, now=0)
        self.assertIsNotNone(g.gpu_wait(person=True))
        said = []
        threading.Timer(0.3, lambda: g.update(GW - 10, 60, 10, now=5)).start()
        t0 = time.monotonic()
        waited = g.pace(said.append)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertGreater(waited, 0)
        self.assertTrue(said and "cool" in said[0]["what"])

    def test_chat_step_stops_waiting_when_stopped(self):
        g = gov()
        g.update(T.DANGER_GPU + 1, 60, 10, now=0)
        stop = threading.Event()
        threading.Timer(0.2, stop.set).start()
        t0 = time.monotonic()
        g.pace(None, stop)
        self.assertLess(time.monotonic() - t0, 4)


class PowerTests(unittest.TestCase):
    """The shutdowns of 2026-10-12 were the battery running to 0 after the mains dropped."""

    def test_mains_changes_nothing(self):
        g = gov()
        g.update(60, 70, 10, now=0, power={"plugged": True, "percent": 15})
        self.assertEqual(g.power_state, "ac")
        self.assertIsNone(g.background_wait(now=0))

    def test_on_battery_background_waits_chat_goes_on(self):
        g = gov()
        g.update(60, 70, 10, now=0, power={"plugged": False, "percent": 60})
        self.assertEqual(g.power_state, "battery")
        self.assertIn("charger", g.background_wait(now=0))
        self.assertIsNone(g.gpu_wait())
        self.assertEqual(g.pace(), 0.0)

    def test_low_battery_holds_gpu_work_and_tells_the_person_once(self):
        g = gov()
        g.update(60, 70, 10, now=0, power={"plugged": False, "percent": 20})
        self.assertIsNotNone(g.gpu_wait())
        said = []
        g.pace(said.append)
        g.pace(said.append)
        self.assertEqual(len(said), 1)
        self.assertIn("plug in", said[0]["what"])
        g.update(60, 70, 10, now=1, power={"plugged": True, "percent": 21})
        self.assertIsNone(g.gpu_wait())

    def test_guard_off_ignores_battery(self):
        g = gov(on=False)
        g.update(60, 70, 10, now=0, power={"plugged": False, "percent": 5})
        self.assertEqual(g.power_state, "ac")


class StuckSensorTests(unittest.TestCase):
    """This laptop's only ACPI zone said 373.0 K whatever the load (2026-10-20)."""

    def test_a_reading_that_never_moves_while_the_load_does_is_ignored(self):
        g = gov()
        for i in range(T.STUCK_SAMPLES):
            g.update(60, 99.85, 10 if i % 2 else 80, now=i * T.PERIOD_S)
        self.assertTrue(g.cpu_stuck)
        self.assertIsNone(g.cpu)
        self.assertEqual(g.name, "cool")          # the GPU (60 °C) decides
        self.assertEqual(g.snapshot()["cpu_sensor"], "fixed value, ignored")

    def test_a_steady_reading_under_steady_load_is_believed(self):
        g = gov()
        for i in range(T.STUCK_SAMPLES):
            g.update(60, CH + 1, 50, now=i * T.PERIOD_S)
        self.assertFalse(g.cpu_stuck)
        self.assertEqual(g.name, "hot")

    def test_a_moving_reading_is_believed_again(self):
        g = gov()
        for i in range(T.STUCK_SAMPLES):
            g.update(60, 99.85, 10 if i % 2 else 80, now=i * T.PERIOD_S)
        g.update(60, 71.0, 30, now=999)
        self.assertFalse(g.cpu_stuck)
        self.assertEqual(g.cpu, 71.0)


if __name__ == "__main__":
    unittest.main()
