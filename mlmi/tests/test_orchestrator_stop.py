"""Cancellation remains observable before and during background model startup."""
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend.orchestrator import Orchestrator  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.model = ""
        self.stopped = False

    def alive(self):
        return False

    def stop(self):
        self.stopped = True


def make_orchestrator():
    orch = Orchestrator.__new__(Orchestrator)
    orch.st = SimpleNamespace(model_path="target.ssf", engine=FakeEngine(), saved=0)
    orch.st.save = lambda: setattr(orch.st, "saved", orch.st.saved + 1)
    orch.lock = threading.Lock()
    orch.cancelled = threading.Event()
    orch.loading = threading.Event()
    orch.load_engine_replaced = False
    orch.on = True
    orch.lamp = "amber"
    orch.stage = ""
    orch.progress = 0.0
    orch.log = []
    return orch


class OrchestratorStopTests(unittest.TestCase):
    def test_cancel_before_worker_starts_is_not_lost(self):
        orch = make_orchestrator()
        threads = []

        class DeferredThread:
            def __init__(self, target, args=(), daemon=None):
                self.target, self.args = target, args
                threads.append(self)

            def start(self):
                pass

        orch._start = unittest.mock.Mock(return_value=True)
        with patch("mlmi.backend.orchestrator.threading.Thread", DeferredThread):
            orch.start_model("target.ssf")
            self.assertTrue(orch.loading.is_set())
            self.assertEqual(orch.cancel_load(), {"cancelled": True})
            threads[0].target(*threads[0].args)

        orch._start.assert_not_called()
        self.assertFalse(orch.loading.is_set())
        self.assertFalse(orch.cancelled.is_set())
        self.assertEqual(orch.st.model_path, "")
        self.assertEqual(orch.st.saved, 1)

    def test_cancel_during_startup_stops_replaced_engine(self):
        orch = make_orchestrator()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def start(path):
            entered.set()
            release.wait(2)
            orch.load_engine_replaced = True
            return False if orch._stop_if_cancelled() else True

        orch._start = start
        worker = threading.Thread(target=lambda: (orch.start_model("target.ssf", on_ready=lambda _: finished.set())))
        worker.start()
        self.assertTrue(entered.wait(2))
        self.assertTrue(orch.loading.is_set())
        self.assertEqual(orch.cancel_load(), {"cancelled": True})
        release.set()
        worker.join(3)

        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.wait(3))
        self.assertTrue(finished.is_set())
        self.assertTrue(orch.st.engine.stopped)
        self.assertEqual(orch.st.model_path, "")
        self.assertFalse(orch.loading.is_set())


if __name__ == "__main__":
    unittest.main()
