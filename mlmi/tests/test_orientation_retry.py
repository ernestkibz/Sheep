"""The UI retry reruns every failed app example without unloading Chat."""
import sys
import threading
import urllib.request
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import orchestrator, server  # noqa: E402


class OrientationRetryTests(unittest.TestCase):
    def test_retries_every_failed_task_and_preserves_chat_ready_state(self):
        tasks = [SimpleNamespace(id="sl008"), SimpleNamespace(id="sh001"), SimpleNamespace(id="fu004")]
        saved = {"orientation_validation": {
            "signature": "same-model", "state": "failed", "passed": 1, "of": 3,
            "done": 3, "total": 3,
            "rows": [
                {"id": "sl008", "ok": False, "why": "placeholder text", "s": 1},
                {"id": "sh001", "ok": True, "why": "", "s": 1},
                {"id": "fu004", "ok": False, "why": "0 list items, 5 asked", "s": 1},
            ],
        }, "knows_sheep": {"old": True}}
        updates = {}
        orch = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
        orch.st = SimpleNamespace(model_path="model.ssf")
        orch.ready = lambda: True
        orch.entry = lambda path=None: saved
        orch.validation_signature = lambda path: "same-model"
        orch.lock = threading.Lock()
        orch.plan = {"device_name": "Test GPU"}
        orch.background_checks = {}
        orch.update_entry_for = lambda path, **kw: updates.update(kw)

        def run_inprocess(state, selected, root):
            self.assertEqual([task.id for task in selected], ["sl008", "fu004"])
            return [{"id": "sl008", "ok": True, "why": "", "s": 3}, {"id": "fu004", "ok": True, "why": "", "s": 2}]

        class InlineThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        with patch.object(orchestrator, "threading") as threads, \
                patch("mlmi.backend.orient.quick", return_value=tasks), \
                patch("mlmi.backend.orient.run_inprocess", side_effect=run_inprocess):
            threads.Thread.side_effect = lambda target, daemon: InlineThread(target, daemon)
            result = orch.retry_orientation_failure()

        self.assertEqual(result, {"ok": True, "task_id": "sl008,fu004"})
        self.assertFalse(orch.lock.locked())
        self.assertEqual(orch.background_checks["orientation"]["state"], "passed")
        self.assertEqual(orch.background_checks["orientation"]["passed"], 3)
        self.assertEqual(updates["orientation_validation"]["rows"][0]["ok"], True)
        self.assertEqual(updates["orientation_validation"]["rows"][2]["ok"], True)
        self.assertEqual(updates["knows_sheep"]["old"], True)
        self.assertEqual(updates["knows_sheep"]["hints"], [])

    def test_retry_refuses_when_chat_is_not_ready(self):
        orch = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
        orch.st = SimpleNamespace(model_path="model.ssf")
        orch.ready = lambda: False
        with self.assertRaisesRegex(RuntimeError, "Chat must be ready"):
            orch.retry_orientation_failure()


class OrientationRetryRouteTests(unittest.TestCase):
    def test_current_backend_exposes_retry_route_and_returns_accepted_job(self):
        state = SimpleNamespace(orch=SimpleNamespace(
            retry_orientation_failure=lambda: {"ok": True, "task_id": "sl008"}))
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/api/orient/retry",
                data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(response.read(), b'{"ok": true, "task_id": "sl008"}')
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
