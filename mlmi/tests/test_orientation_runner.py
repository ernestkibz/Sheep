import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mlmi.tests import orient_run


class OrientationRunnerTests(unittest.TestCase):
    def test_ready_reads_nested_orchestrator_lamp(self):
        payload = io.BytesIO(json.dumps({"orch": {"lamp": "green"}}).encode())
        with patch.object(orient_run.urllib.request, "urlopen", return_value=payload):
            self.assertEqual(orient_run.ready("http://local", timeout=0.1)["lamp"], "green")

    def test_ready_accepts_live_engine_after_clean_canaries_while_amber(self):
        payload = io.BytesIO(json.dumps({
            "lamp": "amber", "engine": {"up": True}, "canaries": [{"ok": True}] * 8,
        }).encode())
        with patch.object(orient_run.urllib.request, "urlopen", return_value=payload):
            self.assertEqual(orient_run.ready("http://local", timeout=0.1)["lamp"], "amber")

    def test_ready_accepts_engine_on_fixed_defaults(self):
        payload = io.BytesIO(json.dumps({"lamp": "off", "engine": {"up": True}, "canaries": []}).encode())
        with patch.object(orient_run.urllib.request, "urlopen", return_value=payload):
            self.assertEqual(orient_run.ready("http://local", timeout=0.1)["lamp"], "off")

    def test_connection_reset_before_headers_is_retried(self):
        response = io.BytesIO(b"")
        with patch.object(orient_run.urllib.request, "urlopen",
                          side_effect=[ConnectionResetError("reset"), response]) as urlopen, \
                patch.object(orient_run.time, "sleep"):
            self.assertEqual(orient_run.turn("http://local", [{"role": "user", "content": "hello"}], "", ""),
                             ("", [], []))
        self.assertEqual(urlopen.call_count, 2)

    def test_backend_busy_409_is_retried_until_it_accepts_a_turn(self):
        busy = orient_run.urllib.error.HTTPError("http://local", 409, "busy", {}, io.BytesIO())
        with patch.object(orient_run.urllib.request, "urlopen", side_effect=[busy, io.BytesIO(b"")]) as urlopen, \
                patch.object(orient_run.time, "sleep"):
            self.assertEqual(orient_run.turn("http://local", [], "", "", timeout=1), ("", [], []))
        self.assertEqual(urlopen.call_count, 2)

    def test_persistent_connection_reset_is_reported_as_infrastructure(self):
        with patch.object(orient_run.urllib.request, "urlopen", side_effect=ConnectionResetError("reset")), \
                patch.object(orient_run.time, "sleep"):
            with self.assertRaises(ConnectionResetError):
                orient_run.turn("http://local", [], "", "")

    def test_report_classifies_run_errors_separately_from_task_failures(self):
        self.assertTrue(orient_run.is_infrastructure_failure({"ok": False, "why": "run: ConnectionResetError"}))
        self.assertFalse(orient_run.is_infrastructure_failure({"ok": False, "why": "5 slides, 6 expected"}))
        self.assertFalse(orient_run.is_infrastructure_failure({"ok": True, "why": ""}))

    def test_infrastructure_rows_are_retried_when_resuming_but_task_fails_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orientation.jsonl"
            rows = [
                {"id": "sl000", "area": "slides", "ok": False, "why": "run: ConnectionResetError"},
                {"id": "sl001", "area": "slides", "ok": False, "why": "grade: missing title"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(orient_run.completed_ids(path), {"sl001"})
            self.assertEqual(orient_run.task_failure_ids(path), {"sl001"})
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"id": "sl000", "area": "slides", "ok": True, "why": ""}) + "\n")
            self.assertEqual(orient_run.completed_ids(path), {"sl000", "sl001"})
            self.assertEqual(orient_run.task_failure_ids(path), {"sl001"})

    def test_report_uses_latest_attempt_and_counts_unique_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orientation.jsonl"
            rows = [
                {"id": "sl000", "area": "slides", "ok": False, "why": "run: ConnectionResetError", "s": 0},
                {"id": "sl000", "area": "slides", "ok": True, "why": "", "s": 4},
                {"id": "sl001", "area": "slides", "ok": False, "why": "grade: missing title", "s": 3},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                orient_run.report(path)
            self.assertIn("orientation total 2/800; passed 1; task failures 1; infrastructure attempts 1; pending infrastructure retries 0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
