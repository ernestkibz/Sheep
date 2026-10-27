import threading
import types
import unittest
from unittest.mock import patch

from mlmi.backend.agent import Session
from mlmi.backend.tools import media
from mlmi.backend.workspace import ToolError


class SafeStopTests(unittest.TestCase):
    def test_session_invokes_registered_safe_cancel_and_rejects_next_action(self):
        session = Session()
        session.reset_stop()
        cancelled = []
        self.assertTrue(session.begin_action("make_video"))
        session.set_cancel_hook(lambda: cancelled.append(True))

        self.assertEqual(session.request_stop(), "make_video")
        self.assertTrue(session.stop.is_set())
        self.assertEqual(cancelled, [True])
        self.assertFalse(session.begin_action("studio_render"))

    def test_media_cancel_does_not_fallback_after_device_error(self):
        worker = object.__new__(media._Worker)
        worker.what = "video"
        worker.proc = types.SimpleNamespace(poll=lambda: None)
        worker.dev = "cuda"
        worker.preference = media.CONFIG.get("media_device", "auto")
        worker.lock = threading.Lock()
        worker.used = 0.0
        cancelled = threading.Event()

        def fail_as_cancelled(*args, **kwargs):
            cancelled.set()
            return {"ok": False, "error": "CUDA out of memory"}

        with patch.object(worker, "_start"), patch.object(worker, "_send", side_effect=fail_as_cancelled), \
                patch.object(worker, "_retry_next", side_effect=AssertionError("cancel must not retry")):
            with self.assertRaisesRegex(ToolError, "cancelled by the user"):
                worker.make({}, cancel_event=cancelled)


if __name__ == "__main__":
    unittest.main()
