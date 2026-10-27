"""The app checks fail only for real (2026-10-20): five headed tips are five items when the ask names no
shape, a media job passes when Sheep reaches the media tool, and the checks' view of the orchestrator hides
nothing the real one has."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import formats as F, orient as O  # noqa: E402


def five_tips(root):
    d = F.new("words")
    d["blocks"].append({"type": "heading", "level": 0, "text": "Five Tips"})
    for i in range(5):
        d["blocks"].append({"type": "heading", "level": 1, "text": f"Tip {i + 1}: drink water"})
        d["blocks"].append({"type": "paragraph", "text": f"Why tip {i + 1} helps, in a sentence."})
    F.save(Path(root) / "tips.sdoc", d)


class DocShape(unittest.TestCase):
    def test_headed_tips_pass_any_shape_and_fail_list(self):
        with tempfile.TemporaryDirectory() as td:
            five_tips(td)
            self.assertEqual(O.doc_ok(Path(td), "tips.sdoc", items=5, min_blocks=5, shape="any"), (True, ""))
            ok, why = O.doc_ok(Path(td), "tips.sdoc", items=5, min_blocks=5)
            self.assertFalse(ok)
            self.assertIn("0 list items", why)

    def test_fu004_grades_with_any_shape(self):
        task = next(t for t in O.quick() if t.id == "fu004")
        with tempfile.TemporaryDirectory() as td:
            five_tips(td)
            ok, _ = task.grade(td, {"texts": ["done"], "files": ["tips.sdoc"], "steps": []})
            self.assertTrue(ok)


class MediaRehearsal(unittest.TestCase):
    def test_reaching_the_picture_tool_passes(self):
        task = next(t for t in O.quick() if t.area == "picture")
        with tempfile.TemporaryDirectory() as td:
            ok, why = task.grade(td, {"texts": ["made"], "files": [], "steps": [["make_image", True, ""]]})
            self.assertTrue(ok, why)
            ok, _ = task.grade(td, {"texts": ["made"], "files": [], "steps": [["make_image", False, "err"]]})
            self.assertFalse(ok)


class QuickList(unittest.TestCase):
    def test_file_first_task_is_in_the_checks(self):
        ids = [t.id for t in O.quick()]
        self.assertEqual(len(ids), 14)
        fu8 = next(t for t in O.quick() if t.id == "fu008")
        self.assertEqual(fu8.turns[1][0], "pptx")


class OldFolders(unittest.TestCase):
    def test_only_other_signature_folders_go(self):
        from mlmi.backend.orchestrator import drop_old_checks
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for n in ("0123456789abcdef", "fedcba9876543210", "flat-layout"):
                (base / n).mkdir()
            drop_old_checks(base / "0123456789abcdef")
            self.assertEqual(sorted(p.name for p in base.iterdir()), ["0123456789abcdef", "flat-layout"])


class RequeueAfterWait(unittest.TestCase):
    def _orch(self, state):
        import threading
        from unittest.mock import Mock
        from mlmi.backend import orchestrator
        o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
        o.st = SimpleNamespace(model_path="m.ssf", engine=SimpleNamespace(alive=lambda: True))
        o.background_checks = {"orientation": {"state": state}}
        o.lock = threading.Lock()
        o._background_should_yield = lambda: False
        o.resume_background_checks = Mock()
        return orchestrator, o

    def test_a_waiting_check_comes_back_by_itself(self):
        from unittest.mock import patch
        orchestrator, o = self._orch("waiting for temperature")
        done = []
        with patch.object(orchestrator.time, "sleep", lambda s: None), \
                patch.object(orchestrator.threading, "Thread",
                             lambda target, name, daemon: SimpleNamespace(start=lambda: done.append(target()))):
            o._requeue_when_clear("m.ssf")
        o.resume_background_checks.assert_called_once()

    def test_a_failed_or_blocked_check_is_not_requeued(self):
        orchestrator, o = self._orch("blocked by canary failures")
        o._requeue_when_clear("m.ssf")
        o.resume_background_checks.assert_not_called()


class OrchestratorView(unittest.TestCase):
    def test_view_delegates_lazily(self):
        from mlmi.backend import agent
        seen = {}

        def fake_run(view, messages, app, path, emit, auto_confirm=None, cache=True):
            seen["has"] = hasattr(view.orch, "abilities")
            seen["kw"] = view.orch.chat_kwargs()
            seen["rehearsal"] = view.rehearsal
            seen["ready"] = view.orch.ready()

        st = SimpleNamespace(orch=SimpleNamespace(chat_kwargs=lambda: {"frame": "own"}))
        task = O.Task("x1", "chat", [("hi", "")], lambda r, s: (True, ""))
        real = agent.run
        agent.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as td:
                rows = O.run_inprocess(st, [task], td, sampling={"temperature": 0.6})
                self.assertTrue((Path(td) / "transcript.jsonl").exists())
        finally:
            agent.run = real
        self.assertFalse(seen["has"])
        self.assertEqual(seen["kw"], {"frame": "own", "temperature": 0.6})
        self.assertTrue(seen["rehearsal"] and seen["ready"])
        self.assertTrue(rows[0]["ok"])


if __name__ == "__main__":
    unittest.main()
