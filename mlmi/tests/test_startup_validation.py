import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlmi.backend.checker import Checker, review_reply
from mlmi.backend.orchestrator import CANARY_TOKENS, Orchestrator, QUICK_CANARIES, device_label, run_canary
from mlmi.backend.orient import run_inprocess


class StartupValidationTests(unittest.TestCase):
    def test_quick_exam_is_small_and_covers_followup_and_tool_routing(self):
        names = {row[0] for row in QUICK_CANARIES}
        self.assertLessEqual(len(QUICK_CANARIES), 5)
        self.assertIn("follow-up", names)
        self.assertIn("tool call", names)

    def test_learned_checker_cannot_veto_a_deterministic_canary(self):
        class Local:
            def chat(self, *args, **kwargs):
                return ({"content": "Paris."}, {"finish": "stop", "tokens": {"completion_tokens": 2}})

        class CheckerThatWouldBlock:
            def check(self, *args, **kwargs):
                raise AssertionError("the learned Checker must not run inside canary acceptance")

        row = run_canary(Local(), "fact", [{"role": "user", "content": "capital?"}], None,
                         lambda text, calls: "paris" in text.lower(), checker=CheckerThatWouldBlock())
        self.assertTrue(row["ok"], row)

    def test_unchanged_model_and_plan_reuse_a_validation_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "candidate.ssf"
            model.write_bytes(b"first model payload")
            engine = Path(folder) / "engine"
            engine.write_bytes(b"runtime build one")
            orch = object.__new__(Orchestrator)
            orch.st = SimpleNamespace(engine=SimpleNamespace())
            orch.frame = "own"
            orch.sampling = None
            cache = {}
            orch.entry = lambda path=None: cache.get(str(path or model), {})
            orch.update_entry_for = lambda path, **fields: cache.setdefault(str(path), {}).update(fields)
            plan = {"device": "cpu", "device_name": "CPU", "ctx": 2048, "slots": 1,
                    "threads": 4, "gpu_layers": 0, "mid_layers": 0}
            with patch.dict("os.environ", {"MLMI_ENGINE_EXE": str(engine)}):
                first = orch.validation_signature(model, plan)
                self.assertEqual(first, orch.validation_signature(model, plan))
                model.write_bytes(b"second model payload")
                self.assertNotEqual(first, orch.validation_signature(model, plan))
                self.assertNotEqual(first, orch.validation_signature(model, {**plan, "threads": 2}))
                second = orch.validation_signature(model, plan)
                engine.write_bytes(b"runtime build two")
                self.assertNotEqual(second, orch.validation_signature(model, plan))

    def test_failed_results_are_completed_cache_entries(self):
        self.assertTrue(Orchestrator._finished_check({"signature": "same", "state": "failed"}, "same"))
        self.assertFalse(Orchestrator._finished_check({"signature": "other", "state": "failed"}, "same"))

    def test_background_resume_progress_uses_only_matching_completed_rows(self):
        stale = {"signature": "old", "state": "running", "progress": 1.0, "done": 0, "rows": []}
        self.assertEqual(Orchestrator._resume_progress(stale, "current", 8), (None, 0))
        partial = {"signature": "current", "state": "running", "progress": 1.0, "done": 2,
                   "rows": [{"ok": True}, {"ok": True}]}
        self.assertEqual(Orchestrator._resume_progress(partial, "current", 8), (0.25, 2))
        malformed = {"signature": "current", "state": "running", "done": "bad", "rows": [{"ok": True}]}
        self.assertEqual(Orchestrator._resume_progress(malformed, "current", 8), (None, 0))

    def test_multi_gpu_status_names_the_whole_assignment(self):
        devices = {
            "gpu": {"name": "NVIDIA RTX 3050"},
            "pool": [{"index": 1, "name": "AMD Radeon"}],
            "amd": {"name": "AMD Radeon"},
        }
        self.assertEqual(device_label(devices, {"device": "multi", "dml_split": [[1, 12]]}),
                         "NVIDIA RTX 3050 + AMD Radeon")
        self.assertEqual(device_label(devices, {"device": "hybrid"}), "NVIDIA RTX 3050 + CPU")

    def test_background_canaries_keep_the_full_answer_budget(self):
        orch = object.__new__(Orchestrator)
        orch.st = SimpleNamespace(model_path="model.ssf", local=object())
        orch.plan = {"device": "gpu"}
        orch.frame, orch.sampling, orch.egypt = "own", None, None
        orch.canaries = []
        orch.background_checks = {"canaries": {}}
        saved = {}
        orch.entry = lambda _path=None: saved
        orch.update_entry_for = lambda _path, **fields: saved.update(fields)
        orch.history = lambda *_args, **_kwargs: None
        orch.say = lambda *_args, **_kwargs: None
        orch._background_should_yield = lambda: False
        with patch("mlmi.backend.orchestrator.run_canary", return_value={"name": "long", "ok": True,
                                                                        "why": "", "ms": 20, "gen": 3}) as run:
            ok = orch.canary_suite("background", [("long", [], None, lambda *_: True)], signature="sig")
        self.assertTrue(ok)
        self.assertEqual(run.call_args.kwargs["max_tokens"], CANARY_TOKENS)

    def test_background_inference_does_not_wait_for_temperature(self):
        orch = object.__new__(Orchestrator)
        orch.st = SimpleNamespace(busy=threading.Lock())
        orch.interactive_requested = threading.Event()
        orch.gpu_jobs = threading.Condition()
        orch.gpu_job, orch.gpu_waiters = None, []
        self.assertIsNone(orch._background_yield_reason())
        orch.interactive_requested.set()
        self.assertEqual(orch._background_yield_reason(),
                         ("waiting for chat", "an interactive request is starting"))

    def test_background_wait_reason_updates_and_clears(self):
        orch = object.__new__(Orchestrator)
        orch.background_checks = {
            "canaries": {"state": "waiting for model load", "why": "stale"},
            "orientation": {"state": "waiting for GPU", "why": "old blocker"},
        }
        orch._set_background_wait_state(("waiting for chat", "interactive work is starting"))
        for row in orch.background_checks.values():
            self.assertEqual(row["state"], "waiting for chat")
            self.assertEqual(row["why"], "interactive work is starting")
        orch._set_background_wait_state(None)
        for row in orch.background_checks.values():
            self.assertEqual(row["state"], "queued")
            self.assertNotIn("why", row)

    def test_failed_orientation_migration_is_not_promoted_to_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "candidate.ssf"
            model.write_bytes(b"candidate")
            rows = [{"ok": i < 9} for i in range(10)]
            entry = {"facts": {"bytes": model.stat().st_size},
                     "plan": {"device": "cpu", "ctx": 2048, "slots": 1, "threads": 4,
                              "gpu_layers": 0, "mid_layers": 0},
                     "knows_sheep": {"validation_signature": "sig", "of": 10, "passed": 9, "rows": rows}}
            orch = object.__new__(Orchestrator)
            orch.frame = "own"
            orch.entry = lambda _path=None: entry
            orch.update_entry = lambda **fields: entry.update(fields)
            orch.update_entry_for = lambda _path, **fields: entry.update(fields)
            orch._adopt_legacy_checks(str(model), "sig", {"device": "cpu", "ctx": 2048, "slots": 1,
                                                            "threads": 4, "gpu_layers": 0, "mid_layers": 0})
            self.assertEqual(entry["orientation_validation"]["state"], "failed")

    def test_learned_checker_verdict_is_advisory_not_a_fault(self):
        checker = object.__new__(Checker)
        checker.judge = SimpleNamespace(judge=lambda *_: (False, "style disagreement"))
        faults, verdict = checker.check("question", "a complete answer")
        self.assertEqual(faults, [])
        self.assertEqual(verdict, "style disagreement")

    def test_readiness_reports_startup_and_background_jobs_separately(self):
        orch = object.__new__(Orchestrator)
        orch.st = SimpleNamespace(model_path="model.ssf", engine=SimpleNamespace(alive=lambda: True))
        orch.on = True
        orch.lamp = "green"
        orch.stage = "ready; background checks queued"
        orch.progress = 1.0
        orch.plan = {"device": "cpu"}
        orch.loading = threading.Event()
        orch.background_checks = {
            "canaries": {"state": "running", "progress": 0.5, "done": 4, "total": 8},
            "orientation": {"state": "queued", "progress": 0.0, "done": 0, "total": 10},
        }
        rows = orch.readiness_checks()
        self.assertEqual([row["id"] for row in rows], ["startup", "canaries", "orientation"])
        self.assertEqual(rows[0]["state"], "passed")
        self.assertEqual(rows[1]["state"], "running")

    def test_orientation_restores_completed_rows_and_saves_each_new_task(self):
        class Task:
            def __init__(self, task_id):
                self.id, self.area = task_id, "chat"
                self.turns = [(f"prompt {task_id}", "chat")]

            def setup(self, _root):
                pass

            def grade(self, _root, _result):
                return True, ""

        tasks = [Task("first"), Task("second")]
        old = {"id": "first", "area": "chat", "ok": True, "why": "", "s": 1, "turns": ["prompt first"]}
        saves = []
        state = SimpleNamespace(orch=SimpleNamespace(chat_kwargs=lambda: {}, write_mode=lambda _kind: True))

        def fake_run(_state, messages, _app, _path, emit, **_kwargs):
            emit({"type": "text", "text": "done"})

        with tempfile.TemporaryDirectory() as folder, patch("mlmi.backend.agent.run", side_effect=fake_run):
            rows = run_inprocess(state, tasks, Path(folder), existing_rows=[old],
                                 save_row=lambda i, row, all_rows: saves.append((i, row["id"], len(all_rows))),
                                 should_yield=lambda: True)
        self.assertEqual([row["id"] for row in rows], ["first", "second"])
        self.assertEqual(saves, [(1, "second", 2)])

    def test_orientation_stops_at_a_task_boundary_when_model_changes(self):
        class Task:
            def __init__(self, task_id):
                self.id, self.area = task_id, "chat"
                self.turns = [(f"prompt {task_id}", "chat")]

            def setup(self, _root):
                pass

            def grade(self, _root, _result):
                return True, ""

        tasks = [Task("first"), Task("second")]
        saved = []
        state = SimpleNamespace(orch=SimpleNamespace(chat_kwargs=lambda: {}, write_mode=lambda _kind: True))

        def fake_run(_state, _messages, _app, _path, emit, **_kwargs):
            emit({"type": "text", "text": "done"})

        with tempfile.TemporaryDirectory() as folder, patch("mlmi.backend.agent.run", side_effect=fake_run):
            rows = run_inprocess(state, tasks, Path(folder), save_row=lambda i, row, all_rows: saved.append(row["id"]),
                                 should_stop=lambda: len(saved) >= 1)
        self.assertEqual([row["id"] for row in rows], ["first"])
        self.assertEqual(saved, ["first"])

    def test_reply_integrity_review_does_not_call_the_learned_judge(self):
        class Judge:
            def judge(self, *_):
                raise AssertionError("learned review is advisory and must not gate this reply")

        checker = SimpleNamespace(judge=Judge())
        self.assertEqual(review_reply(checker, "question", "A valid response."), [])


if __name__ == "__main__":
    unittest.main()
