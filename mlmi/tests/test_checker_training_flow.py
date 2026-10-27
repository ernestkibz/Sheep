import tempfile
import time
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlmi.backend import checker as checker_mod, teach as teach_mod


class CheckerTrainingFlowTests(unittest.TestCase):
    def test_eight_step_trial_runs_full_exam_and_keeps_incumbent_on_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.ssf"
            candidate.write_bytes(b"installed candidate")
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({"num_hidden_layers": 1}), encoding="utf-8")
            (source / "weights.bin").write_bytes(b"weights")
            checker = SimpleNamespace(cfg={}, trial={})
            exam = {"right": 36, "of": 36, "frame": "own", "exam_ok": True, "accuracy": 1.0}
            completed = []
            commands = []

            class FakeProcess:
                def __init__(self, command, **_kwargs):
                    commands.append(command)
                    if command[0] == teach_mod.sys.executable:
                        out = Path(command[command.index("--out") + 1])
                        out.mkdir(parents=True, exist_ok=True)
                        (out / "config.json").write_text(json.dumps({"num_hidden_layers": 1}), encoding="utf-8")
                        self.stdout = [
                            json.dumps({"step": 8, "of": 8, "chunk_step": 8, "s": 1, "loss": 0.1}),
                            json.dumps({"ok": True, "out": str(out), "timings": {"train_seconds": 1.0,
                                                                                   "merge_seconds": 1.0}}),
                        ]
                    else:
                        Path(command[-1]).write_bytes(b"quantized candidate")
                        self.stdout = ["layer 1/1"]

                def wait(self):
                    return 0

            lessons = [{"kind": kind, "messages": [{"role": "user", "content": "Q"},
                                                       {"role": "assistant", "content": "A"}]}
                       for kind in ("arithmetic", "facts", "translation", "open_requests", "faults")
                       for _ in range(20)]

            def recipe(_merged, part, *_args, **_kwargs):
                return ["fake-quantizer", str(part)]

            trainer = teach_mod.Trainer(checker, None, str(candidate), source,
                                        lambda chosen, report: completed.append((chosen, report)),
                                        training_mode="cpu")
            with patch.object(teach_mod, "HOME", root / "sheep-home"), \
                    patch.object(teach_mod, "lessons", return_value=lessons), \
                    patch("mlmi.backend.quantizer.recipe", side_effect=recipe), \
                    patch.object(teach_mod.subprocess, "Popen", side_effect=FakeProcess), \
                    patch("mlmi.backend.training_policy.nvidia_thermal", return_value={"available": False}), \
                    patch("mlmi.backend.orchestrator.probe", return_value={"ram": {"available": 16 * 2**30},
                                                                             "gpu": None}), \
                    patch.object(checker_mod, "configure", side_effect=lambda *_args: dict(exam)):
                trainer._run()

            self.assertEqual(trainer.state["state"], "installed")
            self.assertFalse(trainer.state["trained_better"])
            self.assertEqual(completed[0][0], str(candidate))
            self.assertEqual(len(list((Path(trainer.state["work"]) / "lessons.jsonl").open(encoding="utf-8"))), 64)
            worker_command = commands[0]
            self.assertEqual(worker_command[worker_command.index("--max-steps") + 1], "8")
            self.assertEqual(worker_command[worker_command.index("--checkpoint-every") + 1], "4")
            self.assertEqual(completed[0][1]["right"], 36)

    def test_candidate_must_pass_exam_before_optional_training_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.ssf"
            candidate.write_bytes(b"placeholder model for lifecycle test")
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text("{}", encoding="utf-8")
            checker = checker_mod.Checker({}, object, object, lambda: None)
            failed_exam = {"right": 20, "of": 36, "frame": "own", "exam_ok": False}
            with patch.object(checker_mod.Checker, "source", return_value=str(source)), \
                    patch.object(checker_mod, "configure", return_value=failed_exam), \
                    patch.object(teach_mod, "HOME", root / "sheep-home"):
                checker.install(str(candidate), train=True, training_mode="cpu")
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and checker.trainer.state["state"] not in ("rolled back", "failed"):
                    time.sleep(0.01)
            self.assertEqual(checker.trainer.state["state"], "rolled back")
            self.assertFalse(checker.trainer.proc)
            self.assertIsNone(checker.judge)
            self.assertIn("failed its initial", checker.trainer.state["why"])

    def test_settings_flow_checks_first_and_reuses_saved_training_placement(self):
        source = (Path(__file__).resolve().parents[1] / "desktop" / "ui" / "views.js").read_text(encoding="utf-8")
        self.assertIn('{ path, train: false }', source)
        self.assertIn('$("#cTrain", this.el)', source)
        self.assertIn('id="hCheckerTraining"', source)
        self.assertIn('checker_training_mode: training.value', source)
        self.assertIn('const savedMode = hw.prefs?.checker_training_mode || "auto"', source)
        self.assertIn('option value="gpu_bounded"', source)
        training_start = source.index("const trainingChoice = async")
        training_end = source.index("this.trainingChoice = trainingChoice", training_start)
        self.assertNotIn('className = "modal"', source[training_start:training_end])
        self.assertNotIn("1,600 lessons", source)
        self.assertNotIn("30 to 60 minutes", source)


if __name__ == "__main__":
    unittest.main()
