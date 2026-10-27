import unittest

from pathlib import Path
import tempfile

from mlmi.backend.teach import (TRAINING_TRIAL_EXAMPLES, TRAINING_TRIAL_STEPS,
                               reusable_baseline, stratified_trial)
from mlmi.backend.teach_worker import training_step_count


class ShortCheckerTrainingTests(unittest.TestCase):
    def test_trial_is_small_deterministic_and_keeps_each_lesson_type(self):
        rows = [{"kind": kind, "id": f"{kind}-{i}"}
                for kind in ("arithmetic", "facts", "translation", "open_requests", "faults")
                for i in range(40)]
        first = stratified_trial(rows, TRAINING_TRIAL_EXAMPLES)
        second = stratified_trial(rows, TRAINING_TRIAL_EXAMPLES)
        self.assertEqual(len(first), 64)
        self.assertEqual(first, second)
        self.assertEqual({row["kind"] for row in first},
                         {"arithmetic", "facts", "translation", "open_requests", "faults"})
        counts = {kind: sum(row["kind"] == kind for row in first) for kind in {row["kind"] for row in first}}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_product_run_caps_training_at_eight_steps_without_changing_effective_batch(self):
        self.assertEqual(TRAINING_TRIAL_STEPS, 8)
        self.assertEqual(training_step_count(64, epochs=1, batch=8, max_steps=8), 8)
        self.assertEqual(training_step_count(1600, epochs=2, batch=8, max_steps=8), 8)
        self.assertEqual(training_step_count(64, epochs=1, batch=8), 8)

    def test_baseline_exam_is_reused_only_for_matching_file_and_exam_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "checker.ssf"
            model.write_bytes(b"model")
            saved = {"path": str(model), "model_revision": "sha256:model", "exam_revision": 4,
                     "exam_ok": True, "right": 36, "of": 36, "accuracy": 1.0}
            cached = reusable_baseline(saved, model, "sha256:model", 4, 36)
            self.assertTrue(cached["cached"])
            self.assertEqual(cached["right"], 36)
            self.assertIsNone(reusable_baseline(saved, model, "sha256:changed", 4, 36))
            self.assertIsNone(reusable_baseline(saved, model, "sha256:model", 5, 36))
            self.assertIsNone(reusable_baseline({**saved, "exam_ok": False}, model, "sha256:model", 4, 36))


if __name__ == "__main__":
    unittest.main()
