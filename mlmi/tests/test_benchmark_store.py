import json
import tempfile
import unittest
from pathlib import Path

from tooling.sml_benchmark_store import file_facts, record


class BenchmarkStoreTests(unittest.TestCase):
    def test_appends_jsonl_with_runtime_and_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            run_log = Path(directory) / "benchmarks.jsonl"
            model = Path(directory) / "model.ssf"
            model.write_bytes(b"fixture")

            saved = record(run_log, {
                "comparison": "sheep-vs-reference",
                "models": {"ssf": file_facts(model)},
                "summary": {"median_ratio": 1.2},
            })
            row = json.loads(saved.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(row["schema"], 1)
            self.assertEqual(row["comparison"], "sheep-vs-reference")
            self.assertEqual(row["models"]["ssf"]["bytes"], 7)
            self.assertEqual(row["summary"]["median_ratio"], 1.2)
            self.assertTrue(row["recorded_utc"])


if __name__ == "__main__":
    unittest.main()
