"""Concurrent conversation saves must use independent atomic temp files."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlmi.backend.server import save_json


class AtomicJsonSaveTests(unittest.TestCase):
    def test_concurrent_saves_leave_valid_json_and_no_shared_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "chat.json"

            def write(index):
                save_json(target, {"index": index, "payload": "x" * 4096})

            with ThreadPoolExecutor(max_workers=12) as pool:
                list(pool.map(write, range(80)))

            saved = json.loads(target.read_text(encoding="utf-8"))
            self.assertIn(saved["index"], range(80))
            self.assertEqual(saved["payload"], "x" * 4096)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
