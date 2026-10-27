"""Compatibility mode: replies split as sml-server splits them, the .compat.json the
quantizer writes, and how the app reads one."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlmi.backend import compat_server as C
from mlmi.backend import quantizer
from mlmi.backend.orchestrator import compat_facts, model_facts, originals_for

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


class ParseReplyTests(unittest.TestCase):
    def test_tool_call_tags_become_calls(self):
        content, think, calls = C.parse_reply(
            'Sure.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>', TOOLS)
        self.assertEqual(content, "Sure.")
        self.assertEqual(calls, [{"name": "get_weather", "arguments": '{"city": "Paris"}'}])

    def test_bare_json_only_for_an_offered_tool(self):
        _, _, calls = C.parse_reply('{"name": "get_weather", "parameters": {"city": "Rome"}}', TOOLS)
        self.assertEqual(calls[0]["name"], "get_weather")
        content, _, calls = C.parse_reply('{"name": "other", "parameters": {}}', TOOLS)
        self.assertEqual(calls, [])
        self.assertIn("other", content)

    def test_thinking_is_split_off_and_no_tools_means_prose(self):
        content, think, calls = C.parse_reply("<think>hmm</think>Hello <tool_call>{}</tool_call>", [])
        self.assertEqual((think, calls), ("hmm", []))
        self.assertTrue(content.startswith("Hello"))

    def test_tools_block_is_sml_servers_chatml_frame(self):
        b = C.tools_block(TOOLS)
        self.assertTrue(b.startswith("# Tools\n\n"))
        self.assertIn('"name": "get_weather"', b)
        self.assertTrue(b.endswith("</tool_call>"))


class CompatFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.src = Path(self.dir.name) / "originals" / "org--m"
        self.src.mkdir(parents=True)
        (self.src / "config.json").write_text(json.dumps({
            "model_type": "never_listed_lm", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
            "max_position_embeddings": 4096}), encoding="utf-8")
        (self.src / "model.safetensors").write_bytes(b"\0" * 1000)

    def test_quantizer_writes_a_compat_file_and_hands_it_on(self):
        job, got = quantizer.Job(), []
        out = Path(self.dir.name) / "models"
        out.mkdir()
        with mock.patch.object(quantizer, "OUT_DIR", out):
            job._compat(self.src, "org/m", {"model_type": "x"}, got.append)
        self.assertEqual(job.state, "done")
        self.assertTrue(got[0]["compat"])
        saved = json.loads(Path(got[0]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved, {"src": str(self.src), "repo": "org/m", "trust_remote_code": False})

    def test_remote_code_needs_the_yes(self):
        job = quantizer.Job()
        with mock.patch.object(quantizer, "OUT_DIR", Path(self.dir.name)):
            with self.assertRaises(RuntimeError):
                job._compat(self.src, "org/m", {"auto_map": {"AutoModel": "m.M"}}, None)

    def test_facts_come_from_the_original_config(self):
        p = Path(self.dir.name) / "org--m.compat.json"
        p.write_text(json.dumps({"src": str(self.src), "repo": "org/m"}), encoding="utf-8")
        f = model_facts(str(p))
        self.assertEqual(f, compat_facts(str(p)))
        self.assertTrue(f["compat"])
        self.assertEqual((f["arch"], f["n_layer"], f["n_ctx_train"], f["bytes"]), ("never_listed_lm", 2, 4096, 1000))

    def test_originals_are_found_only_when_still_on_disk(self):
        self.assertEqual(originals_for({"source_dir": str(self.src)}),
                         {"src": str(self.src), "trust_remote_code": False})
        self.assertIsNone(originals_for({"source_dir": str(self.src / "gone")}))
        self.assertIsNone(originals_for({}))


if __name__ == "__main__":
    unittest.main()
