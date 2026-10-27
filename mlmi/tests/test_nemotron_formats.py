"""NVIDIA Nemotron conventions (2026-10-21): tool calls as <TOOLCALL>[...]</TOOLCALL>, and the template's
"detailed thinking off" switch kept when Sheep sends its own system message."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend.compat_server import parse_reply  # noqa: E402
from mlmi.backend.modes import Local  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
TEMPLATE = ("{%- if messages[0].role == 'system' and messages[0].content != '' -%}{{ messages[0].content }}"
            "{%- else -%}detailed thinking off{%- endif -%}|{% for m in messages[1:] %}{{ m.content }}{% endfor %}")


class ToolcallForm(unittest.TestCase):
    def test_nemotron_toolcall_list_is_read(self):
        text = 'Sure. <TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Nairobi"}}]</TOOLCALL>'
        content, _, calls = parse_reply(text, TOOLS)
        self.assertEqual([c["name"] for c in calls], ["get_weather"])
        self.assertIn("Nairobi", calls[0]["arguments"])
        self.assertEqual(content, "Sure.")

    def test_two_calls_in_one_list(self):
        text = '<TOOLCALL>[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]</TOOLCALL>'
        self.assertEqual([c["name"] for c in parse_reply(text, TOOLS)[2]], ["a", "b"])


class ThinkingSwitch(unittest.TestCase):
    def test_system_message_gets_thinking_off(self):
        local = Local("http://127.0.0.1:1")
        local._chat = {"template": TEMPLATE}
        out = local.render([{"role": "system", "content": "You are Sheep."}, {"role": "user", "content": "hi"}])
        self.assertTrue(out.startswith("detailed thinking off"), out)
        self.assertIn("You are Sheep.", out)

    def test_other_templates_untouched(self):
        local = Local("http://127.0.0.1:1")
        local._chat = {"template": "{{ messages[0].content }}"}
        self.assertEqual(local.render([{"role": "system", "content": "You are Sheep."}]), "You are Sheep.")


if __name__ == "__main__":
    unittest.main()
