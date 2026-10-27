"""Nemotron's second opinion (hackathon B6): in Sheep + Nebius mode, when the local model writes a deck,
Nemotron checks the draft against the request before Sheep builds the file, and the local model rewrites
it once with the fixes."""
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import agent, formats as F  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

DRAFT = "# Water\nSaving water\n\n## Why\n- It is scarce\n- It costs money\n\n## How\n- Fix leaks\n- Short showers\n"
FIXED = DRAFT + "\n## At school\n- Turn taps off\n- Report leaks\n"


class Orch:
    lamp, stage, progress = "green", "ready", 1.0

    def ready(self):
        return True

    def holding(self):
        return False

    def chat_kwargs(self):
        return {}

    def write_mode(self, kind):
        return True

    def after_turn(self, info):
        pass


class Local:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, **kw):
        self.calls.append(messages[0]["content"])
        reviewed = "A reviewer read your first draft" in messages[0]["content"]
        return {"role": "assistant", "content": FIXED if reviewed else DRAFT}, {"route": "local"}


class Cloud:
    model = "nvidia/nemotron-3-nano-30b-a3b"
    provider = "Nebius via OpenRouter (pinned)"

    def __init__(self, verdict):
        self.verdict, self.asked = verdict, []

    def ready(self):
        return True

    def chat(self, messages, tools=None, max_tokens=0, why=""):
        self.asked.append(why)
        if why == "check the local model's draft":
            return {"content": json.dumps(self.verdict)}, {"route": "cloud", "ms": 1}
        return {"content": "no plan"}, {"route": "cloud", "ms": 1}      # the planner gives no calls


def run(verdict, text="add three slides about saving water to water.sdeck"):
    root = Path(tempfile.mkdtemp(prefix="mlmi-review-"))
    d = F.new("presentations")
    d["slides"].append({"layout": "title", "title": "Water", "subtitle": "", "bullets": [], "notes": "", "items": []})
    F.save(root / "water.sdeck", d)
    local, cloud = Local(), Cloud(verdict)
    st = types.SimpleNamespace(session=agent.Session(), ws=Workspace(root), model_card=lambda: {"tier": 2}, mode=1,
                               permission="auto", local=local, cloud=cloud, orch=Orch(), busy=threading.Lock())
    ev = []
    agent.run(st, [{"role": "user", "content": text}], emit=ev.append, auto_confirm="yes")
    return ev, local, cloud, F.load(root / "water.sdeck")


class SecondOpinion(unittest.TestCase):
    def test_fixes_are_written_in_before_the_file_is_made(self):
        ev, local, cloud, deck = run({"ok": False, "fixes": ["three slides were asked; add a third one"]})
        review = [e for e in ev if e["type"] == "review"]
        self.assertTrue(review, [e["type"] for e in ev])
        self.assertTrue(review[0]["fixed"])
        self.assertIn("check the local model's draft", cloud.asked)
        self.assertIn("At school", [s["title"] for s in deck["slides"]])

    def test_a_clean_draft_is_left_alone(self):
        ev, local, cloud, deck = run({"ok": True})
        review = [e for e in ev if e["type"] == "review"]
        self.assertTrue(review and review[0]["ok"] and not review[0]["fixed"])
        self.assertFalse(any("A reviewer read your first draft" in c for c in local.calls))


if __name__ == "__main__":
    unittest.main()
