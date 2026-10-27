"""Research that makes files (hackathon B7): a deck about something current is written from a web search
(Tavily when its key is set) and ends with a Sources slide of the links used."""
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import agent, formats as F  # noqa: E402
from mlmi.backend.tools import web  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

DECK = ("# Phones in 2026\nWhat is new\n\n## Cameras\n- Bigger sensors\n- Better zoom\n\n"
        "## Batteries\n- Two-day life\n- Faster charging\n\n## Prices\n- Mid-range got cheaper\n- Flagships held\n")
RESULTS = [{"title": "Phone review roundup", "url": "https://example.com/phones", "snippet": "Cameras grew."},
           {"title": "Battery tests", "url": "https://example.org/battery", "snippet": "Two-day phones."}]


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
        self.systems = []

    def chat(self, messages, tools=None, **kw):
        self.systems.append(messages[0]["content"])
        return {"role": "assistant", "content": DECK}, {"route": "local"}


class NoCloud:
    model, provider, sent = "", "", []

    def ready(self):
        return False


class ResearchFiles(unittest.TestCase):
    def test_current_deck_is_written_from_sources_and_lists_them(self):
        root = Path(tempfile.mkdtemp(prefix="mlmi-research-"))
        local = Local()
        st = types.SimpleNamespace(session=agent.Session(), ws=Workspace(root), model_card=lambda: {"tier": 2}, mode=2,
                                   permission="auto", local=local, cloud=NoCloud(), orch=Orch(), busy=threading.Lock())
        ev = []
        was = web.STATE["on"]
        web.STATE["on"] = True
        try:
            with patch.object(web, "search", return_value=RESULTS), \
                    patch.object(web, "fetch", return_value=("", "Cameras grew a lot this year.")):
                agent.run(st, [{"role": "user", "content": "make a presentation about the latest phones"}],
                          emit=ev.append, auto_confirm="yes")
        finally:
            web.STATE["on"] = was
        self.assertTrue(any("https://example.com/phones" in s for s in local.systems))
        decks = list(root.glob("*.sdeck"))
        self.assertTrue(decks, [e for e in ev if e["type"] in ("text", "error")])
        slides = F.load(decks[0])["slides"]
        self.assertEqual(slides[-1]["title"], "Sources")
        self.assertTrue(any("https://example.org/battery" in b for b in slides[-1]["bullets"]))
        self.assertTrue(any(e["type"] == "sources" for e in ev))

    def test_sources_markdown(self):
        md = agent.sources_md(RESULTS)
        self.assertIn("## Sources", md)
        self.assertIn("- Phone review roundup: https://example.com/phones", md)


if __name__ == "__main__":
    unittest.main()
