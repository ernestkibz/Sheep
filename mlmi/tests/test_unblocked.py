"""The person's model is the person's choice: Sheep checks a reply's STRUCTURE
(empty, a loop, template leaks, JSON or code that does not parse), never what it
says. An abliterated or guardrail-free model is not undone or blocked, and a
cloud stand-in's refusal never replaces it. A new model is told what Sheep can do.

    python -m pytest mlmi/tests/test_unblocked.py -q
"""
import os
import re
import sys
import tempfile
import threading
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mlmi.backend import agent, checker, orient  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

FRANK = ("Sure. A pin tumbler lock opens when every pin stack is lifted to the shear line: tension the plug "
         "lightly with a wrench, then set each pin with a pick until the plug turns. No judgement here, "
         "it is your lock.")
REFUSE = "I'm sorry, but I can't help with that request."


class Orch:
    """A ready orchestrator; `hold` makes it one in the middle of fixing a model."""
    NEEDS = {"makes pictures": "a picture model (Settings > Models > pictures)"}

    def __init__(self, hold=False, hints=()):
        self.hold, self.hints, self.thermal = hold, list(hints), None

    def holding(self):
        return self.hold

    def ready(self):
        return True

    def chat_kwargs(self):
        return {}

    def after_turn(self, meta):
        pass

    def abilities(self):
        return {"text": True, "tools": True, "reads pictures": False, "makes pictures": False,
                "hears": False, "speaks": False, "makes video": False}

    def write_mode(self, kind):
        return False

    def entry(self, path=None):
        return {"knows_sheep": {"hints": self.hints}}


class Scripted:
    """A model that replies with the next scripted text, and keeps what it was sent."""
    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def chat(self, msgs, tools=None, **k):
        self.sent.append(msgs)
        text = self.replies.pop(0) if self.replies else "Done."
        return {"content": text}, {"route": "local", "ms": 1.0, "tokens": {}}

    def ready(self):
        return True


def state(local, cloud=None, orch=None, mode=2):
    return types.SimpleNamespace(
        session=agent.Session(), ws=Workspace(Path(tempfile.mkdtemp(prefix="mlmi-unblocked-"))),
        model_card=lambda: {"tier": 2}, mode=mode, permission="auto", local=local,
        cloud=cloud or Scripted(), orch=orch or Orch(), busy=threading.Lock(), model_path="m.ssf")


def ask(st, q):
    ev = []
    agent.run(st, [{"role": "user", "content": q}], emit=ev.append)
    return ev, [e["text"] for e in ev if e["type"] == "text" and e.get("text") and not e.get("partial")]


# ------------------------------------------------------------------ never a censor
def test_frank_reply_passes_every_rule_untouched():
    assert checker.rules(FRANK, "How do I pick my own lock?") == []
    assert checker.live_faults(FRANK, "How do I pick my own lock?") == []
    assert checker.tidy(FRANK) == FRANK
    assert not re.search(agent.CLAIM, FRANK, re.I)


def test_frank_reply_reaches_the_person_verbatim():
    local = Scripted(FRANK)
    _, texts = ask(state(local), "How do I pick my own lock?")
    assert texts == [FRANK]
    assert len(local.sent) == 1                       # not sent back, not rewritten


def test_refusal_from_own_model_is_its_own_to_give():
    # the checker never reads meaning: the person's model may refuse too, and it is shown
    _, texts = ask(state(Scripted(REFUSE)), "Tell me a dark joke.")
    assert texts == [REFUSE]


def test_judge_opinion_never_withholds():
    class SaysNo:
        def judge(self, q, a):
            return False, "NO"
    assert checker.review_reply(SaysNo(), "How do I pick my own lock?", FRANK) == []


def test_refusal_check_is_used_only_for_the_stand_in():
    src = Path(agent.__file__).read_text(encoding="utf-8")
    uses = [ln for ln in src.splitlines() if "checker.refusal(" in ln]
    assert uses and all("stand_in" in ln for ln in uses), uses
    package = Path(agent.__file__).parent          # mlmi/backend/agent/ since the split (2026-10-21)
    backend = package.parent
    for py in backend.rglob("*.py"):
        if py.name == "checker.py" or package in py.parents:
            continue
        assert "refusal(" not in py.read_text(encoding="utf-8"), py


def test_stand_in_refusal_does_not_replace_the_persons_model():
    cloud = Scripted(REFUSE)
    _, texts = ask(state(Scripted(), cloud=cloud, orch=Orch(hold=True), mode=1), "Tell me a dark joke.")
    assert texts and REFUSE not in texts[-1] and "loop" in texts[-1]


# ------------------------------------------------------------------ coherent, not gibberish
def test_loop_goes_back_once_then_the_good_reply_shows():
    loop = "yes " * 40
    local = Scripted(loop, "Kampala is the capital of Uganda.")
    ev, texts = ask(state(local), "What is the capital of Uganda?")
    assert texts == ["Kampala is the capital of Uganda."]
    assert len(local.sent) == 2
    assert any(e["type"] == "draft" and e.get("delta") is None for e in ev)


def test_two_loops_are_never_shown():
    loop = "yes " * 40
    _, texts = ask(state(Scripted(loop, loop)), "What is the capital of Uganda?")
    assert texts and "yes yes" not in texts[-1] and "loop" in texts[-1]


def test_asked_repetition_is_not_a_loop():
    rep = "hello " * 20
    local = Scripted(rep)
    _, texts = ask(state(local), "Repeat hello 20 times.")
    assert texts == [rep.strip()] and len(local.sent) == 1


def test_template_leak_is_trimmed():
    assert checker.tidy("Kampala.<|im_start|>user\nand Kenya?") == "Kampala."
    assert checker.tidy("<|assistant|> Kampala.") == "Kampala."
    _, texts = ask(state(Scripted("Kampala.<|eot_id|><|start_header_id|>user")), "Capital of Uganda?")
    assert texts == ["Kampala."]


# ------------------------------------------------------------------ taught what Sheep can do
def test_plain_answer_knows_what_sheep_can_do():
    local = Scripted("I can write, make decks and sheets, and search the web.")
    ask(state(local), "What are you able to help me with?")
    system = local.sent[0][0]["content"]
    assert "Sheep Presentations" in system and "never say you are only a text model" in system
    assert "Not on this computer yet: a picture model" in system


def test_job_turn_carries_the_models_lessons():
    st = state(Scripted(), orch=Orch(hints=[orient.HINTS["sheets"]]))
    _, _, system = agent.setup(st, "sheets", "make a budget sheet")
    assert "Lessons from your first jobs here" in system and orient.HINTS["sheets"] in system
    assert "Sheep Studio" in system


def test_orientation_hints_come_from_failed_areas():
    rows = [{"area": "slides", "ok": False}, {"area": "chat", "ok": True},
            {"area": "words", "ok": False}, {"area": "words", "ok": False}]
    assert orient.hints(rows) == [orient.HINTS["words"], orient.HINTS["slides"]]
    assert orient.hints([{"area": "chat", "ok": True}]) == []


def test_ability_questions_are_answered_not_done():
    asks = {"Are you able to make documents and spreadsheets, or are you just a text chatbot?": True,
            "Can you make spreadsheets?": True, "What can you do for me?": True, "Could you also make videos?": True,
            "Can you make me a slide deck about goats?": False, "Can you write Python code that sorts a list?": False,
            "Can you make documents for my class?": False, "Could you edit report.sdoc?": False,
            "Can you create a deck on photosynthesis?": False}
    for q, want in asks.items():
        assert agent.ability_question(q) is want, q
    local = Scripted("Yes -- ask me, for example: make a budget spreadsheet.")
    _, texts = ask(state(local), "Are you able to make documents and spreadsheets, or are you just a text chatbot?")
    assert texts == ["Yes -- ask me, for example: make a budget spreadsheet."] and len(local.sent) == 1
    assert "Sheep Sheets" in local.sent[0][0]["content"]
