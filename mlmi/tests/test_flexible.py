"""A turn belongs to the conversation (Ernest, 2026-10-30): "can you just do the thinking" after Nemotron's
questions was refused as "this turn did not ask to create or render an artifact", on any model and for any
tool. A request stands across hand-backs, answers and retries; the model chooses the tools unless the person
is only talking; a job may make what was asked for and change what it read.

    python -m pytest mlmi/tests/test_flexible.py -q
"""
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mlmi.backend import agent, shepherd  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

PLAN = "Sure. To get started, could you let me know which sections you want?"
ASKED = "can you make an equivalent of this but for the 90 day work plan for tech team"


def chat(*turns, files=None):
    """user, assistant, user, ...; files: what the last assistant turn made."""
    out = [{"role": "assistant" if i % 2 else "user", "content": t} for i, t in enumerate(turns)]
    if files:
        out[-2]["files"] = files
    return out


class Carried(unittest.TestCase):
    def test_his_chat(self):
        m = chat("can you view images?", "I can edit them.", ASKED, PLAN,
                 "can you just do the thinking i dont feel like answerign all your quaestions")
        m[2]["attachments"] = ["attachments/plan.docx"]
        self.assertEqual(shepherd.carried(m), {"text": ASKED, "attachments": ["attachments/plan.docx"]})

    def test_every_kind_of_work_hands_back(self):
        for req in ("draw me a picture of a dog", "make a video of a dog running on a beach", "write a report about tea",
                    "make a budget for the wedding", "read this aloud in a voice: hello there", "create a deck on water",
                    "edit the intro of my studio project", "send an email to Sam about the invoice"):
            for back in ("just do it", "you decide", "go ahead", "stop asking and do it", "no questions, just do it",
                         "use your judgement", "up to you", "whatever you think", "surprise me"):
                held = shepherd.carried(chat(req, "Which style would you like?", back), asks=agent.wants_tools)
                self.assertEqual((held or {}).get("text"), req, (req, back))

    def test_an_answer_joins_the_request(self):
        self.assertEqual(shepherd.carried(chat("make a budget", "How many rows do you want?", "ten rows"))["text"],
                         "make a budget — ten rows")
        self.assertEqual(shepherd.carried(chat("draw a dog", "What style?", "watercolour"))["text"], "draw a dog — watercolour")

    def test_retry_after_a_failure_or_a_result(self):
        failed = "I could not complete this: the last three steps failed."
        self.assertEqual(shepherd.carried(chat("write a report on tea", failed, "try again"))["text"], "write a report on tea")
        self.assertEqual(shepherd.carried(chat("draw a dog", "Done.", "another one", files=["dog.png"]))["text"], "draw a dog")

    def test_a_change_to_a_picture_or_video(self):
        m = chat("draw a dog", "Done.", "with a red hat", files=["dog.png"])
        self.assertEqual(shepherd.carried(m)["text"], "draw a dog — with a red hat")
        self.assertIsNone(shepherd.carried(m, works=True))     # "make it black and white" is an edit, not a redo

    def test_what_stays_alone(self):
        ask = "Which style would you like?"
        for m in (chat("draw a dog", "Done.", "thanks", files=["dog.png"]),
                  chat("draw a dog", "Done.", "what model made it?", files=["dog.png"]),
                  chat("draw a dog", ask, "what styles are there?"),
                  chat("draw a dog", ask, "no thanks"),
                  chat("what is the capital of Uganda?", "Kampala.", "do it"),
                  chat("hello", "Hello! How can I help?", "just do it"),
                  chat("write a poem", "Here it is: ...", "continue"),
                  chat("draw a dog", ask, "just tell me in the chat")):
            self.assertIsNone(shepherd.carried(m), m[-1]["content"])

    def test_a_new_job_is_a_new_job(self):
        self.assertIsNone(shepherd.carried(chat("draw a dog", "What style?", "make a deck about tea"), lambda t: True))


class Pace(unittest.TestCase):
    """A big job goes a part at a time, unless the person says they can wait."""
    def test_how_much_at_once(self):
        for t in ("make a 90 day work plan tracker", "a detailed report on tea", "make a dashboard like the sales one",
                  "make a 20 slides deck on water"):
            self.assertEqual(shepherd.batch_of(t), "batches", t)
        for t in ("make a tracker, i can wait", "do the whole plan all at once", "take your time and write the report"):
            self.assertEqual(shepherd.batch_of(t), "whole", t)
        for t in ("write a note to Sam", "make a deck on tea", "draw a dog"):
            self.assertEqual(shepherd.batch_of(t), "", t)
        self.assertEqual(shepherd.batch_of("make a summary", 2), "batches")

    def test_yes_goes_on_to_the_next_part(self):
        for now in ("yes", "continue", "next", "go on", "ok", "ya"):
            held = shepherd.carried(chat("make a 90 day tracker", "Sheet one is done. Shall I do the next sheet?", now,
                                         files=["plan.ssheet"]))
            self.assertIn("make a 90 day tracker", held["text"], now)
            self.assertIn("plan.ssheet", held["text"], now)
        self.assertIsNone(shepherd.carried(chat("make a 90 day tracker", "All done.", "yes", files=["plan.ssheet"])))
        self.assertTrue(shepherd.carried(chat("make a 90 day tracker", "Rows done. Next: the actions section.", "yes", files=["plan.ssheet"])))

    def test_the_model_is_told_and_may_think_when_asked(self):
        for text, line, kw in (("make a 90 day work plan tracker", agent.BATCHES, {"wait": 180}),
                               ("make a 90 day work plan tracker, i can wait", agent.WHOLE, {"wait": 600, "effort": "high"})):
            seen = {}

            class Note(Calls):
                def chat(self, msgs, tools=None, **k):
                    seen.update(k, msgs=msgs)
                    return super().chat(msgs, tools, **k)
            Whole().run_it(chat(text), True, Note())
            self.assertIn(line, " ".join(str(m.get("content")) for m in seen["msgs"]))
            self.assertEqual({k: seen[k] for k in kw}, kw)


class Formulas(unittest.TestCase):
    def test_what_real_workbooks_use(self):
        from mlmi.backend.tools import formula as F

        class Book:
            sheetnames = ["In put", "S"]
            d = {("In put", "A1"): "2026-10-01", ("In put", "B1"): 90, ("S", "A1"): "x", ("S", "A2"): "y", ("S", "B1"): 3, ("S", "B2"): 7}

            def get(self, s, c):
                return self.d.get((s, c))
        e = F.Evaluator(Book())
        for src, want in (("'In put'!A1+'In put'!B1-1", 46385.0), ('TEXT(\'In put\'!A1,"mmm d, yyyy")', "Oct 1, 2026"),
                          ('TEXT(0.5,"0%")', "50%"), ('REPT("#",3)', "###"), ('MID("abcdef",2,3)', "bcd"),
                          ('INDEX(S!B1:B2,MATCH("y",S!A1:A2,0))', 7.0), ('IFERROR(MATCH("z",S!A1:A2,0),"none")', "none"),
                          ("IF(1>0,5,1/0)", 5.0), ("IFERROR(1/0,0)", 0.0)):
            self.assertEqual(e.formula("S", src), want, src)


class Example(unittest.TestCase):
    """The harness holds the work up against the example it was given: depth, and whether the content changed."""
    def test_how_much_is_in_the_example_and_what_still_is_its_own(self):
        from mlmi.backend import tools as T, verify
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            (Path(d) / "attachments").mkdir()
            T.run(ws, "sheets_create", {"path": "attachments/sales.ssheet", "headers": ["Pipeline", "Target"],
                                        "rows": [["Partners deals", 5], ["Diaspora deals", 7]]}, by="agent")
            T.run(ws, "sheets_set_cell", {"path": "attachments/sales.ssheet", "cell": "B4", "value": "=SUM(B2:B3)"}, by="agent")
            rel, says = verify.example_of(ws, "make one\n\nAttached: attachments/sales.ssheet (a workbook)")
            self.assertEqual(rel, "attachments/sales.ssheet")
            self.assertIn("formulas", says)
            T.run(ws, "copy_file", {"src": "attachments/sales.ssheet", "dst": "tech.ssheet"}, by="agent")
            same, total, left = verify.stale(ws, "tech.ssheet", "attachments/sales.ssheet")
            self.assertEqual((same, total), (total, total))
            self.assertTrue(left)
            T.run(ws, "sheets_set_cell", {"path": "tech.ssheet", "cell": "A2", "value": "Releases shipped"}, by="agent")
            self.assertEqual(verify.stale(ws, "tech.ssheet", "attachments/sales.ssheet")[0], total - 1)


class FilesAreWork(unittest.TestCase):
    def test_yes_after_a_request_that_was_only_files_and_a_few_words(self):
        m = chat("need a version of this but for the tech tema", "Labels done. Shall I go on?", "yes", files=["t.ssheet"])
        m[0]["attachments"] = ["attachments/a.xlsx"]
        held = shepherd.carried(m)
        self.assertIn("need a version", held["text"])
        self.assertEqual(held["attachments"], ["attachments/a.xlsx"])


class ManyCells(unittest.TestCase):
    def test_a_whole_sheet_in_one_call(self):
        from mlmi.backend import formats as F, tools as T
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            T.run(ws, "sheets_create", {"path": "t.ssheet", "headers": ["A", "B"]}, by="agent")
            T.run(ws, "sheets_set_many", {"path": "t.ssheet", "cells": {"A2": "Releases", "C9": 13, "D9": "=C9*2"}}, by="agent")
            cells = F.load(ws.path("t.ssheet"))["sheets"][0]["cells"]
            self.assertEqual((cells["A2"], cells["C9"], cells["D9"]), ("Releases", 13, "=C9*2"))
            # the sheet in the cell's name, and a reply with a broken comma
            T.run(ws, "sheets_set_many", {"path": "t.ssheet", "cells": {"Sheet1!A3": "Named"}}, by="agent")
            T.run(ws, "sheets_set_many", {"path": "t.ssheet", "cells": '{"A4": "Kept" "A5": 7}'}, by="agent")
            cells = F.load(ws.path("t.ssheet"))["sheets"][0]["cells"]
            self.assertEqual((cells["A3"], cells["A4"], cells["A5"]), ("Named", "Kept", 7))


class Guard(unittest.TestCase):
    def test_the_model_decides_unless_this_is_talk(self):
        for t in ("a version for the finance team", "the plan for my team please", "same but in french", "tech team version"):
            self.assertTrue(agent.artifact_action_allowed("words_create", t), t)
        for t in ("hello", "thanks, that looks great", "can you view images?", "what was on the last slide?",
                  "how do i make a chart?", "no file, just tell me", ""):
            self.assertFalse(agent.artifact_action_allowed("words_create", t), t)

    def test_as_many_files_as_were_asked_for(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            args = {"path": "tracker.ssheet"}
            self.assertIsNone(agent.in_scope(ws, "sheets_create", args, "make a work plan document and a tracker sheet", "", {"plan.sdoc"}))
            self.assertIsNone(agent.in_scope(ws, "make_image", {"path": "b.png"}, "draw three pictures of dogs", "", {"a.png"}))
            self.assertIn("already made", agent.in_scope(ws, "sheets_create", args, "write a report on tea", "", {"tea.sdoc"}))

    def test_a_file_it_read_can_be_changed(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            (Path(d) / "q3-roadmap.sdoc").write_text("{}", encoding="utf-8")
            args = {"path": "q3-roadmap.sdoc", "text": "x"}
            self.assertIn("not part", agent.in_scope(ws, "words_add", args, "update my plan", "", set()))
            self.assertIsNone(agent.in_scope(ws, "words_add", args, "update my plan", "", set(), {"q3-roadmap.sdoc"}))


class Finished(unittest.TestCase):
    """A file that exists is not a job that is done: Nemotron's 90-day plan was one heading."""
    def test_a_title_alone_is_short_of_the_request(self):
        from mlmi.backend import tools as T, verify
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            T.run(ws, "words_create", {"path": "plan.sdoc", "title": "Plan"}, by="agent")
            self.assertTrue(verify.shortfalls(ws, "write a work plan", {"plan.sdoc"}))
            self.assertEqual(verify.shortfalls(ws, "make a blank document", {"plan.sdoc"}), [])
            T.run(ws, "words_add", {"path": "plan.sdoc", "text": "Week one."}, by="agent")
            self.assertEqual(verify.shortfalls(ws, "write a work plan", {"plan.sdoc"}), [])
            T.run(ws, "slides_create", {"path": "d.sdeck", "title": "Deck"}, by="agent")
            self.assertTrue(verify.shortfalls(ws, "make a deck on tea", {"d.sdeck"}))

    def test_the_whole_document_in_one_call(self):
        from mlmi.backend import formats as F, tools as T
        md = "# Plan\n\nIntro words.\n\n## Goals\n- One\n- Two\n\n1. First\n\n> Note\n\n| Week | Owner |\n|---|---|\n| 1 | Sam |\n"
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            T.run(ws, "words_create", {"path": "p.sdoc", "content": md}, by="agent")
            blocks = F.load(ws.path("p.sdoc"))["blocks"]
            self.assertEqual([b["type"] for b in blocks], ["heading", "paragraph", "heading", "bullet", "bullet", "numbered",
                                                           "paragraph", "table"])
            self.assertEqual((blocks[0]["level"], blocks[0]["text"]), (0, "Plan"))
            self.assertEqual(blocks[-1]["rows"], [["Week", "Owner"], ["1", "Sam"]])


class Attached(unittest.TestCase):
    """Ernest's chat: "need a version of this but for the tech tema" with a file named ...2027.docx went to the web."""
    def test_a_file_name_is_not_a_question_about_now(self):
        model = Calls()
        off = types.SimpleNamespace(ready=lambda: False)
        st = types.SimpleNamespace(session=agent.Session(), ws=Workspace(Path(tempfile.mkdtemp(prefix="mlmi-flex-"))),
                                   model_card=lambda: {"tier": 0}, mode=1, permission="auto", local=off, cloud=model,
                                   orch=Orch(ready=False), busy=threading.Lock(), model_path="")
        (st.ws.root / "plan 2027.txt").write_text("the plan", encoding="utf-8")
        from mlmi.backend.tools import web
        was, web.STATE["on"] = web.STATE["on"], True
        events = []
        try:
            agent.run(st, [{"role": "user", "content": "need a version of this but for the tech tema",
                            "attachments": ["plan 2027.txt"]}], emit=events.append, auto_confirm="yes")
        finally:
            web.STATE["on"] = was
        self.assertFalse([e for e in events if str(e.get("name", "")).startswith("web_")])
        self.assertTrue([f for f in st.ws.tree() if f["path"].endswith(".sdoc")])


class ChatFiles(unittest.TestCase):
    def test_a_file_this_chat_made_is_open_to_the_next_turn(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d))
            (Path(d) / "tracker.ssheet").write_text("{}", encoding="utf-8")
            args = {"path": "tracker.ssheet", "cells": {"A1": "x"}}
            self.assertIn("not part", agent.in_scope(ws, "sheets_set_many", args, "need a version for the tech team", "", set()))
            self.assertIsNone(agent.in_scope(ws, "sheets_set_many", args, "need a version for the tech team", "", set(), {"tracker.ssheet"}))


class Orch:
    NEEDS = {}

    def __init__(self, ready=True):
        self.up, self.thermal = ready, None

    def holding(self):
        return False

    def ready(self):
        return self.up

    def chat_kwargs(self):
        return {}

    def after_turn(self, meta):
        pass

    def abilities(self):
        return {"text": True, "tools": True}

    def write_mode(self, kind):
        return False

    def entry(self, path=None):
        return {"knows_sheep": {"hints": []}}


class Calls:
    """A model that makes one document, then says so."""
    def __init__(self):
        self.step = 0

    def chat(self, msgs, tools=None, **k):
        self.step += 1
        if self.step == 1 and tools:
            call = {"id": "1", "function": {"name": "words_create",
                                            "arguments": json.dumps({"path": "plan.sdoc", "title": "Plan"})}}
            return {"content": "", "tool_calls": [call]}, {"route": "local", "ms": 1.0, "tokens": {}}
        return {"content": "Made plan.sdoc. I assumed the usual sections."}, {"route": "local", "ms": 1.0, "tokens": {}}

    def ready(self):
        return True


class AsksFirst(Calls):
    """A model that answers a job with a question, then does it when told to."""
    def chat(self, msgs, tools=None, **k):
        if self.step == 0:
            self.step = -1
            return {"content": "Sure! How many sections would you like?"}, {"route": "local", "ms": 1.0, "tokens": {}}
        self.step = max(self.step, 0)
        return super().chat(msgs, tools, **k)


class Whole(unittest.TestCase):
    def test_a_question_back_is_pressed_once_then_it_acts(self):
        st, events = self.run_it(chat(ASKED, PLAN, "up to you"), True, AsksFirst())
        self.assertTrue([f for f in st.ws.tree() if f["path"].endswith(".sdoc")])
        st, events = self.run_it(chat("write a report on tea"), True, AsksFirst())
        self.assertTrue([f for f in st.ws.tree() if f["path"].endswith(".sdoc")])

    def test_a_chat_question_is_never_pressed(self):
        st, events = self.run_it(chat("hello"), True, AsksFirst())
        self.assertFalse([e for e in events if "go ahead" in str(e.get("what"))])

    def run_it(self, messages, cloud_answers, model=None):
        model = model or Calls()
        off = types.SimpleNamespace(ready=lambda: False)
        st = types.SimpleNamespace(
            session=agent.Session(), ws=Workspace(Path(tempfile.mkdtemp(prefix="mlmi-flex-"))),
            model_card=lambda: {"tier": 0 if cloud_answers else 2}, mode=1 if cloud_answers else 2, permission="auto",
            local=model if not cloud_answers else off, cloud=model if cloud_answers else off,
            orch=Orch(ready=not cloud_answers), busy=threading.Lock(), model_path="" if cloud_answers else "m.ssf")
        events = []
        agent.run(st, messages, emit=events.append, auto_confirm="yes")
        return st, events

    def test_his_turn_makes_the_file_on_either_model(self):
        for cloud_answers in (True, False):
            st, events = self.run_it(chat(ASKED, PLAN, "can you just do the thinking"), cloud_answers)
            refused = [e for e in events if e.get("type") == "tool_result" and not e.get("ok")]
            self.assertEqual(refused, [], cloud_answers)
            self.assertTrue([f for f in st.ws.tree() if f["path"].endswith(".sdoc")], cloud_answers)


if __name__ == "__main__":
    unittest.main()
