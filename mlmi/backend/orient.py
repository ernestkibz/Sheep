"""Orientation: about 100 generic tasks per tool area, each graded by opening
what was made (Ernest, 2026-10-06: "run around 100 generic approaches for each
file tool use and a plug-in model, and update the orchestrator to that so as to
orient new models well").

Areas: slides, words, sheets, studio, files, chat (questions that must not touch
files), code, followup (several turns). Each task is a list of turns, an
optional setup (files that exist before it), and a grader that opens the
workspace and says pass or fail with the reason. The same tasks serve the
offline campaign (mlmi/tests/orient_run.py against a backend) and the
orchestrator's short orientation of a newly plugged-in model.

    tasks(area, n)            -> [Task]
    Task.setup(ws_root)       -> writes the files the task starts from
    Task.grade(ws_root, res)  -> (ok, why); res = {"texts", "files", "steps"}
"""
import json
import re
import time
from pathlib import Path

from . import formats as F

TOPICS = ["dog breeds", "the solar system", "healthy eating", "climate change in Uganda", "starting a small business",
          "the history of the internet", "coffee farming", "time management", "the water cycle", "football tactics",
          "renewable energy", "mobile money in East Africa", "first aid basics", "learning Python", "the Nile river",
          "saving money", "volcanoes", "photography tips", "remote work", "bees and pollination", "the human heart",
          "cyber security at home", "growing tomatoes", "public speaking", "electric cars", "the Roman Empire",
          "mental health at work", "chess openings", "recycling plastic", "space exploration", "tea in Kenya",
          "the rainforest", "volunteering", "job interviews", "road safety", "malaria prevention"]
SUBS = ["costs", "risks", "history", "benefits", "next steps", "common mistakes", "key facts", "examples", "tools",
        "a summary"]
THEMES = ["dark", "midnight", "forest", "sunset", "mono"]
PLACEHOLDER = re.compile(r"(lorem ipsum|\[insert|\binsert (text|title|name)\b|<[a-z ]+>|\bTBD\b|\bslide title\b|"
                         r"\bplaceholder\b|\bpoint [0-9]\b|\bbullet [0-9]\b|\.\.\.$)", re.I)
NUMW = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


class Task:
    def __init__(self, tid, area, turns, grade, setup=None, note=""):
        self.id, self.area, self.turns, self._grade, self._setup, self.note = tid, area, turns, grade, setup, note

    def setup(self, root):
        if self._setup:
            self._setup(Path(root))

    def grade(self, root, res):
        try:
            return self._grade(Path(root), res)
        except Exception as e:                       # a grader that cannot read the file: a fail, with why
            return False, f"grader: {type(e).__name__}: {e}"

    def as_dict(self):
        return {"id": self.id, "area": self.area, "turns": self.turns, "note": self.note}


# ------------------------------------------------------------------ reading what was made
def _load(root, rel):
    return F.load(root / rel)


def made(res, ext):
    return [f for f in res["files"] if f.lower().endswith(ext)]


def one(res, ext, root, name=""):
    """The file of this kind the job made (or the named one)."""
    if name:
        return name if (root / name).exists() else None
    fs = made(res, ext)
    return fs[-1] if fs else None


def no_placeholders(texts):
    bad = [t for t in texts if t and PLACEHOLDER.search(str(t))]
    return (False, f"placeholder text: {bad[0][:60]!r}") if bad else (True, "")


def deck_ok(root, rel, want=0, min_content=3, min_points=2):
    d = _load(root, rel)
    content = [s for s in d["slides"] if s.get("bullets") or s.get("layout") != "title"]
    # A request for an N-slide deck means N total slides, including its title
    # slide. Keep the grader aligned with the wording users see.
    if want and len(d["slides"]) < want:
        return False, f"{len(d['slides'])} slides, {want} asked"
    if not want and len(content) < min_content:
        return False, f"only {len(content)} content slides"
    thin = [s.get("title") for s in content if len(s.get("bullets") or []) < min_points]
    if thin:
        return False, f"slides with under {min_points} points: {thin[:3]}"
    titles = [str(s.get("title") or "").strip().lower() for s in d["slides"]]
    dup = {t for t in titles if t and titles.count(t) > 1}
    if dup:
        return False, f"repeated slides: {sorted(dup)[:3]}"
    return no_placeholders([s.get("title") for s in d["slides"]] + [b for s in d["slides"] for b in s.get("bullets") or []])


def doc_blocks(root, rel):
    return _load(root, rel).get("blocks", [])


def doc_text(root, rel):
    out = []
    for b in doc_blocks(root, rel):
        if b.get("type") == "table":
            out += [" ".join(map(str, r)) for r in b.get("rows", [])]
        else:
            out.append(str(b.get("text", "")))
    return "\n".join(out)


def sheet_cells(root, rel, i=0):
    return _load(root, rel)["sheets"][i]["cells"]


def sheet_grid(root, rel):
    cells = sheet_cells(root, rel)
    rows = {}
    for k, v in cells.items():
        m = re.match(r"([A-Z]+)(\d+)$", k)
        if m:
            rows.setdefault(int(m.group(2)), {})[m.group(1)] = v
    return [rows[r] for r in sorted(rows)]


def val(v):
    return v.get("v") if isinstance(v, dict) else v


def is_formula(v):
    return (isinstance(v, dict) and str(v.get("f", "")).startswith("=")) or (isinstance(v, str) and v.startswith("="))


def no_files(res):
    return (True, "") if not res["files"] else (False, f"made files for a question: {res['files'][:3]}")


def said(res, *words):
    t = " ".join(res["texts"]).lower()
    miss = [w for w in words if w.lower() not in t]
    return (True, "") if not miss else (False, f"reply lacks {miss}")


def claimed_nothing(res):
    """A reply that says it did file work when no file changed is the worst fault."""
    t = res["texts"][-1] if res["texts"] else ""
    if re.search(r"\b(i('ve| have)? (added|created|made|updated)|here'?s the updated)", t, re.I) and not res["files"]:
        return False, "claimed work, changed nothing"
    return True, ""


def all_of(*checks):
    for ok, why in checks:
        if not ok:
            return False, why
    return True, ""


# ------------------------------------------------------------------ files a task starts from
def mk_deck(root, rel, title, slides, theme="light"):
    d = F.new("presentations", theme=theme)
    d["slides"].append({"layout": "title", "title": title, "subtitle": "", "bullets": [], "notes": "", "items": []})
    for t, b in slides:
        d["slides"].append({"layout": "bullets", "title": t, "subtitle": "", "bullets": b, "notes": "", "items": []})
    F.save(root / rel, d)


def mk_doc(root, rel, title, paras):
    d = F.new("words")
    d["blocks"].append({"type": "heading", "level": 0, "text": title})
    for p in paras:
        d["blocks"].append({"type": "paragraph", "text": p})
    F.save(root / rel, d)


def mk_sheet(root, rel, rows):
    d = F.new("sheets")
    cells = d["sheets"][0]["cells"]
    for r, row in enumerate(rows, 1):
        for c, v in enumerate(row):
            cells[f"{chr(65 + c)}{r}"] = v
    F.save(root / rel, d)


STD_DECK = [("Overview", ["What it is", "Why it matters", "Who it helps"]),
            ("Key facts", ["Fact one about the subject", "A second fact", "A third fact"]),
            ("Next steps", ["Plan the work", "Share the plan", "Review in a month"])]
STD_ROWS = [["Item", "Cost"], ["Rent", 800], ["Food", 250], ["Transport", 90]]


# ------------------------------------------------------------------ the areas
def slides_tasks(n):
    out = []
    for i in range(n):
        tid = f"sl{i:03d}"
        topic = TOPICS[(i * 7) % len(TOPICS)]
        k = 3 + i % 4
        name = f"deck-{tid}"
        theme = THEMES[i % len(THEMES)]
        sub = SUBS[i % len(SUBS)]
        v = i % 16
        if v == 0:
            out.append(Task(tid, "slides", [(f"create a ppt for {topic}", "")],
                            lambda r, s: (lambda f: deck_ok(r, f) if f else (False, "no deck made"))(one(s, ".sdeck", r))))
        elif v == 1:
            out.append(Task(tid, "slides", [(f"make a {k}-slide deck on {topic}", "")],
                            lambda r, s, k=k: (lambda f: deck_ok(r, f, k) if f else (False, "no deck made"))(one(s, ".sdeck", r))))
        elif v == 2:
            out.append(Task(tid, "slides", [(f"Make a presentation called {name}.sdeck about {topic} with {NUMW.get(k, k)} slides", "")],
                            lambda r, s, k=k, nm=name: deck_ok(r, nm + ".sdeck", k) if (r / (nm + ".sdeck")).exists()
                            else (False, "named deck missing")))
        elif v == 3:
            out.append(Task(tid, "slides", [(f"I need slides on {topic}", "")],
                            lambda r, s: (lambda f: deck_ok(r, f) if f else (False, "no deck made"))(one(s, ".sdeck", r))))
        elif v == 4:
            out.append(Task(tid, "slides", [(f"make a {theme} themed presentation about {topic}", "")],
                            lambda r, s, th=theme: (lambda f: all_of(deck_ok(r, f), ((_load(r, f).get("theme") == th),
                                                                                       f"theme {_load(r, f).get('theme')}, {th} asked"))
                                                    if f else (False, "no deck made"))(one(s, ".sdeck", r))))
        elif v == 5:
            out.append(Task(tid, "slides", [(f"prepare a short presentation on {topic} and export it as pptx", "")],
                            lambda r, s: all_of(((lambda f: deck_ok(r, f) if f else (False, "no deck made"))(one(s, ".sdeck", r))),
                                                (bool(made(s, ".pptx")), "no .pptx exported"))))
        elif v == 6:
            out.append(Task(tid, "slides", [(f"can you make slides explaining {topic}?", "")],
                            lambda r, s: (lambda f: deck_ok(r, f) if f else (False, "no deck made"))(one(s, ".sdeck", r))))
        elif v == 7:
            out.append(Task(tid, "slides", [(f"add a slide about {sub} to {name}.sdeck", "")],
                            lambda r, s, nm=name: all_of(((len(_load(r, nm + ".sdeck")["slides"]) == 5),
                                                          f"{len(_load(r, nm + '.sdeck')['slides'])} slides, 5 expected"),
                                                         deck_ok(r, nm + ".sdeck")),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), STD_DECK)))
        elif v == 8:
            out.append(Task(tid, "slides", [(f"add {NUMW[2]} more slides to {name}.sdeck", "")],
                            lambda r, s, nm=name: all_of(((len(_load(r, nm + ".sdeck")["slides"]) == 6),
                                                          f"{len(_load(r, nm + '.sdeck')['slides'])} slides, 6 expected"),
                                                         deck_ok(r, nm + ".sdeck")),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), STD_DECK)))
        elif v == 9:
            out.append(Task(tid, "slides", [(f"change the theme of {name}.sdeck to {theme}", "")],
                            lambda r, s, nm=name, th=theme: ((_load(r, nm + ".sdeck").get("theme") == th),
                                                             f"theme {_load(r, nm + '.sdeck').get('theme')}"),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), STD_DECK)))
        elif v == 10:
            out.append(Task(tid, "slides", [(f"rename slide 2 of {name}.sdeck to Why It Matters", "")],
                            lambda r, s, nm=name: ((_load(r, nm + ".sdeck")["slides"][1].get("title") == "Why It Matters"),
                                                   f"slide 2 is {_load(r, nm + '.sdeck')['slides'][1].get('title')!r}"),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), STD_DECK)))
        elif v == 11:
            out.append(Task(tid, "slides", [(f"delete the last slide of {name}.sdeck", "")],
                            lambda r, s, nm=name: (len(_load(r, nm + ".sdeck")["slides"]) == 3
                                                   and _load(r, nm + ".sdeck")["slides"][-1]["title"] == "Key facts",
                                                   f"slides now {[x['title'] for x in _load(r, nm + '.sdeck')['slides']]}"),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), STD_DECK)))
        elif v == 12:
            out.append(Task(tid, "slides", [(f"add speaker notes to slide 2 of {name}.sdeck", "")],
                            lambda r, s, nm=name: (bool(_load(r, nm + ".sdeck")["slides"][1].get("notes")), "no notes on slide 2"),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), STD_DECK)))
        elif v == 13:
            thin = ["this deck only has a title slide, fill it in", "there is only one slide here, add the rest",
                    "finish this deck, it is just a title", "the deck is empty apart from the title, build it out",
                    "add the missing slides to this deck", "this needs its content slides",
                    "fill out the rest of this presentation", "only the title is done, do the others",
                    "flesh this deck out", "complete this deck for me", "put some slides in this deck",
                    "the body of this deck is missing", "write the slides that go after the title"][i // 12 % 13]
            out.append(Task(tid, "slides", [(thin, "slides", f"{name}.sdeck")],
                            lambda r, s, nm=name: deck_ok(r, nm + ".sdeck"),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), [])))
        elif v == 14:
            out.append(Task(tid, "slides", [(f"export {name}.sdeck to powerpoint", "")],
                            lambda r, s, nm=name: ((r / (nm + ".pptx")).exists() or bool(made(s, ".pptx")), "no .pptx"),
                            lambda r, nm=name, t=topic: mk_deck(r, nm + ".sdeck", t.title(), STD_DECK)))
        else:
            out.append(Task(tid, "slides", [(f"put together a slideshow about {topic} with {k} slides", "")],
                            lambda r, s, k=k: (lambda f: deck_ok(r, f, k) if f else (False, "no deck made"))(one(s, ".sdeck", r))))
    return out


def doc_ok(root, rel, sections=0, items=0, table=False, min_blocks=3, shape="list"):
    blocks = doc_blocks(root, rel)
    body = [b for b in blocks if not (b.get("type") == "heading" and b.get("level", 1) == 0)]
    if len(body) < min_blocks:
        return False, f"only {len(body)} blocks of body"
    if sections and sum(b.get("type") == "heading" for b in body) < sections:
        return False, f"{sum(b.get('type') == 'heading' for b in body)} sections, {sections} asked"
    if items:
        n = sum(b.get("type") in ("bullet", "numbered") for b in body)
        titled = len(body) < len(blocks)        # a level-0 title is already out of body; else the first heading is it
        heads = sum(b.get("type") == "heading"
                    for b in (body[1:] if not titled and body and body[0].get("type") == "heading" else body))
        n = n if shape == "list" else max(n, heads)    # "any": five headed tips are five items too
        if n < items:
            return False, f"{n} list items, {items} asked"
    if table and not any(b.get("type") == "table" for b in body):
        return False, "no table"
    return no_placeholders([str(b.get("text", "")) for b in blocks])


def words_tasks(n):
    out = []
    for i in range(n):
        tid = f"wd{i:03d}"
        topic = TOPICS[(i * 5 + 3) % len(TOPICS)]
        k = 2 + i % 3
        name = f"doc-{tid}"
        sub = SUBS[i % len(SUBS)]
        v = i % 14
        mkd = lambda r, nm=name, t=topic: mk_doc(r, nm + ".sdoc", t.title(), [
            f"This document is about {t}.", "It was written for the team.", "Read it before the meeting."])
        anydoc = lambda r, s, **kw: (lambda f: doc_ok(r, f, **kw) if f else (False, "no document made"))(one(s, ".sdoc", r))
        if v == 0:
            out.append(Task(tid, "words", [(f"write a report on {topic}", "")], lambda r, s: anydoc(r, s)))
        elif v == 1:
            out.append(Task(tid, "words", [(f"create a document about {topic}", "")], lambda r, s: anydoc(r, s)))
        elif v == 2:
            out.append(Task(tid, "words", [(f"draft a letter to my manager about {topic}", "")],
                            lambda r, s: anydoc(r, s, min_blocks=2)))
        elif v == 3:
            out.append(Task(tid, "words", [(f"Make a one-page brief on {topic} called {name}.sdoc with a short summary and three goals", "")],
                            lambda r, s, nm=name: doc_ok(r, nm + ".sdoc", items=3) if (r / (nm + ".sdoc")).exists()
                            else (False, "named document missing")))
        elif v == 4:
            out.append(Task(tid, "words", [(f"write an essay on {topic} with {k} sections", "")],
                            lambda r, s, k=k: anydoc(r, s, sections=k)))
        elif v == 5:
            out.append(Task(tid, "words", [(f"make a document about {topic} with a table", "")],
                            lambda r, s: anydoc(r, s, table=True, min_blocks=2)))
        elif v == 6:
            out.append(Task(tid, "words", [(f"write a guide to {topic} with a numbered list of steps", "")],
                            lambda r, s: anydoc(r, s, items=3)))
        elif v == 7:
            out.append(Task(tid, "words", [(f"create a word document on {topic}", "")],
                            lambda r, s: all_of(anydoc(r, s), (bool(made(s, ".docx")), "no .docx exported"))))
        elif v == 8:
            out.append(Task(tid, "words", [(f"write meeting notes about {topic} with five bullet points", "")],
                            lambda r, s: anydoc(r, s, items=5, min_blocks=5)))
        elif v == 9:
            out.append(Task(tid, "words", [(f"add a section about {sub} to {name}.sdoc", "")],
                            lambda r, s, nm=name: ((len(doc_blocks(r, nm + ".sdoc")) >= 6
                                                    and any(b.get("type") == "heading" and b.get("level", 1) > 0
                                                            for b in doc_blocks(r, nm + ".sdoc"))),
                                                   f"{len(doc_blocks(r, nm + '.sdoc'))} blocks, no new section"), mkd))
        elif v == 10:
            out.append(Task(tid, "words", [(f"add a conclusion to {name}.sdoc", "")],
                            lambda r, s, nm=name: ("conclusion" in doc_text(r, nm + ".sdoc").lower()
                                                   or len(doc_blocks(r, nm + ".sdoc")) >= 5, "no conclusion added"), mkd))
        elif v == 11:
            out.append(Task(tid, "words", [(f"replace 'team' with 'whole company' in {name}.sdoc", "")],
                            lambda r, s, nm=name: ("whole company" in doc_text(r, nm + ".sdoc")
                                                   and " team" not in doc_text(r, nm + ".sdoc"), "word not replaced"), mkd))
        elif v == 12:
            out.append(Task(tid, "words", [(f"summarize {name}.sdoc", "")],
                            lambda r, s, t=topic, nm=name: all_of(said(s, t.split()[-1]),
                                                                  (len(doc_blocks(r, nm + ".sdoc")) == 4, "the document was changed")),
                            mkd))
        else:
            out.append(Task(tid, "words", [(f"export {name}.sdoc as docx", "")],
                            lambda r, s, nm=name: ((r / (nm + ".docx")).exists() or bool(made(s, ".docx")), "no .docx"), mkd))
    return out


def sheet_ok(root, rel, rows=0, cols=2, total=False):
    g = sheet_grid(root, rel)
    if len(g) < 2:
        return False, "no data rows"
    head = g[0]
    if len(head) < cols:
        return False, f"{len(head)} columns"
    body = g[1:]
    data = [r for r in body if not re.match(r"^\s*(total|sum)", str(val(r.get("A", ""))), re.I)]
    if rows and len(data) < rows:
        return False, f"{len(data)} rows, {rows} asked"
    nums = sum(isinstance(val(v), (int, float)) or is_formula(v) for r in data for v in r.values())
    if nums == 0:
        return False, "no numbers"
    if total and not any(is_formula(v) for r in body for v in r.values()):
        return False, "no total formula"
    return True, ""


def sheets_tasks(n):
    out = []
    things = [("countries", "capital and population"), ("fruits", "price per kg"), ("students", "three test scores"),
              ("months", "income and spending"), ("products", "price and stock"), ("cities", "population"),
              ("employees", "department and salary"), ("books", "author and year")]
    for i in range(n):
        tid = f"sh{i:03d}"
        k = 4 + i % 5
        name = f"book-{tid}"
        th, attrs = things[i % len(things)]
        v = i % 12
        mks = lambda r, nm=name: mk_sheet(r, nm + ".ssheet", STD_ROWS)
        anys = lambda r, s, **kw: (lambda f: sheet_ok(r, f, **kw) if f else (False, "no workbook made"))(one(s, ".ssheet", r))
        if v == 0:
            trio = [("rent, food and transport", "budget"), ("fuel, airtime and water", "monthly costs"),
                    ("seeds, fertiliser and labour", "farm budget"), ("rent, salaries and stock", "shop budget"),
                    ("flights, hotel and food", "trip budget"), ("tuition, books and lunch", "school budget"),
                    ("rent, internet and power", "office budget"), ("feeds, vet and transport", "poultry budget"),
                    ("venue, catering and music", "event budget")][i // 12 % 9]
            out.append(Task(tid, "sheets", [(f"make a {trio[1]} spreadsheet with {trio[0]}", "")],
                            lambda r, s: anys(r, s, rows=3)))
        elif v == 1:
            out.append(Task(tid, "sheets", [(f"Make a workbook called {name}.ssheet with Item and Cost: Rent 800, "
                                             f"Food 250, Transport 90, and a Total row", "")],
                            lambda r, s, nm=name: sheet_ok(r, nm + ".ssheet", rows=3, total=True)
                            if (r / (nm + ".ssheet")).exists() else (False, "named workbook missing")))
        elif v == 2:
            out.append(Task(tid, "sheets", [(f"make a spreadsheet of {k} {th} with their {attrs}", "")],
                            lambda r, s, k=k: anys(r, s, rows=k)))
        elif v == 3:
            what = ["my monthly expenses", "my weekly shopping", "the hours my team worked",
                    "what I spend on transport", "my savings each month", "the stock in my shop",
                    "my water bills", "what the farm earns", "my phone data use"][i // 12 % 9]
            out.append(Task(tid, "sheets", [(f"track {what} in a sheet", "")], lambda r, s: anys(r, s, rows=3)))
        elif v == 4:
            out.append(Task(tid, "sheets", [(f"create an excel file listing {k} {th} with their {attrs}", "")],
                            lambda r, s, k=k: all_of(anys(r, s, rows=k), (bool(made(s, ".xlsx")), "no .xlsx exported"))))
        elif v == 5:
            kind = ["sales", "rainfall", "attendance", "orders", "fuel use", "deliveries",
                    "milk yield", "visitors", "repairs"][i // 12 % 9]
            out.append(Task(tid, "sheets", [(f"make a {kind} table for January to June with a total", "")],
                            lambda r, s: anys(r, s, rows=6, total=True)))
        elif v == 6:
            out.append(Task(tid, "sheets", [(f"add a row for Internet costing 60 to {name}.ssheet", "")],
                            lambda r, s, nm=name: (any(str(val(c.get("A", ""))).lower() == "internet"
                                                       for c in sheet_grid(r, nm + ".ssheet")), "no Internet row"), mks))
        elif v == 7:
            out.append(Task(tid, "sheets", [(f"sort {name}.ssheet by cost, highest first", "")],
                            lambda r, s, nm=name: ([val(c.get("B")) for c in sheet_grid(r, nm + ".ssheet")[1:4]] == [800, 250, 90]
                                                   and str(val(sheet_grid(r, nm + ".ssheet")[1].get("A"))) == "Rent", "not sorted"), mks))
        elif v == 8:
            out.append(Task(tid, "sheets", [(f"add a total row to {name}.ssheet", "")],
                            lambda r, s, nm=name: (any(is_formula(x) for c in sheet_grid(r, nm + ".ssheet") for x in c.values()),
                                                   "no total formula"), mks))
        elif v == 9:
            out.append(Task(tid, "sheets", [(f"set B2 in {name}.ssheet to 950", "")],
                            lambda r, s, nm=name: (val(sheet_cells(r, nm + ".ssheet").get("B2")) == 950,
                                                   f"B2 is {val(sheet_cells(r, nm + '.ssheet').get('B2'))!r}"), mks))
        elif v == 10:
            out.append(Task(tid, "sheets", [(f"what is the total cost in {name}.ssheet?", "")],
                            lambda r, s: said(s, "1140"), mks))
        else:
            out.append(Task(tid, "sheets", [(f"add a chart of the costs in {name}.ssheet", "")],
                            lambda r, s, nm=name: (bool(_load(r, nm + ".ssheet")["sheets"][0].get("charts")), "no chart"), mks))
    return out


def studio_tasks(n):
    out = []
    brands = ["Sheep", "Kampala Coffee", "Nile Tech", "Green Farm", "Bright Kids", "Unity Bank", "Safari Tours",
              "Pearl Motors", "Lake View Hotel", "Mbale Fresh", "Rwenzori Water", "Savannah Books",
              "Jinja Crafts", "Equator Dairy", "Sunrise Clinic", "Kigezi Tea", "Victoria Boats"]
    for i in range(n):
        tid = f"st{i:03d}"
        v = i % 6
        # the brand and the length step once per round of the six jobs, not per
        # task: keyed on i they cycled WITH v and 100 tasks held 21 distinct ones
        b = brands[i // 6 % len(brands)]
        secs = [8, 10, 12, 15, 6, 20, 9, 14][i // 6 % 8]
        proj = lambda r, s: one(s, ".sstudio", r)

        def length_ok(r, s, want=0):
            f = proj(r, s)
            if not f:
                return False, "no Studio project made"
            d = _load(r, f)
            from .tools.studio import length
            if want and length(d) + 0.5 < want:
                return False, f"{length(d):.0f} s long, {want} asked"
            texts = [str(l.get("text", "")) for l in d["layers"] if l.get("kind") == "text"]
            return (True, "") if texts else (False, "no text layers")
        if v == 0:
            out.append(Task(tid, "studio", [(f"make a {secs}-second intro video with the title {b}", "")],
                            lambda r, s, w=secs: length_ok(r, s, w)))
        elif v == 1:
            out.append(Task(tid, "studio", [(f"make a logo reveal for {b}", "")], lambda r, s: length_ok(r, s)))
        elif v == 2:
            out.append(Task(tid, "studio", [(f"make a 9:16 short about {b}", "")],
                            lambda r, s: all_of(length_ok(r, s), ((lambda f: f and _load(r, f).get("h", 0) > _load(r, f).get("w", 1))(proj(r, s)),
                                                                  "not portrait"))))
        elif v == 3:
            says = ["thanks for watching", "subscribe for more", f"follow {b}", "see you next time",
                    f"visit {b} today", "like and share", f"call {b} now", "more coming soon",
                    f"thank you from {b}", "stay tuned", "join us again", f"{b} - see you soon",
                    "goodbye for now", "share with a friend", "until next week", "that is all",
                    "come back tomorrow"][i // 6 % 17]
            out.append(Task(tid, "studio", [(f"create an outro that says {says}", "")],
                            lambda r, s: length_ok(r, s)))
        elif v == 4:
            out.append(Task(tid, "studio", [(f"make a {secs}-second title intro for {b} and render it", "")],
                            lambda r, s, w=secs: all_of(length_ok(r, s, w), (bool(made(s, ".mp4")), "not rendered"))))
        else:
            out.append(Task(tid, "studio", [(f"make a video intro for {b} with the words Welcome to {b}", "")],
                            lambda r, s: length_ok(r, s)))
    return out


def files_tasks(n):
    out = []
    for i in range(n):
        tid = f"fl{i:03d}"
        a = f"note-{tid}.txt"
        folder = f"folder-{tid}"
        v = i % 8
        mk = lambda r, a=a: (r / a).write_text(f"Meeting moved to Friday at 10.\n", encoding="utf-8")
        if v == 0:
            # asked a different way each time round: 100 tasks should be 100 jobs,
            # not one job 13 times (2026-10-07)
            ask = ["list my files", "what files do I have?", "show me what is in my workspace",
                   "which files are in here?", "give me a list of everything I have saved",
                   "what have I got in this folder?", "show my documents", "what did I save here?",
                   "list everything in the workspace", "tell me which files exist",
                   "what is stored in here?", "show me my work", "list the contents of my folder"][i // 8 % 13]
            out.append(Task(tid, "files", [(ask, "")],
                            lambda r, s: (any(p.name.lower() in " ".join(s["texts"]).lower() for p in r.iterdir()),
                                          "names no file that is there"), mk))
        elif v == 1:
            out.append(Task(tid, "files", [(f"create a folder called {folder}", "")],
                            lambda r, s, f=folder: ((r / f).is_dir(), "no folder")))
        elif v == 2:
            out.append(Task(tid, "files", [(f"rename {a} to renamed-{tid}.txt", "")],
                            lambda r, s, t=tid, a=a: ((r / f"renamed-{t}.txt").exists() and not (r / a).exists(), "not renamed"), mk))
        elif v == 3:
            out.append(Task(tid, "files", [(f"make a copy of {a} called copy-{tid}.txt", "")],
                            lambda r, s, t=tid, a=a: ((r / f"copy-{t}.txt").exists() and (r / a).exists(), "not copied"), mk))
        elif v == 4:
            out.append(Task(tid, "files", [(f"delete {a}", "")], lambda r, s, a=a: (not (r / a).exists(), "not deleted"), mk))
        elif v == 5:
            out.append(Task(tid, "files", [(f"write a text file called todo-{tid}.txt with three things to do today", "")],
                            lambda r, s, t=tid: ((r / f"todo-{t}.txt").exists()
                                                 and len((r / f"todo-{t}.txt").read_text(encoding="utf-8").strip().splitlines()) >= 3,
                                                 "no 3-line todo file")))
        elif v == 6:
            out.append(Task(tid, "files", [(f"read {a} and tell me what it says", "")],
                            lambda r, s: said(s, "friday"), mk))
        else:
            out.append(Task(tid, "files", [(f"move {a} into a folder called {folder}", "")],
                            lambda r, s, a=a, f=folder: ((r / f / a).exists(), "not moved"), mk))
    return out


QA = [("What is the capital of Kenya?", "nairobi"), ("How many legs does a spider have?", "8|eight"),
      ("What is 17 times 23?", "391"), ("Who wrote Romeo and Juliet?", "shakespeare"),
      ("What is the boiling point of water in Celsius?", "100"), ("Translate 'thank you' into French.", "merci"),
      ("What is the largest planet in our solar system?", "jupiter"), ("How many days are in a leap year?", "366"),
      ("What gas do plants take in from the air?", "carbon dioxide|co2"), ("What is 144 divided by 12?", "12"),
      ("Which ocean is the largest?", "pacific"), ("What is the square root of 81?", "9|nine"),
      ("Explain photosynthesis in two sentences.", "light"), ("Why is the sky blue?", "scatter"),
      ("Give me three tips for sleeping better.", "sleep"), ("What does CPU stand for?", "central processing unit"),
      ("Who painted the Mona Lisa?", "leonardo|vinci"), ("What is the chemical symbol for gold?", "au"),
      ("How many continents are there?", "7|seven"), ("What year did the Second World War end?", "1945")]


def chat_tasks(n):
    out = []
    for i in range(n):
        q, want = QA[i % len(QA)]
        if i >= len(QA):
            q = ["Quick question: ", "Hi! ", "Please tell me: ", "Can you tell me ", ""][i // len(QA) % 5] + q[0].lower() + q[1:]
        out.append(Task(f"ch{i:03d}", "chat", [(q, "")],
                        lambda r, s, w=want: all_of(no_files(s), (not s["steps"], f"used tools: {[x[0] for x in s['steps']][:3]}"),
                                                    (bool(re.search(w, " ".join(s["texts"]).lower())), f"answer lacks {w!r}"))))
    return out


CODE = [("write a python function that reverses a string", "def "), ("show me javascript to sort an array of numbers", "sort"),
        ("write a SQL query that counts users by country", "count"), ("give me a bash command to count lines in a file", "wc"),
        ("write a python function to check if a number is prime", "def "), ("html for a button that says Hello", "<button"),
        ("write a regex that matches an email address", "@"), ("python code to read a CSV file", "csv"),
        ("write a function in javascript that adds two numbers", "function"), ("css to center a div", "center")]


def code_tasks(n):
    out = []
    for i in range(n):
        tid = f"cd{i:03d}"
        if i % 5 == 4:
            out.append(Task(tid, "code", [(f"save a python script called hello-{tid}.py that prints hello", "")],
                            lambda r, s, t=tid: ((r / f"hello-{t}.py").exists() and "print" in (r / f"hello-{t}.py").read_text(
                                encoding="utf-8"), "no script saved")))
            continue
        q, want = CODE[i % len(CODE)]
        if i >= len(CODE):
            q = ["", "I need ", "Please ", "Quick one: ", "Can you ", "Help me: ", "For my project, ",
                 "Write this for me: ", "At work I need ", "For practice, "][i // len(CODE) % 10] + q[0].lower() + q[1:]
        out.append(Task(tid, "code", [(q, "")],
                        lambda r, s, w=want: all_of(no_files(s), ("```" in " ".join(s["texts"]), "no code block"),
                                                    (w.lower() in " ".join(s["texts"]).lower(), f"code lacks {w!r}"))))
    return out


def followup_tasks(n):
    out = []
    for i in range(n):
        tid = f"fu{i:03d}"
        topic = TOPICS[(i * 11 + 1) % len(TOPICS)]
        if i >= len(TOPICS):                      # a new angle each time round, so no task is asked twice
            topic = f"{topic}: {SUBS[i // len(TOPICS) % len(SUBS)]}"
        v = i % 9
        if v == 0:
            out.append(Task(tid, "followup", [(f"create a presentation about {topic}", ""), ("add two more slides", "slides@")],
                            lambda r, s: (lambda f: all_of(deck_ok(r, f), (len(_load(r, f)["slides"]) >= 6,
                                                                           f"{len(_load(r, f)['slides'])} slides after adding two"))
                                          if f else (False, "no deck"))(one(s, ".sdeck", r))))
        elif v == 1:
            out.append(Task(tid, "followup", [(f"write a short report on {topic}", ""), ("add a conclusion", "words@")],
                            lambda r, s: (lambda f: ("conclusion" in doc_text(r, f).lower(), "no conclusion")
                                          if f else (False, "no document"))(one(s, ".sdoc", r))))
        elif v == 2:
            kinds = ["budget spreadsheet with rent, food and transport",
                     "spreadsheet of my weekly shopping with prices",
                     "spreadsheet of farm costs with seeds, fertiliser and labour",
                     "spreadsheet of my trip costs with flights, hotel and food",
                     "spreadsheet of shop takings for three days",
                     "spreadsheet of my fuel spending by week",
                     "spreadsheet of school fees for three terms",
                     "spreadsheet of rent collected from four tenants",
                     "spreadsheet of my data bundles and their cost",
                     "spreadsheet of the hours I worked this week",
                     "spreadsheet of produce sold at the market",
                     "spreadsheet of my savings goals and amounts",
                     "spreadsheet of repairs and what each cost"][i // 8 % 13]
            out.append(Task(tid, "followup", [(f"make a {kinds}", ""),
                                              ("add a total row", "sheets@")],
                            lambda r, s: (lambda f: sheet_ok(r, f, total=True) if f else (False, "no workbook"))(one(s, ".ssheet", r))))
        elif v == 3:
            out.append(Task(tid, "followup", [(f"create a deck about {topic}", ""), ("make it dark", "slides@")],
                            lambda r, s: (lambda f: (_load(r, f).get("theme") == "dark", f"theme {_load(r, f).get('theme')}")
                                          if f else (False, "no deck"))(one(s, ".sdeck", r))))
        elif v == 4:
            out.append(Task(tid, "followup", [(f"give me five tips about {topic}", ""), ("put that in a document", "")],
                            lambda r, s: (lambda f: doc_ok(r, f, items=5, min_blocks=5, shape="any") if f else (False, "no document"))(one(s, ".sdoc", r))))
        elif v == 5:
            tbl = ["five fruits and their prices", "four cities and their populations",
                   "five tools and what each costs", "three crops and their yields",
                   "four phones and their prices", "five books and their authors",
                   "three routes and their fares", "four shops and their opening hours",
                   "five animals and their weights", "three terms and their fees",
                   "four hotels and their nightly rates", "five snacks and their prices",
                   "three plots and their sizes"][i // 8 % 13]
            out.append(Task(tid, "followup", [(f"give me a table of {tbl}", ""), ("put that in a spreadsheet", "")],
                            lambda r, s: (lambda f: sheet_ok(r, f, rows=5) if f else (False, "no workbook"))(one(s, ".ssheet", r))))
        elif v == 6:
            out.append(Task(tid, "followup", [(f"list the main points about {topic}", ""), ("now make slides from that", "")],
                            lambda r, s: (lambda f: deck_ok(r, f) if f else (False, "no deck"))(one(s, ".sdeck", r))))
        elif v == 8:   # no verb, then a bare format word (Ernest's chat, 2026-10-20)
            out.append(Task(tid, "followup", [(f"powerpoint about {topic}", ""), ("pptx", "")],
                            lambda r, s: (lambda f: all_of(deck_ok(r, f), (bool(made(s, ".pptx")), "no .pptx"))
                                          if f else (False, "no deck"))(one(s, ".sdeck", r))))
        else:
            out.append(Task(tid, "followup", [(f"create a ppt for {topic}", ""), ("only has a few slides, add more", "slides@")],
                            lambda r, s: (lambda f: (len(_load(r, f)["slides"]) >= 7, f"{len(_load(r, f)['slides'])} slides")
                                          if f else (False, "no deck"))(one(s, ".sdeck", r))))
    return out


def _media_file(res, root, exts, min_bytes):
    """The file this job made of one of the kinds: real bytes, not an empty stub."""
    for ext in exts:
        f = one(res, ext, root)
        if f and (root / f).exists() and (root / f).stat().st_size >= min_bytes:
            return f
    return None


NOT_SET_UP = "Sheep is not set up for this yet"


def media_ok(root, res, exts, min_bytes, kind):
    """A media job passes when it made a real file, or when this computer lacks the media
    model and Sheep answered with its setup checklist (agent.py): that is the right
    answer there, not a failure of the text model."""
    if _media_file(res, root, exts, min_bytes):
        return True, ""
    tools = {"picture": ("make_image",), "video": ("make_video", "studio_render"), "audio": ("speak_text", "studio_voice")}
    if any(s and s[0] in tools.get(kind, ()) and s[1] for s in res.get("steps") or []):
        return True, f"Sheep reached {kind} making (checks never render media)"
    if any(NOT_SET_UP in str(t) for t in res.get("texts") or []):
        return True, f"no {kind} model here: Sheep showed its setup checklist"
    return False, f"no {kind} file made"


PICTURE_ASKS = ["a red barn at sunset", "a cup of coffee on a wooden table", "a sheep standing in green grass",
                "a bicycle leaning on a brick wall", "a fishing boat on a calm lake", "a village market in the morning",
                "a tall tree with yellow leaves", "a laptop on a desk by a window"]


def picture_tasks(n):
    """Pictures made from a description: the job passes when a real picture file is made."""
    out = []
    for i in range(n):
        tid = f"pi{i:03d}"
        what = PICTURE_ASKS[i % len(PICTURE_ASKS)]
        ask = [f"make a picture of {what}", f"generate an image of {what}", f"draw {what}",
               f"create a logo-style picture of {what}"][i // len(PICTURE_ASKS) % 4]
        out.append(Task(tid, "picture", [(ask, "")],
                        lambda r, s: media_ok(r, s, (".png", ".jpg", ".jpeg", ".webp"), 2000, "picture")))
    return out


VIDEO_ASKS = ["a sheep walking across a hill", "waves rolling onto a beach", "a candle flickering in the dark",
              "a train passing through snowy mountains", "rain falling on a city street at night"]


def video_tasks(n):
    """A short video made from a description: the job passes when a real .mp4 is made."""
    out = []
    for i in range(n):
        tid = f"vi{i:03d}"
        what = VIDEO_ASKS[i % len(VIDEO_ASKS)]
        ask = [f"make a short video of {what}", f"generate a video of {what}",
               f"create a 3 second video clip of {what}"][i // len(VIDEO_ASKS) % 3]
        out.append(Task(tid, "video", [(ask, "")],
                        lambda r, s: media_ok(r, s, (".mp4",), 10_000, "video")))
    return out


SPEECH_TEXTS = ["Good morning from Sheep.", "The meeting starts at ten.", "Thank you for your patience.",
                "Please save the file before you leave."]


def speech_tasks(n):
    """Spoken audio from text: the job passes when a real audio file is made."""
    out = []
    for i in range(n):
        tid = f"sp{i:03d}"
        said = SPEECH_TEXTS[i % len(SPEECH_TEXTS)]
        ask = [f"read this aloud into voice-{tid}.wav: {said}", f"make a recording of this: {said}",
               f"narrate this into voice-{tid}.wav: {said}"][i // len(SPEECH_TEXTS) % 3]
        out.append(Task(tid, "speech", [(ask, "")],
                        lambda r, s: media_ok(r, s, (".wav", ".mp3"), 1000, "audio")))
    return out


AREAS = {"slides": slides_tasks, "words": words_tasks, "sheets": sheets_tasks, "studio": studio_tasks,
         "files": files_tasks, "chat": chat_tasks, "code": code_tasks, "followup": followup_tasks,
         "picture": picture_tasks, "video": video_tasks, "speech": speech_tasks}


def tasks(area, n=100):
    return AREAS[area](n)


# the short one-time orientation a new model gets after its activation exam:
# making/editing files, a file job, a question, code, a follow-up, and a picture,
# a video and a voice recording (2026-10-14: a model passed without ever making one)
QUICK = [("slides", 0), ("slides", 8), ("words", 0), ("words", 9), ("sheets", 1), ("sheets", 6), ("files", 5),
         ("chat", 2), ("code", 0), ("followup", 4), ("followup", 8), ("picture", 0), ("video", 0), ("speech", 0)]


def quick():
    out = []
    for area, i in QUICK:
        out.append(tasks(area, i + 1)[i])
    return out


def run_inprocess(state, task_list, root, progress=lambda i, n, t: None, existing_rows=None,
                  save_row=None, should_yield=None, should_stop=None, sampling=None):
    """Run tasks through the agent itself in a private workspace. Returns result rows.

    Background orientation can restore completed rows and yield before each task
    so interactive requests keep priority and interrupted runs resume safely.
    """
    from . import agent
    from .workspace import Workspace

    class View:
        """The app's state as the agent sees it, with the orientation's own workspace,
        stop flag and an orchestrator view that is ready (it is the one asking)."""
        def __init__(self, st, ws, sampling=None):
            self._st, self.ws, self.mode, self.permission = st, ws, 2, "auto"
            self.session = agent.Session()
            self.rehearsal = True          # media tools are reached, never run, inside the checks (agent.call)
            o = st.orch

            class O:
                """The real orchestrator (abilities, NEEDS, entry...), except: ready, never holding, no turn
                statistics. A bare stub hid abilities, so every picture/video/speech task failed (2026-10-20)."""
                ready = staticmethod(lambda: True)
                holding = staticmethod(lambda: False)
                after_turn = staticmethod(lambda info: None)

                def __getattr__(self, k):
                    return getattr(o, k)
            self.orch = O()
            if sampling:                   # the second try is sampled, so it is not a copy of the first
                self.orch.chat_kwargs = lambda: dict(o.chat_kwargs(), **sampling)

        def __getattr__(self, k):
            return getattr(self._st, k)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    view = View(state, Workspace(root), sampling)
    rows = list(existing_rows or [])[:len(task_list)]
    for i, t in enumerate(task_list):
        if i < len(rows):
            continue
        if should_stop is not None and should_stop():
            return rows
        # Return at a safe task boundary so the caller releases its scheduler
        # lock for interactive work. Completed rows are persisted and resume on
        # the next queue, rather than waiting here while holding the lock.
        if should_yield is not None and not should_yield():
            return rows
        progress(i, len(task_list), t)
        t.setup(root)
        messages, texts, files, steps, calls = [], [], [], [], {}
        t0 = time.time()
        try:
            for spec in t.turns:
                text, app = spec[0], spec[1]
                path = spec[2] if len(spec) > 2 else ""
                if app.endswith("@"):
                    app = app[:-1]
                    ext = {"slides": ".sdeck", "words": ".sdoc", "sheets": ".ssheet"}.get(app, "#")
                    path = next((f for f in reversed(files) if f.endswith(ext)), "")
                messages.append({"role": "user", "content": text})
                got = {"text": "", "files": []}

                def emit(e, got=got):
                    if e["type"] == "tool_call":
                        calls[e["id"]] = e["name"]
                    elif e["type"] == "tool_result":
                        steps.append([calls.get(e["id"], e.get("name")), bool(e.get("ok")), str(e.get("text", ""))[:200]])
                    elif e["type"] == "text" and not e.get("partial"):
                        got["text"] = e.get("text", "")
                    elif e["type"] == "file_changed" and e["path"] not in got["files"]:
                        got["files"].append(e["path"])
                agent.run(view, messages, app, path, emit, auto_confirm="yes", cache=False)
                messages.append({"role": "assistant", "content": got["text"],
                                 **({"files": got["files"]} if got["files"] else {})})
                texts.append(got["text"])
                files += [f for f in got["files"] if f not in files]
            ok, why = t.grade(root, {"texts": texts, "files": files, "steps": steps})
        except Exception as e:
            ok, why = False, f"run: {type(e).__name__}: {str(e)[:160]}"
        row = {"id": t.id, "area": t.area, "ok": bool(ok), "why": "" if ok else why, "note": why if ok else "",
               "s": round(time.time() - t0, 1), "turns": [x[0] for x in t.turns]}
        try:                           # what was asked, done and made, for Settings › Activity
            with open(Path(root) / "transcript.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**row, "t": time.time(), "steps": steps[:40],
                                     "replies": [x[:1000] for x in texts], "files": files[:20]}) + "\n")
        except OSError:
            pass
        rows.append(row)
        if save_row is not None:
            try:
                save_row(i, row, rows)
            except Exception:
                pass
        if should_stop is not None and should_stop():
            return rows
    return rows


# What a model is told for good after it got an area wrong in orientation: one
# line each, the habit that area's jobs check. A new model is taught by its own
# mistakes, in its system turn, without any training.
HINTS = {
    "slides": "For a deck, call slides_create, then slides_add once per slide with its title and real points "
              "(never placeholders); to change a deck, read it first with slides_read.",
    "words": "For a document, call words_create, then words_add with the content; "
             "to change one, read it first and edit only the block asked for.",
    "sheets": "For a spreadsheet, call sheets_create with the headers and every row in the same call "
              "(total=true adds a Total row); to change one, read it first with sheets_read.",
    "studio": "For video or audio work, use the Studio tools on a .sstudio project and render it when done.",
    "files": "For a file job, call create_folder, move_file, copy_file or delete_file with the exact paths, "
             "then stop.",
    "chat": "A plain question gets a plain answer in words: no tools, no files.",
    "code": "Code goes in one fenced block in the right language, complete and runnable.",
    "followup": "A follow-up (\"make it shorter\", \"add one more\") changes the file made last; "
                "read it, change it, and do not start a new one.",
}


def hints(rows, most=3):
    """The lines a model is taught from the areas it failed, worst area first."""
    bad = {}
    for r in rows:
        if not r.get("ok") and r.get("area") in HINTS:
            bad[r["area"]] = bad.get(r["area"], 0) + 1
    return [HINTS[a] for a in sorted(bad, key=lambda a: -bad[a])[:most]]


def summary(results):
    """{area: {passed, of, s_avg, fails: {reason: count}}} from [{"area","ok","why","s"}]."""
    out = {}
    for r in results:
        a = out.setdefault(r["area"], {"passed": 0, "of": 0, "s": 0.0, "fails": {}})
        a["of"] += 1
        a["s"] += r.get("s", 0)
        if r["ok"]:
            a["passed"] += 1
        else:
            key = re.sub(r"\d+", "N", r["why"])[:70]
            a["fails"][key] = a["fails"].get(key, 0) + 1
    for a in out.values():
        a["s_avg"] = round(a.pop("s") / max(1, a["of"]), 1)
    return out
