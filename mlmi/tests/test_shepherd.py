"""The model writes, Sheep operates (shepherd.py): reading what a model wrote and
knowing which job a turn asks for. Cases from Ernest's dog-breed screenshots
(2026-10-06) and the orientation runs.

    python -m mlmi.tests.test_shepherd
"""
import sys
import unittest
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("CLI regression script; run python -m mlmi.tests.test_shepherd")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import shepherd as S  # noqa: E402

DOG = """I'll add the new slide to the existing presentation. Here's the updated file: dog-breeds.sdeck

**Slide 1: Introduction**

* Title: "Introduction"
* Text: "Welcome to our presentation on dog breeds!"

**Slide 2: Dog Breeds**

* Title: "Popular Dog Breeds"
* Text: "Here are some popular dog breeds, grouped by category:"
* Bullet points:
    + Sporting Dogs
        - Labrador Retriever
        - Golden Retriever
        - German Shorthaired Pointer
    + Herding Dogs
        - Border Collie
        - Australian Shepherd
        - Shetland Sheepdog
    + Working Dogs
        - Doberman Pinscher
        - Rottweiler
        - Boxer
    + Toy Dogs
        - Chihuahua
        - Poodle
        - Pomeranian

I added the new slide as a new slide in the deck. Let me know if you'd like changes!
"""
fails, ran = [], [0]


def check(name, ok, got=""):
    ran[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   got {got!r}"))
    if not ok:
        fails.append(name)


# reading the 3B's reply: the talk is dropped, a crowded slide splits by its groups
t, sub, slides = S.parse_slides(DOG)
titles = [s["title"] for s in slides]
check("chatter is not the deck title", t == "" and sub == "", (t, sub))
check("one slide per breed group", titles[-4:] == ["Sporting Dogs", "Herding Dogs", "Working Dogs", "Toy Dogs"], titles)
check("breeds are the group's points", slides[-1]["bullets"] == ["Chihuahua", "Poodle", "Pomeranian"], slides[-1]["bullets"])
check("no quote marks left", not any('"' in b for s in slides for b in s["bullets"]))
check("closing chatter is not a point", not any("Let me know" in b for s in slides for b in s["bullets"]))

# plain markdown from the writer
t, sub, slides = S.parse_slides("# Coffee\nFrom bean to cup\n\n## Growing\n- Highlands\n- Shade\n\n## Roasting\n- Light\n- Dark\n")
check("plain deck", (t, sub, [s["title"] for s in slides]) == ("Coffee", "From bean to cup", ["Growing", "Roasting"]))
check("thin slides dropped", len(S.drop_thin(("", "", [{"title": "A", "bullets": ["x"], "notes": ""},
                                                         {"title": "B", "bullets": ["x", "y"], "notes": ""}]))[2]) == 1)
check("repeats dropped", [s["title"] for s in S.drop_repeats(("", "", slides), ["growing"])[2]] == ["Roasting"])

title, blocks = S.parse_doc("# Report\n\nIntro line here.\n\n## Effects\n- one\n- two\n\n| A | B |\n|---|---|\n| 1 | 2 |\n")
check("document blocks", title == "Report" and [b[0] for b in blocks] == ["paragraph", "heading", "bullet", "bullet", "table"],
      (title, [b[0] for b in blocks]))
head, rows = S.parse_table("| Item | Cost |\n|---|---|\n| Rent | 1,200 |\n| Food | $250 |\n| Total | 1450 |")
check("table numbers, no total row", head == ["Item", "Cost"] and rows == [["Rent", 1200], ["Food", 250]], (head, rows))

# which job a turn asks for
ex = lambda p: p in ("dog-breeds.sdeck", "budget.ssheet")
J = lambda text, app="", open_path="", last=(): S.intent(text, app, open_path, list(last), DOG, ex)
j = J("create a ppt for dog breeds")
check("create a deck", j and (j["kind"], j["action"]) == ("slides", "create"), j)
j = J("only has one slide", "slides", "dog-breeds.sdeck", ["dog-breeds.sdeck"])
check("a complaint extends the open deck", j and (j["action"], j["target"], j["count"], j["says"]) ==
      ("extend", "dog-breeds.sdeck", 0, 1), j)
j = J("now add that to the ppt you made before", "slides", "dog-breeds.sdeck", ["dog-breeds.sdeck"])
check("'add that' reuses the last reply", j and j["reuse"] and j["target"] == "dog-breeds.sdeck", j)
check("a precise edit is the tool loop's", J("change the theme to dark", "slides", "dog-breeds.sdeck") is None)
check("a question is not a job", J("What is a presentation?") is None)
check("acknowledgement cannot trigger an artifact action", not S.explicit_change("Thanks, that looks great."))
check("a follow-up question cannot trigger an artifact action", not S.explicit_change("What was the last slide about?"))
check("explicit extension remains an artifact action", S.explicit_change("Add two more slides to that deck."))
check("explicit new artifact remains allowed", S.explicit_change("Create a new deck about cats."))
j = J("Make a workbook called budget.ssheet with Item and Cost and a total")
check("a named create stays a create", j and (j["action"], j["name"], j["total"]) == ("create", "budget.ssheet", True), j)
j = J("Make a deck called 'Trip plan.sdeck' with 4 slides")
check("quoted name with a space", j and (j["name"], j["count"]) == ("Trip plan.sdeck", 4), j)
j = J("make a new deck about cats", "slides", "dog-breeds.sdeck", ["dog-breeds.sdeck"])
check("'new deck' is a create", j and j["action"] == "create", j)

# text and code files, notes, and code asked in the chat (2026-10-06 orientation rows)
j = J("write a text file called todo.txt with three things to do today")
check("a named text file", j and (j["kind"], j["name"], j["count"]) == ("text", "todo.txt", 3), j)
j = J("save a python script called hello.py that prints hello")
check("a named script", j and (j["kind"], j["action"], j["name"]) == ("code", "create", "hello.py"), j)
check("code in the chat is not a file", J("write a python function that reverses a string") is None)
j = J("write meeting notes about time management with five bullet points")
check("meeting notes are a document", j and j["kind"] == "words", j)
ex2 = lambda p: p in ("book.ssheet", "deck.sdeck")
j = S.intent("add a row for Internet costing 60 to book.ssheet", "", "", [], "", ex2)
check("'a row' is one row", j and (j["action"], j["count"]) == ("extend", 1), j)
j = S.intent("add two more slides to deck.sdeck", "", "", [], "", ex2)
check("'two more slides' is two", j and (j["action"], j["count"]) == ("extend", 2), j)
check("a code file's code block", S.parse_code("Here:\n```python\nprint(1)\n```\nDone.") == "print(1)\n")
check("a text file without its talk", S.parse_text("Here is your list:\n1. A\n2. B\nLet me know!") == "1. A\n2. B\n")

# one title is one slide, however the rounds spaced it (2026-10-07: a first-aid
# deck kept two "Provide Care" slides because one had a trailing space)
p = ("Deck", "", [{"title": "Provide Care", "bullets": ["a", "b"]}])
check("a taken title is taken whatever its spacing", not S.drop_repeats(p, ["provide care ", "call help"], "Deck")[2])
p = ("Deck", "", [{"title": "Provide Care ", "bullets": ["a", "b"]}, {"title": "provide care", "bullets": ["c", "d"]}])
m = S.merge_repeats(p)[2]
check("slides written twice under one title merge", len(m) == 1 and m[0]["bullets"] == ["a", "b", "c", "d"], m)

print(f"\n{'all ' + str(ran[0]) + ' checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
