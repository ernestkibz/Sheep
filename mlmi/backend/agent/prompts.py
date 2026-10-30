"""The system turns and short prompt texts the agent sends to models.

Split out of agent.py (2026-10-21); every name is re-exported from the agent package."""
import json
import re


MAX_STEPS = 20


# Llama 3.2 3B, first live run (2026-09-30): the list was made in four calls, then
# it kept going -- a herder that did not exist, and the items recoloured green.
# Each good result now ends with this, so the model stops when the job is done.
DONE_HINT = ("\n(If everything the user asked for is now done, stop calling tools and reply to "
             "the user in one sentence. Do nothing they did not ask for.)")


def fit(convo, level):
    """A conversation shortened to fit the model's context, for any model: level 1
    cuts long tool results, level 2 also keeps only this turn (the system prompt,
    the last request and what followed it). Tool calls keep their results."""
    cap = {1: 1500, 2: 500}.get(level, 250)
    out = [dict(m, content=m["content"][:cap] + "\n... (cut to fit the model's context)")
           if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > cap else m
           for m in convo]
    if level >= 2:
        last = max((i for i, m in enumerate(out) if m.get("role") == "user"), default=0)
        out = [m for m in out[:last] if m.get("role") == "system"] + out[last:]
    if level >= 3:
        # one request whose own tool steps ran over (2026-10-13: a render that wandered
        # through three files): the request and its last few steps, whole call/result pairs
        head = [m for m in out if m.get("role") == "system"]
        user = next((i for i, m in enumerate(out) if m.get("role") == "user"), None)
        rest = out[user + 1:] if user is not None else []
        keep = rest[-4:]
        while keep and keep[0].get("role") == "tool":      # never a result without its call
            keep = keep[1:]
        out = head + ([out[user]] if user is not None else []) + keep
    return out


def too_long(e):
    return "does not fit" in str(e) or "context" in str(e).lower() and "400" in str(e)


SYSTEM = """You are Sheep, the assistant. You work inside the user's local workspace in Sheep's own apps: \
Sheep Words (.sdoc documents), Sheep Sheets (.ssheet workbooks) and Sheep Presentations (.sdeck decks), \
by calling the tools you are given.

Rules:
- To change or create a file, call a tool. Never say a file was changed unless a tool result says so.
- Act, do not interrogate: when details are missing, make sensible choices, do the job, and say in your reply \
what you assumed. Ask only when you truly cannot go on.
- Work like a careful professional: read the source files and any example first; match an example's structure, \
sections and depth, not just its topic; use the real names, dates and figures from the sources; then check what \
you made against the request before you call it done.
- A deck, document or spreadsheet that is asked for is made as a file with the tools, never written out in \
the chat. A question, or a card message, poem or caption, is answered in your reply.
- Work in the Sheep apps. Microsoft files come in with import_file and go out with export_file: if the user \
asks for a Word, Excel or PowerPoint file, make it in Sheep, then export it.
- Paths are relative to the workspace, e.g. "reports/q3.sdoc".
- Read a file before editing one you have not seen in this conversation.
- Video choice: use make_video for text-to-video or to animate a photo, and pass its attached or selected workspace path in source_images. Use Studio image layers and render for a montage/slideshow or when the user wants exact original photos. Follow explicit text-only or photo-use instructions. Never omit a selected photo from an animation request or retry it as text-only if photo guidance fails.
- Use only tools listed for this request; never invent a tool name. For a direct edit to an existing file, \
  use the matching edit tool and change only what the person asked for.
- When the job is done, reply with one or two sentences saying what you made or changed."""


# Ernest, 2026-10-30: thinking for one huge reply is slow and can stop halfway. A big job goes a part at a time
# unless the person says they can wait; both are for any model.
BATCHES = ("This is a big job, so do it in batches. Do the next part properly now, with full real content: a whole sheet, "
           "a whole table, a whole section, or about five slides, in as few calls as you can. Then stop and say in two lines what is done, what comes next, "
           "and ask if you should continue. If the person said which part to do, do that part.")
WHOLE = ("The person said they can wait. Take the time to think it through, then finish the whole job in this turn, "
         "with full real content in every part.")


ANSWER = ("You are Sheep, a helpful assistant running on this computer. Answer the question directly and "
          "accurately in plain words. If you are not sure of a fact, say so instead of guessing. Writing asked "
          "for here (a card message, a poem, an email, ideas) is your answer: write it out in full yourself.")


# What Sheep can do, by measured ability (Orchestrator.abilities), so a new model
# never answers "I'm a text model, I can't make slides" about the app it runs in.
# The same abilities give the same words, so the engine's slot cache still holds.
CAN = [
    (None, "write and edit documents (Sheep Words), spreadsheets (Sheep Sheets) and slide decks (Sheep Presentations)"),
    (None, "import and export Word, Excel and PowerPoint files"),
    (None, "edit video and audio projects in Sheep Studio"),
    (None, "search the web and answer with the sources it read"),
    (None, "change a picture that is here: black and white, sepia, crop, rotate, flip, resize, brightness, contrast, blur"),
    ("reads pictures", "look at a picture the person attaches"),
    ("makes pictures", "make pictures from a description"),
    ("makes video", "make short video clips"),
    ("hears", "turn recorded speech into text"),
    ("speaks", "read text aloud"),
]


def brief(state):
    """One paragraph on what Sheep can do on this computer, and what to add for
    the rest. Built from what was measured, never from what a model claims."""
    orch = getattr(state, "orch", None)
    abl = orch.abilities() if orch and hasattr(orch, "abilities") else {}
    needs = getattr(orch, "NEEDS", {}) if orch else {}
    can = [what for need, what in CAN if need is None or abl.get(need)]
    lacks = [needs[need] for need, _ in CAN if need and abl and not abl.get(need) and need in needs]
    text = ("Sheep, the app you work in, can do these jobs when the person asks: " + "; ".join(can) + ". "
            "If you are asked whether you can do one of them, say yes and offer to do it "
            "(for example: \"Ask me to make a slide deck about goats\"); never say you are only a text model.")
    if lacks:
        text += " Not on this computer yet: " + "; ".join(lacks) + "."
    return text


def taught(state):
    """The lines this model was taught by its own orientation mistakes (orient.HINTS)."""
    orch = getattr(state, "orch", None)
    path = getattr(state, "model_path", None)
    if not (orch and path and hasattr(orch, "entry")):
        return []
    try:
        return list(((orch.entry(path) or {}).get("knows_sheep") or {}).get("hints") or [])
    except Exception:
        return []


def answer_prompt(state):
    """The system turn for a plain question: no tools, but it knows Sheep's reach."""
    return ANSWER + "\n\n" + brief(state)


PLAN_ONLY = """\nPLAN MODE: do not change anything. You may read files. Reply with a short numbered list of \
the steps you would take; the user will approve it before anything is done."""


PLANNER = """You plan jobs for a small local assistant that will carry them out with tools. Write a short \
numbered list of concrete steps (at most 8), each naming the tool to use and the file. No explanations.
Use as few steps as possible: when one call can do the whole thing, it is one step (sheets_create takes the \
headers, the rows and a total row at once; words_add_list adds a whole list). Never add a step that repeats, \
checks or overwrites what an earlier step already did."""


REVIEW = ("You check a draft before it is made into a {noun}. A small model on the person's computer wrote it "
          "for the request below. Check only: does it do what was asked (the subject, and any number of "
          "slides, sections, rows or items named), is anything missing, are there placeholders or "
          "unfinished parts. Reply with JSON only: {{\"ok\": true}} or {{\"ok\": false, \"fixes\": "
          "[\"...\"]}} with at most four short fixes. Never rewrite the draft.")


def review_verdict(text):
    """{"ok", "fixes"} from the checker's reply; None when it gave no readable verdict."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(j, dict) or "ok" not in j:
        return None
    fixes = [str(f).strip()[:160] for f in (j.get("fixes") or []) if str(f).strip()][:4]
    return {"ok": bool(j["ok"]) and not fixes, "fixes": fixes}


RESEARCH = ("Answer the question from the web sources below. Cite every fact with its source number, like [1] or "
            "[2]. If the sources do not answer it, say what they do say and that you could not verify the rest. "
            "Be clear and concise: a few sentences, or a short list.")
