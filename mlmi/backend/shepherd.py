"""The model writes; Sheep operates.

Asked "create a ppt for dog breeds", the 3B made one title slide and stopped.
Asked "only has one slide" and "now add that to the ppt you made before", it
twice wrote a good list of dog breeds and said it had added it, with no tool
called (Ernest's screenshots, 2026-10-06). Small models write well and fumble
multi-step tool calls. So for making and filling decks, documents and sheets,
the orchestrator asks the model only to WRITE (markdown, which every chat model
knows), reads what it wrote, and makes the file itself with the app's tools.
Precise edits (rename, move, format, theme, sort, chart) stay with the tool loop.

    intent(...)      -> a job ({kind, action, target, count, ...}) or None
    writer_prompt(job, outline) -> the system turn that asks for markdown
    parse(kind, md)  -> the structure the markdown describes
    calls(job, parsed, ws) -> the tool calls that make or extend the file
"""
import re

KINDS = {
    "slides": {"ext": ".sdeck", "office": ".pptx", "noun": "deck",
               "words": r"\b(ppt|pptx|power ?points?|decks?|slides?|slideshow|presentations?|pitch)\b"},
    "sheets": {"ext": ".ssheet", "office": ".xlsx", "noun": "workbook",
               "words": r"\b(xlsx|excel|spreadsheets?|workbooks?|budget|tracker|ledger|timesheet|sheet)\b"},
    "words": {"ext": ".sdoc", "office": ".docx", "noun": "document",
              "words": r"\b(docx?|documents?|report|essay|letter|brief|article|memo|proposal|cv|resume|story|"
                       r"word (file|document)|write-?up|guide|handout|summary document|"
                       # notes and lists to keep: "write meeting notes with five bullet points" went to
                       # the tool loop and made nothing in 263 s (2026-10-06 orientation)
                       r"(meeting )?notes|minutes|agenda|outline|checklist|to-?do list|reading list|lesson plan)\b"},
    # plain text and code files: the model writes the content, Sheep saves it. Put inside a
    # tool call, the 3B's text came out with literal \n escapes and garbage (2026-10-06)
    "text": {"ext": ".txt", "office": "", "noun": "file",
             "words": r"\b(text file|txt file|notes file|markdown file|plain text file)\b"},
    "code": {"ext": ".py", "office": "", "noun": "script", "words": r"(?!x)x"},     # only by a named file
}
EXT_KIND = {".sdeck": "slides", ".pptx": "slides", ".ssheet": "sheets", ".xlsx": "sheets", ".sdoc": "words",
            ".docx": "words", ".txt": "text", ".md": "text", ".csv": "text", ".py": "code", ".js": "code",
            ".html": "code", ".css": "code", ".sh": "code", ".sql": "code", ".ps1": "code", ".json": "code"}
FILE_EXT = r"(sdeck|pptx|ssheet|xlsx|sdoc|docx|txt|md|csv|py|js|html|css|sh|sql|ps1|json)"
CREATE = (r"\b(make|create|write|draft|build|prepare|generate|produce|design|put together|compose|give me|"
          r"i need|i want|can you (make|create|write|do|build))\b")
EXTEND = (r"\b(add|append|insert|include|expand|extend|more|another|continue|fill|complete|finish|"
          r"only (has|have|got|had|shows?)|too (short|few|little|small|thin)|missing|empty|incomplete|"
          r"put (that|this|it|them|those|these))\b")
REUSE = (r"\b(that|this|those|these|the above|what you (wrote|said|made|listed|gave)|your (answer|list|reply|text)|"
         r"the (list|content|text|points|breeds|items) (you|above))\b")
# precise edits: one tool each, the tool loop does them
PRECISE = (r"\b(agenda (?:slide|page)|rename|delete|remove|move|reorder|swap|bold|italic|underline|colou?r|font|theme|sort|chart|graph|"
           # "make the title bigger in save-water.sdeck" made a NEW deck (2026-10-14 cue audit)
           r"bigger|smaller|larger|wider|narrower|bolder|align\w*|cent(er|re)(ed)?|resize|"
           r"format|export|import|read|open|summari[sz]e|translate|replace|correct|set (the|cell)|change the|"
           r"title of slide|slide \d+|block \d+|cell [a-z]+\d+|undo|redo|number the)\b")
CHANGE_REQUEST = (r"\b(make|create|write|draft|build|prepare|generate|produce|design|compose|add|append|insert|"
                  r"include|expand|extend|edit|change|update|delete|remove|rename|move|copy|export|replace|fix|"
                  r"save|render|shorten|lengthen|continue|complete|finish|put together|i need|i want|"
                  # pictures, sound and photo edits are asks to make something too (2026-10-14 live
                  # run: "Draw me a picture of a dog" was refused as "did not ask to create")
                  r"draw|paint|sketch|illustrate|animate|narrate|speak|record|transcribe|convert|"
                  r"turn (it|this|that) into|crop|resize|rotate|flip|blur|sharpen|brighten|darken|"
                  r"grayscale|greyscale|black and white)\b|"
                  r"\b(read|say) (it|this|that|the \w+ |this \w+ )?(out loud|aloud)\b|\bin a voice\b|"
                  r"\b(picture|photo|image|drawing|painting|illustration|poster|logo|icon) of\b|"
                  r"\b(only (has|have|got|had|shows?)|too (short|few|little|small|thin)|missing|incomplete)\b")
# A file named by kind and subject with no verb: "powerpoint about legend of seeker" got an offer and no file
# (2026-10-20). Strong file nouns at the start only, so "read me a story about a lion" stays a chat answer.
FILE_NOUNS = (r"(?:ppt|pptx|power ?points?|decks?|slides?|slideshows?|presentations?|spreadsheets?|workbooks?|"
              r"excel(?: file| sheet)?|documents?|docs?|docx|word (?:file|document)|reports?)")
NOUN_LED = (r"^\s*(?:please\s+)?(?:(?:i\s+)?(?:want|need|would like|'d like)\s+)?(?:(?:an?|the|some|one|my)\s+)?"
            r"(?:[\w-]+\s+){0,2}?" + FILE_NOUNS + r"\s+(?:about|on|for|of|covering|regarding|explaining)\s+\S")
CLOSERS = (r"^\s*(?:thanks?|thank you|thx|ty|great|good|nice|cool|perfect|awesome|looks? (?:good|great)|"
           r"that'?s (?:it|all|fine|good|great)|never ?mind|no|nope|stop|cancel|bye)\b")
OPT_OUT = r"\b(?:in (?:the|this) chat|as text|just tell me|no file|don'?t make (?:a|the) file|outline only)\b"
AFFIRM = r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|go ahead|do it|please do|go on)\b"
CHOICE = r"^\s*(?:your (?:choice|call|pick)|you (?:choose|decide|pick)|up to you|anything|whatever|surprise me)\b"
FORMAT_WORD = r"\b(?:pptx|docx|xlsx|power ?point|excel|word (?:file|document))\b"
QUESTION = r"^\s*(?:what|why|how|when|where|who|which|whose|is|are|was|were|can|could|do|does|did|should|would|will)\b"
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
NUM = r"(\d{1,2}|" + "|".join(NUM_WORDS) + r")"
UNITS = {"slides": r"(slides?|pages?)", "words": r"(sections?|paragraphs?|parts?|points?|pages?)",
         "sheets": r"(rows?|items?|entries|lines?)", "text": r"(lines?|items?|points?|things|tasks?)",
         "code": r"(functions?|lines?)"}
THEMES = ("light", "dark", "midnight", "forest", "sunset", "mono")


def _n(w):
    return int(w) if w.isdigit() else NUM_WORDS.get(w, 0)


def structured(text):
    """Does a reply hold content worth putting in a file (a list, headings or a table)?"""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    marks = sum(bool(re.match(r"\s*([-*+•]|\d+[.)]|#{1,4}\s|\|)", l)) for l in lines)
    return marks >= 3


def explicit_change(text):
    """Whether this turn itself asks Sheep to create, extend, render or edit an artifact."""
    text = (text or "").lower().strip()
    if re.match(r"^(what|why|when|where|who|how|is|are|was|were|did|does|do)\b", text):
        return False
    return bool(re.search(CHANGE_REQUEST, text))


# "How do I make a chart in Excel?" asks how, not for a chart: it made a new workbook
# (2026-10-14 cue audit). A how-to or what-is question naming no file of this workspace is
# answered in words.
HOWTO = (r"^\s*(how (do|can|should|would|could) (i|you|we|one|someone)|how to|how does|what('?s| is) the (best|right|"
         r"easiest|quickest) way to|what('?s| is) an?|what are|is there a way to|why (does|do|is|are|can't|won't))\b")


def intent(asked, app="", open_path="", last_files=(), prev_answer="", exists=lambda p: False):
    """The job this turn asks of a deck, document or sheet, or None (a question,
    a precise edit, or another app's work).

    open_path: the file open in the editor; last_files: files this chat made or
    named, newest last; prev_answer: the assistant's last reply; exists(path)."""
    t = (asked or "").lower().strip()
    if not t:
        return None
    if re.search(HOWTO, t) and (re.match(r"\s*how\b", t) or not re.search(r"\.(" + FILE_EXT[1:-1] + r"|sstudio)\b", t)):
        return None          # "How do I add a row to budget.ssheet?" is asked, not ordered
    named = [n[0] for n in re.findall(r"[\"']([^\"']+?\." + FILE_EXT + r")[\"']", asked or "", re.I)]
    named += [n[0] for n in re.findall(r"(?<![\w\"'])([\w./-]+\." + FILE_EXT + r")\b", asked or "", re.I)
              if n[0] not in " ".join(named)]
    kind = None
    for n in named:
        kind = EXT_KIND.get("." + n.rsplit(".", 1)[-1].lower())
        if kind:
            break
    if not kind:
        hits = [k for k, v in KINDS.items() if re.search(v["words"], t)]
        if app in KINDS and (not hits or app in hits):
            kind = app
        elif hits:
            kind = hits[0]
    open_kind = EXT_KIND.get("." + open_path.rsplit(".", 1)[-1].lower()) if open_path and "." in open_path else None
    if not kind and open_kind and re.search(EXTEND, t):
        kind = open_kind                      # "only has one slide" with the deck open
    if not kind:
        return None
    if re.search(r"\b(add|insert|put|place|attach)\b", t) and re.search(r"\.(png|jpe?g|webp|bmp|gif)\b", t):
        return None          # "add photo.jpg to trip-plan.sdoc": the picture tool's, not new writing
    if kind in ("text", "code") and not (re.search(CREATE + r"|\b(save|put)\b|" + EXTEND, t) and (named or kind == "text")):
        return None                   # code asked in the chat is answered with a code block, not a file
    creating = bool(re.search(CREATE, t)) or (kind in ("text", "code") and bool(re.search(r"\bsave\b", t)))
    creating = creating or bool(re.search(NOUN_LED, t))
    extending = bool(re.search(EXTEND, t))
    # When the user explicitly asks to put the preceding answer into an artifact,
    # reuse that answer even if the model formatted it as prose instead of a list.
    # The previous markdown-only gate made "put that in a document" fall through
    # to ordinary chat for otherwise useful, unstructured answers.
    move_answer_to_file = bool(re.search(r"\bput\s+(?:that|this|it)\s+(?:in|into)\b", t))
    reuse = bool(re.search(REUSE, t)) and (structured(prev_answer) or (move_answer_to_file and bool(prev_answer.strip())))
    if re.search(PRECISE, t) and not (creating and not named):
        # "change the theme", "rename slide 2", "sort by cost": the tool loop's work
        if not (extending and reuse):
            return None
    if not (creating or extending or reuse):
        return None
    # the file this works on
    target = ""
    for n in named:
        # "make a workbook called budget.ssheet" stays a create even when it exists:
        # the create tool then says so, and nothing is overwritten
        if EXT_KIND.get("." + n.rsplit(".", 1)[-1].lower()) == kind and exists(n) and (extending or reuse
                                                                                      or not creating):
            target = n
            break
    if not target and (extending or reuse) and not (creating and re.search(r"\b(new|another|second|separate)\s+(?:"
                                                                       + KINDS[kind]["words"] + ")", t)):
        if open_kind == kind:
            target = open_path
        else:
            for f in reversed(list(last_files)):
                if EXT_KIND.get("." + f.rsplit(".", 1)[-1].lower()) == kind and exists(f):
                    target = f
                    break
    action = "extend" if target else "create"
    # "only has one slide" is a complaint, not a count
    m = re.search(r"(?<!only has )(?<!only have )(?<!only got )(?<!only had )(?<!just )(?<!only )\b" + NUM
                  + r"[- ](?:content |more |new |extra |additional |short )?" + UNITS[kind] + r"\b", t)
    count = _n(m.group(1)) if m else 0
    if not count and action == "extend":
        # "add a row for Internet": one (read as six rows, the asked row was lost; 2026-10-06)
        m1 = re.search(r"(?<!only has )(?<!only have )(?<!only got )(?<!only had )(?<!just )(?<!only )"
                       r"\b(a|an|one|another) (new |more |extra |single )?" + UNITS[kind] + r"\b", t)
        count = 1 if m1 else 0
    theme = next((th for th in THEMES if re.search(r"\b" + th + r"\b", t)), "") if kind == "slides" else ""
    office = KINDS[kind]["office"]
    want_office = bool(office) and bool(re.search(r"\b(" + office[1:] + r"|" + {
        "slides": "power ?point", "sheets": "excel", "words": "word (file|document)"}[kind] + r")\b", t))
    name = next((n for n in named if EXT_KIND.get("." + n.rsplit(".", 1)[-1].lower()) == kind), "")
    # "only has one slide": what the person sees, checked against the file before anything is added
    m = re.search(r"\bonly (?:has|have|got|had|shows?) " + NUM + r" " + UNITS[kind] + r"\b", t)
    return {"kind": kind, "action": action, "target": target, "count": count, "reuse": reuse, "theme": theme,
            "office": want_office, "name": name, "total": bool(re.search(r"\b(total|sum)\b", t)),
            "says": _n(m.group(1)) if m else 0, "pdf": bool(re.search(r"\bpdf\b", t))}


def follow_on(messages, exists=lambda p: False):
    """A short turn read with the ones before it. After a file request that made no file, it is that request
    again with this detail; after a made deck/document/sheet, a format word exports it and a short subject adds
    to it. None: read the turn alone. (Ernest's chat, 2026-10-20.)"""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) < 3 or turns[-1].get("role") != "user":
        return None
    now = str(turns[-1].get("content") or "").strip()
    low = now.lower()
    if (not now or len(now.split()) > 6 or now.endswith("?") or re.search(CLOSERS, low)
            or re.search(OPT_OUT, low) or re.search(QUESTION, low)):
        return None
    last = turns[-2]
    native = lambda m: [f for f in (m.get("files") or []) if f.lower().endswith((".sdeck", ".sdoc", ".ssheet"))]
    made = native(last)
    if not made and re.search(FORMAT_WORD, low):
        # "pptx" a few turns after the deck was made, with a turn in between that made nothing
        # (Ernest's chat, 2026-10-21: it described a deck in the chat instead)
        made = next((native(m) for m in reversed(turns[-8:-1]) if m.get("role") == "assistant" and native(m)), [])
    if made:
        f = made[-1]
        kind = EXT_KIND["." + f.rsplit(".", 1)[-1].lower()]
        if re.search(FORMAT_WORD, low):
            return f"export {f} to {KINDS[kind]['office'][1:]}"
        if (re.search(AFFIRM, low) or re.search(CHOICE, low) or re.search(r"\bfiles?\b", low)
                or any(re.search(v["words"], low) for v in KINDS.values())):
            return None
        return {"slides": f"add two slides about {now} to {f}", "words": f"add a section about {now} to {f}"}.get(kind)
    details = [now]
    for m in reversed(turns[:-1]):
        if m.get("role") == "assistant":
            if m.get("files"):
                return None
            continue
        text = str(m.get("content") or "").strip()
        j = intent(text, "", "", [], "", exists)
        if j and j["action"] == "create":
            if re.search(AFFIRM, low) and not re.search(r"\?\s*$|would you like|shall i|do you want",
                                                       str(last.get("content") or ""), re.I):
                return None
            return text + " — " + "; ".join(details)
        if len(details) >= 3:
            return None
        details.insert(0, text)
    return None


# The person handing the job back ("just do it", "you decide", "stop asking"): the request stands (Ernest,
# 2026-10-30: "can you just do the thinking" after Nemotron's questions made no file, on any model)
HANDBACK = (r"\b(?:do (?:it|that|this|the thinking|your best|what you (?:can|think))|go ahead|carry on|proceed|"
            r"you (?:decide|choose|pick|figure)|your (?:choice|call|pick)|up to you|figure it out|"
            r"use your (?:judge?ment|best)|whatever you (?:think|want|like)|"
            r"(?:don'?t|dont|stop|no need to|quit) (?:ask|question)|no (?:more )?questions|"
            r"(?:don'?t|dont) (?:feel like answering|care|mind)|any(?:thing)? (?:is|will) (?:fine|do|work)|"
            r"defaults? (?:is|are) fine|make (?:it|them|something) up|surprise me)\b")
RETRY = r"\b(?:try (?:it )?again|retry|again|once more|one more time|redo|another (?:one|go|try)|keep going|finish it)\b"
FAILED = r"\b(?:could not|couldn'?t|failed|nothing was changed|was not (?:made|done)|did not finish)\b"
# A generic "let me know if there's anything else" is not a pending job question.
ASKED_BACK = (r"\?\s*$|\?\s*\n|could you (?:tell|share|confirm)|once i have|"
              r"please (?:tell|confirm|share)|which (?:one )?would you")
ASKING = r"^\s*(?:what|why|how|who|whom|where|when|which|whose)\b"
MEDIA_MADE = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".wav", ".mp3")


# How much a job takes at once (Ernest, 2026-10-30): a big job goes in batches the person can steer, unless they
# say they can wait, and then the model thinks as long as it needs and finishes it in one go
CAN_WAIT = (r"\b(i can wait|i'?ll wait|i don'?t mind waiting|take (?:your|the|as much) time|no rush|however long|"
            r"all at once|all in one go|in one go|everything at once|the whole thing at once)\b")
BIG_JOB = (r"\b(tracker|dashboard|work ?plan|workplan|\d+[- ]day|proposal|business plan|strategy|handbook|manual|"
           r"playbook|roadmap|full report|detailed|comprehensive|complete|in depth|in-depth|whole|entire|"
           r"like (?:the|this|that|our|my)|equivalent|based on|template|\d{2,} (?:slides?|sections?|pages?|rows?))\b")
NEXT = r"^\s*(?:yes|yeah|yep|ya|y|sure|ok(?:ay)?|continue|next|go on|go ahead|carry on|keep going|proceed|do the next|please do)\b"


def batch_of(text, attached=0):
    """"whole": the person can wait; "batches": a big job, a part at a time; "": a small job, done at once."""
    t = (text or "").lower().split("\n\nattached")[0]
    if re.search(CAN_WAIT, t):
        return "whole"
    return "batches" if re.search(BIG_JOB, t) or attached >= 2 else ""


def carried(messages, alone=lambda t: False, works=False, asks=lambda t: False):
    """The request a turn belongs to, for every kind of work, or None to read the turn by itself: a hand-back
    or a retry is the last request again, an answer to the assistant's own question is that request with the
    answer, and a short change after a picture, video or recording is that request changed. alone(text): the
    turn is a whole new job. works: the turn already asks for work by itself. asks(text): an earlier turn asked
    for work. {"text", "attachments"}."""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) < 3 or turns[-1].get("role") != "user" or turns[-2].get("role") != "assistant":
        return None
    now = str(turns[-1].get("content") or "").strip()
    low = now.lower()
    if not now or len(now.split()) > 24 or re.search(ASKING, low) or re.search(OPT_OUT, low) or alone(now):
        return None
    last, handback = turns[-2], bool(re.search(HANDBACK, low))
    if re.search(CLOSERS, low) and not handback:
        return None
    said = str(last.get("content") or "")
    files = last.get("files") or []
    failed = not files and bool(re.search(FAILED, said, re.I))
    asked_back = not files and bool(re.search(ASKED_BACK, said[-700:], re.I))
    retry = bool(re.search(RETRY, low)) and (failed or bool(files))
    # "yes" after a batch and "shall I go on?": the same request, on the files it has made so far
    onward = (bool(files) and bool(re.search(NEXT, low))
              and bool(re.search(ASKED_BACK + r"|\bnext\b\s*(?:step|part|up|batch)?\s*:", said[-600:], re.I)))
    if onward:
        retry = True
    question = now.endswith("?") or bool(re.search(QUESTION, low))
    change = (any(str(f).lower().endswith(MEDIA_MADE) for f in files) and not works and not question
              and len(now.split()) <= 12)
    if not (handback or retry or change or (asked_back and not question)):
        return None
    details, attached = ([] if handback or retry else [now]), []
    for m in reversed(turns[:-1]):
        if m.get("role") != "user":
            continue
        text = str(m.get("content") or "").strip()
        attached = list(m.get("attachments") or []) + attached
        if not (re.search(HANDBACK, text.lower()) or re.search(RETRY, text.lower())):
            # "need a version of this" with files attached names no verb: the files say it is work
            if (explicit_change(text) or re.search(NOUN_LED, text.lower()) or asks(text)
                    or (m.get("attachments") and not just_talk(text))):
                if onward:
                    details = ["done so far: " + ", ".join(str(f) for f in files) + "; carry on with the next part in those "
                               "files, reading them first"]
                return {"text": " — ".join([text] + details), "attachments": list(dict.fromkeys(attached))}
            if not handback and not retry and len(details) < 3:
                details.insert(0, text)
    return None


SOCIAL_CHAT = (r"\b(?:i\s+)?(?:need|want)\s+(?:some\s*one|somebody|a person|a friend)\s+to\s+talk\b|"
              r"\b(?:i feel|i'm feeling|i am feeling)\s+(?:lonely|alone|sad|down|overwhelmed)\b|"
              r"\b(?:can|could)\s+i\s+talk\s+to\s+you\b|"
              r"\b(?:talk|chat)\s+(?:to|with)\s+me\s+like\s+(?:a\s+)?friend\b|"
              r"\b(?:be|act like)\s+(?:my|a)\s+friend\b")


def social_chat(text):
    """A personal conversation request, not a continuation of the previous artifact job."""
    return bool(re.search(SOCIAL_CHAT, str(text or ""), re.I))


def just_talk(text):
    """Is this turn only conversation: a greeting, thanks, a question, or "no file"? Anything else may become
    work if the model judges so; the model chooses the tools (Ernest, 2026-10-30)."""
    t = (text or "").lower().strip()
    return (not t or social_chat(t) or bool(re.search(CLOSERS, t) or re.search(OPT_OUT, t) or re.search(QUESTION, t)
                          or re.search(HOWTO, t) or re.match(r"(?:hi|hey|hello|hell|yo|morning|good (?:morning|evening))\b", t))
            or t.endswith("?"))


# ------------------------------------------------------------------ the writing turn
WRITE = {
    "slides": ("Write a slide deck as markdown and nothing else.\n"
               "Line 1: # <deck title>\nLine 2: <a one-line subtitle>\n"
               "Then each slide:\n## <slide title>\n- <point>\n- <point>\n- <point>\n"
               "Give each slide 3 to 5 short points with real facts. No placeholders, no notes about what you did."),
    "words": ("Write the document as markdown and nothing else.\n"
              "Line 1: # <title>\nThen the body: ## section headings, paragraphs, - bullet lists, 1. numbered lists, "
              "and | tables | when they help.\nReal, specific content. No placeholders, no notes about what you did."),
    "sheets": ("Write the data as one markdown table and nothing else:\n| <column> | <column> |\n|---|---|\n"
               "| <value> | <value> |\nNumbers as plain numbers, with no currency signs or thousands commas. "
               "No total row: Sheep adds totals. Use exactly the names and values the person gives; "
               "otherwise real, specific values."),
    "text": ("Write the content of the file as plain text and nothing else: no line about the file itself, no "
             "introduction, no notes about what you did. Real, specific content, no placeholders."),
    "code": "Write the complete code for the file in one code block and nothing else. It must run as it is.",
}


def writer_prompt(job, outline=""):
    kind, n = job["kind"], job["count"]
    ext = job["action"] == "extend"
    if kind == "slides":
        # the shape spelled out with one heading per slide wanted: told "write 2 slides",
        # the 3B wrote one, titled like the deck (2026-10-06 orientation)
        want = n or (4 if ext else 5)
        head = "" if ext else "# <deck title>\n<a one-line subtitle>\n\n"
        body = "\n\n".join(f"## <title of {'new ' if ext else ''}slide {i + 1}>\n- <point>\n- <point>\n- <point>"
                           for i in range(min(want, 8)))
        if want > 8:
            body += f"\n\n...and so on, until slide {want}."
        s = (f"Write {'the ' + str(want) + ' new slides' if ext else 'a slide deck'} as markdown in exactly this shape "
             "and nothing else:\n\n" + head + body + "\n\nEach slide has its own different title and 3 to 5 short "
             "points with real facts. No placeholders, no notes about what you did.")
        if job.get("taken"):
            s += "\nThese titles are already in the deck, so do not use them: " + "; ".join(job["taken"][:20]) + "."
        if ext:
            s += "\n\nThe deck so far:\n" + (outline or "")[:3000]
        return s
    s = WRITE[kind]
    if kind == "words" and n:
        s += f"\nWrite {n} sections."
    elif kind == "sheets":
        if job.get("columns"):
            s += "\nThe table's columns are: " + " | ".join(map(str, job["columns"])) + ". Start with that header row."
        s += f"\nWrite {n or (3 if ext else 8)} row{'s' if (n or 2) != 1 else ''}."
    elif kind == "text" and n:
        s += f"\nWrite {n} lines."
    if ext:
        s += ("\n\nThis adds to an existing " + KINDS[kind]["noun"] + ". Write ONLY the new part, in the same format "
              "(no # title line). Do not repeat what is already there:\n" + (outline or "")[:3000])
    if job.get("source_answer"):
        request = ("\nOriginal request: " + str(job.get("source_request") or "")[:1000]
                   if job.get("source_request") else "")
        s += ("\n\nTurn the preceding assistant answer into this artifact. Preserve its meaning and useful details; "
              "organize the content to match the format above. Do not add unsupported facts."
              + request + "\nPrevious assistant answer:\n" + str(job["source_answer"])[:4000])
    return s


# ------------------------------------------------------------------ reading what was written
def _clean(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s.strip())
    s = re.sub(r"^\*(.+)\*$", r"\1", s)
    s = s.strip().strip('"“”').strip()
    return s.rstrip(":").strip()


LABEL = re.compile(r"^(bullet points?|points?|content|key points?|details|text|body)\s*:?\s*$", re.I)


CHATTER = re.compile(r"^(i\b|i'[a-z]+|we'?ll|here'?s|here is|here are|since|sure|okay|ok\b|certainly|of course|"
                     r"let me|let's|this (is|will)|below|feel free|if you|note:|hope|great|done|you can)", re.I)


def content_lines(md):
    """The reply without its talk: lines that chat about the work ("I'll add the
    new slide...", "Let me know...") before the first heading or list, and after
    the last, are not content."""
    lines = (md or "").splitlines()
    mark = lambda l: bool(re.match(r"\s*([-*+•]|\d+[.)]|#{1,4}\s|\||\*\*[^*]+\*\*:?\s*$)", l))
    idx = [i for i, l in enumerate(lines) if mark(l)]
    if not idx:
        return lines
    first, last = idx[0], idx[-1]
    head = [l for l in lines[:first] if l.strip() and not CHATTER.match(l.strip()) and not l.strip().endswith(":")]
    tail = [l for l in lines[last + 1:] if l.strip() and not CHATTER.match(l.strip())]
    return head + lines[first:last + 1] + tail


def parse_slides(md):
    """(title, subtitle, [{title, bullets: [(depth, text)], notes}])."""
    title = subtitle = ""
    slides, cur = [], None
    for raw in content_lines(md):
        line = raw.rstrip()
        s = line.strip()
        if not s or s.startswith("```") or re.match(r"^[-=_*]{3,}$", s):
            continue
        m = re.match(r"^#\s+(.+)", s)
        if m and not title and not slides:
            title = _clean(m.group(1))
            continue
        m = (re.match(r"^#{1,4}\s+(.+)", s) or re.match(r"^\*\*(.+?)\*\*:?$", s)
             or re.match(r"^(slide\s*\d+\s*[:.\-–]\s*.+)$", s, re.I))
        if m:
            head = re.sub(r"^slide\s*\d+\s*[:.\-–]?\s*", "", _clean(m.group(1)), flags=re.I) or _clean(m.group(1))
            cur = {"title": head, "bullets": [], "notes": ""}
            slides.append(cur)
            continue
        m = re.match(r"^(notes?|speaker notes?)\s*:\s*(.+)", s, re.I)
        if m and cur:
            cur["notes"] = _clean(m.group(2))
            continue
        m = re.match(r"^(\s*)([-*+•]|\d+[.)])\s+(.+)", line)
        if m:
            text = _clean(m.group(3))
            if not text or LABEL.match(text):
                continue
            k = re.match(r"^(title|heading)\s*:\s*(.+)", text, re.I)
            if k:
                if cur and not cur["bullets"]:
                    cur["title"] = _clean(k.group(2))
                    continue
                cur = {"title": _clean(k.group(2)), "bullets": [], "notes": ""}
                slides.append(cur)
                continue
            text = _clean(re.sub(r"^(text|subtitle|point)\s*:\s*", "", text, flags=re.I))
            if cur is None:
                cur = {"title": title or "Overview", "bullets": [], "notes": ""}
                slides.append(cur)
            cur["bullets"].append((len(m.group(1).expandtabs(4)), text))
            continue
        if LABEL.match(s):
            continue
        k = re.match(r"^(title|heading)\s*:\s*(.+)", s, re.I)
        if k:
            if cur and not cur["bullets"]:
                cur["title"] = _clean(k.group(2))
            else:
                cur = {"title": _clean(k.group(2)), "bullets": [], "notes": ""}
                slides.append(cur)
            continue
        k = re.match(r"^(text|subtitle|point)\s*:\s*(.+)", s, re.I)
        if k and cur is not None:
            cur["bullets"].append((0, _clean(k.group(2))))
            continue
        if not slides and title and not subtitle:
            subtitle = _clean(s)
        elif cur is not None:
            cur["bullets"].append((0, _clean(s)))
        elif not title:
            title = _clean(s)
    out = []
    for sl in slides:
        levels = sorted({ind for ind, _ in sl["bullets"]})
        items = [(levels.index(ind), txt) for ind, txt in sl["bullets"]]
        has_child = lambda i: i + 1 < len(items) and items[i + 1][0] > items[i][0]
        # a slide too full to read: each point with sub-points becomes its own slide
        split = next((r for r in range(len(levels)) if sum(1 for i, (d, _) in enumerate(items)
                                                           if d == r and has_child(i)) >= 2), None)
        if len(items) > 8 and split is not None:
            keep = [txt for i, (d, txt) in enumerate(items) if d < split or (d == split and not has_child(i))]
            if keep:
                out.append({"title": sl["title"], "bullets": keep, "notes": sl["notes"]})
            for i, (d, txt) in enumerate(items):
                if d == split and has_child(i):
                    kids, j = [], i + 1
                    while j < len(items) and items[j][0] > split:
                        kids.append(("  " if items[j][0] > split + 1 else "") + items[j][1])
                        j += 1
                    out.append({"title": txt, "bullets": kids, "notes": ""})
            continue
        out.append({"title": sl["title"], "bullets": [("  " * min(d, 1)) + txt for d, txt in items],
                    "notes": sl["notes"]})
    # the format hint copied word for word: the 3B titled its slides "<title of new slide 1>"
    # (B10 replay, 2026-10-21); such a slide is titled from its first real point. "<point>" bullets
    # are left for the placeholder check (agent.repair_slide_placeholders), which rewrites them
    for sl in out:
        if PLACEHOLDER.fullmatch(sl["title"].strip()):
            first = next((b.strip() for b in sl["bullets"] if not PLACEHOLDER.fullmatch(b.strip())), "")
            words = re.sub(r"[.:;,!]+$", "", first.split(":")[0] if ":" in first[:60] else first).split()
            sl["title"] = " ".join(words[:7])
            if sl["title"] and sl["bullets"] and sl["title"] == first:
                sl["bullets"] = sl["bullets"][1:]
    return title, subtitle, [s for s in out if s["title"] or s["bullets"]]


PLACEHOLDER = re.compile(r"<[^<>]{0,60}>")


def _row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return [_clean(c) for c in cells]


def parse_doc(md):
    """(title, [block]): block = ("heading", level, text) | ("paragraph", 0, text) |
    ("bullet"|"numbered", level, text) | ("quote", 0, text) | ("table", 0, rows)."""
    title, blocks, para, table = "", [], [], []

    def flush():
        if para:
            blocks.append(("paragraph", 0, " ".join(para)))
            para.clear()
        if table:
            rows = [r for r in table if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
            if rows:
                blocks.append(("table", 0, rows))
            table.clear()
    for raw in content_lines(md):
        line = raw.rstrip()
        s = line.strip()
        if s.startswith("```"):
            continue
        if not s:
            flush()
            continue
        if s.startswith("|"):
            if para:
                blocks.append(("paragraph", 0, " ".join(para)))
                para.clear()
            table.append(_row(s))
            continue
        if table:
            flush()
        m = re.match(r"^(#{1,4})\s+(.+)", s)
        if m:
            flush()
            if len(m.group(1)) == 1 and not title and not blocks:
                title = _clean(m.group(2))
            else:
                blocks.append(("heading", max(1, min(3, len(m.group(1)) - 1)), _clean(m.group(2))))
            continue
        m = re.match(r"^(\s*)([-*+•])\s+(.+)", line)
        if m:
            flush()
            blocks.append(("bullet", min(2, len(m.group(1).expandtabs(4)) // 2), m.group(3).strip()))
            continue
        m = re.match(r"^(\s*)\d+[.)]\s+(.+)", line)
        if m:
            flush()
            blocks.append(("numbered", min(2, len(m.group(1).expandtabs(4)) // 2), m.group(2).strip()))
            continue
        if s.startswith(">"):
            flush()
            blocks.append(("quote", 0, s.lstrip("> ").strip()))
            continue
        if not title and not blocks and not para and len(s) < 90 and not s.endswith("."):
            title = _clean(s)
            continue
        para.append(s)
    flush()
    return title, blocks


def _num(v):
    t = re.sub(r"[,$€£\s]|UGX|USD|KES", "", v or "")
    if re.fullmatch(r"-?\d+(\.\d+)?%?", t):
        return float(t[:-1]) / 100 if t.endswith("%") else (int(t) if re.fullmatch(r"-?\d+", t) else float(t))
    return v


def parse_table(md):
    """(headers, rows) from the first markdown table, or from 'a, b' lines."""
    rows = [_row(l) for l in (md or "").splitlines() if l.strip().startswith("|")]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
    if not rows:
        lines = [l.strip() for l in (md or "").splitlines() if l.count(",") >= 1 and not l.strip().startswith("#")]
        rows = [[c.strip() for c in l.split(",")] for l in lines]
    if not rows:
        return [], []
    head, body = rows[0], rows[1:]
    body = [r for r in body if r and not re.match(r"^\s*(total|sum|grand total)\b", r[0], re.I)]
    width = len(head)
    body = [(r + [""] * width)[:width] for r in body]
    return head, [[_num(c) for c in r] for r in body]


def parse_text(md):
    """A text file's content: the reply without its talk and code fences."""
    lines = [l for l in content_lines(md) if not l.strip().startswith("```")]
    return "\n".join(lines).strip() + "\n"


def parse_code(md):
    """A code file's content: the first code block, or the whole reply when it has none."""
    m = re.search(r"```[\w+.-]*[ \t]*\n([\s\S]*?)```", md or "")
    return (m.group(1) if m else (md or "")).strip() + "\n"


def parse(kind, md):
    if kind == "slides":
        return parse_slides(md)
    if kind == "words":
        return parse_doc(md)
    if kind == "text":
        return parse_text(md)
    if kind == "code":
        return parse_code(md)
    return parse_table(md)


def enough(kind, parsed, action="create"):
    """Did the writing give a file worth making? (else the model is asked again)"""
    if kind == "slides":
        return len(parsed[2]) >= 1
    if kind == "words":
        return len(parsed[1]) >= 1
    if kind in ("text", "code"):
        return len(parsed.strip()) > 0
    return bool(parsed[0]) and len(parsed[1]) >= 1


def reuse_needs_reformat(kind, answer, parsed):
    """Whether a reused chat answer needs shaping before it becomes an artifact."""
    return (kind in ("slides", "words", "sheets") and bool((answer or "").strip())
            and (not structured(answer) or not enough(kind, parsed)))


# ------------------------------------------------------------------ the tool calls
def slug(text, ext):
    s = "-".join(re.findall(r"[a-z0-9]+", (text or "").lower())[:5]) or "untitled"
    return s + ext


def new_path(job, title, exists):
    k = KINDS[job["kind"]]
    if job["name"]:
        base = job["name"]
    else:
        base = slug(title, k["office"] if job["office"] else k["ext"])
    stem, ext = base.rsplit(".", 1)
    p, i = base, 2
    native = lambda q: q.rsplit(".", 1)[0] + k["ext"]
    while exists(native(p)) and not job["name"]:
        p = f"{stem}-{i}.{ext}"
        i += 1
    return p


def last_row(ws, path):
    """The last used row of a sheet's first page (0 when empty or unreadable)."""
    from . import formats as F
    try:
        cells = F.load(ws.path(path, must_exist=True))["sheets"][0]["cells"]
    except Exception:
        return 0
    rows = [int(m.group(1)) for k in cells for m in [re.match(r"[A-Z]+(\d+)$", k)] if m]
    return max(rows or [0])


def deck_slides(ws, path):
    """[(title lower case, is a content slide)] of a deck."""
    from . import formats as F
    try:
        slides = F.load(ws.path(path, must_exist=True))["slides"]
    except Exception:
        return []
    return [(str(s.get("title") or "").strip().lower(), bool(s.get("bullets")) or s.get("layout") != "title")
            for s in slides]


def deck_titles(ws, path):
    """Every slide title a deck has, the title slide's too, for not adding them again:
    asked for two more slides, the 3B wrote one titled like the deck (2026-10-06).
    The writer is told these titles are taken and asked for new topics."""
    return [t for t, _ in deck_slides(ws, path) if t]


def user_turn(job, asked, deck_title=""):
    """What the writer is asked: the person's words, made into a writing task when
    they are a complaint about a file ("only has one slide")."""
    if job["action"] != "extend":
        return asked
    noun = {"slides": "slides", "words": "sections", "sheets": "rows", "text": "lines", "code": "lines"}[job["kind"]]
    want = job["count"] or {"slides": 4, "words": 2, "sheets": 3, "text": 3, "code": 5}[job["kind"]]
    about = f' for "{deck_title}"' if deck_title else ""
    return f'Write {want} new {noun}{about}. The person said: "{asked}"'


def drop_repeats(parsed, have, deck_title=""):
    """A deck being extended keeps only slides it does not have yet: told "do not
    repeat", the 3B wrote the same five slides again (2026-10-06). A new slide
    under a taken title, with points the deck lacks, is renamed instead of lost:
    the 3B names an overview slide after the deck itself even when told the title
    is taken ("About The Human Heart"; 2026-10-07, traced)."""
    title, subtitle, slides = parsed
    # normalized here, not at the call: one caller passed titles with a trailing
    # space, so "Provide Care " did not match "Provide Care" and a first-aid deck
    # kept both (2026-10-07 orientation)
    seen = {str(h).strip().lower() for h in have if str(h).strip()}
    keep = []
    for s in slides:
        k = s["title"].strip().lower()
        if k and k in seen:
            # only the deck's own title is renamed; any other repeat is the same slide again
            if not (deck_title and k == deck_title.strip().lower()):
                continue
            new = f"About {s['title'].strip()}"
            if new.lower() in seen or len(s["bullets"]) < 2:
                continue
            s = dict(s, title=new)
            k = new.lower()
        if k and k not in seen:
            keep.append(s)
            seen.add(k)
    return title, subtitle, keep


def merge_repeats(parsed, most=6):
    """Slides written twice under one title are one topic: the points go to the
    first, without doubles, at most `most` (a first-aid deck had two "Provide Care"
    slides; 2026-10-07 orientation)."""
    title, subtitle, slides = parsed
    out, at = [], {}
    for s in slides:
        k = s["title"].strip().lower()
        if k in at:
            first = out[at[k]]
            for b in s["bullets"]:
                if b not in first["bullets"] and len(first["bullets"]) < most:
                    first["bullets"].append(b)
            continue
        at[k] = len(out)
        out.append(dict(s, bullets=list(s["bullets"])))
    return title, subtitle, out


def drop_thin(parsed, least=2):
    """Content slides need points: a written slide with fewer than two (often the
    deck's own title again, with one line) is left out, unless every slide is thin."""
    title, subtitle, slides = parsed
    keep = [s for s in slides if len(s["bullets"]) >= least]
    return title, subtitle, keep if keep else slides


# ------------------------------------------------------------------ Studio: planned from the request
# "short" alone is a short poem or story as often as a video: only a short that is a video
# ("a short video", "a 9:16 short", "YouTube shorts") is Studio's ("Write a short poem about the
# rain" became a Studio project, 2026-10-14 cue audit)
STUDIO = (r"\b(intro|outro|logo reveal|title card|lower third|end card|shorts|reel|promo|video|"
          r"(?:youtube|tiktok|instagram|vertical|9:16) shorts?|short (?:video|clip|ad|film|promo))\b|9:16")


def studio_plan(asked):
    """studio_create's arguments read from the request, or None when it is not a
    Studio video to make. The 3B called studio_template on a project that did not
    exist instead of creating one (2026-10-06 orientation); the template, length,
    shape and title are all in the person's words."""
    t = (asked or "").lower()
    # words inside a file name are not asks: "... intro.sstudio twice as fast" is an edit of that
    # project, read as "make an intro" before (2026-10-14 cue audit)
    t = re.sub(r"[\w./\\-]+\.(sstudio|sdeck|sdoc|ssheet|mp4|mov|png|jpe?g|wav|mp3)\b", " ", t)
    if not re.search(STUDIO, t) or not re.search(CREATE, t):
        return None
    if re.search(r"\b(logo)\b", t):
        tmpl = "logo-reveal"
    elif re.search(r"\b(outro|end card|thanks for watching|closing)\b", t):
        tmpl = "outro"
    elif re.search(r"\b(9:16|shorts?|reel|vertical|tiktok|phone)\b", t):
        tmpl = "short"
    elif "lower third" in t:
        tmpl = "lower-third"
    else:
        tmpl = "title-intro"
    m = re.search(r"\b(\d{1,3})[- ]?(?:seconds?|secs?|s)\b", t)
    dur = int(m.group(1)) if m else (8 if tmpl in ("short", "lower-third") else 10)
    fmt = "9:16" if tmpl == "short" else "1:1" if re.search(r"\b(1:1|square)\b", t) else "16:9"
    a = asked or ""
    pats = [r"[\"“']([^\"”']{2,60})[\"”']",
            r"\bwith the (?:words|text|title|name|heading)\s+(.+?)(?:\s+and (?:render|export|make)\b|[.,!?]|$)",
            r"\b(?:that says|which says|saying|says)\s+(.+?)(?:\s+and (?:render|export|make)\b|[.,!?]|$)",
            r"\b(?:the title|titled|called)\s+(?!\S+\.sstudio)(.+?)(?:\s+and (?:render|export|make)\b|[.,!?]|$)",
            r"\bfor\s+([A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*){0,4})",
            r"\babout\s+(.+?)(?:\s+and (?:render|export|make)\b|[.,!?]|$)"]
    title = ""
    for p in pats:
        mm = re.search(p, a)
        if mm:
            title = mm.group(1).strip()
            break
    sub = ""
    mf = re.search(r"\bfor\s+([A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*){0,4})", a)
    if mf and title and mf.group(1).strip() != title:
        sub = mf.group(1).strip()
    # the file name the person gave: "called intro.sstudio" was saved as sheep.sstudio
    # (named after the title), and the next request, "render intro.sstudio", found nothing
    named = re.search(r"\b([\w./\\-]+\.sstudio)\b", a, re.I)
    return {"template": tmpl, "duration": dur, "format": fmt, "title": title[:60], "subtitle": sub[:60],
            "path": named.group(1) if named else "",
            "render": bool(re.search(r"\b(render|mp4|export)\b", t))}


TOTAL_ROW = r"\b(totals?|sum|grand total)( row| line)?\b"


def total_row(ws, path):
    """(cell, [row]) for a Total row under a sheet's data: SUM formulas over each
    number column. Exact, so the model never writes it: asked to "add a total row",
    the 3B wrote a row of plain numbers (2026-10-06 orientation). None when there
    is nothing to total."""
    from . import formats as F
    try:
        cells = F.load(ws.path(path, must_exist=True))["sheets"][0]["cells"]
    except Exception:
        return None
    pos = [(m.group(1), int(m.group(2))) for k in cells for m in [re.match(r"([A-Z]+)(\d+)$", k)] if m]
    if not pos:
        return None
    last = max(r for _, r in pos)
    cols = sorted({c for c, _ in pos}, key=lambda c: (len(c), c))
    val = lambda v: v.get("v") if isinstance(v, dict) else v
    numeric = [c for c in cols if any(isinstance(val(cells.get(f"{c}{r}")), (int, float)) for r in range(2, last + 1))]
    if not numeric or last < 2:
        return None
    first = str(val(cells.get(f"A{last}")) or "")
    if re.match(r"^\s*(total|sum)", first, re.I):
        return None                       # it has one already
    row = [("Total" if c == cols[0] else (f"=SUM({c}2:{c}{last})" if c in numeric else "")) for c in cols]
    return f"{cols[0]}{last + 1}", row


def native(path, kind):
    """Sheep's own file for an Office name: save-water.pptx -> save-water.sdeck."""
    k = KINDS[kind]
    if k["office"] and path.lower().endswith(k["office"]):
        return path[: -len(k["office"])] + k["ext"]
    return path


FORMAT_LEAD = re.compile(r"^(?:a |an |the )?(?:power ?point|pptx|ppt|deck|presentation|slide ?show|slides|document|"
                         r"docx|doc|report|spreadsheet|workbook|sheet|xlsx)\s+(?:about|on|of|for)\s+", re.I)


def clean_title(title):
    """"Powerpoint about the History of Coffee" -> "The History of Coffee": the request's format words are
    not the file's title (gallery, 2026-10-22)."""
    t = FORMAT_LEAD.sub("", (title or "").strip())
    return t[:1].upper() + t[1:] if t else (title or "")


def calls(job, parsed, exists=lambda p: False, next_row=lambda p: 1):
    """[(tool, args)] that make or extend the file from what was written."""
    kind, out = job["kind"], []
    if kind == "slides":
        title, subtitle, slides = parsed
        title = clean_title(title)
        if job["action"] == "create":
            path = new_path(job, title or (slides[0]["title"] if slides else "deck"), exists)
            out.append(("slides_create", {"path": path, "title": title or (slides[0]["title"] if slides else ""),
                                          "subtitle": subtitle, **({"theme": job["theme"]} if job["theme"] else {})}))
            # "a PowerPoint about ...": the deck is made as .sdeck and exported, so the slides go
            # into the deck, not the .pptx name (2026-10-14 live run: slides_add on save-water.pptx
            # was refused and the job reported "Stopped")
            path = native(path, kind)
        else:
            path = job["target"]
        for s in slides:
            out.append(("slides_add", {"path": path, "title": s["title"], "bullets": s["bullets"],
                                       **({"notes": s["notes"]} if s["notes"] else {})}))
        return out
    if kind == "words":
        title, blocks = parsed
        title = clean_title(title)
        if job["action"] == "create":
            path = new_path(job, title or "document", exists)
            out.append(("words_create", {"path": path, "title": title}))
            path = native(path, kind)
        else:
            path = job["target"]
            if title:
                blocks = [("heading", 1, title)] + list(blocks)
        i = 0
        while i < len(blocks):
            kind_b, level, text = blocks[i]
            if kind_b in ("bullet", "numbered") and level == 0:
                j = i
                while j < len(blocks) and blocks[j][0] == kind_b and blocks[j][1] == 0:
                    j += 1
                out.append(("words_add_list", {"path": path, "items": [b[2] for b in blocks[i:j]],
                                               "numbered": kind_b == "numbered"}))
                i = j
                continue
            if kind_b == "table":
                out.append(("words_add_table", {"path": path, "rows": text}))
            else:
                out.append(("words_add", {"path": path, "text": text, "kind": kind_b,
                                          **({"level": level} if kind_b != "paragraph" else {})}))
            i += 1
        return out
    if kind in ("text", "code"):
        path = job["target"] if job["action"] == "extend" else new_path(job, parsed.splitlines()[0] if parsed else "notes",
                                                                         exists)
        text = (job.get("existing", "").rstrip() + "\n" + parsed) if job["action"] == "extend" else parsed
        return [("write_text_file", {"path": path, "text": text})]
    headers, rows = parsed
    if job["action"] == "create":
        path = new_path(job, " ".join(str(h) for h in headers[:2]) if not job["name"] else job["name"], exists)
        out.append(("sheets_create", {"path": path, "headers": headers, "rows": rows, "total": job["total"]}))
    else:
        out.append(("sheets_write", {"path": job["target"], "cell": f"A{next_row(job['target'])}", "values": rows}))
    return out


def summary(job, parsed, path):
    """What was made, in one line."""
    if job["kind"] == "slides":
        n = len(parsed[2])
        heads = ", ".join(s["title"] for s in parsed[2][:6]) + ("…" if n > 6 else "")
        if job["action"] == "create":
            return f"Made {path}: a title slide and {n} slide{'s' if n != 1 else ''} ({heads})."
        return f"Added {n} slide{'s' if n != 1 else ''} to {path} ({heads})."
    if job["kind"] == "words":
        heads = [b[2] for b in parsed[1] if b[0] == "heading"]
        what = f"{len(heads)} sections ({', '.join(heads[:5])})" if heads else f"{len(parsed[1])} blocks"
        return (f"Made {path} with {what}." if job["action"] == "create" else f"Added {what} to {path}.")
    if job["kind"] in ("text", "code"):
        n = len(parsed.strip().splitlines())
        return (f"Saved {path} ({n} line{'s' if n != 1 else ''})." if job["action"] == "create"
                else f"Added {n} line{'s' if n != 1 else ''} to {path}.")
    rows = len(parsed[1])
    return (f"Made {path}: {len(parsed[0])} columns and {rows} rows" + (" with a Total row." if job["total"] else ".")
            if job["action"] == "create" else f"Added {rows} rows to {path}.")
