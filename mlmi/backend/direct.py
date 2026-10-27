"""One-line commands Sheep runs itself, without a model step.

"Render intro.sstudio", "Export budget.ssheet to Excel", "Copy a.txt to b.txt",
"Give save-water.sdeck a dark theme", "Delete old-notes.txt", "Add a calendar
event called Dentist on 2026-11-03 at 10:00", "In trip-plan.sdoc replace the
word Nairobi with Mombasa". The 3B answered several of these with "Done" and no
tool call, and after a correct replace it went on for 14 steps adding junk to
the document (2026-10-13 simulation). When the whole request is one such
command, the exact tool runs once and the job ends; anything longer or less
certain goes to the model as before.
"""
import re

FILE = r"[\w./\\-]+\.(?:sdoc|ssheet|sdeck|sstudio|txt|md|csv|json|docx|xlsx|pptx|pdf|mp4|png|jpe?g|eml|ics|wav)"
FMT = {"word": "docx", "word document": "docx", "docx": "docx", ".docx": "docx", "pdf": "pdf", "a pdf": "pdf",
       "excel": "xlsx", "xlsx": "xlsx", ".xlsx": "xlsx", "powerpoint": "pptx", "pptx": "pptx", ".pptx": "pptx",
       "csv": "csv", ".csv": "csv", "markdown": "md", "md": "md", ".md": "md"}
THEMES = ("light", "dark", "midnight", "forest", "sunset", "mono")
LEAD = r"^\s*(?:please\s+|can you\s+|could you\s+)?"
END = r"\s*[.!]?\s*$"


def _date(s):
    """'2026-11-03 at 10:00', '2026-11-03 10am', '2026-11-03T10:00' -> '2026-11-03 10:00'."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})(?:\s*(?:at|t)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?$", s.strip(), re.I)
    if not m:
        return None
    day, h, mi, ap = m.groups()
    if h is None:
        return f"{day} 09:00"
    h = int(h) % 12 + (12 if (ap or "").lower() == "pm" else 0) if ap else int(h)
    return f"{day} {h:02d}:{int(mi or 0):02d}"


def parse(asked):
    """(tool, args, words for the reply) when the request is exactly one command, else None."""
    t = (asked or "").strip()
    if not t or len(t) > 200 or re.search(r"\b(and|then|also)\b", t, re.I):
        return None                       # more than one thing asked: the model plans it
    m = re.match(LEAD + rf"render\s+({FILE})(?:\s+(?:to|as|into)\s+(?:an?\s+)?(?:mp4|video))?" + END, t, re.I)
    if m and m.group(1).lower().endswith(".sstudio"):
        return "studio_render", {"path": m.group(1)}, f"Rendered {m.group(1)} to MP4."
    m = re.match(LEAD + rf"(?:export|save|convert)\s+({FILE})\s+(?:to|as|into)\s+(?:an?\s+)?"
                 r"(word(?: document)?|\.?docx|pdf|excel|\.?xlsx|power ?point|\.?pptx|\.?csv|markdown|\.?md)"
                 r"(?:\s+\.?(?:docx|xlsx|pptx|pdf|csv|md))?(?:\s+(?:file|document|deck|workbook))?" + END, t, re.I)
    if m:
        fmt = FMT.get(m.group(2).lower()) or FMT.get(m.group(2).lower().replace(" ", ""))
        if fmt:
            return "export_file", {"path": m.group(1), "format": fmt}, None
    m = re.match(LEAD + rf"copy\s+({FILE})\s+(?:to|into|as)\s+({FILE}|[\w./\\-]+/)" + END, t, re.I)
    if m:
        dst = m.group(2) + (m.group(1).replace("\\", "/").split("/")[-1] if m.group(2).endswith("/") else "")
        return "copy_file", {"src": m.group(1), "dst": dst}, None
    m = re.match(LEAD + rf"(?:move|rename)\s+({FILE})\s+(?:to|as|into)\s+({FILE})" + END, t, re.I)
    if m:
        return "move_file", {"src": m.group(1), "dst": m.group(2)}, None
    m = re.match(LEAD + rf"(?:delete|remove)\s+(?:the\s+file\s+)?({FILE})" + END, t, re.I)
    if m:
        return "delete_file", {"path": m.group(1)}, None
    m = re.match(LEAD + rf"(?:give|set|apply|use)\s+(?:the\s+)?({FILE})\s+(?:a|the)\s+(\w+)\s+theme" + END, t, re.I) \
        or re.match(LEAD + rf"(?:make|change)\s+(?:the\s+)?(?:theme\s+of\s+)?({FILE})(?:'s\s+theme)?\s+(?:to\s+)?(\w+)" + END, t, re.I)
    if m and m.group(1).lower().endswith(".sdeck") and m.group(2).lower() in THEMES:
        return "slides_set_theme", {"path": m.group(1), "theme": m.group(2).lower()}, None
    m = re.match(LEAD + r"(?:add|put|create|make)\s+(?:a|an)\s+(?:calendar\s+)?(?:event|appointment|meeting)\s+"
                 r"(?:called|named|titled)\s+(.+?)\s+on\s+(\d{4}-\d{2}-\d{2}(?:\s*(?:at\s*)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)"
                 r"(?:\s+for\s+(\d{1,3})\s+min(?:ute)?s?)?" + END, t, re.I)
    if m:
        start = _date(m.group(2))
        if start:
            args = {"title": m.group(1).strip(" \"'“”"), "start": start}
            if m.group(3):
                args["minutes"] = int(m.group(3))
            return "calendar_add", args, None
    # a title's size (2026-10-14 live run: the 3B changed the theme instead)
    m = re.match(LEAD + rf"make\s+(?:the\s+)?(?:slide\s+(\d+)(?:'s)?\s+)?(titles?)(?:\s+(?:on|of)\s+slide\s+(\d+))?\s+"
                 rf"(bigger|larger|smaller)\s+(?:in|on|of)\s+({FILE})(?:\s+on\s+slide\s+(\d+))?" + END, t, re.I) \
        or re.match(LEAD + rf"make\s+(?:the\s+)?(?:slide\s+(\d+)(?:'s)?\s+)?(titles?)(?:\s+(?:on|of)\s+slide\s+(\d+))?\s+"
                    rf"(?:in|on|of)\s+({FILE})\s+(bigger|larger|smaller)(?:\s+on\s+slide\s+(\d+))?" + END, t, re.I)
    if m and any(g and g.lower().endswith(".sdeck") for g in m.groups()):
        g = m.groups()
        deck = next(x for x in g if x and x.lower().endswith(".sdeck"))
        size = next(x for x in g if x and x.lower() in ("bigger", "larger", "smaller"))
        n = next((x for x in (g[0], g[2], g[5]) if x), None)
        slide = int(n) if n else (0 if g[1].lower() == "titles" else 1)
        where = f"slide {slide}" if slide else "every slide"
        return ("slides_set", {"path": deck, "slide": slide, "title_size": size.lower()},
                f"Made the title on {where} of {deck} {size.lower()}.")
    # speaker notes for one slide (167 s through the model in the 2026-10-13 run)
    m = re.match(LEAD + rf"(?:add|put|write|set)\s+(?:speaker\s+)?notes?\s+(?:to|on|for)\s+slide\s+(\d+)\s+of\s+({FILE})"
                 r"\s*(?:saying|that says|reading|:)\s*:?\s*[\"“']?(.+?)[\"”']?" + END, t, re.I)
    if m and m.group(2).lower().endswith(".sdeck"):
        return "slides_set", {"path": m.group(2), "slide": int(m.group(1)), "notes": m.group(3).strip()}, None
    # a chart of a sheet's table: the whole used area, which the chart tool trims to the table (224 s before)
    m = re.match(LEAD + rf"(?:add|make|create|insert)\s+(?:a|an)\s+(bar|line|pie)\s+(?:chart|graph)"
                 rf"(?:\s+of\s+(?:the\s+)?([\w ]{{1,40}}?))?\s+(?:in|to|for|from)\s+({FILE})" + END, t, re.I)
    if m and m.group(3).lower().endswith(".ssheet"):
        title = (m.group(2) or "").strip().title()
        return "sheets_add_chart", {"path": m.group(3), "range": "A1:Z500", "kind": m.group(1).lower(),
                                    **({"title": title} if title else {})}, None
    # an email draft with all three parts given (76 s before)
    m = re.match(LEAD + r"(?:draft|write)\s+(?:an?\s+)?(?:e-?mail|mail)\s+to\s+([\w.+-]+@[\w-]+\.[\w.-]+)\s+"
                 r"(?:with\s+)?(?:the\s+)?subject\s+[\"“']?(.+?)[\"”']?,?\s+(?:saying|that says|reading|:)\s*:?\s*"
                 r"[\"“']?(.+?)[\"”']?" + END, t, re.I)
    if m:
        body = m.group(3).strip()
        return "mail_draft", {"to": m.group(1), "subject": m.group(2).strip(),
                              "body": body[:1].upper() + body[1:] + ("" if body.endswith(("?", ".", "!")) else ".")}, None
    m = re.match(LEAD + rf"in\s+({FILE}),?\s+replace\s+(?:the\s+word\s+|all\s+)?[\"“']?(.+?)[\"”']?\s+with\s+"
                 r"[\"“']?(.+?)[\"”']?(?:\s+everywhere)?" + END, t, re.I) \
        or re.match(LEAD + rf"replace\s+(?:the\s+word\s+|all\s+)?[\"“']?(.+?)[\"”']?\s+with\s+[\"“']?(.+?)[\"”']?"
                    rf"\s+in\s+({FILE})(?:\s+everywhere)?" + END, t, re.I)
    if m:
        g = m.groups()
        path, find, repl = (g[0], g[1], g[2]) if re.match(FILE + "$", g[0], re.I) else (g[2], g[0], g[1])
        if path.lower().endswith(".sdoc"):
            return "words_replace", {"path": path, "find": find, "replace": repl}, None
    return None
