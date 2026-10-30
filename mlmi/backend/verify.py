"""Is what a job made what was asked? Read from the files themselves.

When the model says a job is done, the orchestrator opens what it made and
compares it with the request: a file asked for by name exists; a deck has the
slides asked for; a Studio video is as long as asked; a list in a document has
the items asked for. Each shortfall is said in plain words; the model gets one
chance to fix them, and anything still short is told to the person, never
hidden. (The other tool's verify_artifact checked slide counts; rebuilt here
from the request's own numbers.)
"""
import re

from . import formats as F

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}
NUM = r"(\d{1,3}|" + "|".join(WORDS) + r")"
EXTS = r"(sdoc|ssheet|sdeck|sstudio|docx|xlsx|pptx|pdf|txt|md|csv|py|html|js|png|mp4|wav)"


def _n(w):
    return int(w) if w.isdigit() else WORDS.get(w, 0)


def _load(ws, rel):
    try:
        return F.load(ws.path(rel, must_exist=True))
    except Exception:
        return None


def hollow(d, rel):
    """A new file with nothing but its title or headers. Nemotron made a 90-day plan that was one heading and
    said it mirrored the attached plan (2026-10-30): a file that exists is not a job that is done."""
    if rel.endswith(".sdoc"):
        return not any(b.get("text") or b.get("rows") or b.get("src") for b in d.get("blocks", [])
                       if not (b.get("type") == "heading" and b.get("level") == 0))
    if rel.endswith(".sdeck"):
        return not any(s.get("bullets") or s.get("items") for s in d.get("slides", []))
    if rel.endswith(".ssheet"):
        return not any(re.search(r"\d+$", k) and int(re.search(r"\d+$", k).group()) > 1
                       for s in d.get("sheets", []) for k in s.get("cells", {}))
    return False


def _raw(c):
    return c.get("v") if isinstance(c, dict) else c


def book_size(d):
    """(sheet names, cells, formulas) of a workbook."""
    cells = [_raw(c) for s in d.get("sheets", []) for c in s.get("cells", {}).values()]
    return [s.get("name") for s in d.get("sheets", [])], len(cells), sum(str(c).startswith("=") for c in cells)


KEPT_WORDS = {"ahead", "on track", "behind", "status", "metric", "notes", "owner", "total", "target"}


def stale(ws, rel, example):
    """Cells of a made workbook still holding the example's own words: (still the same, all, ["Sheet!A1: words"]).
    A copy with two cells changed was called done (2026-10-30): the design is kept, the content has to change."""
    a, b = _load(ws, example), _load(ws, rel)
    if not a or not b:
        return 0, 0, []
    new = {s.get("name"): s.get("cells", {}) for s in b.get("sheets", [])}
    same, total, left = 0, 0, []
    for s in a.get("sheets", []):
        for k, c in s.get("cells", {}).items():
            v = _raw(c)
            if (not isinstance(v, str) or v.startswith("=") or len(v) < 4 or v.strip().lower() in KEPT_WORDS
                    or not re.search(r"[a-z]{3}", v, re.I)):
                continue
            total += 1
            if s.get("name") in new and k in new[s["name"]] and _raw(new[s["name"]][k]) == v:
                same += 1
                left.append(f"{s['name']}!{k}: {v[:50]}")
    return same, total, left


def example_of(ws, asked):
    """The workbook a request holds up as its example, and a line on how much is in it, so that a model matches
    its depth and not only its topic: Nemotron answered a tracker of 382 formulas with a table of 30 cells."""
    best = None
    for rel in re.findall(r"[\w./ ()'&-]*?attachments/[^;\n]+?\.ssheet", asked or ""):
        d = _load(ws, rel.strip())
        size = book_size(d) if d else None
        if size and (not best or size[2] > best[1][2]):
            best = (rel.strip(), size, d)
    if not best:
        return None, ""
    rel, (names, cells, formulas), d = best
    seen = list(dict.fromkeys(re.sub(r"\$?[A-Z]+\$?\d+", "#", str(_raw(c)))[:70] for s in d["sheets"]
                              for c in s["cells"].values() if str(_raw(c)).startswith("=")))[:5]
    return rel, (f"The example {rel} has {len(names)} sheets ({', '.join(map(str, names))}), {cells} cells and {formulas} "
                 "formulas, so it calculates itself" + (f", such as {' ; '.join(seen)}" if seen else "") + ". Match that "
                 "depth: live formulas for totals, pace and status, and a dashboard that reads from the data. The quickest "
                 "good way is copy_file of the example to a new name, once, then sheets_read and change its text and numbers "
                 "for this job with sheets_set_many, a whole sheet in one call, leaving its formulas alone.")


def _list_items(d):
    """Items in the document's lists, and its paragraphs, for counted requests."""
    items = paras = 0
    for b in d.get("blocks", []):
        k = b.get("type", "")
        if k in ("bullet", "numbered"):
            items += 1
        elif k == "paragraph" and b.get("style") != "quote":
            paras += 1
        elif k == "table":
            items = max(items, len(b.get("rows") or []) - (1 if b.get("header") else 0))
    return items, paras


def shortfalls(ws, asked, made):
    """[plain-words problem] for this request and the files the job made."""
    t = (asked or "").lower()
    made = sorted(made or [])
    out = []
    # a file asked for by name
    for name, _ in re.findall(r"\b(?:called|named|save (?:it |this )?as|name it)\s+['\"]?([\w .-]+?\." + EXTS + r")\b", t):
        name = name.strip()
        if not any(m.lower().split("/")[-1] == name for m in made) and not ws.path(name).exists():
            out.append(f"there is no file called {name}")
    if not re.search(r"\b(blank|empty|template|skeleton)\b", t):
        for rel in made:
            d = _load(ws, rel) if rel.endswith((".sdoc", ".sdeck", ".ssheet")) else None
            if d and hollow(d, rel):
                out.append(f"{rel} holds only its title, with none of the content that was asked for")
    decks = [m for m in made if m.endswith(".sdeck")]
    docs = [m for m in made if m.endswith(".sdoc")]
    vids = [m for m in made if m.endswith(".sstudio")]
    # slides asked for
    m = re.search(r"\b" + NUM + r"[- ](?:content |more )?slides?\b", t)
    if m and decks:
        want = _n(m.group(1))
        for rel in decks:
            d = _load(ws, rel)
            have = len(d.get("slides", [])) if d else 0
            if d and have < want:
                out.append(f"{rel} has {have} slide{'s' if have != 1 else ''}; {want} were asked for")
    # a video's length (Studio)
    m = re.search(r"\b" + NUM + r"[- ]?(?:seconds?|secs?|s)\b", t)
    if m and vids:
        want = _n(m.group(1))
        from .tools.studio import length
        for rel in vids:
            d = _load(ws, rel)
            if d and length(d) + 0.5 < want:
                out.append(f"{rel} is {length(d):.0f} seconds long; {want} were asked for")
    # counted items in a document ("three goals", "five bullet points")
    m = re.search(r"\b" + NUM + r" (?:bullet points?|bullets|points|goals|steps|tips|ideas|reasons|items|questions)\b", t)
    if m and docs:
        want = _n(m.group(1))
        for rel in docs:
            d = _load(ws, rel)
            if not d:
                continue
            items, paras = _list_items(d)
            if max(items, paras) < want:
                out.append(f"{rel} has {max(items, paras)} of the {want} {m.group(0).split(' ', 1)[1]} asked for")
    # a formula over nothing: the 3B added "=SUM(C2:C4)" under a table whose numbers
    # are in column B, a stray 0 row beside the real Total (2026-10-13 simulation)
    from openpyxl.utils import get_column_letter, range_boundaries
    for rel in (m for m in made if m.endswith(".ssheet")):
        d = _load(ws, rel)
        for sh in (d or {}).get("sheets", []):
            cells = sh.get("cells") or {}
            for k, v in cells.items():
                if not (isinstance(v, str) and v.startswith("=")):
                    continue
                for ref in re.findall(r"\b([A-Z]{1,3}\d+:[A-Z]{1,3}\d+)\b", v.upper()):
                    try:
                        c0, r0, c1, r1 = range_boundaries(ref)
                    except ValueError:
                        continue
                    inside = [cells.get(f"{get_column_letter(c)}{r}") for c in range(c0, c1 + 1)
                              for r in range(r0, r1 + 1)]
                    if all(x is None or x == "" for x in inside):
                        out.append(f"{k} in {rel} is {v}, but {ref} is empty: point it at the numbers or remove it")
    return out
