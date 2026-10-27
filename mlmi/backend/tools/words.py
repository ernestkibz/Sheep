"""Sheep Words: .sdoc documents.

A document is a numbered list of blocks, and every operation names blocks by
that number, so the editor and the model point at the same thing. Word files
come in with import_file and go out with export_file (Ernest, 2026-09-30).
Text is written with inline marks: **bold**, *italic*, __underline__.
"""
from . import tool, overwrite_check
from .. import formats as F
from ..workspace import ToolError

KINDS = ("paragraph", "heading", "bullet", "numbered", "quote", "table", "image", "pagebreak")


def ensure_ext(path, ext):
    base = path
    for e in (".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"):
        if base.lower().endswith(e):
            base = base[: -len(e)]
    return base if base.lower().endswith(ext) else base + ext


NATIVE = {"words": ".sdoc", "sheets": ".ssheet", "presentations": ".sdeck", "studio": ".sstudio"}
MAKER = {"words": "words_create", "sheets": "sheets_create", "presentations": "slides_create", "studio": "studio_create"}


def resolve(ctx, path, kind):
    """The Sheep file a path means.

    A model told to make "shopping.docx" goes on calling it that. Refusing the
    name sent a 1B model on the first live run into importing a file that did
    not exist, over the one it had just made. So a Microsoft name means its
    Sheep copy: worked on here, imported first if only the Microsoft file
    exists, and exported back when the job ends."""
    ext = NATIVE[kind]
    low = str(path).lower()
    for office, native in F.OFFICE.items():
        if not low.endswith(office):
            continue
        if native != ext:
            raise ToolError(f"{path} is not a {kind} file")
        nat = path[: -len(office)] + native
        if ctx.ws.path(nat).exists():
            ctx.notes.append(f"(worked on {nat}, the Sheep copy of {path})")
        elif ctx.ws.path(path).exists():
            from .transfer import import_file
            import_file(ctx, path, nat)
            ctx.notes.append(f"(imported {path} as {nat} first)")
        else:
            raise ToolError(f"there is no {path} yet; make it with {MAKER[kind]}")
        ctx.exports.setdefault(nat, path)
        return nat
    return path


def start_new(ctx, path, kind, overwrite):
    """For the *_create tools: (path, None) to go ahead, or (path, message) when it
    already exists -- a small model calls create twice, and the second call must
    not quietly wipe the first one's work."""
    native = ensure_ext(path, NATIVE[kind])
    if native != path and str(path).lower().endswith(tuple(F.OFFICE)):
        ctx.exports[native] = path
    p = ctx.ws.path(native)
    if p.exists() and not overwrite:
        return p, f"{native} already exists; add to it (pass overwrite=true only to start it again)"
    return p, None


def confirm_create(kind):
    def check(ws, args):
        native = ensure_ext(args.get("path", ""), NATIVE[kind])
        return f"replace existing {native}" if args.get("overwrite") and ws.path(native).exists() else None
    return check


def open_doc(ctx, path, kind):
    """Load a Sheep document, or say how to get one."""
    ext = NATIVE[kind]
    path = resolve(ctx, path, kind)
    p = ctx.ws.path(path, must_exist=True)
    if p.suffix.lower() != ext:
        raise ToolError(f"{path} is not a {ext} file")
    try:
        d = F.load(p)
    except (ValueError, OSError) as e:
        raise ToolError(f"{path} could not be read: {e}")
    if d.get("sheep") != kind:
        raise ToolError(f"{path} is a Sheep {d.get('sheep')} file, not {kind}")
    return p, d


def save_doc(ctx, p, d, what):
    with ctx.ws.changing(p, by=ctx.by, what=what):
        p.parent.mkdir(parents=True, exist_ok=True)
        F.save(p, d)
    ctx.made.add(ctx.ws.rel(p))


def _idx(d, index, allow_end=False):
    n = len(d["blocks"])
    try:
        i = int(index)
    except (TypeError, ValueError):
        raise ToolError("index must be a block number from words_read")
    if not 0 <= i < n + (1 if allow_end else 0):
        raise ToolError(f"there is no block {index}; the document has blocks 0-{n - 1}" if n else "the document is empty")
    return i


def _insert(d, block, after):
    if after is None or after == "":
        d["blocks"].append(block)
        return len(d["blocks"]) - 1
    i = max(-1, min(int(after), len(d["blocks"]) - 1)) + 1
    d["blocks"].insert(i, block)
    return i


def model(path, ws):
    class C:
        pass
    c = C()
    c.ws = ws
    _, d = open_doc(c, path, "words")
    return {"path": path, **d}


# ------------------------------------------------------------------ tools

@tool("words_create", "words", 2, "Create a new Sheep Words document (.sdoc). The title becomes the heading at "
      "the top: do not add it again.",
      {"path": ("string", "new file, e.g. report.sdoc"), "title": ("string", "title at the top (optional)"),
       "overwrite": ("boolean", "true only to replace an existing document")},
      ("path",), confirm=confirm_create("words"))
def words_create(ctx, path, title="", overwrite=False):
    p, msg = start_new(ctx, path, "words", overwrite)
    if msg:
        return msg
    d = F.new("words")
    if title:
        d["title"] = title
        d["blocks"].append({"type": "heading", "level": 0, "text": title})
    save_doc(ctx, p, d, f"create {ctx.ws.rel(p)}")
    return f"created {ctx.ws.rel(p)}"


@tool("words_read", "words", 1, "Read a Sheep Words document as numbered blocks.",
      {"path": ("string", "the .sdoc file")}, ("path",))
def words_read(ctx, path):
    _, d = open_doc(ctx, path, "words")
    return F.words_outline(d)


@tool("words_add", "words", 2, "Add a block: a paragraph, heading, bullet, numbered item or quote. "
      "Use **bold** and *italic* inside the text.",
      {"path": ("string", "the .sdoc file"), "text": ("string", "the text"),
       "kind": ("string", "paragraph, heading, bullet, numbered or quote",
                ("paragraph", "heading", "bullet", "numbered", "quote")),
       "level": ("integer", "heading size 1-3, or list indent 0-2"),
       "after": ("integer", "insert after this block number; omit to add at the end")},
      ("path", "text"))
def words_add(ctx, path, text, kind="paragraph", level=None, after=None):
    p, d = open_doc(ctx, path, "words")
    kind = (kind or "paragraph").lower()
    text = F.clean(text)
    if text.strip().lower() in ("heading", "paragraph", "bullet", "numbered", "quote", "title", "text"):
        # the 3B once wrote the word "heading" as a heading's text, copied from a plan step
        raise ToolError(f"'{text}' is the kind of block, not its text: pass the words to write as text")
    if kind in ("bullet", "numbered") and len(F.items(text)) > 1:
        # the whole list in one call: one block per line
        at = None if after in (None, "") else int(after)
        for k, it in enumerate(F.items(text)):
            _insert(d, {"type": kind, "level": max(0, min(int(level or 0), 2)), "text": it},
                    None if at is None else at + k)
        save_doc(ctx, p, d, f"add {kind} list to {path}")
        return f"added {len(F.items(text))} {kind} items"
    if kind == "heading":
        b = {"type": "heading", "level": max(0, min(int(level if level is not None else 1), 3)), "text": text}
    elif kind in ("bullet", "numbered"):
        b = {"type": kind, "level": max(0, min(int(level or 0), 2)), "text": F.item(text)}
    elif kind == "quote":
        b = {"type": "paragraph", "style": "quote", "text": text}
    else:
        b = {"type": "paragraph", "text": text}
    i = _insert(d, b, after)
    save_doc(ctx, p, d, f"add {kind} to {path}")
    return f"added {kind} as block {i}"


@tool("words_add_list", "words", 2, "Add several bullet or numbered items at once.",
      {"path": ("string", "the .sdoc file"), "items": ("array", "the items, one string each"),
       "numbered": ("boolean", "true for 1. 2. 3., false for bullets")}, ("path", "items"))
def words_add_list(ctx, path, items, numbered=False):
    p, d = open_doc(ctx, path, "words")
    flat = [x for it in items for x in (F.items(it) or [""])]
    for it in flat:
        d["blocks"].append({"type": "numbered" if numbered else "bullet", "level": 0, "text": it})
    save_doc(ctx, p, d, f"list in {path}")
    return f"added {len(flat)} {'numbered' if numbered else 'bullet'} items"


@tool("words_add_table", "words", 2, "Add a table. rows is a list of rows, each a list of cell texts; "
      "the first row is the header.",
      {"path": ("string", "the .sdoc file"),
       "rows": ("array", "rows of cells, e.g. [[\"Name\",\"Qty\"],[\"Apples\",\"3\"]]",
                {"type": "array", "items": {"type": "string"}}),
       "after": ("integer", "insert after this block number; omit for the end")}, ("path", "rows"))
def words_add_table(ctx, path, rows, after=None):
    p, d = open_doc(ctx, path, "words")
    rows = [[str(c) for c in (r if isinstance(r, list) else [r])] for r in rows]
    if not rows:
        raise ToolError("rows must have at least one row")
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    i = _insert(d, {"type": "table", "header": True, "rows": rows}, after)
    save_doc(ctx, p, d, f"table in {path}")
    return f"added a {len(rows)}x{w} table as block {i}"


@tool("words_add_image", "words", 2, "Add a picture from the workspace.",
      {"path": ("string", "the .sdoc file"), "image": ("string", "the picture file (.png or .jpg)"),
       "width_inches": ("number", "width in inches, default 5"),
       "after": ("integer", "insert after this block number; omit for the end")}, ("path", "image"))
def words_add_image(ctx, path, image, width_inches=5.0, after=None):
    p, d = open_doc(ctx, path, "words")
    ctx.ws.path(image, must_exist=True)
    i = _insert(d, {"type": "image", "src": image, "width": float(width_inches or 5.0)}, after)
    save_doc(ctx, p, d, f"image in {path}")
    return f"added picture {image} as block {i}"


@tool("words_add_page_break", "words", 2, "Start a new page.", {"path": ("string", "the .sdoc file")}, ("path",))
def words_add_page_break(ctx, path):
    p, d = open_doc(ctx, path, "words")
    d["blocks"].append({"type": "pagebreak"})
    save_doc(ctx, p, d, f"page break in {path}")
    return "added a page break"


@tool("words_set", "words", 2, "Replace one block's text, and optionally change its kind.",
      {"path": ("string", "the .sdoc file"), "index": ("integer", "block number from words_read"),
       "text": ("string", "the new text; **bold** and *italic* allowed"),
       "kind": ("string", "paragraph, heading, bullet, numbered or quote",
                ("paragraph", "heading", "bullet", "numbered", "quote")),
       "level": ("integer", "heading size 1-3, or list indent 0-2")},
      ("path", "index", "text"))
def words_set(ctx, path, index, text, kind="", level=None):
    p, d = open_doc(ctx, path, "words")
    i = _idx(d, index)
    b = d["blocks"][i]
    if b.get("type") in ("table", "image", "pagebreak") and not kind:
        raise ToolError(f"block {i} is a {b.get('type')}; use words_set_cell for tables")
    b["text"] = text
    if kind:
        kind = kind.lower()
        if kind == "quote":
            b.update(type="paragraph", style="quote")
        else:
            b["type"] = kind
            b.pop("style", None)
        for k in ("rows", "src", "width", "header"):
            b.pop(k, None)
    if level is not None:
        b["level"] = int(level)
    elif b.get("type") == "heading" and "level" not in b:
        b["level"] = 1
    save_doc(ctx, p, d, f"edit block {i} of {path}")
    return f"block {i} now reads: {text[:80]}"


@tool("words_set_cell", "words", 2, "Change the text of one table cell.",
      {"path": ("string", "the .sdoc file"), "index": ("integer", "the table's block number"),
       "row": ("integer", "row, from 0"), "col": ("integer", "column, from 0"), "text": ("string", "new text")},
      ("path", "index", "row", "col", "text"))
def words_set_cell(ctx, path, index, row, col, text):
    p, d = open_doc(ctx, path, "words")
    i = _idx(d, index)
    b = d["blocks"][i]
    if b.get("type") != "table":
        raise ToolError(f"block {i} is not a table")
    try:
        b["rows"][int(row)][int(col)] = str(text)
    except IndexError:
        raise ToolError(f"the table has {len(b['rows'])} rows and {len(b['rows'][0])} columns")
    save_doc(ctx, p, d, f"edit cell in {path}")
    return f"cell ({row},{col}) set"


@tool("words_delete", "words", 2, "Delete one block.",
      {"path": ("string", "the .sdoc file"), "index": ("integer", "block number from words_read")},
      ("path", "index"))
def words_delete(ctx, path, index):
    p, d = open_doc(ctx, path, "words")
    i = _idx(d, index)
    gone = d["blocks"].pop(i)
    save_doc(ctx, p, d, f"delete block {i} of {path}")
    return f"deleted block {i} ({gone.get('type')})"


@tool("words_move", "words", 2, "Move a block to a new position.",
      {"path": ("string", "the .sdoc file"), "index": ("integer", "block to move"),
       "to": ("integer", "its new block number")}, ("path", "index", "to"))
def words_move(ctx, path, index, to):
    p, d = open_doc(ctx, path, "words")
    i = _idx(d, index)
    j = max(0, min(int(to), len(d["blocks"]) - 1))
    d["blocks"].insert(j, d["blocks"].pop(i))
    save_doc(ctx, p, d, f"move block {i} to {j} in {path}")
    return f"moved block {i} to {j}"


@tool("words_replace", "words", 2, "Find and replace text everywhere in a document.",
      {"path": ("string", "the .sdoc file"), "find": ("string", "text to find"),
       "replace": ("string", "replacement text")}, ("path", "find", "replace"))
def words_replace(ctx, path, find, replace):
    p, d = open_doc(ctx, path, "words")
    if not find:
        raise ToolError("find must not be empty")
    n = 0
    for b in d["blocks"]:
        if isinstance(b.get("text"), str) and find in b["text"]:
            n += b["text"].count(find)
            b["text"] = b["text"].replace(find, replace)
        for row in b.get("rows") or []:
            for k, c in enumerate(row):
                if find in c:
                    n += c.count(find)
                    row[k] = c.replace(find, replace)
    if not n:
        return f"'{find}' was not found"
    save_doc(ctx, p, d, f"replace in {path}")
    return f"replaced {n} occurrence(s)"


@tool("words_format", "words", 2, "Format one block: alignment, size, color, or make it all bold/italic.",
      {"path": ("string", "the .sdoc file"), "index": ("integer", "block number"),
       "align": ("string", "left, center, right or justify", ("left", "center", "right", "justify")),
       "size": ("number", "font size in points"), "color": ("string", "hex color like 1F4E79"),
       "bold": ("boolean", "make the whole block bold"), "italic": ("boolean", "make the whole block italic")},
      ("path", "index"))
def words_format(ctx, path, index, align=None, size=None, color=None, bold=None, italic=None):
    p, d = open_doc(ctx, path, "words")
    i = _idx(d, index)
    b = d["blocks"][i]
    if align:
        b["align"] = align
    if size:
        b["size"] = float(size)
    if color:
        b["color"] = color.strip("#")[:6].upper()
    if bold is not None or italic is not None:
        # set only the mark asked for: "bold off" used to strip italic and underline too
        rs = F.runs(b.get("text", ""))
        b["text"] = F.marks([(t, bo if bold is None else bool(bold), it if italic is None else bool(italic), u)
                             for t, bo, it, u in rs])
    save_doc(ctx, p, d, f"format block {i} of {path}")
    return f"formatted block {i}"


@tool("words_put", "words", 2, "Editor: replace one block with the given block object.",
      {"path": ("string", "the .sdoc file"), "index": ("integer", "block number (the end to append)"),
       "block": ("object", "the block"), "insert": ("boolean", "insert before instead of replacing")},
      ("path", "index", "block"))
def words_put(ctx, path, index, block, insert=False):
    p, d = open_doc(ctx, path, "words")
    if not isinstance(block, dict) or block.get("type") not in KINDS:
        raise ToolError(f"a block needs a type: {', '.join(KINDS)}")
    i = _idx(d, index, allow_end=True)
    if insert or i == len(d["blocks"]):
        d["blocks"].insert(i, block)
    else:
        d["blocks"][i] = block
    save_doc(ctx, p, d, f"edit block {i} of {path}")
    return f"block {i} saved"
