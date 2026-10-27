"""Herders: the micro tools the AI writes for itself.

A fixed tool list caps what an agent can do. A herder is what a person reaches
for when the menu runs out -- a small, program-specific piece of code written
for exactly this job, kept, and used again. The model writes one with
herder_make, runs it with herder_run (or by its own name, h_<name>), and it
stays in the library for next time. Ernest's name; his idea (2026-09-30).

Every herder is `def herd(doc, args):` over ONE Sheep document -- a .sdoc,
.ssheet or .sdeck as plain data -- and runs in its own sandboxed process
(herder_runner.py). It cannot open files or reach the network; what it returns
is saved through the workspace, so every herder edit has an undo.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import TOOLS, Tool, tool
from .. import formats as F
from ..paths import BUILTIN_HERDERS, HOME
from ..herder_runner import check
from .words import open_doc, save_doc
from ..workspace import ToolError

USER_DIR = HOME / "herders"
RUNNER = Path(__file__).resolve().parent.parent / "herder_runner.py"
APPS = {"words": ("words", ".sdoc"), "sheets": ("sheets", ".ssheet"), "slides": ("presentations", ".sdeck")}
NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
TIMEOUT = 15

PRELUDE = '''
def plain(text):
    """Text without its **bold** / *italic* / __underline__ marks."""
    out, i, s = "", 0, str(text or "")
    while i < len(s):
        if s[i] == "\\\\" and i + 1 < len(s):
            out += s[i + 1]; i += 2; continue
        if s.startswith("**", i) or s.startswith("__", i):
            i += 2; continue
        if s[i] == "*":
            i += 1; continue
        out += s[i]; i += 1
    return out

def col_letter(n):
    """1 -> A, 27 -> AA"""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def col_index(letters):
    """A -> 1, AA -> 27"""
    n = 0
    for ch in letters.upper():
        n = n * 26 + ord(ch) - 64
    return n

def split_cell(ref):
    """'B12' -> ('B', 12)"""
    ref = ref.replace("$", "").upper()
    i = 0
    while i < len(ref) and ref[i].isalpha():
        i += 1
    return ref[:i], int(ref[i:])
'''

HELP = {
    "words": '''A .sdoc (Sheep Words) document as `doc`:
  doc["blocks"] = list of blocks, each a dict:
    {"type": "heading", "level": 0-3, "text": "..."}     (level 0 = title)
    {"type": "paragraph", "text": "...", "style": "normal"|"quote"}   (optional "align", "size", "color")
    {"type": "bullet"|"numbered", "level": 0-2, "text": "..."}
    {"type": "table", "header": true, "rows": [["a","b"],["1","2"]]}
    {"type": "image", "src": "pic.png", "width": 5.0}
    {"type": "pagebreak"}
  Text uses **bold**, *italic*, __underline__. plain(text) strips the marks.''',
    "sheets": '''A .ssheet (Sheep Sheets) workbook as `doc`:
  doc["sheets"] = list of sheets: {"name": "Sheet1", "cells": {"A1": "Item", "B2": 12.5, "B9": "=SUM(B2:B8)"},
                                   "fmt": {"A1": {"b": true, "bg": "DDEBF7", "nf": "0.00", "align": "center"}},
                                   "widths": {"A": 14}, "charts": [...]}
  Cell values are numbers, text, or formulas (text starting with "=").
  VALUES[sheet_name][cell] holds the computed value of each formula cell.
  Helpers: col_letter(3) -> "C", col_index("C") -> 3, split_cell("B12") -> ("B", 12).''',
    "slides": '''A .sdeck (Sheep Presentations) deck as `doc`:
  doc["theme"] = "light"|"dark"|"midnight"|"forest"|"sunset"|"mono"
  doc["slides"] = list of slides: {"layout": "title"|"bullets"|"section"|"title_only"|"blank",
      "title": "...", "subtitle": "...", "bullets": ["point", "  sub-point"], "notes": "...",
      "items": [{"kind": "image", "src": "pic.png", "x": 55, "y": 25, "w": 40},
                {"kind": "text", "text": "...", "x": 10, "y": 80, "w": 80, "size": 18}]}''',
}
CONTRACT = '''Write exactly one function:

def herd(doc, args):
    # change `doc` in place (or return a new one); `args` is a dict of your parameters
    ...
    return "a short message saying what changed"

Allowed: plain Python, and import of math, re, json, statistics, datetime, random, itertools,
collections, string, textwrap, decimal, fractions, functools, operator, copy.
Not allowed: files, network, open(), eval/exec, names starting with _. print() output is returned to you.'''


def _dirs():
    return [(BUILTIN_HERDERS, True), (USER_DIR, False)]


def library():
    """{name: meta} across the built-in and the user's herders; the user's win on a clash."""
    out = {}
    for d, builtin in _dirs():
        if not d.is_dir():
            continue
        for mf in sorted(d.glob("*.json")):
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
                code = mf.with_suffix(".py").read_text(encoding="utf-8")
            except (OSError, ValueError):
                continue
            meta.update(builtin=builtin, code=code, file=str(mf.with_suffix(".py")))
            meta.setdefault("proven", builtin)      # a model-made tool is offered only after one clean run
            meta.setdefault("asks", [])
            out[meta["name"]] = meta
    return out


def _get(name):
    h = library().get(name)
    if not h:
        names = ", ".join(sorted(library())) or "none yet"
        raise ToolError(f"there is no herder '{name}'. Herders: {names}")
    return h


def execute(ctx, h, path, args):
    app = h["app"]
    kind, ext = APPS[app]
    p, d = open_doc(ctx, path, kind)
    vals = {}
    if app == "sheets":
        from .sheets import values
        vals = values(d)
    req = {"code": h["code"], "doc": d, "args": args or {}, "values": vals, "helpers": {"prelude": PRELUDE}}
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, "-I", str(RUNNER)], input=json.dumps(req), capture_output=True,
                           text=True, timeout=TIMEOUT, encoding="utf-8",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise ToolError(f"herder '{h['name']}' ran longer than {TIMEOUT} s and was stopped (an endless loop?)")
    ms = (time.perf_counter() - t0) * 1000
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ToolError(f"herder '{h['name']}' crashed: {(r.stderr or r.stdout)[-400:]}")
    if "error" in out:
        log = "\n".join(out.get("log") or [])
        raise ToolError(f"herder '{h['name']}' failed: {out['error']}" + (f"\nprinted:\n{log[-600:]}" if log else ""))
    new = out["doc"]
    if not isinstance(new, dict) or new.get("sheep") != kind:
        raise ToolError(f"herder '{h['name']}' returned something that is not a {ext} document")
    changed = new != d
    if changed:
        save_doc(ctx, p, new, f"herder {h['name']} on {path}")
    if not h.get("builtin") and not h.get("proven"):
        try:                               # it ran cleanly once: from now on it may be offered by its phrases
            mf = USER_DIR / f"{h['name']}.json"
            meta = json.loads(mf.read_text(encoding="utf-8"))
            meta["proven"] = True
            mf.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        except (OSError, ValueError):
            pass
    msg = out.get("result")
    log = out.get("log") or []
    return (f"{h['name']}: {msg if msg is not None else 'done'}"
            + ("" if changed else " (the document did not change)")
            + (("\nprinted:\n" + "\n".join(log)[-800:]) if log else "")
            + f"  [{ms:.0f} ms]")


def _register(h):
    """A saved herder is also a tool of its own: h_<name>."""
    params = {"path": ("string", f"the {APPS[h['app']][1]} file")}
    for k, spec in (h.get("params") or {}).items():
        params[k] = (spec[0], spec[1]) if isinstance(spec, (list, tuple)) and len(spec) >= 2 else ("string", str(spec))
    tapp = {"words": "words", "sheets": "sheets", "slides": "slides"}[h["app"]]

    def fn(ctx, path, **args):
        return execute(ctx, _get(h["name"]), path, args)
    TOOLS["h_" + h["name"]] = Tool("h_" + h["name"], tapp, 2, f"[herder] {h['description']}", params, ("path",), fn, None)


def load_all():
    for k in [k for k in TOOLS if k.startswith("h_")]:
        TOOLS.pop(k)
    for h in library().values():
        if h.get("app") in APPS:
            _register(h)


# ------------------------------------------------------------------ tools

@tool("herder_help", "files", 1, "How to write a herder (a micro tool of your own) for words, sheets or slides: "
      "the document's structure and the rules.",
      {"app": ("string", "words, sheets or slides", ("words", "sheets", "slides"))}, ("app",))
def herder_help(ctx, app):
    if app not in HELP:
        raise ToolError("app must be words, sheets or slides")
    ex = next((h for h in library().values() if h["app"] == app), None)
    return HELP[app] + "\n\n" + CONTRACT + (f"\n\nExample ({ex['name']}):\n{ex['code']}" if ex else "")


@tool("herder_make", "files", 2, "Write a herder: a small Python function of your own that changes a Sheep document "
      "in a way the other tools cannot. Read herder_help first. It is saved and can be reused as h_<name>.",
      {"name": ("string", "short name, lower_case_with_underscores"),
       "app": ("string", "words, sheets or slides", ("words", "sheets", "slides")),
       "description": ("string", "one line: what it does"),
       "code": ("string", "Python source defining def herd(doc, args):"),
       "params": ("object", "optional: its arguments, e.g. {\"column\": [\"string\", \"column letter\"]}"),
       "asks": ("array", "2-4 short phrases a person would say to use it, e.g. [\"sort by price\"]")},
      ("name", "app", "description", "code"))
def herder_make(ctx, name, app, description, code, params=None, asks=None):
    name = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if not NAME.match(name):
        raise ToolError("name must be lower_case letters, digits and underscores, starting with a letter")
    if app not in APPS:
        raise ToolError("app must be words, sheets or slides")
    lib = library()
    if name in lib and lib[name]["builtin"]:
        raise ToolError(f"'{name}' is a built-in herder; pick another name")
    bad = check(code)
    if bad:
        raise ToolError("the herder was not saved:\n- " + "\n- ".join(bad))
    if params is not None and not isinstance(params, dict):
        raise ToolError("params must be an object like {\"column\": [\"string\", \"column letter\"]}")
    USER_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "app": app, "description": str(description)[:200], "params": params or {},
            "created": time.strftime("%Y-%m-%d %H:%M"), "by": ctx.by,
            "asks": [str(a).lower()[:40] for a in (asks if isinstance(asks, list) else [])][:4], "proven": False}
    (USER_DIR / f"{name}.py").write_text(code, encoding="utf-8")
    (USER_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    load_all()
    verb = "updated" if name in lib else "saved"
    return f"herder '{name}' {verb}. Run it with herder_run (name, path) or as h_{name}."


@tool("herder_run", "files", 2, "Run a herder on a document.",
      {"name": ("string", "the herder's name"), "path": ("string", "the .sdoc, .ssheet or .sdeck file"),
       "args": ("object", "its arguments, if it takes any")}, ("name", "path"))
def herder_run(ctx, name, path, args=None):
    return execute(ctx, _get(str(name).strip()), path, args if isinstance(args, dict) else {})


@tool("herder_list", "files", 1, "List the herders (micro tools) that exist, optionally for one app.",
      {"app": ("string", "words, sheets or slides")})
def herder_list(ctx, app=""):
    hs = [h for h in library().values() if not app or h["app"] == app]
    if not hs:
        return "no herders yet" + (f" for {app}" if app else "") + "; write one with herder_make"
    return "\n".join(f"{h['name']} ({h['app']}{', built-in' if h['builtin'] else ''}): {h['description']}"
                     + (f"  args: {', '.join(h.get('params') or {})}" if h.get("params") else "") for h in hs)


@tool("herder_show", "files", 1, "Show a herder's code.", {"name": ("string", "the herder's name")}, ("name",))
def herder_show(ctx, name):
    h = _get(name)
    return f"# {h['name']} ({h['app']}): {h['description']}\n{h['code']}"


@tool("herder_delete", "files", 2, "Delete a herder you made.", {"name": ("string", "the herder's name")},
      ("name",), confirm="always")
def herder_delete(ctx, name):
    h = _get(name)
    if h["builtin"]:
        raise ToolError(f"'{name}' is built in and cannot be deleted")
    for ext in (".py", ".json"):
        try:
            os.remove(USER_DIR / f"{name}{ext}")
        except OSError:
            pass
    load_all()
    return f"deleted herder '{name}'"


load_all()
