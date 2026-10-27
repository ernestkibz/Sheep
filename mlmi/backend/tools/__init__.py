"""The tool registry: one list of operations, used by the agent AND the editors.

A tool is a name, a short description a small model can act on, a JSON schema,
the tier a model must have earned to be given it, and a function. The editors
call the same functions through /api/op, so a click and a model's call are the
same operation, land in the same file, and share one undo.

Tiers (Sheep's checker grants them):
  0 chat      no tools
  1 reader    read-only tools
  2 editor    create and edit documents, sheets, slides
  3 producer  video, motion, moving and deleting files
"""
import json

from ..workspace import ToolError

TOOLS = {}
APPS = ("files", "words", "sheets", "slides", "studio", "web", "media", "mail")


class Tool:
    def __init__(self, name, app, tier, desc, params, required, fn, confirm):
        self.name, self.app, self.tier, self.desc = name, app, tier, desc
        self.params, self.required, self.fn, self.confirm = params, required, fn, confirm

    def schema(self):
        props = {}
        for k, spec in self.params.items():
            typ, desc = spec[0], spec[1]
            p = {"type": typ, "description": desc}
            if typ == "array":
                p["items"] = spec[2] if len(spec) > 2 else {"type": "string"}
            elif len(spec) > 2 and isinstance(spec[2], (list, tuple)):
                p["enum"] = list(spec[2])
            props[k] = p
        return {"type": "function", "function": {
            "name": self.name, "description": self.desc,
            "parameters": {"type": "object", "properties": props, "required": list(self.required)}}}


def tool(name, app, tier, desc, params=None, required=(), confirm=None):
    """confirm: None, "always" (deletes), or a function(ws, args) -> reason|None."""
    def deco(fn):
        TOOLS[name] = Tool(name, app, tier, desc, params or {}, required, fn, confirm)
        return fn
    return deco


class Ctx:
    """What a tool is given: the workspace and who is acting. An agent run keeps
    one Ctx for the whole job, so tools can leave it notes (appended to their
    result), Microsoft files the job asked for (exported when it finishes), and
    the files this job itself wrote (overwriting those needs no question)."""
    def __init__(self, ws, by="user"):
        self.ws, self.by = ws, by
        self.notes = []
        self.exports = {}      # Sheep file -> the Microsoft file the job asked for
        self.made = set()      # files written by this job
        self.cancel_event = None
        self.register_cancel = None
        self.progress = None
        self.state = None      # the server state, for tools that need the loaded model or Settings


# Not offered to the agent: the editors' whole-object writes (an agent edits
# through the specific tools), and library chores the UI does. Every tool
# offered costs prompt tokens on every step.
NOT_OFFERED = {"words_put", "slides_put", "studio_put", "herder_show", "herder_delete", "create_folder"}


def offered(tier, apps=None):
    """Tool names a model of this tier may use, limited to the given apps."""
    apps = set(apps or APPS)
    return [t.name for t in TOOLS.values() if t.tier <= tier and t.app in apps and t.name not in NOT_OFFERED]


def _coerce(spec, v):
    """Bring a value to the type its schema names. Small models send a list as the
    string "['milk', 'eggs']" and a boolean as "true"; taken literally, the first
    became ONE bullet reading ['milk', 'eggs', 'bread'] on the first live run."""
    typ = spec[0]
    if typ == "array" and isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return json.loads(s)
            except ValueError:
                try:
                    import ast
                    out = ast.literal_eval(s)
                    if isinstance(out, (list, tuple)):
                        return list(out)
                except (ValueError, SyntaxError):
                    pass
        parts = [p.strip() for p in (s.splitlines() if "\n" in s else s.split(","))]
        return [p for p in parts if p]
    if typ == "object" and isinstance(v, str):
        try:
            out = json.loads(v)
            return out if isinstance(out, dict) else v
        except ValueError:
            return v
    if typ == "boolean" and isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "on")
    if typ == "integer" and isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    if typ == "integer" and isinstance(v, float) and v.is_integer():
        return int(v)
    if typ == "number" and isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return v
    if typ == "string" and isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return v


def schemas(names):
    return [TOOLS[n].schema() for n in names if n in TOOLS]


def needs_confirm(ws, name, args, ctx=None):
    """Why this call needs the person's yes, or None. Replacing a file this same
    job wrote is not a question worth asking."""
    t = TOOLS.get(name)
    if not t or not t.confirm:
        return None
    if t.confirm == "always":
        return f"{name} {json.dumps(args)}"
    try:
        reason = t.confirm(ws, args)
    except ToolError:
        return None
    if reason and ctx is not None and reason.startswith("replace existing ")             and reason[len("replace existing "):] in ctx.made:
        return None
    return reason


def run(ws, name, args, by="user", ctx=None):
    """Run one tool; returns its text result. Raises ToolError with a usable message."""
    t = TOOLS.get(name)
    if not t:
        raise ToolError(f"there is no tool named '{name}'. Tools: {', '.join(sorted(TOOLS))}")
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    missing = [k for k in t.required if k not in args or args[k] in (None, "")]
    if missing:
        raise ToolError(f"{name} needs: {', '.join(missing)}")
    unknown = [k for k in args if k not in t.params]
    for k in unknown:          # small models add stray keys; drop them rather than fail
        args.pop(k)
    for k, v in list(args.items()):
        args[k] = _coerce(t.params[k], v)
    ctx = ctx or Ctx(ws, by)
    ctx.notes = []
    if unknown:   # said, not silent: a misspelt key otherwise vanished and the model never learnt why
        ctx.notes.append(f"(ignored {', '.join(unknown)}: {name} takes {', '.join(t.params)})")
    out = t.fn(ctx, **args)
    return out + (" " + " ".join(ctx.notes) if ctx.notes else "")


def overwrite_check(key):
    """confirm hook: ask before replacing a file that already exists."""
    def check(ws, args):
        p = ws.path(args.get(key, ""))
        return f"replace existing {args.get(key)}" if p.exists() else None
    return check


# importing the modules registers their tools
from . import files, words, sheets, slides, studio, transfer, herders, web, media, vision, mail  # noqa: E402,F401
