"""Reading a planner's reply (Nemotron's) into exact tool calls, repairing near-misses.

Split out of agent.py (2026-10-21); every name is re-exported from the agent package."""
import json
import re
import uuid
from ..workspace import ModelSelectionRequired, ToolError


PLANNER_CALLS = """You turn a job into the exact tool calls a local assistant will run on the user's machine.
Reply with ONLY a JSON array, no prose before or after:
[{"tool": "<tool name>", "args": {<arguments>}}, ...]
Rules:
- Use only the tools given, with their exact parameter names and types.
- Write the real content yourself: titles, paragraphs, bullet points, numbers. Specific and useful, never placeholders.
- Fewest calls: one call per slide; one call for a whole list; sheets_create takes headers, rows and total at once.
- For video, pass attached or selected photo paths in make_video.source_images when animating a photo; use Studio image layers and studio_render for photo montages or exact original photos. Respect explicit text-only instructions.
- New files get short names with the right extension: .sdoc (document), .ssheet (workbook), .sdeck (deck).
- Never ask the user for details: choose sensible ones yourself and do the whole job.
- Only what the user asked for: no extra files, no exports unless asked, no reading files unless the job is about one.
- For a photo montage/slideshow, create a plain Studio project, put each selected picture on its own image layer, and render it. Do not add generated scenes or a title template unless asked.
- A video: start from a template (studio_create with template, title, subtitle), then fill the rest of the length with
  more scenes at later start times (studio_template or studio_add_text), each 3-5 seconds, so nothing is empty; use the
  name and words the user gave; end with studio_render."""


def _unescape(v):
    """Nemotron sometimes writes a narrow space as the six characters \\u202f inside a string; a slide
    showed them as typed (journey J4, 2026-10-22). Decode such escapes in every string argument."""
    if isinstance(v, str):
        return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), v)
    if isinstance(v, list):
        return [_unescape(x) for x in v]
    if isinstance(v, dict):
        return {k: _unescape(x) for k, x in v.items()}
    return v


def _plan_calls(text, allowed):
    """Nemotron's plan as [{"tool", "args"}], only tools on offer; [] if it is not that."""
    # the first JSON array that parses: slicing first "[" to last "]" broke on any
    # bracket in the prose around it
    t, v, dec = (text or "").strip(), None, json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch == "[":
            try:
                v, _ = dec.raw_decode(t[i:])
            except ValueError:
                continue
            if isinstance(v, list) and any(isinstance(c, dict) and "tool" in c for c in v):
                break
            v = None
    if v is None:
        return []
    out = []
    for c in v if isinstance(v, list) else []:
        if isinstance(c, dict) and c.get("tool") in allowed and isinstance(c.get("args", {}), dict):
            out.append({"tool": c["tool"], "args": _unescape(c.get("args") or {})})
    return out[:40]


_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I)


_FUNC_TAG = re.compile(r"<function=([A-Za-z_]\w*)>\s*(\{.*?\})?\s*(?:</function>)?", re.S)


def _looks_like_call(text):
    return '"name"' in text or "<function=" in text


def _call(name, a):
    if isinstance(a, str):          # arguments sent as a JSON string
        try:
            a = json.loads(a)
        except ValueError:
            a = {}
    return {"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
            "function": {"name": name, "arguments": a if isinstance(a, dict) else {}}}


def _repair_call(text, known=None):
    """A tool call the engine could not parse, mended when the fault is mechanical.
    Seen from small models: missing closing braces (Llama 3.2 3B, 2026-09-30), the
    call inside a ```json fence, a [{...}] list, a <function=name>{...} tag, or a
    sentence before the JSON. Only a name in `known` counts, so prose that happens
    to hold JSON is never run. Returns an OpenAI-style call, or None."""
    t = _FENCE.sub("", text.strip()).strip()
    m = _FUNC_TAG.search(t)
    if m and (known is None or m.group(1) in known):
        try:
            return _call(m.group(1), json.loads(m.group(2)) if m.group(2) else {})
        except ValueError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch not in "[{":
            continue
        for k in range(0, 4):
            try:
                v, _ = dec.raw_decode(t[i:] + "}" * k)
            except ValueError:
                continue
            if isinstance(v, list):
                v = next((x for x in v if isinstance(x, dict)), None)
            if isinstance(v, dict) and isinstance(v.get("name"), str) and (known is None or v["name"] in known):
                return _call(v["name"], v.get("parameters", v.get("arguments", {})))
            break
    return None


def _args(raw):
    if isinstance(raw, dict):
        v = raw
    else:
        try:
            v = json.loads(raw or "{}")
        except (ValueError, TypeError):
            raise ToolError(f"the arguments were not valid JSON: {str(raw)[:120]}")
    if not isinstance(v, dict):
        raise ToolError("the arguments must be a JSON object")
    # Llama 3.2 3B sends {"properties": {"path": ...}}, or some arguments at the top
    # and the rest one level down under "properties": lift them up
    for key in ("properties", "parameters", "arguments", "args"):
        if isinstance(v.get(key), dict):
            inner = v.pop(key)
            v = {**inner, **v}
    return v
