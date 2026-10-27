"""Sheep Presentations: .sdeck decks, 16:9, with a handful of our own themes.

Slides are numbered from 1, as a person counts them. A bullet indented by two
spaces is a sub-point. Pictures and free text boxes sit on a slide as items
placed in percent of the slide, so they land in the same spot on any screen
and in the exported .pptx.
"""
from . import tool, overwrite_check
from .. import formats as F
from .words import confirm_create, open_doc, save_doc, start_new
from ..workspace import ToolError

LAYOUTS = ("bullets", "title", "section", "title_only", "blank")


def _slide(d, n):
    try:
        i = int(n) - 1
    except (TypeError, ValueError):
        raise ToolError("slide must be a slide number from 1")
    if not 0 <= i < len(d["slides"]):
        raise ToolError(f"there is no slide {n}; the deck has {len(d['slides'])} slides")
    return i, d["slides"][i]


def _bullets(b):
    """Bullet strings; a leading two-space indent (a sub-point) is kept, a typed marker is not."""
    if b is None:
        return []
    lines = [x.rstrip() for x in b.splitlines() if x.strip()] if isinstance(b, str) else [str(x) for x in b]
    return [x[:len(x) - len(x.lstrip())] + F.item(x.lstrip()) for x in lines]


def model(path, ws):
    class C:
        pass
    c = C()
    c.ws = ws
    _, d = open_doc(c, path, "presentations")
    return {"path": path, **d, "themes": F.THEMES}


# ------------------------------------------------------------------ tools

@tool("slides_create", "slides", 2, "Create a new Sheep Presentations deck (.sdeck) with a title slide.",
      {"path": ("string", "new file, e.g. pitch.sdeck"), "title": ("string", "title of the first slide"),
       "subtitle": ("string", "subtitle (optional)"),
       "theme": ("string", "light, dark, midnight, forest, sunset or mono", tuple(F.THEMES)),
       "overwrite": ("boolean", "true only to replace an existing deck")},
      ("path",), confirm=confirm_create("presentations"))
def slides_create(ctx, path, title="", subtitle="", theme="light", overwrite=False):
    p, msg = start_new(ctx, path, "presentations", overwrite)
    if msg:
        return msg
    d = F.new("presentations", theme=theme if theme in F.THEMES else "light")
    if title or subtitle:
        d["slides"].append({"layout": "title", "title": title, "subtitle": subtitle, "bullets": [], "notes": "",
                            "items": []})
    save_doc(ctx, p, d, f"create {ctx.ws.rel(p)}")
    return f"created {ctx.ws.rel(p)} ({len(d['slides'])} slide{'s' if len(d['slides']) != 1 else ''}, theme {d['theme']})"


@tool("slides_read", "slides", 1, "Read a deck: every slide's title, bullets and notes.",
      {"path": ("string", "the .sdeck file")}, ("path",))
def slides_read(ctx, path):
    _, d = open_doc(ctx, path, "presentations")
    return F.deck_outline(d)


@tool("slides_add", "slides", 2, "Add a slide with a title and bullet points (indent a bullet with two spaces "
      "for a sub-point). Optionally a picture and speaker notes.",
      {"path": ("string", "the .sdeck file"), "title": ("string", "slide title"),
       "bullets": ("array", "bullet points, one string each"),
       "layout": ("string", "bullets, title, section or title_only", LAYOUTS[:4]),
       "image": ("string", "a picture from the workspace to place on the slide"),
       "notes": ("string", "speaker notes"), "after": ("integer", "insert after this slide number; omit for the end")},
      ("path",))
def slides_add(ctx, path, title="", bullets=None, layout="bullets", image="", notes="", after=None):
    p, d = open_doc(ctx, path, "presentations")
    bl = _bullets(bullets)
    # a planned step without a title stopped a whole research deck after its title slide
    # (journey J4, 2026-10-21): the first bullet becomes the title, or the slide is numbered
    if not str(title or "").strip():
        title = bl.pop(0).strip() if len(bl) > 1 else f"Slide {len(d['slides']) + 1}"
    s = {"layout": layout if layout in LAYOUTS else "bullets", "title": title, "bullets": bl, "notes": notes or "",
         "items": []}
    if image:
        ctx.ws.path(image, must_exist=True)
        s["items"].append({"kind": "image", "src": image, **({"x": 56, "y": 24, "w": 40} if bl else
                                                              {"x": 15, "y": 24, "w": 70})})
    if after is None or after == "":
        d["slides"].append(s)
        n = len(d["slides"])
    else:
        k = max(0, min(int(after), len(d["slides"])))
        d["slides"].insert(k, s)
        n = k + 1
    save_doc(ctx, p, d, f"add slide to {path}")
    return f"added slide {n}: {title[:60]}"


TITLE_SCALE = (0.5, 2.5)


def _title_scale(now, size):
    """The title's size against the theme's: "bigger" / "smaller" step by a quarter, a number is the scale."""
    w = str(size).strip().lower()
    if w in ("bigger", "larger", "big", "large", "up", "increase"):
        v = now * 1.25
    elif w in ("smaller", "small", "down", "decrease"):
        v = now / 1.25
    elif w in ("normal", "default", "reset"):
        v = 1.0
    else:
        try:
            v = float(w.rstrip("x%")) / (100 if w.endswith("%") else 1)
        except ValueError:
            raise ToolError("title_size is bigger, smaller, normal or a scale like 1.5")
    return round(min(max(v, TITLE_SCALE[0]), TITLE_SCALE[1]), 3)


@tool("slides_set", "slides", 2, "Change a slide's title, subtitle, bullet points, notes or title size.",
      {"path": ("string", "the .sdeck file"), "slide": ("integer", "slide number from 1 (0: every slide, for title_size)"),
       "title": ("string", "new title"), "subtitle": ("string", "new subtitle"),
       "bullets": ("array", "new bullet points"), "notes": ("string", "speaker notes"),
       "title_size": ("string", "bigger, smaller, normal, or a scale like 1.5 (1 = the theme's size)")}, ("path", "slide"))
def slides_set(ctx, path, slide, title=None, subtitle=None, bullets=None, notes=None, title_size=None):
    p, d = open_doc(ctx, path, "presentations")
    if title_size is not None and str(slide).strip() in ("0", "all"):
        # every slide's title (2026-10-14 live run: asked for a bigger title, the 3B changed the theme)
        for s in d["slides"]:
            s["title_scale"] = _title_scale(float(s.get("title_scale") or 1.0), title_size)
        save_doc(ctx, p, d, f"title size on every slide of {path}")
        return f"title size set on all {len(d['slides'])} slides ({d['slides'][0]['title_scale']}x the theme's)"
    _, s = _slide(d, slide)
    if title_size is not None:
        s["title_scale"] = _title_scale(float(s.get("title_scale") or 1.0), title_size)
    if title is not None:
        s["title"] = title
    if subtitle is not None:
        s["subtitle"] = subtitle
    if bullets is not None:
        s["bullets"] = _bullets(bullets)
        if s.get("layout") in ("title_only", "blank"):
            s["layout"] = "bullets"
    if notes is not None:
        s["notes"] = notes
    save_doc(ctx, p, d, f"edit slide {slide} of {path}")
    return f"slide {slide} updated" + (f" (title {s['title_scale']}x the theme's size)" if title_size is not None else "")


@tool("slides_delete", "slides", 2, "Delete a slide.", {"path": ("string", "the .sdeck file"),
      "slide": ("integer", "slide number from 1")}, ("path", "slide"))
def slides_delete(ctx, path, slide):
    p, d = open_doc(ctx, path, "presentations")
    i, _ = _slide(d, slide)
    d["slides"].pop(i)
    save_doc(ctx, p, d, f"delete slide {slide} of {path}")
    return f"deleted slide {slide}; the deck now has {len(d['slides'])} slides"


@tool("slides_move", "slides", 2, "Move a slide to a new position.",
      {"path": ("string", "the .sdeck file"), "slide": ("integer", "slide to move"),
       "to": ("integer", "its new number")}, ("path", "slide", "to"))
def slides_move(ctx, path, slide, to):
    p, d = open_doc(ctx, path, "presentations")
    i, s = _slide(d, slide)
    j = max(1, min(int(to), len(d["slides"]))) - 1
    d["slides"].insert(j, d["slides"].pop(i))
    save_doc(ctx, p, d, f"move slide {slide} to {j + 1} in {path}")
    return f"moved slide {slide} to position {j + 1}"


@tool("slides_set_theme", "slides", 2, "Apply a colour theme to the whole deck.",
      {"path": ("string", "the .sdeck file"),
       "theme": ("string", "light, dark, midnight, forest, sunset or mono", tuple(F.THEMES))}, ("path", "theme"))
def slides_set_theme(ctx, path, theme):
    if theme not in F.THEMES:
        raise ToolError(f"themes are: {', '.join(F.THEMES)}")
    p, d = open_doc(ctx, path, "presentations")
    d["theme"] = theme
    save_doc(ctx, p, d, f"theme {theme} on {path}")
    return f"applied theme {theme} to {len(d['slides'])} slides"


@tool("slides_add_image", "slides", 2, "Place a picture on a slide.",
      {"path": ("string", "the .sdeck file"), "slide": ("integer", "slide number from 1"),
       "image": ("string", "picture file in the workspace"),
       "x": ("number", "left edge, % of slide width"), "y": ("number", "top edge, % of slide height"),
       "width": ("number", "width, % of slide width")}, ("path", "slide", "image"))
def slides_add_image(ctx, path, slide, image, x=55, y=25, width=40):
    p, d = open_doc(ctx, path, "presentations")
    _, s = _slide(d, slide)
    ctx.ws.path(image, must_exist=True)
    s.setdefault("items", []).append({"kind": "image", "src": image, "x": float(x), "y": float(y), "w": float(width)})
    save_doc(ctx, p, d, f"picture on slide {slide} of {path}")
    return f"placed {image} on slide {slide}"


@tool("slides_add_text", "slides", 2, "Place a free text box on a slide.",
      {"path": ("string", "the .sdeck file"), "slide": ("integer", "slide number from 1"),
       "text": ("string", "the text"), "x": ("number", "left edge, % of width"), "y": ("number", "top edge, % of height"),
       "width": ("number", "width, % of slide width"), "size": ("number", "font size in points")},
      ("path", "slide", "text"))
def slides_add_text(ctx, path, slide, text, x=10, y=80, width=80, size=18):
    p, d = open_doc(ctx, path, "presentations")
    _, s = _slide(d, slide)
    s.setdefault("items", []).append({"kind": "text", "text": text, "x": float(x), "y": float(y),
                                      "w": float(width), "h": 10.0, "size": float(size)})
    save_doc(ctx, p, d, f"text on slide {slide} of {path}")
    return f"added a text box to slide {slide}"


@tool("slides_put", "slides", 2, "Editor: replace one slide with the given slide object.",
      {"path": ("string", "the .sdeck file"), "slide": ("integer", "slide number from 1"),
       "data": ("object", "the slide")}, ("path", "slide", "data"))
def slides_put(ctx, path, slide, data):
    p, d = open_doc(ctx, path, "presentations")
    i, _ = _slide(d, slide)
    if not isinstance(data, dict):
        raise ToolError("data must be a slide object")
    d["slides"][i] = data
    save_doc(ctx, p, d, f"edit slide {slide} of {path}")
    return f"slide {slide} saved"
