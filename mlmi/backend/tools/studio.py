"""Sheep Studio: video and motion as one app (.sstudio).

One project, one timeline. Every element -- a clip, a picture, text, a shape,
music -- is a layer with a start, a length, a place on the frame (x, y, w, h
in percent of the frame, x and y at the centre), an entrance and an exit.
Layers later in the list are drawn on top. The same layer model is drawn by
the editor's live preview (studio.js) and by the renderer here, with the same
easing, so what plays in the app is what the MP4 shows.

Rendering: frames are composed with Pillow, clips are read through ffmpeg,
and the result is encoded on the GPU (NVENC) when there is one, libx264
otherwise; audio layers and clip sound are mixed with fades, and music dips
under clip sound when asked to.
"""
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import tool
from .. import formats as F
from ..paths import REPO
from ..workspace import ToolError
from .words import confirm_create, open_doc, save_doc, start_new

FONT = REPO / "mlmi" / "desktop" / "ui" / "fonts" / "Nunito.ttf"
FORMATS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
MOVES = ("none", "fade", "pop", "slide-up", "slide-left", "zoom", "wipe", "typewriter")
KINDS = ("clip", "image", "text", "shape", "audio")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")


# ------------------------------------------------------------------ the shared motion maths (studio.js mirrors it)

def _clamp(u):
    return 0.0 if u < 0 else 1.0 if u > 1 else u


def ease_out(u):
    return 1 - (1 - u) ** 3


def ease_back(u):
    c1 = 1.70158
    return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2


def ease_in_out(u):
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def state(layer, t):
    """Where and how a layer is at time t (seconds), or None when it is not shown."""
    start, dur = float(layer.get("start", 0)), float(layer.get("dur", 3))
    local = t - start
    if local < 0 or local >= dur:
        return None
    s = {"x": float(layer.get("x", 50)), "y": float(layer.get("y", 50)), "w": float(layer.get("w", 60)),
         "h": float(layer.get("h", 20)), "scale": 1.0, "rot": float(layer.get("rotate", 0)),
         "op": float(layer.get("opacity", 1)), "wipe": 1.0, "chars": None}
    keys = sorted(layer.get("keys") or [], key=lambda k: k.get("t", 0))
    if keys:                                     # keyframes, relative to the layer's start
        prev = {"t": 0, **{k: s[v] for k, v in (("x", "x"), ("y", "y"), ("scale", "scale"), ("rotate", "rot"), ("opacity", "op"))}}
        for k in keys:
            if local <= k.get("t", 0):
                u = ease_in_out(_clamp((local - prev["t"]) / max(1e-6, k.get("t", 0) - prev["t"])))
                for a, b in (("x", "x"), ("y", "y"), ("scale", "scale"), ("rotate", "rot"), ("opacity", "op")):
                    if a in k:
                        s[b] = prev.get(a, s[b]) + (k[a] - prev.get(a, s[b])) * u
                break
            prev = {**prev, **k}
        else:
            for a, b in (("x", "x"), ("y", "y"), ("scale", "scale"), ("rotate", "rot"), ("opacity", "op")):
                if a in prev:
                    s[b] = prev[a]
    for side, u in (("in", local / max(1e-6, float((layer.get("in") or {}).get("dur", 0.6)))),
                    ("out", (dur - local) / max(1e-6, float((layer.get("out") or {}).get("dur", 0.5))))):
        mv = (layer.get(side) or {}).get("move", "none")
        if mv == "none" or u >= 1:
            continue
        u = _clamp(u)
        e = ease_out(u)
        if mv == "fade":
            s["op"] *= e
        elif mv == "pop":
            s["scale"] *= max(0.0, ease_back(u)) if side == "in" else e
            s["op"] *= min(1.0, u * 3)
        elif mv == "slide-up":
            s["y"] += (1 - e) * 12
            s["op"] *= e
        elif mv == "slide-left":
            s["x"] += (1 - e) * 15
            s["op"] *= e
        elif mv == "zoom":
            s["scale"] *= 0.6 + 0.4 * e
            s["op"] *= e
        elif mv == "wipe":
            s["wipe"] = min(s["wipe"], e)
        elif mv == "typewriter":
            s["chars"] = u
    return s


# ------------------------------------------------------------------ the project

def new_project(fmt="16:9", duration=10.0, bg="0B0B0B"):
    w, h = FORMATS.get(fmt, FORMATS["16:9"])
    return {"sheep": "studio", "v": 1, "format": fmt if fmt in FORMATS else "16:9", "w": w, "h": h, "fps": 30,
            "duration": float(duration or 10), "bg": str(bg or "0B0B0B").strip("#").upper()[:6], "layers": []}


def length(d):
    ends = [float(l.get("start", 0)) + float(l.get("dur", 0)) for l in d["layers"]]
    return max([float(d.get("duration", 10))] + ends)


def model(path, ws):
    class C:
        pass
    c = C()
    c.ws = ws
    _, d = open_doc(c, path, "studio")
    return {"path": path, **d, "length": length(d), "moves": MOVES, "formats": list(FORMATS)}


def _layer(d, n):
    try:
        i = int(n) - 1
    except (TypeError, ValueError):
        raise ToolError("layer must be a layer number from 1 (see studio_read)")
    if not 0 <= i < len(d["layers"]):
        raise ToolError(f"there is no layer {n}; the project has {len(d['layers'])} layers")
    return i, d["layers"][i]


def _move(name, default):
    name = str(name or default).strip().lower().replace("_", "-").replace(" ", "-")
    return name if name in MOVES else default


def _color(c, default):
    c = str(c or default).strip().lstrip("#").upper()
    return c[:6] if len(c) >= 6 and all(ch in "0123456789ABCDEF" for ch in c[:6]) else default


def _probe(p):
    """(duration seconds, has sound) of a media file, or (None, False)."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None, False
    try:
        out = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(p)],
                             capture_output=True, text=True, timeout=30,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        j = json.loads(out or "{}")
        dur = float(j.get("format", {}).get("duration") or 0) or None
        return dur, any(s.get("codec_type") == "audio" for s in j.get("streams", []))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None, False


def outline(d):
    lines = [f"{d['format']} {d['w']}x{d['h']}, {length(d):.1f} s, background #{d['bg']}"]
    for i, l in enumerate(d["layers"], 1):
        what = l.get("text") or l.get("src") or l.get("shape") or ""
        lines.append(f"{i}. {l['kind']} {float(l.get('start', 0)):.1f}-{float(l.get('start', 0)) + float(l.get('dur', 0)):.1f}s"
                     f" {str(what)[:60]!r} in={(l.get('in') or {}).get('move', 'none')} out={(l.get('out') or {}).get('move', 'none')}")
    return "\n".join(lines) if d["layers"] else lines[0] + "\n(no layers yet)"


# ------------------------------------------------------------------ templates

def template_layers(name, d, title="", subtitle="", logo="", start=0.0):
    """Quick templates: a whole look in one step."""
    portrait = d["h"] > d["w"]
    t0 = float(start or 0)
    L = []
    if name == "title-intro":
        L += [{"kind": "shape", "shape": "rect", "start": t0, "dur": 5, "x": 50, "y": 62 if not portrait else 58, "w": 12, "h": 0.8,
               "fill": "A3A3A3", "radius": 4, "in": {"move": "wipe", "dur": 0.7}, "out": {"move": "fade", "dur": 0.5}},
              {"kind": "text", "text": title or "Your title", "start": t0 + 0.2, "dur": 4.8, "x": 50, "y": 46, "w": 86, "h": 20,
               "size": 120 if not portrait else 110, "color": "FFFFFF", "weight": 900, "in": {"move": "pop", "dur": 0.7}, "out": {"move": "fade", "dur": 0.5}},
              {"kind": "text", "text": subtitle or "A short line about it", "start": t0 + 0.8, "dur": 4.2, "x": 50, "y": 70 if not portrait else 66,
               "w": 80, "h": 10, "size": 48, "color": "AAB6D8", "weight": 600, "in": {"move": "slide-up", "dur": 0.6}, "out": {"move": "fade", "dur": 0.5}}]
    elif name == "logo-reveal":
        if logo:
            L.append({"kind": "image", "src": logo, "start": t0, "dur": 4, "x": 50, "y": 42, "w": 30 if not portrait else 50, "h": 30,
                      "in": {"move": "zoom", "dur": 0.9}, "out": {"move": "fade", "dur": 0.5}})
        else:
            L.append({"kind": "shape", "shape": "circle", "start": t0, "dur": 4, "x": 50, "y": 42, "w": 14 if not portrait else 26, "h": 14,
                      "fill": "A3A3A3", "in": {"move": "pop", "dur": 0.8}, "out": {"move": "fade", "dur": 0.5}})
        L.append({"kind": "text", "text": title or "Brand", "start": t0 + 0.7, "dur": 3.3, "x": 50, "y": 68, "w": 80, "h": 12,
                  "size": 96, "color": "FFFFFF", "weight": 900, "in": {"move": "wipe", "dur": 0.8}, "out": {"move": "fade", "dur": 0.5}})
    elif name == "outro":
        L += [{"kind": "text", "text": title or "Thanks for watching", "start": t0, "dur": 5, "x": 50, "y": 44, "w": 86, "h": 16,
               "size": 100, "color": "FFFFFF", "weight": 900, "in": {"move": "fade", "dur": 0.8}, "out": {"move": "fade", "dur": 0.8}},
              {"kind": "text", "text": subtitle or "Follow for more", "start": t0 + 0.6, "dur": 4.4, "x": 50, "y": 60, "w": 80, "h": 10,
               "size": 52, "color": "7FA8FF", "weight": 700, "in": {"move": "slide-up", "dur": 0.6}, "out": {"move": "fade", "dur": 0.8}}]
        if logo:
            L.append({"kind": "image", "src": logo, "start": t0 + 1, "dur": 4, "x": 50, "y": 80, "w": 14, "h": 12,
                      "in": {"move": "pop", "dur": 0.6}, "out": {"move": "fade", "dur": 0.8}})
    elif name == "short":
        L += [{"kind": "text", "text": title or "Did you know?", "start": t0, "dur": 8, "x": 50, "y": 38, "w": 86, "h": 24,
               "size": 120, "color": "FFFFFF", "weight": 900, "in": {"move": "typewriter", "dur": 1.2}, "out": {"move": "fade", "dur": 0.5}},
              {"kind": "text", "text": subtitle or "One fact, in eight seconds.", "start": t0 + 1.4, "dur": 6.6, "x": 50, "y": 62, "w": 84, "h": 14,
               "size": 60, "color": "AAB6D8", "weight": 700, "in": {"move": "slide-up", "dur": 0.6}, "out": {"move": "fade", "dur": 0.5}},
              {"kind": "shape", "shape": "rect", "start": t0 + 0.4, "dur": 7.6, "x": 50, "y": 52, "w": 20, "h": 0.6,
               "fill": "A3A3A3", "radius": 4, "in": {"move": "wipe", "dur": 0.6}, "out": {"move": "fade", "dur": 0.5}}]
    elif name == "lower-third":
        y = 82 if not portrait else 78
        L += [{"kind": "shape", "shape": "rect", "start": t0, "dur": 5, "x": 26, "y": y, "w": 40, "h": 11, "fill": "0047AB",
               "opacity": 0.92, "radius": 12, "in": {"move": "slide-left", "dur": 0.5}, "out": {"move": "fade", "dur": 0.4}},
              {"kind": "text", "text": title or "Name Surname", "start": t0 + 0.2, "dur": 4.8, "x": 26, "y": y - 1.8, "w": 36, "h": 6,
               "size": 54, "color": "FFFFFF", "weight": 800, "align": "left", "in": {"move": "slide-left", "dur": 0.5}, "out": {"move": "fade", "dur": 0.4}},
              {"kind": "text", "text": subtitle or "Role or place", "start": t0 + 0.4, "dur": 4.6, "x": 26, "y": y + 2.6, "w": 36, "h": 4,
               "size": 34, "color": "C8D6FF", "weight": 600, "align": "left", "in": {"move": "slide-left", "dur": 0.5}, "out": {"move": "fade", "dur": 0.4}}]
    else:
        raise ToolError("templates are: title-intro, logo-reveal, outro, short, lower-third")
    return L


TEMPLATES = ("title-intro", "logo-reveal", "outro", "short", "lower-third")


# ------------------------------------------------------------------ tools

@tool("studio_create", "studio", 2, "Create a Sheep Studio video project (.sstudio). Optionally start it from a "
      "template (title-intro, logo-reveal, outro, short, lower-third) with a title and subtitle.",
      {"path": ("string", "new file, e.g. intro.sstudio"),
       "format": ("string", "16:9 (landscape), 9:16 (phone, shorts) or 1:1", tuple(FORMATS)),
       "duration": ("number", "length in seconds, default 10"),
       "background": ("string", "background colour, hex like 0B0B0B"),
       "template": ("string", "a quick template to start from", TEMPLATES),
       "title": ("string", "the template's title text"), "subtitle": ("string", "the template's second line"),
       "logo": ("string", "a picture in the workspace for logo-reveal or outro"),
       "overwrite": ("boolean", "true only to replace an existing project")},
      ("path",), confirm=confirm_create("studio"))
def studio_create(ctx, path, format="16:9", duration=10, background="0B0B0B", template="", title="", subtitle="", logo="",
                  overwrite=False):
    p, msg = start_new(ctx, path, "studio", overwrite)
    if msg:
        return msg
    d = new_project(format, duration, background)
    if template:
        if logo:
            ctx.ws.path(logo, must_exist=True)
        layers = template_layers(template, d, title, subtitle, logo)
        # a template that starts a project fills it: asked for 10 s, a 5 s template
        # left the second half empty (Nemotron's bakery intro, 2026-10-05)
        end = max(l["start"] + l["dur"] for l in layers)
        if d["duration"] > end:
            for l in layers:
                l["dur"] += d["duration"] - end
        d["layers"] += layers
    save_doc(ctx, p, d, f"create {ctx.ws.rel(p)}")
    return f"created {ctx.ws.rel(p)}: {d['format']}, {length(d):.0f} s" + (f", {len(d['layers'])} layers from {template}" if template else "")


@tool("studio_read", "studio", 1, "Read a Studio project: its format, length and numbered layers.",
      {"path": ("string", "the .sstudio file")}, ("path",))
def studio_read(ctx, path):
    _, d = open_doc(ctx, path, "studio")
    return outline(d)


@tool("studio_template", "studio", 2, "Add a quick template to a project at a time: title-intro, logo-reveal, outro, "
      "short or lower-third.",
      {"path": ("string", "the .sstudio file"), "template": ("string", "which template", TEMPLATES),
       "title": ("string", "main text"), "subtitle": ("string", "second line"),
       "logo": ("string", "a picture in the workspace (optional)"), "start": ("number", "when it starts, in seconds")},
      ("path", "template"))
def studio_template(ctx, path, template, title="", subtitle="", logo="", start=0):
    p, d = open_doc(ctx, path, "studio")
    if logo:
        ctx.ws.path(logo, must_exist=True)
    new = template_layers(template, d, title, subtitle, logo, start)
    d["layers"] += new
    save_doc(ctx, p, d, f"{template} in {path}")
    return f"added {template}: {len(new)} layers from {float(start or 0):.1f} s"


@tool("studio_add_text", "studio", 2, "Add text to the video at a time, with an entrance and exit move.",
      {"path": ("string", "the .sstudio file"), "text": ("string", "the words"),
       "start": ("number", "seconds from the start"), "duration": ("number", "how long it shows, seconds"),
       "size": ("number", "text size in pixels at 1080p, default 72"), "color": ("string", "hex colour, default FFFFFF"),
       "x": ("number", "centre across, % of width (50 = middle)"), "y": ("number", "centre down, % of height (50 = middle)"),
       "enter": ("string", "entrance move", MOVES), "exit": ("string", "exit move", MOVES)},
      ("path", "text"))
def studio_add_text(ctx, path, text, start=0, duration=3, size=72, color="FFFFFF", x=50, y=50, enter="fade", exit="fade"):
    p, d = open_doc(ctx, path, "studio")
    d["layers"].append({"kind": "text", "text": F.clean(text), "start": float(start or 0), "dur": float(duration or 3),
                        "size": float(size or 72), "color": _color(color, "FFFFFF"), "weight": 800, "x": float(x), "y": float(y),
                        "w": 86, "h": 18, "in": {"move": _move(enter, "fade"), "dur": 0.6}, "out": {"move": _move(exit, "fade"), "dur": 0.5}})
    save_doc(ctx, p, d, f"text in {path}")
    return f"added text as layer {len(d['layers'])}"


@tool("studio_add_media", "studio", 2, "Add a video clip, picture or music from the workspace to the timeline.",
      {"path": ("string", "the .sstudio file"), "file": ("string", "the media file in the workspace"),
       "start": ("number", "seconds from the start"), "duration": ("number", "seconds; omit for a clip's or song's own length"),
       "x": ("number", "centre across, %"), "y": ("number", "centre down, %"), "width": ("number", "width, % of the frame"),
       "height": ("number", "height, % of the frame"), "enter": ("string", "entrance move", MOVES),
       "exit": ("string", "exit move", MOVES), "volume": ("number", "sound level 0-1 for clips and music"),
       "duck": ("boolean", "music only: dip under clip sound")},
      ("path", "file"))
def studio_add_media(ctx, path, file, start=0, duration=None, x=50, y=50, width=100, height=100, enter="none", exit="none",
                     volume=1.0, duck=True):
    p, d = open_doc(ctx, path, "studio")
    src = ctx.ws.path(file, must_exist=True)
    ext = src.suffix.lower()
    kind = "clip" if ext in VIDEO_EXT else "image" if ext in IMAGE_EXT else "audio" if ext in AUDIO_EXT else None
    if not kind:
        raise ToolError(f"{file} is not a video, picture or sound file")
    own, _ = _probe(src) if kind != "image" else (None, False)
    dur = float(duration) if duration not in (None, "", 0) else (own or 5.0)
    layer = {"kind": kind, "src": ctx.ws.rel(src), "start": float(start or 0), "dur": dur, "x": float(x), "y": float(y),
             "w": float(width or 100), "h": float(height or 100), "trim": 0.0,
             "in": {"move": _move(enter, "none"), "dur": 0.6}, "out": {"move": _move(exit, "none"), "dur": 0.5}}
    if kind in ("clip", "audio"):
        layer["volume"] = float(volume if volume is not None else 1.0)
    if kind == "audio":
        layer["duck"] = bool(duck)
        layer["in"], layer["out"] = {"move": "fade", "dur": 1.0}, {"move": "fade", "dur": 1.5}
    if kind == "clip" and (width or 100) >= 100 and (height or 100) >= 100:
        d["layers"].insert(0, layer)            # a full-frame clip sits under text and pictures
        pos = 1
    else:
        d["layers"].append(layer)
        pos = len(d["layers"])
    save_doc(ctx, p, d, f"{kind} in {path}")
    return f"added {kind} {ctx.ws.rel(src)} as layer {pos}, {dur:.1f} s"


@tool("studio_add_shape", "studio", 2, "Add a shape (rect, circle or line) to the video, e.g. a bar behind text.",
      {"path": ("string", "the .sstudio file"), "shape": ("string", "rect, circle or line", ("rect", "circle", "line")),
       "start": ("number", "seconds from the start"), "duration": ("number", "seconds"),
       "x": ("number", "centre across, %"), "y": ("number", "centre down, %"), "width": ("number", "% of the frame"),
       "height": ("number", "% of the frame"), "color": ("string", "hex colour"), "opacity": ("number", "0-1"),
       "enter": ("string", "entrance move", MOVES), "exit": ("string", "exit move", MOVES)},
      ("path",))
def studio_add_shape(ctx, path, shape="rect", start=0, duration=3, x=50, y=50, width=30, height=10, color="A3A3A3", opacity=1.0,
                     enter="fade", exit="fade"):
    p, d = open_doc(ctx, path, "studio")
    d["layers"].append({"kind": "shape", "shape": shape if shape in ("rect", "circle", "line") else "rect",
                        "start": float(start or 0), "dur": float(duration or 3), "x": float(x), "y": float(y),
                        "w": float(width), "h": float(height), "fill": _color(color, "A3A3A3"), "opacity": float(opacity),
                        "radius": 12, "in": {"move": _move(enter, "fade"), "dur": 0.6}, "out": {"move": _move(exit, "fade"), "dur": 0.5}})
    save_doc(ctx, p, d, f"shape in {path}")
    return f"added {shape} as layer {len(d['layers'])}"


EDITABLE = {"start": float, "dur": float, "text": str, "x": float, "y": float, "w": float, "h": float, "size": float,
            "color": str, "fill": str, "opacity": float, "volume": float, "rotate": float, "trim": float, "weight": float,
            "align": str, "duck": bool, "speed": float}
SPEED = (0.25, 4.0)                 # a clip or sound plays at 1/4 to 4 times its own pace


def crop_box(layer):
    """The part of the source shown, as (x, y, w, h) in % of the source, or None for all of it."""
    c = layer.get("crop")
    if not c:
        return None
    try:
        x, y, w, h = (float(v) for v in (c if isinstance(c, (list, tuple)) else str(c).replace("%", "").split(",")))
    except (TypeError, ValueError):
        return None
    x, y = max(0.0, min(95.0, x)), max(0.0, min(95.0, y))
    w, h = max(5.0, min(100.0 - x, w)), max(5.0, min(100.0 - y, h))
    return None if (x, y, w, h) == (0.0, 0.0, 100.0, 100.0) else (x, y, w, h)


def _fades(layer):
    """Seconds the sound of a layer fades in and out: its own fade moves when set, else 0.3 / 0.6."""
    fi = (layer.get("in") or {}) if (layer.get("in") or {}).get("move") == "fade" else {}
    fo = (layer.get("out") or {}) if (layer.get("out") or {}).get("move") == "fade" else {}
    return float(fi.get("dur", 0.3)), float(fo.get("dur", 0.6))


def _speed(layer):
    return min(SPEED[1], max(SPEED[0], float(layer.get("speed", 1.0) or 1.0)))


def _atempo(speed):
    """ffmpeg's atempo takes 0.5 .. 2 a stage: a speed beyond is a chain of stages."""
    out, s = [], float(speed)
    while s > 2.0:
        out.append("atempo=2.0"); s /= 2.0
    while s < 0.5:
        out.append("atempo=0.5"); s /= 0.5
    if abs(s - 1.0) > 1e-3:
        out.append(f"atempo={s:.4f}")
    return out


@tool("studio_set", "studio", 2, "Change a layer: timing, text, place, size, colour, moves or sound level.",
      {"path": ("string", "the .sstudio file"), "layer": ("integer", "layer number from studio_read"),
       "start": ("number", "seconds"), "duration": ("number", "seconds"), "text": ("string", "new words (text layers)"),
       "x": ("number", "%"), "y": ("number", "%"), "width": ("number", "%"), "height": ("number", "%"),
       "size": ("number", "text size"), "color": ("string", "hex colour (text or shape)"), "opacity": ("number", "0-1"),
       "enter": ("string", "entrance move", MOVES), "exit": ("string", "exit move", MOVES),
       "volume": ("number", "0-1"), "trim": ("number", "clip: seconds skipped at its start"),
       "speed": ("number", "clip or sound: 0.5 = slow motion, 2 = twice as fast (0.25 to 4)"),
       "crop": ("string", "picture or clip: the part shown, 'x,y,w,h' in % of the source (0,0,100,100 = all)"),
       "fit": ("string", "picture: fill the box (cover) or show all of it (contain)", ["cover", "contain"]),
       "fade_in": ("number", "seconds to fade in (picture, clip, text or sound)"),
       "fade_out": ("number", "seconds to fade out")},
      ("path", "layer"))
def studio_set(ctx, path, layer, **kw):
    p, d = open_doc(ctx, path, "studio")
    i, l = _layer(d, layer)
    names = {"duration": "dur", "width": "w", "height": "h"}
    changed = []
    for k, v in kw.items():
        if v in (None, ""):
            continue
        k2 = names.get(k, k)
        if k in ("enter", "exit"):
            l["in" if k == "enter" else "out"] = {**(l.get("in" if k == "enter" else "out") or {"dur": 0.6}), "move": _move(v, "none")}
        elif k2 == "color":
            l["fill" if l["kind"] == "shape" else "color"] = _color(v, "FFFFFF")
        elif k2 == "text":
            l["text"] = F.clean(v)
        elif k2 == "speed":
            l["speed"] = min(SPEED[1], max(SPEED[0], float(v)))
        elif k2 == "crop":
            l["crop"] = v
            l["crop"] = list(crop_box(l) or (0, 0, 100, 100))
        elif k2 == "fit":
            l["fit"] = "cover" if str(v).lower() == "cover" else "contain"
        elif k2 in ("fade_in", "fade_out"):
            side = "in" if k2 == "fade_in" else "out"
            l[side] = {"move": "fade" if float(v) > 0 else "none", "dur": max(0.0, min(10.0, float(v)))}
        elif k2 in EDITABLE:
            l[k2] = EDITABLE[k2](v)
        changed.append(k)
    save_doc(ctx, p, d, f"edit layer {i + 1} of {path}")
    return f"layer {i + 1}: changed {', '.join(changed) or 'nothing'}"


@tool("studio_split", "studio", 2, "Cut a clip, sound or picture layer in two at a moment of the video; "
      "the second part keeps playing from where the first stopped.",
      {"path": ("string", "the .sstudio file"), "layer": ("integer", "layer number"),
       "at": ("number", "seconds into the video where the cut goes")}, ("path", "layer", "at"))
def studio_split(ctx, path, layer, at):
    p, d = open_doc(ctx, path, "studio")
    i, l = _layer(d, layer)
    start, dur, at = float(l.get("start", 0)), float(l.get("dur", 3)), float(at)
    if not start + 0.1 <= at <= start + dur - 0.1:
        raise ToolError(f"layer {i + 1} runs {start:g}-{start + dur:g} s: the cut must fall inside it")
    second = json.loads(json.dumps(l))
    l["dur"] = round(at - start, 3)
    second["start"], second["dur"] = round(at, 3), round(start + dur - at, 3)
    if l["kind"] in ("clip", "audio"):
        second["trim"] = round(float(l.get("trim", 0)) + (at - start) * _speed(l), 3)
    l["out"] = {"move": "none", "dur": 0.5}
    second["in"] = {"move": "none", "dur": 0.6}
    d["layers"].insert(i + 1, second)
    save_doc(ctx, p, d, f"split layer {i + 1} of {path} at {at:g} s")
    return f"layer {i + 1} split at {at:g} s: layers {i + 1} and {i + 2}"


# An ad from photos (2026-10-27): each photo fills the frame with a slow pan and zoom (Ken Burns),
# photos cross-fade, the headline comes in on the first, the tagline on the last, an end card closes,
# and music sits under an optional voice-over.
# a photo moves off centre only as far as its zoom covers: at scale s, |x - 50| <= (s - 1) * 50
AD_PAN = (({"x": 50, "y": 50, "scale": 1.0}, {"x": 47, "y": 49, "scale": 1.12}),
          ({"x": 53, "y": 51, "scale": 1.12}, {"x": 50, "y": 50, "scale": 1.0}),
          ({"x": 50, "y": 50, "scale": 1.04}, {"x": 51, "y": 52, "scale": 1.12}))


def ad_layers(images, headline, tagline, fmt="16:9", per=3.5, card=3.0, music="", voice="", fade=0.6):
    """The layers of a photo ad, and its length in seconds."""
    layers, t = [], 0.0
    for k, img in enumerate(images):
        a, b = AD_PAN[k % len(AD_PAN)]
        dur = per + (fade if k < len(images) - 1 else 0)
        layers.append({"kind": "image", "src": img, "start": round(t, 3), "dur": round(dur, 3), "x": a["x"], "y": a["y"],
                       "w": 100, "h": 100, "fit": "cover", "rotate": 0, "opacity": 1,
                       "keys": [{"t": 0, **a}, {"t": round(dur, 3), **b}],
                       "in": {"move": "fade" if k else "none", "dur": fade}, "out": {"move": "none", "dur": 0.5}})
        t += per
    body = t
    if headline:
        layers.append({"kind": "text", "text": headline, "start": 0.4, "dur": round(min(per * 1.6, body) - 0.4, 3),
                       "x": 50, "y": 78, "w": 86, "h": 18, "size": 96, "weight": 900, "color": "FFFFFF", "align": "center",
                       "in": {"move": "slide-up", "dur": 0.6}, "out": {"move": "fade", "dur": 0.4}})
    if tagline and len(images) > 1:
        layers.append({"kind": "text", "text": tagline, "start": round(body - per + 0.3, 3), "dur": round(per - 0.3, 3),
                       "x": 50, "y": 80, "w": 86, "h": 14, "size": 64, "weight": 700, "color": "FFFFFF", "align": "center",
                       "in": {"move": "fade", "dur": 0.5}, "out": {"move": "none", "dur": 0.4}})
    layers.append({"kind": "shape", "shape": "rect", "fill": "0B0B0B", "start": round(body, 3), "dur": card,
                   "x": 50, "y": 50, "w": 100, "h": 100, "in": {"move": "fade", "dur": 0.5}, "out": {"move": "none", "dur": 0.5}})
    layers.append({"kind": "text", "text": headline or tagline or "", "start": round(body + 0.2, 3), "dur": round(card - 0.2, 3),
                   "x": 50, "y": 46, "w": 86, "h": 20, "size": 104, "weight": 900, "color": "FFFFFF", "align": "center",
                   "in": {"move": "pop", "dur": 0.6}, "out": {"move": "none", "dur": 0.4}})
    if tagline:
        layers.append({"kind": "text", "text": tagline, "start": round(body + 0.6, 3), "dur": round(card - 0.6, 3),
                       "x": 50, "y": 62, "w": 80, "h": 12, "size": 56, "weight": 600, "color": "CFCFCF", "align": "center",
                       "in": {"move": "fade", "dur": 0.5}, "out": {"move": "none", "dur": 0.4}})
    total = round(body + card, 3)
    if music:
        layers.append({"kind": "audio", "src": music, "start": 0, "dur": total, "volume": 0.35 if voice else 0.8,
                       "duck": True, "in": {"move": "fade", "dur": 1.0}, "out": {"move": "fade", "dur": 1.5}})
    if voice:
        layers.append({"kind": "audio", "src": voice, "start": 0.3, "dur": round(total - 0.3, 3), "volume": 1.0, "duck": False})
    return layers, total


@tool("studio_ad", "studio", 2, "Make an ad from photos: each photo fills the frame with a slow pan and zoom, "
      "photos cross-fade, the headline and tagline come in over them, an end card closes, with optional "
      "music (a sound file) and an optional voice-over line.",
      {"path": ("string", "the new .sstudio file"), "images": ("array", "the photos, in order"),
       "headline": ("string", "the main line, a few words"), "tagline": ("string", "the second line, optional"),
       "music": ("string", "a sound file for music, optional"), "voice": ("string", "words to speak over it, optional"),
       "format": ("string", "16:9, 9:16 or 1:1", list(FORMATS)), "seconds_per_photo": ("number", "2 to 6, default 3.5")},
      ("path", "images"))
def studio_ad(ctx, path, images, headline="", tagline="", music="", voice="", format="16:9", seconds_per_photo=3.5):
    pics = [str(x) for x in (images or []) if str(x).strip()]
    if not pics:
        raise ToolError("choose or attach the photos for the ad first")
    if len(pics) > 12:
        raise ToolError("an ad takes up to 12 photos; choose fewer")
    for img in pics:
        if Path(img).suffix.lower() not in IMAGE_EXT:
            raise ToolError(f"{img} is not a picture")
        ctx.ws.path(img, must_exist=True)
    if music:
        if Path(music).suffix.lower() not in AUDIO_EXT:
            raise ToolError(f"{music} is not a sound file")
        ctx.ws.path(music, must_exist=True)
    per = max(2.0, min(6.0, float(seconds_per_photo or 3.5)))
    voice_file = ""
    if str(voice or "").strip():
        from .media import speak_text
        said = speak_text(ctx, str(voice).strip(), Path(path).with_suffix("").as_posix() + "-voice.wav")
        voice_file = said.rsplit(" as ", 1)[-1].strip()
    layers, total = ad_layers(pics, F.clean(headline or ""), F.clean(tagline or ""), format, per, music=music, voice=voice_file)
    d = new_project(format, total)
    d["layers"] = layers
    p, msg = start_new(ctx, path, "studio", False)
    if msg:
        raise ToolError(msg)
    save_doc(ctx, p, d, f"make the ad {path}")
    return (f"made the ad {path}: {len(pics)} photos, {total:g} s, {format}"
            + (", music" if music else "") + (", voice-over" if voice_file else "") + ". Render it to make the video.")


@tool("studio_delete", "studio", 2, "Remove a layer.",
      {"path": ("string", "the .sstudio file"), "layer": ("integer", "layer number")}, ("path", "layer"))
def studio_delete(ctx, path, layer):
    p, d = open_doc(ctx, path, "studio")
    i, l = _layer(d, layer)
    d["layers"].pop(i)
    save_doc(ctx, p, d, f"delete layer {i + 1} of {path}")
    return f"removed layer {i + 1} ({l['kind']})"


@tool("studio_put", "studio", 2, "Editor: replace the whole project with the given object.",
      {"path": ("string", "the .sstudio file"), "data": ("object", "the project")}, ("path", "data"))
def studio_put(ctx, path, data):
    p, d = open_doc(ctx, path, "studio")
    if not isinstance(data, dict) or data.get("sheep") != "studio" or not isinstance(data.get("layers"), list):
        raise ToolError("data must be a Studio project")
    for l in data["layers"]:
        if l.get("kind") not in KINDS:
            raise ToolError(f"a layer kind must be one of {', '.join(KINDS)}")
    save_doc(ctx, p, data, f"edit {path}")
    return "saved"


@tool("studio_voice", "studio", 2,
      "Speak words over the video: Sheep records them and lays the sound on the timeline.",
      {"path": ("string", "the .sstudio file"), "text": ("string", "what to say"),
       "start": ("number", "seconds from the start"), "voice": ("string", "female or male, optional"),
       "volume": ("number", "sound level 0-1"), "duck": ("boolean", "dip music under the voice")},
      ("path", "text"))
def studio_voice(ctx, path, text, start=0, voice="", volume=1.0, duck=True):
    """A voice-over in one step. The words become a .wav beside the project (the
    same speak_text the chat model uses), then an audio layer, so it mixes, ducks
    the music and renders like any other sound."""
    from .media import speak_text
    p, d = open_doc(ctx, path, "studio")
    stem = Path(ctx.ws.rel(p)).with_suffix("").name
    said = speak_text(ctx, text=text, path=f"media/{stem}-voice.wav", voice=voice)
    rel = said.split(" as ", 1)[-1].split(" ", 1)[0].strip() if " as " in said else ""
    wav = ctx.ws.path(rel, must_exist=True)
    own, _ = _probe(wav)
    layer = {"kind": "audio", "src": ctx.ws.rel(wav), "start": float(start or 0), "dur": own or 3.0,
             "x": 50.0, "y": 50.0, "w": 100.0, "h": 100.0, "trim": 0.0, "voice": True,
             "volume": float(volume if volume is not None else 1.0), "duck": bool(duck),
             "in": {"move": "fade", "dur": 0.2}, "out": {"move": "fade", "dur": 0.3}}
    d["layers"].append(layer)
    save_doc(ctx, p, d, f"voice-over in {path}")
    return f"said it in {own or 3.0:.1f} s as {ctx.ws.rel(wav)}, on the timeline at {float(start or 0):.1f} s"


@tool("studio_transcribe", "studio", 2,
      "Write down what is said in a clip on the timeline, and optionally put it on screen as captions.",
      {"path": ("string", "the .sstudio file"), "layer": ("number", "which layer; omit for the first clip or sound"),
       "captions": ("boolean", "also place the words on the video")},
      ("path",))
def studio_transcribe(ctx, path, layer=None, captions=False):
    """What a clip says, from Sheep's own speech-to-text. With captions=true the
    words are laid over the clip in blocks, so a video can be subtitled without
    anyone typing."""
    from .media import transcribe_audio
    p, d = open_doc(ctx, path, "studio")
    n = int(layer) if layer not in (None, "", 0) else 0
    if n:
        if not 1 <= n <= len(d["layers"]):
            raise ToolError(f"this project has {len(d['layers'])} layers")
        src = d["layers"][n - 1]
    else:
        src = next((l for l in d["layers"] if l.get("kind") in ("clip", "audio")), None)
        if not src:
            raise ToolError("no clip or sound on the timeline to write down")
    out = transcribe_audio(ctx, path=src["src"])
    words = out.split(":\n", 1)[-1].strip()
    if not captions or not words:
        return out
    # Captions at the moment each phrase is said: Whisper's word times, mapped through the
    # layer's start, trim and speed (they were spread evenly over the clip before, 2026-10-14)
    from .media import timed_words
    t0, trim, sp = float(src.get("start", 0)), float(src.get("trim", 0)), _speed(src)
    end = t0 + (float(src.get("dur", 0)) or 5.0)
    try:
        ws_ = timed_words(ctx, src["src"])
    except Exception:
        ws_ = []
    blocks, cur = [], []
    for w in ws_:
        at = t0 + (w["start"] - trim) / sp
        if at < t0 - 0.05 or at > end:
            continue
        cur.append(dict(w, at=at, until=t0 + (w["end"] - trim) / sp))
        if len(cur) >= 7 or w["word"].rstrip().endswith((".", "!", "?")) or cur[-1]["until"] - cur[0]["at"] > 3.0:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    if not blocks:                                     # no word times: one block per sentence, spread
        bits = [b.strip() for b in re.split(r"(?<=[.!?])\s+", words) if b.strip()] or [words]
        each = (end - t0) / len(bits)
        blocks = [[{"word": b, "at": t0 + i * each, "until": t0 + (i + 1) * each}] for i, b in enumerate(bits)]
    for b in blocks:
        text = " ".join(x["word"].strip() for x in b)[:120]
        st, du = b[0]["at"], max(0.6, b[-1]["until"] - b[0]["at"] + 0.15)
        d["layers"].append({"kind": "text", "text": text, "start": round(st, 2), "dur": round(du, 2),
                            "x": 50.0, "y": 86.0, "w": 86.0, "h": 14.0, "size": 4.2, "color": "#ffffff",
                            "font": "", "align": "center", "bold": False,
                            "in": {"move": "fade", "dur": 0.15}, "out": {"move": "fade", "dur": 0.15}})
    save_doc(ctx, p, d, f"captions in {path}")
    timed = "timed to the words" if ws_ else "spread over the clip (no word times)"
    return f"{out}\n{len(blocks)} caption blocks {timed}, over layer {n or 1}"


@tool("studio_listen", "studio", 1,
      "Listen to a clip or sound on the timeline: when speech happens and what is said (with times), the "
      "silences, the loud moments and the beat. Use it to place voice, text, cuts or music on the right moments.",
      {"path": ("string", "the .sstudio file"), "layer": ("number", "which layer; omit for the first clip or sound")},
      ("path",))
def studio_listen(ctx, path, layer=None):
    """What a model can hear of a sound, on the project's own clock: speech lines with their
    times (Whisper, when it is installed), silences longer than half a second, the loudest
    moments, and beats with a tempo. Times are where they fall on the timeline."""
    from .media import hear
    p, d = open_doc(ctx, path, "studio")
    n = int(layer) if layer not in (None, "", 0) else 0
    if n:
        if not 1 <= n <= len(d["layers"]):
            raise ToolError(f"this project has {len(d['layers'])} layers")
        src = d["layers"][n - 1]
    else:
        src = next((l for l in d["layers"] if l.get("kind") in ("clip", "audio")), None)
        if not src:
            raise ToolError("no clip or sound on the timeline to listen to")
    t0, trim, sp = float(src.get("start", 0)), float(src.get("trim", 0)), _speed(src)
    end = t0 + float(src.get("dur", 0) or 0)
    on = lambda s: round(t0 + (s - trim) / sp, 2)
    h = hear(ctx, src["src"])
    keep = lambda t: t0 - 0.05 <= t <= (end or 1e9)
    lines = [f"Layer {n or d['layers'].index(src) + 1} ({src.get('kind')}, {src.get('src')}), times on the timeline:"]
    speech = [(on(a), on(b), t) for a, b, t in h.get("speech", []) if keep(on(a))]
    if speech:
        lines.append("speech:")
        lines += [f"  {a:.2f}-{b:.2f} s: {t}" for a, b, t in speech[:40]]
    elif h.get("speech_note"):
        lines.append(h["speech_note"])
    sil = [(on(a), on(b)) for a, b in h.get("silences", []) if keep(on(a))]
    if sil:
        lines.append("silences: " + ", ".join(f"{a:.2f}-{b:.2f}" for a, b in sil[:20]))
    loud = [on(t) for t in h.get("loud", []) if keep(on(t))]
    if loud:
        lines.append("loudest moments: " + ", ".join(f"{t:.2f}" for t in loud[:10]))
    beats = [on(t) for t in h.get("beats", []) if keep(on(t))]
    if beats:
        lines.append(f"beat: about {h.get('bpm', 0) * sp:.0f} per minute; beats at "
                     + ", ".join(f"{t:.2f}" for t in beats[:32]))
    return "\n".join(lines)


@tool("studio_transition", "studio", 2,
      "A transition from one clip (or picture) into the next: the second overlaps the end of the first and "
      "comes in with a crossfade, slide, wipe or zoom.",
      {"path": ("string", "the .sstudio file"), "layer": ("integer", "the clip that comes in (layer number)"),
       "kind": ("string", "the transition", ("crossfade", "slide", "wipe", "zoom")),
       "duration": ("number", "seconds, 0.2 to 2")}, ("path", "layer"))
def studio_transition(ctx, path, layer, kind="crossfade", duration=0.6):
    """The layer comes in over the end of the clip or picture before it on the timeline,
    which stays put underneath: a cut becomes a transition without retiming the rest."""
    p, d = open_doc(ctx, path, "studio")
    i, nxt = _layer(d, layer)
    if nxt.get("kind") not in ("clip", "image"):
        raise ToolError("a transition goes into a clip or a picture")
    dur = min(2.0, max(0.2, float(duration or 0.6)))
    st = float(nxt.get("start", 0))
    prev = max((l for l in d["layers"] if l is not nxt and l.get("kind") in ("clip", "image")
                and float(l.get("start", 0)) < st), key=lambda l: float(l.get("start", 0)) + float(l.get("dur", 0)),
               default=None)
    if prev is None:
        raise ToolError("there is no clip or picture before this one to go from")
    pend = float(prev.get("start", 0)) + float(prev.get("dur", 0))
    new_start = max(float(prev.get("start", 0)), min(st, pend) - dur)
    nxt["start"] = round(new_start, 2)
    nxt["dur"] = round(float(nxt.get("dur", 0)) + (st - new_start), 2)        # it still ends where it did
    move = {"crossfade": "fade", "slide": "slide-left", "wipe": "wipe", "zoom": "zoom"}.get(str(kind), "fade")
    nxt["in"] = {"move": move, "dur": dur}
    # the incoming one is drawn on top, so the outgoing one stays whole underneath
    if d["layers"].index(nxt) < d["layers"].index(prev):
        d["layers"].remove(nxt)
        d["layers"].insert(d["layers"].index(prev) + 1, nxt)
    save_doc(ctx, p, d, f"{kind} into layer {i + 1} of {path}")
    return f"{kind} of {dur:.1f} s into layer {d['layers'].index(nxt) + 1}, starting at {nxt['start']:.2f} s"


@tool("studio_render", "studio", 2, "Render the project to an MP4 video (on the GPU when there is one).",
      {"path": ("string", "the .sstudio file"), "out": ("string", "the .mp4 to write; default next to the project"),
       "preview": ("boolean", "true for a quick half-size draft")}, ("path",))
def studio_render(ctx, path, out="", preview=False):
    p, d = open_doc(ctx, path, "studio")
    dst = ctx.ws.path(out or str(Path(ctx.ws.rel(p)).with_suffix(".mp4")))
    with ctx.ws.changing(dst, by=ctx.by, what=f"render {path}"):
        info = render(ctx.ws, d, dst, half=bool(preview), cancel_event=getattr(ctx, "cancel_event", None))
    ctx.made.add(ctx.ws.rel(dst))
    return (f"rendered {ctx.ws.rel(dst)}: {info['seconds']:.1f} s, {info['w']}x{info['h']}, {info['encoder']}, "
            f"{info['mb']:.1f} MB in {info['took']:.0f} s")


# ------------------------------------------------------------------ rendering

class _Clip:
    """Frames of one clip layer, read in order through ffmpeg at the project's fps and box size."""

    def __init__(self, src, trim, dur, w, h, fps, speed=1.0, crop=None):
        self.w, self.h, self.size = w, h, w * h * 3
        cut = (f"crop=iw*{crop[2] / 100:.4f}:ih*{crop[3] / 100:.4f}:iw*{crop[0] / 100:.4f}:ih*{crop[1] / 100:.4f},"
               if crop else "")
        # at speed s the layer's dur seconds show dur * s seconds of the source
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{trim:.3f}", "-t", f"{dur * speed:.3f}", "-i", str(src),
             "-vf", f"{cut}setpts=(PTS-STARTPTS)/{speed:.4f},scale={w}:{h}:force_original_aspect_ratio=increase,"
                    f"crop={w}:{h},fps={fps}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.n, self.last = -1, None

    def frame(self, k):
        from PIL import Image
        while self.n < k:
            b = self.proc.stdout.read(self.size)
            if len(b) < self.size:
                break
            self.n += 1
            self.last = b
        return Image.frombytes("RGB", (self.w, self.h), self.last) if self.last else None

    def close(self):
        try:
            self.proc.kill()
        except OSError:
            pass


def _font(size, weight):
    from PIL import ImageFont
    f = ImageFont.truetype(str(FONT), max(6, int(size)))
    try:
        f.set_variation_by_axes([max(200, min(1000, int(weight)))])
    except Exception:
        pass
    return f


def _wrap(draw, text, font, width):
    out = []
    for para in str(text).split("\n"):
        line = ""
        for word in para.split(" "):
            test = (line + " " + word).strip()
            if line and draw.textlength(test, font=font) > width:
                out.append(line)
                line = word
            else:
                line = test
        out.append(line)
    return out


def _rgba(hexs, alpha=1.0):
    h = _color(hexs, "FFFFFF")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(255 * max(0, min(1, alpha))))


def draw_layer(img, layer, s, scale, images, clips, t, fps):
    """Draw one layer's state onto the RGBA frame."""
    from PIL import Image, ImageDraw
    W, H = img.size
    bw, bh = max(1, int(W * s["w"] / 100)), max(1, int(H * s["h"] / 100))
    kind = layer["kind"]
    if kind == "text":
        text = str(layer.get("text", ""))
        if s["chars"] is not None:
            text = text[:max(0, math.ceil(len(text) * s["chars"]))]
        size = float(layer.get("size", 72)) * H / 1080
        font = _font(size, layer.get("weight", 800))
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        lines = _wrap(probe, text, font, bw)
        lh = int(size * 1.18)
        tw = max([int(probe.textlength(ln, font=font)) for ln in lines] + [1])
        piece = Image.new("RGBA", (max(bw, tw) + 8, lh * len(lines) + int(size * 0.35)), (0, 0, 0, 0))
        dd = ImageDraw.Draw(piece)
        align = layer.get("align", "center")
        for k, ln in enumerate(lines):
            lw = probe.textlength(ln, font=font)
            x = 4 if align == "left" else (piece.width - lw - 4 if align == "right" else (piece.width - lw) / 2)
            dd.text((x, k * lh), ln, font=font, fill=_rgba(layer.get("color", "FFFFFF")))
    elif kind == "shape":
        piece = Image.new("RGBA", (bw, max(bh, 2)), (0, 0, 0, 0))
        dd = ImageDraw.Draw(piece)
        col = _rgba(layer.get("fill", "A3A3A3"))
        if layer.get("shape") == "circle":
            dd.ellipse([0, 0, bw - 1, max(bh, 2) - 1], fill=col)
        elif layer.get("shape") == "line":
            dd.rectangle([0, max(bh, 2) // 2 - max(2, bh // 8), bw, max(bh, 2) // 2 + max(2, bh // 8)], fill=col)
        else:
            r = int(float(layer.get("radius", 12)) * H / 1080)
            dd.rounded_rectangle([0, 0, bw - 1, max(bh, 2) - 1], radius=min(r, bh // 2, bw // 2), fill=col)
    elif kind == "image":
        src = images.get(layer["src"])
        if src is None:
            return
        box = crop_box(layer)
        if box:
            x, y, w, h = box
            src = src.crop((int(src.width * x / 100), int(src.height * y / 100),
                            int(src.width * (x + w) / 100), int(src.height * (y + h) / 100)))
        if layer.get("fit") == "cover":
            from PIL import ImageOps
            piece = ImageOps.fit(src, (bw, bh), method=Image.LANCZOS)
        else:
            fit = src.copy()
            fit.thumbnail((bw, bh))
            piece = fit
    elif kind == "clip":
        c = clips.get(id(layer))
        fr = c.frame(int((t - float(layer.get("start", 0))) * fps)) if c else None
        if fr is None:
            return
        piece = fr.convert("RGBA")
    else:
        return
    if s["wipe"] < 1:
        a = piece.getchannel("A")
        cut = int(piece.width * s["wipe"])
        a.paste(0, (cut, 0, piece.width, piece.height))
        piece.putalpha(a)
    sc = s["scale"]
    if abs(sc - 1) > 1e-3:
        if sc <= 0.01:
            return
        piece = piece.resize((max(1, int(piece.width * sc)), max(1, int(piece.height * sc))))
    if s["rot"]:
        piece = piece.rotate(-s["rot"], expand=True, resample=Image.BICUBIC)
    if s["op"] < 0.999:
        a = piece.getchannel("A").point(lambda v: int(v * max(0.0, s["op"])))
        piece.putalpha(a)
    cx, cy = W * s["x"] / 100, H * s["y"] / 100
    img.alpha_composite(piece, (int(cx - piece.width / 2), int(cy - piece.height / 2)))


def frame_at(d, t, size, images, clips):
    from PIL import Image
    W, H = size
    img = Image.new("RGBA", (W, H), _rgba(d.get("bg", "0B0B0B")))
    for layer in d["layers"]:
        if layer["kind"] == "audio":
            continue
        s = state(layer, t)
        if s:
            draw_layer(img, layer, s, W / d["w"], images, clips, t, d.get("fps", 30))
    return img.convert("RGB")


def _encoder():
    """NVENC when this machine's ffmpeg and GPU can use it, else libx264."""
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x144:d=0.1", "-c:v", "h264_nvenc",
                            "-f", "null", "-"], capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0:
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"], "NVENC (GPU)"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20"], "libx264 (CPU)"


def render(ws, d, dst, half=False, cancel_event=None):
    """The project to an MP4. Returns what was made."""
    from PIL import Image
    if not shutil.which("ffmpeg"):
        raise ToolError("rendering needs ffmpeg on this machine")
    t0 = time.time()
    W, H = d["w"], d["h"]
    if half:
        W, H = W // 2 // 2 * 2, H // 2 // 2 * 2
    fps = int(d.get("fps", 30))
    total = length(d)
    n = max(1, int(round(total * fps)))
    images, clips, sounds = {}, {}, []
    for layer in d["layers"]:
        src = layer.get("src")
        if layer["kind"] == "image" and src not in images:
            try:
                with Image.open(ws.path(src, must_exist=True)) as source:
                    images[src] = source.convert("RGBA")
            except (OSError, ToolError) as e:
                raise ToolError(f"cannot read the picture {src}: {e}")
        elif layer["kind"] == "clip":
            bw, bh = max(2, int(W * float(layer.get("w", 100)) / 100) // 2 * 2), max(2, int(H * float(layer.get("h", 100)) / 100) // 2 * 2)
            clips[id(layer)] = _Clip(ws.path(src, must_exist=True), float(layer.get("trim", 0)), float(layer.get("dur", 5)),
                                     bw, bh, fps, _speed(layer), crop_box(layer))
            if _probe(ws.path(src))[1]:
                sounds.append(layer)
        elif layer["kind"] == "audio":
            ws.path(src, must_exist=True)
            sounds.append(layer)
    venc, label = _encoder()
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    for sd in sounds:
        cmd += ["-ss", f"{float(sd.get('trim', 0)):.3f}", "-t", f"{float(sd.get('dur', 5)) * _speed(sd):.3f}",
                "-i", str(ws.path(sd["src"]))]
    if sounds:
        clip_spans = [(float(l["start"]), float(l["start"]) + float(l["dur"])) for l in sounds if l["kind"] == "clip"]
        parts, labels = [], []
        for k, sd in enumerate(sounds, 1):
            st, du = float(sd.get("start", 0)), float(sd.get("dur", 5))
            vol = float(sd.get("volume", 1.0))
            fin, fout = _fades(sd)
            f = _atempo(_speed(sd)) + [f"volume={vol:.3f}", f"afade=t=in:st=0:d={fin:.3f}",
                                       f"afade=t=out:st={max(0.0, du - fout):.3f}:d={fout:.3f}"]
            if sd["kind"] == "audio" and sd.get("duck", True):
                for a, b in clip_spans:          # music dips under clip sound
                    f.append(f"volume=enable='between(t,{max(0.0, a - st):.3f},{max(0.0, b - st):.3f})':volume=0.3")
            f.append(f"adelay={int(st * 1000)}|{int(st * 1000)}")
            parts.append(f"[{k}:a]{','.join(f)}[a{k}]")
            labels.append(f"[a{k}]")
        parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest[aout]")
        cmd += ["-filter_complex", ";".join(parts), "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd += venc + ["-pix_fmt", "yuv420p", "-t", f"{total:.3f}", "-movflags", "+faststart", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    err = b""
    try:
        for k in range(n):
            if cancel_event is not None and cancel_event.is_set():
                raise ToolError("Studio render was cancelled by the user")
            enc.stdin.write(frame_at(d, k / fps, (W, H), images, clips).tobytes())
        enc.stdin.close()
        err = enc.stderr.read()
        enc.wait(timeout=600)
    except (BrokenPipeError, OSError):
        err = enc.stderr.read()
        enc.wait(timeout=60)
    finally:
        if enc.poll() is None:
            enc.kill()
            enc.wait(timeout=5)
        for stream in (enc.stdin, enc.stderr):
            if stream and not stream.closed:
                stream.close()
        for c in clips.values():
            c.close()
    if enc.returncode != 0 or not dst.exists():
        raise ToolError(f"the encoder failed: {err.decode('utf-8', 'replace').strip()[-300:]}")
    return {"seconds": total, "w": W, "h": H, "encoder": label, "mb": dst.stat().st_size / 1e6, "took": time.time() - t0,
            "frames": n}
