"""Reading a picture: Sheep's own engine, with Sheep's own vision model.

A chat model that cannot see is still asked about pictures, so `see_picture`
describes one and hands the words back. The tower and the text side both run in
sml-cli (`--image`), on our own weights, with our own preprocessing: no outside
runtime, nothing sent anywhere (Ernest, 2026-10-07: Sheep is its own ecosystem).

The vision model is any Sheep file whose header carries a vision tower, picked
in Settings; a chat model that has its own tower is used directly instead.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import tool
from ..paths import HOME, REPO
from ..workspace import ToolError

CLI = REPO / "sml" / ("sml-cli.exe" if os.name == "nt" else "sml-cli")
TOOLING = REPO / "tooling"
MODELS = HOME / "models"
PICTURE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def has_tower(ssf):
    """Does this Sheep file carry a vision tower? Read from its own header."""
    try:
        sys.path.insert(0, str(TOOLING))
        import sml_read
        f = sml_read.SmlFile(str(ssf))
        head = f.head
        del f
        return bool(head.get("vision"))
    except Exception:
        return False


def model(state=None):
    """The vision model to use: the loaded chat model if it can see, else the one
    chosen in Settings, else any Sheep file here with a tower. "" when there is none."""
    cur = getattr(state, "model_path", "") if state else ""
    if cur and has_tower(cur):
        return cur
    chosen = (getattr(state, "cfg", {}) or {}).get("vision_model") if state else ""
    if chosen and Path(chosen).exists() and has_tower(chosen):
        return chosen
    for p in sorted(MODELS.glob("*.ssf")) if MODELS.exists() else []:
        if has_tower(p):
            return str(p)
    return ""


def describe(ssf, picture, question, gen=160, threads=0):
    """Run the vision model over one picture; returns (words, seconds)."""
    sys.path.insert(0, str(TOOLING))
    import sml_image
    fd, px = tempfile.mkstemp(suffix=".f32")
    os.close(fd)
    try:
        sml_image.prepare(str(picture)).tofile(px)
        t0 = time.perf_counter()
        r = subprocess.run([str(CLI), str(ssf), "--image", px, "--ask", question or "Describe this picture.",
                            "--gen", str(max(1, min(int(gen), 300))),
                            "--threads", str(threads or os.cpu_count() or 6)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        dt = time.perf_counter() - t0
    finally:
        os.remove(px)
    out = r.stdout or ""
    i, j = out.find('\n  "'), out.find('"\n\ngenerate')
    if r.returncode != 0 or i < 0 or j < 0:
        raise ToolError((r.stderr or out)[-200:].strip() or "the vision model gave no answer")
    return out[i + 4:j].strip(), dt


@tool("see_picture", "media", 1,
      "Look at a picture in the workspace and say what is in it. Use this before answering any "
      "question about a picture; you cannot see it otherwise.",
      {"path": ("string", "the picture file"), "question": ("string", "what to ask about it, optional")},
      ("path",))
def see_picture(ctx, path, question=""):
    src = ctx.ws.path(path, must_exist=True)
    if src.suffix.lower() not in PICTURE:
        raise ToolError(f"{path} is not a picture")
    ssf = model(getattr(ctx, "state", None))
    if not ssf:
        raise ToolError("no model here can see pictures yet. Add one in Settings > Models: a vision model "
                        "such as SmolVLM, quantized by Sheep like any other model.")
    words, dt = describe(ssf, src, question)
    return f"{path} ({Path(ssf).stem}, {dt:.0f} s): {words}"
