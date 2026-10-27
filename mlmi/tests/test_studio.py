"""Sheep Studio, end to end: a project made by tool calls renders to a real MP4.

    python mlmi/tests/test_studio.py

A test clip (with sound) and a picture are generated, a project is started
from a template, text, a shape, the clip, the picture and music are added,
one layer is edited, and the project is rendered; ffprobe then checks the
file's length, size and that it carries sound. The motion maths is checked
at fixed points (the editor's preview uses the same numbers).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.tools import studio as ST  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

ok = 0


def check(what, cond, detail=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + what + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(1)
    ok += 1


def probe(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
                          "-of", "json", str(p)], capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    ws = Workspace(Path(tempfile.mkdtemp(prefix="mlmi-studio-")))
    media = ws.root / "media"
    media.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-shortest", "-c:v", "libx264", "-c:a", "aac",
                    str(media / "clip.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=8", str(media / "music.mp3")],
                   check=True)
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([20, 20, 380, 380], fill=(47, 107, 255, 255))
    im.save(media / "logo.png")

    run = lambda name, **a: T.run(ws, name, a, by="agent")       # noqa: E731
    r = run("studio_create", path="intro.sstudio", duration=6, template="title-intro", title="Sheep", subtitle="Local AI")
    check("a project from a template", "3 layers from title-intro" in r, r)
    run("studio_add_media", path="intro.sstudio", file="media/clip.mp4", start=5, duration=3, enter="fade")
    run("studio_add_media", path="intro.sstudio", file="media/logo.png", start=5.5, duration=2.5, x=82, y=18, width=12, height=20, enter="pop")
    run("studio_add_media", path="intro.sstudio", file="media/music.mp3", start=0, duration=8, volume=0.6)
    run("studio_add_shape", path="intro.sstudio", shape="circle", start=1, duration=2, x=15, y=20, width=6, height=10)
    run("studio_add_text", path="intro.sstudio", text="Made on this laptop", start=5.2, duration=2.8, y=85, size=56, enter="typewriter")
    out = run("studio_read", path="intro.sstudio")
    check("the outline lists every layer", out.count("\n") == 8 and "clip" in out and "audio" in out, out.splitlines()[0])
    first = out.splitlines()[1]
    check("a full-frame clip sits under everything", first.startswith("1. clip"), first)
    run("studio_set", path="intro.sstudio", layer=2, color="FFD166", start=0.1)
    d = T.studio.model("intro.sstudio", ws)
    check("an edit changes the layer", d["layers"][1]["fill"] == "FFD166" and abs(d["layers"][1]["start"] - 0.1) < 1e-6)
    check("the project is as long as its layers", abs(d["length"] - 8.0) < 1e-6, d["length"])

    # the motion maths the preview shares
    txt = {"start": 0, "dur": 3, "in": {"move": "pop", "dur": 0.6}, "out": {"move": "fade", "dur": 0.5}}
    check("before and after a layer it is hidden", ST.state(txt, -0.1) is None and ST.state(txt, 3.0) is None)
    check("a pop starts small and lands at full size", ST.state(txt, 0.0)["scale"] < 0.05 and abs(ST.state(txt, 1.0)["scale"] - 1) < 1e-9)
    check("a fade-out ends near clear", ST.state(txt, 2.99)["op"] < 0.1)
    tw = ST.state({"start": 0, "dur": 2, "in": {"move": "typewriter", "dur": 1}}, 0.5)
    check("typewriter shows half the letters halfway", abs(tw["chars"] - 0.5) < 1e-9)

    r = run("studio_render", path="intro.sstudio")
    check("it renders", r.startswith("rendered intro.mp4"), r)
    p = probe(ws.path("intro.mp4"))
    v = [s for s in p["streams"] if s["codec_type"] == "video"][0]
    check("the MP4 is 1920x1080, 8 s, with sound", v["width"] == 1920 and v["height"] == 1080
          and abs(float(p["format"]["duration"]) - 8.0) < 0.3 and any(s["codec_type"] == "audio" for s in p["streams"]),
          f"{v['width']}x{v['height']} {float(p['format']['duration']):.2f}s")
    r = run("studio_create", path="short.sstudio", format="9:16", duration=8, template="short", title="Did you know?")
    r = run("studio_render", path="short.sstudio", preview=True)
    p = probe(ws.path("short.mp4"))
    v = [s for s in p["streams"] if s["codec_type"] == "video"][0]
    check("a 9:16 short renders as a half-size draft", v["width"] == 540 and v["height"] == 960, r)

    # A rendered MP4 in the workspace root, not just media/, can be added to a
    # later timeline just like an AI-generated clip.
    (ws.root / "rendered.mp4").write_bytes((media / "clip.mp4").read_bytes())
    run("studio_create", path="reuse.sstudio")
    run("studio_add_media", path="reuse.sstudio", file="rendered.mp4")
    reused = ST.model("reuse.sstudio", ws)
    check("workspace MP4 can be selected as a timeline clip",
          reused["layers"][0]["src"] == "rendered.mp4" and reused["layers"][0]["kind"] == "clip")
    print(f"all {ok} checks passed  (workspace: {ws.root})")


if __name__ == "__main__":
    main()
