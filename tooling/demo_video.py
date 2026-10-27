"""The demo video, made by Sheep's own Studio from real screenshots of the app: one scene per screen, a
voice line read by a Windows voice, the same words as a caption, no music, under 3 minutes.

    python tooling/demo_video.py [OUT.mp4]
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.tools import studio as S  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

SHOTS = Path(sys.argv[2]) if len(sys.argv) > 2 else None
GALLERY = ROOT / "mlmi" / "media" / "gallery"
# (picture, line) in order; every picture is a real screenshot of the app or a measured result
SCENES = [
    ("welcome.png", "Most people work on an ordinary laptop, on a connection that is slow or costly. "
                    "Sheep is a personal AI that runs on that laptop and makes real files."),
    (GALLERY / "1-chat.png", "You ask in plain words. NVIDIA Nemotron on Nebius plans the job as exact steps, "
                             "and Sheep carries them out on this computer, in its own apps."),
    (GALLERY / "5-slides.png", "The deck opens in Sheep Slides, and one more word saves it as a PowerPoint file."),
    (GALLERY / "3-words.png", "Documents and workbooks work the same way, and every change can be undone."),
    (GALLERY / "2-what-sheep-did.png", "Bring any model by its name. Sheep fits it to this computer, checks its "
                                       "answers, and shows you every step it took."),
    ("models.png", "One place chooses the model for each job: chat, checking, pictures, video and speech."),
    ("missing.png", "When something is missing, Sheep says what, with a button to fix it, then try again."),
    ("basics.png", "Settings are plain choices. Nothing leaves the computer unless you switch Nemotron on, "
                   "and every request sent out is listed."),
    (ROOT / "mlmi" / "media" / "benchmark.png", "On this four gigabyte laptop, Sheep's own engine reads prompts up to "
                                                "one point four times as fast as llama dot c p p, and ties it when writing."),
    (ROOT / "docs" / "diagrams" / "sheep-1-how-sheep-is-built.png",
     "It is Windows only for now, and local video is short and simple. Sheep: real work, on your own computer."),
]


def main(out):
    ws_dir = Path(tempfile.mkdtemp(prefix="sheep-demo-"))
    ws = Workspace(str(ws_dir))
    ctx = T.Ctx(ws, "user")
    layers, t = [], 0.0
    title = 3.0
    layers.append({"kind": "text", "text": "Sheep", "start": 0.0, "dur": title, "x": 50, "y": 44, "w": 80, "h": 20,
                   "size": 150, "weight": 900, "color": "FFFFFF", "align": "center",
                   "in": {"move": "pop", "dur": 0.6}, "out": {"move": "fade", "dur": 0.5}})
    layers.append({"kind": "text", "text": "Personal AI on your own laptop, with NVIDIA Nemotron on Nebius",
                   "start": 0.5, "dur": title - 0.5, "x": 50, "y": 62, "w": 86, "h": 10, "size": 48, "weight": 600,
                   "color": "CFCFCF", "align": "center", "in": {"move": "fade", "dur": 0.5}, "out": {"move": "fade", "dur": 0.5}})
    t = title
    for k, (pic, line) in enumerate(SCENES):
        src = Path(pic) if Path(pic).is_absolute() else (SHOTS / pic if SHOTS else None)
        if not src or not src.exists():
            print("skipped, no picture:", pic)
            continue
        name = f"scene{k:02d}{src.suffix}"
        shutil.copyfile(src, ws_dir / name)
        voice = f"voice{k:02d}.wav"
        T.run(ws, "speak_text", {"text": line, "path": voice}, by="user", ctx=ctx)
        said, _ = S._probe(ws_dir / voice)
        dur = round(max(4.0, (said or 4.0) + 0.9), 2)
        layers.append({"kind": "image", "src": name, "start": round(t, 2), "dur": dur, "x": 50, "y": 45, "w": 100, "h": 90,
                       "fit": "contain", "keys": [{"t": 0, "scale": 1.0}, {"t": dur, "scale": 1.04}],
                       "in": {"move": "fade", "dur": 0.4}, "out": {"move": "fade", "dur": 0.3}})
        # the voice reads words; the caption shows them as written
        caption = (line.replace("llama dot c p p", "llama.cpp").replace("one point four times", "1.4×")
                   .replace("four gigabyte", "4 GB"))
        layers.append({"kind": "text", "text": caption, "start": round(t + 0.2, 2), "dur": round(dur - 0.3, 2),
                       "x": 50, "y": 94, "w": 92, "h": 9, "size": 30, "weight": 700, "color": "FFFFFF", "align": "center",
                       "in": {"move": "fade", "dur": 0.3}, "out": {"move": "fade", "dur": 0.2}})
        layers.append({"kind": "audio", "src": voice, "start": round(t + 0.35, 2), "dur": round(said or dur, 2),
                       "volume": 1.0, "duck": False})
        t += dur
    layers.append({"kind": "text", "text": "Sheep", "start": round(t, 2), "dur": 3.0, "x": 50, "y": 46, "w": 80, "h": 20,
                   "size": 130, "weight": 900, "color": "FFFFFF", "align": "center", "in": {"move": "fade", "dur": 0.5},
                   "out": {"move": "none", "dur": 0.4}})
    total = round(t + 3.0, 2)
    if total > 178:
        raise SystemExit(f"the video would be {total:.0f} s: over the 3-minute limit")
    d = S.new_project("16:9", total, "0B0B0B")
    d["layers"] = layers
    (ws_dir / "demo.sstudio").write_text(__import__("json").dumps(d), encoding="utf-8")
    T.run(ws, "studio_render", {"path": "demo.sstudio", "out": "demo.mp4"}, by="user", ctx=ctx)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ws_dir / "demo.mp4", out)
    dur, audio = S._probe(Path(out))
    print(f"{out}: {dur:.1f} s, sound: {audio}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "submission" / "sheep-demo.mp4"))
