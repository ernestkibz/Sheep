"""Studio speaks and listens: a voice-over on the timeline, and a clip written
down as captions (PARITY: Studio audio).

    python -m mlmi.tests.test_studio_audio

Needs a Windows voice and Sheep's speech-to-text; skips the half it cannot run.
"""
import sys
import tempfile
import unittest
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("CLI integration test; run python -m mlmi.tests.test_studio_audio")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402

SAID = "Sheep runs on this machine. It makes the video for you."
fails, ran = [], [0]


def check(name, ok, got=""):
    ran[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      got {got}"))
    if not ok:
        fails.append(name)


root = Path(tempfile.mkdtemp(prefix="mlmi-studio-audio-"))
ws = Workspace(root)
run = lambda n, **a: T.run(ws, n, a, by="test")

run("studio_create", path="talk.sstudio", title="Sheep", seconds=8)
try:
    out = run("studio_voice", path="talk.sstudio", text=SAID, start=0.5)
except Exception as e:
    print(f"no voice on this machine ({str(e)[:90]}): skipped")
    sys.exit(0)
print("  " + out)
d = T.run(ws, "studio_read", {"path": "talk.sstudio"}, by="test")
check("the voice is on the timeline", "audio" in d.lower() and "voice" in out.lower(), d[:160])
wav = next((p for p in root.rglob("*.wav")), None)
check("it recorded a sound file", bool(wav) and wav.stat().st_size > 2000, wav)

try:
    out = run("studio_transcribe", path="talk.sstudio", captions=True)
except Exception as e:
    print(f"no speech-to-text on this machine ({str(e)[:90]}): the rest skipped")
    print(f"\n{'all ' + str(ran[0]) + ' checks passed' if not fails else str(len(fails)) + ' failed'}")
    sys.exit(1 if fails else 0)
print("  " + out.replace("\n", " | ")[:220])
check("it wrote down what was said", "sheep" in out.lower() or "machine" in out.lower(), out[:200])
d = T.run(ws, "studio_read", {"path": "talk.sstudio"}, by="test")
check("the words are on screen as captions", "caption blocks" in out and "text" in d.lower(), d[:200])

print(f"\n{'all ' + str(ran[0]) + ' checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
