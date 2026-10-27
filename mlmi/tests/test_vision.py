"""Sheep reads a picture, on this machine, with its own engine and weights.

    python -m mlmi.tests.test_vision [MODEL.ssf]

Draws pictures whose answer is not in doubt, asks about each through the vision
path the see_picture tool uses, and says whether the words match. Needs a Sheep
file carrying a vision tower (SmolVLM, quantized by Sheep); skips when none is
plugged in.
"""
import sys
import tempfile
import unittest
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("CLI integration test; run python -m mlmi.tests.test_vision")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image, ImageDraw  # noqa: E402

from mlmi.backend.tools import vision as V  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="mlmi-vision-"))

CASES = [
    ("circle.png", "What colour is the shape?", ("red",), lambda d: d.ellipse((120, 120, 392, 392), fill=(220, 30, 30))),
    # One fact per question: asked for shape AND colour at once, the model answers
    # the shape only ("A square.") -- right, but not what a two-part check wants.
    ("square.png", "What shape is in the picture?", ("squ",),
     lambda d: d.rectangle((140, 140, 372, 372), fill=(30, 60, 220))),
    ("square2.png", "What colour is the shape?", ("blue",),
     lambda d: d.rectangle((140, 140, 372, 372), fill=(30, 60, 220))),
]

ssf = sys.argv[1] if len(sys.argv) > 1 else V.model(None)
if not ssf:
    print("no model here carries a vision tower: skipped"); sys.exit(0)
print("vision model:", Path(ssf).name, "| tower:", V.has_tower(ssf), flush=True)

bad = 0
for name, ask, want, draw in CASES:
    p = WORK / name
    img = Image.new("RGB", (512, 512), (250, 250, 250))
    draw(ImageDraw.Draw(img))
    img.save(p)
    try:
        words, dt = V.describe(ssf, p, ask)
    except Exception as e:
        print(f"{name}: FAILED to run: {str(e)[:200]}"); bad += 1; continue
    low = words.lower()
    ok = all(w in low for w in want)
    print(f"{name} ({dt:.0f} s): {words[:160]!r}\n   {'RIGHT' if ok else 'wrong, wanted ' + str(want)}")
    bad += 0 if ok else 1

print("\nSheep reads pictures on this machine" if not bad else f"\n{bad} of {len(CASES)} wrong")
sys.exit(1 if bad else 0)
