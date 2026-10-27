"""Plain words on every screen (polish plan, step 1). Collects the text a person sees from the desktop UI
(text between tags, title / aria-label / placeholder, toasts, textContent) and the tile words the
orchestrator sends, and fails on engineer's words. Technical detail belongs inside an element with the class
"tech" (the Details parts); lines that open one are skipped."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "mlmi" / "desktop" / "ui"
sys.path.insert(0, str(ROOT))

JARGON = [r"tok/s", r"\bcanar(y|ies)\b", r"\bslots?\b", r"quantiz", r"\.ssf\b", r"\.sdeck\b", r"\.sdoc\b",
          r"\.ssheet\b", r"\bctx\b", r"\bthreads?\b", r"\blayers?\b", r"\bCUDA\b", r"\bDirectML\b", r"ML\.M\.I",
          r"\bherders?\b", r"\bEgypt\b", r"(?-i:This machine)"]


def visible(text):
    out = []
    for line in text.splitlines():
        if 'class="tech' in line or "class='tech" in line:
            continue
        out += [m.group(1) for m in re.finditer(r">([^<>{}$`=;]{2,})<", line)]
        out += [m.group(2) for m in re.finditer(r'(title|aria-label|placeholder)="([^"${}]{2,})"', line)]
        out += [m.group(1) for m in re.finditer(r"toast\(\s*[`\"']([^`\"'$]{2,})", line)]
        out += [m.group(1) for m in re.finditer(r"textContent\s*=\s*[`\"']([^`\"'$]{2,})", line)]
    return [s.strip() for s in out if re.search(r"[A-Za-z]{2}", s)]


def tile_words():
    src = (ROOT / "mlmi" / "backend" / "orchestrator.py").read_text(encoding="utf-8")
    start = src.find("def kpis(")
    body = src[start:start + 12000] if start >= 0 else ""
    return [m.group(2) for m in re.finditer(r'"(label|word)":\s*"([^"]+)"', body)]


def found():
    hits = []
    for f in ("index.html", "app.js", "views.js", "welcome.js"):
        for s in visible((UI / f).read_text(encoding="utf-8")):
            for pat in JARGON:
                if re.search(pat, s, re.I):
                    hits.append(f"{f}: {pat} in {s[:70]!r}")
    for s in tile_words():
        for pat in JARGON:
            if re.search(pat, s, re.I):
                hits.append(f"orchestrator tiles: {pat} in {s!r}")
    return hits


class PlainWords(unittest.TestCase):
    def test_no_engineer_words_on_screen(self):
        hits = found()
        self.assertFalse(hits, f"{len(hits)} jargon words on screen, first: {hits[:8]}")

    def test_the_collector_sees_screen_text(self):
        self.assertTrue(any("Message Sheep" in s for s in visible((UI / "index.html").read_text(encoding="utf-8"))))


if __name__ == "__main__":
    for h in found():
        print(h)
