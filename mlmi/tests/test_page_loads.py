"""The page's scripts really run: a line in app.js that used MLMI before it existed left the whole app blank, no
icons and no chats, and no other test loads the page (2026-10-30).

    python -m pytest mlmi/tests/test_page_loads.py -q
"""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "desktop" / "ui"
RUN = r"""
const fs = require("fs"), vm = require("vm"), path = require("path");
const stub = () => new Proxy(function () {}, { get: (t, k) => k === Symbol.toPrimitive ? () => "" : k === "length" ? 0 :
  k === Symbol.iterator ? function* () {} : stub(), apply: () => stub(), construct: () => stub(), set: () => true });
const g = { console, setTimeout: () => 0, setInterval: () => 0, clearTimeout() {}, clearInterval() {}, JSON, Math, Date, Object, Array,
  String, Number, Boolean, RegExp, Map, Set, Promise, Symbol, Error, parseInt, parseFloat, isNaN, encodeURIComponent, decodeURIComponent };
for (const k of ["document", "localStorage", "sessionStorage", "navigator", "location", "history", "fetch", "EventSource", "matchMedia",
  "ResizeObserver", "MutationObserver", "IntersectionObserver", "requestAnimationFrame", "getComputedStyle", "speechSynthesis",
  "Audio", "Image", "AudioContext", "FileReader", "Blob", "URL", "CSS", "performance", "addEventListener", "SpeechSynthesisUtterance"]) g[k] = stub();
g.window = g; g.globalThis = g; g.self = g;
const ctx = vm.createContext(g);
for (const f of process.argv.slice(2)) vm.runInContext(fs.readFileSync(f, "utf8"), ctx, { filename: path.basename(f) });
vm.runInContext("if (typeof MLMI.start !== 'function' || typeof MLMI.reveal !== 'function') throw new Error('MLMI is not whole')", ctx);
console.log("loaded");
"""


class PageLoads(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_every_script_runs_in_order(self):
        names = re.findall(r'<script src="([\w.-]+\.js)"></script>', (UI / "index.html").read_text(encoding="utf-8"))
        self.assertGreater(len(names), 5)
        r = subprocess.run(["node", "-e", RUN, *[str(UI / n) for n in names]], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("loaded", r.stdout)


if __name__ == "__main__":
    unittest.main()
