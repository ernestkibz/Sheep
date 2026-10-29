"""The Sheep manual as one PDF: the user guide, the three diagrams and the benchmark chart.

    python tooling/manual_pdf.py            (needs the `markdown` package; prints with Sheep's Electron)

Writes docs/Sheep-Manual.html and docs/Sheep-Manual.pdf.
"""
import base64
import os
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ELECTRON = ROOT / "mlmi" / "desktop" / "node_modules" / "electron" / "dist" / ("electron.exe" if os.name == "nt" else "electron")
PICTURES = [("How Sheep is built", DOCS / "diagrams" / "sheep-1-how-sheep-is-built.png"),
            ("One request, end to end", DOCS / "diagrams" / "sheep-2-one-request-end-to-end.png"),
            ("A model, from a name to Ready", DOCS / "diagrams" / "sheep-3-a-model-from-a-name-to-ready.png"),
            ("Sheep against llama.cpp", ROOT / "mlmi" / "media" / "bench" / "sheep_vs_llamacpp.png"),
            ("Sheep against Ollama", ROOT / "mlmi" / "media" / "bench" / "sheep_vs_ollama.png"),
            ("Engine against engine: SML and GGML", ROOT / "mlmi" / "media" / "bench" / "sml_vs_ggml.png"),
            ("File against file: SSF and GGUF", ROOT / "mlmi" / "media" / "bench" / "ssf_vs_gguf.png"),
            ("Hardware used for the same work", ROOT / "mlmi" / "media" / "bench" / "hardware.png"),
            ("What else each one does", ROOT / "mlmi" / "media" / "bench" / "features.png"),
            ("A model too big for the card", ROOT / "mlmi" / "media" / "bench" / "big_model.png")]
# the app itself, photographed by tooling/manual_shots.ps1 (Sheep's own MLMI_SHOTS mode)
SHOTS = ROOT / "docs" / "shots"
SCREENS = [("Chat, with the start-up card", "home"), ("The tour", "tour"), ("Asking the guide", "tour-ask"), ("Quick guide", "guide"), ("Settings › Basics", "basics"),
           ("Settings › Models: one row per job", "models"), ("Settings › Accounts", "accounts"),
           ("Settings › Activity", "activity"), ("Sheep Studio", "studio")]
CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif; color: #1f2328; max-width: 860px; margin: 32px auto; line-height: 1.55; font-size: 11pt; }
h1 { font-size: 26pt; margin-bottom: 4px; } h2 { font-size: 16pt; border-bottom: 1px solid #d8dee4; padding-bottom: 4px; margin-top: 28px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; margin: 10px 0; } th, td { border: 1px solid #d8dee4; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; } code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; font-size: 9.5pt; }
figure { margin: 18px 0; page-break-inside: avoid; } figure img { max-width: 100%; border: 1px solid #d8dee4; border-radius: 6px; }
figcaption { color: #6e7781; font-size: 9.5pt; margin-top: 4px; } .cover { color: #6e7781; margin-bottom: 24px; }
"""

PRINT_JS = r"""
const { app, BrowserWindow } = require('electron'); const fs = require('fs');
const [src, out] = process.argv.slice(-2);
app.whenReady().then(async () => {
  const w = new BrowserWindow({ show: false });
  await w.loadFile(src);
  const pdf = await w.webContents.printToPDF({ printBackground: true, pageSize: 'A4',
    margins: { top: 0.5, bottom: 0.5, left: 0.5, right: 0.5 } });
  fs.writeFileSync(out, pdf); app.quit();
});
"""


def img(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    guide = (DOCS / "USER_GUIDE.md").read_text(encoding="utf-8")
    body = markdown.markdown(guide, extensions=["tables"])
    figs = "".join(f'<figure><img src="{img(p)}" alt="{t}"><figcaption>{t}</figcaption></figure>'
                   for t, p in PICTURES if p.exists())
    screens = "".join(f'<figure><img src="{img(SHOTS / (n + ".png"))}" alt="{t}"><figcaption>{t}</figcaption></figure>'
                      for t, n in SCREENS if (SHOTS / (n + ".png")).exists())
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>Sheep manual</title><style>{CSS}</style></head><body>"
            f"<p class='cover'>Sheep · local AI on your own computer · manual</p>{body}"
            + (f"<h2>The screens</h2>{screens}" if screens else "")
            + f"<h2>How it fits together, and how it measures up</h2>{figs}</body></html>")
    out_html, out_pdf = DOCS / "Sheep-Manual.html", DOCS / "Sheep-Manual.pdf"
    out_html.write_text(html, encoding="utf-8")
    script = DOCS / ".print_manual.js"
    script.write_text(PRINT_JS, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("ELECTRON_RUN_AS_NODE", "NODE_ENV")}
    try:
        subprocess.run([str(ELECTRON), str(script), str(out_html), str(out_pdf)], env=env, timeout=180, check=True)
    finally:
        script.unlink(missing_ok=True)
    print(out_pdf, f"{out_pdf.stat().st_size // 1024} KB")


if __name__ == "__main__":
    sys.exit(main())
