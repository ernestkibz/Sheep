"""Photographs of every screen, dark and light, wide and narrow, on a fresh home (polish plan, step 3).

    python tooling/ui_audit/photos.py            # the 3B model set, as a returning person
    python tooling/ui_audit/photos.py new        # first launch, no model

Writes <tab>-<theme>-<width>.png and report.json under scratchpad/ui_audit/photos-<time>/ (or $UI_AUDIT_OUT).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from playwright.sync_api import sync_playwright  # noqa: E402
from ux_sim import Sim  # noqa: E402

TABS = ["general", "dashboard", "orchestrator", "machine", "models", "herders", "files", "cloud"]
WIDTHS = [1920, 1280, 900]

sim = Sim("new" if "new" in sys.argv[1:] else "returning")
sim.persona = "photos"


def theme(page, name):
    page.evaluate("""async (t) => { await fetch('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({theme: t})}); document.documentElement.dataset.theme = t; }""", name)
    time.sleep(1.5)


def snap(page, name):
    page.screenshot(path=str(sim.dir / f"{name}.png"), full_page=True)


def run(page):
    took = sim.wait_app(page)
    sim.log("app up", took is not None, s=took)
    time.sleep(4)
    for t in ("dark", "light"):
        theme(page, t)
        for w in WIDTHS:
            page.set_viewport_size({"width": w, "height": 900})
            time.sleep(0.8)
            if page.locator("#settings").is_visible():
                page.click("#closeSettings")
            snap(page, f"chat-{t}-{w}")
            page.click("#openSettings")
            time.sleep(0.8)
            for tab in TABS:
                page.click(f'#setTabs [data-tab="{tab}"]')
                time.sleep(2.0)
                for h in page.locator("#setBody .sect:not(.open) > .sh").all():
                    try:
                        h.click()
                    except Exception:
                        pass
                time.sleep(0.6)
                # the page scrolls inside the Settings sheet: one photo per screenful, top first
                n = page.evaluate("() => { const b = document.querySelector('#settings .setbody'); b.scrollTop = 0;"
                                  " return Math.min(5, Math.ceil(b.scrollHeight / b.clientHeight)); }")
                for i in range(n):
                    page.evaluate("(i) => { const b = document.querySelector('#settings .setbody');"
                                  " b.scrollTop = i * (b.clientHeight - 60); }", i)
                    time.sleep(0.3)
                    snap(page, f"{tab}-{t}-{w}-{i + 1}")
            page.click("#closeSettings")
    sim.log("photographed", True, n=len(list(sim.dir.glob("*.png"))))


cdp = sim.launch()
try:
    with sync_playwright() as pw:
        run(sim.page(pw, cdp))
except Exception as e:
    sim.log("photos ran to the end", False, error=repr(e)[:400])
finally:
    sim.quit()
    sim.finish()
    print(sim.dir)
