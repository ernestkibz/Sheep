"""The README and Devpost pictures, taken from the running app (polish plan, step 6).

    python tooling/ui_audit/gallery.py                 # a fresh home with the 3B, Nemotron on
    python tooling/ui_audit/gallery.py --home DIR      # reuse a journey's home (its model's full setup story)

Writes mlmi/media/gallery/{1-chat,2-what-sheep-did,3-words,4-sheets,5-slides}.png and mlmi/media/thumbnail.png.
Needs the Nebius key in the environment: the deck is planned by Nemotron, as in the video.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from playwright.sync_api import sync_playwright  # noqa: E402
from ux_sim import ROOT, Sim  # noqa: E402

OUT = ROOT / "mlmi" / "media" / "gallery"
W, H = 1440, 900


def main():
    sim = Sim("returning")
    sim.persona = "gallery"
    if "--home" in sys.argv:
        sim.home = Path(sys.argv[sys.argv.index("--home") + 1])
        sim.ws = Path(json.loads((sim.home / "config.json").read_text(encoding="utf-8"))["workspace"])
    cfg_path = sim.home / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.update(mode=1, theme="dark")
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    cdp = sim.launch()
    try:
        with sync_playwright() as pw:
            page = sim.page(pw, cdp)
            page.set_viewport_size({"width": W, "height": H})
            sim.say_yes(page)
            sim.wait_app(page)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline and sim.status(page) != "Ready":
                time.sleep(2)
            OUT.mkdir(parents=True, exist_ok=True)

            def snap(name):
                time.sleep(1.5)
                page.screenshot(path=str(OUT / f"{name}.png"))
                print("took", name)

            def close_canvas():
                if page.locator("#canvas").is_visible():
                    page.click("#closeCanvas")
                    time.sleep(0.5)

            def open_app(app, ext, name):
                close_canvas()
                page.click(f'.tile[data-app="{app}"]')
                time.sleep(2)
                first = page.locator(f'#canvasBody .filelist [data-p$=".{ext}"]').first
                if first.count():
                    first.click()
                    time.sleep(2.5)
                snap(name)

            sim.ask(page, "Make a budget spreadsheet with Rent 800, Food 250, Bus 90 and a Total row", timeout=600)
            sim.ask(page, "Write a one-page project brief for a school vegetable garden", timeout=600)
            page.click("#newChat")
            time.sleep(1)
            sim.ask(page, "powerpoint about the history of coffee", timeout=900)
            if not page.locator("#canvas").is_visible() and page.locator("#workBtn").count():
                page.click("#workBtn")
            snap("1-chat")
            open_app("words", "sdoc", "3-words")
            open_app("sheets", "ssheet", "4-sheets")
            open_app("slides", "sdeck", "5-slides")
            close_canvas()
            page.click("#openSettings")
            page.click('#setTabs [data-tab="orchestrator"]')
            time.sleep(4)
            snap("2-what-sheep-did")
            page.click("#closeSettings")
            from PIL import Image
            im = Image.open(OUT / "1-chat.png").convert("RGB")
            w = round(im.height * 1.5)                         # the thumbnail is 3:2; crop, never squash
            left = (im.width - w) // 2
            im.crop((left, 0, left + w, im.height)).resize((1500, 1000)).save(ROOT / "mlmi" / "media" / "thumbnail.png")
    finally:
        sim.quit()
        sim.finish()


if __name__ == "__main__":
    main()
