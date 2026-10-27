"""Settings in different situations, photographed (2026-10-13). Two launches:
a first-time user with no model, and a returning user with the 3B loaded. In the
second, each device/pool/heat choice is set through the window, the saved config
is read back, and the Orchestrator page is photographed in that state. Everything
is set back at the end. Writes scenarios.json beside the screenshots."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from playwright.sync_api import sync_playwright  # noqa: E402
from ux_sim import Sim  # noqa: E402

shots = []          # (scenario, tab, file, saved config facts, ok)


def cfg(sim):
    try:
        return json.loads((sim.home / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def open_tab(page, tab):
    if not page.locator("#settings").is_visible():
        page.click("#openSettings")
        time.sleep(0.6)
    page.click(f'#setTabs [data-tab="{tab}"]')
    time.sleep(1.5)
    page.evaluate("() => document.querySelectorAll('details.adv').forEach((d) => d.open = true)")   # the expert fold
    for h in page.locator("#setBody .sect:not(.open) > .sh").all():
        try:
            h.click()
        except Exception:
            pass
    time.sleep(0.5)


def photo(sim, page, scenario, tab, ok=True, facts=None):
    """the whole Settings page, one picture per screen of scrolling"""
    body = page.locator("#settings .setbody")
    h, full = body.evaluate("e => [e.clientHeight, e.scrollHeight]")
    n = 0
    for top in range(0, max(full, 1), max(h - 40, 200)):
        body.evaluate(f"e => e.scrollTop = {top}")
        time.sleep(0.3)
        f = sim.dir / f"{len(shots):03d}-{scenario}-{tab}-{n}.png".replace(" ", "_")
        page.locator("#settings .sheet").screenshot(path=str(f))
        shots.append(dict(scenario=scenario, tab=tab, part=n, file=f.name, ok=ok, facts=facts or {}))
        n += 1
        if n >= 4:
            break
    body.evaluate("e => e.scrollTop = 0")


def wait_cfg(sim, key, want):
    for _ in range(14):
        time.sleep(0.5)
        if cfg(sim).get(key) == want:
            return True
    return False


def first_user(page, sim):
    sim.log("app opens (no model)", sim.wait_app(page) is not None)
    time.sleep(3)
    for tab in ("general", "dashboard", "orchestrator", "machine", "models", "herders", "files", "cloud"):
        open_tab(page, tab)
        photo(sim, page, "first-time user, no model", tab)


def returning(page, sim):
    sim.log("app opens (3B)", sim.wait_app(page) is not None)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 300 and sim.status(page) != "Ready":
        time.sleep(1)
    sim.log("model ready", sim.status(page) == "Ready")
    f = sim.dir / f"{len(shots):03d}-chat-with-checks-card.png"
    page.screenshot(path=str(f))
    shots.append(dict(scenario="chat with the checks card", tab="chat", part=0, file=f.name, ok=True, facts={}))
    page.on("dialog", lambda d: d.accept())
    # Sheep's own yes/no box: photographed, then Escape must cancel and leave Settings open
    open_tab(page, "machine")
    page.click("#hHeat")
    time.sleep(0.6)
    boxed = page.locator(".modal.ask .askbox").count() == 1
    f = sim.dir / f"{len(shots):03d}-ask-heat-guard-off.png"
    page.screenshot(path=str(f))
    shots.append(dict(scenario="Sheep's own yes/no box", tab="machine", part=0, file=f.name, ok=boxed, facts={}))
    page.keyboard.press("Escape")
    time.sleep(1)
    kept = cfg(sim).get("thermal_guard", True) is not False and page.locator("#settings").is_visible()         and page.locator(".modal.ask").count() == 0
    sim.log("heat guard off asks in Sheep's box; Escape keeps it on", boxed and kept)
    sim.say_yes(page)
    for tab in ("general", "dashboard", "orchestrator", "machine", "models", "herders"):
        open_tab(page, tab)
        photo(sim, page, "model loaded, defaults", tab)

    def scene(name, key, want, act):
        open_tab(page, "machine")
        page.locator("#hText button").first.wait_for(timeout=20000)
        try:
            act()
            ok = wait_cfg(sim, key, want)
        except Exception as e:
            ok = False
            sim.log(name, False, error=repr(e)[:200])
        time.sleep(2.5)           # let the plan redraw
        open_tab(page, "machine")
        c = cfg(sim)
        facts = {k: c.get(k) for k in ("device", "pool_mode", "pool_gap", "cpu_offload", "thermal_guard",
                                       "vram_reserve_gb", "gpus_off", "theme") if k in c}
        sim.log(name, ok, saved=facts)
        photo(sim, page, name, "machine", ok, facts)

    open_tab(page, "machine")
    page.locator("#hText button").first.wait_for(timeout=20000)
    devices = [b.get_attribute("data-t") for b in page.locator("#hText button:not(.off)").all()]
    labels = {b.get_attribute("data-t"): b.inner_text().strip() for b in page.locator("#hText button").all()}
    sim.log("text device choices", bool(devices), choices=labels)
    for d in devices:
        if d != "auto":
            scene(f"text model on {labels.get(d, d)}", "device", d,
                  lambda d=d: page.click(f'#hText button[data-t="{d}"]'))
    scene("text model on Auto", "device", "auto", lambda: page.click('#hText button[data-t="auto"]'))
    for m in ("off", "always", "smart"):
        scene(f"pool GPUs {m}", "pool_mode", m, lambda m=m: page.select_option("#hPool", m))
    scene("speed gap 8x", "pool_gap", 8.0, lambda: (page.fill("#hGap", "8"), page.dispatch_event("#hGap", "change")))
    scene("keep 1 GB VRAM free", "vram_reserve_gb", 1.0,
          lambda: (page.fill("#hReserve", "1"), page.dispatch_event("#hReserve", "change")))
    scene("GPU and CPU together off", "cpu_offload", False, lambda: page.click("#hOffload"))
    scene("heat guard off", "thermal_guard", False, lambda: page.click("#hHeat"))
    boxes = page.locator("#hGpusOn [data-gpu]")
    if boxes.count() > 1:
        name = boxes.last.get_attribute("data-gpu")
        open_tab(page, "machine")
        page.locator("#hGpusOn [data-gpu]").last.uncheck()
        time.sleep(3)
        open_tab(page, "machine")
        off = name in (cfg(sim).get("gpus_off") or [])
        sim.log(f"second GPU off ({name})", off)
        photo(sim, page, f"GPU off: {name}", "machine", off, {"gpus_off": cfg(sim).get("gpus_off")})
        page.locator("#hGpusOn [data-gpu]").last.check()
    open_tab(page, "general")
    page.click('#lTheme button[data-t="light"]')
    wait_cfg(sim, "theme", "light")
    for tab in ("general", "dashboard", "orchestrator", "machine", "models", "herders"):
        open_tab(page, tab)
        photo(sim, page, "light theme", tab, cfg(sim).get("theme") == "light", {"theme": "light"})

    # everything back
    open_tab(page, "general")
    page.click('#lTheme button[data-t="dark"]')
    open_tab(page, "machine")
    page.locator("#hText button").first.wait_for(timeout=20000)
    page.fill("#hGap", "4"); page.dispatch_event("#hGap", "change")
    page.fill("#hReserve", "0"); page.dispatch_event("#hReserve", "change")
    if not page.locator("#hOffload.on").count():
        page.click("#hOffload")
    if not page.locator("#hHeat.on").count():
        page.click("#hHeat")
    time.sleep(2)
    c = cfg(sim)
    sim.log("all set back", c.get("cpu_offload") is not False and c.get("thermal_guard") is not False
            and c.get("theme") == "dark", saved={k: c.get(k) for k in ("pool_gap", "vram_reserve_gb", "cpu_offload", "thermal_guard", "theme")})


for persona, fn in (("new", first_user), ("returning", returning)):
    sim = Sim(persona)
    sim.persona = f"settings-{persona}"
    cdp = sim.launch()
    try:
        with sync_playwright() as pw:
            fn(sim.page(pw, cdp), sim)
    except Exception as e:
        sim.log("ran to the end", False, error=repr(e)[:400])
    finally:
        sim.quit()
        sim.finish()
    for s in shots:
        s.setdefault("dir", sim.dir.name)
out = Path(os.environ.get("UI_AUDIT_OUT") or (Path(__file__).resolve().parents[2] / "scratchpad" / "ui_audit")) / "scenarios.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(shots, indent=1), encoding="utf-8")
print("shots:", len(shots), "->", out)
