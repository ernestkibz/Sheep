"""Every Settings control, used through the real window and checked in the saved
config (2026-10-13). Each change is made, read back from config.json, and set back.
Console errors are collected throughout.

    python tooling/ui_audit/settings_audit.py
    SETTINGS_AUDIT_PERSONA=new python tooling/ui_audit/settings_audit.py   # no model loaded: a cool run
"""
import os
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from playwright.sync_api import sync_playwright  # noqa: E402
from ux_sim import Sim  # noqa: E402

PERSONA = os.environ.get("SETTINGS_AUDIT_PERSONA", "returning")
sim = Sim(PERSONA)
sim.persona = "settings"
cfg_path = sim.home / "config.json"


def cfg():
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def settle(page, s=1.2):
    time.sleep(s)


def open_tab(page, tab):
    if not page.locator("#settings").is_visible():
        page.click("#openSettings")
        settle(page, 0.6)
    page.click(f'#setTabs [data-tab="{tab}"]')
    settle(page, 1.5)
    page.evaluate("() => document.querySelectorAll('details.adv').forEach((d) => d.open = true)")   # the expert fold
    for h in page.locator("#setBody .fold:not(.open) > .sh").all():    # Advanced: every fold open
        try:
            h.click()
            settle(page, 0.8)
        except Exception:
            pass
    page.evaluate("() => document.querySelectorAll('details.adv').forEach((d) => d.open = true)")
    for h in page.locator("#setBody .sect:not(.open) > .sh").all():    # every section open, as a person would
        try:
            h.click()
        except Exception:
            pass
    settle(page, 0.6)
    body = page.locator("#setBody").inner_text().strip()
    sim.shot(page, f"tab-{tab}")
    sim.log(f"tab {tab} draws", len(body) > 20, chars=len(body))


def expect(name, key, want, act, page):
    """do act(), then config.json[key] must equal want (or satisfy it, if callable)"""
    before = cfg().get(key)
    try:
        act()
    except Exception as e:
        return sim.log(name, False, error=repr(e)[:200])
    for _ in range(12):
        settle(page, 0.5)
        got = cfg().get(key)
        if (want(got) if callable(want) else got == want):
            return sim.log(name, True, key=key, before=before, after=got)
    sim.log(name, False, key=key, before=before, after=cfg().get(key), wanted=str(want))


def mini(page, host, value):
    page.click(f'{host} button[data-t="{value}"]')


def run(page):
    up = sim.wait_app(page)
    sim.log("app opens", up is not None, seconds=up and round(up, 1))
    t0 = time.monotonic()
    while PERSONA == "returning" and time.monotonic() - t0 < 300 and sim.status(page) != "Ready":
        time.sleep(1)
    if PERSONA == "returning":
        sim.log("model ready", sim.status(page) == "Ready", status=sim.status(page))
    page.on("dialog", lambda d: d.accept())
    sim.say_yes(page)
    for tab in ("basics", "models", "accounts", "activity", "advanced"):
        open_tab(page, tab)

    # Basics: the three plain choices are bundles of the Advanced settings
    open_tab(page, "basics")
    expect("preset cool and quiet", "pool_mode", "off", lambda: page.click('#bPreset [data-preset="quiet"]'), page)
    expect("quiet keeps 2 GB free", "vram_reserve_gb", 2.0, lambda: None, page)
    expect("preset fastest", "pool_mode", "always", lambda: page.click('#bPreset [data-preset="fast"]'), page)
    expect("preset balanced", "pool_mode", "smart", lambda: page.click('#bPreset [data-preset="balanced"]'), page)
    expect("balanced keeps 0 GB free", "vram_reserve_gb", 0.0, lambda: None, page)
    expect("web search off (Basics)", "web", False, lambda: page.click("#bWeb"), page)
    expect("web search on (Basics)", "web", True, lambda: page.click("#bWeb"), page)
    expect("theme same as Windows", "theme", "system", lambda: mini(page, "#lTheme", "system"), page)
    expect("theme light", "theme", "light", lambda: mini(page, "#lTheme", "light"), page)
    expect("theme back to dark", "theme", "dark", lambda: mini(page, "#lTheme", "dark"), page)
    expect("permission ask first", "permission", "ask", lambda: mini(page, "#wPerm", "ask"), page)
    expect("permission back", "permission", "auto", lambda: mini(page, "#wPerm", "auto"), page)

    open_tab(page, "advanced")
    page.locator("#hText button").first.wait_for(timeout=15000)
    choices = [b.get_attribute("data-t") for b in page.locator("#hText button:not(.off)").all()]
    other = next((c for c in choices if c != cfg().get("device", "auto")), None)
    if other:
        expect(f"text device -> {other}", "device", other, lambda: mini(page, "#hText", other), page)
        page.locator("#hText button").first.wait_for(timeout=15000)
        expect("text device back to auto", "device", "auto", lambda: mini(page, "#hText", "auto"), page)
    media = [b.get_attribute("data-t") for b in page.locator("#hMedia button:not(.off)").all()]
    mo = next((c for c in media if c != cfg().get("media_device", "auto")), None)
    if mo:
        expect(f"media device -> {mo}", "media_device", mo, lambda: mini(page, "#hMedia", mo), page)
        page.locator("#hMedia button").first.wait_for(timeout=15000)
        expect("media device back to auto", "media_device", "auto", lambda: mini(page, "#hMedia", "auto"), page)
    expect("checker training cpu", "checker_training_mode", "cpu",
           lambda: page.select_option("#hCheckerTraining", "cpu"), page)
    expect("checker training auto", "checker_training_mode", "auto",
           lambda: page.select_option("#hCheckerTraining", "auto"), page)
    expect("pool GPUs off", "pool_mode", "off", lambda: page.select_option("#hPool", "off"), page)
    expect("pool GPUs smart", "pool_mode", "smart", lambda: page.select_option("#hPool", "smart"), page)
    expect("speed gap 6", "pool_gap", 6.0, lambda: (page.fill("#hGap", "6"), page.dispatch_event("#hGap", "change")), page)
    expect("VRAM keep free 1 GB", "vram_reserve_gb", 1.0,
           lambda: (page.fill("#hReserve", "1"), page.dispatch_event("#hReserve", "change")), page)
    expect("VRAM keep free 0", "vram_reserve_gb", 0.0,
           lambda: (page.fill("#hReserve", "0"), page.dispatch_event("#hReserve", "change")), page)
    expect("GPU+CPU off", "cpu_offload", False, lambda: page.click("#hOffload"), page)
    expect("GPU+CPU on", "cpu_offload", True, lambda: page.click("#hOffload"), page)
    boxes = page.locator("#hGpusOn [data-gpu]")
    if boxes.count():
        name = boxes.last.get_attribute("data-gpu")
        expect(f"GPU off: {name}", "gpus_off", lambda v: name in (v or []), lambda: boxes.last.uncheck(), page)
        expect(f"GPU on: {name}", "gpus_off", lambda v: name not in (v or []), lambda: boxes.last.check(), page)
    expect("heat guard off", "thermal_guard", False, lambda: page.click("#hHeat"), page)
    expect("heat guard on", "thermal_guard", True, lambda: page.click("#hHeat"), page)
    open_tab(page, "basics")
    custom = page.locator("#bPresetNote").inner_text()
    sim.log("a hand change shows Custom", "Custom" in custom or "custom" in custom, text=custom[:120])
    expect("back to balanced", "pool_gap", 4.0, lambda: page.click('#bPreset [data-preset="balanced"]'), page)
    open_tab(page, "advanced")
    rt = page.locator("#hGpuRt").inner_text()
    sim.log("GPU support lists each GPU's runtime", "ready" in rt or "missing" in rt, text=rt[:200])
    setup = page.locator("#hSetupList").inner_text()
    sim.log("setup checklist draws", "✓" in setup or "✗" in setup, text=setup[:200])

    open_tab(page, "advanced")
    expect("web search off", "web", False, lambda: page.click("#wSwitch"), page)
    settle(page, 1)
    expect("web search on", "web", True, lambda: page.click("#wSwitch"), page)

    page.click("#closeSettings")
    settle(page, 0.5)
    page.click("#modelBtn")
    settle(page, 1)
    menu = page.locator("#modelMenu").inner_text()
    sim.shot(page, "model-menu")
    sim.log("model menu lists models or 'No ... model'", "chat" in menu.lower() and ("No " in menu or "model" in menu.lower()),
            text=menu[:200])
    if not page.locator("#modelMenu [data-model]").count():
        page.keyboard.press("Escape")
        return
    page.locator("#modelMenu [data-model]").first.click()
    settle(page, 0.6)
    acts = page.locator("#modelMenu").inner_text()
    sim.shot(page, "model-actions")
    sim.log("a model's menu is Use / Remove / Back", "Remove" in acts and "Back" in acts, text=acts[:160])
    page.locator('#modelMenu [data-act="back"]').click()
    settle(page, 0.6)
    page.keyboard.press("Escape")


cdp = sim.launch()
try:
    with sync_playwright() as pw:
        run(sim.page(pw, cdp))
except Exception as e:
    sim.log("audit ran to the end", False, error=repr(e)[:400])
finally:
    sim.quit()
    sim.finish()
