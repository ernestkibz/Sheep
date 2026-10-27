"""The seven journeys of the polish plan (step 4), each as a new person on a fresh home, photographed.

    python tooling/ui_audit/journeys.py            # all of them, in order
    python tooling/ui_audit/journeys.py j1 j6      # some

J1  no model, a Nebius key: the first answer, "powerpoint about ...", then "pptx"
J2  no key: the first suggested model downloads and builds, then the first local answer
J3  add any model: the original, then GGUF, GPTQ and EXL2 pages lead to it; wrong pages are refused in words
J4  a research deck with a Sources slide
J5  the local model's answer checked by Nemotron
J6  no network, a wrong key, an engine crash, a nearly full disk: each in words, each with a way back
J7  on battery the checks wait and say why; on the charger they go on

Each journey writes report.json and photos under scratchpad/ui_audit/<journey>-<time>/ (or $UI_AUDIT_OUT).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from playwright.sync_api import sync_playwright  # noqa: E402
from ux_sim import ROOT, Sim  # noqa: E402

NO_NET = {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost"}
KEYS = ("NEBIUS_API_KEY", "NEBIUS_BASE_URL", "NEBIUS_MODEL")


def wait_for(fn, timeout, every=1.0):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        try:
            v = fn()
            if v:
                return v
        except Exception:
            pass
        time.sleep(every)
    return None


def ready(sim, page, timeout):
    return wait_for(lambda: sim.status(page) == "Ready", timeout, 2)


def set_cfg(sim, **kv):
    path = sim.home / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.update(kv)
    path.write_text(json.dumps(cfg), encoding="utf-8")


def dismiss(page):
    """Close whatever sits over the chat (the add-model dialog, a question, Settings), as a person would."""
    for _ in range(2):
        page.keyboard.press("Escape")
        time.sleep(0.4)
    page.evaluate("() => { const b = document.querySelector('#closeSettings'); if (b && b.offsetParent) b.click(); "
                  "document.querySelectorAll('.media-choice-overlay [data-close]').forEach((x) => x.click()); }")
    time.sleep(0.5)


def made(sim, ext):
    return [f for f in sim.files() if f.lower().endswith(ext) and not f.startswith(".mlmi/")]


def models_tab(page):
    if not page.locator("#settings").is_visible():
        page.click("#openSettings")
        time.sleep(0.6)
    page.click('#setTabs [data-tab="models"]')
    time.sleep(1.5)


def check_in_models(sim, page, repo, name, timeout=120):
    """Type an ID in Add any model, press Check, and return what the page says."""
    models_tab(page)
    page.fill("#qSrc", repo)
    page.click("#qLook")
    said = wait_for(lambda: (page.locator("#qInfo").inner_text().strip() or None)
                    if "Checking" not in page.locator("#qInfo").inner_text() else None, timeout)
    time.sleep(1)
    sim.shot(page, name)
    return (said or "") + " " + page.locator("#qLevels").inner_text()


# ------------------------------------------------------------------ J1
def j1(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    sim.shot(page, "welcome")
    page.click("#wCloud")
    time.sleep(1.5)
    first, done, reply = sim.ask(page, "Hello! In one sentence, what can you help me with?")
    sim.log("the first answer comes from Nemotron", done is not None and len(reply) > 20, first_s=first, reply=reply[-200:])
    first, done, reply = sim.ask(page, "Make a powerpoint about saving water at home", timeout=600)
    sim.shot(page, "deck")
    sim.log("a deck is made", bool(made(sim, ".sdeck")), files=made(sim, ".sdeck"), reply=reply[-200:])
    first, done, reply = sim.ask(page, "pptx", timeout=300)
    sim.shot(page, "pptx")
    sim.log("'pptx' saves the PowerPoint", bool(made(sim, ".pptx")), files=made(sim, ".pptx"), reply=reply[-200:])


# ------------------------------------------------------------------ J2
def j2(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    sim.shot(page, "welcome-no-key")
    sim.log("with no key the welcome offers the key, not a dead cloud button",
            page.locator("#wKeys").count() == 1 and page.locator("#wCloud").count() == 0)
    page.click("#wChoose")
    page.wait_for_selector(".model-slot-choice", timeout=10000)
    page.click('.model-slot-choice [data-ex="Qwen/Qwen3-0.6B"]')
    ok = wait_for(lambda: page.locator(".model-slot-choice [data-use]").is_enabled(), 180)
    sim.shot(page, "checked")
    sim.log("the suggested model is checked and offered", bool(ok),
            said=page.locator(".model-slot-choice [data-slot-result]").inner_text()[:300])
    if not ok:
        return
    page.click(".model-slot-choice [data-use]")
    t = time.monotonic()
    seen = set()
    while time.monotonic() - t < 3600 and sim.status(page) != "Ready":
        steps = page.locator(".model-slot-choice .story li").all_inner_texts() if page.locator(".model-slot-choice").count() else []
        key = tuple(steps)
        if steps and key not in seen:
            seen.add(key)
            sim.shot(page, f"steps-{len(seen)}")
        time.sleep(5)
    sim.log("it downloads, builds and becomes Ready", sim.status(page) == "Ready",
            minutes=round((time.monotonic() - t) / 60, 1), steps_seen=[list(s) for s in seen][-1:] if seen else [])
    # The header can turn Ready just before the chooser's next poll clears its
    # close guard. Wait for Check to reappear so dismiss() does not race that poll.
    closable = wait_for(lambda: page.locator(".model-slot-choice [data-check]").is_visible(), 30, 0.5)
    sim.log("the model chooser is ready to close", bool(closable))
    if not closable:
        return
    dismiss(page)
    first, done, reply = sim.ask(page, "What is the capital of Kenya? Answer in one sentence.")
    sim.shot(page, "first-local-answer")
    sim.log("the first local answer", done is not None and "nairobi" in reply.lower(), first_s=first, reply=reply[-200:])
    page.click("#openSettings")
    page.click('#setTabs [data-tab="orchestrator"]')
    def setup_story():
        text = page.locator("#sStory").inner_text()
        return text if "Added" in text and "Placed it" in text else None
    story = wait_for(setup_story, 30, 0.5) or page.locator("#sStory").inner_text()
    sim.shot(page, "what-sheep-did")
    sim.log("What Sheep did for you tells the setup", "Added" in story and "Placed it" in story, story=story[:400])


# ------------------------------------------------------------------ J3
def j3(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    original = "Qwen/Qwen2.5-0.5B-Instruct"
    for repo, name in (("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "gguf"), ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "gptq"),
                       ("Volko76/Qwen2.5-0.5B-Instruct-EXL2-4bits", "exl2")):
        said = check_in_models(sim, page, repo, "check-" + name)
        sim.log(f"a {name.upper()} page leads to its original", original in said, said=said[:300])
    for repo, name, words in (("nobody-here/no-such-model-xyz", "missing", "no model called"),
                              ("openai/whisper-tiny", "not-text", "not for chat")):
        said = check_in_models(sim, page, repo, "check-" + name)
        sim.log(f"a {name} page is refused in words", words in said.lower(), said=said[:300])
    said = check_in_models(sim, page, original, "check-original")
    sim.log("the original is checked with sizes for this computer", "GB" in said, said=said[:300])
    page.click("#qGo")
    t = time.monotonic()
    shots = 0
    while time.monotonic() - t < 3600:
        q = page.evaluate("() => fetch('/api/quantize').then((r) => r.json())")
        if q.get("state") in ("done", "failed"):
            break
        if shots < 6 and int(time.monotonic() - t) % 60 < 6:
            shots += 1
            sim.shot(page, f"building-{shots}")
        time.sleep(5)
    sim.shot(page, "built")
    sim.log("the original builds", q.get("state") == "done", minutes=round((time.monotonic() - t) / 60, 1),
            log=(q.get("log") or [])[-3:])
    sim.log("and becomes Ready", bool(ready(sim, page, 900)))
    dismiss(page)
    first, done, reply = sim.ask(page, "Name three colours of the rainbow.")
    sim.log("it answers", done is not None and len(reply) > 5, reply=reply[-200:])


# ------------------------------------------------------------------ J4
def j4(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    sim.log("model Ready", bool(ready(sim, page, 600)))
    first, done, reply = sim.ask(page, "Research the latest news about solar power in Kenya and make a short deck with a Sources slide",
                                 timeout=1200)
    sim.shot(page, "research-deck")
    decks = made(sim, ".sdeck")
    body = (sim.ws / decks[0]).read_text(encoding="utf-8").lower() if decks else ""
    sim.log("a research deck with a Sources slide", bool(decks) and "sources" in body and "http" in body,
            files=decks, reply=reply[-200:])


# ------------------------------------------------------------------ J5
def j5(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    sim.log("model Ready", bool(ready(sim, page, 600)))
    # the video's own two prompts: Nemotron plans the deck, then the local model adds slides and Nemotron checks them
    sim.ask(page, "powerpoint about the history of coffee", timeout=900)
    first, done, reply = sim.ask(page, "add three slides about coffee in Uganda to the deck", timeout=900)
    time.sleep(2)
    card = page.locator("details.plan").all_inner_texts()
    sim.shot(page, "checked-by-nemotron")
    sim.log("the local model's slides are checked by Nemotron", any("Checked by" in c for c in card),
            card=card[-1:] if card else [], reply=reply[-200:])


# ------------------------------------------------------------------ J6 (four homes)
def j6_offline(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    said = check_in_models(sim, page, "Qwen/Qwen2.5-0.5B-Instruct", "offline-check")
    sim.log("no network: the check says so in words", "could not reach hugging face" in said.lower(), said=said[:300])
    page.click("#closeSettings")
    if page.locator("#wCloud").count():
        page.click("#wCloud")
        time.sleep(1)
        first, done, reply = sim.ask(page, "Hello?", timeout=180)
        sim.shot(page, "offline-chat")
        sim.log("no network: chat says what happened", done is not None and any(
            w in reply.lower() for w in ("connect", "network", "internet", "reach")), reply=reply[-300:])


def j6_wrong_key(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    page.click("#wCloud")
    time.sleep(1)
    first, done, reply = sim.ask(page, "Hello?", timeout=180)
    sim.shot(page, "wrong-key")
    sim.log("a wrong key: chat says the key was refused and where to fix it", done is not None and "key" in reply.lower()
            and "settings" in reply.lower(), reply=reply[-300:])


def j6_crash(sim, page):
    import psutil
    sim.log("app up", sim.wait_app(page) is not None)
    sim.log("model Ready", bool(ready(sim, page, 600)))
    eng = [p for p in psutil.process_iter(["name", "cmdline"]) if "sml-server" in (p.info["name"] or "")
           and str(sim.home) in " ".join(p.info["cmdline"] or []) + str(p.environ().get("MLMI_HOME", "") if hasattr(p, "environ") else "")]
    if not eng:
        backend = [p for p in psutil.process_iter(["cmdline"]) if str(sim.app_port) in " ".join(p.info["cmdline"] or [])]
        eng = [c for b in backend for c in b.children(recursive=True) if "sml-server" in c.name()]
    for p in eng:
        p.kill()
    sim.log("the engine is killed", bool(eng))
    time.sleep(3)
    sim.shot(page, "engine-killed")
    first, done, reply = sim.ask(page, "Are you back? Answer yes or no.", timeout=600)
    sim.shot(page, "after-crash")
    sim.log("after an engine crash chat comes back by itself", done is not None and len(reply) > 1, reply=reply[-200:],
            status=sim.status(page))


def j6_disk(sim, page):
    sim.log("app up", sim.wait_app(page) is not None)
    said = check_in_models(sim, page, "Qwen/Qwen2.5-0.5B-Instruct", "full-disk")
    sim.log("a nearly full disk: says how much to free, no download starts", "free up" in said.lower(), said=said[:300])


# ------------------------------------------------------------------ J7
def j7(sim, page, power_file):
    sim.log("app up", sim.wait_app(page) is not None)
    sim.log("model Ready on battery", bool(ready(sim, page, 600)))
    time.sleep(20)
    page.click("#openSettings")
    page.click('#setTabs [data-tab="dashboard"]')
    time.sleep(3)
    word = page.locator('[data-k="w-hardware"]').inner_text()
    sim.shot(page, "battery")
    # a hot laptop says "Cooling" first (heat outranks the battery); either way the checks wait and say why
    sim.log("on battery the checks wait and say so", any(x in word.lower() for x in ("battery", "cooling")), tile=word)
    power_file.write_text("ac", encoding="utf-8")
    time.sleep(40)
    page.click('#setTabs [data-tab="dashboard"]')
    time.sleep(3)
    word = page.locator('[data-k="w-hardware"]').inner_text()
    sim.shot(page, "charger")
    sim.log("on the charger they go on", "battery" not in word.lower(), tile=word)


def run(name, fn, persona="new", extra=None, drop=(), cfg=None, *args):
    sim = Sim(persona)
    sim.persona = name
    if cfg:
        set_cfg(sim, **cfg)
    cdp = sim.launch(extra=extra, drop=drop)
    try:
        with sync_playwright() as pw:
            page = sim.page(pw, cdp)
            sim.say_yes(page)
            fn(sim, page, *args)
    except Exception as e:
        sim.log(f"{name} ran to the end", False, error=repr(e)[:400])
    finally:
        sim.quit()
        sim.finish()


def main():
    want = [a.lower() for a in sys.argv[1:]] or ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]
    if "j1" in want:
        run("j1", j1)
    if "j2" in want:
        run("j2", j2, "new", None, KEYS)
    if "j3" in want:
        run("j3", j3, "new", None, KEYS)
    if "j4" in want:
        run("j4", j4, "returning", None, (), {"mode": 1})
    if "j5" in want:
        run("j5", j5, "returning", None, (), {"mode": 1})
    if "j6" in want:
        run("j6-offline", j6_offline, "new", NO_NET)
        run("j6-wrong-key", j6_wrong_key, "new", {"NEBIUS_API_KEY": "sk-or-v1-not-a-real-key"})
        run("j6-crash", j6_crash, "returning")
        run("j6-disk", j6_disk, "new", {"MLMI_FAKE_FREE_GB": "0.4"})
    if "j7" in want:
        power = Path(ROOT / "scratchpad" / "fake-power.txt")
        power.parent.mkdir(parents=True, exist_ok=True)
        power.write_text("battery:60", encoding="utf-8")
        run("j7", j7, "returning", {"MLMI_FAKE_POWER": "@" + str(power)}, (), None, power)


if __name__ == "__main__":
    main()
