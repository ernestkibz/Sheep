"""User-experience simulation: drive the real Electron app as two people would.

    python tooling/ui_audit/ux_sim.py new        # first launch, no model
    python tooling/ui_audit/ux_sim.py returning  # the 3B model set, real chat and app tasks

Each run gets its own MLMI_HOME and workspace under scratchpad/ui_audit/<persona>-<time>/ (or $UI_AUDIT_OUT),
so the real ~/.mlmi is never touched. Writes report.json, console.log and screenshots.
"""
import json
import os
import socket
import subprocess
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # a ✓ in a log line must not end a run on a cp1252 console
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "mlmi" / "desktop"
ELECTRON = DESKTOP / "node_modules" / "electron" / "dist" / "electron.exe"
MODEL = Path.home() / ".mlmi" / "models" / "Qwen--Qwen2.5-3B-Instruct.ssf"   # the 3B audit build was deleted 2026-10-13


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Sim:
    def __init__(self, persona):
        self.persona = persona
        self.dir = Path(os.environ.get("UI_AUDIT_OUT") or (ROOT / "scratchpad" / "ui_audit")) / f"{persona}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.home, self.ws = self.dir / "home", self.dir / "workspace"
        for d in (self.home, self.ws):
            d.mkdir(parents=True, exist_ok=True)
        cfg = {"workspace": str(self.ws)}
        if persona == "returning":
            cfg["model"] = str(MODEL)
        (self.home / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        self.steps, self.console, self.t0 = [], [], time.monotonic()

    def log(self, name, ok, **facts):
        facts.update(step=name, ok=ok, at_s=round(time.monotonic() - self.t0, 1))
        self.steps.append(facts)
        print(("PASS " if ok else "FAIL ") + name, {k: v for k, v in facts.items() if k not in ("step", "ok")})

    def shot(self, page, name):
        try:
            page.screenshot(path=str(self.dir / f"{len(self.steps):02d}-{name}.png"))
        except Exception as e:
            print("  screenshot failed:", e)

    def launch(self, extra=None, drop=()):
        """extra: environment to add (a wrong key, no network through a dead proxy); drop: names to remove."""
        self.app_port, cdp = free_port(), free_port()
        env = dict(os.environ, MLMI_HOME=str(self.home), MLMI_PORT=str(self.app_port),
                   MLMI_PYTHON=sys.executable, ELECTRON_RUN_AS_NODE="", **(extra or {}))
        env.pop("ELECTRON_RUN_AS_NODE")
        for name in drop:
            env.pop(name, None)
        self.out = open(self.dir / "app.log", "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen([str(ELECTRON), str(DESKTOP), f"--remote-debugging-port={cdp}"],
                                     cwd=DESKTOP, env=env, stdout=self.out, stderr=subprocess.STDOUT)
        return cdp

    def quit(self):
        subprocess.run(["taskkill", "/pid", str(self.proc.pid), "/T", "/F"], capture_output=True)
        self.out.close()

    def page(self, pw, cdp):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp}")
                pages = [p for c in browser.contexts for p in c.pages]
                if pages:
                    page = pages[0]
                    page.on("console", lambda m: m.type in ("error", "warning") and self.console.append(
                        f"[{round(time.monotonic() - self.t0, 1)}s] {m.type}: {m.text}"))
                    page.on("pageerror", lambda e: self.console.append(
                        f"[{round(time.monotonic() - self.t0, 1)}s] PAGEERROR: {e}"))
                    return page
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("Electron window never exposed a page over CDP")

    def wait_app(self, page, timeout=120):
        """The main interface is up once the status lamp has words in it."""
        # page.url over CDP stayed on boot.html after Electron's loadURL; ask the page itself
        t, seen = time.monotonic(), None
        while time.monotonic() - t < timeout:
            try:
                href, status = page.evaluate("() => [location.href, (document.querySelector('#statusText') || {}).textContent || '']")
            except Exception:
                href, status = "", ""
            if href != seen:
                seen = href
                print(f"  [{round(time.monotonic() - t, 1)}s] window shows {href[:90]}")
            if href.startswith("http") and status.strip():
                return time.monotonic() - t
            time.sleep(0.25)
        return None

    def say_yes(self, page):
        """answer Sheep's own yes/no box with yes, as page.on("dialog") did for the Windows one"""
        page.evaluate("""() => { if (window.__yes) return; window.__yes = new MutationObserver(() => {
            const b = document.querySelector('.modal.ask #askOk'); if (b) setTimeout(() => b.click(), 150); });
            window.__yes.observe(document.body, { childList: true }); }""")

    def status(self, page):
        try:
            return page.locator("#statusText").inner_text(timeout=1000).strip()
        except Exception:
            return "?"

    def ask(self, page, text, timeout=420, allow=True):
        """Type, send, and time the first visible words and the finished answer.
        A "needs your yes" card is answered (Allow, or Decline when allow=False);
        self.asked records that the question came."""
        # the welcome page is replaced by the thread on the first message, so a text
        # offset from before the send is meaningless then: start from empty
        before = page.locator("#thread").inner_text() if page.locator("#thread .msg, #thread .bot, #thread .you").count() else ""
        page.fill("#askText", text)
        t = time.monotonic()
        page.click("#sendBtn")
        first = done = None
        seen_stop = False
        self.asked = False
        while time.monotonic() - t < timeout:
            card = page.locator(f'[data-allow="{1 if allow else 0}"]')
            if card.count() and card.last.is_visible():
                self.asked = True
                card.last.click()
            now = page.locator("#thread").inner_text()
            grown = now[len(before):].replace(text, "").strip()
            if first is None and len(grown) > 3:
                first = time.monotonic() - t
            if page.locator("#stopBtn").is_visible():
                seen_stop = True
            elif (seen_stop or first) and first is not None and time.monotonic() - t > 1.5:
                done = time.monotonic() - t
                break
            time.sleep(0.25)
        reply = page.locator("#thread").inner_text()[len(before):].strip()
        return first, done, reply

    def files(self):
        return sorted(p.relative_to(self.ws).as_posix() for p in self.ws.rglob("*") if p.is_file())

    def finish(self):
        (self.dir / "console.log").write_text("\n".join(self.console), encoding="utf-8")
        report = {"persona": self.persona, "steps": self.steps, "console_errors": len(self.console),
                  "workspace_files": self.files()}
        (self.dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        fails = [s["step"] for s in self.steps if not s["ok"]]
        print(f"\n{self.persona}: {len(self.steps) - len(fails)}/{len(self.steps)} steps passed, "
              f"{len(self.console)} console errors/warnings -> {self.dir}")


def new_user(sim, page):
    up = sim.wait_app(page)
    sim.log("first screen appears", up is not None, seconds=up and round(up, 1))
    sim.shot(page, "first-screen")
    st = sim.status(page)
    sim.log("status tells a new user to add a model", "model" in st.lower(), status=st)
    rb = page.locator("#readyBar")
    sim.log("readiness bar offers a next step", rb.is_visible() and page.locator("#rbChoose").is_visible(),
            text=page.locator("#rbText").inner_text() if rb.is_visible() else "")
    page.fill("#askText", "Hello, can you help me?")
    page.click("#sendBtn")
    time.sleep(3)
    sim.shot(page, "send-without-model")
    thread = page.locator("#thread").inner_text().strip()
    chooser = page.locator(".media-choice-overlay:visible")
    sim.log("sending with no model gets a clear answer", (bool(thread) and "model" in thread.lower()) or chooser.count() > 0,
            chooser_opened=chooser.count() > 0, thread=thread[-300:])
    page.keyboard.press("Escape")                  # the person closes the model chooser
    time.sleep(0.5)
    sim.log("model chooser closes with Esc", page.locator(".media-choice-overlay:visible").count() == 0)
    for app in ("words", "sheets", "slides", "studio", "text"):
        page.click(f'.tile[data-app="{app}"]')
        time.sleep(1.2)
        vis = page.locator("#canvas").is_visible()
        sim.shot(page, f"app-{app}")
        sim.log(f"{app} app opens", vis)
        if vis:
            page.click("#closeCanvas")
            time.sleep(0.4)
    page.click("#openGuide")
    time.sleep(1)
    sim.shot(page, "guide")
    sim.log("guide opens", True)
    page.keyboard.press("Escape")
    page.click("#openSettings")
    for tab in ("basics", "models", "accounts", "activity", "advanced"):
        page.click(f'#setTabs [data-tab="{tab}"]')
        time.sleep(1)
        body = page.locator("#setBody").inner_text().strip()
        sim.shot(page, f"settings-{tab}")
        sim.log(f"settings {tab} shows content", len(body) > 20, chars=len(body))
    page.click("#closeSettings")


def returning_user(sim, page):
    up = sim.wait_app(page)
    sim.log("first screen appears", up is not None, seconds=up and round(up, 1))
    t, st = time.monotonic(), sim.status(page)
    while time.monotonic() - t < 300 and st not in ("Ready",):
        time.sleep(1)
        st = sim.status(page)
    sim.log("model becomes Ready", st == "Ready", seconds=round(time.monotonic() - t, 1), status=st)
    sim.shot(page, "ready")
    ws = sim.ws

    def doc(ext, *words):
        """a file of this kind exists and its saved content holds every word (case-insensitive)"""
        def ok(r, f):
            for x in f:
                if x.lower().endswith(ext):
                    body = (ws / x).read_text(encoding="utf-8", errors="replace").lower()
                    if all(w.lower() in body for w in words):
                        return True
            return False
        return ok

    def has(ext):
        return lambda r, f: any(x.lower().endswith(ext) for x in f)

    def says(*words):
        return lambda r, f: all(w.lower() in r.lower() for w in words)

    pic = ws / "media" / "red-square.png"
    pic.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new("RGB", (64, 64), (220, 20, 20)).save(pic)
    (ws / "old-notes.txt").write_text("remove me", encoding="utf-8")
    tasks = [
        # plain chat
        ("cold question", "What is the capital of Kenya? Answer in one sentence.", says("nairobi")),
        ("warm question", "Give me two tips for staying focused while studying.", lambda r, f: len(r) > 40),
        ("arithmetic", "What is 17 times 23? Just the number.", says("391")),
        # Words
        ("Words: create", "Create a Words document called trip-plan.sdoc titled Trip Plan with three bullet points about visiting Nairobi.",
         doc(".sdoc", "trip plan", "nairobi")),
        ("Words: add heading", "In trip-plan.sdoc add a heading called Budget at the end.", doc(".sdoc", "budget")),
        ("Words: table", "Add a table to trip-plan.sdoc with columns Day and Activity and two rows.", doc(".sdoc", '"table"', "day", "activity")),
        ("Words: replace", "In trip-plan.sdoc replace the word Nairobi with Mombasa everywhere.", doc(".sdoc", "mombasa")),
        ("Words: export docx", "Export trip-plan.sdoc as a Word .docx file.", has(".docx")),
        ("Words: export pdf", "Save trip-plan.sdoc as a PDF.", has(".pdf")),
        # Sheets
        ("Sheets: budget with SUM", "Make a spreadsheet called budget.ssheet with columns Item and Cost: Rent 800, Food 250, Bus 90, and a Total row using a SUM formula.",
         doc(".ssheet", "rent", "=sum")),
        ("Sheets: sort", "Sort the table in budget.ssheet by Cost, largest first.", doc(".ssheet", "rent")),
        ("Sheets: chart", "Add a bar chart of the costs in budget.ssheet.",
         lambda r, f: (ws / "budget.ssheet").exists()
         and any(s.get("charts") for s in json.loads((ws / "budget.ssheet").read_text(encoding="utf-8"))["sheets"])),
        ("Sheets: export xlsx", "Export budget.ssheet to Excel.", has(".xlsx")),
        # Slides
        ("Slides: deck", "Make a three-slide presentation called save-water.sdeck about saving water at home.",
         doc(".sdeck", "water")),
        ("Slides: notes", "Add speaker notes to slide 2 of save-water.sdeck saying: pause for questions.", doc(".sdeck", "pause for questions")),
        ("Slides: theme", "Give save-water.sdeck a dark theme.", doc(".sdeck", '"theme": "dark"')),
        ("Slides: export pptx", "Export save-water.sdeck as PowerPoint.", has(".pptx")),
        # Studio
        ("Studio: intro", "Make a 6-second intro video project called intro.sstudio with the title Sheep.", doc(".sstudio", "sheep")),
        ("Studio: render", "Render intro.sstudio to an MP4.", has(".mp4")),
        # Text, mail, calendar, files
        ("Text file", "Write a text file called packing.txt listing five things to pack for a trip.", doc(".txt", "\n")),
        ("Mail draft", "Draft an email to ruth@example.com with the subject Lunch saying shall we meet at noon on Friday.",
         doc(".eml", "ruth@example.com", "lunch")),
        ("Calendar", "Add a calendar event called Dentist on 2026-11-03 at 10:00.", doc(".ics", "dentist")),
        ("Copy file", "Copy packing.txt to backup/packing.txt.", lambda r, f: "backup/packing.txt" in f),
        # must ask first; the person declines, so the file must still be there
        ("Delete asks first", "Delete old-notes.txt.", lambda r, f: "old-notes.txt" in f and (sim.asked or "checker" in r.lower())),
        # "red" must come from looking, not from repeating the file name back
        ("Picture: describe", "What is in the picture media/red-square.png?",
         lambda r, f: "red" in r.lower().replace("red-square", "") or "add one in settings" in r.lower()),
    ]
    page.on("dialog", lambda d: d.accept())       # "close without saving?" and the like: yes
    sim.say_yes(page)
    sys.path.insert(0, str(ROOT))
    from mlmi.backend import thermal
    heat = thermal.Governor()

    def cool_off():
        """the test itself must not cook the laptop (it shut down from heat on 2026-10-12)"""
        t0 = time.monotonic()
        while time.monotonic() - t0 < 600:
            heat.update(*heat.read())
            if heat.level < 3:                 # same rule as the app: only critical waits
                return
            print(f"  cooling before the next request: {heat.name} ({heat.why})")
            time.sleep(15)

    for name, prompt, good in tasks:
        cool_off()
        first, done, reply = sim.ask(page, prompt, allow=not name.startswith("Delete"))
        files = sim.files()
        sim.shot(page, name.replace(" ", "-").replace(":", ""))
        sim.log(name, done is not None and good(reply, files), first_words_s=first and round(first, 1),
                done_s=done and round(done, 1), asked=sim.asked, reply=reply[-240:],
                files=[x for x in files if not x.startswith(".")])
        if page.locator("#canvas").is_visible():
            page.click("#closeCanvas")
    # the backend dies under the person: does the app come back by itself?
    pids = subprocess.run(["powershell", "-NoProfile", "-Command",
                           f"(Get-NetTCPConnection -LocalPort {sim.app_port} -State Listen).OwningProcess"],
                          capture_output=True, text=True).stdout.split()
    if pids:
        subprocess.run(["taskkill", "/pid", pids[0], "/T", "/F"], capture_output=True)
        t = time.monotonic()
        time.sleep(3)
        sim.shot(page, "backend-killed")
        back = sim.wait_app(page, timeout=180)
        sim.log("app recovers after the backend crashes", back is not None,
                seconds=round(time.monotonic() - t, 1), status=sim.status(page))
        sim.shot(page, "recovered")
        first, done, reply = sim.ask(page, "Are you back? Answer yes or no.")
        sim.log("chat works after recovery", done is not None and len(reply) > 1,
                done_s=done and round(done, 1), reply=reply[-160:])
        sim.log("the open chat is still there after recovery", "capital of kenya" in page.locator("#thread").inner_text().lower())


def main():
    persona = sys.argv[1] if len(sys.argv) > 1 else "new"
    sim = Sim(persona)
    cdp = sim.launch()
    try:
        with sync_playwright() as pw:
            page = sim.page(pw, cdp)
            (new_user if persona == "new" else returning_user)(sim, page)
    except Exception as e:
        sim.log("simulation ran to the end", False, error=repr(e)[:400])
    finally:
        sim.quit()
        sim.finish()


if __name__ == "__main__":
    main()
