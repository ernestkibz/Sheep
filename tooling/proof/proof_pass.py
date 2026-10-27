"""The proof pass, run before every release: a fresh isolated home on the 3B (ports 8141/8142).

    python tooling/proof/proof_pass.py [output folder]     (default: scratchpad/proof)

Needs the charger in: on battery the background checks wait, by design. 1) the app checks, row by row;
2) Ernest's 7-turn legend-of-seeker replay; 3) the video's prompts, twice, each on a fresh home, timed."""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SP = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "scratchpad" / "proof"
SP.mkdir(parents=True, exist_ok=True)
MODEL = str(Path.home() / ".mlmi" / "models" / "Qwen--Qwen2.5-3B-Instruct.ssf")
B = "http://127.0.0.1:8141"


def say(*a):
    print(*a, flush=True)


def get(path, t=20):
    with urllib.request.urlopen(B + path, timeout=t) as r:
        return json.load(r)


def post(path, body, t=60):
    req = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                     "Origin": B})
    with urllib.request.urlopen(req, timeout=t) as r:
        return json.load(r)


def fresh(tag):
    home, ws = SP / f"b10_{tag}_home", SP / f"b10_{tag}_ws"
    shutil.rmtree(home, ignore_errors=True)
    shutil.rmtree(ws, ignore_errors=True)
    home.mkdir(); ws.mkdir()
    (home / "config.json").write_text(json.dumps({"model": MODEL, "device": "cpu", "workspace": str(ws)}), encoding="utf-8")
    log = open(SP / f"b10_{tag}_server.log", "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-m", "mlmi.backend.server", "--port", "8141", "--engine-port", "8142"],
                         cwd=REPO, env=dict(os.environ, MLMI_HOME=str(home)), stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            get("/health", 3)
            break
        except Exception:
            time.sleep(2)
    t0 = time.time()
    while time.time() - t0 < 900:
        try:
            if get("/api/orchestrator").get("lamp") == "green":
                break
        except Exception:
            pass
        time.sleep(3)
    say(f"[{tag}] green after {round(time.time() - t0)} s")
    return p, home


def agent(messages):
    req = urllib.request.Request(B + "/api/agent", json.dumps({"messages": messages, "auto_confirm": "yes",
                                                               "cache": False}).encode(),
                                 {"Content-Type": "application/json", "Origin": B})
    t0, tools, text, files, checked = time.time(), [], "", [], ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            t = ev.get("type")
            if t == "tool_result":
                tools.append(("ok " if ev.get("ok") else "FAIL ") + str(ev.get("name")))
            elif t == "text" and not ev.get("partial"):
                text = ev.get("text") or ""
            elif t == "file_changed" and ev.get("path") not in files:
                files.append(ev["path"])
            elif t in ("checked", "check", "review"):
                checked = json.dumps(ev)[:160]
            elif t == "done":
                break
    messages.append({"role": "assistant", "content": text, **({"files": files} if files else {})})
    return {"s": round(time.time() - t0), "files": [Path(f).name for f in files], "tools": tools,
            "checked": checked, "reply": text[:160]}


def stop(p):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(3)


# 1: the replay, on a fresh home
p, home = fresh("replay")
say("REPLAY")
msgs = []
for i, ask in enumerate(["powerpoint about legend of seeker", "season 2", "your shoice", "search web for legend of seeker",
                         "want a powerpoint about legend of seeker", "i want a file", "pptx"], 1):
    msgs.append({"role": "user", "content": ask})
    r = agent(msgs)
    say(json.dumps({"turn": i, "ask": ask, **r}))
stop(p)

# 3: the video's prompts, twice, each on a fresh home
for run in (1, 2):
    p, home = fresh(f"video{run}")
    post("/api/settings", {"mode": 1})
    msgs, T = [], time.time()
    for ask, mode in (("powerpoint about the history of coffee", 1), ("pptx", 1),
                      ("make a presentation about the latest NVIDIA Nemotron models", 1),
                      ("add three slides to the deck", 2)):
        if mode == 2:
            post("/api/settings", {"mode": 2})
        msgs.append({"role": "user", "content": ask})
        r = agent(msgs)
        say(json.dumps({"video": run, "mode": mode, "ask": ask, **r}))
    say(f"VIDEO RUN {run}: {round(time.time() - T)} s in all")
    try:
        sent = get("/api/cloudlog")
        rows = sent if isinstance(sent, list) else sent.get("sent") or sent.get("rows") or []
        say(f"  sent to Nebius: {len(rows)} calls")
        (SP / f"b10_video{run}_cloudlog.json").write_text(json.dumps(sent, indent=1)[:200000], encoding="utf-8")
    except Exception as e:
        say("  cloudlog:", e)
    stop(p)
# 3: the app checks last: on battery they wait for the charger (by design), up to 6 hours
p, home = fresh("checks")
t0, last = time.time(), None
while time.time() - t0 < 21600:
    time.sleep(15)
    try:
        e = json.loads((home / "models.json").read_text(encoding="utf-8")).get(MODEL, {})
    except (OSError, ValueError):
        continue
    ov = e.get("orientation_validation") or {}
    now = (ov.get("state"), ov.get("done"))
    if now != last:
        say(f"  {round(time.time() - t0)} s: app checks {now[0]} {now[1]}")
        last = now
    if ov.get("state") in ("passed", "failed"):
        for r in ov.get("rows", []):
            say("  ", r.get("id"), r.get("ok"), "|", (r.get("why") or "")[:90], "| tries", r.get("tries", 1))
        say(f"APP CHECKS: {ov.get('passed')} of {ov.get('of')} ({ov.get('state')})")
        break
stop(p)
say("B10 DONE")
