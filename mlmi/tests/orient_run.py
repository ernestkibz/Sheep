"""Run the orientation tasks against a Sheep backend, the way the app talks to it,
and grade each by opening what was made.

    python mlmi/tests/orient_run.py --port 8090 --ws WORKSPACE --areas slides,words --per 100 --out res.jsonl
    python mlmi/tests/orient_run.py --report res.jsonl

Resumable: a task already in the results file is skipped. --every k runs every
k-th task (a quick sample of the same spread). Use a backend whose workspace is
a scratch folder: the tasks make and change files there.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import orient  # noqa: E402
from mlmi.backend.orchestrator import QUICK_CANARIES  # noqa: E402

KIND_EXT = {"slides": ".sdeck", "words": ".sdoc", "sheets": ".ssheet", "studio": ".sstudio"}


def ready(base, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            d = json.load(urllib.request.urlopen(base + "/api/dashboard", timeout=10))
            orch = d.get("orch") or {}
            lamp = d.get("lamp") or orch.get("lamp")
            canaries = d.get("canaries") or orch.get("canaries") or []
            engine_up = bool((d.get("engine") or {}).get("up"))
            checks = d.get("checks") or []
            if not checks:
                # Compatibility with earlier test backends that did not expose
                # the readiness rows yet.
                checked = len(canaries) >= 8 and all(row.get("ok") for row in canaries)
                if lamp in ("green", "red") or (engine_up and (checked or lamp == "off")):
                    return {**d, "lamp": lamp or "amber"}
                time.sleep(2)
                continue
            startup = next((row for row in checks if row.get("id") == "startup"), {})
            required = {row[0] for row in QUICK_CANARIES}
            quick = [row for row in canaries if row.get("name") in required]
            checked = required.issubset({row.get("name") for row in quick}) and all(row.get("ok") for row in quick)
            # The model is usable after its compact activation exam. Full
            # canaries and orientation continue in the background.
            if (engine_up and startup.get("state") == "passed" and checked) or lamp == "off":
                return {**d, "lamp": lamp or "amber", "checks": checks, "quick_canaries": quick}
        except Exception:
            pass
        time.sleep(2)
    return {}


def turn(base, messages, app, path, timeout=1800):
    # cache off: orientation grades the model, not the engine's reply cache. With
    # it on, repeated wording came back in 0 s and counted as the model's work.
    body = {"messages": messages, "app": app, "path": path, "auto_confirm": "yes", "cache": False}
    req = urllib.request.Request(base + "/api/agent", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    final, files, steps, calls = "", [], [], {}
    connection_failures = 0
    busy_since = time.monotonic()
    for attempt in range(360):                     # 409: the backend is still finishing a run
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            break
        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise
            if time.monotonic() - busy_since >= timeout:
                raise TimeoutError(f"the backend stayed busy for {timeout} s") from e
            time.sleep(5)
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, ConnectionAbortedError, OSError):
            # No response headers means the agent could not have started a tool
            # action, so retrying this request cannot duplicate a file change.
            connection_failures += 1
            if connection_failures >= 6:
                raise
            time.sleep(min(connection_failures * 2, 10))
    else:
        raise TimeoutError(f"the backend stayed busy for {timeout} s")
    with r:
        buf = b""
        for chunk in iter(lambda: r.read(512), b""):
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                if not ev.startswith(b"data: "):
                    continue
                e = json.loads(ev[6:])
                if e["type"] == "tool_call":
                    calls[e["id"]] = e["name"]
                elif e["type"] == "tool_result":
                    steps.append([calls.get(e["id"], e.get("name")), bool(e.get("ok")), str(e.get("text", ""))[:200]])
                elif e["type"] == "text" and not e.get("partial"):
                    final = e.get("text", "")
                elif e["type"] == "file_changed" and e["path"] not in files:
                    files.append(e["path"])
                elif e["type"] == "error":
                    final = "ERROR: " + e.get("text", "")
    return final, files, steps


def gpu_temp():
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def cool(hot=96, cold=92, most=90):
    """Let the card cool between tasks: the laptop shut down overnight after hours at
    about 90 C with the GPU and CPU both at 100% (2026-10-07).

    Measured while a pass ran (2026-10-07): the card sits at 85-93 C and 15-21 W
    between tasks and touches 88-94 C at 35 W while working, already clock-limited
    by itself to 210/712 MHz -- which is the hardware's own protection, well under
    this chip's 60 W. At hot=88 almost every task waited the full five minutes for a
    temperature the card never leaves, and 104 tasks would have taken five hours.
    So the bar is where the card is genuinely hot, the heat is printed beside each
    result, and the protection that matters is running one heavy job at a time."""
    t = gpu_temp()
    if t is None or t < hot:
        return
    print(f"  GPU at {t} C: pausing until {cold} C (at most {most // 60} min)", flush=True)
    t0 = time.time()
    while (t := gpu_temp()) is not None and t > cold and time.time() - t0 < most:
        time.sleep(15)


def run_task(base, root, t):
    t.setup(root)
    messages, texts, files, steps = [], [], [], []
    t0 = time.time()
    for spec in t.turns:
        text, app = spec[0], spec[1]
        path = spec[2] if len(spec) > 2 else ""
        if app.endswith("@"):                         # the app, with the file the chat made open in it
            app = app[:-1]
            path = next((f for f in reversed(files) if f.endswith(KIND_EXT.get(app, "#"))), "")
        messages.append({"role": "user", "content": text})
        final, fs, st = turn(base, messages, app, path)
        messages.append({"role": "assistant", "content": final, **({"files": fs} if fs else {})})
        texts.append(final)
        files += [f for f in fs if f not in files]
        steps += st
    res = {"texts": texts, "files": files, "steps": steps}
    ok, why = t.grade(root, res)
    return {"id": t.id, "area": t.area, "ok": bool(ok), "why": why, "s": round(time.time() - t0, 1),
            "turns": [x[0] for x in t.turns], "files": files, "steps": [s[0] for s in steps],
            "failed_steps": [s for s in steps if not s[1]][:3], "reply": texts[-1][:400] if texts else ""}


def is_infrastructure_failure(row):
    why = str(row.get("why") or "")
    return not row.get("ok") and (why.startswith("run:") or why.startswith("infra:"))


def latest_results(rows):
    latest = {}
    for row in rows:
        latest[row["id"]] = row
    return list(latest.values())


def completed_ids(path):
    if not Path(path).exists():
        return set()
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"] for row in latest_results(rows) if not is_infrastructure_failure(row)}


def task_failure_ids(path):
    if not Path(path).exists():
        return set()
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"] for row in latest_results(rows)
            if not row.get("ok") and not is_infrastructure_failure(row)}


def report(path, total=800):
    attempts = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = latest_results(attempts)
    pending = [row for row in rows if is_infrastructure_failure(row)]
    infrastructure = [row for row in attempts if is_infrastructure_failure(row)]
    task_failures = [row for row in rows if not row.get("ok") and not is_infrastructure_failure(row)]
    print(f"orientation total {len(rows)}/{total}; passed {sum(bool(row.get('ok')) for row in rows)}; "
          f"task failures {len(task_failures)}; infrastructure attempts {len(infrastructure)}; "
          f"pending infrastructure retries {len(pending)}")
    for area, a in orient.summary(rows).items():
        print(f"{area:9} {a['passed']:3}/{a['of']:<3} passed  avg {a['s_avg']:5.1f} s")
        for why, n in sorted(a["fails"].items(), key=lambda x: -x[1])[:8]:
            print(f"            {n:3} x {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--ws", default="")
    ap.add_argument("--areas", default=",".join(orient.AREAS))
    ap.add_argument("--per", type=int, default=100)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--total", type=int, default=0,
                    help="expected sample size for --report (run mode derives it from --areas/--per/--every)")
    ap.add_argument("--out", default="orient_results.jsonl")
    ap.add_argument("--report", default="")
    ap.add_argument("--retry", default="", help="comma-separated task IDs to rerun even if already graded")
    ap.add_argument("--retry-failures", action="store_true", help="rerun tasks whose latest result is a task failure")
    a = ap.parse_args()
    if a.report:
        return report(a.report, a.total or 800)
    base = f"http://127.0.0.1:{a.port}"
    d = ready(base)
    print("backend", d.get("lamp"), "device", (d.get("plan") or {}).get("device"), flush=True)
    root = Path(a.ws)
    done = completed_ids(a.out)
    retry = {task_id.strip() for task_id in a.retry.split(",") if task_id.strip()}
    if a.retry_failures:
        retry |= task_failure_ids(a.out)
    selected = [t for area in a.areas.split(",") for t in orient.tasks(area, a.per)[::a.every]]
    for t in selected:
            if t.id in done and t.id not in retry:
                continue
            cool()
            try:
                r = run_task(base, root, t)
            except Exception as e:
                r = {"id": t.id, "area": t.area, "ok": False, "why": f"run: {type(e).__name__}: {e}", "s": 0,
                     "turns": [x[0] for x in t.turns]}
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['id']} {r['s']:5.0f}s {r['turns'][0][:60]!r} {'' if r['ok'] else r['why']}",
                  flush=True)
    report(a.out, len(selected))


if __name__ == "__main__":
    main()
