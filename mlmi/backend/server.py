"""The Sheep backend: engine, modes, agent, tools, workspace -- one local server.

    python -m mlmi.backend.server [--port 8080] [--model C:/models/x.ssf] [--workspace DIR]

It starts sml-server (the SHEEP engine) on the chosen model and restarts it when
another model is plugged in; the desktop app only has to start this. Everything
listens on 127.0.0.1.

  GET  /                    the desktop UI (also works in a browser)
  GET  /api/state           mode, model and its grade, workspace, engine stats
  POST /api/settings        {mode, permission, workspace, model, device, media_device}
  GET  /api/hardware        adapters found, the DirectML check, text and media modes (greyed with reasons)
  GET  /api/levels?src=     a checkpoint identified, and verified quantized/exact build cards for this machine
  GET  /api/checker         the Checker: its rules, the plugged-in judge model, the last exam
  GET  /api/weblog          ABIS: web access on/off, how search runs, and every query and page sent
  GET  /api/preview?path=   a few lines of a file, for its Work Canvas block
  POST /api/run_code        {lang, code} a Work Canvas code block run sandboxed: {output, code, ms}
  POST /api/checker         {path} exam a judge model and install it if it passes; {remove} back to the rules
  GET  /api/files           the workspace tree
  GET  /api/media_install   the current on-demand picture/video/speech model install status
  GET  /api/raw?path=       a file's bytes (previews)
  GET  /api/doc?path=       an editor's model of a .docx/.xlsx/.pptx
  POST /api/op              {tool, args}: an editor action -- the same tools the agent uses
  POST /api/agent           {messages, app, path}: run the agent, events as SSE
  POST /api/confirm         {id, allow}: answer a confirmation the agent is waiting on
  POST /api/stop            stop the running agent
  POST /api/undo, /api/redo
  GET  /api/history?path=   the journal
  GET  /api/cloudlog        exactly what Mode 1 sent to Nebius
  GET  /api/events          SSE: files changed and media-install progress
  GET  /api/dashboard       KPIs (red/amber/green), the device, the plan, the mode's figures
  GET  /api/orchestrator    the lamp, the plan and why, canaries, the Egypt protocol, the log
  POST /api/orchestrator    {action: "replan" | "canaries" | "cancel" | "recheck", check} (always on)
  GET  /api/registry        every model run on this machine; ?path= one model's history
  GET  /api/quantize        the build in progress; POST {src, name, budget_mb, licence} starts one
"""
import argparse
import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import agent
from . import checker as checker_mod
from . import hardware
from . import levels
from . import tools as T
from .modes import Cloud, Local
from .orchestrator import Orchestrator, kpis, load_registry, probe, save_registry
from .orchestrator import cuda_env as _cuda_env
from .quantizer import Job
from .workspace import ToolError, Workspace

_JSON_SAVE_LOCKS_GUARD = threading.Lock()
_JSON_SAVE_LOCKS = {}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                       # the sheep repo
UI = HERE.parent / "desktop" / "ui"
# MLMI_ENGINE_EXE: another engine build (a second instance testing a new engine
# while the app keeps the installed one, which Windows locks while it runs)
ENGINE_EXE = (Path(os.environ["MLMI_ENGINE_EXE"]) if os.environ.get("MLMI_ENGINE_EXE")
              else ROOT / "sml" / ("sml-server.exe" if os.name == "nt" else "sml-server"))
from .paths import HOME  # noqa: E402
CONFIG = HOME / "config.json"
CHATS = HOME / "chats"          # one JSON file per conversation, like a chat app's history
MODELS = HOME / "models.json"
UNCHECKED_TIER = 2     # a model not yet through the checker may edit office files, not delete or render


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def chosen_device(st):
    """Where the person chose text to run: their choice, even while the GPU is lent to a job."""
    return getattr(st.orch, "lent_from", None) or st.cfg.get("device", "auto")


def plan_settings(cfg):
    """The pool, VRAM and GPU+CPU settings a plan takes, from the saved config."""
    return {"pool_mode": cfg.get("pool_mode", "smart"), "gap": float(cfg.get("pool_gap", 4) or 4),
            "exclude": cfg.get("gpus_off") or [], "reserve_gb": float(cfg.get("vram_reserve_gb", 0) or 0),
            "cpu_offload": bool(cfg.get("cpu_offload", True))}


def mode_fits(st, dev, modes):
    """For the loaded model, how each "text and code run on" choice would carry it, ranked
    by SPEED: green for the fastest here (or within 15% of it), amber for one that works but
    is slower, red for one that cannot hold it. Speeds are the model's measured ones from
    its speed trial on this machine when there is one, else the planner's estimate.
    (Ernest, 2026-10-14: the Radeon alone showed green and multi amber, though the Radeon
    was the slowest: the lamps said only "fits on a GPU", not "best".)"""
    facts = getattr(st.orch, "facts", None)
    if not facts or not facts.get("n_layer"):
        return {}
    from .orchestrator import plan as _plan, hardware_sig
    entry = load_json(MODELS, {}).get(st.model_path) or {}
    measured = ((entry.get("speed_trials") or {}).get(hardware_sig(dev)) or {}).get("tok_s") or {}
    labels = {m["id"]: m.get("label", m["id"]) for m in modes}
    rows = {}
    for m in modes:
        if not m.get("ok"):
            continue
        try:
            p = _plan(dev, facts, m["id"], **plan_settings(st.cfg))
        except Exception:
            continue
        colour, words = levels.fit(p, facts, dev, m["id"])
        speed = measured.get(p.get("device"))
        rows[m["id"]] = {"fit": colour, "words": words, "device": p.get("device"),
                         "est_tok_s": round(speed, 1) if speed else p.get("est_tok_s"),
                         "measured": bool(speed), "shared_gpu": bool(p.get("device") in ("amd", "multi") and (
                             (dev.get("amd") or {}).get("integrated")))}
    usable = {k: r for k, r in rows.items() if r["fit"] != "red" and r.get("est_tok_s")}
    if not usable:
        return rows
    best_id = max(usable, key=lambda k: usable[k]["est_tok_s"])
    best = usable[best_id]["est_tok_s"]
    for k, r in usable.items():
        how = "measured" if r["measured"] else "estimated"
        if r["est_tok_s"] >= 0.85 * best:
            r["fit"] = "green"
            r["words"] = f"the fastest here: about {r['est_tok_s']} tokens a second ({how})"
        else:
            r["fit"] = "amber"
            r["words"] = (f"works, but slower here: about {r['est_tok_s']} tokens a second, against {best} on "
                          f"{labels.get(best_id, best_id)} ({how})")
            if r["shared_gpu"]:
                r["words"] += "; this GPU shares the processor's memory, so it adds room, not speed"
    return rows


def save_json(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Requests for one chat can save concurrently. A shared ``.tmp`` filename
    # lets one request replace a file another request is still writing (or
    # replacing), which can fail on Windows and lose the newest conversation.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            json.dump(v, stream, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        key = str(p.resolve())
        with _JSON_SAVE_LOCKS_GUARD:
            replace_lock = _JSON_SAVE_LOCKS.setdefault(key, threading.Lock())
        # Serialize only the final replace for a destination; JSON encoding
        # remains concurrent, while Windows never races two replacements of
        # the same open target.
        with replace_lock:
            for attempt in range(8):
                try:
                    os.replace(tmp, p)
                    break
                except PermissionError:
                    # Windows scanners can briefly hold a just-written file.
                    # Retry only this atomic persistence boundary, then surface
                    # a persistent permission error to the caller.
                    if attempt == 7:
                        raise
                    time.sleep(0.01 * (attempt + 1))
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class Engine:
    """sml-server as a child process, restarted when the model changes."""

    def __init__(self, port=8099, ctx=8192, slots=4, threads=None):
        self.port, self.ctx, self.slots = port, ctx, slots
        self.threads = threads or max(1, min(6, (os.cpu_count() or 4) // 2))
        self.device, self.gpu_layers = "cpu", 0       # set by the orchestrator's plan
        self.ext_cmd = ""                             # the DirectML worker for device "amd" (or "multi")
        self.mid_layers = 0                           # device "multi": the AMD/Intel card's layers
        self.compat = None                            # {"src", "trust_remote_code", "kind"}: the maker's code
        self.tune_env = {}                            # this model's tuned GPU settings (the orchestrator's engine tune)
        self.proc = None
        self.model = ""
        self.log = []
        self.lock = threading.Lock()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                return r.status == 200
        except OSError:
            return False

    def alive(self):
        """The engine process is up. Not healthy(): a busy engine answers nothing
        until its current request is done, and busy is not down."""
        return self.proc is not None and self.proc.poll() is None

    def start(self, model):
        with self.lock:
            self.stop()
            self.model = model
            if not model:
                return False
            if model.lower().endswith(".compat.json") or self.compat:
                return self._start_compat(model)
            if not ENGINE_EXE.exists():
                self.log.append(f"engine not built: {ENGINE_EXE}")
                return False
            self.reap()
            self.proc = subprocess.Popen(
                [str(ENGINE_EXE), model, "--port", str(self.port), "--ctx", str(self.ctx),
                 "--slots", str(self.slots), "--threads", str(self.threads), "--device", self.device]
                + (["--gpu-layers", str(self.gpu_layers)] if self.device in ("hybrid", "multi") and self.gpu_layers > 0 else [])
                + (["--mid-layers", str(self.mid_layers)] if self.device in ("multi", "amd") and self.mid_layers > 0 else [])
                + (["--ext-cmd", self.ext_cmd] if self.ext_cmd else []),
                # UTF-8, not the Windows code page: Youtu-LLM's vocabulary printed byte 0x81, the reader
                # thread died on cp1252, and the full pipe stalled the engine's start (2026-10-20)
                cwd=str(ROOT / "sml"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=({**_cuda_env(), **self.tune_env} if self.device in ("gpu", "hybrid", "multi") else None))
            threading.Thread(target=self._drain, args=(self.proc,), daemon=True).start()
            # a DirectML worker loads the model before the engine answers: minutes, not seconds
            for _ in range(2400 if self.ext_cmd else 240):
                if self.healthy():
                    return True
                if self.proc.poll() is not None:
                    return False
                time.sleep(0.25)
            return False

    def _start_compat(self, model):
        """Compatibility mode (compat_server.py): the maker's own code on the original
        weights, under a PyTorch runtime, behind the engine's own HTTP API."""
        from . import gpu_runtime
        try:
            c = self.compat or json.loads(Path(model).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.log.append(f"compatibility mode: cannot read {model}: {exc}")
            return False
        kind = c.get("kind") or ""
        if kind and gpu_runtime.ready(kind):
            py = gpu_runtime.py(kind)
        else:                              # the runtime whose Transformers knows this model type
            try:
                mt = json.loads((Path(c["src"]) / "config.json").read_text(encoding="utf-8")).get("model_type", "")
            except (OSError, ValueError, KeyError):
                mt = ""
            py = gpu_runtime.torch_python_for(mt, install_cpu=False)[0]
        if not py:
            self.log.append("compatibility mode needs the PyTorch runtime (Settings: install GPU support, or "
                            "python -m mlmi.backend.gpu_runtime install cpu)")
            return False
        self.reap()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        if kind == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            from .orchestrator import cuda_env
            env = cuda_env(env)
        self.proc = subprocess.Popen(
            [str(py), "-m", "mlmi.backend.compat_server", str(c["src"]), "--port", str(self.port)]
            + (["--trust-remote-code"] if c.get("trust_remote_code") else []),
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._drain, args=(self.proc,), daemon=True).start()
        for _ in range(2400):                     # loading the originals: up to 10 minutes
            if self.healthy():
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.25)
        return False

    def reap(self):
        """Stop engines a dead backend left on this port. One was found holding
        3.8 GB beside the live engine (2026-09-30): the machine had 0.4 GB free
        and the orchestrator planned the model into a 2K context."""
        try:
            import psutil
        except ImportError:
            return
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                cmd = p.info["cmdline"] or []
                ours = (p.info["name"] or "").lower().startswith("sml-server") or any(
                    "mlmi.backend.compat_server" in str(c) for c in cmd)      # a compatibility-mode server too
                if ours and "--port" in cmd \
                        and cmd[cmd.index("--port") + 1] == str(self.port) and (not self.proc or p.pid != self.proc.pid):
                    p.kill()
                    self.log.append(f"stopped a stale engine (pid {p.pid}) left on port {self.port}")
            except (psutil.Error, IndexError, ValueError):
                continue

    def _drain(self, proc):
        for line in proc.stdout:
            self.log.append(line.rstrip())
            self.log = self.log[-300:]

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def run_code(lang, code, timeout=15):
    """A Work Canvas code block, run in its own process in an empty folder: Python
    in isolated mode (-I), JavaScript with node. Stopped after `timeout` seconds."""
    import shutil
    import tempfile
    exe = {"python": [sys.executable, "-I", "-"], "javascript": [shutil.which("node") or "", "-"]}.get(lang)
    if not exe or not exe[0]:
        return {"output": f"{lang} cannot run here", "code": -1, "ms": 0}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sheep-run-") as d:
        try:
            r = subprocess.run(exe, input=code, capture_output=True, text=True, cwd=d, timeout=timeout,
                               encoding="utf-8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            out, rc = (r.stdout or "") + (r.stderr or ""), r.returncode
        except subprocess.TimeoutExpired:
            out, rc = f"stopped after {timeout} s (an endless loop?)", -1
    return {"output": out[-6000:], "code": rc, "ms": round((time.perf_counter() - t0) * 1000)}


# keys entered in Settings › Accounts: judges and most people cannot set environment variables.
# A key set there wins over the environment; the interface only ever learns whether one is set.
KEY_ENV = {"nebius_key": "NEBIUS_API_KEY", "nebius_base": "NEBIUS_BASE_URL", "nebius_model": "NEBIUS_MODEL",
           "tavily_key": "TAVILY_API_KEY"}


def apply_keys(cfg):
    for k, env in KEY_ENV.items():
        v = str((cfg.get("keys") or {}).get(k) or "").strip()
        if v:
            os.environ[env] = v


def keys_view():
    return {"nebius": bool(os.environ.get("NEBIUS_API_KEY")), "tavily": bool(os.environ.get("TAVILY_API_KEY")),
            "nebius_base": os.environ.get("NEBIUS_BASE_URL", ""), "nebius_model": os.environ.get("NEBIUS_MODEL", "")}


# Settings › Basics: "How Sheep runs" as three plain choices. Each is a bundle of the Advanced
# settings below it; the choice shown is worked out from those settings, so a hand change shows "custom".
PRESETS = {
    "fast": {"pool_mode": "always", "pool_gap": 8.0, "vram_reserve_gb": 0.0, "cpu_offload": True, "thermal_guard": True},
    "balanced": {"pool_mode": "smart", "pool_gap": 4.0, "vram_reserve_gb": 0.0, "cpu_offload": True, "thermal_guard": True},
    "quiet": {"pool_mode": "off", "pool_gap": 4.0, "vram_reserve_gb": 2.0, "cpu_offload": True, "thermal_guard": True},
}
PRESET_DEFAULTS = PRESETS["balanced"]


def preset_of(cfg):
    now = {k: cfg.get(k, d) for k, d in PRESET_DEFAULTS.items()}
    for name, want in PRESETS.items():
        if all((float(now[k] or 0) == float(v)) if isinstance(v, float) else now[k] == v for k, v in want.items()):
            return name
    return "custom"


class State:
    def __init__(self, a):
        HOME.mkdir(parents=True, exist_ok=True)
        cfg = load_json(CONFIG, {})
        self.cfg = cfg
        apply_keys(cfg)
        self.mode = int(cfg.get("mode", 2))
        self.permission = cfg.get("permission", "auto")
        self.ws = Workspace(a.workspace or cfg.get("workspace") or (Path.home() / "Documents" / "Sheep"))
        self.engine = Engine(port=a.engine_port, ctx=a.ctx, slots=a.slots)
        self.local = Local(self.engine.url)
        self.cloud = Cloud()
        self.session = agent.Session()
        self.busy = threading.Lock()
        self.warming, self.warmed = set(), {}
        self.subscribers = []
        # The engine answers one request at a time, so asking it for /stats while
        # it reads a prompt waits minutes. The last figures it gave are kept here
        # and the interface reads those; it never waits on a busy engine.
        self.stats = {}
        threading.Thread(target=self._poll_stats, daemon=True).start()
        self.ws.listeners.append(self._changed)
        self.model_path = a.model or cfg.get("model", "")
        self.orch = Orchestrator(self)
        self.local.retry_device = self.orch.retry_after_device_error
        self.quant = Job()
        self.checker = checker_mod.Checker(self.cfg, Engine, Local, self.save)
        from .tools import web as _web
        _web.STATE["on"] = bool(cfg.get("web", True))
        from .tools import media as _media
        _media.CONFIG = self.cfg                  # media device and models chosen in Settings
        _media.LEND = (lambda what, gb, cancel_event=None: self.orch.lend_gpu(
                           what, gb, wait=True, priority=10, cancel_event=cancel_event),
                       self.orch.give_back_gpu)   # interactive media outranks background training/builds
        from .adapters.install import set_publisher
        set_publisher(self._publish)
        self.quant.lend = (lambda: self.orch.lend_gpu("the Sheep Quantizer", 1.5, wait=True, priority=5),
                           self.orch.give_back_gpu)
        hardware.start()

    def save(self):
        save_json(CONFIG, {"mode": self.mode, "permission": self.permission, "workspace": str(self.ws.root),
                           "model": self.model_path, "orchestrator": self.orch.on,
                           "theme": self.cfg.get("theme", "dark"),
                           # while the GPU is lent to a media job the chat runs on the CPU; the choice saved is the person's
                           "device": getattr(self.orch, "lent_from", None) or self.cfg.get("device", "auto"),
                           "media_device": self.cfg.get("media_device", "auto"),
                           "checker_training_mode": self.cfg.get("checker_training_mode", "auto"),
                           "checker": self.cfg.get("checker"), "web": self.cfg.get("web", True),
                           "thermal_guard": self.cfg.get("thermal_guard", True),
                           "pool_mode": self.cfg.get("pool_mode", "smart"), "pool_gap": self.cfg.get("pool_gap", 4),
                           "gpus_off": self.cfg.get("gpus_off") or [], "vram_reserve_gb": self.cfg.get("vram_reserve_gb", 0),
                           "cpu_offload": self.cfg.get("cpu_offload", True),
                           "media_models": self.cfg.get("media_models") or {},
                           "keys": self.cfg.get("keys") or {},
                           "disabled_media_models": self.cfg.get("disabled_media_models") or []})

    def model_card(self):
        grades = load_json(MODELS, {})
        g = grades.get(self.model_path, {})
        name = g.get("display") or (Path(self.model_path).name if self.model_path else "")
        return {"path": self.model_path, "name": name, "tier": int(g.get("tier", UNCHECKED_TIER)),
                "checked": "tier" in g, "grade": g.get("grade", {}), "licence": g.get("licence", ""),
                "duties": g.get("duties", [])}

    def _poll_stats(self):
        while True:
            s = self.local.stats()
            if s:
                self.stats = s
            time.sleep(2)

    def _changed(self, rel):
        self._publish({"type": "file_changed", "path": rel})

    def _publish(self, event):
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def snapshot(self):
        return {"mode": self.mode, "permission": self.permission, "workspace": str(self.ws.root),
                "model": self.model_card(), "engine": {"up": self.engine.alive(),
                                                       "url": self.engine.url, "stats": self.stats},
                "cloud": {"ready": self.cloud.ready(), "model": self.cloud.model, "sent": len(self.cloud.sent),
                          "provider": self.cloud.provider, "keys": keys_view()},
                "busy": self.busy.locked(), "warming": sorted(self.warming), "warmed": self.warmed,
                "orch": {"on": self.orch.on, "lamp": self.orch.lamp, "stage": self.orch.stage,
                         "holding": self.orch.holding(), "ready": self.orch.ready(), "progress": self.orch.progress,
                         "loading": self.orch.loading.is_set(), "checks": self.orch.readiness_checks(),
                         "heat": self.orch.thermal.snapshot() if getattr(self.orch, "thermal", None) else None},
                "theme": self.cfg.get("theme", "dark"), "quant": self.quant.state,
                "pool_mode": self.cfg.get("pool_mode", "smart"), "pool_gap": self.cfg.get("pool_gap", 4),
                "gpus_off": self.cfg.get("gpus_off") or [], "vram_reserve_gb": self.cfg.get("vram_reserve_gb", 0),
                "cpu_offload": self.cfg.get("cpu_offload", True), "preset": preset_of(self.cfg),
                "web": bool(self.cfg.get("web", True))}


def activity_view(st):
    """Settings › Activity: every check's rows (an app check with what was asked, done and made) and the
    automatic repairs this model had."""
    if not st.model_path:
        return {}
    e = load_registry().get(st.model_path) or {}
    orient = dict(e.get("orientation_validation") or {})
    if orient.get("rows") and orient.get("signature"):
        latest = {}
        try:
            path = HOME / "orientation" / Path(st.model_path).stem / str(orient["signature"])[:16] / "transcript.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                latest[j.get("id")] = j
        except OSError:
            pass
        orient["rows"] = [{**r, **{k: latest[r.get("id")][k] for k in ("steps", "replies", "files")
                                   if k in latest.get(r.get("id"), {})}} for r in orient["rows"]]
    return {"quick": e.get("quick_validation"), "deep": e.get("deep_validation"), "orientation": orient or None,
            "speed": e.get("speed_trials"), "fix": e.get("fix"),
            "repairs": [h for h in e.get("history") or [] if str(h.get("event", "")).startswith("egypt")][-20:]}


def setup_checklist(st, kind="", cloud_chat=False):
    """What Sheep needs for each kind of work, what is here and what is missing,
    for a person who asks for something Sheep is not set up for (Ernest, 2026-10-13).
    kind narrows it to one need ("video", "picture", "speech", "voice", "vision")."""
    import shutil as _sh
    from . import adapters as _ad, gpu_runtime, hardware
    cat = {m["id"]: m for m in _ad.catalogue(st.cfg)}
    ab = st.orch.abilities() if getattr(st, "orch", None) else {}
    nv = gpu_runtime.nvidia()
    # (kind, label, ready, how to get it, what it is for, optional, the button's action)
    rows = [
        ("text", "Chat (text) model", bool(st.model_path) or bool(cloud_chat), "Add a model from the model menu at the top",
         "Nebius can write video prompts; video generation still runs locally." if cloud_chat
         else "Everything: Sheep needs one model to chat and write.",
         False, {"do": "slot", "slot": "chat", "label": "Add a model"}),
        ("picture", "Picture model (Stable Diffusion 1.5)", bool(cat.get("sd15-lcm", {}).get("ready")),
         "Settings › Models › pictures", "Only for making pictures. Skip it if you will not ask for any.", True,
         {"do": "media", "id": "sd15-lcm", "label": "Download (about 2 GB)"}),
        ("video", "Video model (AnimateDiff)", bool(cat.get("animatediff-lightning", {}).get("ready")),
         "Settings › Models › video", "Only for making short videos. Skip it if you will not ask for any.", True,
         {"do": "media", "id": "animatediff-lightning", "label": "Download (about 1 GB, plus the picture model)"}),
        ("video", "ffmpeg (writes the video file)", bool(_sh.which("ffmpeg")), "install ffmpeg and restart Sheep",
         "Only for saving videos as files. Skip it if you will not make videos.", True,
         {"do": "link", "url": "https://ffmpeg.org/download.html", "label": "Get ffmpeg"}),
        ("speech", "Speech to text (Whisper small)", bool(cat.get("whisper-small", {}).get("ready")),
         "Settings › Models › sound", "Only for talking to Sheep with your voice. Typing works without it.", True,
         {"do": "media", "id": "whisper-small", "label": "Download (about 1 GB)"}),
        ("voice", "Windows voices (read aloud)", bool(cat.get("windows-voices", {}).get("ready")),
         "Windows Settings › Time & language › Speech", "Only for Sheep reading answers aloud.", True,
         {"do": "link", "url": "ms-settings:speech", "label": "Open Windows speech settings"}),
        ("vision", "Picture-reading (vision) model", bool(ab.get("reads pictures")),
         "a vision model such as SmolVLM, added in Settings › Models",
         "Only for reading photos you attach. Text and documents work without it.", True,
         {"do": "slot", "slot": "chat", "label": "Add a vision model"}),
    ]
    # each GPU here needs its PyTorch runtime (NVIDIA: CUDA, AMD/Intel: DirectML)
    try:
        runtimes = [r for r in gpu_runtime.status_all() if r["needed"]]
    except Exception:
        runtimes = []
    for r in runtimes:
        ok = hardware.media_cuda() if r["kind"] == "cuda" else r["ready"]
        for k in ("picture", "video"):
            card = {"cuda": "NVIDIA graphics card", "directml": "AMD / Intel graphics card"}.get(r["kind"], r["label"])
            rows.append((k, f"{card} support for pictures and video", ok,
                         f"Settings › Advanced › This computer › GPU support · Download ({r['size']})",
                         f"Makes pictures and video much faster on the {card}. Without it they run on the processor, slowly.",
                         True, {"do": "runtime", "kind": r["kind"], "label": f"Download ({r['size']})"}))
    if nv and not runtimes:
        rows += [(k, "NVIDIA graphics card support for pictures and video", hardware.media_cuda(),
                  "Settings › Advanced › This computer › GPU support · Download (about 2.5 GB)",
                  "Makes pictures and video much faster on the NVIDIA card.", True,
                  {"do": "runtime", "kind": "cuda", "label": "Download (about 2.5 GB)"}) for k in ("picture", "video")]
    seen, out = set(), []
    for k, label, ok, fix, need, optional, action in rows:
        if (kind and k not in (kind, "text")) or label in seen:
            continue
        seen.add(label)
        out.append({"kind": k, "label": label, "ok": bool(ok), "fix": "" if ok else fix, "need": need,
                    "optional": optional, "action": None if ok else action,
                    # GPU support makes the work faster; without it the work still runs (slowly)
                    "speedup": bool(action and action.get("do") == "runtime")})
    return out


def make_handler(st):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (json.dumps(body) if not isinstance(body, str) else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n))
            except ValueError:
                return {}

        def _foreign(self):
            """A request a web page elsewhere made: any site can POST to 127.0.0.1, and a
            renamed domain can point there (DNS rebinding). Only this app's own page, the
            Electron frame and local tools (no Origin) get through."""
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in ("127.0.0.1", "localhost", "::1"):
                return True
            if (self.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
                return True
            origin = self.headers.get("Origin")
            if origin is None:
                return False
            o = urllib.parse.urlparse(origin)
            return not (o.scheme == "http" and o.hostname in ("127.0.0.1", "localhost", "::1")
                        and o.port == self.server.server_address[1])

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _sse(self, obj):
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
            self.wfile.flush()

        def do_GET(self):
            if self._foreign():
                self.close_connection = True
                return self._send(403, {"error": "requests from other sites are refused"})
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            try:
                if u.path == "/api/state":
                    prune_model_catalogue(st)
                    return self._send(200, st.snapshot())
                if u.path == "/api/files":
                    return self._send(200, {"root": str(st.ws.root), "files": st.ws.tree()})
                if u.path == "/api/raw":
                    p = st.ws.path(q.get("path", ""), must_exist=True)
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    return self._send(200, p.read_bytes(), ctype)
                if u.path == "/api/doc":
                    rel = q.get("path", "")
                    ext = Path(rel).suffix.lower()
                    mod = {".sdoc": T.words, ".ssheet": T.sheets, ".sdeck": T.slides, ".sstudio": T.studio}.get(ext)
                    if not mod:
                        return self._send(400, {"error": f"no editor for {ext}"
                                                + ("; import it first" if ext in (".docx", ".xlsx", ".pptx") else "")})
                    return self._send(200, mod.model(rel, st.ws))
                if u.path == "/api/herders":
                    return self._send(200, {"herders": list(T.herders.library().values())})
                if u.path == "/api/history":
                    return self._send(200, {"entries": st.ws.history(q.get("path") or None)})
                if u.path == "/api/cloudlog":
                    return self._send(200, {"sent": st.cloud.sent[-100:]})
                if u.path == "/api/tools":
                    card = st.model_card()
                    return self._send(200, {"tier": card["tier"], "tools": [
                        {"name": t.name, "app": t.app, "tier": t.tier, "desc": t.desc,
                         "allowed": t.tier <= card["tier"]} for t in T.TOOLS.values()]})
                if u.path == "/api/models":
                    prune_model_catalogue(st)
                    from . import adapters as _ad
                    chosen = _ad.defaults(st.cfg)
                    dev = probe()
                    try:
                        dml = hardware.directml_adapters()
                    except Exception:
                        dml = []
                    try:
                        cuda_ready = hardware.media_cuda()
                    except Exception:
                        cuda_ready = False
                    media = [{**m, "role": _ad.role(m), "default": chosen.get(_ad.role(m)) == m["id"],
                              "match": _ad.fit(m, dev, dml, cuda_ready)} for m in _ad.catalogue(st.cfg)]
                    from . import model_slots as _slots
                    return self._send(200, {"current": st.model_path, "grades": load_json(MODELS, {}),
                                            "found": find_models(st.model_path), "media": media,
                                            "slots": _slots.current(st)})
                if u.path == "/api/hf_login":
                    from . import hf_source
                    return self._send(200, {"user": hf_source.signed_in()})
                if u.path == "/api/model_fit":
                    try:
                        path = validate_ssf_path(q.get("path", ""))
                        from .orchestrator import model_facts, plan as device_plan
                        facts = model_facts(path)
                        planned = device_plan(probe(), facts, st.cfg.get("device", "auto"))
                    except Exception as exc:
                        return self._send(400, {"error": f"cannot check this Sheep .ssf model: {str(exc)[:220]}"})
                    target = planned.get("device", "cpu")
                    ram_fit = bool(planned.get("ram_fit"))
                    gpu = planned.get("gpu") or {}
                    if not ram_fit:
                        level, label = "red", "Poor match · exceeds available RAM"
                    elif target == "gpu" and gpu.get("fit") == "whole":
                        level, label = "green", "Good match · the model fits the NVIDIA GPU"
                    elif target == "multi":
                        level, label = "yellow", "Medium match · split across local devices"
                    elif target == "amd" and gpu.get("fit") == "whole":
                        level, label = "yellow", "Medium match · fits a DirectML GPU"
                    elif target == "hybrid":
                        level, label = "yellow", "Medium match · shared between GPU and CPU"
                    else:
                        level, label = "yellow", "Medium match · runs on CPU"
                    return self._send(200, {"path": path, "name": Path(path).name, "arch": facts.get("arch"),
                                            "match": {"level": level, "label": label,
                                                      "device": planned.get("device_name") or target.upper(),
                                                      "reason": "; ".join(planned.get("why") or [])[:400]},
                                            "plan": planned})
                if u.path == "/api/media_install":
                    from .adapters.install import status
                    return self._send(200, {"job": status()})
                if u.path == "/api/dashboard":
                    dev = probe()
                    o = st.orch
                    sent = st.cloud.sent
                    from .model_slots import current as current_model_slots
                    slots = current_model_slots(st)
                    preferences = {"device": st.cfg.get("device", "auto"),
                                   "media_device": st.cfg.get("media_device", "auto")}
                    return self._send(200, {
                        "kpis": kpis(o, dev, slots, preferences), "slots": slots,
                        "preferences": preferences, "device": dev, "plan": o.plan, "facts": o.facts,
                        "lamp": o.lamp, "stage": o.stage, "on": o.on, "egypt": o.egypt, "mode": st.mode,
                        "model": st.model_card(), "engine": {"up": st.engine.alive(), "stats": st.stats},
                        "checks": o.readiness_checks(), "loading": o.loading.is_set(),
                        "turns": o.turns, "canaries": o.canaries, "activity": activity_view(st),
                        "cloud": {"ready": st.cloud.ready(), "model": st.cloud.model, "sent": len(sent), "provider": st.cloud.provider,
                                  "chars": sum(x.get("chars", 0) for x in sent),
                                  "last": sent[-1]["t"] if sent else None}})
                if u.path == "/api/orchestrator":
                    from .story import story
                    o = st.orch
                    return self._send(200, {**o.status(), "device": probe(),
                                            "story": story(o.entry(), o.facts, o.plan) if st.model_path else []})
                if u.path == "/api/checker":
                    return self._send(200, st.checker.status())
                if u.path == "/api/gpu_runtime":
                    from . import gpu_runtime
                    return self._send(200, {"runtimes": gpu_runtime.status_all(), "devices": gpu_runtime.gpus(),
                                            **gpu_runtime.status("cuda")})
                if u.path == "/api/setup_check":
                    return self._send(200, {"items": setup_checklist(st)})
                if u.path == "/api/preview":
                    rel = q.get("path", "")
                    office = {".docx": "Microsoft Word file · opens in Sheep Words",
                              ".xlsx": "Microsoft Excel file · opens in Sheep Sheets",
                              ".pptx": "Microsoft PowerPoint file · opens in Sheep Slides"}
                    ext = Path(rel).suffix.lower()
                    if ext in office:
                        # read_file's answer here is an instruction to the model ("bring it in with
                        # import_file"), shown to people on their file cards (2026-10-13)
                        return self._send(200, {"text": office[ext]})
                    try:
                        text = T.run(st.ws, "read_file", {"path": rel}, by="user")
                        # a person reads the document, not its block numbers ("[0] title: Trip Plan")
                        text = re.sub(r"(?m)^\[\d+\] [\w ]+?(?: \d)?: ", "", str(text))
                    except Exception as e:
                        text = str(e)
                    return self._send(200, {"text": str(text)[:900]})
                if u.path == "/api/weblog":
                    from .tools import web as _web
                    return self._send(200, {"on": _web.STATE["on"],
                                            "via": "Tavily" if os.environ.get("TAVILY_API_KEY") else "Bing, then Wikipedia (no key needed)",
                                            "sent": _web.SENT[-200:]})
                if u.path == "/api/levels":
                    src = (q.get("src") or "").strip()
                    if not src:
                        return self._send(400, {"error": "give a folder or a Hugging Face id"})
                    original_of, note = "", ""
                    from .model_slots import HF_ID
                    from .levels import LEVELS

                    def convert_reply(fname, words, keep=None, orig=""):
                        # a GGUF, GPTQ/AWQ copy or Meta .pth is converted, then built (Parts C1, C6). An
                        # original (Meta .pth, F16 GGUF) is offered every level; a copy that is already
                        # quantized (keep) only its own, packed into SSF as it is (Ernest, 2026-10-29)
                        if keep:
                            words = (f"Already quantized ({keep['tag']}). Sheep packs it into SSF as it is, at its own "
                                     "width, and does not shrink it again." +
                                     (f" To choose a level, add its original, {orig}." if orig else
                                      " To choose a level, add the original model it was made from.") +
                                     (" Its file name does not say its level, so Sheep treats it as 4-bit: rename it with its "
                                      "level (for example model-Q8_0.gguf) and press Check again." if keep["tag"] == "GGUF quantized" else ""))
                        return self._send(200, {"gguf": fname, "convert": words, "info": {"src": src}, "as_is": keep,
                                                "levels": [{"id": x["id"], "name": x["name"], "what": x["what"]}
                                                           for x in LEVELS]})
                    from . import hf_source
                    if src.lower().endswith(".gguf") and Path(src).is_file():
                        return convert_reply(Path(src).name, "no original weights to build from, so Sheep converts "
                                                             "this GGUF first.", hf_source.as_is("gguf", Path(src).name))
                    if Path(src).is_dir():
                        k = hf_source.folder_kind(src)
                        if k in hf_source.UNPACK:
                            cfg = load_json(Path(src) / "config.json", {})
                            return convert_reply(Path(src).name, hf_source.convert_note({"how": "unpack", "kind": k}, Path(src).name, local=True),
                                                 hf_source.as_is(k, cfg=cfg))
                        if k not in ("original", "none", "no-config"):
                            return self._send(400, {"error": f"{Path(src).name} is {hf_source.WORDS.get(k, 'a converted copy')}, "
                                                             "which Sheep cannot read; use the original model it was made from"})
                    if not Path(src).is_dir() and HF_ID.match(src):
                        # a GGUF/GPTQ/MLX page: the original it was made from is read instead
                        try:
                            orig, note, _ = hf_source.resolve(src)
                        except LookupError as e:
                            fb = hf_source.fallback(src)
                            if fb:
                                return convert_reply(fb.get("file") or src, hf_source.convert_note(fb, src), fb.get("as_is"))
                            return self._send(400, {"error": str(e)})
                        except Exception as e:
                            return self._send(400, {"error": hf_source.hub_error(e, src)})
                        if orig != src:
                            # a copy: kept as it is when Sheep can read it, else its original is named
                            fb = hf_source.fallback(src)
                            if fb and fb.get("as_is"):
                                return convert_reply(fb.get("file") or src, "", fb["as_is"], orig)
                            if not fb:
                                return self._send(400, {"error": f"{note.split(';')[0]}, which Sheep cannot pack as it is. "
                                                                 f"Add its original, {orig}, to choose a level."})
                            original_of, src = src, orig
                    try:
                        info = levels.inspect(src)
                    except Exception as e:
                        from . import hf_source
                        return self._send(400, {"error": hf_source.hub_error(e, src) if not Path(src).is_dir() else
                                                f"Sheep could not read {Path(src).name} as a model: {str(e)[:160]}"})
                    info.update(original_of=original_of, note=note, src=src)
                    if info.get("kind") != "text":
                        return self._send(400, {"error": f"{src} is made for {info.get('kind')}, not for chat; "
                                                         "pick a text model here, or add it under Extras"})
                    # never start a download that cannot finish (the add dialog says the same)
                    from .model_slots import free_bytes
                    need = (0 if Path(src).is_dir() else 2 * info.get("n_weights", 0)) + 0.6 * info.get("n_weights", 0)
                    free = free_bytes()
                    if need and free < need:
                        return self._send(400, {"error": f"Not enough free space: this model needs about {need / 1e9:.1f} GB and "
                                                         f"{free / 1e9:.1f} GB is free. Free up {(need - free) / 1e9:.1f} GB, "
                                                         "then press Check again."})
                    from .orchestrator import plan as _plan
                    cards = (levels.cards(info, st.orch.freed(probe(0)), _plan, chosen_device(st), **plan_settings(st.cfg))
                             if info["builds"] else [])
                    return self._send(200, {"info": {k: v for k, v in info.items() if k != "tensors"}, "cards": cards})
                if u.path == "/api/video_tip":
                    from .tools import media as _media
                    return self._send(200, _media.video_tip((st.orch.plan or {}).get("device", "")))
                if u.path == "/api/hardware":
                    if q.get("refresh"):
                        hardware.adapters(refresh=True)
                        hardware.directml(refresh=True)
                    current_device = probe(0)
                    after_chat_yields = st.orch.freed(current_device)
                    rep = hardware.report(current_device, {"device": chosen_device(st),
                                                           "media_device": st.cfg.get("media_device", "auto"),
                                                           "checker_training_mode": st.cfg.get("checker_training_mode", "auto"),
                                                           "checker_path": (st.cfg.get("checker") or {}).get("path")},
                                          training_dev=after_chat_yields)
                    rep["fits"] = mode_fits(st, after_chat_yields, rep["modes"]["text"])
                    return self._send(200, rep)
                if u.path == "/api/registry":
                    prune_model_catalogue(st)
                    reg = load_registry()
                    if q.get("path"):
                        e = reg.get(q["path"])
                        if e is None:
                            return self._send(404, {"error": "no such model in the registry"})
                        return self._send(200, {"path": q["path"], **e})
                    return self._send(200, {"current": st.model_path, "models": [
                        {"path": k, "name": v.get("display") or Path(k).name, "exists": Path(k).exists(),
                         "added": v.get("added"), "last": (v.get("history") or [{}])[-1].get("t"),
                         "events": len(v.get("history") or []),
                         "canary": {x: (v.get("canary") or {}).get(x) for x in ("passed", "of")},
                         "licence": v.get("licence", ""),
                         "mb": round(Path(k).stat().st_size / 1e6) if Path(k).exists() else None}
                        for k, v in reg.items()]})
                if u.path == "/api/quantize":
                    return self._send(200, st.quant.status())
                if u.path == "/api/chats":
                    if q.get("id"):
                        c = load_json(CHATS / f"{q['id']}.json", None)
                        return self._send(200 if c else 404, c or {"error": "no such chat"})
                    # ?q= finds chats by title or by anything said in them; ?limit= pages past 100
                    words = [w for w in (q.get("q") or "").lower().split() if w]
                    limit = max(1, min(int(q.get("limit") or 100), 1000))
                    out = []
                    for f in CHATS.glob("*.json") if CHATS.is_dir() else []:
                        c = load_json(f, {})
                        if words:
                            hay = (c.get("title", "") + " " + " ".join(str(m.get("content", ""))
                                   for m in c.get("messages") or [] if isinstance(m, dict))).lower()
                            if not all(w in hay for w in words):
                                continue
                        out.append({"id": f.stem, "title": c.get("title", "Chat"), "t": c.get("t", 0)})
                    out.sort(key=lambda x: -x["t"])
                    return self._send(200, {"chats": out[:limit], "total": len(out)})
                if u.path == "/api/engine_log":
                    return self._send(200, {"log": st.engine.log[-200:]})
                if u.path == "/api/events":
                    qq = queue.Queue(maxsize=500)
                    st.subscribers.append(qq)
                    self._sse_start()
                    try:
                        while True:
                            try:
                                self._sse(qq.get(timeout=15))
                            except queue.Empty:
                                self.wfile.write(b": ping\n\n")
                                self.wfile.flush()
                    except OSError:
                        pass
                    finally:
                        st.subscribers.remove(qq)
                    return
                if u.path == "/health":
                    return self._send(200, {"ok": True})
                if u.path == "/favicon.ico":
                    icon = UI.parent / "icon.ico"
                    if icon.is_file():
                        return self._send(200, icon.read_bytes(), "image/x-icon")
                # static UI
                rel = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
                f = (UI / rel).resolve()
                if UI.resolve() in f.parents and f.is_file():
                    return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "text/plain")
                return self._send(404, {"error": "not found"})
            except ToolError as e:
                return self._send(400, {"error": str(e)})

        def do_POST(self):
            if self._foreign():
                self.close_connection = True
                return self._send(403, {"error": "requests from other sites are refused"})
            u = urllib.parse.urlparse(self.path)
            if u.path == "/api/upload":
                # a file attached to a message: raw bytes, the name in ?name=; saved under attachments/
                n = int(self.headers.get("Content-Length") or 0)
                name = urllib.parse.parse_qs(u.query).get("name", ["file"])[0]
                from . import attachments as _att
                if not n or n > _att.MAX_BYTES:
                    return self._send(400, {"error": "no file, or one larger than 500 MB"})
                try:
                    rel = _att.save(st.ws, name, self.rfile.read(n))
                except ToolError as e:
                    return self._send(400, {"error": str(e)})
                return self._send(200, {"path": rel})
            if u.path == "/api/transcribe":
                # the microphone's recording (webm/opus): speech to text on this computer
                n = int(self.headers.get("Content-Length") or 0)
                if not n or n > 50_000_000:
                    return self._send(400, {"error": "no recording, or one too long"})
                audio = self.rfile.read(n)
                import tempfile as _tf
                ext = ".webm" if "webm" in (self.headers.get("Content-Type") or "webm") else ".wav"
                f = _tf.NamedTemporaryFile(suffix=ext, delete=False)
                try:
                    f.write(audio)
                    f.close()
                    if "whisper-small" in set(st.cfg.get("disabled_media_models") or []):
                        return self._send(409, {"error": "Whisper small is disabled. Choose it in the model menu to enable it."})
                    from .adapters.install import ensure
                    ensure("audio")
                    from .tools.media import _whisper, for_whisper
                    t0 = time.time()
                    segs, info = _whisper().transcribe(for_whisper(f.name), vad_filter=True)
                    text = " ".join(s.text.strip() for s in segs).strip()
                    return self._send(200, {"text": text, "language": info.language, "seconds": round(time.time() - t0, 2)})
                except Exception as e:
                    return self._send(500, {"error": f"speech to text failed: {str(e)[:200]}"})
                finally:
                    Path(f.name).unlink(missing_ok=True)
            b = self._body()
            try:
                if u.path == "/api/hf_login":
                    from . import hf_source
                    try:
                        if b.get("logout"):
                            hf_source.sign_out()
                            return self._send(200, {"user": ""})
                        return self._send(200, {"user": hf_source.sign_in(str(b.get("token") or ""))})
                    except ValueError as exc:
                        return self._send(400, {"error": str(exc)})
                    except Exception as exc:
                        return self._send(400, {"error": "Hugging Face did not accept that token: " + str(exc)[:160]})
                if u.path == "/api/model_check":
                    from . import model_slots as _slots
                    try:
                        from .orchestrator import plan as _plan
                        try:
                            dml = hardware.directml_adapters()
                        except Exception:
                            dml = []
                        try:
                            cuda = hardware.media_cuda()
                        except Exception:
                            cuda = False
                        checked = _slots.check(str(b.get("slot") or ""), str(b.get("repo_id") or ""),
                                               device=st.orch.freed(probe(0)), directml=dml, cuda_ready=cuda,
                                               plan_fn=_plan, device_pref=chosen_device(st),
                                               plan_kw=plan_settings(st.cfg))     # the same sizing as the Quantizer
                    except Exception as exc:
                        return self._send(400, {"error": f"could not check this model: {str(exc)[:200]}"})
                    return self._send(200, checked)
                if u.path == "/api/model_publish":
                    # Part C5: a built .ssf to the signed-in person's Hugging Face account (private first)
                    from . import publish as _pub
                    path = str(b.get("path") or "")
                    try:
                        return self._send(200, _pub.publish(path, load_registry().get(path) or {},
                                                            str(b.get("name") or ""), b.get("private", True) is not False))
                    except ValueError as exc:
                        return self._send(400, {"error": str(exc)})
                    except Exception as exc:
                        # a read-only token is the usual cause: say which token to make
                        m = str(exc).lower()
                        if any(x in m for x in ("403", "forbidden", "write access", "not authorized", "unauthorized")):
                            return self._send(400, {"error": "Your Hugging Face token can only read. Make one of type Write at "
                                                             "huggingface.co/settings/tokens and sign in with it."})
                        from . import hf_source
                        return self._send(400, {"error": "Could not publish: " + hf_source.hub_error(exc, path)})
                if u.path == "/api/model_remove":
                    try:
                        removed = remove_managed_model(st, str(b.get("path") or ""))
                    except ValueError as exc:
                        return self._send(400, {"error": str(exc)})
                    except RuntimeError as exc:
                        return self._send(409, {"error": str(exc)})
                    return self._send(200, removed)
                if u.path == "/api/model_use":
                    from . import model_slots as _slots
                    slot, repo_id = str(b.get("slot") or ""), str(b.get("repo_id") or "")
                    checked = _slots.check(slot, repo_id)
                    if not checked.get("available") or not checked.get("compatible"):
                        return self._send(400, {"error": checked.get("reason") or "check the repository before using it"})
                    if checked.get("remote_code") and (checked.get("fit_needed") or checked.get("compat_needed")) \
                            and not b.get("allow_remote_code"):
                        return self._send(400, {"error": REMOTE_CODE_ASK})
                    if slot in ("chat", "checker"):
                        level = str(b.get("quant_level") or (checked.get("recommendation") or {}).get("quant_level") or "balanced")
                        if level not in {item["id"] for item in levels.LEVELS}:
                            return self._send(400, {"error": "choose a supported Sheep quantization level"})
                        try:
                            # checked["repo_id"] is the original when a GGUF/GPTQ/MLX page was given
                            start_slot_quantize(st, slot, checked.get("repo_id") or repo_id, level,
                                                checked.get("license", ""),
                                                fallback="fit" if checked.get("fit_needed") else
                                                "compat" if checked.get("compat_needed") else "",
                                                allow_remote_code=bool(b.get("allow_remote_code")))
                        except RuntimeError as exc:
                            return self._send(409, {"error": str(exc)})
                        return self._send(202, {"accepted": True, "slot": slot, "quantize": st.quant.status(),
                                                "message": "Downloading the model, then building it for this computer."})
                    rec = checked.get("recommendation") or {}
                    if rec.get("style") or rec.get("speech_repo"):
                        from . import media_styles as _ms
                        from .tools import media as _tm
                        mm = st.cfg.setdefault("media_models", {})
                        key = "image_style" if rec.get("style") else "speech_repo"
                        before = mm.get(key)

                        def _restart():
                            for w in list(getattr(_tm, "_Worker").ALL):
                                w.stop()                          # the next picture loads the chosen base
                            _tm._WHISPER.clear()
                        try:
                            if rec.get("style"):
                                _ms.install_style(rec["style"])
                                from .adapters import MEDIA_DIR as _md
                                if not any((_md / "lcm").glob("*.safetensors")):
                                    from .adapters.fetch import fetch as _fetch
                                    _fetch("lcm")                 # the 4-step add-on the style runs with
                            else:
                                _ms.install_whisper(rec["speech_repo"])
                            mm[key] = rec.get("style") or rec["speech_repo"]
                            _restart()
                            # a test run before it is trusted: a broken or incomplete download fails here,
                            # not on the person's first real request (2026-10-27)
                            ok, why = _ms.test_run(key, st)
                        except Exception as exc:
                            ok, why = False, f"could not install it: {str(exc)[:200]}"
                        if not ok:
                            if before:
                                mm[key] = before
                            else:
                                mm.pop(key, None)
                            _restart()
                            st.save()
                            return self._send(409, {"error": f"{repo_id} did not pass its test run, so the previous "
                                                             f"model stays in use: {why}"})
                        st.save()
                        return self._send(200, {"accepted": True, "slots": _slots.current(st),
                                                "message": f"{repo_id} is installed and chosen."})
                    from .model_slots import MEDIA_REPOS
                    media_id = MEDIA_REPOS[slot][1]
                    try:
                        result = media_model_action(st, media_id, "use")
                    except (ValueError, RuntimeError, ToolError) as exc:
                        return self._send(409, {"error": str(exc)})
                    st.save()
                    return self._send(200, {**result, "slots": _slots.current(st), "accepted": True})
                if u.path == "/api/model_slot_job":
                    return self._send(200, {"quantize": st.quant.status(), "checker": st.checker.status(),
                                            "orchestrator": st.orch.status(), "model": st.model_card(),
                                            "engine_up": st.engine.alive()})
                if u.path == "/api/model_slot_cancel":
                    slot = str(b.get("slot") or "")
                    if slot in ("chat", "checker") and st.quant.state in ("fetching", "converting", "fitting", "building"):
                        st.quant.stop()
                        return self._send(200, {"cancelled": True, "job": st.quant.status()})
                    if slot == "checker" and st.checker.trainer and st.checker.trial.get("state") == "training":
                        st.checker.trainer.stop()
                        st.checker.trial = {"state": "stopped"}
                        return self._send(200, {"cancelled": True, "checker": st.checker.status()})
                    from .adapters.install import cancel as cancel_install
                    return self._send(200, cancel_install())
                if u.path == "/api/media_model":
                    try:
                        result = media_model_action(st, str(b.get("id") or ""), str(b.get("action") or ""))
                    except ValueError as exc:
                        return self._send(400, {"error": str(exc)})
                    except (RuntimeError, ToolError) as exc:
                        return self._send(409, {"error": str(exc)})
                    st.save()
                    return self._send(200, result)
                if u.path == "/api/op":
                    # the editors ask the same yes the agent asks (deletes, sends, overwrites)
                    name, args = b.get("tool", ""), dict(b.get("args") or {})
                    why = None if b.get("confirmed") else T.needs_confirm(st.ws, name, args)
                    if why:
                        return self._send(409, {"error": f"needs your yes: {why}", "confirm": why})
                    ctx = T.Ctx(st.ws, "user")
                    ctx.state = st
                    res = T.run(st.ws, name, args, by="user", ctx=ctx)
                    return self._send(200, {"ok": True, "result": res})
                if u.path == "/api/settings":
                    if "preset" in b:                   # a Basics choice: its bundle, unless a key was also sent
                        if b["preset"] not in PRESETS:
                            return self._send(400, {"error": "choose Fastest, Balanced or Cool and quiet"})
                        b = {**PRESETS[b["preset"]], **{k: v for k, v in b.items() if k != "preset"}}
                    if isinstance(b.get("keys"), dict):
                        keys = st.cfg.setdefault("keys", {})
                        for k in KEY_ENV:
                            if k in b["keys"]:
                                v = str(b["keys"][k] or "").strip()
                                if v:
                                    keys[k] = v
                                else:
                                    keys.pop(k, None)
                        apply_keys(st.cfg)
                        st.cloud.reload()
                    if "mode" in b:
                        st.mode = 1 if int(b["mode"]) == 1 else 2
                    if b.get("permission") in ("ask", "auto", "plan"):
                        st.permission = b["permission"]
                    if b.get("workspace"):
                        st.ws = Workspace(b["workspace"])
                        st.ws.listeners.append(st._changed)
                    if b.get("theme") in ("dark", "light", "system"):
                        st.cfg["theme"] = b["theme"]
                    if isinstance(b.get("media_model"), dict):
                        # a picture, sound or video model chosen in the drop-down: the default
                        # for its job; the chat stays on the chat model
                        from . import adapters as _ad
                        mm = b["media_model"]
                        m = next((x for x in _ad.catalogue(st.cfg) if x["id"] == mm.get("id")), None)
                        if not m or not m["ready"] or m.get("disabled"):
                            return self._send(400, {"error": "that media model is not ready on this machine"})
                        st.cfg.setdefault("media_models", {})[_ad.role(m)] = m["id"]
                    if "web" in b:
                        from .tools import web as _web
                        st.cfg["web"] = _web.STATE["on"] = bool(b["web"])
                    if "thermal_guard" in b:          # Sheep paces itself by heat (thermal.py)
                        st.cfg["thermal_guard"] = bool(b["thermal_guard"])
                    if "checker_training_mode" in b:
                        training_mode = str(b["checker_training_mode"] or "auto")
                        if training_mode not in ("auto", "cpu", "gpu_bounded"):
                            return self._send(400, {"error": "choose Auto, CPU, or bounded GPU for optional Checker training"})
                        st.cfg["checker_training_mode"] = training_mode
                    replan = False
                    for key, kind in (("device", "text"), ("media_device", "media")):
                        if key in b and b[key] != st.cfg.get(key, "auto"):
                            m = next((x for x in hardware.modes(probe())[kind] if x["id"] == b[key]), None)
                            if not m:
                                return self._send(400, {"error": f"no such choice: {b[key]}"})
                            if not m["ok"]:
                                return self._send(400, {"error": f"{m['label']} is not available here: {m['why']}"})
                            if key == "device" and getattr(st.orch, "lent_from", None) is not None:
                                # the GPU is lent to a job: remember the choice and use it when the GPU
                                # comes back (writing cfg here was undone by the give-back, 2026-10-14)
                                st.orch.lent_from = b[key]
                                continue
                            st.cfg[key] = b[key]
                            replan = replan or key == "device"
                    # Pool GPUs (off / smart / always) and how far behind the main GPU another
                    # may be: the model is planned again with them
                    if "pool_mode" in b and b["pool_mode"] != st.cfg.get("pool_mode", "smart"):
                        from .orchestrator import POOL_MODES
                        if b["pool_mode"] not in POOL_MODES:
                            return self._send(400, {"error": "Pool GPUs is off, smart or always"})
                        st.cfg["pool_mode"] = b["pool_mode"]
                        replan = True
                    if "gpus_off" in b:                 # GPUs the person switched off for the model
                        names = sorted({str(n).strip() for n in (b["gpus_off"] or []) if str(n).strip()})
                        replan = replan or names != sorted(st.cfg.get("gpus_off") or [])
                        st.cfg["gpus_off"] = names
                    if "vram_reserve_gb" in b:          # VRAM kept free on the main GPU for other programs
                        try:
                            res = float(b["vram_reserve_gb"])
                        except (TypeError, ValueError):
                            return self._send(400, {"error": "VRAM to keep free is a number of GB, like 0.5"})
                        if not 0 <= res <= 16:
                            return self._send(400, {"error": "VRAM to keep free is between 0 and 16 GB"})
                        replan = replan or res != float(st.cfg.get("vram_reserve_gb", 0) or 0)
                        st.cfg["vram_reserve_gb"] = res
                    if "cpu_offload" in b:              # may one model be split between GPU and CPU
                        replan = replan or bool(b["cpu_offload"]) != bool(st.cfg.get("cpu_offload", True))
                        st.cfg["cpu_offload"] = bool(b["cpu_offload"])
                    if "pool_gap" in b:
                        try:
                            gap = float(b["pool_gap"])
                        except (TypeError, ValueError):
                            return self._send(400, {"error": "the speed gap is a number, like 4"})
                        if not 1.0 <= gap <= 20.0:
                            return self._send(400, {"error": "the speed gap is between 1 and 20 times"})
                        replan = replan or gap != float(st.cfg.get("pool_gap", 4))
                        st.cfg["pool_gap"] = gap
                    if replan and st.model_path and "model" not in b:
                        st.orch.start_model(st.model_path)       # re-plans and restarts on the chosen device
                    if "model" in b and b["model"] != st.model_path:
                        try:
                            st.model_path = validate_ssf_path(b["model"])
                        except ValueError as exc:
                            return self._send(400, {"error": str(exc)})
                        st.orch.egypt = None
                        st.orch.start_model(st.model_path)       # plans, starts, runs the canaries
                    st.save()
                    return self._send(200, st.snapshot())
                if u.path == "/api/warm":
                    app = b.get("app", "")
                    if st.engine.alive() and app not in st.warming:
                        st.warming.add(app)

                        def go():
                            try:
                                ms = agent.warm(st, app)
                                st.warmed[app] = round(ms)
                            except Exception as e:      # a failed warm-up only costs speed
                                st.warmed[app] = f"failed: {e}"
                            finally:
                                st.warming.discard(app)
                        threading.Thread(target=go, daemon=True).start()
                    return self._send(200, {"warming": sorted(st.warming), "warmed": st.warmed})
                if u.path == "/api/chats":
                    import re as _re
                    import uuid as _uuid
                    cid = str(b.get("id") or "")
                    if not _re.fullmatch(r"[a-f0-9]{8,32}", cid):
                        cid = _uuid.uuid4().hex[:12]
                    if b.get("delete"):
                        (CHATS / f"{cid}.json").unlink(missing_ok=True)
                        return self._send(200, {"ok": True})
                    old = load_json(CHATS / f"{cid}.json", {})
                    if "rename" in b:
                        # a title the person chose stays: later saves no longer replace it
                        if not old:
                            return self._send(404, {"error": "no such chat"})
                        old["title"] = str(b.get("rename") or "").strip()[:60] or old.get("title", "Chat")
                        old["renamed"] = True
                        save_json(CHATS / f"{cid}.json", old)
                        return self._send(200, {"id": cid, "title": old["title"]})
                    msgs = b.get("messages") or []
                    first = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "") or "New chat"
                    title = old.get("title") if old.get("renamed") else str(b.get("title") or first)[:60]
                    save_json(CHATS / f"{cid}.json", {"id": cid, "title": title, "renamed": bool(old.get("renamed")),
                                                      "t": time.time(), "messages": msgs, "view": b.get("view") or [],
                                                      "canvas": b.get("canvas") or {}})
                    return self._send(200, {"id": cid})
                if u.path == "/api/import_media":
                    # the person picked a file outside the workspace: copy it into media/
                    import shutil as _sh
                    src = Path(str(b.get("path") or ""))
                    if not src.is_file():
                        return self._send(400, {"error": "no such file"})
                    dst = st.ws.path("media/" + src.name)
                    k = 1
                    while dst.exists():
                        dst = st.ws.path(f"media/{src.stem} ({k}){src.suffix}")
                        k += 1
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _sh.copy2(src, dst)
                    st._changed(st.ws.rel(dst))
                    return self._send(200, {"path": st.ws.rel(dst)})
                if u.path == "/api/orchestrator":
                    prune_model_catalogue(st)
                    if "on" in b:
                        return self._send(400, {"error": "the orchestrator is always on"})
                    elif b.get("action") == "recheck":
                        try:
                            st.orch.recheck(str(b.get("check") or ""))
                        except (RuntimeError, KeyError) as e:
                            return self._send(400, {"error": str(e)})
                    elif b.get("action") == "replan":
                        st.orch.start_model(st.model_path)
                    elif b.get("action") == "cancel":
                        return self._send(200, {**st.orch.cancel_load(), **st.orch.status()})
                    elif b.get("action") == "canaries":
                        if st.orch.lock.locked():
                            return self._send(409, {"error": "the orchestrator is busy"})

                        def go():
                            with st.orch.lock:
                                st.orch.canary_suite("asked")
                        threading.Thread(target=go, daemon=True).start()
                    return self._send(200, st.orch.status())
                if u.path == "/api/check_text":
                    # Sheep Text & Code's Check: the Checker's parse rules on the file as it stands
                    lang, text = str(b.get("lang") or ""), str(b.get("text") or "")
                    fence = {"python": "python", "javascript": "javascript", "json": "json"}.get(lang)
                    if not text.strip():
                        return self._send(200, {"faults": []})
                    faults = checker_mod.rules(f"```{fence}\n{text}\n```" if fence else text, "", None)
                    if fence:
                        faults = [f for f in faults if "does not parse" in f]     # loops are not faults in code
                    return self._send(200, {"faults": faults, "lang": lang})
                if u.path == "/api/run_code":
                    lang = str(b.get("lang") or "")
                    return self._send(200, run_code(lang, str(b.get("code") or "")))
                if u.path == "/api/orient":
                    try:
                        st.orch.reorient()
                    except RuntimeError as e:
                        return self._send(400, {"error": str(e)})
                    return self._send(200, {"ok": True})
                if u.path == "/api/orient/retry":
                    try:
                        return self._send(202, st.orch.retry_orientation_failure())
                    except RuntimeError as e:
                        return self._send(409, {"error": str(e)})
                if u.path == "/api/checker":
                    try:
                        if b.get("remove"):
                            st.checker.remove()
                        elif b.get("resume_training"):
                            if not st.checker.trainer:
                                raise RuntimeError("there is no Checker training checkpoint to resume")
                            training_mode = str(b.get("training_mode") or "cpu")
                            if training_mode not in ("cpu", "gpu_bounded"):
                                raise RuntimeError("choose CPU or bounded GPU training")
                            st.checker.trial = {"state": "training", "path": st.checker.trainer.path}
                            st.checker.trainer.resume(training_mode)
                        elif b.get("stop_training") and st.checker.trainer:
                            st.checker.trainer.stop()
                        elif b.get("path"):
                            # train: Sheep teaches it the Checker's job first (the pop-up's yes)
                            training_mode = str(b.get("training_mode") or "cpu")
                            if training_mode not in ("cpu", "gpu_bounded"):
                                raise RuntimeError("choose CPU or bounded GPU training")
                            st.checker.install(str(b["path"]).strip(), train=bool(b.get("train")), orch=st.orch,
                                               training_mode=training_mode)
                    except RuntimeError as e:
                        return self._send(400, {"error": str(e)})
                    return self._send(200, st.checker.status())
                if u.path == "/api/quantize":
                    if b.get("stop"):
                        st.quant.stop()
                        return self._send(200, st.quant.status())

                    def built(res):
                        # staged build (A4): the orchestrator plans it and the Checker runs;
                        # it only enters the registry after the canaries pass. The person sees
                        # a green lamp or a refusal with the reason — nothing more asked of them.
                        st.model_path = res["path"]
                        st.orch.egypt = None
                        st.save()
                        # register after the canaries succeed, so a bad quantize never enters
                        # the registry. The orchestrator's start_model will call back with the
                        # result; we catch the pass here.
                        def on_ready(ok):
                            if ok:
                                reg = load_registry()
                                e = reg.setdefault(res["path"], {})
                                e.update(source_dir=res["source_dir"], built={"t": time.time(), "minutes": res["minutes"],
                                                                              "mb": res["mb"], "recipe": "sheep flat " + str(res.get("level", "balanced")),
                                                                              "level": res.get("level", "balanced"),
                                                                              "from": getattr(st.quant, "converted", ""),
                                                                              "origin": st.quant.src})
                                if b.get("licence"):
                                    e["licence"] = b["licence"]
                                save_registry(reg)
                        st.orch.start_model(st.model_path, on_ready=on_ready)
                    try:
                        st.quant.start(str(b.get("src", "")).strip(), str(b.get("name", "")).strip(),
                                       b.get("budget_mb") or 0, str(b.get("licence", "")).strip(), built,
                                       str(b.get("level") or "balanced"))
                    except RuntimeError as e:
                        return self._send(409, {"error": str(e)})
                    st.orch.replan_for_job()            # the chat engine makes room for the quantizer
                    return self._send(200, st.quant.status())
                if u.path == "/api/gpu_runtime":
                    from . import gpu_runtime
                    if b.get("action") != "install":
                        return self._send(400, {"error": "action must be install"})
                    kind = str(b.get("kind") or "cuda")
                    if kind not in gpu_runtime.KINDS:
                        return self._send(400, {"error": "the runtime is cuda or directml"})
                    return self._send(202, gpu_runtime.start(kind))
                if u.path == "/api/restore":
                    return self._send(200, {"ok": True, "result": st.ws.restore_before(b.get("t", 0), str(b.get("path") or ""))})
                if u.path == "/api/undo":
                    e = st.ws.undo()
                    return self._send(200, {"undone": e})
                if u.path == "/api/redo":
                    e = st.ws.redo()
                    return self._send(200, {"redone": e})
                if u.path == "/api/confirm":
                    return self._send(200, {"ok": st.session.answer(b.get("id"), b.get("allow"))})
                if u.path == "/api/stop":
                    active_action = st.session.request_stop()
                    for p in list(st.session.pending.values()):
                        p["event"].set()
                    return self._send(200, {"ok": True, "active_action": active_action})
                if u.path == "/api/agent":
                    # Let low-priority validation know a foreground request is
                    # arriving before it begins the next background probe.
                    if not st.busy.locked():
                        st.orch.interactive_requested.set()
                    if not st.busy.acquire(blocking=False):
                        return self._send(409, {"error": "the agent is already working; stop it first"})
                    try:
                        self._sse_start()
                        lock = threading.Lock()

                        def emit(ev):
                            with lock:
                                try:
                                    self._sse(ev)
                                except OSError:
                                    st.session.request_stop()
                        if not st.model_path and not (st.mode == 1 and st.cloud.ready()):
                            emit({"type": "error", "text": "no model is running: plug one in on the Models page"})
                            emit({"type": "done", "changed": []})
                        else:
                            try:
                                agent.run(st, b.get("messages") or [], b.get("app", ""), b.get("path", ""), emit,
                                          auto_confirm=b.get("auto_confirm"), cache=b.get("cache", True) is not False)
                            except ToolError as e:
                                # once the stream has started a 400 cannot be sent: the chat stopped with
                                # no word and no end (B10, 2026-10-21). Said in the chat, and the turn ends
                                emit({"type": "text", "route": "sheep", "text": f"I couldn't finish this: {e}"})
                                emit({"type": "done", "failed": True, "changed": []})
                    finally:
                        st.busy.release()
                        st.orch.interactive_requested.clear()
                        st.orch.resume_background_checks()
                    return
                return self._send(404, {"error": "not found"})
            except ToolError as e:
                return self._send(400, {"error": str(e)})

    return H


def find_models(current=""):
    """Sheep model files in usual folders plus the current or registered choices."""
    out = []
    reg = load_registry()
    seen = set()
    candidates = [Path(path) for path in reg]
    if current:
        candidates.append(Path(current))
    for directory in (Path.home() / "models", HOME / "models"):
        if directory.is_dir():
            candidates.extend(directory.glob("*.ssf"))
            candidates.extend(directory.glob("*/*.ssf"))
            candidates.extend(directory.glob("*.compat.json"))     # compatibility mode (the maker's code)
    for path in candidates:
        compat = path.name.lower().endswith(".compat.json")
        if path.suffix.lower() != ".ssf" and not compat:
            continue
        try:
            p = path.expanduser().resolve(strict=True)
            if not p.is_file() or str(p).casefold() in seen:
                continue
            size = p.stat().st_size
            if compat:                     # its weights are the originals it points at, not this small file
                src = Path(json.loads(p.read_text(encoding="utf-8")).get("src") or "")
                size = sum(f.stat().st_size for f in src.glob("*.safetensors")) or size
        except (OSError, ValueError):
            continue
        seen.add(str(p).casefold())
        e = reg.get(str(p), {})
        managed = False
        if e.get("hf_repo") and e.get("source_dir"):
            try:
                original_root = (HOME / "originals").resolve()
                original = Path(e["source_dir"]).resolve(strict=True)
                managed = original != original_root and original_root in original.parents
            except OSError:
                managed = False
        name = e.get("display") or (p.name[:-len(".compat.json")] + " (compatibility mode)" if compat else p.name)
        out.append({"path": str(p), "name": name, "mb": round(size / 1e6), "compat": compat,
                    "licence": e.get("licence", ""), "canary": (e.get("canary") or {}).get("passed"),
                    "kind": "code" if "code" in name.lower() else "text", "managed": managed,
                    "hf_repo": e.get("hf_repo", "") if managed else ""})
    out.sort(key=lambda model: model["name"].casefold())
    return out


def remove_managed_model(state, value):
    """Remove an inactive HF-built SSF and only its Sheep-owned source folder."""
    import shutil
    path = Path(validate_ssf_path(value)).resolve(strict=True)
    models_root = (HOME / "models").resolve()
    if path == models_root or models_root not in path.parents:
        raise ValueError("only Sheep-managed model downloads can be removed here")
    registry = load_registry()
    entry = registry.get(str(path)) or {}
    repo_id = str(entry.get("hf_repo") or "")
    source_value = str(entry.get("source_dir") or "")
    if not repo_id or not source_value:
        raise ValueError("this .ssf was not installed through Sheep's Hugging Face model flow")
    active = {str(Path(p).resolve()) for p in (state.model_path, (state.cfg.get("checker") or {}).get("path", "")) if p}
    if str(path) in active:
        raise RuntimeError("this model is active in Chat or Checker; switch that slot before removing it")
    originals_root = (HOME / "originals").resolve()
    source = Path(source_value).resolve()
    expected = (originals_root / repo_id.replace("/", "--")).resolve()
    if source != expected or originals_root not in source.parents:
        raise RuntimeError("the recorded original is not in Sheep's managed model folder; refusing to remove it")
    path.unlink()
    registry.pop(str(path), None)
    save_registry(registry)
    reused = any(str(Path(row.get("source_dir", "")).resolve()) == str(source) for row in registry.values()
                 if row.get("source_dir"))
    original_removed = False
    if not reused and source.exists():
        shutil.rmtree(source)
        original_removed = True
    return {"removed": str(path), "original_removed": original_removed,
            "shared_hugging_face_cache_preserved": True}


def validate_ssf_path(value):
    """Validate and normalize a user-picked native Sheep model file."""
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    if path.suffix.lower() != ".ssf" and not path.name.lower().endswith(".compat.json"):
        raise ValueError("choose a Sheep model file with the .ssf extension")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("no such model file") from exc
    if not path.is_file():
        raise ValueError("no such model file")
    try:
        sys.path.insert(0, str(ROOT / "tooling"))
        import sml_read
        model = sml_read.SmlFile(str(path))
        if not isinstance(model.head, dict) or not model.head.get("arch"):
            raise ValueError("the file has no Sheep model header")
        del model
    except Exception as exc:
        raise ValueError(f"cannot read Sheep .ssf model: {str(exc)[:180]}") from exc
    return str(path)


def media_model_action(state, model_id, action):
    """Check, install and choose a media feature, or remove/disable its model."""
    from . import adapters as _ad
    from .adapters import install as _install
    if action not in ("use", "remove"):
        raise ValueError("choose use or remove")
    model = next((m for m in _ad.catalogue(state.cfg) if m["id"] == model_id), None)
    if not model:
        raise ValueError("choose a listed media model")
    chosen = state.cfg.setdefault("media_models", {})
    disabled = set(state.cfg.get("disabled_media_models") or [])
    if action == "use":
        if model.get("install") and not model.get("ready"):
            if not model.get("installable"):
                raise RuntimeError("the required installer or runtime is not available on this computer")
            _install.ensure(model["install"])
            model = next((m for m in _ad.catalogue(state.cfg) if m["id"] == model_id), model)
        if not model.get("ready"):
            raise RuntimeError(model.get("why") or "this media model is unavailable")
        disabled.discard(model_id)
        state.cfg["disabled_media_models"] = sorted(disabled)
        chosen[_ad.role(model)] = model_id
    else:
        if model_id == "windows-voices":
            disabled.add(model_id)  # Windows owns these voices; Sheep can disable their use.
        else:
            _install.remove(model_id)
            disabled.add(model_id)
        if chosen.get(_ad.role(model)) == model_id:
            chosen.pop(_ad.role(model), None)
        state.cfg["disabled_media_models"] = sorted(disabled)
    return {"media": [{**m, "role": _ad.role(m)} for m in _ad.catalogue(state.cfg)],
            "defaults": _ad.defaults(state.cfg)}


REMOTE_CODE_ASK = "This model runs its own code from Hugging Face. Allow?"


def start_slot_quantize(state, slot, repo_id, level, licence="", fallback="", allow_remote_code=False):
    """Build a confirmed text checkpoint and activate it only after its slot gate.
    fallback "fit": fit the unknown architecture to the engine first; "compat": run it in
    compatibility mode. A fit that fails falls through to compatibility mode too."""
    if slot not in ("chat", "checker"):
        raise ValueError("only Chat and Checker accept quantized text checkpoints")
    if state.quant.state in ("fetching", "converting", "fitting", "building"):
        raise RuntimeError("a model build is already running")

    def remember(res):
        reg = load_registry()
        entry = reg.setdefault(res["path"], {})
        entry.update(source_dir=res["source_dir"], licence=licence,
                     built={"t": time.time(), "minutes": res["minutes"], "mb": res["mb"],
                            "recipe": "compatibility mode" if res.get("compat") else "sheep flat " + str(res.get("level", level)),
                            "level": res.get("level", level)},
                     hf_repo=repo_id, slot_candidate=slot)
        save_registry(reg)

    if slot == "checker":
        def checker_candidate(res):
            remember(res)  # Checker training can use the original source if requested later.
            try:
                state.checker.install(res["path"], train=False, orch=state.orch)
            except Exception as exc:
                state.checker.trial = {"state": "rolled back", "path": res["path"], "why": str(exc)[:200]}
        callback = checker_candidate
    else:
        previous = state.model_path

        def chat_candidate(res):
            state.model_path = res["path"]
            state.orch.egypt = None
            state.save()

            def ready(ok):
                if ok:
                    remember(res)
                    return

                def restore():
                    state.model_path = previous if previous and Path(previous).is_file() else ""
                    state.orch.egypt = None
                    state.save()
                    if state.model_path:
                        state.orch.start_model(state.model_path)
                    else:
                        state.orch.lamp = "amber" if state.orch.on else "off"
                        state.orch.stage = "new model failed its checks; previous selection restored"
                threading.Thread(target=restore, daemon=True).start()

            state.orch.start_model(res["path"], on_ready=ready)
        callback = chat_candidate

    state.quant.start(repo_id, "", 0, licence, callback, level, fallback=fallback,
                      allow_remote_code=allow_remote_code)
    state.orch.replan_for_job()
    return state.quant.status()


def prune_model_catalogue(st=None):
    """Drop deleted model references from the catalogue and clear a stale selection."""
    reg = load_registry()
    cleaned = {path: entry for path, entry in reg.items() if Path(path).is_file()}
    if len(cleaned) != len(reg):
        save_registry(cleaned)
    if st and st.model_path and not Path(st.model_path).is_file():
        st.orch.cancel_load()
        st.engine.stop()
        st.model_path = ""
        st.orch.plan = None
        st.orch.facts = None
        st.orch.lamp = "amber" if st.orch.on else "off"
        st.orch.stage = "selected model was removed from disk"
        st.orch.progress = 0.0
        st.save()


def main():
    # the backend's own lines carry model text: a full-width "！" in a Youtu-LLM reply crashed a log print on
    # the Windows code page and the model never started (2026-10-20)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Sheep backend")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--engine-port", type=int, default=8099)
    ap.add_argument("--model", default="")
    ap.add_argument("--workspace", default="")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=4)
    a = ap.parse_args()
    # Another Sheep already serving this port: leave now, before State() starts an
    # engine (whose start-up clean-up would stop the running app's engine)
    import socket as _socket
    probe = _socket.socket()
    try:
        probe.bind(("127.0.0.1", a.port))
    except OSError:
        print(f"Sheep is already running on port {a.port}", flush=True)
        sys.exit(1)
    finally:
        probe.close()
    # several NVIDIA cards: every program Sheep starts (engine, checks, PyTorch workers) uses the
    # one the plans are made for, the card with the most memory (Part C7)
    os.environ.update({k: v for k, v in _cuda_env().items() if k.startswith("CUDA_")})
    st = State(a)
    st.orch.start_model(st.model_path)          # probes, plans, starts the engine, runs the canaries
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st))
    srv.daemon_threads = True
    print(f"Sheep on http://127.0.0.1:{a.port}  workspace {st.ws.root}  mode {st.mode}  "
          f"Nebius {'ready: ' + st.cloud.model if st.cloud.ready() else 'off (no NEBIUS_API_KEY)'}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        st.engine.stop()


if __name__ == "__main__":
    main()
