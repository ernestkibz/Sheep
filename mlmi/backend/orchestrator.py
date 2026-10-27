"""The orchestrator: the system's admin (STANDARDS.md section 6).

It probes this machine, plans how a model runs on it, starts the engine with
that plan, and runs a short deterministic activation exam. Full canaries and
model orientation continue in the background and yield to interactive work.
Every decision goes to the log with its reason. The only thing a person sets is
the on/off switch: with it off, the engine runs on fixed defaults and nothing
is checked.

  lamp   off     the switch is off
         amber   planning or loading the model, or recovering from a fault
         green   the engine is usable; the compact activation exam passed
         red     the engine or compact activation exam failed
"""
import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .paths import HOME, REPO

MODELS = HOME / "models.json"
TOOLING = REPO / "tooling"
KV_BYTES = 2                     # the engine keeps K and V as fp16
HEADROOM = 1.5e9                 # RAM left for the OS, the browser and the app
GPU_RESERVE = 1.0e9              # VRAM the engine keeps free (700 MB) plus its prompt scratch
GPU_CTX = 4096                   # the GPU keeps K and V as fp32: 8192 would crowd a 4 GB card
DEVICES = ("auto", "cpu", "gpu", "hybrid", "amd", "multi")
FP16_PER_BYTE = 16 / 4.5         # a Sheep file averages ~4.5 bits a weight; fp16 holds 16
DML_LAYER_S = 0.001              # DirectML's Python and dispatch per layer per token (~15 ops), an estimate
CANARY_TOKENS = 160
QUICK_CANARY_TOKENS = 64
from .orient import QUICK as _ORIENT_QUICK
CANARY_REVISION = 3
ORIENTATION_REVISION = 6  # 6: media rehearsed, fu004 shape, second try, file-first task (2026-10-20)
LOG = HOME / "logs" / "activity.jsonl"   # the decision log, kept across restarts
ORIENTATION_TASKS = len(_ORIENT_QUICK)        # the jobs a new model does before first use


def drop_old_checks(root):
    """Earlier signatures' check folders of this model are never read again. Only 16-hex folders: the
    Llama-3.2 folder's old flat layout is left alone."""
    for old in Path(root).parent.glob("*"):
        if old.is_dir() and old != Path(root) and re.fullmatch(r"[0-9a-f]{16}", old.name):
            shutil.rmtree(old, ignore_errors=True)


def load_registry():
    try:
        return json.loads(MODELS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_registry(reg):
    MODELS.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(MODELS) + ".tmp")
    tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    os.replace(tmp, MODELS)


# ------------------------------------------------------------------ the machine

_probe_cache = {"t": 0.0, "v": None}


def _nvidia():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    q = "index,name,memory.total,memory.used,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,utilization.gpu"
    try:
        out = subprocess.run([exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return pick_nvidia(out)


def pick_nvidia(out):
    """nvidia-smi's rows to the card Sheep uses: the one with the most memory (then the least
    in use). Only the first row was read before (2026-10-21, Part C7), so on a machine with
    several cards a small first one was planned and the big ones never seen."""
    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    cards = []
    for line in (out or "").strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue
        cards.append({"index": int(num(f[0]) or 0), "name": f[1], "vram_total": (num(f[2]) or 0) * 2**20,
                      "vram_used": (num(f[3]) or 0) * 2**20, "temp": num(f[4]), "clock": num(f[5]),
                      "clock_max": num(f[6]), "power": num(f[7]), "util": num(f[8])})
    if not cards:
        return None
    best = max(cards, key=lambda c: (c["vram_total"], -c["vram_used"], -c["index"]))
    return dict(best, count=len(cards), others=[c["name"] for c in cards if c is not best])


def cuda_env(env=None, gpu=None):
    """The environment for a program that uses the NVIDIA card: pinned to the card Sheep
    planned for when there are several (nvidia-smi counts in PCI order, CUDA fastest first)."""
    env = dict(os.environ if env is None else env)
    g = gpu if gpu is not None else (probe() or {}).get("gpu")
    if g and g.get("count", 1) > 1 and "CUDA_VISIBLE_DEVICES" not in env:
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g.get("index", 0)))
    return env


def probe(max_age=4.0):
    """This machine now: CPU, RAM, power, GPU. Cached for a few seconds."""
    if _probe_cache["v"] and time.time() - _probe_cache["t"] < max_age:
        return _probe_cache["v"]
    import psutil
    vm = psutil.virtual_memory()
    bat = None
    try:
        b = psutil.sensors_battery()
        if b:
            bat = {"percent": round(b.percent), "plugged": bool(b.power_plugged)}
    except (AttributeError, OSError):
        pass
    from .thermal import fake_power
    bat = fake_power() or bat                               # the journeys' battery (tooling/ui_audit)
    cpu_name = ""
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    v = {"cpu": {"name": cpu_name or "CPU", "cores": psutil.cpu_count(logical=False) or 1,
                 "threads": psutil.cpu_count() or 1, "load": psutil.cpu_percent(interval=None)},
         "ram": {"total": vm.total, "available": vm.available},
         "power": bat or {"percent": None, "plugged": True},
         "gpu": _nvidia(), "t": time.time()}
    try:
        from . import hardware
        v["others"] = [a for a in hardware.adapters() if a["vendor"] != "nvidia"]
        v["amd"] = hardware.amd_adapter()
        v["pool"] = hardware.dml_pool(cuda_in_use=bool(v["gpu"]))     # every other GPU, any brand
    except Exception:
        v["others"], v["amd"], v["pool"] = [], None, []
    _probe_cache.update(t=time.time(), v=v)
    return v


# ------------------------------------------------------------------ the model

def compat_facts(path):
    """model_facts for compatibility mode: the original checkpoint's config, not a Sheep header."""
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    src = Path(c["src"])
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    tc = cfg.get("text_config") or cfg
    n_layer, n_embd = int(tc.get("num_hidden_layers") or 0), int(tc.get("hidden_size") or 0)
    n_head = int(tc.get("num_attention_heads") or 1) or 1
    n_kv = int(tc.get("num_key_value_heads") or n_head)
    hd = int(tc.get("head_dim") or (n_embd // n_head if n_embd else 128))
    size = sum(f.stat().st_size for f in src.iterdir() if f.suffix in (".safetensors", ".bin"))
    return {"arch": str(cfg.get("model_type", "?")), "n_layer": n_layer, "n_embd": n_embd, "n_kv": n_kv,
            "head_dim": hd, "n_ctx_train": int(tc.get("max_position_embeddings") or 0), "bytes": size,
            "n_weights": size // 2, "kv_per_token": n_layer * n_kv * hd * 2 * KV_BYTES, "template": True,
            "stops": [], "exact": False, "widths": [], "gpu_widths_ok": False, "source": c.get("repo", ""),
            "licence": "UNKNOWN", "duties": [], "compat": True}


def originals_for(entry):
    """{"src", "trust_remote_code"} of a native model's original checkpoint, when it is still on disk."""
    src = Path(str((entry or {}).get("source_dir") or ""))
    try:
        cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not any(src.glob("*.safetensors")):
        return None
    return {"src": str(src), "trust_remote_code": bool(cfg.get("auto_map"))}


def model_facts(path):
    """What the plan needs from the file's own header: shape, size, chat block, licence."""
    if str(path).lower().endswith(".compat.json"):
        return compat_facts(path)
    sys.path.insert(0, str(TOOLING))
    import sml_read
    f = sml_read.SmlFile(path)
    h = f.head
    # weights by count, so a device holding them in fp16 (DirectML) is sized exactly
    n_weights = sum(int(t["rows"]) * int(t["cols"]) for t in f.tensors.values())
    # the NVIDIA engine has 4- and 6-bit kernels only (sml-gpu-fwd.c gpu_setup): a Q8 7B was
    # planned "hybrid, 9 of 28 layers on the GPU" and then ran on the CPU (2026-10-14)
    widths = sorted({int(t["nbits"]) for t in f.tensors.values() if t.get("layout") == 1 and t.get("nbits")})
    del f
    hp = h.get("hparams", {})
    n_layer, n_embd = int(hp.get("n_layer", 0)), int(hp.get("n_embd", 0))
    n_head = int(hp.get("n_head", 1)) or 1
    n_kv = int(hp.get("n_head_kv", n_head)) or n_head
    hd = int(hp.get("head_dim", 0) or (n_embd // n_head))
    chat = h.get("chat") or {}
    prov = h.get("provenance") or {}
    return {"arch": h.get("arch", "?"), "n_layer": n_layer, "n_embd": n_embd, "n_kv": n_kv, "head_dim": hd,
            "n_ctx_train": int(hp.get("n_ctx_train", 0) or 0), "bytes": os.path.getsize(path), "n_weights": n_weights,
            "kv_per_token": n_layer * n_kv * hd * 2 * KV_BYTES, "n_head": n_head,
            "template": bool(chat.get("template")), "stops": list(chat.get("stop_ids") or []),
            "exact": h.get("precision") == "exact-f32",
            "widths": widths, "gpu_widths_ok": all(w in GPU_WIDTHS for w in widths),
            "source": h.get("source", ""), "licence": prov.get("license", "UNKNOWN"),
            "duties": prov.get("duties") or []}


def engine_stamp():
    """Size and time of the engine and its GPU kernels. A model's GPU fault is remembered
    until either changes: then the GPU is tried again, so a fixed engine gives it back."""
    name = "sml-server.exe" if os.name == "nt" else "sml-server"
    out = []
    vk = "sml-vk.exe" if os.name == "nt" else "sml-vk"
    for f in (Path(os.environ.get("MLMI_ENGINE_EXE", REPO / "sml" / name)), REPO / "sml" / "sml-gpu-fwd.ptx",
              REPO / "sml" / "sml-gpu-fwd-wide.ptx", REPO / "sml" / vk, REPO / "sml" / "vk" / "mv4.spv",
              REPO / "sml" / "vk" / "mv6.spv"):
        try:
            st_ = f.stat()
            out.append(f"{st_.st_size}:{st_.st_mtime_ns}")
        except OSError:
            out.append("-")
    return "|".join(out)


GPU_CHECK_IDS = "100,200,300,400,500"      # any vocabulary has these; the maths, not the words, is compared


def parse_gpu_agreement(text):
    """sml-gpu-fwd's "GPU vs CPU over N positions: mean logit corr C, worst W, top-1 agree A/N".
    ok: True when the card computes the model as the CPU does, False when it does not, None
    when no comparison ran (the engine kept the model off the GPU, and says why itself)."""
    m = re.search(r"mean logit corr (-?[\d.]+), worst (-?[\d.]+), top-1 agree (\d+)/(\d+)", text or "")
    if not m:
        why = next((ln.strip() for ln in (text or "").splitlines() if "not on the GPU" in ln or "missing" in ln), "")
        return {"ok": None, "why": why or "no comparison ran"}
    corr, worst, agree, of = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
    r = {"corr": round(corr, 4), "worst": round(worst, 4), "agree": agree, "of": of}
    return dict(r, ok=agreement_ok(r))


def agreement_ok(r):
    """A right GPU path agrees with the CPU to ~0.9997 (3B 0.9997, 7B 0.9998); the overrun one
    was 0.014. Fixed ids make near-ties, so the top word may differ in places on a right path
    (the 3B on its own card: corr 0.9997, top word 5/7, wrongly faulted under the first rule):
    corr >= 0.999 passes; 0.99 .. 0.999 passes when the top word agrees nearly everywhere."""
    c = r.get("corr")
    if c is None:
        return None
    return bool(c >= 0.999 or (c >= 0.99 and r.get("agree", 0) >= r.get("of", 1) - 1))


def add_fault(prior, new):
    """A model's remembered GPU faults, one word for the planner: "gpu" (the NVIDIA card),
    "amd" (the other GPUs), "multi" (both)."""
    nv = {"gpu", "hybrid", "multi"}
    other = {"amd", "multi"}
    has_nv = prior in nv or new in nv
    has_other = prior in other or new in other
    return "multi" if has_nv and has_other else ("gpu" if has_nv else ("amd" if has_other else new))


def gpu_agreement(path, layers=1, timeout=600):
    """Run the model's first layer on the NVIDIA card and the rest on the CPU, against the
    CPU alone, on fixed ids (sml-gpu-fwd). None when the check cannot run here."""
    exe = REPO / "sml" / ("sml-gpu-fwd.exe" if os.name == "nt" else "sml-gpu-fwd")
    if not exe.exists() or not _nvidia():
        return None
    try:
        r = subprocess.run([str(exe), str(path), GPU_CHECK_IDS, "2"], cwd=str(exe.parent), capture_output=True,
                           text=True, errors="replace", timeout=timeout,
                           env=cuda_env(dict(os.environ, SML_GPU_LAYERS=str(layers))),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_gpu_agreement((r.stdout or "") + (r.stderr or ""))


def hardware_sig(dev):
    """The machine and its state a speed trial was measured in: its GPUs, its RAM, and
    whether it runs on the charger or the battery (a laptop's GPU is slower on battery, so
    the fastest mode can differ; a desktop is always "ac"). Each gets its own trial."""
    g, a = dev.get("gpu") or {}, dev.get("amd") or {}
    power = "battery" if (dev.get("power") or {}).get("plugged") is False else "ac"
    return f"{g.get('name', '-')}|{a.get('name', '-')}|{round((dev.get('ram') or {}).get('total', 0) / 1e9)}|{power}"


TRIAL_PROMPT = "Write a short paragraph about how rain forms."
TRIAL_TOKENS = 48
AGREE_PROMPTS = ("What is the capital of France? Answer in one word.", "What is 7 times 8? Answer with the number.",
                 "Name the largest planet in the solar system in one word.")
SPEED_WORDS = {"compat": "compatibility, GPU", "cpu": "native, CPU", "gpu": "native, GPU", "hybrid": "native, GPU + CPU"}

# The engine tune (Ernest, 2026-10-26: "refine Sheep so it tunes well by itself"). The GPU
# launch settings were tuned once, on Llama 3.2; each model now tries a few others on this
# machine. A setting is kept only when its answer is the same word for word and it wins every
# one of TUNE_PAIRS head-to-head pairs by TUNE_GAIN: the laptop moves 5-15% with heat, so one
# fast run proves nothing.
TUNE_CANDIDATES = ({"SML_GRID": "32"}, {"SML_GRID": "64"}, {"SML_GRID": "96"}, {"SML_BLOCK": "256"},
                   {"SML_GRAPH": "0"}, {"SML_TCM5": "0"}, {"SML_T3SWAP": "0"})
TUNE_PAIRS, TUNE_GAIN = 3, 1.03
# The grid cap (48 blocks) was tuned on a 16-SM laptop card; a big card has many more SMs and a
# 48-block job leaves most of them idle. nvidia-smi does not report SMs, so the card's memory
# stands in for its size: bigger cards also try wider grids.
TUNE_WIDE = ((8, ("128",)), (16, ("128", "256")), (40, ("128", "256", "512")))


def tune_candidates(dev):
    vram_gb = ((dev or {}).get("gpu") or {}).get("vram_total", 0) / 2**30
    wide = ()
    for at, grids in TUNE_WIDE:
        if vram_gb >= at - 0.5:
            wide = grids
    return TUNE_CANDIDATES + tuple({"SML_GRID": g} for g in wide)
TUNE_TOKENS = 48
# a prompt long enough that reading it is timed too (about 300 tokens), then the trial question
TUNE_PROMPT = ("The river town kept its market on the old stone bridge. Every morning the traders came before "
               "sunrise, carrying baskets of mangoes, dried fish, cloth dyed in deep blues, and clay pots still "
               "warm from the kiln. ") * 6 + "\n" + TRIAL_PROMPT
TUNE_DEVICES = ("gpu", "hybrid", "multi")
TUNES = HOME / "engine_tunes.json"


def tech_sig(facts):
    """A model as the GPU kernels see it, read from the file's own header, never its name:
    a renamed file keeps its tune, and a file named after another model is not mistaken for it."""
    keys = ("arch", "n_layer", "n_embd", "n_head", "n_kv", "head_dim", "n_weights", "widths", "exact")
    return "|".join(str(facts.get(k)) for k in keys)


def load_tunes():
    try:
        return json.loads(TUNES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_tune(key, value):
    t = load_tunes()
    t[key] = value
    TUNES.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(TUNES) + ".tmp")
    tmp.write_text(json.dumps(t, indent=1), encoding="utf-8")
    os.replace(tmp, TUNES)


POOL_MODES = ("off", "smart", "always")
GPU_WIDTHS = (4, 6)               # the code widths sml-gpu-fwd has kernels for


def _plan_context(dev, facts, pref, pool_mode, gap, exclude, reserve_gb, cpu_offload):
    off = {str(name).lower() for name in exclude or ()}
    is_off = lambda name: str(name or "").replace("(TM)", "").replace("(R)", "").lower() in off \
        or str(name or "").lower() in off
    why = []
    pref = pref if pref in DEVICES else "auto"
    best = facts.get("measured_best")
    if pref == "auto" and best in DEVICES and best != "auto":
        pref = best
        why.append(f"Auto: {best} measured fastest for this model here ({facts.get('measured_note', '')})")
    if facts.get("exact"):
        pref = "cpu"
        why.append("exact f32 weights use the CPU dense path")
    elif facts.get("gpu_widths_ok") is False and dev.get("gpu"):
        dev = dict(dev, gpu=None)
        why.append(f"{'/'.join(f'{width}-bit' for width in facts.get('widths') or [])} weights: the NVIDIA engine runs "
                   "4- and 6-bit models only, so this one runs without it (rebuild at Q6 or Balanced for the GPU)")
    fault = facts.get("gpu_fault")
    if fault in ("gpu", "hybrid", "multi") and dev.get("gpu") and not facts.get("exact"):
        dev = dict(dev, gpu=None)
        why.append("the NVIDIA path gave this model wrong answers in its checks, so it runs without it "
                   "(tried again when the engine is updated)")
    if fault in ("amd", "multi") and (dev.get("amd") or dev.get("pool")):
        dev = dict(dev, amd=None, pool=[])
        why.append("the DirectML path gave this model wrong answers in its checks, so the AMD/Intel GPUs are left out")
    ram = dev["ram"]["available"]
    cores = dev["cpu"]["cores"]
    on_battery = dev["power"].get("plugged") is False
    threads = max(2, cores // 2) if on_battery else cores
    why.append(f"{threads} threads: {'on battery, a low-power profile' if on_battery else 'one per physical core'}")
    budget = ram - facts["bytes"] - HEADROOM
    kv = facts["kv_per_token"] or 1
    ctx, slots = min(4096, facts["n_ctx_train"] or 4096), 1
    for context, count in ((8192, 4), (8192, 2), (8192, 1), (4096, 2), (4096, 1)):
        if facts["n_ctx_train"] and context > facts["n_ctx_train"]:
            continue
        if context * count * kv <= budget:
            ctx, slots = context, count
            break
    why.append(f"context {ctx} x {slots} conversation{'s' if slots > 1 else ''}: "
               f"{ctx * slots * kv / 1e9:.1f} GB of cache fits the {max(budget, 0) / 1e9:.1f} GB RAM left beside the weights")
    return {"dev": dev, "facts": facts, "pref": pref, "pool_mode": pool_mode, "gap": gap,
            "reserve_gb": reserve_gb, "cpu_offload": cpu_offload, "why": why, "is_off": is_off,
            "ram": ram, "threads": threads, "kv": kv, "ctx": ctx, "slots": slots,
            "ram_fit": facts["bytes"] + ctx * kv < ram}


def _plan_gpu_capacity(state):
    dev, facts, why = state["dev"], state["facts"], state["why"]
    g = dev.get("gpu")
    n_layer = max(1, facts["n_layer"])
    gpu = {"fit": "none", "layers": 0}
    gctx = min(GPU_CTX, facts["n_ctx_train"] or GPU_CTX)
    if g and state["is_off"](g.get("name")):
        why.append(f"the {g.get('name')} is switched off in Settings")
        g = None
    if g and g["vram_total"]:
        free = g["vram_total"] - g["vram_used"]
        kvg = 2 * state["kv"]
        room = free - GPU_RESERVE - max(0.0, float(state["reserve_gb"] or 0)) * 1e9
        if state["reserve_gb"]:
            why.append(f"{float(state['reserve_gb']):.1f} GB of VRAM kept free for other programs (Settings)")
        need = facts["bytes"] + gctx * kvg
        if need <= room:
            heads = facts.get("n_head") or 64
            for context in (32768, 16384, 8192):
                if gctx < context <= (facts["n_ctx_train"] or GPU_CTX) and \
                        facts["bytes"] + context * kvg + 512 * heads * context * 4 <= 0.9 * room:
                    gctx, need = context, facts["bytes"] + context * kvg
                    break
            gpu = {"fit": "whole", "layers": n_layer}
            why.append(f"the whole model fits the GPU ({need / 1e9:.1f} GB of {free / 1e9:.1f} GB free)")
        else:
            per = need / n_layer
            layers = min(n_layer, int(max(0, room) // per))
            gpu = {"fit": "split" if layers else "none", "layers": layers}
            why.append(f"{layers} of {n_layer} layers fit the GPU ({free / 1e9:.1f} GB free)" if layers
                       else f"the GPU has no room for this model now ({free / 1e9:.1f} GB free)")
    amd = dev.get("amd") if dev.get("amd") and not dev["amd"].get("missing") else None
    if amd and state["is_off"](amd.get("name")):
        amd = None
    amd_fit = False
    if amd:
        w16 = 2 * facts["n_weights"] if facts.get("n_weights") else facts["bytes"] * FP16_PER_BYTE
        if amd.get("backend") == "vk":
            w16 = facts["bytes"] + gctx * state["kv"]
        need16 = w16 + gctx * state["kv"]
        room16 = (state["ram"] - HEADROOM) if amd["integrated"] else amd["vram"] - 0.5e9
        amd_fit = need16 <= room16
        state.update(need16=need16, room16=room16)
    state.update(g=g, n_layer=n_layer, gpu=gpu, gctx=gctx, amd=amd, amd_fit=amd_fit)


def _plan_pooling(state):
    dev, facts, why = state["dev"], state["facts"], state["why"]
    ram, g, gpu, amd = state["ram"], state["g"], state["gpu"], state["amd"]
    n_layer, gctx, kv = state["n_layer"], state["gctx"], state["kv"]
    device, layers, mid, split = "cpu", 0, 0, []
    per16 = (2 * facts["n_weights"] if facts.get("n_weights") else facts["bytes"] * FP16_PER_BYTE) / n_layer \
        + gctx * kv / n_layer
    t_cuda = facts["bytes"] / n_layer / 70e9
    t_cpu = facts["bytes"] / n_layer / 21e9
    pervk = facts["bytes"] / n_layer + gctx * kv * 2 / n_layer

    def per_of(adapter):
        return pervk if adapter.get("backend") == "vk" else per16

    def t16(adapter):
        return per_of(adapter) / ((adapter.get("gbps") or 0) * 1e9 or
                                  (14e9 if adapter.get("integrated") else 150e9)) + DML_LAYER_S

    vendor_order = {"nvidia": 0, "amd": 1, "intel": 2, "other": 3}
    pool = sorted((adapter for adapter in (dev.get("pool") or [])
                   if not adapter.get("integrated") and not state["is_off"](adapter.get("name"))),
                  key=lambda adapter: (vendor_order.get(adapter.get("vendor"), 3), t16(adapter), -adapter["vram"]))
    pool_mode = state["pool_mode"] if state["pool_mode"] in POOL_MODES else "smart"
    pref = state["pref"]
    if pref == "amd":
        pass
    elif pool_mode == "off":
        pool = []
    elif pool_mode == "smart" or pref != "multi":
        kept = [adapter for adapter in pool if t16(adapter) < t_cpu and (not g or t16(adapter) <= state["gap"] * t_cuda)]
        for adapter in pool if gpu["fit"] != "whole" else []:
            if adapter not in kept:
                why.append(f"left out the {adapter['name'].replace('(TM)', '')}: "
                           + (f"{t16(adapter) / t_cuda:.0f}x slower a layer than the main GPU" if g else "slower than the CPU"))
        pool = kept
    host_without = facts["bytes"] * (n_layer - gpu["layers"]) / n_layer + gctx * kv
    if pref == "multi" and pool_mode == "always" and not pool and amd:
        pool = [amd]
    elif (pref == "multi" and not pool and amd and amd.get("integrated") and amd.get("vram")
          and gpu["fit"] != "whole" and host_without + HEADROOM > ram):
        pool = [amd]
        why.append(f"the {amd['name'].replace('(TM)', '')} joined for room: its own "
                   f"{amd['vram'] / 1e9:.1f} GB holds layers the RAM cannot")
    elif pref == "multi" and amd and amd.get("integrated") and gpu["fit"] != "whole":
        why.append(f"left out the {amd['name'].replace('(TM)', '')}: it shares the system RAM and is no faster "
                   "than the CPU (Pool GPUs: always uses it anyway)")
    byidx = {adapter["index"]: adapter for adapter in pool}

    def secs(k, parts):
        on = k + sum(count for _, count, _ in parts)
        return k * t_cuda + sum(count * t16(byidx[index]) for index, count, _ in parts) \
            + max(0, n_layer - on) * t_cpu

    def fill(left):
        out = []
        for adapter in pool:
            if left <= 0:
                break
            room = ((adapter["vram"] - 0.5e9) if adapter.get("vram") else (ram - HEADROOM - facts["bytes"]) * 0.5) \
                if adapter.get("integrated") else adapter["vram"] - 0.5e9
            count = min(max(0, int(room // per_of(adapter))), left)
            if count > 0:
                out.append((adapter["index"], count, adapter["name"].replace("(TM)", "").replace("(R)", "")))
                left -= count
        return out

    nv_name = (g or {}).get("name", "the NVIDIA GPU") if g else ""
    alone = fill(n_layer) if pool else []
    got = sum(count for _, count, _ in alone)
    go_alone = False
    state.update(device=device, layers=layers, mid=mid, split=split, per16=per16, t_cuda=t_cuda, t_cpu=t_cpu,
                 pervk=pervk, per_of=per_of, t16=t16, pool=pool, pool_mode=pool_mode, pref=pref,
                 byidx=byidx, secs=secs, fill=fill, nv_name=nv_name, alone=alone, got=got, go_alone=go_alone)


def _choose_plan_device(state):
    facts, why, gpu, amd = state["facts"], state["why"], state["gpu"], state["amd"]
    device, layers, mid, split = state["device"], state["layers"], state["mid"], state["split"]
    pref, pool, pool_mode = state["pref"], state["pool"], state["pool_mode"]
    n_layer, fill = state["n_layer"], state["fill"]
    if pref == "multi" and gpu["fit"] == "whole":
        why.append(f"GPUs pooled only when needed: the whole model fits {state['nv_name']}, so it runs there alone")
        pref = "gpu"
    if pref == "multi" or (pref == "auto" and gpu["fit"] == "split" and pool):
        if gpu["fit"] != "none" and pool:
            k = gpu["layers"] if gpu["fit"] == "split" else max(1, n_layer // 2)
            split = fill(n_layer - k)
            if split:
                mid = sum(count for _, count, _ in split)
                device, layers = "multi", k
                at, parts = k, [f"layers 0-{k - 1} on {state['nv_name']} (CUDA)"]
                for _, count, name in split:
                    parts.append(f"{at}-{at + count - 1} on the {name} (DirectML)")
                    at += count
                cpu_n = n_layer - k - mid
                why.append("GPUs pooled: " + ", ".join(parts) + (f", {cpu_n} on the CPU" if cpu_n else "")
                           + (": your choice in Settings" if pref == "multi" else ""))
                pref = "done"
        if pref == "multi":
            why.append("pooling GPUs is not possible now: " + ("no NVIDIA GPU room for the first layers"
                                                               if gpu["fit"] == "none" else "no room on the other GPUs"))
    elif pref in ("auto", "amd") and gpu["fit"] == "none" and pool:
        state["go_alone"] = True
    if state["go_alone"]:
        whole = state["alone"]
        got = state["got"]
        if got >= n_layer:
            device, split = "amd", whole
            why.append("GPUs pooled: the whole model across " + " + ".join(f"the {name} ({count} layers)"
                                                                          for _, count, name in whole) + " (DirectML)")
            pref = "done"
        elif 0 < got < n_layer and (got * 8 >= n_layer or pref == "amd"):
            device, split, mid = "amd", whole, got
            at, parts = 0, []
            for _, count, name in whole:
                parts.append(f"layers {at}-{at + count - 1} on the {name} (DirectML)")
                at += count
            why.append("GPUs pooled: " + ", ".join(parts) + f", {n_layer - got} on the CPU: the model is larger "
                       "than the GPUs' memory")
            pref = "done"
    if pref == "done":
        pass
    elif pref == "amd":
        if amd and state["amd_fit"]:
            device = "amd"
            why.append(f"on the {amd['name'].replace('(TM)', '')}: your choice in Settings")
        else:
            why.append("on the CPU: " + ("no AMD or Intel GPU passed its check" if not amd else
                                         f"the model in fp16 needs {state['need16'] / 1e9:.1f} GB, more than the "
                                         f"{state['room16'] / 1e9:.1f} GB the {amd['name']} can hold"))
    elif pref == "cpu":
        why.append("on the CPU: your choice in Settings")
    elif pref == "auto" and gpu["fit"] == "none" and amd and state["amd_fit"] and not amd["integrated"]:
        device = "amd"
        why.append(f"on the {amd['name']}: no NVIDIA room, and a discrete AMD/Intel GPU outruns the CPU")
    elif gpu["fit"] == "none":
        why.append("on the CPU: " + ("no NVIDIA GPU found" if not (state["g"] and state["g"]["vram_total"])
                                     else "no GPU room"))
    elif pref == "hybrid":
        device, layers = "hybrid", max(1, min(gpu["layers"], n_layer // 2))
        why.append(f"CPU+GPU: your choice in Settings, {layers} of {n_layer} layers on the GPU")
    elif gpu["fit"] == "whole":
        device, layers = "gpu", n_layer
        why.append("on the GPU: every layer fits")
    elif not state["cpu_offload"]:
        why.append(f"on the CPU: only {gpu['layers']} of {n_layer} layers fit the GPU, and splitting a model "
                   "between GPU and CPU is off in Settings")
    elif gpu["layers"] * 4 >= n_layer or pref == "gpu":
        device, layers = "hybrid", gpu["layers"]
        why.append(f"CPU+GPU: {layers} of {n_layer} layers on the GPU, the rest on the CPU")
    else:
        why.append(f"on the CPU: only {gpu['layers']} of {n_layer} layers fit the GPU, too few to pay")
    ctx, slots = state["ctx"], state["slots"]
    if device != "cpu":
        ctx, slots = state["gctx"], 1
        why[:] = [item for item in why if not item.startswith("context ")]
        why.append(f"context {ctx} x 1 conversation: the GPU holds one conversation's cache")
    state.update(device=device, layers=layers, mid=mid, split=split, pref=pref, ctx=ctx, slots=slots)


def _finish_plan(state):
    facts, device, layers, mid, split = state["facts"], state["device"], state["layers"], state["mid"], state["split"]
    n_layer, ctx, kv, amd = state["n_layer"], state["ctx"], state["kv"], state["amd"]
    f_gpu = layers / n_layer
    sec = facts["bytes"] * (f_gpu / 70e9 + (1 - f_gpu) / 21e9)
    if device == "amd":
        if split:
            sec = state["secs"](0, split)
        else:
            sec = (facts["bytes"] if amd.get("backend") == "vk" else
                   (2 * facts["n_weights"] if facts.get("n_weights") else facts["bytes"] * FP16_PER_BYTE)) \
                / (((amd.get("gbps") or 0) * 1e9) or (14e9 if amd["integrated"] else 150e9))
    if device == "multi":
        sec = state["secs"](layers, split)
    est = 1 / max(1e-9, sec)
    on_gpus = {"gpu": n_layer, "hybrid": layers, "multi": layers + mid,
               "amd": sum(count for _, count, _ in split) if split else n_layer}.get(device, 0)
    n_cpu = max(0, n_layer - on_gpus)
    shared = sum(count * state["per_of"](state["byidx"][index]) for index, count, _ in split
                 if state["byidx"].get(index, {}).get("integrated") and not state["byidx"].get(index, {}).get("vram"))
    if device == "amd" and not split and amd and amd.get("integrated"):
        shared = facts["bytes"] if amd.get("backend") == "vk" else \
            (2 * facts["n_weights"] if facts.get("n_weights") else facts["bytes"] * FP16_PER_BYTE)
    host = facts["bytes"] * n_cpu / n_layer + shared + (ctx * kv if n_cpu else 0)
    ram_fit = host < state["ram"]
    if device != "cpu" and 0 < n_cpu < n_layer:
        state["why"].append(f"RAM holds {host / 1e9:.1f} GB here (the CPU's {n_cpu} of {n_layer} layers"
                            + (" and the shared-memory GPU's part" if shared else "")
                            + f"), of {state['ram'] / 1e9:.1f} GB free")
    vk_chain = len(split) > 1 and all(state["byidx"].get(index, {}).get("backend") == "vk"
                                      for index, _, _ in split)
    if device == "amd" and mid >= n_layer and not vk_chain:
        mid = 0
    elif device == "amd" and vk_chain and sum(count for _, count, _ in split) >= n_layer:
        index, count, name = split[-1]
        split[-1] = (index, count - 1, name)
        mid = n_layer - 1
    return {"device": device, "gpu_layers": layers if device in ("hybrid", "multi") else 0, "mid_layers": mid,
            "dml_split": [[index, count] for index, count, _ in split], "pref": state["pref"],
            "threads": state["threads"], "ctx": ctx, "slots": state["slots"], "gpu": state["gpu"],
            "ram_fit": ram_fit, "est_tok_s": round(est, 1), "why": state["why"]}


def plan(dev, facts, pref="auto", pool_mode="smart", gap=4.0, exclude=(), reserve_gb=0.0, cpu_offload=True):
    """Return a reproducible device plan and the reasons for each placement choice."""
    state = _plan_context(dev, facts, pref, pool_mode, gap, exclude, reserve_gb, cpu_offload)
    _plan_gpu_capacity(state)
    _plan_pooling(state)
    _choose_plan_device(state)
    return _finish_plan(state)

def device_label(dev, planned):
    """Name every device in the selected plan instead of reporting only its last backend."""
    mode = planned.get("device")
    if mode == "cpu":
        return "CPU"
    gpu = dev.get("gpu") or {}
    pool = {int(row.get("index", -1)): row.get("name") for row in dev.get("pool", [])}
    directml = [pool.get(int(row[0])) for row in planned.get("dml_split", []) if row]
    directml = list(dict.fromkeys(name for name in directml if name))
    if mode == "multi":
        names = ([gpu.get("name")] if gpu.get("name") else []) + directml
        return " + ".join(dict.fromkeys(names)) or "Multiple GPUs"
    if mode == "amd":
        if directml:
            return " + ".join(directml)
        return (dev.get("amd") or {}).get("name") or "DirectML GPU"
    if mode == "hybrid":
        return f"{gpu.get('name') or 'GPU'} + CPU"
    if mode == "gpu":
        return gpu.get("name") or "GPU"
    return mode.upper() if mode else ""


PLAN_KEYS = ("device", "gpu_layers", "mid_layers", "dml_split", "threads", "ctx", "slots", "gpu", "est_tok_s", "device_name")


# ------------------------------------------------------------------ canaries

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get today's weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "the city"}},
                   "required": ["city"]}}}
ROLE_MARK = re.compile(r"<\|[a-z_]+\|>|^(assistant|user|system)\s*[:\n]", re.I | re.M)


def _has(t, *words):
    t = t.lower()
    return all(w.lower() in t for w in words)


CANARIES = [
    ("greeting", [{"role": "user", "content": "Say hello in one short sentence."}], None,
     lambda t, c: bool(re.search(r"[A-Za-z]", t)) and len(t.split()) <= 40),
    ("fact", [{"role": "user", "content": "What is the capital of France? Answer in one word."}], None,
     lambda t, c: _has(t, "paris")),
    ("list", [{"role": "user", "content": "Name the three primary colours of paint, one per line."}], None,
     lambda t, c: sum(w in t.lower() for w in ("red", "blue", "yellow")) >= 2),
    ("arithmetic", [{"role": "user", "content": "What is 17 + 25? Reply with the number only."}], None,
     lambda t, c: "42" in t),
    ("tool call", [{"role": "user", "content": "What is the weather in Nairobi today?"}], [WEATHER],
     lambda t, c: any((x.get("function") or {}).get("name") == "get_weather"
                      and "nairobi" in json.dumps(x.get("function", {}).get("arguments", "")).lower() for x in c)),
    ("long answer", [{"role": "user", "content": "In about 80 words, explain why sleep matters."}], None,
     lambda t, c: len(t.split()) >= 35),
    ("stop", [{"role": "user", "content": "Reply with just the word OK."}], None,
     lambda t, c: "ok" in t.lower() and len(t.split()) <= 4),
    ("follow-up", [{"role": "user", "content": "My name is Amani."},
                   {"role": "assistant", "content": "Nice to meet you, Amani!"},
                   {"role": "user", "content": "What is my name? Answer in one word."}], None,
     lambda t, c: _has(t, "amani")),
]

# These short, deterministic checks are the activation gate. The remaining
# canaries and file-app orientation run after the model is usable, at low
# priority, and never make ordinary replies wait.
QUICK_CANARIES = [CANARIES[i] for i in (1, 2, 3, 4, 7)]


def run_canary(local, name, msgs, tools, check, frame="own", sampling=None, checker=None, max_tokens=None):
    """Run one deterministic model check. Checker opinions are diagnostic only."""
    body_extra = dict(sampling or {})
    max_tokens = max_tokens or CANARY_TOKENS
    t0 = time.perf_counter()
    try:
        if frame == "own":
            msg, meta = local.chat(msgs, tools=tools, max_tokens=max_tokens, cache=False, **body_extra)
        else:
            msg, meta = local.chat(msgs, tools=tools, max_tokens=max_tokens, render=False, cache=False, **body_extra)
    except Exception as e:           # the engine refused or fell over: a failed canary, with the reason
        return {"name": name, "ok": False, "why": f"engine error: {str(e)[:120]}", "ms": 0}
    text = (msg.get("content") or "").strip()
    calls = msg.get("tool_calls") or []
    finish = meta.get("finish", "")
    faults = []
    if meta.get("degenerate"):
        faults.append(f"guard: {meta['degenerate']}")
    if finish == "length":
        faults.append("never ended its turn")
    question = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    verdict = None
    # The Checker must not veto a model's useful output. Canary acceptance is
    # based on explicit task outcomes and deterministic output-integrity rules.
    from .checker import rules
    faults += rules(text, question, calls)
    passed = not faults and bool(check(text, calls))
    if not faults and not passed:
        faults.append("wrong answer")
    sml = meta.get("sml") or {}
    return {"name": name, "ok": passed, "why": "; ".join(faults), "reply": (text or json.dumps(calls))[:160],
            "ms": round((time.perf_counter() - t0) * 1000), "finish": finish,
            "gen": (meta.get("tokens") or {}).get("completion_tokens", 0), "sml": {k: sml.get(k) for k in ("reused",)},
            "judge": verdict}


# ------------------------------------------------------------------ the orchestrator

class Orchestrator:
    def __init__(self, st):
        self.st = st
        self.on = True             # always on (2026-10-20): an old "orchestrator": false is ignored
        st.cfg["orchestrator"] = True
        self.lamp = "amber"
        self.stage = ""            # what it is doing now, in words
        self.log = []              # decisions, newest last
        try:
            self.log = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines()[-200:] if l.strip()]
        except (OSError, ValueError):
            pass
        self.plan = None
        self.facts = None
        self.canaries = []         # the last suite's rows
        self.egypt = None          # the protocol's state for the loaded model, or None
        self.frame = "own"         # which chat frame the doctor settled on
        self.sampling = None       # the model's own sampling, if the doctor chose it
        self.lock = threading.Lock()
        self.gpu_jobs = threading.Condition()
        self.gpu_job = None
        self.gpu_waiters = []
        self.gpu_ticket = 0
        self.lent_from = None
        self.turns = {"n": 0, "tok": 0, "ms": 0.0, "prompt": 0, "prompt_ms": 0.0, "trips": 0}
        self.progress = 0.0        # how far the model is from ready, 0..1, for the loading bar
        self.background_checks = {
            "canaries": {"state": "not started", "progress": 0.0, "done": 0, "total": len(CANARIES)},
            "orientation": {"state": "not started", "progress": 0.0, "done": 0, "total": ORIENTATION_TASKS},
        }
        self._startup_started = None
        self._startup_elapsed = None
        self.interactive_requested = threading.Event()
        self._background_thread = None
        self._background_key = None
        self._resume_thread = None
        self.restarts = []         # when the engine died and was restarted
        self.cancelled = threading.Event()   # the person stopped this load (a 7B on a slow disk is minutes)
        self.loading = threading.Event()      # set before the background worker is scheduled
        self.load_engine_replaced = False
        # Sheep paces itself by heat: the laptop shut down from it on 2026-10-12 (thermal.py)
        from . import thermal
        self.thermal = thermal.Governor(lambda: bool(self.st.cfg.get("thermal_guard", True))).start()
        if getattr(self.st, "local", None) is not None:
            self.st.local.pace = self.thermal.pace_ms    # every chat request carries the heat guard's pace
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        """A crashed engine is noticed and restarted with the same plan (the plan's
        rule: "a crash: restart with the last good settings"). The test engine
        died after its canaries and the lamp stayed green for minutes (2026-10-06)."""
        silent = 0
        while True:
            time.sleep(5)
            st = self.st
            eng = getattr(st, "engine", None)
            if not (eng and self.on and st.model_path and self.lamp == "green" and eng.proc is not None):
                silent = 0
                continue
            if self.lock.locked():
                silent = 0
                continue
            if eng.alive():
                # alive but hung: no answer for 30 s while nothing waits on it. A GPU job beside
                # the chat engine took its memory and it stopped answering, process up (2026-10-06)
                local = getattr(st, "local", None)
                if getattr(local, "inflight", 1) == 0 and not eng.healthy():
                    silent += 1
                else:
                    silent = 0
                if silent < 6:
                    continue
                silent = 0
                self.say("the engine stopped answering; restarting it", "no reply for 30 s with nothing asked of it",
                         "fault")
                self.history("fault", note="engine hung; restarted")
                eng.stop()
                eng.proc = None
                with self.lock:
                    self._start(st.model_path, check=False)
                continue
            code = eng.proc.poll()
            last = " | ".join(l for l in eng.log[-3:] if l)[:200]
            self.restarts = [t for t in self.restarts if time.time() - t < 600] + [time.time()]
            if len(self.restarts) > 3:
                self.lamp, self.stage = "red", "the engine keeps stopping"
                self.say("the engine stopped again; not restarting it a fourth time in 10 minutes",
                         f"exit code {code}; {last}", "fault")
                self.history("fault", note=f"engine exit {code}, restarts stopped")
                continue
            self.say("the engine stopped; restarting it", f"exit code {code}; {last}", "fault")
            self.history("fault", note=f"engine exit {code}; restarted")
            eng.proc = None                 # start() starts afresh
            with self.lock:
                self._start(st.model_path, check=False)

    # -------------------------------------------------------------- log + history
    def say(self, what, why="", kind="decision"):
        e = {"t": time.time(), "what": what, "why": why, "kind": kind}
        self.log.append(e)
        self.log = self.log[-300:]
        try:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**e, "model": Path(getattr(getattr(self, "st", None), "model_path", "") or "").name}) + "\n")
        except OSError:
            pass
        print(f"orchestrator: {what}" + (f" ({why})" if why else ""), flush=True)
        return e

    def history(self, event, **detail):
        """One line in the loaded model's own log (the dashboard's model history)."""
        path = self.st.model_path
        if not path:
            return
        reg = load_registry()
        m = reg.setdefault(path, {})
        h = m.setdefault("history", [])
        h.append({"t": time.time(), "event": event, **detail})
        m["history"] = h[-400:]
        save_registry(reg)

    def entry(self, path=None):
        return load_registry().get(path or self.st.model_path, {})

    def update_entry(self, **kv):
        self.update_entry_for(self.st.model_path, **kv)

    def update_entry_for(self, path, **kv):
        reg = load_registry()
        m = reg.setdefault(path, {})
        m.update(kv)
        save_registry(reg)

    def validation_signature(self, path, plan=None):
        """Cheap content/config fingerprint for reusing validation on unchanged models.

        Reading just the ends plus file metadata avoids hashing multi-gigabyte
        checkpoints at every launch, while detecting normal edits and replacements.
        """
        model = Path(path).resolve()
        stat = model.stat()
        edge = hashlib.sha256()
        with model.open("rb") as f:
            edge.update(f.read(64 * 1024))
            if stat.st_size > 64 * 1024:
                f.seek(max(0, stat.st_size - 64 * 1024))
                edge.update(f.read(64 * 1024))
        edge_hash = edge.hexdigest()
        cached_hash = (self.entry(path).get("validation_model_hash") or {})
        if (cached_hash.get("bytes") == stat.st_size and cached_hash.get("mtime_ns") == stat.st_mtime_ns
                and cached_hash.get("ctime_ns") == stat.st_ctime_ns and cached_hash.get("file_id") == stat.st_ino
                and cached_hash.get("edge") == edge_hash and cached_hash.get("sha256")):
            model_hash = cached_hash["sha256"]
        else:
            digest = hashlib.sha256()
            with model.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            after = model.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != \
                    (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino):
                raise OSError(f"model changed while its validation fingerprint was computed: {model}")
            model_hash = digest.hexdigest()
            self.update_entry_for(path, validation_model_hash={"bytes": stat.st_size,
                                                                      "mtime_ns": stat.st_mtime_ns,
                                                                  "ctime_ns": stat.st_ctime_ns,
                                                                  "file_id": stat.st_ino,
                                                                      "edge": edge_hash,
                                                                      "sha256": model_hash})
        p = plan or self.plan or {}
        engine_name = "sml-server.exe" if os.name == "nt" else "sml-server"
        engine = Path(os.environ.get("MLMI_ENGINE_EXE", REPO / "sml" / engine_name))
        try:
            engine_stat = engine.stat()
            engine_digest = hashlib.sha256()
            with engine.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    engine_digest.update(block)
            runtime = {"path": os.path.normcase(str(engine.resolve())), "bytes": engine_stat.st_size,
                       "mtime_ns": engine_stat.st_mtime_ns, "sha256": engine_digest.hexdigest()}
        except OSError:
            runtime = {"path": os.path.normcase(str(engine)), "missing": True}
        payload = {
            "path": os.path.normcase(str(model)), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": model_hash,
            "device": p.get("device"), "device_name": p.get("device_name"),
            "ctx": p.get("ctx"), "slots": p.get("slots"), "threads": p.get("threads"),
            "gpu_layers": p.get("gpu_layers"), "mid_layers": p.get("mid_layers"),
            "frame": self.frame, "sampling": self.sampling,
            "canary_revision": CANARY_REVISION, "orientation_revision": ORIENTATION_REVISION,
            "runtime": {"os": os.name, "engine": runtime},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _adopt_legacy_checks(self, path, signature, plan):
        """Promote only older evidence already pinned to the current fingerprint."""
        entry = self.entry(path)
        canary = entry.get("canary") or {}
        rows = canary.get("rows") or []
        old_plan = entry.get("plan") or {}
        try:
            size_matches = Path(path).stat().st_size == (entry.get("facts") or {}).get("bytes")
        except OSError:
            size_matches = False
        plan_matches = all(old_plan.get(k) == (plan or {}).get(k)
                           for k in ("device", "ctx", "slots", "threads", "gpu_layers", "mid_layers"))
        if (canary.get("validation_signature") == signature
                and canary.get("passed") == len(CANARIES) and canary.get("of") == len(CANARIES)
                and len(rows) == len(CANARIES) and all(r.get("ok") for r in rows)
                and canary.get("frame", "own") == self.frame and size_matches and plan_matches):
            full = {"signature": signature, "state": "passed", "passed": len(rows), "of": len(rows),
                    "rows": rows, "t": canary.get("t"), "migrated": True}
            self.update_entry(quick_validation={**full, "of": len(QUICK_CANARIES),
                                                "passed": len(QUICK_CANARIES),
                                                "rows": [r for r in rows if r.get("name") in {x[0] for x in QUICK_CANARIES}]},
                              deep_validation=full)
            return full
        # Preserve an older orientation result only when it was pinned to the
        # current fingerprint; older unpinned results remain visible in history.
        knows = entry.get("knows_sheep") or {}
        if (knows.get("validation_signature") == signature and knows.get("of") == 10
                and len(knows.get("rows") or []) == 10 and size_matches):
            orientation_passed = knows.get("passed", 0) == 10 and all(r.get("ok") for r in knows["rows"])
            self.update_entry(orientation_validation={"signature": signature,
                                                       "state": "passed" if orientation_passed else "failed",
                                                       "passed": knows.get("passed", 0), "of": 10,
                                                       "rows": knows.get("rows", []), "migrated": True,
                                                       "t": knows.get("t")})
        return None

    @staticmethod
    def _valid_check(record, signature):
        return bool(record and record.get("signature") == signature and record.get("state") == "passed")

    @staticmethod
    def _finished_check(record, signature):
        """A completed failure is evidence too; do not rerun it on every launch."""
        return bool(record and record.get("signature") == signature
                    and record.get("state") in ("passed", "failed", "blocked"))

    @staticmethod
    def _resume_progress(record, signature, total):
        """Only reuse progress from completed rows for this exact validation signature."""
        if not record or record.get("signature") != signature:
            return None, 0
        rows = record.get("rows") or []
        try:
            recorded = int(record.get("done", len(rows)) or 0)
        except (TypeError, ValueError):
            recorded = 0
        done = min(max(0, recorded), len(rows), max(0, int(total)))
        if not done or not total:
            return None, 0
        return done / total, done

    def readiness_checks(self):
        """Small serializable startup and background-check view for the desktop."""
        startup_state = ("blocked" if not self.st.model_path else "passed" if self.ready()
                         else "running" if self.loading.is_set() else "failed")
        started = getattr(self, "_startup_started", None)
        elapsed = (max(0, round(time.monotonic() - started))
                   if started is not None and self.loading.is_set() else getattr(self, "_startup_elapsed", None))
        startup = {"id": "startup", "label": "Model load and quick exam", "state": startup_state,
                   "stage": self.stage,
                   # The startup bar has historically been a hand-set stage estimate
                   # (3%, 6%, 10%), not measured model-load progress. Report only a
                   # completed result; the UI uses an indeterminate bar while loading.
                   "progress": 1.0 if startup_state == "passed" else None,
                   "device": (self.plan or {}).get("device_name")
                   or (self.plan or {}).get("device"), "elapsed_seconds": elapsed}
        out = [startup] + [dict({"id": key, "label": label}, **self.background_checks.get(key, {}))
                           for key, label in (("canaries", "Full canaries"), ("orientation", "Model orientation"))]
        # A check that is running while the heat guard holds background work is paused, not
        # stuck: say so, with the temperatures (2026-10-14: the 7B's orientation sat at 3/13
        # reading "In progress" while the laptop was at 101 C and waiting to cool)
        heat = self.thermal.background_wait() if getattr(self, "thermal", None) else None
        for c in out:
            # every task passed is a pass (a retried task once left "failed" beside 10/10 passed)
            if c.get("state") == "failed" and c.get("passed") and c["passed"] >= (c.get("of") or c.get("total") or 1e9):
                c.update(state="passed", why="")
            if heat and c.get("state") in ("running", "queued") and c.get("id") != "startup":
                c.update(state="waiting for temperature", stage=f"paused until it cools: {heat}")
        return out

    def _background_yield_reason(self):
        busy = getattr(self.st, "busy", None)
        if busy and busy.locked():
            return "waiting for chat", "an interactive request is in progress"
        if self.interactive_requested.is_set():
            return "waiting for chat", "an interactive request is starting"
        with self.gpu_jobs:
            if self.gpu_job:
                return "waiting for GPU", f"{self.gpu_job.get('what', 'a higher-priority GPU job')} has the GPU"
            if any(int(row.get("priority", 0)) >= 10 for row in self.gpu_waiters):
                return "waiting for GPU", "a higher-priority GPU job is queued"
        # Background checks pause while the machine is warm: hours of checks at
        # full speed beside chat overheated the laptop until it shut down
        # (2026-10-12). They resume at task boundaries once it has cooled.
        heat = self.thermal.background_wait() if getattr(self, "thermal", None) else None
        if heat:
            return "waiting for temperature", heat
        return None

    def _background_should_yield(self):
        return self._background_yield_reason() is not None

    def _set_background_wait_state(self, reason):
        """Keep the two background rows in sync with the current transient wait."""
        for key in ("canaries", "orientation"):
            row = self.background_checks.get(key, {})
            state = str(row.get("state", ""))
            if reason and (state == "queued" or state.startswith("waiting for ")):
                row.update(state=reason[0], why=reason[1])
            elif not reason and state.startswith("waiting for "):
                row["state"] = "queued"
                row.pop("why", None)

    # -------------------------------------------------------------- the switch
    def set_on(self, on):
        self.on = bool(on)
        self.st.cfg["orchestrator"] = self.on
        self.say("switched " + ("on" if on else "off"), "by the person, in Settings", "switch")
        if on:
            self.start_model(self.st.model_path)
        else:
            self.lamp, self.stage = "off", ""

    # -------------------------------------------------------------- load
    def freed(self, dev):
        """This machine as if the running engine were stopped: a plan for a new
        model (or a level card) may use the RAM and VRAM the old one holds.
        Windows reports no VRAM per process, so the engine's own is measured at
        its start (VRAM after, less VRAM before)."""
        eng = self.st.engine
        if not eng.alive():
            return dev
        try:
            import psutil
            held = psutil.Process(eng.proc.pid).memory_info().rss
            dev = dict(dev, ram=dict(dev["ram"], available=dev["ram"]["available"] + held))
        except Exception:
            pass
        g = dev.get("gpu")
        if g and getattr(self, "engine_vram", 0):
            dev = dict(dev, gpu=dict(g, vram_used=max(0.0, g["vram_used"] - self.engine_vram)))
        return dev

    def start_model(self, path, background=True, on_ready=None):
        if background:
            self.cancelled.clear()
            self.loading.set()
            self._startup_started = time.monotonic()
            self._startup_elapsed = None
            self.load_engine_replaced = False
            threading.Thread(target=self.start_model, args=(path, False, on_ready), daemon=True).start()
            return
        if not self.loading.is_set():
            self.cancelled.clear()
            self.loading.set()
            self._startup_started = time.monotonic()
            self._startup_elapsed = None
            self.load_engine_replaced = False
        try:
            with self.lock:
                ok = False if self._stop_if_cancelled() else self._start(path)
                if on_ready:
                    try:
                        on_ready(ok)
                    except Exception:
                        pass
        finally:
            if self._startup_started is not None:
                self._startup_elapsed = max(0, round(time.monotonic() - self._startup_started))
            self._startup_started = None
            self.loading.clear()

    def cancel_load(self):
        """Stop the load in progress and leave the machine as it was. A big model on
        a slow disk takes minutes, and the person who picked the wrong one had no way
        out but to quit Sheep. Each stage of _start checks this between steps."""
        if not self.loading.is_set():
            return {"cancelled": False, "why": "nothing is loading"}
        self.cancelled.set()
        self.stage = "stopping the load"
        return {"cancelled": True}

    def _stop_if_cancelled(self):
        """True when the person cancelled: the engine is stopped and the lamp says so."""
        if not self.cancelled.is_set():
            return False
        eng = self.st.engine
        if self.load_engine_replaced:
            eng.stop()
        previous = getattr(eng, "model", "") if not self.load_engine_replaced and eng.alive() else ""
        self.st.model_path = previous if previous and Path(previous).is_file() else ""
        self.lamp = "green" if self.st.model_path else ("amber" if self.on else "off")
        self.stage = "load stopped; previous model kept running" if self.st.model_path else "load stopped; no model is running"
        self.progress = 1.0 if self.st.model_path else 0.0
        self.say("the load was stopped", "the person cancelled it", "decision")
        self.cancelled.clear()
        self.load_engine_replaced = False
        if hasattr(self.st, "save"):
            self.st.save()
        return True

    # -------------------------------------------------------------- memory other jobs need
    def reserved(self):
        """Bytes of RAM a running Sheep Quantizer needs beside the chat engine. The
        chat engine planned a 3.8 GB cache while the 7B was being quantized and left
        the machine 1 GB (2026-10-06). The quantizer's own size now, plus room to grow."""
        q = getattr(self.st, "quant", None)
        if not q or q.state not in ("fetching", "converting", "fitting", "building"):
            return 0
        held = 0
        try:
            import psutil
            if q.proc and q.proc.poll() is None:
                held = psutil.Process(q.proc.pid).memory_info().rss
        except Exception:
            pass
        return max(held + 2 * 2**30, 4 * 2**30)

    def replan_for_job(self):
        """A quantize just started: a chat engine on the CPU is planned again around it
        (smaller cache), without the canaries (same model, same file)."""
        if (self.plan or {}).get("device") not in ("cpu", "hybrid") or not self.st.model_path:
            return

        def go():
            time.sleep(20)                     # the quantizer's process is up and has its size
            with self.lock:
                self._start(self.st.model_path, check=False)
        threading.Thread(target=go, daemon=True).start()

    # -------------------------------------------------------------- how this model works best
    def write_mode(self, kind):
        """True when this model makes decks, documents and sheets best by writing
        the content and letting Sheep make the file (the default for every local
        model); False when its orientation run showed its own tool calls do better."""
        prof = (self.entry(self.st.model_path).get("profile") or {}) if self.st.model_path else {}
        return prof.get(kind, "write") == "write"

    # -------------------------------------------------------------- lending the GPU to media
    def lend_gpu(self, what, need_gb, *, wait=False, priority=0, timeout=None, cancel_event=None):
        """Lease GPU-heavy work exclusively; pause Chat on CPU when it owns a GPU.

        Higher priority jobs pass queued background jobs. A lease is returned by
        give_back_gpu(), which restores Chat before waking the next waiter.
        """
        st = self.st
        heat = getattr(self, "thermal", None)
        person = priority >= 10                 # the person's own pictures and video: only the danger line holds them
        if heat and heat.gpu_wait(person):
            # no new GPU work on a machine this hot: wait for it to cool, or decline
            if not wait:
                self.say(f"held {what}", heat.gpu_wait(person), "plan")
                return False
            self.say(f"holding {what} until this computer cools", heat.gpu_wait(person), "plan")
            t0 = time.monotonic()
            with heat.changed:
                while heat.gpu_wait(person):
                    if cancel_event is not None and cancel_event.is_set():
                        return False
                    if timeout is not None and time.monotonic() - t0 >= timeout:
                        return False
                    heat.changed.wait(timeout=5)
        with self.gpu_jobs:
            self.gpu_ticket += 1
            ticket = self.gpu_ticket
            waiter = {"ticket": ticket, "priority": int(priority), "what": str(what), "queued": time.time()}
            self.gpu_waiters.append(waiter)
            self.gpu_waiters.sort(key=lambda row: (-row["priority"], row["ticket"]))
            deadline = time.monotonic() + timeout if timeout is not None else None
            while self.gpu_job is not None or not self.gpu_waiters or self.gpu_waiters[0]["ticket"] != ticket:
                if cancel_event is not None and cancel_event.is_set():
                    self.gpu_waiters = [row for row in self.gpu_waiters if row["ticket"] != ticket]
                    self.gpu_jobs.notify_all()
                    return False
                if not wait:
                    self.gpu_waiters = [row for row in self.gpu_waiters if row["ticket"] != ticket]
                    self.gpu_jobs.notify_all()
                    return False
                remaining = deadline - time.monotonic() if deadline is not None else None
                if remaining is not None and remaining <= 0:
                    self.gpu_waiters = [row for row in self.gpu_waiters if row["ticket"] != ticket]
                    self.gpu_jobs.notify_all()
                    return False
                self.gpu_jobs.wait(min(remaining, 5) if remaining is not None else 5)
            if cancel_event is not None and cancel_event.is_set():
                self.gpu_waiters = [row for row in self.gpu_waiters if row["ticket"] != ticket]
                self.gpu_jobs.notify_all()
                return False
            self.gpu_waiters.pop(0)
            self.gpu_job = {"what": str(what), "need_gb": float(need_gb), "priority": int(priority),
                            "started": time.time()}

        if (self.plan or {}).get("device") not in ("gpu", "hybrid", "multi", "amd") or not st.model_path:
            self.lent_from = None
            self.say(f"reserved the GPU for {what}", f"exclusive local compute lease; about {need_gb:.1f} GB requested", "plan")
            return True
        try:
            with self.lock:
                self.lent_from = st.cfg.get("device", "auto")
                st.cfg["device"] = "cpu"
                self.say(f"lent the GPU to {what}",
                         f"it needs about {need_gb:.1f} GB; Chat answers from the CPU while this job runs", "plan")
                if not self._start(st.model_path, check=False):
                    st.cfg["device"], self.lent_from = self.lent_from, None
                    self._release_gpu_job()
                    return False
        except Exception:
            if self.lent_from is not None:
                st.cfg["device"], self.lent_from = self.lent_from, None
            self._release_gpu_job()
            raise
        return True

    def _release_gpu_job(self):
        with self.gpu_jobs:
            self.gpu_job = None
            self.gpu_jobs.notify_all()

    def give_back_gpu(self):
        st = self.st
        if getattr(self, "gpu_job", None) is None:
            return False
        if getattr(self, "lent_from", None) is None:
            self._release_gpu_job()
            return True
        st.cfg["device"], self.lent_from = self.lent_from, None
        self.say("the GPU is back with Chat", "the higher-priority GPU job finished", "plan")

        def back():
            try:
                with self.lock:
                    self._start(st.model_path, check=False)
            finally:
                self._release_gpu_job()
        threading.Thread(target=back, daemon=True).start()
        return True

    @staticmethod
    def _fallback_preferences(device):
        if device in ("gpu", "hybrid", "multi"):
            return ("amd", "cpu")
        if device == "amd":
            return ("cpu",)
        return ()

    def retry_after_device_error(self, error):
        """For Auto only, restart local inference on the next capable backend."""
        st = self.st
        if not self.on or st.cfg.get("device", "auto") != "auto" or self.lock.locked() or not st.model_path:
            return None
        device = (self.plan or {}).get("device", "cpu")
        text = str(error or "").lower()
        device_fault = any(s in text for s in (
            "cuda", "directml", "gpu", "device", "out of memory", "allocation", "driver",
            "connection refused", "connection reset", "worker exited", "engine died", "timed out"))
        if not device_fault and st.engine.alive():
            return None
        preferences = self._fallback_preferences(device)
        if not preferences:
            return None
        with self.lock:
            for pref in preferences:
                try:
                    started = self._start(st.model_path, check=False, pref_override=pref)
                except Exception as exc:
                    self.say("local hardware fallback did not start", str(exc)[:220], "fault")
                    continue
                if not started:
                    continue
                used = (self.plan or {}).get("device", "cpu")
                if used == device or not st.engine.alive():
                    continue
                note = f"{device.upper()} failed during local inference; Auto retried on {used.upper()}"
                self.plan.setdefault("why", []).append(note)
                self.update_entry(plan={k: self.plan.get(k) for k in PLAN_KEYS})
                self.say(note, str(error)[:220], "plan")
                return note
        return None

    def _verify_plan_devices(self, path, dev, p, make_plan):
        # Each GPU route is checked once before any model reply trusts its output.
        if p["device"] in ("gpu", "hybrid", "multi") and not self.facts.get("gpu_fault"):
            stamp = engine_stamp()
            chk = self.entry(path).get("gpu_check") or {}
            if chk.get("engine") != stamp:
                self.stage, self.progress = "checking that the GPU computes this model as the CPU does", 0.08
                res = gpu_agreement(path)
                if res is not None:
                    self.update_entry_for(path, gpu_check=dict(res, engine=stamp, t=time.time()))
                    chk = res
            if agreement_ok(chk) is False:
                why = (f"the NVIDIA card and the CPU disagree on this model (logit corr {chk.get('corr')}, "
                       f"top word {chk.get('agree')}/{chk.get('of')})")
                fault = add_fault(self.facts.get("gpu_fault"), "gpu")
                self.update_entry_for(path, gpu_fault={"device": fault, "engine": stamp, "t": time.time(), "why": why})
                self.facts["gpu_fault"] = fault
                self.say("this model computes wrongly on the NVIDIA card; it runs without it", why, "plan")
                p = make_plan()
        # The Vulkan path checks its first layer against CPU once per model, card, and engine build.
        if p["device"] in ("amd", "multi") and self.facts.get("gpu_fault") not in ("amd", "multi"):
            from . import hardware as _hw
            if _hw.use_vk():
                stamp = engine_stamp()
                idx = (p.get("dml_split") or [[None]])[0][0]
                if idx is None:
                    idx = (dev.get("amd") or {}).get("index")
                vchk = self.entry(path).get("vk_check") or {}
                if idx is not None and (vchk.get("engine") != stamp or vchk.get("index") != idx):
                    self.stage, self.progress = "checking that the other GPU computes this model as the CPU does", 0.09
                    vchk = dict(parse_gpu_agreement(_hw.vk_agreement(path, idx)), engine=stamp, index=idx, t=time.time())
                    self.update_entry_for(path, vk_check=vchk)
                v_ok = agreement_ok(vchk) if vchk.get("corr") is not None else vchk.get("ok")
                if idx is not None and v_ok is not True:
                    why = (f"the other GPU and the CPU disagree on this model (logit corr {vchk.get('corr')}, "
                           f"top word {vchk.get('agree')}/{vchk.get('of')})" if v_ok is False
                           else f"the other GPU cannot run this model: {vchk.get('why') or 'its check did not run'}")
                    fault = add_fault(self.facts.get("gpu_fault"), "amd")
                    self.update_entry_for(path, gpu_fault={"device": fault, "engine": stamp, "t": time.time(), "why": why})
                    self.facts["gpu_fault"] = fault
                    self.say("the other GPU is left out for this model", why, "plan")
                    p = make_plan()
        return p

    def _record_plan(self, path, p, dev, held):
        p["device_name"] = device_label(dev, p)
        if held:
            p["why"].append(f"{held / 2**30:.1f} GB kept free for the Sheep Quantizer, which is running")
        prev = self.entry(path).get("plan", {})
        self.plan = p
        self.say(f"plan: {p['device'].upper()}, {p['threads']} threads, context {p['ctx']} x {p['slots']}",
                 "; ".join(p["why"]), "plan")
        if not p["ram_fit"]:
            self.say("this model does not fit the free RAM", "offer a smaller build", "fault")
        e = self.entry(path)
        if not e.get("added"):
            self.update_entry(added=time.time(), name=Path(path).name, facts=self.facts)
            self.history("added", note=f"{self.facts['arch']}, {self.facts['n_layer']} layers, "
                                       f"{self.facts['bytes'] / 1e9:.2f} GB")
        self.update_entry(plan={k: p.get(k) for k in PLAN_KEYS}, facts=self.facts)
        return e, prev

    def _start(self, path, check=True, pref_override=None):
        st = self.st
        if self._stop_if_cancelled():
            return False
        if not path:
            self.lamp, self.stage = ("amber" if self.on else "off"), "no model plugged in"
            return
        if not self.on:
            self.lamp, self.stage = "off", "engine on fixed defaults"
            ok = st.engine.start(path)
            self.say("engine started on fixed defaults" if ok else "engine did not start", "orchestrator off")
            return
        self.lamp, self.stage, self.progress = "amber", "probing this machine", 0.03
        dev = probe(0)
        try:
            self.facts = model_facts(path)
            e_ = self.entry(path)
            fault = e_.get("gpu_fault") or {}
            if fault.get("device") and fault.get("engine") == engine_stamp():
                kind = fault["device"]
                # the agreement checks are re-read under the current rule: an older, stricter
                # one faulted the 3B on its own card at corr 0.9997
                nv_ok = agreement_ok(e_.get("gpu_check") or {}) is True
                vk_ok = agreement_ok(e_.get("vk_check") or {}) is True
                if str(fault.get("why", "")).startswith(("the NVIDIA card and the CPU disagree",
                                                         "the other GPU and the CPU disagree")):
                    if nv_ok and kind in ("gpu", "hybrid"):
                        kind = None
                    elif vk_ok and kind == "amd":
                        kind = None
                    elif kind == "multi":
                        kind = None if (nv_ok and vk_ok) else ("amd" if nv_ok else ("gpu" if vk_ok else kind))
                if kind:
                    self.facts["gpu_fault"] = kind
                else:
                    self.update_entry_for(path, gpu_fault=None)
            trial = (self.entry(path).get("speed_trials") or {}).get(hardware_sig(dev)) or {}
            if trial.get("best") and trial.get("engine") == engine_stamp() and trial.get("hw") == hardware_sig(dev):
                self.facts["measured_best"] = trial["best"]
                self.facts["measured_note"] = ", ".join(f"{k} {v} tok/s" for k, v in (trial.get("tok_s") or {}).items())
        except Exception as e:
            self.lamp, self.stage = "red", f"cannot read the model file: {e}"
            self.say("refused the model", str(e), "fault")
            return
        if self._stop_if_cancelled():
            return
        if self.facts.get("compat"):
            return self._start_compat(path, None, check)
        if self.facts.get("measured_best") == "compat" and pref_override is None:
            from . import gpu_runtime
            orig = originals_for(self.entry(path))
            if orig and gpu_runtime.ready("cuda"):
                return self._start_compat(path, dict(orig, kind="cuda"), check)
        st.engine.compat = None
        self.stage, self.progress = "planning", 0.06
        dev = self.freed(dev)          # the engine about to be replaced holds memory the new plan may use
        held = self.reserved()
        if held:                       # a running quantize keeps its memory: the plan takes what is left
            dev = dict(dev, ram=dict(dev["ram"], available=max(0.0, dev["ram"]["available"] - held)))
        pref = pref_override if pref_override is not None else st.cfg.get("device", "auto")

        def make_plan():
            return plan(dev, self.facts, pref, st.cfg.get("pool_mode", "smart"), float(st.cfg.get("pool_gap", 4) or 4),
                        exclude=st.cfg.get("gpus_off") or [], reserve_gb=float(st.cfg.get("vram_reserve_gb", 0) or 0),
                        cpu_offload=bool(st.cfg.get("cpu_offload", True)))
        p = make_plan()
        p = self._verify_plan_devices(path, dev, p, make_plan)
        e, prev = self._record_plan(path, p, dev, held)
        if self._stop_if_cancelled():
            return
        self.stage, self.progress = "starting the engine", 0.1
        eng = st.engine
        need_restart = (eng.model != path or not eng.alive() or eng.ctx != p["ctx"] or eng.slots != p["slots"]
                        or getattr(eng, "compat_running", False)
                        or eng.threads != p["threads"] or eng.device != p["device"] or eng.gpu_layers != p["gpu_layers"])
        ext = ""
        if p["device"] == "amd":
            from . import hardware
            ext = hardware.ext_cmd(path, p["ctx"], (0, p["mid_layers"]) if p.get("mid_layers") else None,
                                   p.get("dml_split") or None)
        elif p["device"] == "multi":
            from . import hardware
            ext = hardware.ext_cmd(path, p["ctx"], (p["gpu_layers"], p["gpu_layers"] + p["mid_layers"]),
                                   p.get("dml_split") or None)
        tune = self.tuned_env(path, p["device"], dev)
        need_restart = need_restart or eng.ext_cmd != ext or getattr(eng, "tune_env", {}) != tune
        if need_restart:
            g0 = _nvidia()
            self.load_engine_replaced = True
            eng.stop()
            time.sleep(0.5)
            g0 = _nvidia() or g0           # VRAM with no engine: the baseline for what it takes
            eng.ctx, eng.slots, eng.threads = p["ctx"], p["slots"], p["threads"]
            eng.device, eng.gpu_layers, eng.ext_cmd = p["device"], p["gpu_layers"], ext
            eng.mid_layers = p.get("mid_layers", 0)
            eng.compat, eng.compat_running = None, False
            eng.tune_env = tune
            t0 = time.time()
            ok = eng.start(path)
            if not ok:
                if pref_override is None and st.cfg.get("device", "auto") == "auto":
                    failed_device = p["device"]
                    for fallback_pref in self._fallback_preferences(failed_device):
                        self.say(f"{failed_device.upper()} did not start; trying the next local device",
                                 "Auto keeps inference local", "plan")
                        try:
                            fallback_started = self._start(path, check=check, pref_override=fallback_pref)
                        except Exception as exc:
                            self.say("local hardware fallback did not start", str(exc)[:220], "fault")
                            fallback_started = False
                        if fallback_started:
                            note = f"{failed_device.upper()} did not start; Auto continued on {self.plan['device'].upper()}"
                            self.plan.setdefault("why", []).append(note)
                            self.update_entry(plan={k: self.plan.get(k) for k in PLAN_KEYS})
                            self.say(note, "local device fallback", "plan")
                            return True
                self.lamp, self.stage = "red", "the engine did not start"
                self.say("the engine did not start", " | ".join(eng.log[-3:]), "fault")
                self.history("fault", note="the engine did not start")
                return False
            g1 = _nvidia()
            self.engine_vram = max(0.0, (g1 or {}).get("vram_used", 0) - (g0 or {}).get("vram_used", 0)) if g0 else 0.0
            self.history("loaded", ms=round((time.time() - t0) * 1000),
                         plan=f"{p['device']} {p['threads']}t ctx {p['ctx']}x{p['slots']}",
                         changed=bool(prev) and prev.get("ctx") != p["ctx"])
        return self._reconcile_engine_plan(path, p, e, dev, check)

    def _reconcile_engine_plan(self, path, p, e, dev, check):
        st = self.st
        st.local._chat = None           # re-read the file's chat block
        st.local.ctx = p["ctx"]
        # the plan is what was asked for; the engine reports what it could hold.
        es = st.local.stats() or {}
        got = es.get("slots")
        if got and got < p["slots"]:
            p["why"].append(f"the engine could hold {got} of {p['slots']} conversations in memory; using {got}")
            self.say(f"the engine holds {got} conversation{'s' if got > 1 else ''}, not {p['slots']}", "memory at start", "plan")
            p["slots"] = got
        ran = es.get("device")
        if ran and (ran != p["device"] or (ran == "hybrid" and es.get("gpu_layers") != p["gpu_layers"])):
            why = es.get("device_why") or f"{es.get('gpu_layers')} of {es.get('gpu_of')} layers fit at load"
            p["why"].append(f"the engine runs on {ran.upper()}, not {p['device'].upper()}: {why}")
            self.say(f"running on {ran.upper()} instead of {p['device'].upper()}", why, "plan")
            p["device"], p["gpu_layers"] = ran, es.get("gpu_layers") or 0
        if ran in ("gpu", "hybrid", "multi"):
            p["gpu_of"] = es.get("gpu_of")
            p["gpu_layers"] = es.get("gpu_layers") or p["gpu_layers"]
        if p.get("device") in ("multi", "amd", "hybrid"):
            p["device_name"] = device_label(dev, p)
        elif es.get("device_name"):
            p["device_name"] = es["device_name"]
        self.update_entry(plan={k: p.get(k) for k in PLAN_KEYS})
        return self._finish_start(path, p, e, check)

    def _finish_start(self, path, p, e, check):
        """The model's checks once its engine runs: the quick exam, then the background ones."""
        fix = e.get("fix") or {}
        self.frame = fix.get("frame", "own")
        self.sampling = fix.get("sampling")
        if self._stop_if_cancelled():        # the canaries and orientation are the long part
            return
        if check:
            signature = self.validation_signature(path, p)
            entry = self.entry(path)
            if not self._valid_check(entry.get("deep_validation"), signature):
                self._adopt_legacy_checks(path, signature, p)
                entry = self.entry(path)
            quick = entry.get("quick_validation") or {}
            deep = entry.get("deep_validation") or {}
            orientation = entry.get("orientation_validation") or {}
            if (self._finished_check(quick, signature) and quick.get("state") == "failed"
                    and not self._valid_check(deep, signature)):
                self.canaries = quick.get("rows") or self.canaries
                self.lamp, self.stage, self.progress = "red", "quick model exam failed (cached)", 1.0
                self.say("kept the model out of service after its quick exam failed",
                         quick.get("why") or "the saved validation result matches this model, runtime, and device",
                         "fault")
                return False
            if self._finished_check(deep, signature):
                self.canaries = deep.get("rows") or self.canaries
                self.background_checks["canaries"] = {"state": deep.get("state", "passed"), "progress": 1.0,
                                                       "done": deep.get("of", len(CANARIES)),
                                                        "total": deep.get("of", len(CANARIES)), "cached": True,
                                                        "elapsed_seconds": deep.get("elapsed_seconds"),
                                                        "why": deep.get("why"),
                                                        "device": (p or {}).get("device_name") or (p or {}).get("device")}
            else:
                self.background_checks["canaries"] = {"state": "queued", "progress": 0.0,
                                                       "done": 0, "total": len(CANARIES),
                                                       "device": (p or {}).get("device_name") or (p or {}).get("device")}
            if self._finished_check(orientation, signature):
                self.background_checks["orientation"] = {"state": orientation.get("state", "passed"), "progress": 1.0,
                                                           "done": orientation.get("passed", 10),
                                                            "total": orientation.get("of", 10), "cached": True,
                                                            "elapsed_seconds": orientation.get("elapsed_seconds"),
                                                            "why": orientation.get("why"),
                                                            "device": (p or {}).get("device_name") or (p or {}).get("device")}
            else:
                self.background_checks["orientation"] = {"state": "queued", "progress": 0.0,
                                                          "done": 0, "total": ORIENTATION_TASKS,
                                                          "device": (p or {}).get("device_name") or (p or {}).get("device")}

            if self._valid_check(quick, signature) or self._valid_check(deep, signature):
                self.canaries = (deep.get("rows") or quick.get("rows") or self.canaries)
                self.lamp, self.stage, self.progress = "green", "ready; background checks continue", 1.0
            else:
                self.background_checks["canaries"] = {"state": "waiting for quick exam", "progress": 0.0,
                                                       "done": 0, "total": len(QUICK_CANARIES),
                                                       "device": (p or {}).get("device_name") or (p or {}).get("device")}
                quick_ok = self.canary_suite("quick", QUICK_CANARIES, signature=signature)
                if not quick_ok and not self.ready():
                    return False
                # If the doctor selected another compatible frame or sampling,
                # cache under the resulting settings rather than the failed ones.
                signature = self.validation_signature(path, self.plan)
                self.update_entry(quick_validation={"signature": signature, "state": "passed",
                                                    "passed": sum(r.get("ok") for r in self.canaries),
                                                    "of": len(QUICK_CANARIES), "rows": self.canaries,
                                                    "t": time.time()})
                self.lamp, self.stage, self.progress = "green", "ready; background checks queued", 1.0
            self._queue_background_checks(path, signature)
        else:
            self.lamp, self.stage, self.progress = "green", "in sync", 1.0
        return True

    def upgrade_compat(self, path):
        """Part C4: a compatibility-mode model whose type Sheep's engine now runs (a new fit, a newer
        build) is rebuilt natively in the background, and used once its build passes its checks.
        Returns True when a rebuild was started."""
        try:
            c = json.loads(Path(path).read_text(encoding="utf-8"))
            src = Path(c["src"])
            cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
            if str(TOOLING) not in sys.path:
                sys.path.insert(0, str(TOOLING))
            import sml_archs
            import sml_hf
            row = sml_archs.check(sml_hf.model_type_of(cfg), cfg)
        except Exception:
            return False
        q = getattr(self.st, "quant", None)
        if not row or row[0] != "ok" or q is None or q.state in ("fetching", "converting", "fitting", "building"):
            return False
        if self.st.model_path != path:
            return False
        self.say("rebuilding a compatibility-mode model in Sheep's own engine",
                 f"{src.name}: the engine now runs this model type; it switches once the build passes its checks")
        self.history("upgrade: started", source=str(src))

        def done(res):
            if self.st.model_path == path:
                self.st.model_path = res["path"]
                self.st.save()
                self.start_model(res["path"])
        q.start(str(src), name=Path(path).name.replace(".compat.json", ""), on_done=done)
        return True

    def _start_compat(self, path, compat, check=True, trial=False):
        """Compatibility mode: the maker's own code (compat_server.py). compat None: a
        .compat.json; else a native model's originals, which its speed trial found faster
        on the NVIDIA card. Same canaries, same lamp."""
        st = self.st
        eng = st.engine
        e = self.entry(path)
        if compat is None and not trial:
            up = threading.Timer(120, self.upgrade_compat, args=(path,))
            up.daemon = True
            up.start()
        n_ctx = int((self.facts or {}).get("n_ctx_train") or 8192)
        p = {"device": "compat", "gpu_layers": 0, "mid_layers": 0, "dml_split": None, "threads": eng.threads,
             "ctx": min(8192, n_ctx) if n_ctx else 8192, "slots": 1, "gpu": None, "est_tok_s": None, "ram_fit": True,
             "why": ["compatibility mode: the maker's own code on the original weights" if compat is None else
                     "timing the maker's own code on the NVIDIA card (speed trial)" if trial else
                     "the maker's own code on the NVIDIA card measured faster than Sheep's file on the CPU"]}
        self.plan = p
        self.stage, self.progress = "starting compatibility mode (the maker's own code)", 0.1
        self.load_engine_replaced = True
        eng.stop()
        eng.compat = compat
        eng.ctx, eng.slots = p["ctx"], 1
        t0 = time.time()
        ok = eng.start(path)
        eng.compat_running = bool(ok)
        if not ok:
            eng.compat = None
            self.lamp, self.stage = "red", "compatibility mode did not start"
            self.say("compatibility mode did not start", " | ".join(eng.log[-3:]), "fault")
            self.history("fault", note="compatibility mode did not start")
            return False
        self.history("loaded", ms=round((time.time() - t0) * 1000), plan="compat")
        st.local._chat = None
        st.local.ctx = p["ctx"]
        es = st.local.stats() or {}
        p["device_name"] = f"compatibility mode, {es.get('device_name') or 'CPU'}"
        self.say(f"running in compatibility mode on {es.get('device_name') or 'the CPU'}", p["why"][0], "plan")
        if not e.get("added"):
            self.update_entry(added=time.time(), name=Path(path).name, facts=self.facts)
        self.update_entry(plan={k: p.get(k) for k in PLAN_KEYS}, facts=self.facts)
        return self._finish_start(path, p, e, check)

    # -------------------------------------------------------------- orientation
    def orient(self, background=False, signature=None):
        """Before a new model is first used, it does real jobs in a private
        workspace (making and editing in each file app, a file job, a question,
        code, a follow-up, a picture, a video, a recording), graded by opening
        what it made (orient.py). It runs
        after activation and yields between tasks when the user needs the model."""
        from . import orient as O
        from .paths import HOME
        signature = signature or self.validation_signature(self.st.model_path)
        model_path = self.st.model_path
        root = HOME / "orientation" / Path(model_path).stem / signature[:16]
        drop_old_checks(root)
        tasks = O.quick()
        saved = (self.entry().get("orientation_validation") or {}) if background else {}
        rows_saved = saved.get("rows", []) if saved.get("signature") == signature else []
        started_clock = time.monotonic()
        device = (self.plan or {}).get("device_name") or (self.plan or {}).get("device")
        durations_seen = [float(r.get("s") or 0) for r in rows_saved if r.get("s")]
        if background:
            self.background_checks["orientation"] = {"state": "running", "progress": len(rows_saved) / len(tasks),
                                                       "done": len(rows_saved), "total": len(tasks),
                                                       "elapsed_seconds": 0, "device": device}
        else:
            self.lamp = "amber"

        def prog(i, n, t):
            if background:
                eta = round(sum(durations_seen) / len(durations_seen) * max(0, n - i)) if len(durations_seen) >= 2 else None
                self.background_checks["orientation"] = {"state": "running", "progress": i / max(n, 1),
                                                           "done": i, "total": n,
                                                           "stage": f"orientation task {i + 1} of {n}",
                                                           "elapsed_seconds": max(0, round(time.monotonic() - started_clock)),
                                                           "eta_seconds": eta, "device": device}
            else:
                self.stage = f"orienting the model: job {i + 1} of {n}"
                self.progress = 0.6 + 0.39 * i / n

        def save_orientation(i, row, rows):
            passed = sum(bool(r.get("ok")) for r in rows)
            if row.get("s"):
                durations_seen.append(float(row["s"]))
            self.update_entry_for(model_path, orientation_validation={"signature": signature, "state": "running",
                                                                       "progress": len(rows) / len(tasks), "done": len(rows),
                                                                       "total": len(tasks), "passed": passed, "rows": rows,
                                                                        "elapsed_seconds": round(time.monotonic() - started_clock),
                                                                       "t": time.time()})
            self.background_checks["orientation"] = {"state": "running", "progress": len(rows) / len(tasks),
                                                       "done": len(rows), "total": len(tasks),
                                                       "elapsed_seconds": max(0, round(time.monotonic() - started_clock)),
                                                       "device": device}

        def orientation_can_run():
            reason = self._background_yield_reason()
            if reason:
                self.background_checks["orientation"] = {
                    **self.background_checks.get("orientation", {}),
                    "state": reason[0], "why": reason[1],
                    "elapsed_seconds": max(0, round(time.monotonic() - started_clock)), "device": device,
                }
                return False
            return True

        t0 = time.time()
        rows = O.run_inprocess(self.st, tasks, root, prog, existing_rows=rows_saved,
                               save_row=save_orientation if background else None,
                               should_yield=orientation_can_run if background else None,
                               should_stop=(lambda: self.st.model_path != model_path) if background else None)
        if background and self.st.model_path != model_path:
            self.background_checks["orientation"] = {"state": "deferred", "progress": len(rows) / len(tasks),
                                                        "done": len(rows), "total": len(tasks),
                                                        "elapsed_seconds": round(time.monotonic() - started_clock),
                                                        "device": device,
                                                        "why": "the active model changed; its own validation is queued"}
            return rows
        if background and len(rows) < len(tasks):
            reason = self._background_yield_reason() or ("waiting for chat", "interactive work has priority")
            self.background_checks["orientation"] = {
                "state": reason[0], "progress": len(rows) / max(1, len(tasks)),
                "done": len(rows), "total": len(tasks),
                "elapsed_seconds": round(time.monotonic() - started_clock), "device": device,
                "why": reason[1],
            }
            return rows
        # one sampled second try per missed task: a small model's single slip is not a fault (2026-10-20)
        missed = [t for t in tasks if any(r["id"] == t.id and not r.get("ok") for r in rows)]
        if missed:
            again = {r["id"]: r for r in O.run_inprocess(self.st, missed, root, prog,
                                                          sampling={"temperature": 0.6, "top_p": 0.9},
                                                          should_yield=orientation_can_run if background else None)}
            rows = [dict(again[r["id"]], tries=2) if again.get(r["id"], {}).get("ok")
                    else dict(r, tries=2) if r["id"] in again else r for r in rows]
        passed = sum(r["ok"] for r in rows)
        fails = [f"{r['turns'][0][:40]}: {r['why']}" for r in rows if not r["ok"]]
        self.update_entry_for(model_path, knows_sheep={"validation_signature": signature,
                                                       "t": time.time(), "passed": passed, "of": len(rows), "rows": rows,
                                                       "hints": O.hints(rows), "s": round(time.time() - t0)},
                              orientation_validation={"signature": signature, "state": "passed" if passed == len(rows) else "failed",
                                                      "progress": 1.0, "done": len(rows), "total": len(rows),
                                                      "passed": passed, "of": len(rows), "rows": rows,
                                                      "elapsed_seconds": round(time.monotonic() - started_clock),
                                                      "why": "; ".join(fails)[:400],
                                                      "t": time.time(), "s": round(time.time() - t0)})
        self.say(f"oriented: {passed} of {len(rows)} jobs done right", "; ".join(fails)[:400] or "all clean", "orient")
        self.history("oriented", passed=passed, of=len(rows), s=round(time.time() - t0))
        self.background_checks["orientation"] = {"state": "passed" if passed == len(rows) else "failed",
                                                   "progress": 1.0, "done": len(rows), "total": len(rows),
                                                   "passed": passed, "why": "; ".join(fails)[:400],
                                                   "elapsed_seconds": round(time.monotonic() - started_clock),
                                                   "device": device}
        if not background:
            self.lamp, self.stage, self.progress = "green", "in sync", 1.0
        return rows

    # -------------------------------------------------------------- the canaries (Egypt step 1)
    def canary_suite(self, reason="load", items=None, signature=None):
        st = self.st
        items = list(items or CANARIES)
        background = reason == "background"
        quick = reason == "quick"
        path = st.model_path
        signature = signature or self.validation_signature(st.model_path)
        started_clock = time.monotonic()
        device = (self.plan or {}).get("device_name") or (self.plan or {}).get("device")
        if not background:
            self.lamp, self.stage = "amber", "running the quick model exam" if quick else "running the canary suite"
            self.progress = 0.32
        saved = self.entry(st.model_path).get("deep_validation") or {} if background else {}
        rows = (saved.get("rows", []) if saved.get("signature") == signature else [])
        # Keep completed probes across app restarts. Background work yields at
        # each boundary; it never claims the main startup bar or blocks chat.
        start_index = min(len(rows), len(items))
        for i in range(start_index, len(items)):
            name, msgs, tools, check = items[i]
            if background:
                if self._background_should_yield():
                    reason = self._background_yield_reason() or ("waiting for chat", "interactive work has priority")
                    self.background_checks["canaries"] = {"state": reason[0], "why": reason[1],
                                                           "progress": i / len(items),
                                                           "done": i, "total": len(items), "stage": name,
                                                           "elapsed_seconds": max(0, round(time.monotonic() - started_clock)),
                                                           "device": device}
                    # Completed rows are persisted after every probe. Yield at
                    # this boundary so the caller can release the orchestrator
                    # lock before interactive inference or media needs it.
                    return None
                if st.model_path != path:
                    return None
                self.background_checks["canaries"] = {"state": "running", "progress": i / len(items),
                                                       "done": i, "total": len(items), "stage": name,
                                                       "elapsed_seconds": max(0, round(time.monotonic() - started_clock)),
                                                       "device": device}
            else:
                self.stage = f"canary: {name}"
                self.progress = max(self.progress, 0.32 + 0.66 * i / len(items))
            row = run_canary(st.local, name, msgs, tools, check, self.frame, self.sampling,
                             max_tokens=QUICK_CANARY_TOKENS if quick else CANARY_TOKENS)
            rows.append(row)
            if background:
                self.update_entry_for(path, deep_validation={"signature": signature, "state": "running",
                                                             "progress": len(rows) / len(items), "done": len(rows),
                                                             "total": len(items), "passed": sum(r.get("ok") for r in rows),
                                                              "rows": rows, "elapsed_seconds": round(time.monotonic() - started_clock),
                                                              "t": time.time()})
                durations = [float(r.get("ms") or 0) for r in rows if r.get("ms")]
                remaining = max(0, len(items) - len(rows))
                eta = round(sum(durations) / len(durations) * remaining / 1000) if len(durations) >= 2 else None
                self.background_checks["canaries"] = {"state": "running", "progress": len(rows) / len(items),
                                                       "done": len(rows), "total": len(items),
                                                       "elapsed_seconds": max(0, round(time.monotonic() - started_clock)),
                                                       "eta_seconds": eta, "device": device}
        if background and st.model_path != path:
            return None
        self.canaries = rows
        passed = sum(r["ok"] for r in rows)
        gen = sum(r.get("gen", 0) for r in rows)
        ms = sum(r["ms"] for r in rows)
        failures = [r for r in rows if not r["ok"]]
        accepted = not failures or (len(failures) <= 1 and all(r.get("why") == "wrong answer" for r in failures))
        state = "passed" if accepted else "failed"
        record = {"signature": signature, "state": state, "t": time.time(), "passed": passed,
                  "of": len(items), "rows": rows, "frame": self.frame, "progress": 1.0,
                  "elapsed_seconds": round(time.monotonic() - started_clock),
                  "why": "; ".join(f"{r['name']}: {r['why']}" for r in failures)[:500]}
        if background:
            self.update_entry_for(path, deep_validation=record,
                                  canary={"validation_signature": signature, "t": time.time(), "passed": passed,
                                          "of": len(rows), "rows": rows,
                                          "frame": self.frame})
            self.background_checks["canaries"] = {"state": state, "progress": 1.0, "done": len(rows),
                                                   "total": len(items), "passed": passed,
                                                   "elapsed_seconds": record["elapsed_seconds"], "device": device}
        elif quick:
            self.update_entry(quick_validation=record)
        else:
            self.update_entry_for(path, canary={"validation_signature": signature, "t": time.time(),
                                                "passed": passed, "of": len(rows), "rows": rows,
                                                "frame": self.frame},
                                  deep_validation=record)
        self.history("canaries", passed=passed, of=len(rows), tok_s=round(gen / max(ms, 1) * 1000, 1))
        fails = [r for r in rows if not r["ok"]]
        self.say(f"canaries {passed}/{len(rows)}", "; ".join(f"{r['name']}: {r['why']}" for r in fails) or "all clean",
                 "canary")
        if not fails:
            if not background:
                self.lamp, self.stage, self.progress = "green", "ready", 1.0
            if self.egypt and self.egypt.get("stage") in ("warned", "fled", "fixing"):
                self.out_of_egypt()
            return True
        # one wrong fact alone is the model's knowledge, not gibberish; more is a fault
        if all(r["why"] == "wrong answer" for r in fails) and len(fails) <= 1:
            if background:
                self.background_checks["canaries"] = {"state": "passed", "progress": 1.0,
                                                       "done": len(rows), "total": len(items), "passed": passed,
                                                        "note": "one weak factual answer; no structural faults",
                                                        "elapsed_seconds": record["elapsed_seconds"], "device": device}
            if not background:
                self.lamp, self.stage = "green", f"ready ({len(fails)} check answer{'s' if len(fails) > 1 else ''} weak)"
            return True
        if background:
            self.background_checks["canaries"] = {"state": "failed", "progress": 1.0,
                                                   "done": len(rows), "total": len(items),
                                                   "passed": passed,
                                                    "why": "; ".join(f"{r['name']}: {r['why']}" for r in fails)[:500],
                                                    "elapsed_seconds": record["elapsed_seconds"], "device": device}
            self.update_entry(deep_validation={**record, "why": self.background_checks["canaries"]["why"]})
            return False
        self.warning(fails)
        return False

    def speed_trial(self, path):
        """Time each mode that can hold this model, here, and keep the fastest for Auto
        (Ernest, 2026-10-14: time the modes on a new model and keep the fastest). Once per
        model, engine build and machine; only on Auto; it stops when the person needs the
        model. Returns True when it finished (the engine then runs the winner)."""
        st = self.st
        if st.cfg.get("device", "auto") != "auto" or not path:
            return True
        # the running engine holds memory each candidate would get back: without this the
        # CPU-only plan looked too big beside the hybrid engine ("only one way", 2026-10-14)
        dev = self.freed(probe(0))
        stamp, hw = engine_stamp(), hardware_sig(dev)
        trials = self.entry(path).get("speed_trials") or {}
        trial = trials.get(hw) or {}
        if trial.get("engine") == stamp and trial.get("hw") == hw:
            return True
        facts = dict(self.facts or model_facts(path))
        facts.pop("measured_best", None)
        if facts.get("compat"):                 # the maker's code has one way to run here
            return True
        kw = dict(pool_mode=st.cfg.get("pool_mode", "smart"), gap=float(st.cfg.get("pool_gap", 4) or 4),
                  exclude=st.cfg.get("gpus_off") or [], reserve_gb=float(st.cfg.get("vram_reserve_gb", 0) or 0),
                  cpu_offload=bool(st.cfg.get("cpu_offload", True)))
        seen, cands = set(), []
        for pref in ("gpu", "hybrid", "multi", "amd", "cpu"):
            p = plan(dev, facts, pref, **kw)
            key = (p["device"], p["gpu_layers"], p["mid_layers"], json.dumps(p.get("dml_split")))
            if p["ram_fit"] and key not in seen:
                seen.add(key)
                cands.append((pref, p["device"]))
        # the maker's own code on the NVIDIA card, for a model whose newest maths the engine
        # runs on the CPU only: timed too, and kept only when its answers agree with Sheep's
        from . import gpu_runtime
        orig = originals_for(self.entry(path)) if gpu_runtime.ready("cuda") and not facts.get("compat") else None
        if orig:
            cands.append(("compat", "compat"))
        if len(cands) < 2:
            # nothing to compare now (often the free RAM: other programs are open). Not
            # remembered: the trial runs at a later start, when more modes can hold it.
            return True
        speeds, prompt = {}, {}

        def clear_to_run():
            """Wait out heat or another GPU job (the laptop runs warm; giving up there meant
            the trial never finished); give way at once to the person's own chat."""
            t0 = time.monotonic()
            while time.monotonic() - t0 < 900:
                why = self._background_yield_reason()
                if not why:
                    return True
                if why[0] == "waiting for chat" or path != st.model_path:
                    return False
                self.stage = f"speed trial waits: {why[1]}"
                time.sleep(5)
            return False
        firsts, agree = {}, None
        for pref, device in cands:
            if not clear_to_run():
                self._start(path, check=False)            # back to the planned mode; tried again later
                return False
            if pref == "compat":
                if "cpu" in firsts:            # every path is timed; agreement decides if it counts
                    self.stage = "timing the maker's own code on the NVIDIA card (speed trial)"
                    if self._start_compat(path, dict(orig, kind="cuda"), check=False, trial=True) is False:
                        continue
                    got = self._first_words()
                    agree = sum(a == b for a, b in zip(got, firsts["cpu"]))
                    if agree < len(got) - 1:
                        self.say("speed trial: compatibility mode on the NVIDIA card disagrees with Sheep's file",
                                 f"first words agree {agree}/{len(got)}", "plan")
                        continue
                else:
                    continue
            else:
                self.stage = f"timing this model on {device.upper()} (speed trial)"
                if self._start(path, check=False, pref_override=pref) is False:
                    continue
                if device == "cpu" and orig and "cpu" not in firsts:
                    firsts["cpu"] = self._first_words()
            try:
                st.local.chat([{"role": "user", "content": TRIAL_PROMPT}], max_tokens=TRIAL_TOKENS, cache=False,
                              verify=False)
                s = st.local.stats() or {}
            except Exception as e:
                self.say(f"speed trial: could not time {device.upper()}", str(e)[:200], "plan")
                continue
            if s.get("last_gen_ms"):
                ran = (self.plan or {}).get("device", device)
                speeds[ran] = round(s.get("last_gen", 0) / s["last_gen_ms"] * 1000, 2)
                if s.get("last_prompt_ms"):
                    prompt[ran] = round(s.get("last_prompt", 0) / s["last_prompt_ms"] * 1000, 1)
        if not speeds:
            self._start(path, check=False)
            return True
        # writing speed decides; within 3% of the best, the one that reads prompts faster wins
        top = max(speeds.values())
        best = max((d for d in speeds if speeds[d] >= 0.97 * top), key=lambda d: (prompt.get(d, 0), speeds[d]))
        trials[hw] = {"engine": stamp, "hw": hw, "best": best, "tok_s": speeds, "prompt_tok_s": prompt,
                      "t": time.time()}
        if agree is not None:
            trials[hw]["compat_agree"] = f"{agree}/{len(AGREE_PROMPTS)}"
        self.update_entry_for(path, speed_trials=trials)
        self.facts["measured_best"] = best
        self.facts["measured_note"] = ", ".join(f"{k} {v} tok/s" for k, v in speeds.items())
        self.say(f"speed trial: {SPEED_WORDS.get(best, best.upper())} is fastest for this model here",
                 self.facts["measured_note"], "plan")
        self._start(path, check=False)                    # Auto now plans the winner
        return True

    def tuned_env(self, path, device, dev):
        """The GPU settings this model's engine tune kept on this machine and engine build, or {}."""
        if device not in TUNE_DEVICES or not path:
            return {}
        try:
            sig = tech_sig(self.facts if self.facts and path == self.st.model_path else model_facts(path))
        except Exception:
            return {}
        t = load_tunes().get(f"{sig}#{hardware_sig(dev)}") or {}
        return dict(t.get("env") or {}) if t.get("engine") == engine_stamp() else {}

    def engine_tune(self, path):
        """Try a few GPU launch settings for this model here and keep one only when it gives the
        same answer and wins every paired race (TUNE_PAIRS, TUNE_GAIN). Once per model, machine
        and engine build; gives way to the person's chat and to heat like the speed trial.
        Returns True when it finished (or had nothing to do)."""
        st = self.st
        eng = st.engine
        if eng.device not in TUNE_DEVICES or not path or getattr(eng, "compat_running", False):
            return True
        dev = self.freed(probe(0))
        stamp, hw = engine_stamp(), hardware_sig(dev)
        key_ = f"{tech_sig(self.facts or model_facts(path))}#{hw}"
        if (load_tunes().get(key_) or {}).get("engine") == stamp:
            return True
        planned = dict(eng.tune_env or {})

        def clear():
            t0 = time.monotonic()
            while time.monotonic() - t0 < 900:
                why = self._background_yield_reason()
                if not why:
                    return True
                if why[0] == "waiting for chat" or path != st.model_path:
                    return False
                self.stage = f"engine tune waits: {why[1]}"
                time.sleep(5)
            return False

        def run(env):
            """(seconds for the tune prompt and reply, the reply) on a fresh engine with env."""
            eng.stop()
            eng.tune_env = env
            if not eng.start(path):
                return None, ""
            try:
                msg, _ = st.local.chat([{"role": "user", "content": TUNE_PROMPT}], max_tokens=TUNE_TOKENS,
                                       cache=False, verify=False)
                s_ = st.local.stats() or {}
            except Exception:
                return None, ""
            ms = (s_.get("last_prompt_ms") or 0) + (s_.get("last_gen_ms") or 0)
            return (ms / 1000.0 if ms else None), str(msg.get("content") or "")

        def back(why):
            eng.stop()
            eng.tune_env = planned
            eng.start(path)
            self.stage = "in sync"
            return why

        self.stage = "tuning the engine for this model (same answers, faster settings)"
        if not clear():
            return back(False)
        base_s, base_text = run({})
        if base_s is None:
            return back(True)
        times, rejected = {}, []
        for cand in tune_candidates(dev):
            if not clear():
                return back(False)
            sec, text = run(dict(cand))
            key = " ".join(f"{k}={v}" for k, v in cand.items())
            if sec is None or text != base_text:
                rejected.append(key)
                continue
            times[key] = (sec, cand)
        win, gains = None, []
        if times:
            key, (sec, cand) = min(times.items(), key=lambda kv: kv[1][0])
            if base_s / sec >= TUNE_GAIN:
                for i in range(TUNE_PAIRS):          # head to head, the order swapped every pair
                    if not clear():
                        return back(False)
                    order = ((cand, "c"), ({}, "b")) if i % 2 == 0 else (({}, "b"), (cand, "c"))
                    got = {}
                    for env, tag in order:
                        got[tag], _ = run(dict(env))
                    if not got.get("b") or not got.get("c"):
                        break
                    gains.append(got["b"] / got["c"])
                if len(gains) == TUNE_PAIRS and min(gains) >= TUNE_GAIN:
                    win = (key, cand)
        save_tune(key_, {"engine": stamp, "env": dict(win[1]) if win else {}, "t": time.time(),
                         "gain": round(min(gains), 3) if win else 1.0, "first_file": Path(path).name,
                         "tried": {k: round(base_s / v[0], 3) for k, v in times.items()}, "rejected": rejected})
        if win:
            self.say(f"engine tune: {win[0]} makes this model {round((min(gains) - 1) * 100)}% faster here",
                     f"won {TUNE_PAIRS}/{TUNE_PAIRS} paired races; answers word for word the same", "plan")
            eng.stop()
            eng.tune_env = dict(win[1])
            eng.start(path)
            self.stage = "in sync"
            return True
        self.say("engine tune: the standard settings are already the fastest for this model here",
                 f"{len(times)} settings timed, {len(rejected)} changed the answer and were dropped", "plan")
        return back(True)

    def _first_words(self):
        """The running engine's greedy first words on AGREE_PROMPTS: native and compatibility
        mode must say the same before the faster one is trusted."""
        out = []
        for q in AGREE_PROMPTS:
            try:
                msg, _ = self.st.local.chat([{"role": "user", "content": q}], max_tokens=6, cache=False, verify=False)
                out.append(" ".join(str(msg.get("content") or "").split()[:2]).lower())
            except Exception:
                out.append("")
        return out

    def _queue_background_checks(self, path, signature):
        """Resume full quality checks and orientation after chat/model activation."""
        entry = self.entry(path)
        deep = entry.get("deep_validation") or {}
        orientation = entry.get("orientation_validation") or {}
        if self._finished_check(deep, signature):
            self.background_checks["canaries"] = {"state": deep.get("state", "passed"), "progress": 1.0,
                                                   "done": deep.get("of", len(CANARIES)),
                                                   "total": deep.get("of", len(CANARIES)), "cached": True,
                                                   "elapsed_seconds": deep.get("elapsed_seconds"),
                                                   "device": (self.plan or {}).get("device_name") or (self.plan or {}).get("device"),
                                                   "why": deep.get("why")}
        else:
            progress, done = self._resume_progress(deep, signature, len(CANARIES))
            self.background_checks["canaries"] = {
                "state": "queued", "progress": progress,
                "done": done, "total": len(CANARIES),
                "device": (self.plan or {}).get("device_name") or (self.plan or {}).get("device"),
            }
        if self._finished_check(orientation, signature):
            self.background_checks["orientation"] = {"state": orientation.get("state", "passed"), "progress": 1.0,
                                                       "done": orientation.get("of", ORIENTATION_TASKS),
                                                       "total": orientation.get("of", ORIENTATION_TASKS), "cached": True,
                                                       "elapsed_seconds": orientation.get("elapsed_seconds"),
                                                       "device": (self.plan or {}).get("device_name") or (self.plan or {}).get("device"),
                                                       "why": orientation.get("why")}
        else:
            progress, done = self._resume_progress(orientation, signature, ORIENTATION_TASKS)
            self.background_checks["orientation"] = {
                "state": "queued", "progress": progress,
                "done": done, "total": ORIENTATION_TASKS,
                "device": (self.plan or {}).get("device_name") or (self.plan or {}).get("device"),
            }
        if self._finished_check(deep, signature) and self._finished_check(orientation, signature):
            return
        if self._background_thread and self._background_thread.is_alive() and self._background_key == signature:
            return
        self._background_key = signature

        def go():
            nonlocal signature                  # the speed trial may move the model to another device
            while path == self.st.model_path and (self.loading.is_set() or self._background_should_yield()):
                reason = (("waiting for model load", "the model is still starting") if self.loading.is_set()
                          else self._background_yield_reason())
                self._set_background_wait_state(reason)
                time.sleep(0.5)
            if path != self.st.model_path:
                return
            # A model-load or resource wait is transient. Once the blocker clears,
            # return the rows to queued until the resumed worker records new work.
            self._set_background_wait_state(None)
            while not self.lock.acquire(timeout=0.5):
                if path != self.st.model_path:
                    return
            try:
                if path != self.st.model_path or not self.st.engine.alive():
                    return
                # The checks the person sees go first; the speed trial and the engine tune run after
                # them. They ran first, and the card said "Starts in a moment" for minutes after every
                # engine update (2026-10-27). If the trial moves the model, the checks run once more there.
                current = self.entry(path)
                deep_now = current.get("deep_validation") or {}
                orient_now = current.get("orientation_validation") or {}
                if not self._finished_check(deep_now, signature):
                    outcome = self.canary_suite("background", CANARIES, signature=signature)
                    if outcome is None and path == self.st.model_path:
                        # The suite saved its completed rows and yielded. Release
                        # this lock before interactive work reserves a GPU.
                        return
                current = self.entry(path)
                deep_now = current.get("deep_validation") or {}
                if (path == self.st.model_path and self._valid_check(deep_now, signature)
                        and not self._finished_check(current.get("orientation_validation"), signature)):
                    self.orient(background=True, signature=signature)
                elif path == self.st.model_path and not self._valid_check(deep_now, signature):
                    self.background_checks["orientation"] = {"state": "blocked by canary failures", "progress": 0.0,
                                                               "done": 0, "total": ORIENTATION_TASKS,
                                                               "why": "the full canary check needs attention first"}
                if path != self.st.model_path or self._background_should_yield():
                    return
                before = (self.plan or {}).get("device")
                if not self.speed_trial(path):
                    return                                  # yielded to the person; resumed later
                if not self.engine_tune(path):
                    return                                  # yielded to the person; resumed later
                if (self.plan or {}).get("device") != before and path == self.st.model_path:
                    signature = self.validation_signature(path, self.plan)   # the checks again, on the faster device
                    if self.canary_suite("background", CANARIES, signature=signature) is not None:
                        self.orient(background=True, signature=signature)
            finally:
                self.lock.release()
                self._requeue_when_clear(path)

        self._background_thread = threading.Thread(target=go, name="sheep-background-validation", daemon=True)
        self._background_thread.start()

    def _requeue_when_clear(self, path):
        """A check that waited (heat, a chat turn, the GPU) comes back by itself once that clears. It
        used to wait for the next chat message: an app check sat at task 3 for hours (2026-10-20)."""
        if not any(str(c.get("state", "")).startswith("waiting for") for c in self.background_checks.values()):
            return

        def wait():
            time.sleep(5)
            while path == self.st.model_path and (self._background_should_yield() or self.lock.locked()):
                time.sleep(5)
            if path == self.st.model_path and self.st.engine.alive():
                self.resume_background_checks()
        threading.Thread(target=wait, name="sheep-requeue-checks", daemon=True).start()

    def resume_background_checks(self):
        """Resume yielded audits after an interactive request releases the model."""
        path = self.st.model_path
        if not path or not self.st.engine.alive():
            return False
        signature = self.validation_signature(path, self.plan)
        current = self._background_thread
        if current and current.is_alive():
            # The prior worker may be just finishing after yielding. Join it off
            # the request thread, then requeue from its saved rows if necessary.
            if self._resume_thread and self._resume_thread.is_alive():
                return True

            def after_current():
                current.join()
                if (self.st.model_path == path and self.st.engine.alive()
                        and not self.loading.is_set() and not self._background_should_yield()):
                    self._queue_background_checks(path, signature)

            self._resume_thread = threading.Thread(target=after_current, name="sheep-resume-validation", daemon=True)
            self._resume_thread.start()
            return True
        self._queue_background_checks(path, signature)
        return True

    # -------------------------------------------------------------- Egypt
    def warning(self, fails):
        """Step 1: sensed. Step 2 (flight) follows at once."""
        self.egypt = {"stage": "warned", "t": time.time(), "fails": [f"{r['name']}: {r['why']}" for r in fails],
                      "tried": [], "never": list(self.entry().get("never", []))}
        self.history("egypt: warned", fails=self.egypt["fails"])
        self.say("the model is producing faults", "; ".join(self.egypt["fails"]), "egypt")
        cloud = self.st.cloud.ready() and self.st.mode == 1
        self.egypt["stage"] = "fled"
        self.egypt["standin"] = "nebius" if cloud else "stopped"
        self.history("egypt: fled", standin=self.egypt["standin"])
        self.say("turns go to Nebius while the model is fixed" if cloud
                 else "replies from this model are held back while it is fixed",
                 "Sheep + Nebius mode with a key" if cloud else "Sheep-only mode: nothing leaves the machine", "egypt")
        self.doctor()

    def _try(self, label, frame, sampling):
        """One candidate fix: the whole suite must pass with it."""
        key = json.dumps({"frame": frame, "sampling": sampling}, sort_keys=True)
        if key in self.egypt["never"]:
            self.egypt["tried"].append({"fix": label, "ok": False, "why": "known bad: not used again"})
            return False
        self.stage = f"Repairing: trying {label}"
        rows = [run_canary(self.st.local, n, m, t, c, frame, sampling, max_tokens=QUICK_CANARY_TOKENS)
                for n, m, t, c in QUICK_CANARIES]
        bad = [r for r in rows if not r["ok"] and r["why"] != "wrong answer"]
        # a fix must pass all but one: under a broken setting babble also reads as
        # "wrong answer". Recovery is limited to the compact exam before use.
        ok = not bad and sum(r["ok"] for r in rows) >= len(rows) - 1
        self.egypt["tried"].append({"fix": label, "ok": ok, "passed": sum(r["ok"] for r in rows), "of": len(rows)})
        self.history("egypt: tried", fix=label, ok=ok, passed=sum(r["ok"] for r in rows))
        if ok:
            self.canaries = rows
        return ok

    def doctor(self):
        """Step 3: the known causes, in order. A fix is kept only if the canaries pass with it."""
        self.egypt["stage"] = "fixing"
        self.lamp = "amber"
        self.history("egypt: fixing")
        f = self.facts or {}
        bad_key = json.dumps({"frame": self.frame, "sampling": self.sampling}, sort_keys=True)
        # 0. the device: faults on a GPU that the CPU does not make are the GPU path's, for
        #    this model. Remembered per model and engine build; the person's setting stays.
        on = (self.plan or {}).get("device")
        if on and on != "cpu" and self.st.model_path:
            self.stage = f"Repairing: checking the model on the CPU (it faulted on {on.upper()})"
            started = self._start(self.st.model_path, check=False, pref_override="cpu")
            if started is not False and self._try("the CPU path", self.frame, self.sampling):
                fault = add_fault((self.facts or {}).get("gpu_fault"), on)
                self.update_entry(gpu_fault={"device": fault, "engine": engine_stamp(), "t": time.time(),
                                             "why": "; ".join(self.egypt.get("fails") or [])[:300]})
                if self.facts is not None:
                    self.facts["gpu_fault"] = fault
                self.say(f"this model gives wrong answers on {on.upper()} and right ones on the CPU",
                         "it runs on the CPU from now on; the GPU is tried again when the engine is updated", "plan")
                return self._fixed({"frame": self.frame, "sampling": self.sampling},
                                   f"the {on.upper()} path gave wrong answers; it runs on the CPU")
            self._start(self.st.model_path, check=False)      # not the device: back to the plan, other fixes
        self.egypt["never"].append(bad_key)
        # 1. template: the other frame (the file's own template vs the engine's built-in frames)
        other = "builtin" if self.frame == "own" else ("own" if f.get("template") else None)
        if other and self._try(f"the {'built-in' if other == 'builtin' else 'model' + chr(39) + 's own'} chat frame",
                               other, self.sampling):
            return self._fixed({"frame": other, "sampling": self.sampling}, "chat template")
        # 2. stops: a file without its end-of-turn ids gets them from the original checkpoint
        src = self.entry().get("source_dir")
        if not f.get("stops") and src and Path(src).is_dir():
            self.stage = "Repairing: reading the stop tokens from the original files"
            r = subprocess.run([sys.executable, str(TOOLING / "sml_chat_block.py"), self.st.model_path, src],
                               capture_output=True, text=True)
            self.egypt["tried"].append({"fix": "stop tokens from the original", "ok": r.returncode == 0})
            if r.returncode == 0:
                self.st.engine.start(self.st.model_path)
                self.st.local._chat = None
                self.facts = model_facts(self.st.model_path)
                if self._try("the original stop tokens", self.frame, self.sampling):
                    return self._fixed({"frame": self.frame, "sampling": self.sampling}, "stop tokens")
        # 3. sampling: the model's own generation settings, for loops that only greedy decoding falls into
        s = {"temperature": 0.6, "top_p": 0.9}
        if self._try("the model's own sampling", self.frame, s):
            return self._fixed({"frame": self.frame, "sampling": s}, "sampling")
        # 4-5. quantization and architecture need the original weights and a rebuild: offered, not guessed
        self.egypt["stage"] = "stuck"
        self.egypt["offer"] = ("add it again from its original in Settings › Models" +
                               (", or keep it on Nebius" if self.st.cloud.ready() else ""))
        self.lamp, self.stage = "red", "the model is not fixed: " + self.egypt["offer"]
        self.history("egypt: stuck", offer=self.egypt["offer"])
        self.say("could not fix the model locally", self.egypt["offer"], "egypt")
        self.update_entry(never=self.egypt["never"])

    def _fixed(self, fix, cause):
        self.frame, self.sampling = fix["frame"], fix.get("sampling")
        self.update_entry(fix=fix, never=self.egypt["never"])
        self.egypt["cause"] = cause
        self.out_of_egypt()

    def out_of_egypt(self):
        """Step 4: back to local, and never back to the bad configuration."""
        self.egypt["stage"] = "returned"
        self.egypt["t_back"] = time.time()
        self.lamp, self.stage = "green", "in sync (fixed: " + self.egypt.get("cause", "passed on re-check") + ")"
        self.history("egypt: returned", cause=self.egypt.get("cause", ""))
        self.say("Your model is fixed and running on this machine again", self.egypt.get("cause", ""), "egypt")

    # -------------------------------------------------------------- every turn
    def after_turn(self, meta):
        """Called after each local reply: the live guard (Egypt's warning during use) and the speed history."""
        sml = meta.get("sml") or {}
        s = self.st.stats or {}
        self.turns["n"] += 1
        if s.get("last_gen_ms"):
            self.turns["tok"] += s.get("last_gen", 0)
            self.turns["ms"] += s.get("last_gen_ms", 0)
        if s.get("last_prompt_ms"):
            self.turns["prompt"] += max(0, s.get("last_prompt", 0) - s.get("last_reused", 0))
            self.turns["prompt_ms"] += s.get("last_prompt_ms", 0)
        if meta.get("degenerate") or sml.get("degenerate"):
            self.turns["trips"] += 1
            self.history("guard trip", reason=str(meta.get("degenerate") or sml.get("degenerate")))
            self.say("the guard cut a reply", str(meta.get("degenerate")), "guard")
            if self.on and not self.lock.locked():
                threading.Thread(target=self._recheck, daemon=True).start()
        elif self.turns["n"] % 10 == 0:
            self.history("turns", n=self.turns["n"], tok_s=self.tok_s(), trips=self.turns["trips"])

    def _recheck(self):
        with self.lock:
            self.canary_suite("guard trip")

    def tok_s(self):
        t = self.turns
        if t["ms"] > 0:
            return round(t["tok"] / t["ms"] * 1000, 1)
        s = self.st.stats or {}
        if s.get("last_gen_ms"):
            return round(s["last_gen"] / s["last_gen_ms"] * 1000, 1)
        c = [r for r in self.canaries if r.get("gen") and r.get("ms")]
        if c:
            return round(sum(r["gen"] for r in c) / sum(r["ms"] for r in c) * 1000, 1)
        return None

    def chat_kwargs(self):
        """The frame and sampling the doctor settled on, for every local call."""
        kw = {"render": self.frame == "own"}
        if self.sampling:
            kw.update(self.sampling)
        return kw

    def ready(self):
        """The local model can answer now: checked and green, or running unchecked
        with the orchestrator switched off."""
        if not self.on:
            return self.st.engine.alive()
        return self.lamp == "green" and self.st.engine.alive()

    def holding(self):
        """True while a faulty model's replies must not reach the person."""
        return bool(self.egypt and self.egypt.get("stage") in ("warned", "fled", "fixing", "stuck"))

    # -------------------------------------------------------------- what the app reads
    def abilities(self):
        """What the loaded model, and this machine, can actually do (spec 3 and 8).
        Tools attach by ability, and when one is missing the app says which model
        to add rather than letting a call fail (spec N).

        Measured, not declared: the tool ability is the canary that made a real
        call; vision is a tower in the file itself, or a Sheep vision model here;
        media tools are offered when ready or when their local runtime can install
        weights on first use."""
        from . import adapters
        from .tools import vision as V
        path = self.st.model_path
        canary = {r.get("name"): r.get("ok") for r in (self.canaries or [])}
        ready = {m["kind"] for m in adapters.catalogue() if m.get("ready")}
        sees = (bool(path) and V.has_tower(path)) or bool(V.model(self.st))
        return {
            "text": bool(path) or self.st.mode == 1,
            "tools": bool(canary.get("tool call")),
            "reads pictures": sees,
            "makes pictures": "image" in ready,
            "hears": any(m["id"] == "whisper-small" and m.get("ready") for m in adapters.catalogue()),
            "speaks": any(m["id"] == "windows-voices" and m.get("ready") for m in adapters.catalogue()),
            "makes video": "video" in ready or adapters.video_installable(),
        }

    # What to add when an ability is missing, in the person's words (spec N).
    NEEDS = {
        "reads pictures": "a vision model such as SmolVLM, quantized by Sheep like any other model",
        "makes pictures": "a picture model (Settings > Models > pictures)",
        "hears": "the speech-to-text model (Settings > Models > sound)",
        "speaks": "a Windows voice (Settings > Models > sound)",
        "makes video": "ffmpeg and the local AI video runtime, or use Sheep Studio",
    }

    def status(self):
        ks = self.entry(self.st.model_path).get("knows_sheep") or {} if self.st.model_path else {}
        return {"on": self.on, "lamp": self.lamp, "stage": self.stage, "plan": self.plan, "facts": self.facts,
                "canaries": self.canaries, "egypt": self.egypt, "frame": self.frame, "sampling": self.sampling,
                "log": self.log[-60:], "busy": self.lock.locked(), "gpu_job": self.gpu_job,
                "gpu_queue": [{k: row[k] for k in ("what", "priority", "queued")} for row in self.gpu_waiters],
                "abilities": self.abilities(),
                "checks": self.readiness_checks(),
                "loading": self.loading.is_set(),
                "heat": self.thermal.snapshot() if getattr(self, "thermal", None) else None,
                "knows_sheep": {k: ks.get(k) for k in ("passed", "of", "t")} if ks else None}

    def reorient(self):
        """Settings › Activity › Check again on the app checks: run them again in the background."""
        self.recheck("orientation")

    def recheck(self, check):
        """Settings › Activity › Check again: forget that check's result and run it in the background."""
        path = self.st.model_path
        if not path:
            raise RuntimeError("no model is selected")
        if check == "startup":
            return self.start_model(path)
        keys = {"canaries": ("deep_validation",), "orientation": ("orientation_validation", "knows_sheep")}[check]
        reg = load_registry()
        for k in keys:
            reg.get(path, {}).pop(k, None)
        save_registry(reg)
        self.background_checks[check] = {"state": "queued", "progress": 0.0}
        self.resume_background_checks()

    def retry_orientation_failure(self):
        """Retry every failed app task, without taking Chat offline."""
        from . import orient as O
        from .paths import HOME

        model_path = self.st.model_path
        if not model_path or not self.ready():
            raise RuntimeError("Chat must be ready before an app check can be retried")
        saved = self.entry(model_path).get("orientation_validation") or {}
        signature = self.validation_signature(model_path)
        if saved.get("signature") != signature:
            raise RuntimeError("The model changed; its app checks need to refresh first")
        rows = list(saved.get("rows") or [])
        tasks = O.quick()
        retry = [t for t in tasks if t.id in {r.get("id") for r in rows if not r.get("ok")}]
        if not retry:
            raise RuntimeError("There is no failed app task to retry")

        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Sheep is busy with another task; try again when it finishes")

        root = HOME / "orientation" / Path(model_path).stem / signature[:16]
        passed_before = sum(bool(row.get("ok")) for row in rows)
        self.background_checks["orientation"] = {
            "state": "running", "progress": None, "done": len(rows), "total": len(tasks),
            "passed": passed_before, "of": len(rows), "stage": f"retrying {len(retry)} app task" + ("s" if len(retry) > 1 else ""),
            "device": (self.plan or {}).get("device_name") or (self.plan or {}).get("device")}

        def go():
            try:
                if self.st.model_path != model_path:
                    self.background_checks["orientation"] = {
                        "state": "deferred", "progress": 0.0, "done": passed_before,
                        "total": len(tasks), "passed": passed_before, "of": len(rows),
                        "why": "The selected model changed before the app task could be retried"}
                    return
                result = {r["id"]: r for r in O.run_inprocess(self.st, retry, root)}
                if not result:
                    return
                merged = [result.get(row.get("id"), row) for row in rows]
                passed = sum(bool(row.get("ok")) for row in merged)
                why = "; ".join(f"{row.get('id')}: {row.get('why')}" for row in merged if not row.get("ok"))[:400]
                elapsed = round(float(saved.get("elapsed_seconds") or 0) + sum(float(r.get("s") or 0) for r in result.values()))
                record = {**saved, "signature": signature, "state": "passed" if passed == len(merged) else "failed",
                          "progress": 1.0, "done": len(merged), "total": len(tasks), "passed": passed,
                          "of": len(merged), "rows": merged, "elapsed_seconds": elapsed, "why": why, "t": time.time()}
                old_knows = self.entry(model_path).get("knows_sheep") or {}
                self.update_entry_for(model_path, orientation_validation=record,
                                      knows_sheep={**old_knows, "validation_signature": signature,
                                                   "t": time.time(), "passed": passed, "of": len(merged),
                                                   "rows": merged, "hints": O.hints(merged)})
                self.background_checks["orientation"] = {"state": record["state"], "progress": 1.0,
                                                          "done": len(merged), "total": len(tasks),
                                                          "passed": passed, "of": len(merged), "why": why,
                                                          "elapsed_seconds": elapsed,
                                                          "device": (self.plan or {}).get("device_name") or
                                                                    (self.plan or {}).get("device")}
            except Exception as exc:
                self.background_checks["orientation"] = {
                    "state": "failed", "progress": 1.0, "done": len(rows), "total": len(tasks),
                    "passed": passed_before, "of": len(rows),
                    "why": f"The retry could not finish: {type(exc).__name__}: {str(exc)[:180]}"}
            finally:
                self.lock.release()

        threading.Thread(target=go, daemon=True).start()
        return {"ok": True, "task_id": ",".join(t.id for t in retry)}


# ------------------------------------------------------------------ the dashboard's KPIs

def rag(v, green, amber, higher=True):
    if v is None:
        return "none"
    if higher:
        return "green" if v >= green else "amber" if v >= amber else "red"
    return "green" if v <= green else "amber" if v <= amber else "red"


def feature_models_kpi(slots):
    """Summarize the feature models selected in Settings for the dashboard."""
    slots = slots or []
    selected = [s for s in slots if s.get("active") and not s.get("disabled")]
    ready = [s for s in selected if s.get("ready")]
    if not selected:
        status, rag_color, value, unit = "No models selected", "none", None, ""
    elif len(ready) == len(selected):
        status, rag_color, value, unit = "All ready", "green", len(ready), f"/ {len(selected)}"
    elif ready:
        status, rag_color, value, unit = "Some unavailable", "amber", len(ready), f"/ {len(selected)}"
    else:
        status, rag_color, value, unit = "Unavailable", "red", 0, f"/ {len(selected)}"

    detail = []
    for slot in slots:
        if slot.get("disabled"):
            state = "disabled in Settings"
        elif not slot.get("active"):
            state = "not selected"
        elif slot.get("ready"):
            state = "ready"
        else:
            state = "selected · not ready"
        detail.append([slot.get("label") or slot.get("id") or "model",
                       f"{slot.get('active_name') or 'Not selected'} · {state}"])
    return {"id": "feature-models", "label": "Extra models", "rag": rag_color,
            "tip": "Models selected for Chat, Checker, image, video, photo guidance, and speech in Settings → Models.",
            "word": status, "value": value, "unit": unit,
            "fill": len(ready) / max(1, len(selected)), "detail": detail}


def check_tile(cid, label, tip, c, extra):
    """One check as a tile: its state in words, the count, and what it checked behind Details."""
    state, done = c.get("state") or "not started", c.get("done")
    of = c.get("of") or c.get("total")
    passed = c.get("passed")
    if state == "passed":
        colour, word = "green", "Ready" if cid == "startup" else f"All {of} right" if of else "Done"
    elif state == "failed":
        colour = "red" if cid == "startup" else "amber"
        word = ("Did not start" if cid == "startup" else f"{passed} of {of} right" if of and passed is not None
                else "Needs a retry")
    elif state == "running":
        colour, word = "amber", f"Checking · {done} of {of}" if of and done is not None else "Checking"
    elif state == "waiting for temperature":
        colour, word = "amber", "Paused: computer is hot"
    elif state.startswith("waiting") or state == "queued":
        colour, word = "amber", "Waiting its turn"
    else:
        colour, word = "none", "Not checked yet"
    share = (passed or 0) / of if of else (1.0 if state == "passed" else 0)
    count = passed if state in ("passed", "failed") else done if state == "running" else None
    detail = [["result", word]] + ([["why", str(c["why"])[:120]]] if c.get("why") else []) + extra
    return {"id": "check-" + cid, "label": label, "rag": colour, "tip": tip, "word": word,
            "value": count if of and count is not None else None, "unit": f"of {of}" if of else "",
            "fill": share, "detail": detail}


def kpis(o, dev, slots=None, preferences=None):
    """Red, amber or green first, a number second, detail on expand (STANDARDS.md section 8)."""
    p, f = o.plan or {}, o.facts or {}
    preferences = preferences or {}
    text_setting = {"auto": "Auto · recommended", "gpu": "GPU (CUDA first)", "hybrid": "CPU + GPU split",
                    "multi": "Multi-GPU + CPU", "amd": "DirectML (AMD / Intel)", "cpu": "CPU"}.get(
                        preferences.get("device", "auto"), str(preferences.get("device", "auto")))
    media_setting = {"auto": "Auto · recommended", "cuda": "NVIDIA CUDA", "directml": "DirectML (AMD / Intel)",
                     "cpu": "CPU"}.get(preferences.get("media_device", "auto"), str(preferences.get("media_device", "auto")))
    e = o.entry() if o.st.model_path else {}
    tok = o.tok_s()
    t = o.turns
    pp = round(t["prompt"] / t["prompt_ms"] * 1000) if t["prompt_ms"] > 0 else None
    can = e.get("canary") or {}
    passed, of = can.get("passed"), can.get("of") or 8
    grade = e.get("grade") or {}
    g = dev.get("gpu") or {}
    ram = dev["ram"]
    out = []
    # speed: reading pace is ~5 words/s, ~7 tokens/s
    # a word is about 1.3 tokens; a person reads about five words a second
    out.append({"id": "speed", "label": "Speed", "rag": rag(tok, 7, 3.5),
                "tip": "How fast the model writes. Green is faster than you read.",
                "word": {"green": "Faster than you read", "amber": "About reading pace", "red": "Slow",
                         "none": "Not measured yet"}[rag(tok, 7, 3.5)],
                "value": round(tok / 1.3, 1) if tok else None, "unit": "words/s", "fill": min(1.0, (tok or 0) / 14),
                "detail": [["writing", f"{tok} tokens/s" if tok else "–"],
                           ["reading prompts", f"{pp} tokens/s" if pp else "–"],
                           ["expected", f"~{p.get('est_tok_s', '–')} tokens/s from the file size"],
                           ["replies", str(t["n"])]]})
    # The three checks, under the names and from the same records as the checks card: a
    # "Canary health · Not yet checked" tile above "Model ready 5/5" read as one check
    # contradicting itself (2026-10-21)
    rows = {c["id"]: c for c in (o.readiness_checks() if hasattr(o, "readiness_checks") else [])}
    ori = e.get("knows_sheep") or {}
    extra = {"canaries": [["guard trips", str(t["trips"])],
                          ["chat frame", "model's own" if o.frame == "own" else "built-in"]]
             + [[k, str(v)] for k, v in grade.items() if k not in ("t",)],
             "orientation": [[r["area"], ("✓ " if r["ok"] else "✕ ") + r["turns"][0][:48]
                              + ("" if r["ok"] else f" ({r['why'][:60]})")]
                             for r in (e.get("knows_sheep") or {}).get("rows", [])]}
    for cid, label, tip in (
            ("startup", "Model ready", "The model loaded and answered a short first exam."),
            ("canaries", "Answers checked", "Check questions with known answers, each answer read for "
                                            "structure. An operational check, not a quality score."),
            ("orientation", "Apps checked", "Real jobs done in Sheep before first use (decks, documents, "
                                            "sheets, files, code), each checked by opening what it made.")):
        c = rows.get(cid)
        # a check waiting to run again keeps showing its last result: "Paused" over a fresh
        # 8 of 8 in the timeline read as two stories (2026-10-21)
        kept = ({"passed": passed, "of": of} if cid == "canaries" and passed is not None
                else {"passed": ori.get("passed"), "of": ori.get("of")} if cid == "orientation" and ori.get("of")
                else None)
        if kept and (c is None or c.get("state") not in ("passed", "failed", "running")):
            c = {"state": "passed" if kept["passed"] >= kept["of"] else "failed", **kept}
        out.append(check_tile(cid, label, tip, c or {}, extra.get(cid, [])))
    temp = g.get("temp")
    ram_free = ram["available"] / 1e9
    hw = "green"
    if ram_free < 1.0 or (temp is not None and temp >= 92):
        hw = "red"
    elif ram_free < 2.5 or (temp is not None and temp >= 83) or dev["power"].get("plugged") is False:
        hw = "amber"
    heat = o.thermal.snapshot() if getattr(o, "thermal", None) else {}
    if heat.get("level") == "critical":
        hw = "red"
    elif heat.get("level") in ("warm", "hot") and hw == "green":
        hw = "amber"
    pacing = {"cool": "full speed", "warm": "full speed",
              "hot": "background checks run in bursts", "critical": "background and new GPU work wait; cooling"}.get(heat.get("level"), "–")
    power_words = {"battery": "on battery: background checks wait for the charger",
                   "low": "battery low: background checks and new GPU work wait; plug in",
                   "critical": "battery very low: plug in the charger now"}.get(heat.get("power"))
    if heat.get("power") in ("low", "critical"):
        hw = "red"
    if power_words:
        pacing = f"{power_words} ({heat.get('battery')}%)"
    clock = g.get("clock")
    cm = g.get("clock_max") or 0
    # amber or red only for what changes Sheep, and the word says which
    why_hw = ("plug in the charger" if heat.get("power") in ("low", "critical")
              else "cooling: checks wait" if heat.get("level") == "critical" or (temp is not None and temp >= 92)
              else "low on memory" if ram_free < 1.0
              else "on battery: checks wait" if dev["power"].get("plugged") is False
              else "warm: checks run slower" if heat.get("level") in ("warm", "hot") or (temp is not None and temp >= 83)
              else "memory is tight" if ram_free < 2.5 else "Healthy")
    out.append({"id": "hardware", "label": "This computer", "rag": hw,
                "tip": "Free memory, heat and power, as they affect Sheep.",
                "word": why_hw[0].upper() + why_hw[1:],
                "value": round(ram_free, 1), "unit": "GB free", "fill": min(1.0, ram_free / max(1, ram["total"] / 1e9)),
                "detail": [["CPU", f"{dev['cpu']['name']} · {p.get('threads', '–')} of {dev['cpu']['threads']} threads"],
                           ["RAM", f"{ram_free:.1f} of {ram['total'] / 1e9:.1f} GB free"],
                           ["GPU", g.get("name", "–")],
                           ["other GPUs", ", ".join(a["name"] for a in dev.get("others") or []) or "–"],
                           ["text / code setting", text_setting],
                           ["media setting", media_setting],
                           ["GPU heat", f"{temp:.0f} °C" if temp is not None else "–"],
                           ["CPU heat", f"{heat['cpu']:.0f} °C" if heat.get("cpu") is not None else "–"],
                           ["heat guard", (f"{heat.get('level')}: {pacing}" if heat.get("on") else "off (Settings)")
                            if heat else "–"],
                           ["GPU clock", f"{clock:.0f} of {cm:.0f} MHz" + (" (throttled)" if cm and clock and clock < cm * 0.5 else "")
                            if clock else "–"],
                           ["VRAM", f"{g['vram_used'] / 2**30:.1f} of {g['vram_total'] / 2**30:.1f} GB used" if g else "–"],
                           ["power", "AC" if dev["power"].get("plugged") is not False else f"battery {dev['power'].get('percent')}%"]]})
    fit = p.get("gpu", {}).get("fit")
    comp = "none" if not p else ("green" if p.get("ram_fit") else "red")
    out.append({"id": "fit", "label": "Fits this computer", "rag": comp,
                "tip": "Whether the model fits this computer's memory, and where it runs.",
                "word": {"green": "Runs well here", "red": "Too big for RAM", "none": "No model"}[comp],
                "value": None, "unit": "", "fill": 1.0 if comp == "green" else 0.3 if comp == "red" else 0,
                "detail": [["runs on", {"gpu": "GPU", "hybrid": f"CPU+GPU ({p.get('gpu_layers')} of {f.get('n_layer')} layers on the GPU)",
                                        "cpu": "CPU", "amd": f"{p.get('device_name') or 'AMD GPU'} (DirectML)",
                                        "multi": f"NVIDIA {p.get('gpu_layers')} layers + {p.get('device_name') or 'AMD'} "
                                                 f"{p.get('mid_layers')} + CPU {(f.get('n_layer') or 0) - (p.get('gpu_layers') or 0) - (p.get('mid_layers') or 0)}"
                                        }.get(p.get("device"), "–")],
                           ["Settings preference", f"text/code: {text_setting} · media: {media_setting}"],
                           ["GPU", {"whole": "the whole model fits", "split": f"{p['gpu']['layers']} of {f.get('n_layer')} layers fit",
                                    "none": "no room"}.get(fit, "–") if p else "–"],
                           ["context", f"{p.get('ctx', '–')} tokens x {p.get('slots', '–')}"],
                           ["model", f"{f.get('bytes', 0) / 1e9:.2f} GB, {f.get('arch', '–')}, {f.get('n_layer', '–')} layers"]]})
    out.append(feature_models_kpi(slots))
    return out
