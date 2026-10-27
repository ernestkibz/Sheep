"""GPU support for pictures and video, one runtime per kind of GPU.

Pictures and video need PyTorch built for the GPU. On an NVIDIA card without it
they ran through DirectML, which does not fit a 4 GB RTX 3050 (2026-10-13). So:

    cuda      NVIDIA      %LOCALAPPDATA%\\Sheep\\runtimes\\cuda      CUDA PyTorch (cu124)
    directml  AMD, Intel  %LOCALAPPDATA%\\Sheep\\runtimes\\directml  torch-directml

each a Python 3.10-3.12 venv with the same diffusers stack, checked by asking the
GPU itself. Installed once (about 2.5 GB for CUDA, 1.5 GB for DirectML); the
installer starts the one this computer needs, and Settings › Advanced › This computer shows
each GPU with its runtime and an Install button.

    python -m mlmi.backend.gpu_runtime install [cuda|directml] [--if-nvidia]
    python -m mlmi.backend.gpu_runtime status
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Sheep" / "runtimes"
_BIN = "Scripts/python.exe" if os.name == "nt" else "bin/python"
# the versions the DirectML runtime has run since 2026-10, so both media paths behave the same
STACK = ["diffusers==0.35.2", "transformers==4.57.6", "accelerate==1.15.0", "safetensors>=0.4", "peft==0.18.1",
         "huggingface_hub==0.36.2", "numpy<2.3", "pillow"]
KINDS = {
    "cuda": {"label": "NVIDIA (CUDA)", "vendors": ("nvidia",), "size": "about 2.5 GB",
             "torch": ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"],
             "canary": "import torch, json; ok = torch.cuda.is_available(); print(json.dumps({'ready': ok, "
                       "'name': torch.cuda.get_device_name(0) if ok else '', 'torch': torch.__version__}))"},
    "directml": {"label": "AMD / Intel (DirectML)", "vendors": ("amd", "intel"), "size": "about 1.5 GB",
                 "torch": ["torch-directml"],
                 "canary": "import torch, torch_directml, json; n = torch_directml.device_count(); "
                           "print(json.dumps({'ready': n > 0, 'name': torch_directml.device_name(0) if n else '', "
                           "'torch': torch.__version__}))"},
    # no GPU needed: fitting an unknown architecture, and compatibility mode (the maker's own
    # code) where no CUDA runtime is installed; the newest transformers, for the newest models
    "cpu": {"label": "CPU (PyTorch)", "vendors": (), "size": "about 300 MB",
            "torch": ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
            "stack": ["transformers>=4.57.6", "accelerate", "safetensors>=0.4", "huggingface_hub", "numpy<2.3",
                      "sentencepiece", "tiktoken", "protobuf"],
            "canary": "import torch, json; print(json.dumps({'ready': True, 'name': 'CPU', 'torch': torch.__version__}))"},
}
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_jobs = {k: {"state": "idle", "step": "", "progress": 0.0, "error": "", "t": 0} for k in KINDS}
_lock = threading.Lock()


def root(kind="cuda"):
    return BASE / kind


def py(kind="cuda"):
    return root(kind) / _BIN


def _state_file(kind):
    return BASE / f"{kind}-state.json"


# what the rest of Sheep already uses (NVIDIA)
ROOT, PY, STATE = root("cuda"), py("cuda"), _state_file("cuda")


def nvidia():
    return shutil.which("nvidia-smi") is not None


def gpus():
    """This computer's GPUs as [{name, vendor, kind}] (kind: the runtime it needs)."""
    try:
        from . import hardware
        ads = hardware.adapters()
    except Exception:
        ads = []
    out = []
    for a in ads:
        vendor = a.get("vendor", "other")
        kind = next((k for k, v in KINDS.items() if vendor in v["vendors"]), None)
        if kind:
            out.append({"name": a.get("name", "GPU").replace("(TM)", "").replace("(R)", ""), "vendor": vendor, "kind": kind})
    if not out and nvidia():
        out.append({"name": "NVIDIA GPU", "vendor": "nvidia", "kind": "cuda"})
    return out


def _base_python():
    """A Python PyTorch builds for these runtimes: 3.12, 3.11 or 3.10."""
    cands = []
    if sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)):
        cands.append(Path(sys.executable))
    progs = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    for v in ("312", "311", "310"):
        cands += [progs / f"Python{v}" / "python.exe", Path(f"C:/Python{v}/python.exe")]
    for k in KINDS:                                   # the Python an existing runtime was made from
        cfg = root(k) / "pyvenv.cfg"
        if cfg.exists():
            for line in cfg.read_text(encoding="utf-8").splitlines():
                if line.startswith("home"):
                    cands.append(Path(line.split("=", 1)[1].strip()) / "python.exe")
    return next((c for c in cands if c.is_file()), None)


def check(kind="cuda"):
    """{"ready": bool, "name": GPU, "torch": version}, asked of the runtime itself."""
    if not py(kind).exists():
        return {"ready": False}
    try:
        out = subprocess.run([str(py(kind)), "-c", KINDS[kind]["canary"]], capture_output=True, text=True,
                             timeout=180, creationflags=_NOWIN)
        info = json.loads(out.stdout.strip().splitlines()[-1])
        return {"ready": bool(info.get("ready")), "name": str(info.get("name", "")).strip("\x00 "), "torch": info.get("torch", "")}
    except Exception as e:
        return {"ready": False, "error": str(e)[:200]}


def _saved(kind):
    try:
        return json.loads(_state_file(kind).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def status(kind="cuda"):
    saved = _saved(kind)
    if not saved and kind == "directml" and py(kind).exists():
        # the DirectML runtime predates this file: ask it once, and remember
        saved = check(kind)
        try:
            _state_file(kind).write_text(json.dumps({**saved, "t": time.time()}), encoding="utf-8")
        except OSError:
            pass
    with _lock:
        job = dict(_jobs[kind])
    present = [g for g in gpus() if g["kind"] == kind]
    return {"kind": kind, "label": KINDS[kind]["label"], "size": KINDS[kind]["size"],
            "nvidia": nvidia(), "gpus": [g["name"] for g in present], "needed": bool(present),
            "installed": py(kind).exists(), "ready": bool(saved.get("ready")) and py(kind).exists(),
            "gpu": saved.get("name", ""), "torch": saved.get("torch", ""), "job": job,
            "base_python": str(_base_python() or "")}


def status_all():
    """Every runtime, with whether this computer has a GPU that needs it."""
    return [status(k) for k in KINDS if k != "cpu"]


def torch_python(install_cpu=True, say=None):
    """(python, kind) of a ready PyTorch runtime: CUDA when ready, else the CPU one,
    installed on first need (about 300 MB). (None, "") when neither can be had."""
    for kind in ("cuda", "cpu"):
        if ready(kind):
            return py(kind), kind
    if not install_cpu:
        return None, ""
    if say:
        say("installing PyTorch for the CPU (about 300 MB, once)")
    got = install("cpu")
    if got.get("ready"):
        return py("cpu"), "cpu"
    if say:
        say("PyTorch could not be installed: " + str((got.get("job") or {}).get("error", ""))[:200])
    return None, ""


_knows = {}


def knows(kind, model_type):
    """Whether this runtime's Transformers has a class for model_type (asked once per run)."""
    key = (kind, model_type)
    if key not in _knows:
        try:
            r = subprocess.run([str(py(kind)), "-c", "import sys, transformers.models.auto.configuration_auto as c; "
                                "sys.exit(0 if sys.argv[1] in c.CONFIG_MAPPING_NAMES else 3)", model_type],
                               capture_output=True, timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _knows[key] = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _knows[key] = False
    return _knows[key]


def torch_python_for(model_type, install_cpu=True, say=None):
    """torch_python, but a ready runtime whose Transformers knows model_type comes first: the CUDA
    runtime had Transformers 4.57 and the CPU one 5.20, which alone knows youtu (2026-10-20)."""
    if model_type:
        for kind in ("cuda", "cpu"):
            if ready(kind) and knows(kind, model_type):
                return py(kind), kind
    return torch_python(install_cpu=install_cpu, say=say)


def ready(kind="cuda"):
    """Cheap: the saved answer, not a new subprocess."""
    return bool(_saved(kind).get("ready")) and py(kind).exists()


def _set(kind, **kw):
    with _lock:
        _jobs[kind].update(kw, t=time.time())


def _pip(kind, args, step, progress):
    _set(kind, step=step, progress=progress)
    r = subprocess.run([str(py(kind)), "-m", "pip", "install", "--disable-pip-version-check", *args],
                       capture_output=True, text=True, creationflags=_NOWIN)
    if r.returncode != 0:
        raise RuntimeError(f"{step} failed: {(r.stderr or r.stdout).strip().splitlines()[-1][:300]}")


def install(kind="cuda"):
    """Build or repair one runtime; returns its status. Safe to run again."""
    if kind not in KINDS:
        raise ValueError("the runtime is cuda, directml or cpu")
    if kind == "cuda" and not nvidia():
        _set(kind, state="failed", error="no NVIDIA GPU found on this computer")
        return status(kind)
    if kind == "directml" and os.name != "nt":
        _set(kind, state="failed", error="DirectML runs on Windows only")
        return status(kind)
    base = _base_python()
    if not base:
        _set(kind, state="failed", error="GPU support needs Python 3.10, 3.11 or 3.12 on this computer (python.org)")
        return status(kind)
    _set(kind, state="running", step="preparing", progress=0.02, error="")
    try:
        if not py(kind).exists():
            _set(kind, step="making the runtime folder", progress=0.05)
            BASE.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([str(base), "-m", "venv", str(root(kind))], capture_output=True, text=True,
                               creationflags=_NOWIN)
            if r.returncode != 0:
                raise RuntimeError("could not make the runtime: " + (r.stderr or r.stdout)[-300:])
        _pip(kind, ["--upgrade", "pip"], "updating pip", 0.08)
        _pip(kind, KINDS[kind]["torch"], f"downloading PyTorch for {KINDS[kind]['label']} ({KINDS[kind]['size']})", 0.15)
        _pip(kind, KINDS[kind].get("stack", STACK), "installing the libraries", 0.8)
        _set(kind, step="asking the GPU", progress=0.95)
        got = check(kind)
        _state_file(kind).write_text(json.dumps({**got, "t": time.time()}), encoding="utf-8")
        if not got.get("ready"):
            raise RuntimeError("PyTorch is installed but the GPU did not answer: " + str(got.get("error", ""))[:200])
        _set(kind, state="done", step=f"ready on {got.get('name')}", progress=1.0)
    except Exception as e:
        _set(kind, state="failed", error=str(e)[:400])
    return status(kind)


def start(kind="cuda"):
    """install() in the background (Settings' button); returns at once."""
    if kind not in KINDS:
        raise ValueError("the runtime is cuda or directml")
    with _lock:
        if _jobs[kind]["state"] == "running":
            return status(kind)
    threading.Thread(target=install, args=(kind,), daemon=True).start()
    return status(kind)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "status"
    kind = args[1] if len(args) > 1 else "cuda"
    if cmd == "install":
        if "--if-nvidia" in sys.argv and (not nvidia() or ready("cuda")):
            print(json.dumps(status("cuda")))
            sys.exit(0)
        print(json.dumps(install(kind), indent=1))
    else:
        print(json.dumps(status_all(), indent=1))
