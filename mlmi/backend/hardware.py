"""What this machine has, and which ways of running a model it allows.

Display adapters come from Windows (the video controllers that are present,
with their true VRAM from the driver's registry key: the WMI field stops at
4 GB). AMD and Intel GPUs are proven with a DirectML canary (a 2x2 matmul) in
Sheep's separate DirectML runtime, so a listed GPU is one that computes. The
modes say what the person may choose; an unavailable one carries the reason,
which Settings shows greyed out.
"""
import importlib.util
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .paths import HOME, REPO

CACHE = HOME / "hardware.json"
EXPORT_DLL = REPO / "sml" / "sml-export.dll"
WORKER = Path(__file__).resolve().parent / "dml_worker.py"
PTX = REPO / "sml" / "sml-gpu-fwd.ptx"
DML_PY = Path(os.environ.get("LOCALAPPDATA", "")) / "Sheep" / "runtimes" / "directml" / "Scripts" / "python.exe"
# Sheep's own engine for any Vulkan GPU (sml/sml-vk.c): the file's 4/6-bit codes on AMD,
# Intel or a second NVIDIA card, no Python runtime. Preferred over the DirectML worker,
# which held fp16 (3.6x the file) and failed its own grade on the Radeon (2026-10-14).
VK_EXE = REPO / "sml" / ("sml-vk.exe" if os.name == "nt" else "sml-vk")
VK_DIR = REPO / "sml" / "vk"
_vk = {"rows": None}
VENDORS = {"10de": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel"}
VENDOR_ORDER = {"nvidia": 0, "amd": 1, "intel": 2, "other": 3}
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_lock = threading.Lock()
_state = {"adapters": None, "directml": {"state": "not probed", "devices": []}}

DML_CANARY = r'''
import json, time, torch, torch_directml
rows = []
for i in range(int(torch_directml.device_count())):
    r = {"index": i, "name": str(torch_directml.device_name(i)).rstrip("\x00"), "ok": False}
    try:
        d = torch_directml.device(i)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=d)
        r["ok"] = (x @ x).cpu().tolist() == [[7.0, 10.0], [15.0, 22.0]]
        if r["ok"]:
            # its speed at what decode does: read a 128 MB fp16 matrix once per token
            w = torch.randn(4096, 16384, dtype=torch.float16, device=d)
            v = torch.randn(1, 16384, dtype=torch.float16, device=d)
            for _ in range(3):
                y = torch.nn.functional.linear(v, w)
            y.cpu()
            t0 = time.perf_counter()
            for _ in range(20):
                y = torch.nn.functional.linear(v, w)
            y.cpu()
            r["gbps"] = round(20 * w.numel() * 2 / (time.perf_counter() - t0) / 1e9, 1)
            del w, v, y
    except Exception as e:
        r["why"] = str(e)[:200]
    rows.append(r)
print(json.dumps({"torch": torch.__version__, "devices": rows}))
'''


def _vram_from_registry(pnp):
    """The adapter's VRAM from its driver key (HardwareInformation.qwMemorySize)."""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Enum\\" + pnp)
        drv = winreg.QueryValueEx(k, "Driver")[0]
        c = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Class\\" + drv)
        for name in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
            try:
                v = winreg.QueryValueEx(c, name)[0]
                return int.from_bytes(v[:8], "little") if isinstance(v, bytes) else int(v)
            except OSError:
                continue
    except (OSError, ImportError):
        pass
    return 0


def adapters(refresh=False):
    """Present display adapters: name, vendor, VRAM, integrated or not."""
    with _lock:
        if _state["adapters"] is not None and not refresh:
            return _state["adapters"]
    rows = []
    if os.name == "nt":
        ps = ("Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,AdapterRAM,DriverVersion,Status "
              "| ConvertTo-Json -Compress")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 timeout=20, creationflags=NOWIN).stdout.strip()
            items = json.loads(out) if out else []
            items = items if isinstance(items, list) else [items]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            items = []
        for it in items:
            pnp = it.get("PNPDeviceID") or ""
            low = pnp.lower()
            if "ven_" not in low:
                continue                    # virtual and basic display adapters
            vendor = VENDORS.get(low.split("ven_")[1][:4], "other")
            vram = _vram_from_registry(pnp) or int(it.get("AdapterRAM") or 0)
            name = (it.get("Name") or "").strip()
            # an integrated GPU shares system RAM: Intel's, and AMD's "Radeon Graphics"
            # (Intel Arc cards are discrete)
            integrated = (vendor == "intel" and "arc" not in name.lower()) or (
                vendor == "amd" and "radeon" in name.lower()
                and not any(s in name.lower() for s in (" rx ", "rx ", "pro w", "vega 56", "vega 64")))
            rows.append({"name": name, "vendor": vendor, "vram": vram, "integrated": integrated,
                         "driver": it.get("DriverVersion") or "", "ok": (it.get("Status") or "OK") == "OK"})
    rows.sort(key=lambda a: (not a.get("ok", True), VENDOR_ORDER.get(a.get("vendor"), 3),
                             a.get("integrated", False), -(a.get("vram") or 0), a.get("name", "").casefold()))
    with _lock:
        _state["adapters"] = rows
    return rows


def avx2():
    """Whether the CPU has AVX2, which Sheep's CPU engine is built for."""
    if os.name != "nt":
        try:
            return "avx2" in Path("/proc/cpuinfo").read_text()
        except OSError:
            return True
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))   # PF_AVX2_INSTRUCTIONS_AVAILABLE
    except (OSError, AttributeError):
        return True


def torch_cuda():
    """The main Python's torch is a CUDA build (media adapters on NVIDIA), read
    from its version file so nothing heavy is imported."""
    try:
        spec = importlib.util.find_spec("torch")
        if not spec or not spec.origin:
            return False
        txt = (Path(spec.origin).parent / "version.py").read_text(encoding="utf-8")
        return "cuda = None" not in txt and "cuda: Optional[str] = None" not in txt and "cuda" in txt
    except (OSError, ImportError, ValueError):
        return False


def media_cuda():
    """Pictures and video can run on CUDA: the app's own PyTorch is a CUDA build, or
    the separate CUDA runtime (gpu_runtime.py) is installed and answered."""
    try:
        from . import gpu_runtime
        return torch_cuda() or gpu_runtime.ready()
    except Exception:
        return torch_cuda()


def directml(refresh=False, wait=False):
    """The DirectML canary, cached per set of adapters (it takes seconds)."""
    with _lock:
        cur = dict(_state["directml"])
    if cur["state"] in ("ok", "failed", "missing", "probing") and not refresh:
        return cur
    if not DML_PY.exists():
        res = {"state": "missing", "devices": [], "why": "the DirectML runtime is not installed"}
        with _lock:
            _state["directml"] = res
        return res
    key = [a["name"] for a in adapters()]
    try:
        c = json.loads(CACHE.read_text(encoding="utf-8")).get("directml") or {}
        measured = all("gbps" in x for x in (c.get("res") or {}).get("devices", []) if x.get("ok"))
        if not refresh and c.get("key") == key and time.time() - c.get("t", 0) < 7 * 86400 and measured:
            with _lock:
                _state["directml"] = c["res"]
            return c["res"]
    except (OSError, ValueError):
        pass

    def run():
        try:
            out = subprocess.run([str(DML_PY), "-c", DML_CANARY], capture_output=True, text=True, timeout=120,
                                 creationflags=NOWIN)
            d = json.loads(out.stdout.strip().splitlines()[-1])
            devs = d.get("devices") or []
            res = {"state": "ok" if any(x.get("ok") for x in devs) else "failed", "devices": devs,
                   "torch": d.get("torch")}
            if res["state"] == "failed":
                res["why"] = "no adapter passed the DirectML check"
        except Exception as e:
            res = {"state": "failed", "devices": [], "why": f"the DirectML check did not run: {str(e)[:160]}"}
        with _lock:
            _state["directml"] = res
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps({"directml": {"key": key, "t": time.time(), "res": res}}), encoding="utf-8")
        except OSError:
            pass

    with _lock:
        _state["directml"] = {"state": "probing", "devices": []}
    if wait:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()
    with _lock:
        return dict(_state["directml"])


def directml_adapters(include_nvidia=True):
    """DirectML devices that passed the local canary, ordered for Auto fallback.

    NVIDIA is first when present, followed by AMD and Intel; discrete adapters,
    measured throughput and memory break ties within those groups.
    """
    ad = {a["name"]: a for a in adapters()}
    rows = []
    for d in directml().get("devices", []):
        if not d.get("ok"):
            continue
        a = ad.get(d["name"]) or next((x for n, x in ad.items() if d["name"] in n or n in d["name"]), None)
        if not a or not a.get("ok", True) or (not include_nvidia and a["vendor"] == "nvidia"):
            continue
        rows.append({"index": d["index"], "name": a["name"], "vendor": a["vendor"],
                     "integrated": a["integrated"], "vram": a["vram"], "gbps": d.get("gbps") or 0})
    rows.sort(key=lambda a: (VENDOR_ORDER.get(a.get("vendor"), 3), a.get("integrated", False),
                             -(a.get("gbps") or 0), -(a.get("vram") or 0), a.get("index", 0)))
    return rows


def vk_ready():
    """sml-vk and its kernels are here (SML_VK=0 turns it off)."""
    return VK_EXE.exists() and (VK_DIR / "mv4.spv").exists() and os.environ.get("SML_VK", "1") != "0"


def _plain(name):
    return " ".join(str(name or "").lower().replace("(tm)", "").replace("(r)", "").split())


def vk_devices(refresh=False):
    """The GPUs Vulkan sees, from sml-vk --list: index, name, vendor, type (1 integrated,
    2 discrete, 3 virtual, 4 CPU), device-local and host-visible MB. [] without sml-vk."""
    if _vk["rows"] is not None and not refresh:
        return _vk["rows"]
    rows = []
    if vk_ready():
        try:
            r = subprocess.run([str(VK_EXE), "--list"], capture_output=True, text=True, timeout=60,
                               creationflags=NOWIN)
            for line in (r.stdout or "").splitlines():
                parts = line.strip().split("|")
                if len(parts) == 6 and parts[0].isdigit():
                    rows.append({"index": int(parts[0]), "name": parts[1].strip(),
                                 "vendor": VENDORS.get(parts[2].lower(), "other"), "type": int(parts[3]),
                                 "local_mb": float(parts[4]), "host_mb": float(parts[5])})
        except (OSError, subprocess.TimeoutExpired, ValueError):
            rows = []
    _vk["rows"] = rows
    return rows


def use_vk():
    """Whether the other GPUs run through sml-vk: then every index the planner hands
    ext_cmd is a Vulkan device index (else a DirectML one, as before)."""
    return vk_ready() and any(d["type"] in (1, 2, 3) for d in vk_devices())


def vk_pool(cuda_in_use=True):
    """Every GPU sml-vk can drive for text, any brand, matched to the adapters for memory
    and integrated; the NVIDIA card the CUDA engine drives is left out when cuda_in_use.
    gbps: the DirectML canary's figure for the same card when there is one, else 0 (the
    planner then assumes 14 GB/s integrated, 150 discrete)."""
    if not use_vk():
        return []
    ad = adapters()
    nv = sorted((a for a in ad if a["vendor"] == "nvidia"), key=lambda a: -a["vram"])
    cuda_name = _plain(nv[0]["name"]) if (nv and cuda_in_use) else None
    try:
        dml = {_plain(r["name"]): r.get("gbps") or 0 for r in directml_adapters()}
    except Exception:
        dml = {}
    rows = []
    for d in vk_devices():
        if d["type"] not in (1, 2, 3):
            continue                                    # software renderers (llvmpipe) are not GPUs
        name = _plain(d["name"])
        if cuda_name and d["vendor"] == "nvidia" and (name == cuda_name or name in cuda_name or cuda_name in name):
            continue
        a = next((x for x in ad if _plain(x["name"]) == name or _plain(x["name"]) in name or name in _plain(x["name"])), None)
        rows.append({"index": d["index"], "name": a["name"] if a else d["name"], "vendor": d["vendor"],
                     "integrated": a["integrated"] if a else d["type"] == 1,
                     "vram": (a or {}).get("vram") or d["local_mb"] * 1048576,
                     "gbps": dml.get(name, 0), "backend": "vk"})
    rows.sort(key=lambda a: (VENDOR_ORDER.get(a.get("vendor"), 3), a.get("integrated", False),
                             -(a.get("gbps") or 0), -(a.get("vram") or 0), a.get("index", 0)))
    return rows


def vk_agreement(model, index, timeout=900):
    """sml-vk's grade on this card: layer 0 here, the rest on the CPU, against the CPU
    alone on fixed ids. The orchestrator reads it with parse_gpu_agreement."""
    try:
        r = subprocess.run([str(VK_EXE), str(model), "--grade", "--device", str(int(index)), "--layers", "0:1",
                            "--ctx", "512"], capture_output=True, text=True, errors="replace", timeout=timeout,
                           creationflags=NOWIN)
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def amd_adapter():
    """The best AMD or Intel GPU for text models: through sml-vk when it is here, else the
    best one that passed the DirectML check."""
    if use_vk():
        rows = [r for r in vk_pool(cuda_in_use=False) if r["vendor"] != "nvidia"]
        return rows[0] if rows else None
    rows = directml_adapters(include_nvidia=False)
    best = rows[0] if rows else None
    if best and not (EXPORT_DLL.exists() and WORKER.exists() and DML_PY.exists()):
        best = dict(best, missing="the DirectML text worker or sml-export.dll is missing")
    return best


def dml_pool(cuda_in_use=True):
    """Every GPU the DirectML worker can use for text, of any brand: AMD, Intel and
    NVIDIA alike, except the one NVIDIA card the CUDA engine already drives when
    cuda_in_use. Discrete cards first, then by memory. [] when DirectML is not here.
    Ernest, 2026-10-07: "whatever brand ... as long as multiple GPUs are on the
    machine, make sure Sheep uses their strength"."""
    if use_vk():
        # every other GPU: sml-vk chains its workers card to card (--chain)
        return vk_pool(cuda_in_use)
    if not (EXPORT_DLL.exists() and WORKER.exists() and DML_PY.exists()):
        return []
    ad = adapters()
    rows, skipped_cuda = [], not cuda_in_use
    # the CUDA card: the NVIDIA adapter with the most memory
    nv = sorted((a for a in ad if a["vendor"] == "nvidia"), key=lambda a: -a["vram"])
    cuda_name = nv[0]["name"] if nv else None
    for row in directml_adapters():
        a = next((x for x in ad if x["name"] == row["name"]), None)
        if not a:
            continue
        if not skipped_cuda and a["name"] == cuda_name:
            skipped_cuda = True
            continue
        rows.append(row)
    return rows


def ext_cmd(model, ctx, layers=None, split=None):
    """The command sml-server runs (with --connect PORT added) for a model on
    DirectML GPUs, or "" when there is none. layers=(A, B): the worker holds only
    those, as the middle stage of a CUDA + DirectML + CPU model. split: [(adapter
    index, layer count), ...] to spread its layers over several GPUs, any brand."""
    span = f" --layers {int(layers[0])}:{int(layers[1])}" if layers else ""
    if use_vk():
        idx = int(split[0][0]) if split else (amd_adapter() or {}).get("index")
        if idx is None:
            return ""
        if split and len(split) > 1:
            # card to card: the first worker runs its layers, then starts the next card's
            # worker for the following ones (absolute layer ends), and so on
            at = int(layers[0]) if layers else 0
            first_end = at + int(split[0][1])
            ends, at = [], first_end
            for i, n in split[1:]:
                at += int(n)
                ends.append(f"{int(i)}:{at}")
            return (f'"{VK_EXE}" "{model}" --device {int(idx)} --ctx {int(ctx)} --layers {layers[0] if layers else 0}:{first_end}'
                    f' --chain {",".join(ends)}')
        return f'"{VK_EXE}" "{model}" --device {int(idx)} --ctx {int(ctx)}{span}'
    if split and len(split) > 1:
        ends, at = [], int(layers[0]) if layers else 0      # the worker reads absolute layer ends
        for idx, n in split:
            at += int(n)
            ends.append(f"{int(idx)}:{at}")
        return f'"{DML_PY}" "{WORKER}" "{model}" --adapters {",".join(ends)} --ctx {int(ctx)}{span}'
    if split:
        idx = int(split[0][0])
    else:
        a = amd_adapter()
        if not a or a.get("missing"):
            return ""
        idx = a["index"]
    return f'"{DML_PY}" "{WORKER}" "{model}" --adapter {idx} --ctx {int(ctx)}{span}'


def start():
    """At backend start: list the adapters and run the DirectML canary in the background."""
    threading.Thread(target=lambda: (adapters(), directml()), daemon=True).start()


def _nv(dev):
    return dev.get("gpu") if dev.get("gpu") and dev["gpu"].get("vram_total") else None


def amd_text(amd, others):
    a = amd_adapter()
    label = (a["name"].replace("(TM)", "").strip() if a else
             "Intel GPU" if any(x["vendor"] == "intel" for x in others) else "AMD GPU")
    if a and not a.get("missing"):
        why = f"the {a['name'].replace('(TM)', '')} through {'Vulkan' if a.get('backend') == 'vk' else 'DirectML'}" + (
            ": it shares system memory, so it is slower than the CPU here" if a["integrated"] else "")
        return {"id": "amd", "label": label, "ok": True, "why": why}
    if a:
        return {"id": "amd", "label": label, "ok": False, "why": a["missing"]}
    if not (amd or others):
        return {"id": "amd", "label": label, "ok": False, "why": "no AMD or Intel GPU found"}
    st = directml()
    return {"id": "amd", "label": label, "ok": False,
            "why": "checking the DirectML runtime…" if st["state"] == "probing" else
                   st.get("why") or "the GPU did not pass the DirectML check"}


def multi_text(gpu_ok, nv_name):
    """NVIDIA + AMD/Intel on one model: Sheep's layers on the NVIDIA card, the
    next on the other card, the rest on the CPU."""
    a = amd_adapter()
    ok = gpu_ok and bool(a) and not (a or {}).get("missing")
    amd_name = a["name"].replace("(TM)", "").strip() if a else "AMD / Intel GPU"
    why = (f"the first layers on the {nv_name}, the next on the {amd_name}, the rest on the CPU: "
           "for models larger than one card" if ok else
           "needs an NVIDIA GPU and an AMD or Intel GPU that passed its check")
    return {"id": "multi", "label": f"{nv_name} + {amd_name}", "ok": ok, "why": why}


def modes(dev):
    """The text and media choices, each with ok and a one-line reason."""
    ad = adapters()
    nv = _nv(dev)
    amd = [a for a in ad if a["vendor"] == "amd"]
    others = [a for a in ad if a["vendor"] in ("amd", "intel")]
    dml = directml()
    has_avx2 = avx2()
    nv_name = nv["name"].replace("NVIDIA GeForce ", "") if nv else ""
    gpu_ok = bool(nv) and PTX.exists()
    gpu_label = (nv_name if nv_name.lower().startswith("nvidia ") else f"NVIDIA {nv_name}") if nv_name else "NVIDIA GPU"
    gpu_why = (f"every layer on the {nv_name}" if gpu_ok else
               "the GPU kernels file (sml-gpu-fwd.ptx) is missing" if nv else "no NVIDIA GPU found")
    text = [
        {"id": "auto", "label": "Auto · recommended", "ok": True,
         "why": "NVIDIA first, then a checked AMD / Intel GPU, then CPU when the model fits"},
        {"id": "gpu", "label": gpu_label, "ok": gpu_ok, "why": gpu_why},
        {"id": "hybrid", "label": f"{gpu_label} + CPU", "ok": gpu_ok,
         "why": "the first layers on the GPU, the rest on the CPU: for models larger than the GPU" if gpu_ok else gpu_why},
        multi_text(gpu_ok, nv_name),
        amd_text(amd, others),
        {"id": "cpu", "label": "CPU", "ok": has_avx2,
         "why": "the processor and system RAM" if has_avx2 else "this CPU lacks AVX2, which Sheep's engine needs"},
    ]
    dml_names = [d["name"] for d in dml.get("devices", []) if d.get("ok") and "nvidia" not in d["name"].lower()]
    cuda_ok = bool(nv) and media_cuda()
    media = [
        {"id": "auto", "label": "Auto", "ok": True, "why": "the fastest that works for each model: NVIDIA, then AMD/Intel, then CPU"},
        {"id": "cuda", "label": gpu_label, "ok": cuda_ok,
         "why": f"CUDA on the {nv_name}" if cuda_ok else ("no NVIDIA GPU found" if not nv else "the CUDA runtime for media models is not installed")},
        {"id": "directml", "label": (dml_names[0] if dml_names else "AMD / Intel GPU"), "ok": bool(dml_names),
         "why": (f"DirectML on {', '.join(dml_names)}" if dml_names else
                 "checking the DirectML runtime…" if dml["state"] == "probing" else
                 "no AMD or Intel GPU found" if not others else dml.get("why") or "DirectML did not pass its check")},
        {"id": "cpu", "label": "CPU", "ok": True, "why": "always works; slowest for pictures and video"},
    ]
    return {"text": text, "media": media}


def setup_name(dev):
    """One line for the machine: "NVIDIA RTX 3050 Laptop GPU + AMD Radeon (integrated)"."""
    parts = []
    nv = _nv(dev)
    if nv:
        parts.append(nv["name"].replace("NVIDIA GeForce ", "NVIDIA "))
    for a in adapters():
        if a["vendor"] == "nvidia":
            continue
        parts.append(a["name"].replace("(TM)", "") + (" (integrated)" if a["integrated"] else ""))
    return " + ".join(parts) if parts else "CPU only"


def report(dev, prefs, training_dev=None):
    ad = adapters()
    nv = _nv(dev)
    rows = []
    for a in ad:
        live = nv if a["vendor"] == "nvidia" and nv else None
        dm = next((d for d in directml().get("devices", []) if d.get("name") and d["name"] in a["name"]), None)
        rows.append({**a, "vram": int(live["vram_total"]) if live else a["vram"],
                     "runs": [x for x, ok in (("Sheep engine (CUDA)", a["vendor"] == "nvidia" and PTX.exists()),
                                              ("DirectML", bool(dm and dm.get("ok")))) if ok]})
    from .training_policy import checker_training_advice, nvidia_thermal
    thermal = nvidia_thermal()
    # Training may borrow the GPU after Sheep moves Chat to CPU. Use that
    # projected VRAM for its recommendation while retaining live CPU RAM below.
    training_dev = training_dev or dev
    gpu = training_dev.get("gpu") or {}
    free_vram = max(0, (gpu.get("vram_total", 0) - gpu.get("vram_used", 0)) / 2**30) if gpu else None
    source_bytes = None
    checker_path = prefs.get("checker_path")
    if checker_path:
        try:
            from .checker import Checker
            source = Checker.source(checker_path)
            if source:
                root = Path(source)
                index = root / "model.safetensors.index.json"
                if index.is_file():
                    source_bytes = int(json.loads(index.read_text(encoding="utf-8")).get("metadata", {}).get("total_size") or 0)
                if not source_bytes:
                    source_bytes = sum(p.stat().st_size for pattern in ("*.safetensors", "*.bin", "*.pt")
                                       for p in root.glob(pattern) if p.is_file()) or None
        except (OSError, ValueError, TypeError):
            source_bytes = None
    available_ram = (dev.get("ram", {}).get("available") or 0) / 2**30
    training = checker_training_advice(cuda_ready=bool(gpu and torch_cuda()), free_vram_gb=free_vram,
                                       available_ram_gb=available_ram, source_bytes=source_bytes, thermal=thermal)
    training["gpu_fit_basis"] = "estimated after Chat moves to CPU" if training_dev is not dev else "currently free"
    training["gpu_free_gb"] = round(free_vram, 2) if free_vram is not None else None
    return {"setup": setup_name(dev), "adapters": rows, "avx2": avx2(), "directml": directml(),
            "modes": modes(dev), "prefs": prefs, "thermal": thermal, "checker_training": training}
