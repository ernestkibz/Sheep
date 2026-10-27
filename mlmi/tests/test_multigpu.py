"""The orchestrator pools a machine's GPUs, any brand, for a model too big for one
(Ernest, 2026-10-07). The plan is pure, so machines this laptop does not have are
described here and planned: which device holds which layers, and why.

    python -m mlmi.tests.test_multigpu
"""
import sys
import unittest
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("CLI regression script; run python -m mlmi.tests.test_multigpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend.orchestrator import plan  # noqa: E402

GB = 1e9
fails, ran = [], [0]


def check(name, ok, got=""):
    ran[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      got {got}"))
    if not ok:
        fails.append(name)


def machine(nv=None, pool=(), amd=None, ram=28 * GB):
    """nv: (name, total GB, used GB); pool: [(index, name, vram GB, integrated)]."""
    return {"ram": {"available": ram, "total": ram + 4 * GB}, "cpu": {"cores": 8, "name": "CPU"},
            "power": {"plugged": True},
            "gpu": {"name": nv[0], "vram_total": nv[1] * GB, "vram_used": nv[2] * GB} if nv else None,
            "amd": amd, "pool": [{"index": t[0], "name": t[1], "vendor": "x", "vram": t[2] * GB, "integrated": t[3],
                                  "gbps": t[4] if len(t) > 4 else 0} for t in pool]}


# a 7B at Q4/6 (Qwen2.5-Coder-7B-like) and a 3B
B7 = {"bytes": 4.6 * GB, "n_weights": 7.6e9, "n_layer": 28, "kv_per_token": 57344, "n_ctx_train": 32768}
B3 = {"bytes": 2.0 * GB, "n_weights": 3.2e9, "n_layer": 28, "kv_per_token": 114688, "n_ctx_train": 131072}


def show(p):
    return f"{p['device']} cuda={p['gpu_layers']} mid={p['mid_layers']} split={p['dml_split']} est={p['est_tok_s']} | " \
           + " / ".join(w for w in p["why"] if "GPU" in w or "pooled" in w or "on the" in w)


# 1. this laptop: RTX 3050 4 GB + an integrated Radeon (shared memory, slower than the CPU)
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)],
                 amd={"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2 * GB}), B7)
check("laptop: the integrated GPU is left out, the 7B splits NVIDIA + CPU", p["device"] == "hybrid" and not p["dml_split"], show(p))

# 2. NVIDIA 4 GB + a discrete AMD RX 7600 8 GB: the 7B across both cards
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon RX 7600", 8, False)]), B7)
check("NVIDIA + discrete AMD: pooled", p["device"] == "multi" and p["mid_layers"] > 0 and p["dml_split"][0][0] == 1, show(p))
check("NVIDIA + discrete AMD: every layer has a home", p["gpu_layers"] + p["mid_layers"] <= 28 and p["gpu_layers"] >= 1, show(p))

# 3. two NVIDIA cards: the second one through DirectML (same brand)
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "NVIDIA GeForce RTX 3060", 12, False)]), B7)
check("two NVIDIA cards: the second joins through DirectML", p["device"] == "multi"
      and p["gpu_layers"] + p["mid_layers"] == 28, show(p))

# 4. three GPUs: NVIDIA 4 GB + AMD 8 GB + Intel Arc 8 GB, a model bigger than any one
B14 = {"bytes": 9.0 * GB, "n_weights": 14.8e9, "n_layer": 48, "kv_per_token": 196608, "n_ctx_train": 32768}
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon RX 7600", 8, False), (2, "Intel Arc A770", 16, False)],
                 ram=64 * GB), B14)
check("three GPUs: both other cards take layers, the larger first",
      p["device"] == "multi" and len(p["dml_split"]) == 2 and p["dml_split"][0][0] == 2, show(p))

# 5. no NVIDIA, two AMD cards of 12 GB: the whole 7B across both
p = plan(machine(None, [(0, "AMD Radeon RX 7700 XT", 12, False), (1, "AMD Radeon RX 7700 XT", 12, False)]), B7)
check("two AMD cards, no NVIDIA: the model spread across both", p["device"] == "amd"
      and sum(n for _, n in p["dml_split"]) == 28 and len(p["dml_split"]) == 2, show(p))

# 6. the model fits the NVIDIA card whole: no pooling (one card is fastest)
p = plan(machine(("RTX 4070", 12, 0.5), [(1, "AMD Radeon RX 7600", 8, False)]), B3)
check("fits one card: the whole model on it, no pooling", p["device"] == "gpu", show(p))

# 7. Settings > multi with Pool GPUs "always": the integrated GPU's memory is used
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)],
                 amd={"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2 * GB}), B7, "multi", "always")
check("Pool GPUs always: the integrated GPU joins when asked", p["device"] == "multi" and p["mid_layers"] > 0, show(p))
# 7a. ...with "smart" (the default): it is slower than the CPU, so it is left out and said
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)],
                 amd={"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2 * GB}), B7, "multi", "smart")
check("Pool GPUs smart: a GPU slower than the CPU is left out, with the reason",
      p["device"] != "multi" and any("left out" in w for w in p["why"]), show(p))
# 7c. a discrete card more than `gap` times slower a layer than the NVIDIA card stays out
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "Slow Card", 8, False, 100)]), B7, "auto", "smart", 2.0)
check("Pool GPUs smart, gap 2: a card well behind the main GPU is left out", p["device"] != "multi", show(p))
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "Slow Card", 8, False, 100)]), B7, "auto", "smart", 8.0)
check("...and joins with gap 8 (still faster than the CPU)", p["device"] == "multi", show(p))
# 7d. Pool GPUs off: NVIDIA and the CPU only, whatever else is there
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon RX 7600", 8, False)]), B7, "auto", "off")
check("Pool GPUs off: no other GPU", not p["dml_split"] and p["device"] in ("hybrid", "gpu"), show(p))

# 7b. Settings > multi with a model that fits the NVIDIA card whole: it runs there alone.
# SmolLM2 1.7B was split in half onto the integrated GPU and every step crawled (2026-10-13)
B17 = {"bytes": 1.0 * GB, "n_weights": 1.7e9, "n_layer": 24, "kv_per_token": 49152, "n_ctx_train": 8192}
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)],
                 amd={"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2 * GB}), B17, "multi")
check("Settings > multi, model fits the NVIDIA card: no split", p["device"] == "gpu" and not p["dml_split"], show(p))

# 7e. the person's switches: a GPU off, VRAM kept free, no GPU+CPU split
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon RX 7600", 8, False)]), B7, "auto", "smart", 4.0,
         exclude=["AMD Radeon RX 7600"])
check("a GPU switched off is not used", not p["dml_split"], show(p))
p = plan(machine(("RTX 3050", 4, 0.4)), B3, "auto", "smart", 4.0)
p2 = plan(machine(("RTX 3050", 4, 0.4)), B3, "auto", "smart", 4.0, reserve_gb=2.0)
check("VRAM kept free: fewer layers go on the GPU", p2["gpu_layers"] < p["gpu_layers"]
      and any("kept free" in w for w in p2["why"]), f"{show(p)} || {show(p2)}")
p = plan(machine(("RTX 3050", 4, 0.4)), B7, "auto", "smart", 4.0, cpu_offload=False)
check("GPU+CPU off: a 7B too big for the card runs on the CPU alone", p["device"] == "cpu", show(p))
p = plan(machine(("RTX 3050", 4, 0.4)), B7, "auto", "smart", 4.0, exclude=["RTX 3050"])
check("the main GPU switched off: no CUDA layers", p["gpu_layers"] == 0, show(p))

# 8. pooling is faster than leaving the layers on the CPU
p_pool = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon RX 7600", 8, False)]), B7)
p_cpu = plan(machine(("RTX 3050", 4, 0.4)), B7)
check("pooled estimate beats NVIDIA + CPU", p_pool["est_tok_s"] > p_cpu["est_tok_s"],
      f"{p_pool['est_tok_s']} vs {p_cpu['est_tok_s']}")
print(f"  (estimates: pooled {p_pool['est_tok_s']} tok/s, NVIDIA + CPU {p_cpu['est_tok_s']} tok/s)")

# 9. no NVIDIA, one AMD 8 GB card, a 14B: the card takes the first layers, the CPU the rest
p = plan(machine(None, [(0, "AMD Radeon RX 7600", 8, False)], ram=64 * GB), B14)
check("AMD only, model larger than the card: front layers on it, the rest on the CPU",
      p["device"] == "amd" and 0 < p["mid_layers"] < 48 and p["dml_split"][0][0] == 0
      and sum(n for _, n in p["dml_split"]) == p["mid_layers"], show(p))
p_cpu = plan(machine(None, ram=64 * GB), B14)
check("AMD + CPU beats the CPU alone", p["est_tok_s"] > p_cpu["est_tok_s"], f"{p['est_tok_s']} vs {p_cpu['est_tok_s']}")

# 10. no NVIDIA, one AMD 24 GB card, the 7B (15 GB in fp16) fits: the whole model on it
p = plan(machine(None, [(0, "AMD Radeon RX 7900 XTX", 24, False)]), B7)
check("AMD only, model fits: the whole model on the card", p["device"] == "amd" and p["mid_layers"] == 0, show(p))

# 11. the front stage's command: the worker holds layers 0..M and embeds the ids
from mlmi.backend import hardware  # noqa: E402
import os  # noqa: E402
# the DirectML worker's command (sml-vk off); the Vulkan one is checked after
os.environ["SML_VK"] = "0"
cmd = hardware.ext_cmd("m.ssf", 4096, (0, 20), [[0, 20]])
check("front stage command: --layers 0:20 on adapter 0", "--layers 0:20" in cmd and "--adapter 0" in cmd, cmd)
cmd = hardware.ext_cmd("m.ssf", 4096, (0, 30), [[1, 18], [0, 12]])
check("front stage on two cards: absolute ends 1:18,0:30", "--adapters 1:18,0:30" in cmd and "--layers 0:30" in cmd, cmd)
os.environ.pop("SML_VK", None)
if hardware.use_vk():
    cmd = hardware.ext_cmd("m.ssf", 4096, (12, 22), [[0, 10]])
    check("Vulkan middle stage: sml-vk on device 0, layers 12:22", "sml-vk" in cmd and "--device 0" in cmd
          and "--layers 12:22" in cmd, cmd)

# Auto keeps the NVIDIA/CUDA stage first when it has room, then uses the DirectML pool.
# 12. a 4 GB NVIDIA card + an RX 7900 XTX measured at 700 GB/s: both cards take layers
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon RX 7900 XTX", 24, False, 700)]), B7)
check("Auto uses NVIDIA first, then the AMD card",
      p["device"] == "multi" and p["gpu_layers"] > 0 and p["mid_layers"] > 0
      and p["dml_split"][0][0] == 1, show(p))

# 13. a discrete card measured slower per layer than the CPU (30 GB/s in fp16): left out
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "Old GPU", 8, False, 30)]), B7)
check("by speed: a card slower than the CPU is not pooled", p["device"] == "hybrid" and not p["dml_split"], show(p))

# 14. two cards, no NVIDIA: the faster one (smaller) takes the first layers
p = plan(machine(None, [(0, "Intel Arc A770", 16, False, 200), (1, "AMD Radeon RX 7600", 8, False, 400)], ram=64 * GB), B14)
check("by speed: the faster card fills first", p["device"] == "amd" and p["dml_split"][0][0] == 1, show(p))

# 15. the model fits the NVIDIA card, the other card is only a little faster: stay on CUDA
p = plan(machine(("RTX 4070", 12, 0.5), [(1, "AMD Radeon RX 7600", 8, False, 250)]), B3)
check("fits the NVIDIA card and the other is not much faster: stays on CUDA", p["device"] == "gpu", show(p))

# 16. a Q8 7B: the NVIDIA engine has no 8-bit kernel, so no plan may put layers on it
#     (2026-10-14: planned "hybrid, 9 of 28 layers on the GPU", then ran on the CPU)
B7Q8 = dict(B7, bytes=8.1 * GB, widths=[8], gpu_widths_ok=False)
for pref in ("auto", "gpu", "hybrid", "multi"):
    p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)],
                     amd={"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2 * GB}), B7Q8, pref)
    check(f"Q8 7B, {pref}: no layers on the NVIDIA card, and it says why",
          p["gpu_layers"] == 0 and p["device"] not in ("gpu", "hybrid", "multi")
          and any("8-bit" in w for w in p["why"]), show(p))
# the same 7B at Q6 still splits NVIDIA + CPU
p = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)],
                 amd={"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2 * GB}),
         dict(B7, bytes=6.2 * GB, widths=[6], gpu_widths_ok=True))
check("Q6 7B on the laptop: NVIDIA + CPU", p["device"] == "hybrid" and p["gpu_layers"] > 0, show(p))

# 17. a model larger than the free RAM: too big on the CPU alone, but hybrid fits it,
#     because the GPU holds part of it (2026-10-14: RAM was charged the whole file)
B12 = {"bytes": 8.0 * GB, "n_weights": 13e9, "n_layer": 40, "kv_per_token": 81920, "n_ctx_train": 32768,
       "widths": [4], "gpu_widths_ok": True}
lap = machine(("RTX 3050", 4, 0.4), ram=7 * GB)
p_cpu, p_hyb = plan(lap, B12, "cpu"), plan(lap, B12, "hybrid")
check("8 GB model, 7 GB RAM free: too big on the CPU alone", not p_cpu["ram_fit"], show(p_cpu))
check("8 GB model, 7 GB RAM free: hybrid fits it (the GPU holds part)", p_hyb["ram_fit"] and p_hyb["device"] == "hybrid",
      show(p_hyb))

# 18. more = better: on multi, the integrated Radeon's own 2 GB joins when NVIDIA + RAM cannot hold the model
radeon = {"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2 * GB}
lap2 = machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)], amd=radeon, ram=6 * GB)
p_m = plan(lap2, B12, "multi")
check("multi, model too big for NVIDIA + RAM: the Radeon joins for room",
      p_m["device"] == "multi" and p_m["mid_layers"] > 0 and any("joined for room" in w for w in p_m["why"]), show(p_m))
p_small = plan(machine(("RTX 3050", 4, 0.4), [(1, "AMD Radeon Graphics", 2, True)], amd=radeon), B7, "multi")
check("multi, model fits NVIDIA + RAM: the slower Radeon is left out", p_small["mid_layers"] == 0, show(p_small))

# 19. autonomy: a model whose NVIDIA path faulted in its checks is planned without the card, in every mode
for pref in ("auto", "gpu", "hybrid", "multi"):
    p = plan(machine(("RTX 3050", 4, 0.4)), dict(B7, gpu_fault="hybrid"), pref)
    check(f"NVIDIA path faulted, {pref}: no layers on it, and it says why",
          p["gpu_layers"] == 0 and any("wrong answers" in w for w in p["why"]), show(p))

print(f"\n{'all ' + str(ran[0]) + ' checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
