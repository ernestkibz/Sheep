"""The quantize pick follows the machine (2026-10-14): the best quality the GPUs hold
whole, else the build that puts the most of the model on them. Any brand, any mix."""
from mlmi.backend import levels, orchestrator as o

GB = 1e9
SHAPES = (("attn_q", 3584, 3584), ("attn_k", 512, 3584), ("attn_v", 512, 3584), ("attn_output", 3584, 3584),
          ("ffn_gate", 18944, 3584), ("ffn_up", 18944, 3584), ("ffn_down", 3584, 18944))
INFO = {"n_layer": 28, "n_weights": 7.6e9, "n_ctx_train": 32768, "n_kv": 4, "head_dim": 128,
        "tensors": [(f"blk.{l}.{n}.weight", (r, c)) for l in range(28) for n, r, c in SHAPES]
        + [("token_embd.weight", (152064, 3584)), ("output.weight", (152064, 3584))]}


def machine(nv, ram, pool=(), amd=None):
    return {"ram": {"available": ram * GB, "total": (ram + 3) * GB}, "cpu": {"cores": 6, "name": "CPU"},
            "power": {"plugged": True},
            "gpu": {"name": nv[0], "vram_total": nv[1] * GB, "vram_used": 0.3 * GB} if nv else None, "amd": amd,
            "pool": [{"index": 1, "name": p[0], "vendor": "amd", "vram": p[1] * GB, "integrated": p[2]} for p in pool]}


def pick(dev, pref="auto"):
    return next(c for c in levels.cards(INFO, dev, o.plan, pref) if c["recommended"])


def test_a_small_card_gets_the_build_that_puts_most_on_it():
    c = pick(machine(("RTX 3050", 4), 10, [("Radeon", 2, True)],
                     {"index": 1, "name": "Radeon", "integrated": True, "vram": 2 * GB}))
    assert c["id"] == "balanced" and c["device"] == "hybrid"


def test_a_big_card_gets_the_best_quality_it_holds_whole():
    c = pick(machine(("RTX 4070", 12), 16))
    assert c["id"] == "high" and c["device"] == "gpu" and "whole" in c["why_recommended"]


def test_no_gpu_still_gets_a_pick_of_4_bit_quality_or_better():
    c = pick(machine(None, 16))
    assert c["device"] == "cpu" and c["id"] in ("small", "balanced", "q5", "high")


def test_never_a_level_the_nvidia_card_cannot_run_as_the_whole_gpu_pick():
    c = pick(machine(("RTX 4090", 24), 32))
    assert c["id"] in ("small", "balanced", "high")         # Q8 would run on the CPU only
