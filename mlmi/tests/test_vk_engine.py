"""sml-vk in the planner (2026-10-14): other GPUs driven by Sheep's Vulkan engine hold the
file's own 4/6-bit bytes, not fp16, so each holds about 3.6x the layers; a remembered
fault on one card leaves only that card out."""
from mlmi.backend import hardware, orchestrator as o

GB = 1e9
B7 = {"bytes": 6.2 * GB, "n_weights": 7.6e9, "n_layer": 28, "kv_per_token": 57344, "n_ctx_train": 32768,
      "widths": [6], "gpu_widths_ok": True}


def laptop(backend=None, ram=6):
    radeon = {"index": 0, "name": "AMD Radeon Graphics", "vendor": "amd", "integrated": True, "vram": 2 * GB,
              "gbps": 22}
    if backend:
        radeon["backend"] = backend
    return {"ram": {"available": ram * GB, "total": 14 * GB}, "cpu": {"cores": 6, "name": "CPU"},
            "power": {"plugged": True}, "gpu": {"name": "RTX 3050", "vram_total": 4 * GB, "vram_used": 0.3 * GB},
            "amd": dict(radeon), "pool": [dict(radeon)]}


def test_vulkan_cards_hold_far_more_layers_than_directml():
    dml = o.plan(laptop(), B7, "multi", pool_mode="always")
    vk = o.plan(laptop("vk"), B7, "multi", pool_mode="always")
    assert vk["device"] == "multi" and vk["mid_layers"] >= 3 * max(1, dml["mid_layers"]), (dml["mid_layers"], vk["mid_layers"])


def test_an_amd_fault_leaves_only_the_amd_card_out():
    p = o.plan(laptop("vk"), dict(B7, gpu_fault="amd"), "multi", pool_mode="always")
    assert p["mid_layers"] == 0 and p["gpu_layers"] > 0


def test_faults_combine_per_card():
    assert o.add_fault(None, "gpu") == "gpu"
    assert o.add_fault(None, "amd") == "amd"
    assert o.add_fault("gpu", "amd") == "multi"
    assert o.add_fault("amd", "hybrid") == "multi"
    assert o.add_fault("amd", "amd") == "amd"


def test_vk_list_is_parsed(monkeypatch):
    class R:
        stdout = "0|AMD Radeon(TM) Graphics|1002|1|2048|6838\n1|NVIDIA GeForce RTX 3050 Laptop GPU|10de|2|3965|7094\n"
    monkeypatch.setattr(hardware, "vk_ready", lambda: True)
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: R())
    rows = hardware.vk_devices(refresh=True)
    assert [(r["index"], r["vendor"], r["type"]) for r in rows] == [(0, "amd", 1), (1, "nvidia", 2)]
    hardware._vk["rows"] = None


def test_the_check_line_is_read_by_the_same_parser():
    out = "  GPU vs CPU over 7 positions: mean logit corr 0.999740, worst 0.999476, top-1 agree 7/7"
    assert o.parse_gpu_agreement(out)["ok"] is True
    assert o.parse_gpu_agreement("  not on Vulkan yet: experts or state-space layers")["ok"] is None


def test_a_vulkan_chain_holding_every_layer_leaves_the_last_to_the_cpu():
    # two discrete cards through sml-vk that hold the whole model: the server's CPU stage
    # needs a layer to run (with none it answered garbage), so the last one stays there
    cards = [{"index": 0, "name": "RX 7600", "vendor": "amd", "integrated": False, "vram": 8 * GB, "gbps": 200, "backend": "vk"},
             {"index": 1, "name": "Arc A770", "vendor": "intel", "integrated": False, "vram": 16 * GB, "gbps": 300, "backend": "vk"}]
    dev = {"ram": {"available": 28 * GB, "total": 32 * GB}, "cpu": {"cores": 8, "name": "CPU"}, "power": {"plugged": True},
           "gpu": None, "amd": dict(cards[1]), "pool": cards}
    p = o.plan(dev, dict(B7, bytes=9 * GB, n_weights=14e9, n_layer=48), "amd")
    assert p["device"] == "amd" and len(p["dml_split"]) == 2 and p["mid_layers"] == 47, p
