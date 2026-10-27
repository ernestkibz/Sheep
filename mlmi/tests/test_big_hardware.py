"""Bigger and different hardware (Part C7, 2026-10-21): with several NVIDIA cards the one with the
most memory is used and every program Sheep starts is pinned to it; a 14B model plans fully onto
an 80 GB card; a machine with no NVIDIA card plans on its CPU or its AMD/Intel GPU."""
from mlmi.backend import orchestrator as o

SMI = ("0, NVIDIA GeForce RTX 3050, 4096, 300, 50, 1500, 1800, 20.5, 3\n"
       "1, NVIDIA H100 80GB HBM3, 81559, 1200, 40, 1980, 1980, 70.1, 0\n"
       "2, NVIDIA H100 80GB HBM3, 81559, 9000, 41, 1980, 1980, 90.0, 12\n")
GB = 2**30
F14 = {"bytes": 9.0e9, "n_weights": 14.8e9, "n_layer": 48, "kv_per_token": 196608, "n_ctx_train": 32768}


def dev(gpu=None, amd=None, ram=64e9, cores=16, name="CPU"):
    return {"ram": {"available": ram, "total": ram * 1.1}, "cpu": {"cores": cores, "threads": cores * 2, "name": name},
            "power": {"plugged": True}, "gpu": gpu, "amd": amd, "pool": [amd] if amd else [], "others": []}


def test_the_card_with_most_memory_and_least_in_use_is_picked():
    g = o.pick_nvidia(SMI)
    assert (g["index"], g["count"]) == (1, 3)
    assert g["vram_total"] == 81559 * 2**20
    assert "NVIDIA GeForce RTX 3050" in g["others"]


def test_one_card_keeps_the_old_reading():
    g = o.pick_nvidia("0, NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 500, 55, 1200, 1800, 15.0, 7\n")
    assert g["name"] == "NVIDIA GeForce RTX 3050 Laptop GPU" and g["count"] == 1 and g["util"] == 7
    assert o.pick_nvidia("") is None


def test_programs_are_pinned_only_when_there_are_several_cards():
    env = o.cuda_env({}, gpu=o.pick_nvidia(SMI))
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}
    assert o.cuda_env({}, gpu={"index": 0, "count": 1}) == {}
    assert o.cuda_env({"CUDA_VISIBLE_DEVICES": "2"}, gpu=o.pick_nvidia(SMI))["CUDA_VISIBLE_DEVICES"] == "2"


def test_a_14b_model_runs_wholly_on_an_80gb_card():
    p = o.plan(dev(gpu=dict(o.pick_nvidia(SMI))), F14)
    assert p["device"] == "gpu" and p["gpu"] == {"fit": "whole", "layers": 48}, p
    assert p["ctx"] == 32768, p                       # the card's room, not the 4 GB card's 4096


def test_a_4gb_card_keeps_its_4096_context():
    small = {"index": 0, "count": 1, "name": "RTX 3050", "vram_total": 4096 * 2**20, "vram_used": 300 * 2**20}
    p = o.plan(dev(gpu=small, ram=16e9), {"bytes": 1.9e9, "n_weights": 3.1e9, "n_layer": 36, "kv_per_token": 36864,
                                         "n_ctx_train": 32768})
    assert p["device"] == "gpu" and p["ctx"] == 4096, p


def test_an_intel_laptop_with_no_gpu_runs_on_its_cpu():
    p = o.plan(dev(ram=15e9, cores=8, name="Intel(R) Core(TM) i7-1165G7"), dict(F14, bytes=2.0e9, n_layer=36))
    assert p["device"] == "cpu" and p["gpu_layers"] == 0


def test_a_model_too_big_for_memory_says_so():
    p = o.plan(dev(ram=6e9, cores=4), dict(F14, bytes=9.0e9))
    assert p.get("fits") is False or "RAM" in " ".join(p.get("why", [])), p
