"""The check before a model first runs on the NVIDIA card (2026-10-14): the card must
compute the model as the CPU does, or the model is planned without it."""
from mlmi.backend import orchestrator as o


def test_a_right_gpu_path_passes():
    r = o.parse_gpu_agreement("  GPU vs CPU over 7 positions: mean logit corr 0.999544, worst 0.997839, top-1 agree 7/7")
    assert r["ok"] is True and r["agree"] == 7


def test_one_near_tie_is_allowed():
    assert o.parse_gpu_agreement("mean logit corr 0.9998, worst 0.9995, top-1 agree 6/7")["ok"] is True


def test_the_overrun_bug_is_caught():
    # Qwen2.5-7B on the standard kernels, 2026-10-14
    r = o.parse_gpu_agreement("  GPU vs CPU over 7 positions: mean logit corr 0.014100, worst -0.028900, top-1 agree 0/7")
    assert r["ok"] is False


def test_a_model_the_engine_keeps_off_the_gpu_is_not_a_fault():
    r = o.parse_gpu_agreement("  inputs 40000 wide: more than the GPU kernels stage (32768): not on the GPU yet")
    assert r["ok"] is None and "not on the GPU" in r["why"]


def test_a_remembered_fault_keeps_every_mode_off_the_card():
    dev = {"ram": {"available": 28e9, "total": 32e9}, "cpu": {"cores": 8, "name": "CPU"}, "power": {"plugged": True},
           "gpu": {"name": "RTX 3050", "vram_total": 4e9, "vram_used": 0.4e9}, "amd": None, "pool": []}
    facts = {"bytes": 6.2e9, "n_weights": 7.6e9, "n_layer": 28, "kv_per_token": 57344, "n_ctx_train": 32768,
             "gpu_fault": "hybrid"}
    for pref in ("auto", "gpu", "hybrid", "multi"):
        assert o.plan(dev, facts, pref)["gpu_layers"] == 0, pref


def test_near_ties_on_a_right_path_are_not_a_fault():
    # the 3B on its own card, 2026-10-14: corr 0.9997 but the top word 5/7 on fixed ids
    assert o.parse_gpu_agreement("mean logit corr 0.999700, worst 0.999500, top-1 agree 5/7")["ok"] is True
    assert o.agreement_ok({"corr": 0.995, "agree": 4, "of": 7}) is False
    assert o.agreement_ok({"corr": 0.995, "agree": 6, "of": 7}) is True
