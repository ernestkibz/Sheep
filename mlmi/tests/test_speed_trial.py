"""The speed trial (2026-10-14): Auto takes the mode a model ran fastest in on this
machine, so a device joins only where it measurably helps."""
from mlmi.backend import orchestrator as o

DEV = {"ram": {"available": 10e9, "total": 14e9}, "cpu": {"cores": 6, "name": "CPU"}, "power": {"plugged": True},
       "gpu": {"name": "RTX 3050", "vram_total": 4e9, "vram_used": 0.3e9},
       "amd": {"index": 1, "name": "AMD Radeon Graphics", "integrated": True, "vram": 2e9},
       "pool": [{"index": 1, "name": "AMD Radeon Graphics", "vendor": "amd", "vram": 2e9, "integrated": True}]}
B7 = {"bytes": 6.2e9, "n_weights": 7.6e9, "n_layer": 28, "kv_per_token": 57344, "n_ctx_train": 32768,
      "widths": [6], "gpu_widths_ok": True}


def test_auto_follows_the_measured_winner():
    p = o.plan(DEV, dict(B7, measured_best="hybrid", measured_note="cpu 4.95, hybrid 5.35, multi 4.0"), "auto")
    assert p["device"] == "hybrid"
    assert any("measured fastest" in w for w in p["why"])


def test_auto_follows_a_cpu_winner_too():
    p = o.plan(DEV, dict(B7, measured_best="cpu", measured_note="cpu 5.0, hybrid 3.1"), "auto")
    assert p["device"] == "cpu" and p["gpu_layers"] == 0


def test_a_chosen_mode_is_not_overridden_by_the_trial():
    p = o.plan(DEV, dict(B7, measured_best="cpu"), "hybrid")
    assert p["device"] == "hybrid"


def test_the_machine_signature_changes_with_the_hardware():
    a = o.hardware_sig(DEV)
    b = o.hardware_sig(dict(DEV, gpu={"name": "RTX 4070", "vram_total": 12e9, "vram_used": 0}))
    assert a != b and "RTX 3050" in a


def test_charger_and_battery_get_their_own_trial():
    on_ac = dict(DEV, power={"plugged": True})
    on_bat = dict(DEV, power={"plugged": False, "percent": 60})
    assert o.hardware_sig(on_ac).endswith("|ac") and o.hardware_sig(on_bat).endswith("|battery")


def test_a_desktop_without_a_battery_is_ac():
    assert o.hardware_sig(dict(DEV, power={"percent": None, "plugged": True})).endswith("|ac")
