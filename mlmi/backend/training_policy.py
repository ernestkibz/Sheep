"""Thermal-aware placement policy for optional Checker training.

The policy is intentionally advisory when a sensor is unavailable: CPU is the
default, while a user may explicitly choose short GPU chunks.
"""
import re
import shutil
import subprocess


def nvidia_thermal():
    """Return NVIDIA temperatures and driver limits, or an explicit no-sensor result."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "NVIDIA temperature telemetry is unavailable."}
    try:
        result = subprocess.run([exe, "-q", "-d", "TEMPERATURE"], capture_output=True, text=True,
                                timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"GPU temperature could not be read: {exc}"}
    if result.returncode:
        return {"available": False, "reason": (result.stderr or "nvidia-smi temperature query failed")[:180]}

    values = {}
    for label, key in (("GPU Current Temp", "current_c"), ("GPU Target Temperature", "target_c"),
                       ("GPU Slowdown Temp", "slowdown_c"), ("GPU Shutdown Temp", "shutdown_c")):
        match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(\d+)\s*C\s*$", result.stdout, re.M | re.I)
        values[key] = int(match.group(1)) if match else None
    values["available"] = values["current_c"] is not None
    values["reason"] = "" if values["available"] else "NVIDIA did not report a current GPU temperature."
    if values["available"]:
        # The driver may throttle at target. Keep optional training below the
        # slowdown point, then resume at the reported target temperature.
        caps = [x for x in (values["slowdown_c"] - 8 if values["slowdown_c"] else None,
                            values["shutdown_c"] - 11 if values["shutdown_c"] else None) if x is not None]
        values["pause_at_c"] = min(caps) if caps else None
        values["resume_below_c"] = (values["target_c"] if values["target_c"] is not None else
                                    values["pause_at_c"] - 2 if values["pause_at_c"] is not None else None)
    return values


def checker_training_advice(*, cuda_ready=False, free_vram_gb=None, available_ram_gb=None,
                            source_bytes=None, thermal=None, need_vram_gb=None):
    """Describe CPU and bounded-GPU choices without silently selecting GPU training."""
    thermal = thermal or {"available": False}
    if source_bytes:
        # The trainer uses a single-example CUDA micro-batch with accumulation,
        # keeping the effective batch unchanged while lowering activation memory.
        # Reserve room for CUDA workspaces and scale for larger source weights.
        need_vram_gb = max(2.5, source_bytes / 2**30 * 0.25 + 1.0)
    elif need_vram_gb is None:
        need_vram_gb = 2.5
    cpu_need_gb = max(6.0, (source_bytes or 0) / 2**30 * 2.0 + 2.0)
    # Unknown RAM is not evidence that a full-precision CPU training job fits.
    cpu_fits = available_ram_gb is not None and available_ram_gb >= cpu_need_gb
    fits = cuda_ready and free_vram_gb is not None and free_vram_gb >= need_vram_gb
    if not cuda_ready:
        recommended, reason = "cpu", ("CUDA training is unavailable; use the CPU." if cpu_fits else
                                        f"Neither training device fits: CPU needs about {cpu_need_gb:.1f} GB available RAM.")
    elif not fits:
        recommended, reason = ("cpu", f"The GPU needs about {need_vram_gb:.1f} GB free; use the CPU for this model.") if cpu_fits else (
            "cpu", f"This job exceeds current fit: GPU needs about {need_vram_gb:.1f} GB free; CPU needs about {cpu_need_gb:.1f} GB available RAM.")
    elif not thermal.get("available"):
        recommended, reason = "cpu", "Temperature cannot be measured; CPU is safer for a long training run."
    elif thermal.get("pause_at_c") is None:
        recommended, reason = "cpu", "The temperature sensor has no usable driver limit; choose CPU or short GPU chunks."
    elif thermal.get("pause_at_c") is not None and thermal.get("current_c", 0) >= thermal["pause_at_c"]:
        recommended, reason = "cpu", (f"The GPU is already at {thermal['current_c']} °C, near its training pause point "
                                       f"of {thermal['pause_at_c']} °C.")
    else:
        recommended, reason = "gpu_bounded", "CUDA fits and temperature is readable; training will checkpoint and pause at the driver-aware limit."
    return {"recommended": recommended, "reason": reason, "gpu_available": bool(fits),
            "cpu_available": bool(cpu_fits), "cpu_need_gb": round(cpu_need_gb, 1),
            "gpu_need_gb": round(need_vram_gb, 1),
            "thermal": thermal, "bounded_gpu_chunk_seconds": 300 if fits and not thermal.get("pause_at_c") else None,
            "sensorless_chunk_seconds": 300 if fits and not thermal.get("available") else None,
            "need_vram_gb": need_vram_gb,
            "fit_basis": "available now; GPU is rechecked after Chat yields"}
