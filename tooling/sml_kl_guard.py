"""Build and measure one local KL level without starving the workstation."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scratchpad" / "audit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tooling"))

import psutil
from mlmi.backend.levels import inspect
from mlmi.backend.quantizer import recipe
from sml_kl_levels import kl_measure, speed_measure

SOURCE = Path(r"C:\Users\earne\models\llama32-3b")
PROMPTS = [
    "Sheep helps people create useful files and videos on a local device. The quality of a smaller model depends on careful conversion, the inference hardware, and the selected quantization level.",
    "A good response answers the question, follows the user's request, and avoids inventing facts.",
    "For video, Sheep can animate selected photos, make an exact-photo montage, or generate from text.",
]
MEMORY_FLOOR = int(2.25 * 1024**3)
# A quantizer can allocate over a gigabyte between one-second polls. Require
# enough startup headroom before launching, and begin shutdown well above the
# hard floor so a fast allocation does not consume the safety margin.
STARTUP_HEADROOM = int(2.0 * 1024**3)
STOP_HEADROOM = int(0.75 * 1024**3)
POLL_SECONDS = 0.25


def can_start_with_memory(available_bytes):
    return int(available_bytes) >= MEMORY_FLOOR + STARTUP_HEADROOM


def should_stop_for_memory(available_bytes):
    return int(available_bytes) < MEMORY_FLOOR + STOP_HEADROOM


def persist(path, report):
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def stop_tree(proc):
    try:
        root = psutil.Process(proc.pid)
        children = root.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            root.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs(children + [root], timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def build_with_memory_floor(source, level, output):
    available_before_start = psutil.virtual_memory().available
    minimum = available_before_start
    if not can_start_with_memory(available_before_start):
        return {"state": "resource_blocked", "minimum_available_ram_bytes": minimum,
                "available_ram_bytes_at_stop": available_before_start,
                "memory_floor_bytes": MEMORY_FLOOR,
                "required_available_ram_to_start_bytes": MEMORY_FLOOR + STARTUP_HEADROOM,
                "partial_file": None, "partial_bytes": 0,
                "reason": "not started: insufficient RAM headroom to preserve the 2.25 GiB safety floor"}

    info = inspect(str(source))
    if not info.get("builds"):
        raise RuntimeError(f"source architecture is not buildable: {info.get('note', '')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    command = recipe(str(source), str(part), info["n_layer"], 0, info.get("licence", ""), level)
    proc = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        psutil.Process(proc.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (psutil.Error, AttributeError):
        pass
    tail = []
    lines = queue.Queue()

    def collect_output():
        for line in proc.stdout:
            lines.put(line.rstrip())

    reader = threading.Thread(target=collect_output, daemon=True)
    reader.start()
    while proc.poll() is None:
        available = psutil.virtual_memory().available
        minimum = min(minimum, available)
        if should_stop_for_memory(available):
            stop_tree(proc)
            after_stop = psutil.virtual_memory().available
            minimum = min(minimum, after_stop)
            return {"state": "resource_blocked", "minimum_available_ram_bytes": minimum,
                    "available_ram_bytes_at_stop": after_stop, "memory_floor_bytes": MEMORY_FLOOR,
                    "stop_threshold_bytes": MEMORY_FLOOR + STOP_HEADROOM,
                    "partial_file": str(part) if part.exists() else None,
                    "partial_bytes": part.stat().st_size if part.exists() else 0,
                    "reason": ("quantization stopped at the early RAM guard; post-stop available RAM was "
                               f"{after_stop / 1024**3:.2f} GiB" +
                               (" (below the 2.25 GiB safety floor)" if after_stop < MEMORY_FLOOR else ""))}
        try:
            line = lines.get_nowait()
            tail.append(line.rstrip())
            tail = tail[-30:]
        except queue.Empty:
            pass
        time.sleep(POLL_SECONDS)
    reader.join(timeout=2)
    output_text = "\n".join(tail)
    if proc.returncode or not part.is_file():
        return {"state": "build_failed", "return_code": proc.returncode, "output_tail": output_text,
                "partial_file": str(part), "partial_bytes": part.stat().st_size if part.exists() else 0,
                "minimum_available_ram_bytes": minimum}
    os.replace(part, output)
    return {"state": "built", "output": str(output), "bytes": output.stat().st_size,
            "minimum_available_ram_bytes": minimum, "output_tail": output_text}


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "q2"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = AUDIT / f"kl_level_{level}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    report_path = outdir / "measurement.json"
    base = outdir / "llama32-3b.ssf"
    candidate = outdir / f"llama32-3b-{level}.ssf"
    report = {"model": "Llama 3.2 3B", "level": level, "source": str(SOURCE),
              "source_kind": "local full-precision checkpoint", "status": "starting",
              "memory_floor_gib": 2.25, "prompt_count": len(PROMPTS), "prompt_results": [],
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "output_dir": str(outdir)}
    persist(report_path, report)
    if not SOURCE.is_dir():
        raise SystemExit("the already-local full-precision checkpoint is unavailable")
    if not candidate.exists():
        report["status"] = "building"
        persist(report_path, report)
        build = build_with_memory_floor(SOURCE, level, candidate)
        report["build"] = build
        if build["state"] != "built":
            report["status"] = build["state"]
            report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            persist(report_path, report)
            print(json.dumps(report, indent=2), flush=True)
            return 2
    report["candidate"] = str(candidate)
    report["candidate_bytes"] = candidate.stat().st_size
    report["status"] = "measuring"
    persist(report_path, report)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        started = time.perf_counter()
        value, seconds = kl_measure(candidate, SOURCE, tokens)
        row = {"prompt": prompt, "tokens": tokens, "positions": len(tokens), "kl_nats": value,
               "comparison_seconds": round(seconds, 2), "complete": True}
        report["prompt_results"].append(row)
        report["positions_measured"] = sum(r["positions"] for r in report["prompt_results"])
        report["measured_kl_nats"] = sum(r["kl_nats"] * r["positions"] for r in report["prompt_results"]) / report["positions_measured"]
        report["last_measurement_wall_seconds"] = round(time.perf_counter() - started, 2)
        persist(report_path, report)
        print(json.dumps({"progress": len(report["prompt_results"]), "of": len(PROMPTS),
                          "positions": row["positions"], "kl_nats": value}), flush=True)
    report["speed"] = speed_measure(candidate, gen=32, threads=2, tokens=[1, 11, 358, 1079])
    report["metric"] = "mean categorical KL(full_precision || quantized), nats, averaged across every input prefix for three fixed prompts"
    report["status"] = "measured"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    persist(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
