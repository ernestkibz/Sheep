#!/usr/bin/env python3
"""GPU prompt-reading race against the reference, many rounds, with a confidence
interval -- so a throttling laptop cannot decide the result by luck.

    python tooling/sml_gpu_race_ci.py MODEL.ssf REFERENCE.gguf [--p 128,512] [--rounds 10]

Each round runs Sheep and the reference back to back, alternating which goes
first (ABBA), one prompt size per run, and records tok/s and the GPU temperature.
The ratio is Sheep / reference per round; the verdict is the median ratio with a
bootstrap 95% interval. A size counts as won only when the whole interval sits
above 1.0 -- "won 3 of 5 rounds" is not a result, it is a coin.

Sheep's run also re-checks correctness every round (its next token after 40
tokens must agree with the CPU engine); a round that disagrees is reported and
excluded, not quietly averaged in.
"""
import argparse
import os
import random
import re
import statistics as st
import subprocess
from sml_benchmark_store import default_path, file_facts, record

HERE = os.path.dirname(os.path.abspath(__file__))
SML = os.path.join(HERE, "..", "sml")
GPU = os.path.join(SML, "sml-gpu-fwd" + (".exe" if os.name == "nt" else ""))
BENCH = os.environ.get("REF_BENCH", r"C:\Users\earne\llama-cuda\llama-bench.exe")
# Runs per side per round; the round's figure is their mean (the reference's
# bench averages its own -r repetitions; Sheep gets the same count of runs).
REPS = 3


def mean_ok(fn, *args):
    got = [fn(*args) for _ in range(REPS)]
    xs = [x for x, _ in got]
    ok = all(k for _, k in got)
    if any(x is None for x in xs):
        return None, ok
    return sum(xs) / len(xs), ok


def temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "?"


def gpu_facts():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10)
        if r.returncode != 0:
            return {"available": False, "error": r.stderr.strip() or f"exit code {r.returncode}"}
        return {"available": True,
                "devices": [line.strip() for line in r.stdout.splitlines() if line.strip()]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def calm(limit, max_wait):
    """Mark 4:39 -- "Peace, be still": rest until the GPU is at or below `limit` C,
    so every run starts from the same calm rather than from the last run's heat.
    Returns the temperature it started at (or waited out to)."""
    if not limit:
        return temp()
    import time
    t0 = time.time()
    while time.time() - t0 < max_wait:
        t = temp()
        if t.isdigit() and int(t) <= limit:
            return t
        time.sleep(5)
    return temp() + "!"   # waited the limit out: marked, not hidden


def sheep(ssf, p):
    env = dict(os.environ, SML_CTX=str(max(256, p + 64)), SML_PP=str(p))
    r = subprocess.run([GPU, os.path.abspath(ssf), "791,6864,315,9822,374", "0"], env=env,
                       cwd=SML, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    ok = "next token agrees" in r.stdout
    m = re.search(r"GPU prompt +%d tokens: .*= ([0-9.]+|inf) tok/s" % p, r.stdout)
    return (float(m.group(1)) if m else None), ok


def sheep_decode(ssf, extra):
    """Greedy decode of 64 tokens, graph replay when the path allows it; the run
    also checks the GPU's first logits against the CPU engine (top-1 must agree)."""
    env = dict(os.environ, SML_CTX="256", SPEED_ONLY="1", **extra)
    r = subprocess.run([GPU, os.path.abspath(ssf), "791,6864,315,9822,374", "0"], env=env,
                       cwd=SML, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    ok = re.search(r"top-1 agree (\d+)/(\d+)", r.stdout)
    ok = bool(ok) and ok.group(1) == ok.group(2) and "DIFFER" not in r.stdout
    m = re.search(r"GPU greedy decode: 64 tokens in [0-9.]+ s = ([0-9.]+|inf) tok/s", r.stdout)
    return (float(m.group(1)) if m else None), ok


def reference_decode(gguf, n):
    r = subprocess.run([BENCH, "-m", gguf, "-p", "0", "-n", str(n), "-ngl", "99", "-r", str(REPS)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    for line in r.stdout.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 8 and cells[-3] == "tg%d" % n:
            m = re.match(r"\s*([0-9.]+)", cells[-2])
            return float(m.group(1)) if m else None
    return None


def reference(gguf, p):
    r = subprocess.run([BENCH, "-m", gguf, "-p", str(p), "-n", "0", "-ngl", "99", "-r", str(REPS)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    for line in r.stdout.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 8 and cells[-3] == "pp%d" % p:
            # the cell is "mean ± sd"; the console codepage mangles the sign
            m = re.match(r"\s*([0-9.]+)", cells[-2])
            return float(m.group(1)) if m else None
    return None


def boot_ci(xs, n=4000, seed=7):
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choice(xs) for _ in xs) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ssf")
    ap.add_argument("gguf")
    ap.add_argument("--p", default="128,512")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--cool", type=int, default=0,
                    help="rest before every run until the GPU is at or below this many C")
    ap.add_argument("--max-wait", type=int, default=240, help="seconds to wait for --cool")
    ap.add_argument("--reps", type=int, default=3, help="runs per side per round (mean)")
    ap.add_argument("--decode", action="store_true",
                    help="race greedy decode (64 tokens each) instead of prompt reading")
    ap.add_argument("--env", default="",
                    help="extra settings for Sheep's run, e.g. 'SML_GRID=48 SML_BLOCK=128'")
    ap.add_argument("--output", default=str(default_path()),
                    help="append JSONL run records here (default: ~/.mlmi/benchmark-runs.jsonl)")
    ap.add_argument("--notes", default="", help="optional hardware, thermal, or workload notes")
    a = ap.parse_args()
    global REPS
    REPS = a.reps
    extra = dict(kv.split("=", 1) for kv in a.env.split()) if a.env else {}
    sizes = [64] if a.decode else [int(x) for x in a.p.split(",")]
    for p in sizes:
        ratios = []
        rounds = []
        what = f"decode tg{p}" if a.decode else f"pp{p}"
        print(f"== {os.path.basename(a.ssf)} vs {os.path.basename(a.gguf)}, {what}, {a.rounds} rounds"
              + (f", Sheep with {a.env}" if a.env else ""))
        for r in range(a.rounds):
            order = ("sheep", "ref") if r % 2 == 0 else ("ref", "sheep")
            got, start = {}, {}
            for who in order:
                start[who] = calm(a.cool, a.max_wait)
                if a.decode:
                    got[who] = mean_ok(sheep_decode, a.ssf, extra) if who == "sheep" else (reference_decode(a.gguf, p), True)
                else:
                    got[who] = mean_ok(sheep, a.ssf, p) if who == "sheep" else (reference(a.gguf, p), True)
            (s, ok), (f, _) = got["sheep"], got["ref"]
            if s is None or f is None or not ok:
                print(f"  round {r + 1}: EXCLUDED (sheep {s}, ref {f}, correct {ok})")
                rounds.append({"number": r + 1, "first": order[0], "sheep_tok_s": s,
                               "reference_tok_s": f, "ratio": None, "sheep_correct": bool(ok),
                               "sheep_start_c": start["sheep"], "reference_start_c": start["ref"],
                               "valid": False})
                continue
            ratios.append(s / f)
            rounds.append({"number": r + 1, "first": order[0], "sheep_tok_s": s,
                           "reference_tok_s": f, "ratio": s / f, "sheep_correct": True,
                           "sheep_start_c": start["sheep"], "reference_start_c": start["ref"],
                           "valid": True})
            print(f"  round {r + 1} ({order[0]} first): sheep {s:8.1f}  ref {f:8.1f}  ratio {s / f:5.2f}  "
                  f"[started {start['sheep']} / {start['ref']} C]")
        if not ratios:
            print("  no valid rounds")
            lo = hi = med = None
            verdict = "NO VALID ROUNDS"
        else:
            lo, hi = boot_ci(ratios)
            med = st.median(ratios)
            verdict = "WON" if lo > 1.0 else ("LOST" if hi < 1.0 else "NOT DECIDED")
            print(f"  {what}: median ratio {med:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
                  f"{sum(x > 1 for x in ratios)}/{len(ratios)} rounds above 1 -> {verdict}")
        saved = record(a.output, {
            "comparison": "sheep-vs-llama.cpp",
            "device": "gpu",
            "device_metadata": gpu_facts(),
            "workload": ("decode-" if a.decode else "prompt-") + f"tokens-{p}",
            "status": verdict,
            "models": {"sheep_ssf": file_facts(a.ssf), "reference_gguf": file_facts(a.gguf)},
            "runtimes": {"sheep_gpu": file_facts(GPU), "reference_bench": file_facts(BENCH)},
            "configuration": {"rounds_requested": a.rounds, "repetitions_per_round": REPS,
                              "alternating_order": "ABBA", "cool_target_c": a.cool or None,
                              "cool_max_wait_seconds": a.max_wait, "sheep_environment": extra},
            "correctness": {"checked": True,
                            "method": "Sheep next-token agreement for prompts; GPU/CPU top-1 agreement for decode."},
            "measurements": rounds,
            "summary": {"valid_rounds": len(ratios), "median_ratio": med,
                        "bootstrap_95_ci": [lo, hi], "bootstrap_seed": 7,
                        "bootstrap_samples": 4000},
            "notes": a.notes,
        })
        print(f"  recorded: {saved}")


if __name__ == "__main__":
    main()
