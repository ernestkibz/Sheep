#!/usr/bin/env python3
"""CPU prompt-reading race against the reference, many rounds, with a confidence
interval.

    python tooling/sml_cpu_race_ci.py MODEL.ssf REFERENCE.gguf [--p 28,128] [--rounds 10] [--threads 6]

Both engines read warm (Sheep with --warmup, the reference's bench warms up by
itself), on the SAME thread count, one prompt size per run, back to back in ABBA
order. The verdict is the median of per-round ratios (Sheep / reference) with a
bootstrap 95% interval; a size is WON only when the whole interval is above 1.
"""
import argparse
import os
import re
import subprocess

from sml_gpu_race_ci import boot_ci
from sml_benchmark_store import default_path, file_facts, record
# the reference's CPU-only build, run from its own folder: llama.cpp also loads
# backend DLLs from the current directory, and a CUDA one there would be used
BENCH = os.environ.get("REF_BENCH_CPU", r"C:\Users\earne\llama-cpu\llama-bench.exe")
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
# Runs per side per round; the round's figure is their mean. The reference's
# bench averages its own -r repetitions, so Sheep gets the same count of runs.
REPS = 3
CLI = os.path.join(HERE, "..", "sml", "sml-cli" + (".exe" if os.name == "nt" else ""))


def ids(n):
    return ",".join(str(1000 + (i * 37) % 20000) for i in range(n))


def mean_of(fn, *args):
    xs = [fn(*args) for _ in range(REPS)]
    return None if any(x is None for x in xs) else sum(xs) / len(xs)


def sheep(ssf, p, threads):
    r = subprocess.run([CLI, ssf, "--tokens", ids(p), "--gen", "0", "--threads", str(threads),
                        "--warmup"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    m = re.search(r"prompt %d tokens in [0-9.]+s \(([0-9.]+|inf) tok/s\)" % p, r.stdout)
    return float(m.group(1)) if m else None


def sheep_decode(ssf, g, threads):
    r = subprocess.run([CLI, ssf, "--tokens", ids(8), "--gen", str(g), "--threads", str(threads),
                        "--warmup"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    m = re.search(r"generate %d tokens in [0-9.]+s\s+([0-9.]+|inf) tok/s" % g, r.stdout)
    return float(m.group(1)) if m else None


def reference_decode(gguf, g, threads):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([BENCH, "-m", gguf, "-p", "0", "-n", str(g), "-t", str(threads), "-ngl", "0",
                        "-dev", "none", "-r", str(REPS)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, env=env,
                       cwd=os.path.dirname(BENCH))
    for line in r.stdout.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 8 and cells[-3] == "tg%d" % g:
            m = re.match(r"\s*([0-9.]+)", cells[-2])
            return float(m.group(1)) if m else None
    return None


def reference(gguf, p, threads):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([BENCH, "-m", gguf, "-p", str(p), "-n", "0", "-t", str(threads), "-ngl", "0",
                        "-dev", "none", "-r", str(REPS)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, env=env,
                       cwd=os.path.dirname(BENCH))
    for line in r.stdout.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 8 and cells[-3] == "pp%d" % p:
            m = re.match(r"\s*([0-9.]+)", cells[-2])
            return float(m.group(1)) if m else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ssf")
    ap.add_argument("gguf")
    ap.add_argument("--p", default="28,128")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3, help="runs per side per round (mean)")
    ap.add_argument("--decode", type=int, default=0,
                    help="race decode of this many tokens instead of prompt reading")
    ap.add_argument("--output", default=str(default_path()),
                    help="append JSONL run records here (default: ~/.mlmi/benchmark-runs.jsonl)")
    ap.add_argument("--notes", default="", help="optional hardware, thermal, or workload notes")
    a = ap.parse_args()
    global REPS
    REPS = a.reps
    sizes = [a.decode] if a.decode else [int(x) for x in a.p.split(",")]
    for p in sizes:
        ratios = []
        rounds = []
        workload = f"decode-tg{p}" if a.decode else f"prompt-pp{p}"
        print(f"== {os.path.basename(a.ssf)} vs {os.path.basename(a.gguf)}, CPU pp{p}, "
              f"{a.threads} threads each, {a.rounds} rounds")
        for r in range(a.rounds):
            order = ("sheep", "ref") if r % 2 == 0 else ("ref", "sheep")
            got = {}
            for who in order:
                if a.decode:
                    got[who] = mean_of(sheep_decode, a.ssf, p, a.threads) if who == "sheep" else reference_decode(a.gguf, p, a.threads)
                else:
                    got[who] = mean_of(sheep, a.ssf, p, a.threads) if who == "sheep" else reference(a.gguf, p, a.threads)
            s, f = got["sheep"], got["ref"]
            if s is None or f is None:
                print(f"  round {r + 1}: EXCLUDED (sheep {s}, ref {f})")
                rounds.append({"number": r + 1, "first": order[0], "sheep_tok_s": s,
                               "reference_tok_s": f, "ratio": None, "valid": False})
                continue
            ratios.append(s / f)
            rounds.append({"number": r + 1, "first": order[0], "sheep_tok_s": s,
                           "reference_tok_s": f, "ratio": s / f, "valid": True})
            print(f"  round {r + 1} ({order[0]} first): sheep {s:7.1f}  ref {f:7.1f}  ratio {s / f:5.2f}")
        if ratios:
            lo, hi = boot_ci(ratios)
            median = st.median(ratios)
            verdict = "WON" if lo > 1.0 else ("LOST" if hi < 1.0 else "NOT DECIDED")
            print(f"  {workload}: median ratio {median:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
                  f"{sum(x > 1 for x in ratios)}/{len(ratios)} rounds above 1 -> {verdict}")
        else:
            lo = hi = median = None
            verdict = "NO VALID ROUNDS"
        saved = record(a.output, {
            "comparison": "sheep-vs-llama.cpp",
            "device": "cpu",
            "workload": workload,
            "status": verdict,
            "models": {"sheep_ssf": file_facts(a.ssf), "reference_gguf": file_facts(a.gguf)},
            "runtimes": {"sheep_cli": file_facts(CLI), "reference_bench": file_facts(BENCH)},
            "configuration": {"threads": a.threads, "rounds_requested": a.rounds,
                              "repetitions_per_round": REPS, "alternating_order": "ABBA"},
            "correctness": {"checked": False,
                            "reason": "This throughput harness does not compare generated outputs."},
            "measurements": rounds,
            "summary": {"valid_rounds": len(ratios), "median_ratio": median,
                        "bootstrap_95_ci": [lo, hi], "bootstrap_seed": 7,
                        "bootstrap_samples": 4000},
            "notes": a.notes,
        })
        print(f"  recorded: {saved}")


if __name__ == "__main__":
    main()
