#!/usr/bin/env python3
"""Discover this machine's best runtime settings and persist them.

Sheep's defaults are not optimal on any particular box, and the optimum is
not guessable from theory - on a Ryzen 5 5600H, generation runs ~76% faster on
4 threads than on 12, because generation is memory-bandwidth bound and extra
threads only add contention. Prompt processing is compute bound and wants every
core. No amount of reasoning finds that; measurement does.

So this sweeps the real knobs with repeated timed runs, picks winners only when
they clear the measured noise, and writes a profile keyed by a hardware+model
fingerprint to ~/.sheep/tuning.json. sheep-server loads it at startup, so a
user never has to know that -t and -tb should differ.

Usage:
    python tooling/autotune.py --bench <path/to/sml-cli> --model <model.ssf>
    python tooling/autotune.py ... --quick      # fewer candidates, fewer reps
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".sheep")
PROFILE_PATH = os.path.join(PROFILE_DIR, "tuning.json")
SCHEMA_VERSION = 1


# ----------------------------------------------------------------- fingerprint

def _proc_cpuinfo():
    """Parsed /proc/cpuinfo, or None where it does not exist."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=60)
            name = out.stdout.strip().splitlines()
            if name:
                return name[0].strip()
        except Exception:
            pass

    # Linux: the profile is keyed on this, so a wrong value means a machine
    # silently reuses another machine's tuning
    info = _proc_cpuinfo()
    if info:
        for line in info.splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()

    return platform.processor() or platform.machine()


def physical_cores():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).NumberOfCores"],
                capture_output=True, text=True, timeout=60)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            pass

    # Linux: count distinct physical ids, so hyperthreads are not counted twice.
    # This matters more than it looks - the whole finding was that generation
    # wants *fewer* threads than logical CPUs.
    info = _proc_cpuinfo()
    if info:
        cores = set()
        phys = core = None
        for line in info.splitlines():
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
            elif not line.strip():
                if phys is not None and core is not None:
                    cores.add((phys, core))
                phys = core = None
        if phys is not None and core is not None:
            cores.add((phys, core))
        if cores:
            return len(cores)
        # some ARM kernels omit those fields entirely
        for line in info.splitlines():
            if line.lower().startswith("cpu cores"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

    return os.cpu_count() or 1


def model_fingerprint(path):
    """Size plus a hash of the head - enough to tell models apart without
    reading gigabytes."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
    h.update(str(size).encode())
    return {
        "file": os.path.basename(path),
        "size": size,
        "digest": h.hexdigest()[:16],
    }


def hardware_fingerprint():
    return {
        "cpu": cpu_name(),
        "cores_physical": physical_cores(),
        "cores_logical": os.cpu_count() or 1,
        "system": platform.system(),
        "machine": platform.machine(),
    }


def profile_key(hw, model):
    raw = "%s|%s|%s|%s" % (hw["cpu"], hw["cores_logical"],
                           model["digest"], model["size"])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- measurement

class Bench:
    def __init__(self, bench_path, model_path, reps, verbose=True):
        self.bench = bench_path
        self.model = model_path
        self.reps = reps
        self.verbose = verbose
        self.runs = 0

    def run(self, *, gen, threads, batch=None, ubatch=None,
            type_k=None, type_v=None, flash=None, depth=0):
        """One configuration. `gen` selects generation vs prompt throughput."""
        cmd = [self.bench, "-m", self.model, "-t", str(threads),
               "-r", str(self.reps), "-o", "json"]
        cmd += ["-p", "0", "-n", "64"] if gen else ["-p", "512", "-n", "0"]
        if depth:
            cmd += ["-d", str(depth)]
        if batch is not None:
            cmd += ["-b", str(batch)]
        if ubatch is not None:
            cmd += ["-ub", str(ubatch)]
        if type_k is not None:
            cmd += ["-ctk", type_k]
        if type_v is not None:
            cmd += ["-ctv", type_v]
        if flash is not None:
            cmd += ["-fa", flash]

        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=1800)
            rows = json.loads(out.stdout)
        except Exception as exc:
            if self.verbose:
                print("      run failed: %s" % exc)
            return None

        self.runs += 1
        row = rows[0]
        return {"ts": row["avg_ts"], "sd": row.get("stddev_ts", 0.0)}


def pick(results, label, verbose=True):
    """Best candidate, but only trusted if it clears the noise of the runner-up."""
    scored = [(v["ts"], k, v["sd"]) for k, v in results.items() if v]
    if not scored:
        return None, None
    scored.sort(reverse=True)
    best_ts, best_key, best_sd = scored[0]

    decisive = True
    if len(scored) > 1:
        second_ts, _, second_sd = scored[1]
        margin = best_ts - second_ts
        noise = max(best_sd, second_sd, 0.05 * best_ts)
        decisive = margin > noise

    if verbose:
        for ts, key, sd in scored:
            mark = " <-" if key == best_key else "   "
            print("      %-22s %7.2f t/s  (sd %.2f)%s" % (key, ts, sd, mark))
        if not decisive:
            print("      (margin within noise - keeping best but low confidence)")
    return best_key, {"ts": best_ts, "decisive": decisive}


# ------------------------------------------------------------------- sweeps

def candidate_threads(logical, physical, quick):
    cands = {1, 2, physical // 2, physical, physical + 2, logical}
    cands = sorted(c for c in cands if 1 <= c <= logical)
    if quick:
        cands = sorted({max(1, physical // 2), physical, logical})
    return cands


def autotune(bench, hw, quick, depths):
    logical, physical = hw["cores_logical"], hw["cores_physical"]
    threads = candidate_threads(logical, physical, quick)
    tuning = {}

    # Generation threads, measured at several KV depths. Tuning at depth 0 alone
    # picks a setting that runs ~15% SLOWER on long contexts: with a large KV
    # cache the attention work per token grows, so threads that were pure
    # contention at depth 0 start to pay for themselves. Score each candidate by
    # its WORST showing relative to the best at each depth, so the profile is
    # robust across workloads rather than optimal at one operating point.
    print("\n[1/4] generation threads  (scored across KV depths, worst case wins)")
    per_depth = {}
    for d in depths:
        row = {}
        for t in threads:
            r = bench.run(gen=True, threads=t, depth=d)
            if r:
                row[t] = r
        per_depth[d] = row
        if row:
            print("      depth %-5d %s" % (d, "  ".join(
                "t=%d:%.2f" % (t, r["ts"]) for t, r in sorted(row.items()))))

    scores = {}
    for t in threads:
        ratios = []
        for d in depths:
            row = per_depth.get(d) or {}
            if t not in row:
                continue
            best_at_depth = max(r["ts"] for r in row.values())
            if best_at_depth > 0:
                ratios.append(row[t]["ts"] / best_at_depth)
        if ratios:
            scores[t] = min(ratios)

    if scores:
        n_threads = max(scores, key=scores.get)
        print("      worst-case      %s" % "  ".join(
            "t=%d:%.2f" % (t, sc) for t, sc in sorted(scores.items())))
        print("      -> t=%d (never below %.0f%% of the best at any depth)"
              % (n_threads, 100 * scores[n_threads]))
    else:
        n_threads = physical
    tuning["n_threads"] = n_threads
    gen_baseline = (per_depth.get(depths[0]) or {}).get(n_threads, {}).get("ts", 0.0)

    print("\n[2/4] prompt threads  (compute bound - more usually wins)")
    pp_res = {}
    for t in threads:
        pp_res["tb=%d" % t] = bench.run(gen=False, threads=t)
    best, meta = pick(pp_res, "prompt")
    n_threads_batch = int(best.split("=")[1]) if best else logical
    tuning["n_threads_batch"] = n_threads_batch
    pp_baseline = meta["ts"] if meta else 0.0

    print("\n[3/4] KV cache type  (q8_0 halves KV memory - is it slower?)")
    kv_res = {}
    for k, v in (("f16", "f16"), ("q8_0", "q8_0")):
        kv_res["kv=%s" % k] = bench.run(gen=True, threads=n_threads,
                                        type_k=k, type_v=v)
    best, meta = pick(kv_res, "kv")
    kv_type = best.split("=")[1] if best else "f16"
    tuning["cache_type_k"] = kv_type
    tuning["cache_type_v"] = kv_type

    print("\n[4/4] batch size  (prompt throughput)")
    batch_res = {}
    for b, ub in ((2048, 512), (1024, 256), (512, 128)) if not quick else ((2048, 512), (1024, 256)):
        batch_res["b=%d/ub=%d" % (b, ub)] = bench.run(
            gen=False, threads=n_threads_batch, batch=b, ubatch=ub)
    best, meta = pick(batch_res, "batch")
    if best:
        # split first: a bare replace("b=") would also eat the b= inside ub=
        left, right = best.split("/")
        tuning["n_batch"] = int(left.split("=")[1])
        tuning["n_ubatch"] = int(right.split("=")[1])

    return tuning, {"gen_ts": gen_baseline, "prompt_ts": pp_baseline}


# --------------------------------------------------------------------- main

def load_profiles():
    if not os.path.exists(PROFILE_PATH):
        return {"schema": SCHEMA_VERSION, "profiles": {}}
    try:
        with open(PROFILE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema") != SCHEMA_VERSION:
            return {"schema": SCHEMA_VERSION, "profiles": {}}
        return data
    except (OSError, ValueError):
        return {"schema": SCHEMA_VERSION, "profiles": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--depths", default="0,4096",
                    help="KV depths to score generation at (comma separated)")
    ap.add_argument("--print-only", action="store_true",
                    help="show the stored profile for this machine+model")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    hw = hardware_fingerprint()
    model = model_fingerprint(args.model)
    key = profile_key(hw, model)
    store = load_profiles()

    if args.print_only:
        print(json.dumps(store["profiles"].get(key, {}), indent=2))
        return 0

    print("hardware : %s (%d physical / %d logical)"
          % (hw["cpu"], hw["cores_physical"], hw["cores_logical"]))
    print("model    : %s (%.2f GB)" % (model["file"], model["size"] / 2**30))
    print("profile  : %s" % key)

    started = time.time()
    bench = Bench(args.bench, args.model, args.reps)
    tuning, baseline = autotune(bench, hw, args.quick, depths)
    elapsed = time.time() - started

    store["profiles"][key] = {
        "hardware": hw,
        "model": model,
        "tuning": tuning,
        "measured": baseline,
        "depths": depths,
        "bench_runs": bench.runs,
        "seconds": round(elapsed, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    os.makedirs(PROFILE_DIR, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)

    print("\n" + "=" * 58)
    print("tuned in %.0f s over %d benchmark runs" % (elapsed, bench.runs))
    for k, v in tuning.items():
        print("  %-18s %s" % (k, v))
    print("\nwritten to %s" % PROFILE_PATH)
    print("equivalent flags:")
    print("  -t %d -tb %d --cache-type-k %s --cache-type-v %s -b %d -ub %d"
          % (tuning.get("n_threads", 0), tuning.get("n_threads_batch", 0),
             tuning.get("cache_type_k", "f16"), tuning.get("cache_type_v", "f16"),
             tuning.get("n_batch", 2048), tuning.get("n_ubatch", 512)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
