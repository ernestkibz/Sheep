"""Autotuner for the engine: search, keep only what is correct and faster.

The plain-search form of recursive improvement. Every knob the engine already
reads from the environment is sampled, raced on the real engine, the slower
half dropped, the rest raced again, and the survivor settled against the
shipped default in an interleaved paired race with a bootstrap interval. The
laptop moves 5-15% run to run with heat, so a single fast run proves nothing;
only the paired interval decides.

Correctness first: a candidate must produce the same answer as the default --
the same logits for prompt reading, the same generated text for decode -- or it
is discarded before it is timed a second time. On the GPU the gate is the
binary own grading: every position teacher-forced against the CPU engine,
top-1 agreement on all of them. A tuner will happily hand you a faster wrong
answer.

    python tooling/sml_tune.py MODEL.ssf --mode prefill [--budget 100] [--seed 1]
    python tooling/sml_tune.py MODEL.ssf --mode decode  [--budget 60]
    python tooling/sml_tune.py MODEL.ssf --mode gpu     [--ids 1,2,3,4,5]

Measured 2026-09-18 on the CPU: every knob is at its optimum already (paired
ratios 0.975-1.037, all inside the noise but one +1.7%); the prompt-reading gap
to the reference engine is in the kernel arithmetic, not in a setting.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""
CLI = os.path.join(HERE, "..", "sml", "sml-cli" + EXE)
GPU = os.path.join(HERE, "..", "sml", "sml-gpu-fwd" + EXE)

LONG_PROMPT = (
    "You are Sheep, a live assistant that sees through the camera and hears "
    "through the microphone. Keep answers short, friendly and correct. If you are "
    "not sure, say so. Earlier in this conversation the user showed you a kitchen "
    "counter with a green apple, a knife, a cutting board and a glass of water, "
    "and asked how to cut the apple safely. You explained how to hold the knife "
    "and keep fingers clear of the blade. Now the user says: what am I holding in "
    "my hand right now, and is it safe to eat? The user then asks a follow-up "
    "question about whether the apple should be washed first, and whether the "
    "knife needs to be sharp or blunt to be safe for a child to use under watch."
)

# Token ids the GPU race uses per model file, so a tuned result is comparable.
GPU_IDS = {
    "qwen3b-v7.ssf": "785,6722,315,9625,374",
    "llama32-1b-flat.ssf": "791,6864,315,9822,374",
    "pythia-flat.ssf": "510,5347,273,6181,310",
}

# Knobs the engine reads today. None means "unset" (the engine default).
SPACE = {
    "prefill": {
        "SML_PREFILL_NB":        ["4", "6", "8", "12", "16", "24", "32"],
        "SML_CHUNK_ROWS":        ["0", "8", "16", "32", "64", "128", "256"],
        "SML_PF_DIST":           ["0", "2", "4", "8", "16"],
        "SML_LANES4":            [None, "1"],
        "threads":               ["4", "5", "6", "7", "8", "12"],
    },
    "decode": {
        "SML_DECODE_CHUNK_ROWS": ["0", "8", "16", "32", "64", "128", "256"],
        "SML_PF_DIST":           ["0", "2", "4", "8", "16"],
        "SML_LANES4":            [None, "1"],
        "threads":               ["4", "5", "6", "7", "8", "12"],
    },
    "gpu": {
        "SML_FUSE2":             ["1"],
        "SML_FAST":              ["1"],
        "SML_GRAPH":             ["0", "1"],
        "SML_GRID":              ["8", "16", "24", "32", "48", "64", "96", "128", "256"],
        "SML_BLOCK":             ["64", "128", "256", "512"],
        "SML_SHARED":            ["0"],
        "SML_CTXFLAGS":          ["0", "1", "2", "4"],
    },
}
DEFAULT = {
    "prefill": {"SML_PREFILL_NB": "8", "SML_CHUNK_ROWS": "32", "SML_PF_DIST": "0",
                "SML_LANES4": None, "threads": "6"},
    "decode":  {"SML_DECODE_CHUNK_ROWS": "32", "SML_PF_DIST": "0",
                "SML_LANES4": None, "threads": "6"},
    "gpu":     {"SML_FUSE2": "1", "SML_FAST": "1", "SML_GRAPH": "1", "SML_GRID": "48",
                "SML_BLOCK": "128", "SML_SHARED": "0", "SML_CTXFLAGS": "1"},
}
ALL_KNOBS = {k for s in SPACE.values() for k in s if k != "threads"}
RE_PREFILL = re.compile(r"prompt (\d+) tokens in [0-9.]+s \(([0-9.]+) tok/s\)")
RE_DECODE = re.compile(r"generate \d+ tokens in [0-9.]+s\s*\n\s*([0-9.]+) tok/s")
RE_GPU = re.compile(r"= ([0-9.]+) tok/s")
RE_GPU_GRADE = re.compile(r"mean logit corr ([0-9.]+), worst ([0-9.]+), top-1 agree (\d+)/(\d+)")
RE_TEXT = re.compile(r'^  "(.*)"$', re.S | re.M)


def key(cfg):
    return " ".join(f"{k}={'-' if v is None else v}" for k, v in sorted(cfg.items()))


def env_for(cfg, extra=None):
    env = dict(os.environ)
    for k in ALL_KNOBS | {"SML_CHECK_PREFILL", "SML_SEQ_PROMPT", "SPEED_ONLY", "PROFILE"}:
        env.pop(k, None)
    for k, v in cfg.items():
        if k != "threads" and v is not None:
            env[k] = v
    if extra:
        env.update(extra)
    return env


class Target:
    """How to run, time and grade one engine binary."""

    def __init__(self, a):
        self.a = a

    def run(self, cfg, logits=None, extra=None, grade=False, gen=None):
        a = self.a
        if a.mode == "gpu":
            args = [GPU, a.model, a.ids, "0"]
            env = env_for(cfg, None if grade else {"SPEED_ONLY": "1"})
        else:
            n = "1" if a.mode == "prefill" else str(gen or a.gen)
            args = [CLI, a.model, "--prompt", a.prompt, "--threads", cfg["threads"],
                    "--ctx", "1024", "--gen", n]
            if logits:
                args += ["--logits", logits]
            env = env_for(cfg, extra)
        # the GPU binary loads its kernels from sml-gpu-fwd.ptx by relative path,
        # so it has to run from its own directory
        cwd = os.path.dirname(os.path.abspath(GPU)) if a.mode == "gpu" else None
        p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=900,
                           cwd=cwd)
        return p.stdout

    def measure(self, cfg):
        out = self.run(cfg)
        if self.a.mode == "gpu":
            m = RE_GPU.search(out)
            return float(m.group(1)) if m else None
        m = (RE_PREFILL if self.a.mode == "prefill" else RE_DECODE).search(out)
        if not m:
            return None
        return float(m.group(2) if self.a.mode == "prefill" else m.group(1))


class Gate:
    """Same answer as the default (CPU) or as the CPU engine (GPU), or out."""

    def __init__(self, target, default):
        self.t = target
        self.mode = target.a.mode
        self.ref = None if self.mode == "gpu" else self.answer(default)
        if self.mode != "gpu" and self.ref is None:
            sys.exit("the default configuration produced no answer; nothing to gate against")
        if self.mode == "gpu":
            ok, why = self.passes(default)
            print(f"  default graded against the CPU engine: {why}")
            if not ok:
                sys.exit("the default GPU path does not match the CPU engine; fix that first")

    def answer(self, cfg):
        if self.mode == "prefill":
            fd, path = tempfile.mkstemp(suffix=".bin")
            os.close(fd)
            try:
                self.t.run(cfg, logits=path)
                v = np.fromfile(path, dtype=np.float32, offset=8).astype(np.float64)
            finally:
                os.remove(path)
            return v if v.size else None
        m = RE_TEXT.search(self.t.run(cfg, gen=48))
        return m.group(1) if m else None

    def passes(self, cfg):
        if self.mode == "gpu":
            out = self.t.run(cfg, grade=True)
            if "DIFFER FROM" in out:
                return False, "graph replay tokens differ from the per-token path"
            m = RE_GPU_GRADE.search(out)
            if not m:
                return False, "no grade (failed to run)"
            corr, worst, agree, total = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
            ok = agree == total and worst >= 0.99
            return ok, f"top-1 {agree}/{total}, corr {corr:.4f}, worst {worst:.4f}"
        got = self.answer(cfg)
        if got is None:
            return False, "no answer"
        if self.mode == "prefill":
            if got.size != self.ref.size:
                return False, "size"
            rel = float(np.linalg.norm(got - self.ref) / np.linalg.norm(self.ref))
            same = int(np.argmax(got)) == int(np.argmax(self.ref))
            # kernels that sum in a different order may differ in the last bits;
            # anything past rounding is a different answer
            return (same and rel < 1e-5), f"rel {rel:.1e}"
        return got == self.ref, "text identical" if got == self.ref else "text differs"


def bootstrap_ratio(a, b, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    point = float(np.median(a) / np.median(b))
    ra = rng.choice(a, (n, a.size)); rb = rng.choice(b, (n, b.size))
    r = np.median(ra, axis=1) / np.median(rb, axis=1)
    return point, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    ap = argparse.ArgumentParser(description="search the engine knobs, keep what is correct and faster")
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["prefill", "decode", "gpu"], default="prefill")
    ap.add_argument("--budget", type=int, default=100, help="random candidates besides the default")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gen", type=int, default=64, help="tokens timed in decode mode")
    ap.add_argument("--prompt", default=LONG_PROMPT)
    ap.add_argument("--ids", help="token ids for gpu mode (defaults to the race ids)")
    ap.add_argument("--pairs", type=int, default=10, help="paired races in the final settle")
    ap.add_argument("--out", help="write the result as JSON")
    cfg = ap.parse_args()
    if cfg.mode == "gpu" and not cfg.ids:
        cfg.ids = GPU_IDS.get(os.path.basename(cfg.model))
        if not cfg.ids:
            sys.exit("gpu mode needs --ids for this model")

    space, default = SPACE[cfg.mode], DEFAULT[cfg.mode]
    rng = random.Random(cfg.seed)
    t_start = time.time()
    target = Target(cfg)

    # the default first, always carried as the control
    cands = {key(default): dict(default)}
    tries = 0
    while len(cands) < cfg.budget + 1 and tries < cfg.budget * 20:
        c = {k: rng.choice(v) for k, v in space.items()}
        cands.setdefault(key(c), c)
        tries += 1
    print(f"{cfg.mode}: {len(cands)} candidates on {os.path.basename(cfg.model)}")

    # warm the page cache so the first candidate is not paying for the file
    target.measure(default)

    gate = Gate(target, default)
    scores = {k: [] for k in cands}
    checked = {}
    alive = list(cands)
    dkey = key(default)
    rnd = 0
    while len(alive) > 3:
        rnd += 1
        order = alive[:]
        rng.shuffle(order)
        for k in order:
            v = target.measure(cands[k])
            if v is not None:
                scores[k].append(v)
        ranked = sorted((k for k in alive if scores[k]),
                        key=lambda k: -float(np.median(scores[k])))
        keep = []
        for k in ranked:
            if len(keep) >= max(3, len(alive) // 2):
                break
            if k not in checked:
                checked[k] = gate.passes(cands[k])
            if checked[k][0]:
                keep.append(k)
            else:
                print(f"  rejected, not the same answer ({checked[k][1]}): {k}")
        if dkey not in keep:
            keep.append(dkey)                         # the control never leaves
        alive = keep
        best = ranked[0]
        print(f"  round {rnd}: {len(order)} raced, {len(alive)} kept; leader "
              f"{np.median(scores[best]):.1f} tok/s  [{best}]  "
              f"default {np.median(scores[dkey]):.1f}  ({time.time() - t_start:.0f} s)")

    finalists = sorted((k for k in alive if k != dkey),
                       key=lambda k: -float(np.median(scores[k])))
    if not finalists:
        print("no candidate survived the gate; the default stands")
        return
    win = finalists[0]
    print(f"\nsettle: {win}\n    vs default: {dkey}\n  {cfg.pairs} interleaved pairs")
    a, b = [], []
    for i in range(cfg.pairs):
        order = [(win, a), (dkey, b)] if i % 2 == 0 else [(dkey, b), (win, a)]
        for k, sink in order:
            v = target.measure(cands[k])
            if v is not None:
                sink.append(v)
    point, lo, hi = bootstrap_ratio(a, b)
    verdict = ("REAL WIN -- ship it" if lo > 1.0 else
               "REAL LOSS -- keep the default" if hi < 1.0 else
               "NO DIFFERENCE the noise can separate -- keep the default")
    print(f"  candidate median {np.median(a):.1f} tok/s, default median {np.median(b):.1f}")
    print(f"  ratio {point:.3f}x   95% CI [{lo:.3f}, {hi:.3f}]   {verdict}")
    print(f"  answer check: {checked.get(win, ('?', '?'))[1]}")
    print(f"  total {time.time() - t_start:.0f} s")
    if cfg.out:
        with open(cfg.out, "w") as f:
            json.dump({"mode": cfg.mode, "model": cfg.model, "winner": cands[win],
                       "default": default, "ratio": point, "ci": [lo, hi],
                       "verdict": verdict, "candidate": a, "control": b}, f, indent=1)


if __name__ == "__main__":
    main()
