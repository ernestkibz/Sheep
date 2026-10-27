#!/usr/bin/env python3
"""The engine gate: record a baseline, then prove a change kept the engine's answers.

    python tooling/sml_engine_gate.py record  OUT_DIR   # before a change
    python tooling/sml_engine_gate.py compare OUT_DIR   # after it

For every combination of the kernel switches (SML_FLAT6, SML_PROMPT_SB,
SML_REPACK) the CPU engine scores a fixed prompt with --verify-batch (batched
reading must match stepping) and writes its logits; `compare` requires those
logits to be BYTE-IDENTICAL to the recorded ones. The GPU program grades its
forward pass against the CPU (worst correlation, next-token agreement), and
`compare` requires the same verdicts. Model: the 3B, which holds both flat
4-bit and flat 6-bit tensors, so the 6-bit path is exercised (a past change
silently dropped its low plane on the unfused path).
"""
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SML = os.path.join(HERE, "..", "sml")
MODEL = os.environ.get("SML_GATE_MODEL", r"C:\Users\earne\models\Llama-3.2-3B-Instruct.ssf")
TOKENS = "128000,791,6864,315,9822,374,12366,13,578,6864,315,9635,374"
SWITCHES = ("SML_FLAT6", "SML_PROMPT_SB", "SML_REPACK")


def run(cmd, env=None, timeout=900):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(cmd, cwd=SML, capture_output=True, text=True, timeout=timeout, env=e)
    return r.returncode, r.stdout + r.stderr


def cpu_cases(out):
    res = {}
    for bits in itertools.product("01", repeat=len(SWITCHES)):
        env = dict(zip(SWITCHES, bits))
        tag = "".join(f"{k[4:].lower()}{v}" for k, v in env.items())
        lg = os.path.join(out, f"cpu_{tag}.bin")
        cli = [os.path.join(SML, "sml-cli.exe"), MODEL, "--tokens", TOKENS, "--threads", "6", "--ctx", "256"]
        rc, o = run(cli + ["--gen", "4", "--verify-batch"], env)
        ok = rc == 0 and "PASS - batched scoring matches stepping" in o
        rc2, o2 = run(cli + ["--gen", "4"], {**env, "SML_CHECK_PREFILL": "1"})
        pf = rc2 == 0 and "\nPASS" in "\n" + o2
        if os.path.exists(lg):
            os.remove(lg)
        run(cli + ["--gen", "8", "--logits", lg], env)          # the stepped path's logits, for byte comparison
        digest = hashlib.sha256(open(lg, "rb").read()).hexdigest() if os.path.exists(lg) else None
        res[tag] = {"verify_batch": ok, "check_prefill": pf, "logits_sha256": digest}
        print(f"  cpu {tag}: verify-batch {'PASS' if ok else 'FAIL'}  prefill {'PASS' if pf else 'FAIL'}"
              f"  logits {digest[:12] if digest else '-'}", flush=True)
    return res


def gpu_case():
    rc, o = run([os.path.join(SML, "sml-gpu-fwd.exe"), MODEL, TOKENS, "32"], timeout=900)
    w = re.search(r"worst ([0-9.]+)", o)
    a = re.search(r"top-1 agree (\d+)/(\d+)", o)
    res = {"rc": rc, "worst_corr": float(w.group(1)) if w else None, "agree": f"{a.group(1)}/{a.group(2)}" if a else None,
           "graph_identical": "tokens identical to the per-token path" in o, "tail": o.strip().splitlines()[-4:]}
    print(f"  gpu: rc {rc} worst corr {res['worst_corr']} agree {res['agree']} graph identical {res['graph_identical']}", flush=True)
    return res


def main():
    mode, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    print(f"engine gate {mode}: {MODEL}", flush=True)
    now = {"cpu": cpu_cases(out if mode == "record" else os.path.join(out, "_now")) if mode == "record" else None}
    if mode == "compare":
        os.makedirs(os.path.join(out, "_now"), exist_ok=True)
        now["cpu"] = cpu_cases(os.path.join(out, "_now"))
    now["gpu"] = gpu_case()
    if mode == "record":
        json.dump(now, open(os.path.join(out, "baseline.json"), "w"), indent=1)
        print("baseline recorded")
        return
    base = json.load(open(os.path.join(out, "baseline.json")))
    bad = []
    for tag, b in base["cpu"].items():
        n = now["cpu"].get(tag, {})
        if not n.get("verify_batch"):
            bad.append(f"cpu {tag}: verify-batch failed")
        if b.get("check_prefill") and not n.get("check_prefill"):
            bad.append(f"cpu {tag}: batched prompt reading disagrees with stepping")
        if b.get("check_prefill") and not n.get("check_prefill"):
            bad.append(f"cpu {tag}: batched prompt reading disagrees with stepping")
        if n.get("logits_sha256") != b["logits_sha256"]:
            bad.append(f"cpu {tag}: logits changed")
    g0, g1 = base["gpu"], now["gpu"]
    if g1["rc"] != 0 or (g0["worst_corr"] and (g1["worst_corr"] or 0) < min(0.999, g0["worst_corr"] - 1e-4)):
        bad.append(f"gpu: worst corr {g1['worst_corr']} (was {g0['worst_corr']})")
    if g0["agree"] and g1["agree"] != g0["agree"]:
        bad.append(f"gpu: agreement {g1['agree']} (was {g0['agree']})")
    if g0.get("graph_identical") and not g1.get("graph_identical"):
        bad.append("gpu: graph replay no longer matches the per-token path")
    print("GATE PASS" if not bad else "GATE FAIL\n  " + "\n  ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
