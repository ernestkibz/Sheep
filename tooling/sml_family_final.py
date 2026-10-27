#!/usr/bin/env python3
"""The final family run: every family head against llama.cpp, one command.

    python tooling/sml_family_final.py --preflight     every family at the sweep's tiny
                                                       size through every step, 1 round
    python tooling/sml_family_final.py                 the final run: race size, 6 rounds

The Ten Commandments (plan, 2026-09-30) this enforces:
  I    one verdict, fixed here: bytes <=, KL CI < 0, CPU and GPU prompt and
       decode CIs > 1 (sml_family_race.finish)
  II   one recipe for every family (sml_family_race.sheep_build)
  III  no tick without the family's own race; "llama.cpp cannot run" is its own row
  IV   the Sabbath: binaries hashed at the start and checked before every race;
       a changed binary stops the run. One family at a time: build, race, clean.
  V    llama.cpp's own converter, quantizer, perplexity and bench tools
  IX   every result saved as measured, before it is printed
Resumable: a family with a saved result is skipped unless --redo names it.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sml_family_race as R  # noqa: E402
import sml_family_heads as FH  # noqa: E402

ROOT = os.path.join(HERE, "..")
SML = os.path.join(ROOT, "sml")
EXE = ".exe" if os.name == "nt" else ""


def frozen_files():
    fs = [os.path.join(SML, "sml-cli" + EXE), os.path.join(SML, "sml-gpu-fwd" + EXE),
          os.path.join(SML, "sml-gpu-fwd.ptx"), R.QUANTIZE, R.PERPLEXITY, R.CONVERT]
    from sml_cpu_race_ci import BENCH as CPU_BENCH
    from sml_gpu_race_ci import BENCH as GPU_BENCH
    return fs + [CPU_BENCH, GPU_BENCH]


def hashes():
    out = {}
    for f in frozen_files():
        if os.path.exists(f):
            with open(f, "rb") as fh:
                out[os.path.basename(f) + ("@" + os.path.basename(os.path.dirname(f)))] = \
                    hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def one_family(label, mt, over, work, size, rounds, reps, do_gpu):
    d = os.path.join(work, label)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    ssf = os.path.join(d, "sheep.ssf")
    f16, q4 = os.path.join(d, "ref-f16.gguf"), os.path.join(d, "ref-q4km.gguf")
    res = {"label": label, "model_type": mt, "size": size, "t": time.strftime("%Y-%m-%d %H:%M")}
    steps = res["steps"] = {}
    try:
        model = R.prepare_head(label, mt, over, d, size=size)
    except Exception as e:
        res["error"] = f"transformers: {type(e).__name__}: {e}"[:300]
        steps["head"] = False
        return res
    steps["head"] = True
    res["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    n_layer = int(getattr(model.config, "num_hidden_layers", 0) or getattr(model.config, "n_layer", 0) or 2)
    res["alloc"] = R.sheep_alloc(n_layer)
    # llama.cpp's file first: its size is Sheep's byte budget (commandment I:
    # bytes <= theirs), spent by the builder's one fixed order
    c = R.convert(d, f16)
    steps["their_convert"] = os.path.exists(f16)
    if steps["their_convert"]:
        q = R.run([R.QUANTIZE, f16, q4, "Q4_K_M"], timeout=1800)
        steps["their_quantize"] = os.path.exists(q4)
    t0 = time.time()
    # the preflight proves the steps run, not the quantizer: a thin grid there
    env = {"SML_GRID_CAND": "6", "SML_GRID_ITERS": "1"} if size == "tiny" else None
    b = R.sheep_build(d, ssf, res["alloc"], env=env,
                      budget=os.path.getsize(q4) if steps.get("their_quantize") else 0)
    steps["sheep_build"] = b.returncode == 0 and os.path.exists(ssf)
    if not steps["sheep_build"]:
        res["error"] = "sheep build: " + " | ".join((b.stdout + b.stderr).strip().splitlines()[-3:])[:300]
        return res
    res["sheep_build_s"] = round(time.time() - t0)
    if not steps["their_convert"]:
        # Sheep built and runs; the reference cannot: its own row (commandment III)
        res["error"] = "llama.cpp converter: " + " | ".join((c.stdout + c.stderr).strip().splitlines()[-3:])[:300]
        res["reference_cannot_load"] = True
        return res
    if not steps["their_quantize"]:
        res["error"] = "llama-quantize: " + (q.stdout + q.stderr).strip()[-300:]
        return res
    for big in (f16, os.path.join(d, "model.safetensors")):
        try:
            os.remove(big)
        except OSError:
            pass
    res = R.finish(res, model, d, ssf, q4, rounds, do_gpu, extra=["--reps", str(reps)])
    res["steps"] = steps
    steps["quality"] = "kld_sheep" in res.get("quality", {})
    for k in ("cpu_pp", "cpu_tg", "gpu_pp", "gpu_tg"):
        steps[k] = "ratio" in res.get(k, {}) or (k.startswith("gpu") and "gpu_refused" in res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--redo", default="", help="families to run again even if saved")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    size = "tiny" if a.preflight else "race"
    rounds, reps = (1, 1) if a.preflight else (a.rounds, a.reps)
    out = a.json or os.path.join(HERE, "family_preflight.json" if a.preflight else "family_final.json")
    work = a.work or tempfile.mkdtemp(prefix="sml_final_")
    os.makedirs(work, exist_ok=True)
    # the evaluation text is frozen for the whole run: a RESULTS.md edited
    # mid-run would change the tokens between families
    text = os.path.join(work, "eval.txt")
    if not os.path.exists(text):
        shutil.copyfile(R.TEXT, text)
    R.TEXT = text
    doc = {"results": []}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
    frozen = hashes()
    if doc.get("hashes") and doc["hashes"] != frozen and not a.preflight:
        print("binaries differ from the ones this run started with:")
        for k in sorted(set(doc["hashes"]) | set(frozen)):
            if doc["hashes"].get(k) != frozen.get(k):
                print(f"  {k}: {doc['hashes'].get(k)} -> {frozen.get(k)}")
        print("start a new results file (--json) or restore them; a run is one set of binaries")
        sys.exit(2)
    doc["hashes"] = frozen
    want = [x for x in a.only.split(",") if x]
    redo = set(x for x in a.redo.split(",") if x)
    done = {r["label"] for r in doc["results"] if r["label"] not in redo}
    heads = [h for h in FH.HEADS if (not want or h[0] in want)]
    seen = set()
    heads = [h for h in heads if not (h[0] in seen or seen.add(h[0]))]     # gpt_oss is listed twice
    print(f"{len(heads)} families, {len(done & {h[0] for h in heads})} already done, work {work}", flush=True)
    for label, mt, over, key, what in heads:
        if label in done:
            continue
        if hashes() != frozen:
            print("a binary changed during the run -- stopping (commandment IV)")
            sys.exit(3)
        t0 = time.time()
        try:
            r = one_family(label, mt, over, work, size, rounds, reps, not a.no_gpu)
        except Exception as e:                      # one family's failure must not end the run
            r = {"label": label, "model_type": mt, "error": f"harness: {type(e).__name__}: {e}"[:300]}
        r["roster"], r["represents"], r["minutes"] = key, what, round((time.time() - t0) / 60, 1)
        doc["results"] = [x for x in doc["results"] if x["label"] != label] + [r]
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
        os.replace(out + ".tmp", out)
        shutil.rmtree(os.path.join(work, label), ignore_errors=True)
        try:
            if a.preflight:
                st = r.get("steps", {})
                bad = [k for k, v in st.items() if not v]
                print(f"{label:<22} {'GREEN' if st and not bad and 'error' not in r else 'RED'}  "
                      f"{' '.join(bad)}  {r.get('error', '')[:120]}", flush=True)
            else:
                print(R.show(r) if "error" not in r else f"{label:<18} ERROR {r['error'][:150]}", flush=True)
        except Exception as e:
            print(f"{label}: (could not format: {e})", flush=True)
    rs = [r for r in doc["results"] if not want or r["label"] in want]
    won = sum(1 for r in rs if r.get("family_won"))
    cannot = sum(1 for r in rs if r.get("reference_cannot_load"))
    print(f"\n{won}/{len(rs)} families WON; llama.cpp cannot run {cannot}", flush=True)


if __name__ == "__main__":
    main()
