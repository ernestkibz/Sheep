#!/usr/bin/env python3
"""Pilot race: one REAL checkpoint, Sheep against llama.cpp, every column.

    python tooling/sml_pilot_race.py C:/Users/earne/models/llama32-3b --label llama32-3b

From the one checkpoint:
  1. llama.cpp: THEIR converter (f16) and THEIR quantizer (Q4_K_M)
  2. Sheep: the family recipe (sml_family_race.sheep_build) with llama.cpp's
     file size as the byte budget, calibrated on more text than a head needs
  3. quality: llama-perplexity saves the f16 model's own distributions on held-
     out text (their reference, their tool) and Q4_K_M's on the same tokens;
     Sheep is run on exactly those tokens and packed the same way. KL from f16
     and top-1, paired per position, bootstrap CI.
  4. speed: sml_cpu_race_ci.py and sml_gpu_race_ci.py, pp128 and decode,
     equal threads, ABBA rounds, equal repetitions per side
Results go to tooling/pilot_race.json (one entry per label, replaced on re-run).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sml_family_race as R  # noqa: E402

OUT = os.path.join(HERE, "pilot_race.json")


def logprob_rows(path):
    n_ctx, toks, lp = R.read_llama_logits(path)
    return n_ctx, toks, lp


def quality(d, ssf, f16, q4, work):
    base, qf = os.path.join(work, "base_f16.bin"), os.path.join(work, "q4.bin")
    for m, out in ((f16, base), (q4, qf)):
        if not os.path.exists(out):
            r = R.run([R.PERPLEXITY, "-m", m, "-f", R.TEXT, "-c", str(R.N_CTX), "--chunks", str(R.CHUNKS),
                       "-t", "6", "-b", str(R.N_CTX), "--kl-divergence-base", out], timeout=3600)
            if not os.path.exists(out):
                return {"error": "llama-perplexity: " + (r.stdout + r.stderr).strip()[-300:]}
    n_ctx, toks, lb = logprob_rows(base)
    _, toks2, lq = logprob_rows(qf)
    if not np.array_equal(toks, toks2):
        return {"error": "the two llama.cpp runs saw different tokens"}
    add_bos, bos_id = R.gguf_add_bos(q4)
    first = n_ctx // 2
    kl = {"sheep": [], "llama": []}
    top = {"sheep": 0, "llama": 0}
    n = 0
    for c in range(toks.shape[0]):
        ids = [int(t) for t in toks[c]]
        if add_bos and bos_id is not None:
            ids[0] = bos_id
        out = os.path.join(work, "sheep_rows.bin")
        s = R.run([R.CLI, ssf, "--tokens", ",".join(map(str, ids)), "--gen", "0", "--threads", "6",
                   "--logits", out], env={"SML_ALL_LOGITS": str(first)}, timeout=1800)
        if s.returncode != 0 or not os.path.exists(out):
            return {"error": "sheep: " + (s.stdout + s.stderr).strip()[-300:]}
        meta = np.fromfile(out, dtype=np.int32, count=2)
        rows = np.fromfile(out, dtype=np.float32, offset=8).reshape(int(meta[1]), int(meta[0])).astype(np.float64)
        nv = lb.shape[-1]
        for i in range(n_ctx - 1 - first):
            pr = lb[c, i]
            p = np.exp(pr)
            for who, lp in (("sheep", R.pack_like_llama(rows[i][:nv])), ("llama", lq[c, i])):
                kl[who].append(float((p * (pr - lp)).sum()))
                top[who] += int(np.argmax(lp) == np.argmax(pr))
            n += 1
    ks, kl_ = np.array(kl["sheep"]), np.array(kl["llama"])
    lo, hi = R.boot_ci(ks - kl_)
    return {"kld_sheep": float(ks.mean()), "kld_llama": float(kl_.mean()),
            "top1_sheep": top["sheep"] / n, "top1_llama": top["llama"] / n, "positions": n,
            "kld_diff_ci": [lo, hi], "won": hi < 0, "lost": lo > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hf")
    ap.add_argument("--label", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--keep-f16", action="store_true")
    a = ap.parse_args()
    work = os.path.join(os.path.dirname(os.path.abspath(a.hf)), a.label + "-race")
    os.makedirs(work, exist_ok=True)
    text = os.path.join(work, "eval.txt")
    if not os.path.exists(text):
        shutil.copyfile(R.TEXT, text)
    R.TEXT = text
    f16, q4, ssf = (os.path.join(work, "ref-f16.gguf"), os.path.join(work, "ref-q4km.gguf"),
                    os.path.join(work, "sheep.ssf"))
    res = {"label": a.label, "hf": a.hf, "t": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    if not os.path.exists(q4):
        if not os.path.exists(f16):
            c = R.run([sys.executable, R.CONVERT, a.hf, "--outtype", "f16", "--outfile", f16,
                       "--model-name", a.label], timeout=7200)
            if not os.path.exists(f16):
                print("llama.cpp converter failed:", (c.stdout + c.stderr)[-500:])
                sys.exit(1)
        R.run([R.QUANTIZE, f16, q4, "Q4_K_M"], timeout=7200)
    cfg = json.load(open(os.path.join(a.hf, "config.json"), encoding="utf-8"))
    n_layer = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    res["alloc"] = R.sheep_alloc(n_layer)
    if not os.path.exists(ssf):
        b = R.run([sys.executable, os.path.join(HERE, "sml_quantize.py"), a.hf, ssf, "--layout", "flat",
                   "--alloc", res["alloc"], "--no-gptq", "--scale6", "--seqs", str(a.seqs),
                   "--seq-len", str(a.seq_len), "--budget", str(os.path.getsize(q4))], timeout=14400)
        if not os.path.exists(ssf) or b.returncode != 0:
            print("sheep build failed:", (b.stdout + b.stderr)[-800:])
            sys.exit(1)
    res["build_min"] = round((time.time() - t0) / 60, 1)
    res["bytes"] = {"sheep": os.path.getsize(ssf), "llama": os.path.getsize(q4)}
    res["quality"] = quality(a.hf, ssf, f16, q4, work)
    if not a.keep_f16 and os.path.exists(f16):
        os.remove(f16)
    ex = ["--reps", str(a.reps)]
    res["cpu_pp"] = R.race("sml_cpu_race_ci.py", ssf, q4, ["--p", "128"] + ex, a.rounds)
    res["cpu_tg"] = R.race("sml_cpu_race_ci.py", ssf, q4, ["--decode", "64"] + ex, a.rounds)
    if not a.no_gpu:
        why = R.gpu_refusal(ssf)
        if why:
            res["gpu_refused"] = why
        else:
            res["gpu_pp"] = R.race("sml_gpu_race_ci.py", ssf, q4, ["--p", "128"] + ex, a.rounds)
            res["gpu_tg"] = R.race("sml_gpu_race_ci.py", ssf, q4, ["--decode"] + ex, a.rounds)
    won = lambda k: res.get(k, {}).get("verdict") == "WON"  # noqa: E731
    res["won"] = {"bytes": res["bytes"]["sheep"] <= res["bytes"]["llama"],
                  "quality": bool(res["quality"].get("won")),
                  "cpu": won("cpu_pp") and won("cpu_tg"),
                  "gpu": (not a.no_gpu) and won("gpu_pp") and won("gpu_tg")}
    res["family_won"] = all(res["won"].values())
    allr = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else []
    allr = [x for x in allr if x["label"] != a.label] + [res]
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(allr, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(R.show(res))


if __name__ == "__main__":
    main()
