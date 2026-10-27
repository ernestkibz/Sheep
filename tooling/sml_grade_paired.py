#!/usr/bin/env python3
"""Paired grade of several .ssf builds of ONE model on many held-out prefixes --
the measure to use before saying one build is better or worse than another.

    python tooling/sml_grade_paired.py HF_CHECKPOINT A.ssf B.ssf [C.ssf ...] [--n 100]

Each prefix (8-96 tokens of RESULTS.md, a text no build calibrates on) is scored
by its last-token logits' rel-RMS to Hugging Face full precision, for every file
on the SAME prefixes; the verdict for B, C, ... is the mean paired difference to
A with a bootstrap 95% interval.

Why it exists (2026-09-25): sml_hf_grade.py's 20 prompts (sd ~0.09) said a fresh
1B build was 21% worse than the Sep 10 one -- a "builder regression" that sent the
sprint looking through the builder's diff. Headers, float tensors, tokenizer and
per-tensor output error on true activations were all identical, and on 100 paired
prefixes the difference was -0.0026 [-0.0128, +0.0079]: one quantization draw
against another. The same test found the real effect it was hiding: --scale6
costs the 1B +0.0137 [+0.0046, +0.0235], 8%.
"""
import argparse
import os
import random
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "sml", "sml-cli" + (".exe" if os.name == "nt" else ""))
TEXT = os.path.join(HERE, "..", "RESULTS.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hf")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf)
    m = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32)
    m.eval()
    ids_all = tok(open(TEXT, encoding="utf-8").read())["input_ids"]
    if tok.bos_token_id is not None and ids_all and ids_all[0] == tok.bos_token_id:
        ids_all = ids_all[1:]
    rng = random.Random(a.seed)
    prefixes = []
    for _ in range(a.n):
        k = rng.randint(8, 96)
        s = rng.randint(0, len(ids_all) - k - 1)
        head = [tok.bos_token_id] if tok.bos_token_id is not None else []
        prefixes.append(head + ids_all[s:s + k])
    with torch.no_grad():
        ref = [m(torch.tensor([p])).logits[0, -1].numpy().astype(np.float64) for p in prefixes]

    errs = {}
    out = os.path.join(tempfile.gettempdir(), "sml_grade_paired.bin")
    for f in a.files:
        e = []
        for p, r in zip(prefixes, ref):
            subprocess.run([CLI, f, "--tokens", ",".join(map(str, p)), "--gen", "0", "--threads", "6",
                            "--logits", out], capture_output=True,
                           env=dict(os.environ, SML_SEQ_PROMPT="1"))
            l = np.fromfile(out, dtype=np.float32)[2:].astype(np.float64)
            e.append(np.linalg.norm(l - r) / np.linalg.norm(r))
        errs[f] = np.array(e)
        print(f"{os.path.basename(f):32s} mean rel-RMS {errs[f].mean():.4f}  median {np.median(errs[f]):.4f}",
              flush=True)
    base = a.files[0]
    brng = np.random.default_rng(3)
    for f in a.files[1:]:
        d = errs[f] - errs[base]
        boots = [d[brng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        v = "WORSE" if lo > 0 else ("BETTER" if hi < 0 else "EQUAL (interval spans 0)")
        print(f"{os.path.basename(f)} - {os.path.basename(base)}: mean {d.mean():+.4f}, 95% CI "
              f"[{lo:+.4f}, {hi:+.4f}], worse on {int((d > 0).sum())}/{len(d)} -> {v}")


if __name__ == "__main__":
    main()
