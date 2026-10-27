#!/usr/bin/env python3
"""Batched serving against the reference's own batched bench, with a confidence
interval per batch size.

    python tooling/sml_batched_race_ci.py MODEL.ssf REFERENCE.gguf [--npl 1,2,4,8] [--rounds 6]

Each round runs sml-batched and llama-batched-bench back to back (ABBA order),
same prompt length, generation length and thread count. For every batch size B
the verdict is the median of per-round ratios (Sheep / reference) of generation
throughput (S_TG) and of total throughput (S), with a bootstrap 95% interval; WON
only when the whole interval is above 1.
"""
import argparse
import os
import re
import statistics as st
import subprocess

from sml_gpu_race_ci import boot_ci

HERE = os.path.dirname(os.path.abspath(__file__))
SHEEP = os.path.join(HERE, "..", "sml", "sml-batched" + (".exe" if os.name == "nt" else ""))
REF = os.environ.get("REF_BATCHED", r"C:\Users\earne\llama-cpu\llama-batched-bench.exe")


def parse(out):
    got = {}
    for line in out.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 11 or not cells[1].isdigit():
            continue
        try:
            b = int(cells[3])
            got[b] = (float(cells[8]), float(cells[10]))      # S_TG, S
        except ValueError:
            pass
    return got


def sheep(ssf, a):
    r = subprocess.run([SHEEP, ssf, "--npp", str(a.npp), "--ntg", str(a.ntg), "--npl", a.npl,
                        "--threads", str(a.threads)], capture_output=True, text=True, timeout=3600)
    return parse(r.stdout)


def reference(gguf, a):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([REF, "-m", gguf, "-c", str(max(4096, 8 * (a.npp + a.ntg) + 64)),
                        "-npp", str(a.npp), "-ntg", str(a.ntg), "-npl", a.npl, "-t", str(a.threads),
                        "-ngl", "0", "-dev", "none"],
                       capture_output=True, text=True, timeout=3600, env=env, cwd=os.path.dirname(REF))
    return parse(r.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ssf")
    ap.add_argument("gguf")
    ap.add_argument("--npl", default="1,2,4,8")
    ap.add_argument("--npp", type=int, default=128)
    ap.add_argument("--ntg", type=int, default=128)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()
    bs = [int(x) for x in a.npl.split(",")]
    tg = {b: [] for b in bs}
    tot = {b: [] for b in bs}
    print(f"== {os.path.basename(a.ssf)} vs {os.path.basename(a.gguf)}: {a.npp}-token prompts, "
          f"{a.ntg} generated, B = {a.npl}, {a.threads} threads each, {a.rounds} rounds")
    for r in range(a.rounds):
        order = ("sheep", "ref") if r % 2 == 0 else ("ref", "sheep")
        got = {}
        for who in order:
            got[who] = sheep(a.ssf, a) if who == "sheep" else reference(a.gguf, a)
        line = []
        for b in bs:
            if b in got["sheep"] and b in got["ref"]:
                (st_s, s_s), (st_r, s_r) = got["sheep"][b], got["ref"][b]
                tg[b].append(st_s / st_r)
                tot[b].append(s_s / s_r)
                line.append(f"B={b} tg {st_s:.1f}/{st_r:.1f}")
        print(f"  round {r + 1} ({order[0]} first): " + "  ".join(line))
    for b in bs:
        for name, xs in (("generation", tg[b]), ("total", tot[b])):
            if not xs:
                continue
            lo, hi = boot_ci(xs)
            v = "WON" if lo > 1.0 else ("LOST" if hi < 1.0 else "NOT DECIDED")
            print(f"  B={b} {name:<10}: median ratio {st.median(xs):.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
                  f"{sum(x > 1 for x in xs)}/{len(xs)} rounds above 1 -> {v}")


if __name__ == "__main__":
    main()
