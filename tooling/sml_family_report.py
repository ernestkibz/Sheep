#!/usr/bin/env python3
"""The final family run as a tick table (commandment IX: every result as measured).

    python tooling/sml_family_report.py [tooling/family_final.json] > table.md

One row per family head: bytes, quality (KL from full precision, Sheep vs
llama.cpp), CPU and GPU prompt (pp128) and decode (tg64) ratios with their
verdicts. Families llama.cpp cannot run, and families Sheep cannot build yet,
are listed on their own -- never counted as ticks (commandment III).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TICK, CROSS = "✅", "❌"


def col(r, k):
    v = r.get(k)
    if k.startswith("gpu") and r.get("gpu_refused"):
        return f"{CROSS} refused"
    if not v or "ratio" not in v:
        return f"{CROSS} -"
    return f"{TICK if v['verdict'] == 'WON' else CROSS} {v['ratio']:.2f}x"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "family_final.json")
    doc = json.load(open(path, encoding="utf-8"))
    rs = doc["results"]
    won = [r for r in rs if r.get("family_won")]
    raced = [r for r in rs if "quality" in r and not r.get("family_won")]
    noref = [r for r in rs if r.get("reference_cannot_load")]
    nobuild = [r for r in rs if "error" in r and not r.get("reference_cannot_load") and "quality" not in r]
    out = []
    out.append(f"**{len(won)} of {len(rs)} family heads WON every column** against llama.cpp "
               f"(Q4_K_M, their converter, quantizer and tools). "
               f"{len(noref)} run on Sheep where llama.cpp cannot; {len(nobuild)} are not yet built by Sheep "
               f"or its reference could not be made.\n")
    out.append("| Family | Bytes | Quality KL (Sheep / llama.cpp) | CPU prompt | CPU decode | GPU prompt | GPU decode | |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in won + raced:
        q = r.get("quality", {})
        qs = (f"{TICK if q.get('won') else CROSS} {q['kld_sheep']:.4f} / {q['kld_llama']:.4f}"
              if "kld_sheep" in q else f"{CROSS} {q.get('error', '')[:30]}")
        b = r.get("bytes", {})
        bs = (f"{TICK if r['won']['bytes'] else CROSS} {b['sheep'] / 1e6:.0f} / {b['llama'] / 1e6:.0f} MB"
              if b else "-")
        out.append(f"| {r['label']} | {bs} | {qs} | {col(r, 'cpu_pp')} | {col(r, 'cpu_tg')} | "
                   f"{col(r, 'gpu_pp')} | {col(r, 'gpu_tg')} | {'WON' if r.get('family_won') else 'not yet'} |")
    if noref:
        out.append("\n**Sheep runs, llama.cpp cannot** (no tick; their converter or loader refuses the model):\n")
        for r in noref:
            out.append(f"- {r['label']}: {r.get('error', '')[:140]}")
    if nobuild:
        out.append("\n**Not raced yet:**\n")
        for r in nobuild:
            out.append(f"- {r['label']}: {r.get('error', '')[:140]}")
    hs = doc.get("hashes", {})
    if hs:
        out.append("\nBinaries (sha256, first 16): " + ", ".join(f"{k} {v}" for k, v in sorted(hs.items())))
    print("\n".join(out))


if __name__ == "__main__":
    main()
