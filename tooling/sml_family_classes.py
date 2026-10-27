#!/usr/bin/env python3
"""Kernel-path classes: which families run the same engine code for the parts
that decide speed and quality, so one race stands for all of them.

The exact signature (sml_family_signatures.py) separates families on details
the fast path does not care about -- a bias, a tied head, a rotary variant, a
scale -- and gave 52 signatures for 63 families: a day of racing. What decides
a race is the kernel path: norm, FFN activation, Q/K norm, how position is
carried, windows, parallel residual, post-norms, soft caps, experts, state-space
layers. Families sharing those run the same kernels; they inherit their class
representative's race, as children inherit their family head's.

    python tooling/sml_family_classes.py            -> prints classes, writes family_classes.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sml_family_heads as FH  # noqa: E402

SIG = os.path.join(HERE, "family_signatures.json")
OUT = os.path.join(HERE, "family_classes.json")


def kclass(sig):
    f, roles = sig["flags"], set(sig["roles"])

    def has(s):
        return any(s in r for r in roles)
    k = ["layernorm" if f.get("norm_type") == 1 else "rmsnorm"]
    if has("ssm_in"):
        k.append("mamba2")
    if has("ssm1_") or has("m1_"):
        k.append("mamba1")
    if has("gdn_"):
        k.append("deltanet")
    if has("shortconv"):
        k.append("shortconv")
    if has(".E."):
        k.append("experts" + ("+shared" if has("shexp") or has("shared") else "")
                 + ("+v3" if f.get("moe_sigmoid") or f.get("moe_groups") else ""))
    gate = has("ffn_gate")
    if f.get("relu2"):
        k.append("relu2")
    elif f.get("gelu_ffn"):
        k.append("geglu" if gate else "gelu")
    else:
        k.append("swiglu" if gate or has(".E.") else "gelu")
    if has("attn_q_norm"):
        k.append("qknorm")
    if f.get("alibi"):
        k.append("alibi")
    elif has("position_embd"):
        k.append("learned-pos")
    if has("attn_window") or f.get("n_swa"):
        k.append("windows")
    if f.get("parallel_residual"):
        k.append("parallel")
    if has("attn_post_norm") or has("ffn_post_norm"):
        k.append("post-norms")
    if not has("attn_norm"):
        k.append("no-pre-norm")
    if f.get("attn_softcap") or f.get("logit_softcap"):
        k.append("softcap")
    if has("attn_gate"):
        k.append("attn-gate")
    return " ".join(k)


def main():
    groups = json.load(open(SIG, encoding="utf-8"))
    order = [h[0] for h in FH.HEADS]
    classes = {}
    for g in groups.values():
        c = kclass(g["signature"])
        for lab in g["labels"]:
            ent = classes.setdefault(c, {"labels": [], "env": g["env"]})
            if lab not in ent["labels"]:
                ent["labels"].append(lab)
    for c in classes.values():
        c["labels"].sort(key=lambda x: order.index(x) if x in order else 999)
        c["representative"] = c["labels"][0]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=1)
    n = sum(len(c["labels"]) for c in classes.values())
    print(f"{n} families -> {len(classes)} kernel-path classes")
    for k, c in sorted(classes.items(), key=lambda kv: -len(kv[1]["labels"])):
        print(f"  [{c['env']}] {c['representative']:<16} ({len(c['labels'])}) {k}: {', '.join(c['labels'][1:])}")


if __name__ == "__main__":
    main()
