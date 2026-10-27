#!/usr/bin/env python3
"""Group the family heads by ENGINE SIGNATURE: the exact set of engine features a
family exercises -- its tensor roles and its header flags, sizes stripped.

Families with one signature run the same kernels in the same order; only sizes
differ. So one representative per signature is raced against llama.cpp
(sml_family_race.py) and the others inherit its verdict, as a family's
children inherit its head's -- Ernest's family-head idea applied one level up,
because racing 80 heads in full is a day of machine time.

    python tooling/sml_family_signatures.py        (and again with the venv-tf5 python)

Writes tooling/family_signatures.json: {signature: [labels]} per environment.
Each family's own arithmetic is already graded against transformers by
sml_family_heads.py; this only decides which of them need their own race.
"""
import hashlib
import shutil
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sml_family_heads as FH  # noqa: E402
from sml_hf import open_weights  # noqa: E402

OUT = os.path.join(HERE, "family_signatures.json")
# values whose SIZE changes nothing about which kernel runs: only presence counts
PRESENCE = {"attn_scale", "logit_softcap", "attn_softcap", "n_swa", "emb_scale", "res_scale",
            "logit_scale", "clip_qkv", "moe_scale", "n_expert_used", "rope_theta", "n_rot",
            "ssm_dstate", "ssm_groups", "ssm_norm_group", "ssm_dt_lo", "ssm_dt_hi", "ssm_bcdt_eps",
            "gdn_nk", "gdn_dk", "moe_groups", "moe_topk_group", "rms_eps", "n_ctx_train"}


def role(name):
    name = re.sub(r"^blk\.\d+\.", "", name)
    return re.sub(r"\.\d+\.", ".E.", name)        # expert index


def signature(d, W=None):
    W = W or open_weights(d)
    fl = W.flags()
    flags = {}
    for k, v in sorted(fl.items()):
        if isinstance(v, (list, tuple, dict)):
            v = bool(v)
        if k in PRESENCE:
            v = bool(v)
        flags[k] = v
    hd = W.cfg.get("head_dim") or (W.cfg.get("hidden_size", 0) // max(1, W.cfg.get("num_attention_heads", 1)))
    flags["partial_rotary"] = bool(fl.get("n_rot")) and bool(hd) and int(fl["n_rot"]) < int(hd)
    roles = sorted({role(n) for n in W.t})
    sig = {"flags": flags, "roles": roles}
    return sig, hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:10]


def main():
    env = "tf5" if sys.version_info and "venv-tf5" in sys.executable else "main"
    groups = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            groups = json.load(f)
    groups = {k: v for k, v in groups.items() if v.get("env") != env}
    work = tempfile.mkdtemp(prefix="sml_sig_")
    for label, mt, over, key, what in FH.HEADS:
        d = os.path.join(work, label)
        try:
            FH.build_head(label, mt, over, d, seed=1)
            sig, h = signature(d)
        except (Exception, SystemExit) as e:    # this env cannot build it, or the reader refuses it
            print(f"  {label:<20} skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        shutil.rmtree(d, ignore_errors=True)       # ~100 MB a head; 85 of them filled the disk
        g = groups.setdefault(h, {"env": env, "labels": [], "signature": sig})
        g["labels"].append(label)
        print(f"  {label:<20} {h}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    mine = [g for g in groups.values() if g["env"] == env]
    print(f"\n{sum(len(g['labels']) for g in mine)} families -> {len(mine)} signatures ({env})")
    for h, g in groups.items():
        if g["env"] == env:
            print(f"  {h}: {', '.join(g['labels'])}")


if __name__ == "__main__":
    main()
