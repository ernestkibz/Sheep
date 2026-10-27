#!/usr/bin/env python3
"""Fit an architecture nobody listed to the engine's existing settings.

    python tooling/sml_autofit.py fit <folder with config.json> [--out fits.json]
    python tooling/sml_autofit.py blind <known model_type> [--out fits.json]

A model_type with no roster row is usually a known model under a new name: the
same norms, rotary, activation and residual wiring as something the engine
already runs, in a different order. So instead of a hand-written row, a tiny
copy of the model is built from transformers' OWN modelling code for it (two
layers, random weights, the real config's behaviour keys), Sheep builds and runs
that copy, and the behaviour flags are searched -- one axis at a time, keeping
whatever lowers the error -- until Sheep's logits land where the family heads'
do (sml_family_heads._measure's pass rule). The fit is saved per model_type; the
builder (sml_hf.arch_flags) and the roster (sml_archs.check) then read it, and the
real checkpoint builds at native speed.

Nothing passes, the builder cannot place a tensor, or transformers has no class
for the type (remote code): the fit says so, and the app runs the model in
compatibility mode (the maker's own code) instead.

`blind` is the proof: a known family's head, its config's model_type renamed so
no name list applies, fitted from nothing.

Needs transformers and torch: run it with the PyTorch runtime's python
(mlmi.backend.gpu_runtime); SML_BUILD_PYTHON names the python that builds.
"""
import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                       # noqa: E402

import sml_family_heads as heads         # noqa: E402
from sml_archs import fits_path          # noqa: E402

# The real config's sizes are dropped: the tiny copy takes the heads' small shape
SIZE_KEYS = {"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
             "moe_intermediate_size", "n_routed_experts", "num_local_experts", "num_experts",
             "shared_expert_intermediate_size", "n_embd", "n_layer", "n_head", "n_inner",
             "d_model", "n_layers", "n_heads", "n_positions", "ffn_hidden_size"}
# what describes the file, not the maths
SKIP_KEYS = {"architectures", "auto_map", "torch_dtype", "dtype", "transformers_version",
             "quantization_config", "_name_or_path", "model_type", "bos_token_id", "eos_token_id",
             "pad_token_id", "tie_word_embeddings", "use_cache", "text_config", "vision_config"}

# (label, the change) -- one axis at a time; activation is one axis of five
ACTS = {"silu": dict(gelu_ffn=False, gelu_erf=False, relu2=False, relu=False),
        "gelu tanh": dict(gelu_ffn=True, gelu_erf=False, relu2=False, relu=False),
        "gelu erf": dict(gelu_ffn=True, gelu_erf=True, relu2=False, relu=False),
        "relu^2": dict(gelu_ffn=False, gelu_erf=False, relu2=True, relu=False),
        "relu": dict(gelu_ffn=False, gelu_erf=False, relu2=False, relu=True)}
AXES = [
    ("norm", {"RMSNorm": dict(norm_type=0), "LayerNorm": dict(norm_type=1)}),
    ("rotary", {"halved rotary": dict(rope_type=0), "interleaved rotary": dict(rope_type=1)}),
    ("activation", ACTS),
    ("late Q/K norm", {"Q/K norm before rotary": dict(qknorm_late=False), "Q/K norm after rotary": dict(qknorm_late=True)}),
    ("residual", {"serial residual": dict(parallel_residual=False), "parallel residual": dict(parallel_residual=True)}),
    ("expert weights", {"experts renormalized": dict(moe_norm=True), "experts as scored": dict(moe_norm=False)}),
    ("router", {"softmax router": dict(moe_sigmoid=False), "sigmoid router": dict(moe_sigmoid=True)}),
]


def tiny_over(cfg):
    """The real config's behaviour keys, at the tiny copy's two layers."""
    n = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    over = {}
    for k, v in cfg.items():
        if k in SIZE_KEYS or k in SKIP_KEYS or k.startswith("_"):
            continue
        if isinstance(v, list) and n and len(v) == n:            # one entry per layer
            v = v[:2]
        elif isinstance(v, list) and v and all(isinstance(x, int) for x in v) and k.endswith(("layers", "indices")):
            v = [x for x in v if x < 2]                          # layer indices
        over[k] = v
    return over


def _score(res):
    if "error" in res:
        return float("inf")
    return max(res["stepped"]["rel"], res["batched"]["rel"])


def search(model, d, mt, log=print, sweeps=2):
    """Greedy search over AXES for the flags that make Sheep compute what `model` does."""
    import sml_hf
    with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    rng = np.random.default_rng(7)
    vocab = min(heads.VOCAB, int(cfg.get("vocab_size") or heads.VOCAB))
    seqs = [list(map(int, rng.integers(300, vocab - 300, size=20))) for _ in range(4)]
    os.environ["SML_UNGRADED_ARCH"] = mt
    base = sml_hf.arch_flags(cfg)
    ssf = os.path.join(d, "model.ssf")

    def measure(delta, what):
        os.environ["SML_FLAGS_OVERRIDE"] = json.dumps(delta)
        if os.path.exists(ssf):
            os.remove(ssf)
        res = {}
        try:
            heads._measure(res, model, d, ssf, seqs, int(os.environ.get("SML_HEAD_BITS", "6")))
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"[:300]}
        s = _score(res)
        log(f"fitting: {what}: " + (f"rel {s:.4f}" + (" PASS" if res.get("pass") else "")
                                     if s < float("inf") else res.get("error", "")[:160]))
        return res, s

    best, n = {}, 0
    try:
        res, s = measure(best, "the settings its config implies")
        n += 1
        if "no place in the name table" in res.get("error", ""):
            return {"pass": False, "why": "the builder cannot place its tensors: " + res["error"][:200], "builds": n}
        for _ in range(sweeps):
            if res.get("pass"):
                break
            moved = False
            for axis, options in AXES:
                if res.get("pass"):
                    break
                for label, change in options.items():
                    if all(best.get(k, base.get(k)) == v for k, v in change.items()):
                        continue                          # what it has already
                    r2, s2 = measure(dict(best, **change), f"trying {label}")
                    n += 1
                    if s2 < s:
                        best, res, s, moved = dict(best, **change), r2, s2, True
                        if r2.get("pass"):
                            break
            if not moved:
                break
    finally:
        os.environ.pop("SML_FLAGS_OVERRIDE", None)
    out = {"pass": bool(res.get("pass")), "rel": s if s < float("inf") else None,
           "sens": res.get("sens"), "builds": n, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if out["pass"]:
        out["flags"] = {k: v for k, v in best.items()}
    else:
        out["why"] = res.get("error") or f"no setting of the engine's matches it (best rel {s:.3f})"
    return out


def save(mt, fit, path):
    try:
        with open(path, encoding="utf-8") as f:
            fits = json.load(f) or {}
    except (OSError, ValueError):
        fits = {}
    fits[mt] = fit
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(fits, f, indent=1)
    os.replace(path + ".tmp", path)


def fit_folder(folder, out=None, log=print):
    """Fit the model_type of a folder's config.json; saved and returned."""
    from transformers import AutoConfig
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg.get("text_config"), dict) and not cfg.get("hidden_size"):
        cfg = dict(cfg["text_config"], **{k: v for k, v in cfg.items() if k == "auto_map"})
    mt = cfg.get("model_type", "")
    try:
        AutoConfig.for_model(mt)
    except Exception:
        return {"pass": False, "why": "no transformers class", "model_type": mt}
    work = tempfile.mkdtemp(prefix="sml_fit_")
    d = os.path.join(work, mt)
    try:
        log(f"fitting: building a tiny {mt} from transformers' own code")
        # its own tokenizer and vocabulary: the builder reads the tokenizer beside the weights
        heads.TOK_SRC = folder
        over = tiny_over(cfg)
        if cfg.get("vocab_size"):
            over["vocab_size"] = int(cfg["vocab_size"])
        for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if k in cfg:
                over[k] = cfg[k]
        # the real K/V-to-Q head ratio (latent attention needs as many K/V heads as Q heads)
        nh, nkv = int(cfg.get("num_attention_heads") or 0), int(cfg.get("num_key_value_heads") or 0)
        if nh and nkv:
            over["num_key_value_heads"] = max(1, heads.BASE["num_attention_heads"] * nkv // nh)
        try:
            model = heads.build_head(mt, mt, over, d, seed=1234)
            heads.hf_last_logits(model, [300, 301, 302])
        except Exception as e:
            # sizes that depend on one another in ways the small shape breaks: the real head
            # geometry, two layers
            log(f"fitting: the small shape does not run ({type(e).__name__}); keeping its real head sizes")
            import shutil as _sh
            _sh.rmtree(d, ignore_errors=True)
            over = dict(over, **{k: cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads",
                                                    "head_dim") if k in cfg})
            model = heads.build_head(mt, mt, over, d, seed=1234)
            heads.hf_last_logits(model, [300, 301, 302])
        import glob
        import shutil
        for f in glob.glob(os.path.join(folder, "*")):     # sentencepiece / tiktoken files too
            n = os.path.basename(f).lower()
            if os.path.isfile(f) and (n.startswith(("tokenizer", "special_tokens")) or n.endswith((".model", ".tiktoken"))
                                      or n in ("vocab.json", "merges.txt", "vocab.txt")):
                shutil.copy(f, os.path.join(d, os.path.basename(f)))
    except Exception as e:
        return {"pass": False, "why": f"transformers: {type(e).__name__}: {e}"[:300], "model_type": mt}
    try:
        fit = search(model, d, mt, log=log)
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    save(mt, fit, out or fits_path())
    fit["model_type"] = mt
    return fit


def blind(known, out=None, log=print):
    """A known family's head with its model_type renamed blind_<type>, fitted from nothing."""
    h = next(x for x in heads.HEADS if x[0] == known)
    work = tempfile.mkdtemp(prefix="sml_blind_")
    d = os.path.join(work, known)
    model = heads.build_head(h[0], h[1], h[2], d, seed=1234)
    p = os.path.join(d, "config.json")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    mt = "blind_" + cfg["model_type"]
    cfg["model_type"] = mt
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    try:
        fit = search(model, d, mt, log=log)
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    save(mt, fit, out or fits_path())
    fit["model_type"] = mt
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "blind"))
    ap.add_argument("target", help="fit: a folder with config.json; blind: a head label (glm, stablelm, ...)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    fn = fit_folder if args.cmd == "fit" else blind
    r = fn(args.target, out=args.out or None, log=lambda s: print(s, flush=True))
    print("RESULT " + json.dumps(r), flush=True)
    return 0 if r.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
