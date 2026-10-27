"""Quantization levels: what each one is, and what it means for one model here.

Every offered level has a distinct builder/runtime path; exact-F32 is a
non-quantizing CPU build rather than another quantization recipe:

    Small      every weight 4-bit; the embedding 6-bit (it is the head too on a
               tied model, and carried a third of all error at 4-bit)
    Balanced   4-bit, 6-bit where it matters: Sheep's race recipe, which beat
               llama.cpp's Q4_K_M on quality and size
    High       every weight 6-bit

For a checkpoint (a local folder or a Hugging Face id) the cards give each
level's file size from the tensor headers alone (nothing is downloaded), the
writing speed this machine's plan predicts, and whether it fits.
"""
import json
import struct
import sys
from pathlib import Path

from .paths import REPO

TOOLING = REPO / "tooling"
BUILD_RATE = 0.5e6               # weights a second the builder reaches (Python/numpy, measured)

# Q2 to Q8 (Ernest, 2026-10-05). width: every weight's code width; emb: the
# embedding's (it is the head too on a tied model, and carries the most error).
ALL_LEVELS = [
    {"id": "q2", "name": "Q2 · Tiny", "bits": "2-bit", "width": 2, "emb": 6,
     "what": "every weight 2-bit, the embedding 6-bit", "closeness": "rough: a big loss on small models"},
    {"id": "q3", "name": "Q3 · Compact", "bits": "3-bit", "width": 3, "emb": 6,
     "what": "every weight 3-bit, the embedding 6-bit", "closeness": "a noticeable loss on small models"},
    {"id": "small", "name": "Q4 · Small", "bits": "4-bit", "width": 4, "emb": 6,
     "what": "every weight 4-bit, the embedding 6-bit", "closeness": "close to the original"},
    {"id": "balanced", "name": "Q4/6 · Balanced", "bits": "4/6-bit",
     "what": "4-bit, with 6-bit where it matters most: Sheep's race recipe",
     "closeness": "very close: beat llama.cpp's Q4_K_M"},
    {"id": "q5", "name": "Q5 · Fine", "bits": "5-bit", "width": 5, "emb": 6,
     "what": "every weight 5-bit, the embedding 6-bit", "closeness": "very close"},
    {"id": "high", "name": "Q6 · High", "bits": "6-bit", "width": 6, "emb": 6,
     "what": "every weight 6-bit", "closeness": "closer still"},
    {"id": "q7", "name": "Q7 · Very high", "bits": "7-bit", "width": 7, "emb": 8,
     "what": "every weight 7-bit, the embedding 8-bit", "closeness": "nearly the original"},
    {"id": "q8", "name": "Q8 · Near-lossless", "bits": "8-bit", "width": 8, "emb": 8,
     "what": "every weight 8-bit", "closeness": "nearly the original"},
    {"id": "exact", "name": "As-is · Exact weights", "bits": "No quantization",
     "what": "Original numeric weights stored as f32; larger files, no quantization",
     "closeness": "exact numeric round-trip"},
]
# A level is offered only once its kernels passed the engine gates (kernel vs
# format, forward-pass wiring) on a build of it (Ernest, 2026-10-05: all said must
# work). Q8 2026-10-06: wiring corr 0.99944; Q2: 0.99946; Q5: 0.99946; Q3: 0.99936; Q7: 0.99941.
VERIFIED = {"q2", "q3", "small", "balanced", "q5", "high", "q7", "q8", "exact"}
LEVELS = [lv for lv in ALL_LEVELS if lv["id"] in VERIFIED]
ROLES = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
         "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down",
         "embed_tokens": "token_embd", "lm_head": "output", "wte": "token_embd"}
ALL_ROLES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
             "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "token_embd", "output")


def alloc(level, n_layer):
    """The builder's --alloc for a level."""
    sys.path.insert(0, str(TOOLING))
    from sml_recipe import sheep_alloc
    lv = next((x for x in ALL_LEVELS if x["id"] == level), None)
    if not lv or "width" not in lv:
        return sheep_alloc(n_layer)                 # Balanced: the race recipe
    w, e = lv["width"], lv["emb"]
    return ",".join(f"{r}={e if r in ('token_embd', 'output') else w}" for r in ALL_ROLES)


def _bits(level, role, layer, n_layer):
    """Code width the level gives a tensor (the builder's own rule, for sizing)."""
    a = alloc(level, n_layer)
    for part in a.split(","):
        r, _, b = part.partition("=")
        if r != role:
            continue
        if "@" in b:
            bits, _, lst = b.partition("@")
            if layer is not None and str(layer) in lst.split(";"):
                return int(bits)
        else:
            return int(b.partition(":")[0])
    return 4


def _role(name):
    """(role, layer) for an original tensor name like model.layers.3.self_attn.q_proj.weight."""
    parts = name.split(".")
    layer = next((int(p) for p in parts if p.isdigit()), None)
    for p in parts:
        if p in ROLES:
            return ROLES[p], layer
    return None, layer


def _local_tensors(d):
    out = []
    for f in sorted(Path(d).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        for k, v in head.items():
            if k != "__metadata__":
                out.append((k, list(v.get("shape") or [])))
    return out


def inspect(src):
    """What the checkpoint is, from its config and tensor headers."""
    d = Path(src)
    looks_local = ":" in src or "\\" in src or src.count("/") != 1 or src.startswith((".", "/", "~"))
    if not d.is_dir() and looks_local:
        raise ValueError(f"no such folder: {src}")
    if d.is_dir():
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else {}
        tensors = _local_tensors(d)
        lic = ""
        readme = d / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8", errors="replace")
            if text.startswith("---"):
                for line in text.split("---", 2)[1].splitlines():
                    if line.startswith("license:"):
                        lic = line.split(":", 1)[1].strip().strip("'\"")
        diffusers = (d / "model_index.json").exists()
    else:
        from huggingface_hub import hf_hub_download, model_info
        try:
            cfg = json.loads(Path(hf_hub_download(src, "config.json")).read_text(encoding="utf-8"))
            diffusers = False
        except Exception as exc:
            if any(x in str(exc).lower() for x in ("gated", "401", "403", "unauthorized", "restricted")):
                raise                                     # a gated original: the caller says sign in
            # no config.json: a diffusers pipeline only when it has a model_index.json; anything
            # else (a GGUF page read as "image", 2026-10-19) is no Transformers checkpoint
            cfg = {}
            try:
                hf_hub_download(src, "model_index.json")
                diffusers = True
            except Exception:
                raise ValueError(f"{src} has no config.json: it is not an original Transformers checkpoint")
        tensors = []
        try:
            lic = (model_info(src).card_data or {}).get("license", "") or ""
        except Exception:
            lic = ""
    sys.path.insert(0, str(TOOLING))
    import sml_archs
    import sml_hf
    tc = cfg.get("text_config") or {}
    mt = sml_hf.model_type_of(cfg) if cfg else ""
    n_layer = int(cfg.get("num_hidden_layers") or tc.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    # a vision wrapper whose text model Sheep builds (Gemma 3, Mistral Small 3.x,
    # Qwen 3.5) is a text model here; its images are not run
    text_path = mt in sml_hf._MM_TEXT
    kind = "image" if diffusers else ("audio" if mt in ("whisper", "wav2vec2", "speecht5") else
                                      "vision" if cfg.get("vision_config") and not text_path else "text")
    row = sml_archs.check(mt, cfg) if mt else None
    ok = kind == "text" and row is not None and row[0] == "ok"
    note = sml_archs.explain(mt, cfg) if mt else "no model_type in config.json"
    if kind == "text" and not diffusers:
        # the builder's own checks on the checkpoint's headers (sml_compat.py):
        # names, layouts, rotary scaling, tokenizer -- seconds for any size,
        # so a model that would fail an hour into its build fails here instead
        import shutil
        import tempfile
        import sml_compat
        work = tempfile.mkdtemp(prefix="sheep_inspect_")
        try:
            stub, _ = sml_compat._stub_dir(str(src), work)
            tensors = tensors or _local_tensors(stub)
            if ok:
                got = sml_compat.check_dir(stub)
                if not got.get("ok"):
                    ok, note = False, f"'{mt}': " + str(got.get("why", "the header check failed"))
        except (Exception, SystemExit) as e:
            # headers that could not be read (a Hub hiccup) say nothing against the model: the build
            # reads them again and stops plainly if they are wrong. Taken as a failed check, a fitted
            # Youtu-LLM was sent to compatibility mode (2026-10-20)
            note = f"{note}; the headers could not be read now ({str(e)[:160]}), the build checks them"
        finally:
            shutil.rmtree(work, ignore_errors=True)
    n_head = int(cfg.get("num_attention_heads") or tc.get("num_attention_heads") or 1)
    n_kv = int(cfg.get("num_key_value_heads") or tc.get("num_key_value_heads") or n_head)
    hidden = int(cfg.get("hidden_size") or tc.get("hidden_size") or 0)
    head_dim = int(cfg.get("head_dim") or tc.get("head_dim") or (hidden // n_head if hidden else 128))
    n_ctx = int(cfg.get("max_position_embeddings") or tc.get("max_position_embeddings") or 8192)
    # not built natively: an architecture nobody listed is fitted to the engine first
    # (sml_autofit.py); one that needs maths the engine lacks runs in compatibility mode
    fallback = ""
    if kind == "text" and not ok and mt:
        fallback = "fit" if row is None or row[0] == "unknown" else "compat" if row[0] in ("gap", "ok") else ""
    n_weights = sum(_numel(s) for _, s in tensors if len(s) >= 2)
    if not n_weights and kind == "text" and hidden and n_layer:
        # the headers could not be read (a Hub hiccup): size it from its config, so the cards never
        # say "0.00 GB" at four million words a second (2026-10-21)
        inter = int(cfg.get("intermediate_size") or tc.get("intermediate_size") or 4 * hidden)
        vocab = int(cfg.get("vocab_size") or tc.get("vocab_size") or 32000)
        attn = 2 * hidden * n_head * head_dim + 2 * hidden * n_kv * head_dim
        tied = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))
        n_weights = n_layer * (attn + 3 * hidden * inter) + vocab * hidden * (1 if tied else 2)
    return {"src": str(src), "model_type": mt, "kind": kind, "n_layer": n_layer, "licence": str(lic),
            "fallback": fallback, "remote_code": bool(cfg.get("auto_map")),
            "n_kv": n_kv, "head_dim": head_dim, "n_ctx_train": n_ctx,
            "tensors": tensors, "builds": ok, "note": note, "text_path": text_path,
            "n_weights": n_weights, "n_weights_estimated": bool(n_weights) and not tensors}


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def size(tensors, level, n_layer, n_weights=0):
    """Bytes of the Sheep file at a level: codes, the two-level scales and mins
    (about 0.53 bits a weight, fitted to the 3B file), the column order, and the f32 vectors."""
    total = 4096
    if not tensors and n_weights:
        # headers unread (a Hub hiccup): the whole model at the level's main width, from its size
        bits = 32 if level == "exact" else next((x.get("width", 4) for x in ALL_LEVELS if x["id"] == level), 4) + 0.53
        return int(total + n_weights * bits / 8)
    for name, shape in tensors:
        n = _numel(shape)
        if level == "exact":
            total += 4 * n
            continue
        if len(shape) < 2:
            total += 4 * n
            continue
        role, layer = _role(name)
        b = _bits(level, role, layer, n_layer) if role else next((x.get("width", 4) for x in ALL_LEVELS if x["id"] == level), 4)
        total += n * (b + 0.53) / 8 + 4 * int(shape[-1])
    return int(total)


def fit(p, facts, dev, pref="auto"):
    """Green, amber or red for one plan, in words a person reads: where the layers go in
    the chosen mode. One rule for the quantize cards and the mode buttons (2026-10-14:
    Ernest: a big model red on one GPU or the CPU should turn amber or green on hybrid
    or multi, where there is more memory)."""
    n = int(facts.get("n_layer") or 0) or 1
    dev_ = p.get("device", "cpu")
    if not p.get("ram_fit", True):
        return "red", "too big for this machine in this mode"
    if facts.get("exact"):
        return "green", "fits · runs on CPU"
    on_gpu = {"gpu": n, "hybrid": int(p.get("gpu_layers") or 0),
              "multi": int(p.get("gpu_layers") or 0) + int(p.get("mid_layers") or 0),
              "amd": sum(int(k) for _, k in (p.get("dml_split") or [])) or n}.get(dev_, 0)
    on_gpu = min(on_gpu, n)
    asked_gpu = pref in ("gpu", "hybrid", "multi", "amd")
    if facts.get("gpu_widths_ok") is False and dev.get("gpu") and dev_ not in ("amd",):
        return "amber", "fits · no NVIDIA GPU at this width yet (CPU)"
    if asked_gpu and dev_ == "cpu":
        return "amber", "fits · this mode cannot hold it on a GPU, so it runs on the CPU"
    if dev_ == "cpu":
        has_gpu = bool(dev.get("gpu") or dev.get("amd") or dev.get("pool"))
        return ("amber", "fits · CPU only (slower)") if has_gpu else ("green", "fits · CPU")
    if on_gpu >= n:
        return "green", "fits on the GPU" + ("s" if dev_ == "multi" else "")
    return "amber", f"fits · {on_gpu} of {n} layers on the GPU{'s' if dev_ == 'multi' else ''}, the rest on the CPU"


def cards(info, dev, plan_fn, pref="auto", **plan_kw):
    """One card per level: size, the plan's predicted speed and device, the fit in the
    chosen mode (plan_kw: the pool, VRAM and GPU+CPU settings), and the recommendation."""
    out = []
    n_layer = info["n_layer"] or 1
    for lv in LEVELS:
        b = size(info["tensors"], lv["id"], n_layer, info.get("n_weights", 0))
        facts = {"bytes": b, "n_layer": n_layer, "n_ctx_train": info.get("n_ctx_train") or 8192,
                 "n_weights": info["n_weights"],
                 "kv_per_token": n_layer * info.get("n_kv", 8) * info.get("head_dim", 128) * 2 * 2,
                 "exact": lv["id"] == "exact",
                 # Balanced mixes 4 and 6 bits; Q2, Q3, Q5, Q7, Q8 have no NVIDIA kernel yet
                 "gpu_widths_ok": lv.get("width") in (None, 4, 6),
                 "widths": [lv["width"]] if lv.get("width") else [4, 6]}
        p = plan_fn(dev, facts, pref, **plan_kw)
        colour, words = fit(p, facts, dev, pref)
        out.append({**lv, "bytes": b, "device": p["device"], "gpu_layers": p.get("gpu_layers", 0),
                    "est_tok_s": p["est_tok_s"], "fit": colour, "fit_word": words,
                    "build_min": max(1, round(info["n_weights"] / BUILD_RATE / 60))})
    # The pick for THIS machine, whatever cards it has (Ernest, 2026-10-14: the benefit of
    # more GPUs without the cost). The slowest part holding layers sets the pace, so:
    # 1. the best quality that the GPUs hold whole (green): every layer on fast memory;
    # 2. else the fastest planned build of 4-bit quality or better (more layers on the GPUs);
    # 3. else the largest that fits at all. Q2/Q3 and exact weights are never the pick.
    usable = [c for c in out if c["fit"] != "red" and c["id"] not in ("q2", "q3", "exact")]
    whole = [c for c in usable if c["fit"] == "green" and c["device"] != "cpu"]
    rec, why = None, ""
    if whole:
        rec = max(whole, key=lambda c: (c.get("width") or 4.5, c["est_tok_s"]))
        why = "the best quality your GPUs hold whole"
    elif usable:
        four_up = [c for c in usable if (c.get("width") or 4.5) >= 4] or usable
        rec = max(four_up, key=lambda c: (c["est_tok_s"], c.get("width") or 4.5))
        why = ("the most of it on your GPUs, so the fastest here" if rec["device"] != "cpu"
               else "the fastest that fits here")
    for c in out:
        c["recommended"] = rec is not None and c["id"] == rec["id"]
        if c["recommended"]:
            c["why_recommended"] = why
    return out
