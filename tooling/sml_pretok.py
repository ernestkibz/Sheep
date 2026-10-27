"""A BPE tokenizer's own text pipeline, packed for the engine (2026-10-19).

Measured with sml_tok_check.py --engine before this: the engine cut text into the same ids as
the model's own tokenizer on 122/135 lines for Qwen2.5, 112 for Llama-3.2, 96 for Gemma 2, 2 for
Mistral v0.3 and 0 for Phi-3 -- it split at every space and knew no model's rules. A
tokenizer.json says exactly how its text is cut before the merges run: a normalizer (prepend
U+2581, replace spaces), a pre-tokenizer (regex splits with a behaviour, byte-level mapping,
Metaspace, digit splits) and model options (ignore_merges, byte_fallback). llama.cpp follows
them per model; so does Sheep now, from the file itself, with no per-model code.

The trailer after the merges (see sml-vocab.h):
  "PRET", u32 bytes after this field, u32 version (1), u32 flags (PRET_*), u32 unk (0xFFFFFFFF none)
  u32 n_norm, n_norm ops; u32 n_pre, n_pre ops       op: u8 kind, u8 a, u8 b, u8 c,
                                                         u32 len1, bytes1, u32 len2, bytes2
  u32 n_added, u32 ids[n_added]                       added tokens, cut out whole first
  u32 n_cats, each: char name[4], u32 n, (u32 lo, u32 hi)[n]   Unicode classes the regexes use
"""
import functools
import re
import struct
import unicodedata

SPACE = "\u2581"
MAGIC = b"PRET"

# flags
BYTE_LEVEL = 1        # pieces are mapped byte by byte to GPT-2's printable code points before the merges
IGNORE_MERGES = 2     # a piece that is itself a token is taken whole (Llama 3)
BYTE_FALLBACK = 4     # an unknown character becomes its <0xNN> tokens
FUSE_UNK = 8          # a run of unknown characters is one <unk>

# normalizer ops
N_PREPEND, N_REPLACE, N_NFC, N_LOWER, N_STRIP = 1, 2, 3, 4, 5
# pre-tokenizer ops
P_SPLIT_REGEX, P_SPLIT_STRING, P_BYTELEVEL, P_METASPACE, P_DIGITS, P_PUNCT = 1, 2, 3, 4, 5, 6
# split behaviours (tokenizers' SplitDelimiterBehavior)
BEHAVIOUR = {"removed": 0, "isolated": 1, "mergedwithprevious": 2, "mergedwithnext": 3, "contiguous": 4}
SCHEME = {"never": 0, "first": 1, "always": 2}

GPT2_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def _op(kind, a=0, b=0, c=0, s1="", s2=""):
    return {"kind": kind, "a": a, "b": b, "c": c, "s1": s1, "s2": s2}


def from_tokenizer_json(tj):
    """The pipeline of a tokenizer.json whose model is BPE; SystemExit names any step the engine
    does not follow (a build refused beats a model fed text cut the wrong way)."""
    model = tj.get("model") or {}
    flags = 0
    if model.get("ignore_merges"):
        flags |= IGNORE_MERGES
    if model.get("byte_fallback"):
        flags |= BYTE_FALLBACK
    if model.get("fuse_unk"):
        flags |= FUSE_UNK
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise SystemExit("BPE tokenizer with a subword prefix/suffix: not one the engine follows")
    norm, pre = [], []

    def walk_norm(n):
        if not n:
            return
        t = n.get("type")
        if t == "Sequence":
            for x in n.get("normalizers") or []:
                walk_norm(x)
        elif t == "Prepend":
            norm.append(_op(N_PREPEND, s1=n.get("prepend", "")))
        elif t == "Replace":
            pat = n.get("pattern") or {}
            if "String" not in pat:
                raise SystemExit(f"BPE tokenizer: Replace by regex {pat} is not followed yet")
            norm.append(_op(N_REPLACE, s1=pat["String"], s2=n.get("content", "")))
        elif t in ("NFC", "NFKC"):
            norm.append(_op(N_NFC))           # text arrives composed in practice; recorded, not applied
        elif t == "Lowercase":
            norm.append(_op(N_LOWER))
        elif t == "Strip":
            norm.append(_op(N_STRIP, a=int(bool(n.get("strip_left", True))), b=int(bool(n.get("strip_right", True)))))
        else:
            raise SystemExit(f"BPE tokenizer: normalizer {t} is not one the engine follows")

    def walk_pre(p):
        nonlocal flags
        if not p:
            return
        t = p.get("type")
        if t == "Sequence":
            for x in p.get("pretokenizers") or []:
                walk_pre(x)
        elif t == "Split":
            pat = p.get("pattern") or {}
            beh = BEHAVIOUR[str(p.get("behavior", "Isolated")).lower()]
            inv = int(bool(p.get("invert")))
            if "Regex" in pat:
                pre.append(_op(P_SPLIT_REGEX, a=beh, b=inv, s1=pat["Regex"]))
            else:
                pre.append(_op(P_SPLIT_STRING, a=beh, b=inv, s1=pat.get("String", "")))
        elif t == "ByteLevel":
            # tokenizers' order: the prefix space first, then GPT-2's regex split, per piece
            use = bool(p.get("use_regex", True))
            pre.append(_op(P_BYTELEVEL, a=int(bool(p.get("add_prefix_space", False))), b=int(use),
                           s1=GPT2_REGEX if use else ""))
            flags |= BYTE_LEVEL
        elif t == "Metaspace":
            scheme = p.get("prepend_scheme")
            if scheme is None:
                scheme = "always" if p.get("add_prefix_space", True) else "never"
            pre.append(_op(P_METASPACE, a=SCHEME[scheme], b=int(bool(p.get("split", True))),
                           s1=p.get("replacement", SPACE)))
        elif t == "Digits":
            pre.append(_op(P_DIGITS, a=int(bool(p.get("individual_digits", False)))))
        elif t == "Punctuation":
            pre.append(_op(P_PUNCT, a=BEHAVIOUR[str(p.get("behavior", "Isolated")).lower()]))
        else:
            raise SystemExit(f"BPE tokenizer: pre-tokenizer {t} is not one the engine follows")

    walk_norm(tj.get("normalizer"))
    walk_pre(tj.get("pre_tokenizer"))
    if not flags & BYTE_LEVEL and (tj.get("decoder") or {}).get("type") == "ByteLevel":
        flags |= BYTE_LEVEL                      # byte-level vocab with no ByteLevel pre-tokenizer step
    added = [a["id"] for a in tj.get("added_tokens") or []]
    vocab = model.get("vocab") or {}
    unk = vocab.get(model.get("unk_token")) if model.get("unk_token") else None
    return {"flags": flags, "norm": norm, "pre": pre, "added": added, "unk": unk}


def for_gpt2_files():
    """vocab.json + merges.txt (GPT-2 lineage): GPT-2's own regex and byte map."""
    return {"flags": BYTE_LEVEL, "norm": [], "pre": [_op(P_BYTELEVEL, b=1, s1=GPT2_REGEX)], "added": [], "unk": None}


def for_spm_bpe(add_dummy_prefix=True, byte_fallback=True, unk=0):
    """A SentencePiece BPE tokenizer.model (Llama 2 lineage) as transformers' converter writes it:
    U+2581 before the text, spaces as U+2581, the whole stretch merged."""
    norm = ([_op(N_PREPEND, s1=SPACE)] if add_dummy_prefix else []) + [_op(N_REPLACE, s1=" ", s2=SPACE)]
    return {"flags": (BYTE_FALLBACK | FUSE_UNK) if byte_fallback else FUSE_UNK, "norm": norm, "pre": [], "added": [],
            "unk": unk}


SYNTH = {}      # class names made up for a pattern's set operations: name -> ranges


def _set_ranges(items):
    """The union of \\p{..} classes in `items` (a regex fragment of only \\p{X} terms)."""
    out = []
    for nm in re.findall(r"\\p\{(\w+)\}", items):
        out += ranges(nm)
    return _merge(out)


def _merge(rs):
    rs = sorted(rs)
    out = []
    for lo, hi in rs:
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _minus(a, b):
    out = []
    for lo, hi in a:
        cur = lo
        for blo, bhi in b:
            if bhi < cur or blo > hi:
                continue
            if blo > cur:
                out.append((cur, blo - 1))
            cur = max(cur, bhi + 1)
        if cur <= hi:
            out.append((cur, hi))
    return out


def resolve_sets(pattern):
    """Kimi K2's split pattern subtracts classes: [\\p{Lu}\\p{Lt}..&&[^\\p{Han}]]. Each such
    class becomes one made-up class of plain code-point ranges (\\p{X0}, \\p{X1} ...), so
    the engine's matcher needs no set algebra."""
    def sub(m):
        rs = _minus(_set_ranges(m.group(1)), _set_ranges(m.group(2)))
        for nm, v in SYNTH.items():
            if v == rs:
                return "\\p{" + nm + "}"
        nm = f"X{len(SYNTH)}"
        SYNTH[nm] = rs
        return "\\p{" + nm + "}"
    return re.sub(r"\[((?:\\p\{\w+\})+)&&\[\^((?:\\p\{\w+\})+)\]\]", sub, pattern)


def short_names(pattern):
    """The file keeps a class name in 4 bytes: \\p{Hangul} was stored as "Hang", and the engine,
    asked for "Hangul", found no such class and refused the model (Youtu-LLM, 2026-10-20). A
    longer name becomes a made-up short one with the same code points."""
    def sub(m):
        nm = m.group(2)
        if len(nm) <= 4:
            return m.group(0)
        rs = ranges(nm)
        for k, v in SYNTH.items():
            if v == rs:
                return "\\" + m.group(1) + "{" + k + "}"
        k = f"X{len(SYNTH)}"
        SYNTH[k] = rs
        return "\\" + m.group(1) + "{" + k + "}"
    return re.sub(r"\\([pP])\{(\w+)\}", sub, pattern)


def for_tiktoken(pattern):
    """A tiktoken BPE (Qwen v1): its split pattern, then GPT-2's byte map over the pieces; a
    piece that is itself a token is taken whole, as tiktoken does (ignore_merges)."""
    return {"flags": BYTE_LEVEL | IGNORE_MERGES, "norm": [], "pre": [_op(P_SPLIT_REGEX, a=BEHAVIOUR["isolated"],
                                                                         s1=resolve_sets(pattern)),
                                              _op(P_BYTELEVEL)], "added": [], "unk": None}


# ------------------------------------------------------------------- Unicode classes

def _categories(pre):
    """The class names the regexes use: \\p{L}, \\p{Lu} ..., always 's' (White_Space)."""
    names = {"s"}
    if any(op["kind"] == P_DIGITS for op in pre):
        names.add("N")                  # Digits takes every numeric character (char::is_numeric)
    if any(op["kind"] == P_PUNCT for op in pre):
        names.add("P")                  # Punctuation: ASCII punctuation or any P* character
    for op in pre:
        if op["kind"] in (P_SPLIT_REGEX, P_BYTELEVEL):
            names |= set(re.findall(r"\\[pP]\{(\w+)\}", op["s1"]))
            if re.search(r"\\[pP]\{", op["s1"]) is None and re.search(r"\\[pP]([A-Z])", op["s1"]):
                names |= set(re.findall(r"\\[pP]([A-Z])", op["s1"]))
    return sorted(names)


WHITE_SPACE = [(0x9, 0xD), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
               (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


@functools.lru_cache(maxsize=None)
def _script(name):
    """A Unicode script's code points (\\p{Han}), read through the `regex` module."""
    import regex
    rx = regex.compile(r"\p{" + name + "}")
    out, lo = [], None
    for cp in range(0x110000):
        hit = rx.match(chr(cp)) is not None
        if hit and lo is None:
            lo = cp
        elif not hit and lo is not None:
            out.append((lo, cp - 1))
            lo = None
    if lo is not None:
        out.append((lo, 0x10FFFF))
    return out


@functools.lru_cache(maxsize=None)
def _category(name):
    out, lo = [], None
    for cp in range(0x110000):
        c = unicodedata.category(chr(cp))
        hit = c.startswith(name) if len(name) == 1 else c == name
        if hit and lo is None:
            lo = cp
        elif not hit and lo is not None:
            out.append((lo, cp - 1))
            lo = None
    if lo is not None:
        out.append((lo, 0x10FFFF))
    return out


def ranges(name):
    """Codepoint ranges of a general category ("L", "Lu", "N" ...), 's' (White_Space), a
    script (Han) or a class resolve_sets made up."""
    if name in SYNTH:
        return SYNTH[name]
    if name == "s":
        return WHITE_SPACE
    if len(name) > 2:
        return _script(name)
    return _category(name)


def pack(pret):
    def op_bytes(op):
        b1, b2 = op["s1"].encode("utf-8"), op["s2"].encode("utf-8")
        return struct.pack("<4B", op["kind"], op["a"], op["b"], op["c"]) + struct.pack("<I", len(b1)) + b1 \
            + struct.pack("<I", len(b2)) + b2
    unk = pret.get("unk")
    pre = [dict(o, s1=short_names(o["s1"])) if o["kind"] in (P_SPLIT_REGEX, P_BYTELEVEL) else o for o in pret["pre"]]
    body = struct.pack("<III", 1, pret["flags"], 0xFFFFFFFF if unk is None else int(unk))
    body += struct.pack("<I", len(pret["norm"])) + b"".join(op_bytes(o) for o in pret["norm"])
    body += struct.pack("<I", len(pre)) + b"".join(op_bytes(o) for o in pre)
    body += struct.pack("<I", len(pret["added"])) + struct.pack(f"<{len(pret['added'])}I", *pret["added"])
    cats = _categories(pre)
    body += struct.pack("<I", len(cats))
    for c in cats:
        rs = ranges(c)
        body += c.encode("ascii").ljust(4, b"\0")[:4] + struct.pack("<I", len(rs)) \
            + b"".join(struct.pack("<II", lo, hi) for lo, hi in rs)
    return MAGIC + struct.pack("<I", len(body)) + body
