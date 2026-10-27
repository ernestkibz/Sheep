"""Unigram (SentencePiece unigram) tokenizers for the engine (2026-10-14).

Helium, XGLM and Persimmon ship a Unigram model: no merge order, a score per piece,
and the text is split by the best-scoring path through every way of cutting it
(Viterbi). Refused until now ("only BPE ones have a merge order"). This module reads
both sources -- tokenizer.json's Unigram model and a SentencePiece tokenizer.model of
type 1 -- into one description, packs it as a trailer after the vocab's merges (see
sml/sml-vocab.h), and encodes in Python exactly as sml-vocab.c does, so calibration
text and the tokenizer check use the same cut.

Normalization is part of the tokenizer: nmt_nfkc's precompiled charsmap (XGLM) is a
darts-clone trie of byte strings -> replacement strings; it is unpacked here into
plain (source, target) pairs the engine matches longest first.
"""
import base64
import struct

import numpy as np

SPACE = "▁"
MAGIC = b"UNIG"

# flags, mirrored in sml-vocab.h
PREFIX_ALWAYS = 1       # put a U+2581 before the text (SentencePiece's dummy prefix, Helium's Prepend)
PREFIX_IF_MISSING = 2   # ... unless it already starts with one (tokenizers' Metaspace)
BYTE_FALLBACK = 4       # an unknown character is spelled as its <0xNN> bytes
SPLIT_WORDS = 8         # cut before every U+2581 and run each word alone (Metaspace)
SPLIT_NEWLINES = 16     # a run of newlines is its own piece (Helium's Split("\n", Contiguous))
COLLAPSE_SPACES = 32    # two or more spaces become one
STRIP_SPACES = 64       # SentencePiece remove_extra_whitespaces: collapse, and strip both ends
SPM_MATH = 128          # SentencePiece's own arithmetic: path scores kept as float32, a piece's
                        # candidate added in double and compared against that rounded float
                        # (its ties fall by rounding: "2026" -> 202|6, "8 2026" -> 8|20|26)

# piece kinds: in the lattice, in it with a bonus, never chosen, and an added token -- cut out
# whole BEFORE normalization, as tokenizers does (Helium's newline is one: each line is
# then normalized alone, so every line gets its own U+2581 prefix)
NORMAL, USER, SKIP, ADDED = 0, 1, 2, 3
UNK_PENALTY = 10.0                # both implementations score an unknown character min - 10


# ------------------------------------------------------------ precompiled charsmap

def charsmap_pairs(blob):
    """[(source bytes, target bytes)] from a SentencePiece precompiled charsmap."""
    if not blob:
        return []
    n = struct.unpack_from("<I", blob, 0)[0]
    units = np.frombuffer(blob, dtype="<u4", count=n // 4, offset=4).tolist()
    pool = blob[4 + n:]

    def has_leaf(u): return (u >> 8) & 1
    def value(u): return u & 0x7FFFFFFF
    def label(u): return u & (0x80000000 | 0xFF)
    def offset(u): return (u >> 10) << ((u & (1 << 9)) >> 6)

    out = []
    stack = [(offset(units[0]), b"")]
    while stack:
        pos, key = stack.pop()
        for c in range(1, 256):
            child = pos ^ c
            if child >= len(units) or label(units[child]) != c:
                continue
            nxt = child ^ offset(units[child])
            k2 = key + bytes([c])
            if has_leaf(units[child]):
                v = value(units[nxt])
                end = pool.index(b"\0", v)
                out.append((k2, bytes(pool[v:end])))
            stack.append((nxt, k2))
    out.sort()
    return out


# ------------------------------------------------------------------- the two readers

def from_tokenizer_json(tj):
    """(tokens, info) from a tokenizer.json whose model is Unigram."""
    model = tj["model"]
    vocab = model["vocab"]
    n = len(vocab)
    for a in tj.get("added_tokens", []):
        n = max(n, a["id"] + 1)
    tokens = [""] * n
    scores = np.full(n, 0.0, dtype=np.float32)
    kinds = np.full(n, SKIP, dtype=np.uint8)
    for i, (piece, score) in enumerate(vocab):
        tokens[i], scores[i], kinds[i] = piece, score, NORMAL
    for a in tj.get("added_tokens", []):
        tokens[a["id"]] = a["content"]
        kinds[a["id"]] = ADDED
    flags, norm = 0, []
    if model.get("byte_fallback"):
        flags |= BYTE_FALLBACK

    def walk_norm(nz):
        nonlocal flags, norm
        if not nz:
            return
        t = nz.get("type")
        if t == "Sequence":
            for x in nz["normalizers"]:
                walk_norm(x)
        elif t == "Precompiled":
            norm = charsmap_pairs(base64.b64decode(nz.get("precompiled_charsmap") or ""))
        elif t == "Prepend" and nz.get("prepend") in (" ", SPACE):
            flags |= PREFIX_ALWAYS
        elif t == "Replace":
            pat = nz.get("pattern", {})
            if pat.get("String") == " " and nz.get("content") == SPACE:
                pass                                  # spaces become U+2581: the engine always does this
            elif pat.get("Regex") == " {2,}" and nz.get("content") == " ":
                flags |= COLLAPSE_SPACES
            else:
                raise SystemExit(f"Unigram tokenizer: normalizer Replace {pat} is not one the engine follows")
        elif t in ("NFKC",) and not norm:
            raise SystemExit("Unigram tokenizer: a plain NFKC normalizer (no charsmap) is not followed yet")
        else:
            raise SystemExit(f"Unigram tokenizer: normalizer {t} is not one the engine follows")

    def walk_pre(pt):
        nonlocal flags
        if not pt:
            return
        t = pt.get("type")
        if t == "Sequence":
            for x in pt["pretokenizers"]:
                walk_pre(x)
        elif t == "Metaspace":
            if pt.get("prepend_scheme", "always" if pt.get("add_prefix_space", True) else "never") != "never":
                flags |= PREFIX_IF_MISSING
            if pt.get("split", True):
                flags |= SPLIT_WORDS
        elif t == "Split" and pt.get("pattern", {}).get("String") == "\n" and pt.get("behavior") == "Contiguous":
            flags |= SPLIT_NEWLINES
        else:
            raise SystemExit(f"Unigram tokenizer: pre-tokenizer {t} is not one the engine follows")

    walk_norm(tj.get("normalizer"))
    walk_pre(tj.get("pre_tokenizer"))
    live = scores[kinds == NORMAL]
    info = {"unk": int(model.get("unk_id") or 0), "flags": flags, "scores": scores, "kinds": kinds,
            "unk_score": float(live.min()) - UNK_PENALTY, "max_score": float(live.max()), "norm": norm}
    return tokens, info


def from_spm(pieces, norm_spec, trainer):
    """info from a SentencePiece model of type 1. pieces: [(piece, score, type)] with the
    proto's types (1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte)."""
    n = len(pieces)
    scores = np.asarray([p[1] for p in pieces], dtype=np.float32)
    kinds = np.asarray([NORMAL if p[2] == 1 else USER if p[2] == 4 else SKIP for p in pieces], dtype=np.uint8)
    flags = 0
    if norm_spec.get("add_dummy_prefix", True):
        flags |= PREFIX_ALWAYS
    if norm_spec.get("remove_extra_whitespaces", True):
        flags |= STRIP_SPACES
    if trainer.get("byte_fallback"):
        flags |= BYTE_FALLBACK
    norm = charsmap_pairs(norm_spec.get("precompiled_charsmap") or b"")
    flags |= SPM_MATH
    live = scores[kinds == NORMAL]
    unk = next((i for i, p in enumerate(pieces) if p[2] == 2), 0)
    # SentencePiece starts its max at FLT_MIN (the smallest POSITIVE float), so with every
    # score negative a user-defined piece scores length * 1e-38 - 0.1: about -0.1 each
    return {"unk": unk, "flags": flags, "scores": scores, "kinds": kinds,
            "unk_score": float(live.min()) - UNK_PENALTY, "max_score": max(float(live.max()), 1.1754944e-38),
            "norm": norm, "n": n}


def pack(info, n_tokens):
    """The trailer the engine reads after the merges."""
    scores = np.zeros(n_tokens, dtype="<f4")
    kinds = np.full(n_tokens, SKIP, dtype=np.uint8)
    m = min(n_tokens, len(info["scores"]))
    scores[:m] = info["scores"][:m]
    kinds[:m] = info["kinds"][:m]
    recs = bytearray()
    for s, t in info["norm"]:
        if len(s) > 255 or len(t) > 255:
            continue
        recs += bytes([len(s), len(t)]) + s + t
    kb = kinds.tobytes()
    kb += b"\0" * (-len(kb) % 4)
    return (MAGIC + struct.pack("<IIIffII", 1, int(info["unk"]), int(info["flags"]), float(info["unk_score"]),
                                float(info["max_score"]), len(info["norm"]), len(recs))
            + scores.tobytes() + kb + bytes(recs))


# --------------------------------------------------------------------- the encoder

class Unigram:
    """Encodes as sml-vocab.c does. specials: token strings matched whole, longest first."""

    def __init__(self, tokens, info, specials=()):
        self.flags = info["flags"]
        self.unk, self.unk_score, self.max_score = info["unk"], info["unk_score"], info["max_score"]
        self.ids = {}
        for i, t in enumerate(tokens):
            if t and info["kinds"][i] != SKIP:
                self.ids.setdefault(t.encode("utf-8"), i)
        self.score = info["scores"]
        self.kind = info["kinds"]
        self.maxlen = max((len(k) for k in self.ids), default=1)
        self.norm = {s: t for s, t in info["norm"]}
        self.normmax = max((len(s) for s in self.norm), default=0)
        self.byte = {}
        for i, t in enumerate(tokens):
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                self.byte[int(t[3:5], 16)] = i
        full = {t: i for i, t in enumerate(tokens) if t}
        specials = list(specials) + [t for i, t in enumerate(tokens) if t and info["kinds"][i] == ADDED]
        self.special_ids = {s: full[s] for s in specials if s in full}
        self.specials = sorted(self.special_ids, key=len, reverse=True)

    def _normalize(self, b):
        out, i = bytearray(), 0
        while i < len(b):
            for L in range(min(self.normmax, len(b) - i), 0, -1):
                t = self.norm.get(b[i:i + L])
                if t is not None:
                    out += t
                    i += L
                    break
            else:
                out.append(b[i])
                i += 1
        s = out.decode("utf-8", errors="replace")
        f = self.flags
        if f & STRIP_SPACES:
            s = " ".join(x for x in s.split(" ") if x)
        elif f & COLLAPSE_SPACES:
            while "  " in s:
                s = s.replace("  ", " ")
        s = s.replace(" ", SPACE)
        if f & PREFIX_ALWAYS or (f & PREFIX_IF_MISSING and not s.startswith(SPACE)):
            s = SPACE + s
        return s

    def _pieces(self, s):
        if self.flags & SPLIT_NEWLINES:
            parts, cur = [], ""
            for ch in s:
                if (ch == "\n") != (cur[-1:] == "\n") and cur:
                    parts.append(cur)
                    cur = ""
                cur += ch
            if cur:
                parts.append(cur)
        else:
            parts = [s]
        if self.flags & SPLIT_WORDS:
            words = []
            for p in parts:
                cur = ""
                for ch in p:
                    if ch == SPACE and cur:
                        words.append(cur)
                        cur = ""
                    cur += ch
                if cur:
                    words.append(cur)
            parts = words
        return parts

    def _viterbi(self, s, out):
        b = s.encode("utf-8")
        n = len(b)
        starts = [i for i in range(n) if (b[i] & 0xC0) != 0x80] + [n]
        nxt = {starts[k]: starts[k + 1] for k in range(len(starts) - 1)}
        best = {0: (0.0, None, None)}           # position -> (score, from, id)
        spm = bool(self.flags & SPM_MATH)
        f32 = (lambda x: float(np.float32(x))) if spm else (lambda x: x)
        for i in starts[:-1]:
            if i not in best:
                continue
            base = best[i][0]
            single = False
            j, chars = i, 0
            while j < n and j - i < self.maxlen:
                j = nxt[j]
                chars += 1
                if j - i > self.maxlen:
                    break
                tid = self.ids.get(b[i:j])
                if tid is None:
                    continue
                sc = (chars * self.max_score - 0.1) if self.kind[tid] == USER else float(self.score[tid])
                if chars == 1:
                    single = True
                if j not in best or base + sc > best[j][0]:
                    best[j] = (f32(base + sc), i, tid)
            if not single:
                j = nxt[i]
                cand = f32(f32(self.unk_score) + base) if spm else base + self.unk_score
                if j not in best or cand > best[j][0]:
                    best[j] = (cand, i, -1)
        path, j = [], n
        while j:
            _, i, tid = best[j]
            path.append((i, j, tid))
            j = i
        path.reverse()
        k = 0
        while k < len(path):
            i, j, tid = path[k]
            if tid >= 0:
                out.append(tid)
                k += 1
                continue
            while k + 1 < len(path) and path[k + 1][2] < 0:     # a run of unknowns is one
                k += 1
                j = path[k][1]
            raw = b[i:j]
            if self.flags & BYTE_FALLBACK and all(c in self.byte for c in raw):
                out.extend(self.byte[c] for c in raw)
            else:
                out.append(self.unk)
            k += 1

    def encode(self, text):
        out, seg, i = [], "", 0
        while i < len(text):
            sp = next((s for s in self.specials if text.startswith(s, i)), None)
            if sp is None:
                seg += text[i]
                i += 1
                continue
            if seg:
                self._segment(seg, out)
                seg = ""
            out.append(self.special_ids[sp])
            i += len(sp)
        if seg:
            self._segment(seg, out)
        return out

    def _segment(self, seg, out):
        for p in self._pieces(self._normalize(seg.encode("utf-8"))):
            if p:
                self._viterbi(p, out)
