#!/usr/bin/env python3
"""Pieces shared by the build and the gates: tokenizer, calibration text, ops.

Shared by the builder and the gates so neither depends on the other.
"""
import numpy as np


CALIB_TEXT = ("""The capital of France is Paris, a city known for its museums.
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
In 1969, Apollo 11 landed on the Moon. Neil Armstrong stepped onto the surface.
Water boils at 100 degrees Celsius at standard atmospheric pressure.
The mitochondrion is the powerhouse of the cell, producing ATP through respiration.
SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1;
Dear Sir or Madam, I am writing to enquire about the position advertised last week.
The derivative of x squared with respect to x is two x.
Once upon a time, in a village at the edge of a great forest, there lived a miller.
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
The Treaty of Westphalia in 1648 ended the Thirty Years War in Europe.
Photosynthesis converts carbon dioxide and water into glucose using sunlight.
git rebase --interactive HEAD~5 lets you squash commits before pushing.
The integral of one over x with respect to x is the natural logarithm of x.
She had not expected the letter, nor the news that it carried about her brother.
A binary search halves the interval each step, so it runs in logarithmic time.
The Pacific Ocean covers roughly one third of the surface of the Earth.
import numpy as np
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
Inflation measures the rate at which prices for goods and services rise.
The quick brown fox jumps over the lazy dog near the riverbank at dawn.
Hydrogen is the lightest and most abundant element in the observable universe.
He argued that the premise was false, and therefore the conclusion did not follow.
SELECT u.id, o.total FROM users u JOIN orders o ON o.user_id = u.id WHERE o.total > 100;
Shakespeare wrote thirty seven plays, among them Hamlet, Macbeth and King Lear.
The speed of light in a vacuum is approximately 299,792,458 metres per second.
To reverse a linked list, walk it while rewriting each next pointer in turn.
Antibiotics act on bacteria and have no effect whatsoever on viral infections.
The company reported quarterly revenue of 4.2 billion dollars, up nine percent.
for i, row in enumerate(matrix):
    for j, value in enumerate(row):
        result[j][i] = value
Plate tectonics explains earthquakes, mountain ranges and the drift of continents.
If the temperature falls below zero, the water in the pipes may freeze and burst.
""" * 8)


class Bpe:
    """Byte-level BPE encoder, enough for calibration text.

    Ported to C later for the engine; kept here so calibration does not depend
    on a binary that does not exist yet.
    """

    def __init__(self, tokens, merges):
        self.rank = {tuple(m.split(" ")): i for i, m in enumerate(merges)} if merges else {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        # Byte-level BPE gives all 256 byte values a printable symbol, and WHICH
        # bytes keep their own character is fixed by the vocabularies we have to
        # read: change the set and the merge table stops matching. The set is
        # exactly "printable and not whitespace" in Latin-1 -- so ask Python that
        # question rather than transcribing the three ranges everyone quotes.
        # The bytes that fail it are parked in the private range above 255, in
        # byte order, which is what makes the map a bijection.
        keep = [b for b in range(256) if chr(b).isprintable() and not chr(b).isspace()]
        spare = iter(range(256, 256 + (256 - len(keep))))
        kept = set(keep)
        self.b2u = {b: chr(b if b in kept else next(spare)) for b in range(256)}
        assert len(set(self.b2u.values())) == 256, "byte map must be a bijection"

    def _merge_word(self, word):
        syms = list(word)
        while len(syms) > 1:
            best, bi = None, None
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
        return syms

    def encode(self, text):
        out = []
        for chunk in text.replace("\n", " \n").split(" "):
            if not chunk:
                continue
            w = "".join(self.b2u[b] for b in chunk.encode("utf-8"))
            for sym in self._merge_word(w):
                i = self.ids.get(sym)
                if i is not None:
                    out.append(i)
        return out


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def layer_norm(x, w, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    v = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + (0.0 if b is None else b)


def norm(x, w, b, eps, norm_type):
    """The calibration walk must compute the SAME architecture the engine does.
    It is not a sanity check -- it produces the activations GPTQ fits its error
    feedback against, so a wrong forward pass here degrades the built model
    silently, with nothing to catch it downstream."""
    return layer_norm(x, w, b, eps) if norm_type == 1 else rms_norm(x, w, eps)


def yarn_attention_factor(scaling):
    """The factor YaRN or LongRoPE multiplies cos and sin by (1 for the rest)."""
    kind = str((scaling or {}).get("rope_type", (scaling or {}).get("type", ""))).lower()
    if kind == "longrope":
        if scaling.get("attention_factor") is not None:
            return float(scaling["attention_factor"])
        orig = float(scaling.get("original_max_position_embeddings") or scaling.get("_orig_max") or 0)
        factor = float(scaling.get("factor") or (float(scaling.get("_max_pos") or 0) / orig if orig else 1.0))
        return 1.0 if factor <= 1 or not orig else float(np.sqrt(1 + np.log(factor) / np.log(orig)))
    if kind != "yarn":
        return 1.0
    if scaling.get("attention_factor") is not None:
        return float(scaling["attention_factor"])
    factor = float(scaling.get("factor", 1.0))

    def mscale(s, m=1.0):
        return 1.0 if s <= 1 else 0.1 * m * np.log(s) + 1.0
    if scaling.get("mscale") and scaling.get("mscale_all_dim"):
        return float(mscale(factor, scaling["mscale"]) / mscale(factor, scaling["mscale_all_dim"]))
    return float(mscale(factor))


def rope_freqs(theta, half, scaling=None):
    """The inverse frequencies RoPE rotates by, including any context-extension
    scaling (Llama 3.x, linear, dynamic).

    These go into the FILE as a tensor, not into the engine as another
    per-variant branch. The reference engine implements each scaling type in
    C++; we compute it once at build time and store the answer, so a scaling
    scheme it has never heard of costs the engine nothing and cannot drift
    between builder and runtime. It is the same reason the ladder stores bit
    slices instead of a dequantization recipe."""
    inv = theta ** (-np.arange(0, half, dtype=np.float64) / half)
    if not scaling:
        return inv.astype(np.float32)
    kind = str(scaling.get("rope_type", scaling.get("type", "default"))).lower()
    factor = float(scaling.get("factor", 1.0))
    if kind in ("default", "none"):
        return inv.astype(np.float32)
    if kind == "dynamic" and scaling.get("alpha"):
        # HunYuan's NTK-alpha: the base itself is raised, from the start
        dim = 2 * half
        base = theta * float(scaling["alpha"]) ** (dim / (dim - 2))
        return (base ** (-np.arange(0, half, dtype=np.float64) / half)).astype(np.float32)
    if kind == "dynamic":
        # Dynamic NTK changes nothing inside the trained length (transformers
        # recomputes the base only past max_position_embeddings). This used to
        # divide by the factor -- a different rotation at every position.
        return inv.astype(np.float32)
    if kind == "linear":
        return (inv / factor).astype(np.float32)
    if kind == "longrope":
        # Phi-3/Phi-4 (transformers' _compute_longrope_parameters): per-dimension
        # rescale factors. The short ones hold up to the original context, which
        # is where the builder caps such a model (sml_hf); the long ones would
        # need the whole cache re-rotated as the context passes it.
        sf = np.asarray(scaling["short_factor"], dtype=np.float64)
        if sf.size != half:
            raise SystemExit(f"longrope short_factor has {sf.size} entries for {half} rotary pairs")
        return (inv / sf).astype(np.float32)
    if kind == "yarn":
        # transformers' _compute_yarn_parameters: interpolate the low
        # frequencies by the factor, keep the high ones, ramp between the
        # dimensions that turn beta_fast and beta_slow times over the original
        # context. Its attention factor scales cos and sin; arch_flags folds
        # that into the attention scale (yarn_attention_factor below).
        dim = 2 * half
        old_ctx = float(scaling.get("original_max_position_embeddings") or scaling.get("_max_pos") or 0)
        if not old_ctx:
            raise SystemExit("yarn rope scaling without original_max_position_embeddings")
        fast, slow = float(scaling.get("beta_fast") or 32), float(scaling.get("beta_slow") or 1)

        def corr(rot):
            return dim * np.log(old_ctx / (rot * 2 * np.pi)) / (2 * np.log(theta))
        lo, hi = corr(fast), corr(slow)
        if scaling.get("truncate", True):
            lo, hi = np.floor(lo), np.ceil(hi)
        lo, hi = max(lo, 0), min(hi, dim - 1)
        if lo == hi:
            hi += 0.001
        ramp = np.clip((np.arange(half, dtype=np.float64) - lo) / (hi - lo), 0, 1)
        extra = 1 - ramp
        return (inv / factor * (1 - extra) + inv * extra).astype(np.float32)
    if kind == "llama3":
        low = float(scaling["low_freq_factor"])
        high = float(scaling["high_freq_factor"])
        old_ctx = float(scaling["original_max_position_embeddings"])
        low_wl, high_wl = old_ctx / low, old_ctx / high
        wl = 2.0 * np.pi / inv
        out = np.where(wl > low_wl, inv / factor, inv)
        smooth = (old_ctx / wl - low) / (high - low)
        smoothed = (1 - smooth) * out / factor + smooth * out
        medium = ~(wl < high_wl) & ~(wl > low_wl)
        return np.where(medium, smoothed, out).astype(np.float32)
    raise SystemExit(f"rope scaling '{kind}' is not implemented; it would "
                     f"silently rotate at the wrong frequencies")


def rope(x, pos, theta, n_rot=0, interleaved=False, freqs=None):
    """x: (T, H, D). Rotate by absolute position.

    n_rot rotates only the leading dimensions of each head (GPT-NeoX, Phi,
    StableLM) and leaves the tail untouched. interleaved pairs ADJACENT
    elements (GPT-J) rather than across the halves."""
    T, H, D = x.shape
    rot = n_rot if n_rot else D
    half = rot // 2
    inv = rope_freqs(theta, half) if freqs is None else np.asarray(freqs)[:half]
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    r, tail = x[..., :rot], x[..., rot:]
    if interleaved:
        a, b = r[..., 0::2], r[..., 1::2]
        out = np.empty_like(r)
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
    else:
        a, b = r[..., :half], r[..., half:]
        out = np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)
    return np.concatenate([out, tail], axis=-1) if tail.shape[-1] else out

def bias(W, name):
    """QKV biases exist in Qwen2 and not in Llama/Mistral. Reading them
    unconditionally is what stopped the first non-Qwen build."""
    try:
        return W.get(name)
    except KeyError:
        return 0.0


def opt(W, name):
    """Optional tensor, or None -- for norm biases, which LayerNorm
    architectures carry and RMSNorm ones do not."""
    try:
        return W.get(name)
    except KeyError:
        return None
