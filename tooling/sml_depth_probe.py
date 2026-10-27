"""Per-tensor read depth: what SSF can do that a fixed-precision format cannot.

A conventional quantized file commits to one precision when it is written. The
ladder stores bit slices, so depth is a per-tensor decision made at RUN time
from a single file. We have only ever exercised one global depth, which leaves
that capability unused.

Mixed depth per ROW was measured and failed: most rows ended up at depth 1,
where error is 27-51%, and those rows dominated the output regardless of their
true contribution. Per TENSOR is a different bet -- a handful of coarse choices
tuned offline against measured output error, with no row ever landing in that
band. This measures whether it pays.

Graded on end-to-end logits against the full-precision checkpoint, never on
per-tensor weight error, because weight error is what made mixed-depth look
promising in the first place.
"""
import argparse, itertools, json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sml_logits_check import SsfWeights, reference_logits   # noqa: E402
from sml_read import SmlFile                                # noqa: E402

# Tensor classes that can carry different depths. Grouped by role because that
# is the granularity a schedule can actually exploit.
CLASSES = {
    "embd":  lambda n: n == "token_embd.weight",
    "head":  lambda n: n == "output.weight",
    "attn":  lambda n: ".attn_q." in n or ".attn_k." in n or ".attn_v." in n,
    "attno": lambda n: ".attn_output." in n,
    "ffnu":  lambda n: ".ffn_up." in n or ".ffn_gate." in n,
    "ffnd":  lambda n: ".ffn_down." in n,
}


def class_of(name):
    for c, f in CLASSES.items():
        if f(name):
            return c
    return None


class MixedWeights(SsfWeights):
    """SsfWeights with a per-class depth map instead of one global depth."""

    def __init__(self, path, depths):
        super().__init__(path, max(depths.values()))
        self.depths = depths

    def get(self, name):
        m = self.f.tensors[name]
        if m["kind"] == "f32":
            return (self.f.f32(name).reshape(-1) if m["rows"] == 1
                    else self.f.f32(name))
        d = self.depths.get(class_of(name), self.depth)
        perm = self.f.perm(name)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return self.f.dequant(name, depth=d)[:, inv]


def bytes_per_token(path, depths):
    """Weight bytes one forward pass moves under this schedule."""
    f = SmlFile(path)
    total = 0
    for n, m in f.tensors.items():
        if m["kind"] != "ladder":
            total += m["nbytes"]
            continue
        if n == "token_embd.weight":       # one row, not streamed
            continue
        total += m["sd_n"] + m["sq_n"]
        d = depths.get(class_of(n), max(depths.values()))
        for lv in m["lv"][:d]:
            total += lv["codes_n"]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ssf")
    ap.add_argument("ref", help="HF checkpoint, the full-precision truth")
    ap.add_argument("--tokens", default="")
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    ids = ([int(x) for x in args.tokens.split(",")] if args.tokens
           else [785, 6722, 315, 9625, 374])
    truth = reference_logits(args.ref, ids)

    def score(depths):
        got = reference_logits.__wrapped__(args.ssf, ids, depths) \
            if hasattr(reference_logits, "__wrapped__") else None
        return got

    # score() needs reference_logits to accept our weights object; do it directly
    import sml_logits_check as LC

    def run(depths):
        W = MixedWeights(args.ssf, depths)
        orig = LC.SsfWeights
        try:
            LC.SsfWeights = lambda p, d: W
            got = LC.reference_logits(args.ssf, ids, depth=max(depths.values()))
        finally:
            LC.SsfWeights = orig
        n = min(len(got), len(truth))
        return float(np.corrcoef(got[:n], truth[:n])[0, 1]), int(got.argmax())

    base_top1 = int(truth.argmax())
    print(f"reference top1 = {base_top1}\n")

    uniform = {}
    for d in range(1, args.max_depth + 1):
        dep = {c: d for c in CLASSES}
        c, t1 = run(dep)
        b = bytes_per_token(args.ssf, dep)
        uniform[d] = (c, b)
        print(f"uniform depth {d}: corr {c:.5f}  {b/1e9:.4f} GB/token  "
              f"top1 {'OK' if t1 == base_top1 else 'MISS'}")

    target_c, target_b = uniform[2]
    print(f"\nbeating uniform depth 2 means corr >= {target_c:.5f} "
          f"AND < {target_b/1e9:.4f} GB/token\n")

    # Search: each class at depth 1..3. 3^6 = 729, but only score the ones that
    # are actually cheaper than the target, which is most of the cost.
    best = []
    for combo in itertools.product(range(1, args.max_depth + 1), repeat=len(CLASSES)):
        dep = dict(zip(CLASSES, combo))
        b = bytes_per_token(args.ssf, dep)
        if b >= target_b:
            continue
        best.append((b, tuple(sorted(dep.items())), dep))
    best.sort(key=lambda t: t[0])
    print(f"{len(best)} schedules are cheaper than uniform depth 2; "
          f"scoring the cheapest 40")
    hits = []
    for b, _k, dep in best[:40]:
        c, t1 = run(dep)
        ok = c >= target_c and t1 == base_top1
        if ok:
            hits.append((b, c, tuple(sorted(dep.items())), dep))
        print(f"  {b/1e9:.4f} GB  corr {c:.5f}  top1 {'OK ' if t1==base_top1 else 'MISS'}"
              f"  {'<== BEATS depth 2' if ok else ''}  {dep}")
    print()
    if hits:
        b, c, _k, dep = min(hits, key=lambda t: t[0])
        print(f"BEST: {b/1e9:.4f} GB/token at corr {c:.5f} "
              f"({target_b/b:.3f}x fewer bytes than uniform depth 2)")
        print(f"      {json.dumps(dep)}")
    else:
        print("No cheaper schedule matched uniform depth 2 on quality.")


if __name__ == "__main__":
    main()
