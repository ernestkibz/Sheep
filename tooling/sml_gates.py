#!/usr/bin/env python3
"""Run every SSF gate and print one verdict.

The gates exist because this project keeps producing numbers that grade the
mechanism instead of the outcome. Thirteen separate measurements faked a result
while this was being built; each looked fine in isolation. So correctness is
checked before speed, and a configuration that fails correctness has its
throughput reported as void rather than as a win.

    python tooling/sml_gates.py MODEL.ssf --ref HF_CHECKPOINT_DIR

Two things are deliberately separated here, because conflating them is what
made the previous version useless:

  * WIRING -- does the C forward pass compute the architecture correctly?
    Graded by running the numpy reference on the SAME quantized weights, so
    quantization error cancels and only a RoPE convention, GQA repeat,
    residual placement or tied-head mistake can move the number. This is
    architecture- and size-independent, and it is what decides pass/fail.

  * QUALITY -- how much did quantization cost against the fp16 checkpoint?
    Reported, never gated. It legitimately depends on model size: a 410m model
    lands near 0.976 and the reference engine's own Q4_K_M lands at 0.9766 on
    the same model. The old gate held every model to the 3B's 0.99 and so
    declared a correct small model broken.

Everything else is read FROM THE MODEL too. An earlier version hardcoded the
Qwen tensor names, so it asked a GPT-NeoX model for an ffn_gate it does not
have and reported FAIL. Same mistake in both places: a gate that assumes what
it is supposed to be checking.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SML = os.path.join(_HERE, "..", "sml")
sys.path.insert(0, _HERE)

from sml_read import SmlFile          # noqa: E402

LAYOUT_LADDER, LAYOUT_FLAT = 0, 1
TOKENS = "9707,11,358,1079"

# A wiring bug destroys the structure of the distribution: the ones this
# project actually hit read 0.379, 0.873 and -0.04. Quantization noise with the
# SAME weights on both sides costs far less than 1%. 0.99 sits in the wide gap
# between those two regimes and is not tuned to any model.
WIRING_MIN = 0.99


def run(cmd):
    # the engine's output is UTF-8 (generated text included): read as the Windows
    # code page, one byte it could not map threw away the whole gate (2026-10-06)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def pick_tensors(f):
    """Two quantized weights of DIFFERENT shapes, whatever this arch calls them.

    Shape matters: the scale-superblock count is a truncating division, and a
    bug there only shows on a row count that is not a whole multiple of the
    superblock size. Picking by NAME instead is what made this gate demand an
    ffn_gate from an architecture that has none.
    """
    cand = [(n, m) for n, m in f.tensors.items()
            if m.get("kind") == "ladder" and m.get("lv")]
    if not cand:
        return []
    by_shape = {}
    for n, m in cand:
        by_shape.setdefault((m["rows"], m["cols"]), n)
    names = sorted(by_shape.values())
    return [names[0], names[-1]] if len(names) > 1 else names[:1]


def corr_of(out):
    m = re.search(r"correlation\s+([0-9.\-]+)", out)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ssf")
    ap.add_argument("--ref", help="HF checkpoint dir the model was built from. "
                                  "Without it the correctness gates are "
                                  "SKIPPED, not passed.")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--baseline", help="TOKS_PER_S[,GB_PER_TOKEN] measured for "
                                       "THIS model on THIS machine")
    args = ap.parse_args()

    base_tps = base_gb = None
    if args.baseline:
        parts = args.baseline.split(",")
        base_tps = float(parts[0])
        if len(parts) > 1:
            base_gb = float(parts[1])

    ext = ".exe" if os.name == "nt" else ""
    cli = os.path.join(_SML, "sml-cli" + ext)
    chk = os.path.join(_SML, "sml-check" + ext)
    lgc = os.path.join(_HERE, "sml_logits_check.py")
    tmp = tempfile.mkdtemp(prefix="smlgate")

    f = SmlFile(args.ssf)
    quant = [m for m in f.tensors.values()
             if m.get("kind") == "ladder" and m.get("lv")]
    layouts = {m.get("layout", LAYOUT_LADDER) for m in quant}
    has_ladder = LAYOUT_LADDER in layouts
    widths = sorted({m.get("nbits") or m["lv"][0]["bits"] for m in quant})

    lname = "+".join(n for n, b in (("ladder", has_ladder),
                                    ("flat", LAYOUT_FLAT in layouts)) if b)
    print("=== SSF gates: %s ===\n" % os.path.basename(args.ssf))
    print("    format v%d, %d quantized tensors, group %d"
          % (f.version, len(quant), f.group))
    print("    layout %s   widths %s\n" % (lname, widths))

    # --- [1] kernel vs format, on shapes this architecture actually has
    print("[1] kernel vs format spec")
    names = pick_tensors(f)
    ok_kernel = bool(names)
    if not names:
        print("    no quantized tensors found -- nothing to cross-check")
    for tname in names:
        out_bin = os.path.join(tmp, "g.bin")
        run([chk, args.ssf, tname, out_bin, "128"])
        out = run([sys.executable, os.path.join(_HERE, "sml_check.py"),
                   args.ssf, tname, out_bin])
        good = "PASS" in out
        ok_kernel &= good
        print("    %-30s %s" % (tname, "PASS" if good else "FAIL"))

    # Depth only means something when a tensor is stored level-major. On a flat
    # model every depth reads the same bytes, so iterating them would print the
    # same row three times and imply a dial that is not there.
    depths = (1, 2, 3) if has_ladder else (0,)

    # --- [2] wiring, and [3] quality: measured together, reported apart
    print("\n[2] forward-pass wiring   (numpy vs C on the SAME quantized weights)")
    if not args.ref:
        print("    SKIPPED - no --ref checkpoint given. Unknown is not a pass.")
    hdr = "    %5s" % "depth" if has_ladder else "    "
    if args.ref:
        print(hdr + " %10s %s" % ("corr", "verdict"))

    rows = []
    for d in depths:
        dep = ["--depth", str(d)] if d else []
        wire = qual = float("nan")
        if args.ref:
            lg = os.path.join(tmp, "lg%d.bin" % d)
            run([cli, args.ssf, "--tokens", TOKENS] + dep +
                ["--threads", str(args.threads), "--logits", lg])
            wire = corr_of(run([sys.executable, lgc, args.ref, lg,
                                "--tokens", TOKENS, "--ssf", args.ssf,
                                "--ssf-depth", str(d)]))
            qual = corr_of(run([sys.executable, lgc, args.ref, lg,
                                "--tokens", TOKENS]))
            lead = "    %5d" % d if has_ladder else "    "
            good = wire > WIRING_MIN
            print("%s %10.5f %s" % (lead, wire,
                                    "PASS" if good else "FAIL - wiring bug"))

        out = run([cli, args.ssf, "--tokens", TOKENS] + dep +
                  ["--gen", str(args.gen), "--threads", str(args.threads)])
        m = re.search(r"generate\s+\d+\s+tokens? in [0-9.]+s\s+([0-9.]+) tok/s",
                      out)
        toks = float(m.group(1)) if m else float("nan")
        m = re.search(r"([0-9.]+) GB/token moved", out)
        gb = float(m.group(1)) if m else float("nan")
        rows.append((d, gb, wire, qual, toks, bool(args.ref) and wire > WIRING_MIN))

    ok_wiring = bool(args.ref) and all(r[5] for r in rows)

    print("\n[3] quality vs fp16   SMOKE ONLY -- one scored position")
    print(hdr + " %10s" % "corr")
    for d, gb, wire, qual, toks, _ in rows:
        lead = "    %5d" % d if has_ladder else "    "
        cs = "%.5f" % qual if qual == qual else "unreadable"
        print("%s %10s" % (lead, cs))
    # This grades the LAST position only, on four tokens of context. A quality
    # claim measured that way was already published and withdrawn once: it read
    # 6.5% at four tokens, 2.2% at 128, and reversed at 512. So this is a smoke
    # check that the model is not broken, and nothing more.
    print("    NOT a quality measurement: one position, 4 tokens of context.")
    print("    A real number needs hundreds of held-out positions, scored")
    print("    against the reference engine ON THIS MODEL. See RESULTS.md.")

    print("\n[4] throughput   (void unless wiring passed)")
    print(hdr + " %8s %8s %13s" % ("GB/tok", "tok/s", "vs baseline"))
    for d, gb, wire, qual, toks, usable in rows:
        if base_tps and usable:
            ratio = "%.2fx" % (toks / base_tps)
        elif not args.ref:
            ratio = "UNGRADED"
        elif not usable:
            ratio = "VOID"
        else:
            ratio = "no baseline"
        lead = "    %5d" % d if has_ladder else "    "
        print("%s %8.3f %8.2f %13s" % (lead, gb, toks, ratio))

    print("\n=== VERDICT ===")
    if not ok_kernel:
        print("FAIL - kernel and format disagree; nothing else is meaningful")
        sys.exit(1)
    if not args.ref:
        print("INCOMPLETE - kernel and format agree, but the forward pass was")
        print("             never graded. Pass --ref HF_CHECKPOINT_DIR.")
        sys.exit(2)
    if not ok_wiring:
        print("FAIL - the C forward pass does not match the architecture")
        sys.exit(1)

    best = max(rows, key=lambda r: r[4])
    d, gb, wire, qual, toks, _ = best
    where = "depth %d: " % d if has_ladder else ""
    qs = "%.5f" % qual if qual == qual else "unreadable"
    print("CORRECT  %swiring %.5f, fp16 smoke %s" % (where, wire, qs))
    print("         %.3f GB/token, %.2f tok/s" % (gb, toks))
    if base_tps:
        ratio = toks / base_tps
        if base_gb:
            print("         bytes %.2fx fewer than the supplied baseline"
                  % (base_gb / gb))
        verdict = ("BETTER than the supplied baseline"
                   if ratio >= 1.05 else
                   "PARITY with the supplied baseline" if ratio >= 0.95 else
                   "NOT better than the supplied baseline")
        print("\n%s: %.2fx" % (verdict, ratio))
        print("Speed alone is not the claim, and [3] is a smoke check, not a")
        print("quality result. Race the reference engine on THIS model.")
    else:
        print("\nNo --baseline supplied, so no ratio is claimed. Measure the")
        print("reference engine on THIS model and THIS machine, interleaved.")
    sys.exit(0)


if __name__ == "__main__":
    main()
