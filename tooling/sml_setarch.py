#!/usr/bin/env python3
"""Copy an .ssf file with an architecture bit set -- a fact about the model that the
file got wrong, fixed without touching the weights.

    python tooling/sml_setarch.py IN.ssf OUT.ssf --gelu-erf

--gelu-erf  the model's GELU is the exact erf form (Hugging Face "gelu"). Files
            built before 2026-09-25 never said so, and the engine ran every GELU
            model with the tanh approximation -- right for GPT-2 ("gelu_new") and
            BLOOM, wrong for GPT-NeoX (pythia), whose config says "gelu".

The weights, scales and tables are copied byte for byte; only arch_bits changes.
"""
import argparse
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sml_format as F  # noqa: E402

# arch_bits is the second field of the header's "n_swa arch_bits" pair
_PREFIX = ("<4s" "I" "7I" "2f" "2I" f"{F.MAX_LEVELS}I" "Q" "6Q" "2I" "2f" "I")
ARCH_OFF = struct.calcsize(_PREFIX)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gelu-erf", action="store_true")
    a = ap.parse_args()
    if not a.gelu_erf:
        sys.exit("nothing to set")
    with open(a.src, "rb") as f:
        head = f.read(ARCH_OFF + 4)
    if head[:4] != b"SSF ":
        sys.exit(f"{a.src} is not an .ssf file")
    version = struct.unpack_from("<I", head, 4)[0]
    if version < 5:
        sys.exit(f"{a.src} is SSF v{version}; arch bits start at v5")
    bits = struct.unpack_from("<I", head, ARCH_OFF)[0]
    if not bits & F.ARCH_GELU_FFN:
        sys.exit(f"{a.src} has no GELU FFN (arch_bits {bits:#x})")
    shutil.copyfile(a.src, a.dst)
    with open(a.dst, "r+b") as f:
        f.seek(ARCH_OFF)
        f.write(struct.pack("<I", bits | F.ARCH_GELU_ERF))
    print(f"{a.dst}: arch_bits {bits:#x} -> {bits | F.ARCH_GELU_ERF:#x}")


if __name__ == "__main__":
    main()
