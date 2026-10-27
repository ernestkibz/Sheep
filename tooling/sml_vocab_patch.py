#!/usr/bin/env python3
"""Rewrite an .ssf file's tokenizer section from JSON to the binary form.

The engine parses no JSON, so the vocab has to be binary (sml/sml-vocab.h). This
upgrades a file in place rather than re-quantizing it -- a full GPTQ build is
45 minutes and the weights do not change.

The vocab and the JSON metadata are the last two sections, so only the tail is
rewritten and the header offsets are updated.

    python tooling/sml_vocab_patch.py MODEL.ssf
"""
import json
import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sml_format import HEADER_FMT, MAGIC, build_vocab_blob   # noqa: E402


def main():
    path = sys.argv[1]
    n = struct.calcsize(HEADER_FMT)
    with open(path, "rb") as f:
        head = f.read(n)
    fields = list(struct.unpack(HEADER_FMT, head))
    if fields[0] != MAGIC:
        raise SystemExit(f"not an .ssf file: {path}")

    n_tensors, table_off, json_off, json_n, tok_off, tok_n, _ = fields[17:24]

    with open(path, "rb") as f:
        f.seek(tok_off);  tok_raw = f.read(tok_n)
        f.seek(json_off); jblob = f.read(json_n)

    if tok_raw[:1] != b"{":
        print("already binary; nothing to do")
        return

    tok = json.loads(tok_raw.decode("utf-8"))
    blob = build_vocab_blob(tok.get("tokens") or [], tok.get("merges") or [],
                            tok.get("bos"), tok.get("eos"))
    print(f"vocab: {len(tok.get('tokens') or [])} tokens, "
          f"{len(tok.get('merges') or [])} merges  "
          f"{tok_n / 1e6:.1f} MB JSON -> {len(blob) / 1e6:.1f} MB binary")

    new_json_off = tok_off + len(blob)
    # field order: [17] n_tensors [18] table_off [19] json_off [20] json_n
    #              [21] tok_off   [22] tok_n     [23] reserved
    fields[19] = new_json_off          # json_off
    fields[22] = len(blob)             # tok_n  (NOT [21], which is tok_off)
    with open(path, "r+b") as f:
        f.seek(tok_off);      f.write(blob)
        f.seek(new_json_off); f.write(jblob)
        f.truncate(new_json_off + len(jblob))
        f.seek(0);            f.write(struct.pack(HEADER_FMT, *fields))
    print(f"patched {path}")


if __name__ == "__main__":
    main()
