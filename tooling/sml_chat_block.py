#!/usr/bin/env python3
"""Give an already-built SSF the model's own chat frame and stop tokens.

    python tooling/sml_chat_block.py MODEL.ssf HF_DIR

Files built before 2026-09-30 carry no "chat" block. This reads the template,
BOS/EOS and end-of-turn ids from the checkpoint's own files (sml_build.chat_block,
with the same vocabulary gate), appends a new JSON section with them to the end
of the file, and points the header at it. Tensors are untouched.
"""
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sml_format import HEADER_FMT  # noqa: E402
import sml_quantize as sml_build  # noqa: E402  (the Sheep Quantizer)
import sml_read  # noqa: E402


def main():
    ssf, hf = sys.argv[1], sys.argv[2]
    f = sml_read.SmlFile(ssf)
    head = dict(f.head)
    import sml_hf
    W = sml_hf.open_weights(hf)
    toks = W.field("tokenizer.tokens")
    head["chat"] = sml_build.chat_block(hf, toks, int(f.head["tensors"]["token_embd.weight"]["rows"]))
    blob = json.dumps(head, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode("utf-8")
    n = struct.calcsize(HEADER_FMT)
    b = list(struct.unpack(HEADER_FMT, bytes(f.mm[:n])))
    del f
    size = os.path.getsize(ssf)
    with open(ssf, "r+b") as out:
        out.seek(size)
        out.write(blob)
        b[19], b[20] = size, len(blob)          # json_off, json_n
        out.seek(0)
        out.write(struct.pack(HEADER_FMT, *b))
    c = head["chat"]
    print(f"chat block written: template {len(c['template'])} chars, stops {c['stop_ids']}, bos {c['bos_id']}")


if __name__ == "__main__":
    main()
