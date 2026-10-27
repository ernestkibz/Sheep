#!/usr/bin/env python3
"""Where does a family head diverge? The residual stream, layer by layer.

    python tooling/sml_head_dump.py HEAD_DIR   (a directory sml_family_heads.py --keep wrote)

Runs transformers on the head's checkpoint with output_hidden_states, and the
engine stepped with SML_DUMP, over the same 20 tokens, and prints the relative
error of the LAST position's residual after the embedding and after each layer.
The first layer whose error jumps is where the behaviour differs.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "sml", "sml-cli" + (".exe" if os.name == "nt" else ""))


def main():
    import torch
    from transformers import AutoModelForCausalLM
    d = sys.argv[1]
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
    model.eval()
    ids = list(map(int, np.random.default_rng(7).integers(300, 50000, size=20)))
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_hidden_states=True)
    hs = [h[0, -1].double().numpy() for h in out.hidden_states]
    fd, dump = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    env = dict(os.environ, SML_DUMP=dump, SML_SEQ_PROMPT="1")
    subprocess.run([CLI, os.path.join(d, "model.ssf"), "--tokens", ",".join(map(str, ids)),
                    "--gen", "0", "--threads", "1"], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = np.fromfile(dump, dtype=np.float32)
    os.remove(dump)
    n = hs[0].size
    rows = raw.reshape(-1, n)
    per_tok = len(hs)            # embedding + one row per layer, per position
    last = rows[-per_tok:]
    for i, (a, b) in enumerate(zip(last, hs)):
        # HF's last hidden state has the final norm applied; the engine's does not
        tag = "embed" if i == 0 else f"layer {i}"
        if i == len(hs) - 1:
            tag += " (HF: after final norm)"
        e = np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30)
        print(f"  {tag:<28} rel {e:.4f}")


if __name__ == "__main__":
    main()
