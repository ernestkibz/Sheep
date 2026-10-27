#!/usr/bin/env python3
"""Grade a Sheep model against the full-precision original, prompt by prompt.

    python tooling/sml_hf_grade.py MODEL.ssf HF_CHECKPOINT_DIR [--gpu]

For each prompt: the last-token logits from Hugging Face transformers in fp32 (an
implementation that shares no code with this engine), then Sheep's with the prompt
stepped one token at a time, batched exactly, and batched with per-256 activation
scales (the default). Reports each path's distance to full precision and whether
its top token matches. --gpu adds the GPU prompt path (sml-gpu-fwd, the batched
tensor-core kernels) as a fourth row, scored the same way.

This is the gate every new architecture passes before its row in sml_archs.py
says "ok". It used to live in a scratch directory, and a wiped scratch directory
took the gate with it -- which is why it is in the repository now.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "sml", "sml-cli" + (".exe" if os.name == "nt" else ""))
GPU = os.path.join(HERE, "..", "sml", "sml-gpu-fwd" + (".exe" if os.name == "nt" else ""))
ENV_KEYS = ("SML_SEQ_PROMPT", "SML_PREFILL_NB", "SML_LANES4", "SML_PROMPT_SB")

PROMPTS = [
    "You are a helpful live assistant. The user says: what am I holding in my hand right now, and is it safe to eat?",
    "Write a haiku about the rain in Nairobi.",
    "Explain in two sentences why the sky is blue.",
    "List three safe ways to store fresh milk without a fridge.",
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Translate to Swahili: Good morning, how did you sleep?",
    "A train leaves at 9:40 and the trip takes 2 hours 35 minutes. It arrives at",
    "In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was",
    "Summarize: The meeting was moved from Tuesday to Thursday because the projector broke and the manager was travelling.",
    "What is 17 multiplied by 23? Show your working step by step.",
    "My phone battery drains fast even when I am not using it. What should I check first?",
    "Once upon a time, in a small village at the foot of a mountain, there lived a girl who",
    "SELECT name, COUNT(*) FROM orders GROUP BY name ORDER BY",
    "Give me a short, friendly reminder to drink water.",
    "Why do onions make people cry when they are cut?",
    "The three primary colors of light are red, green and",
    "Write a polite email declining a meeting invitation for Friday afternoon.",
    "Is it safe to charge a laptop overnight? Answer briefly.",
    "You are Sheep, a live assistant that sees through the camera and hears through the microphone. Keep answers short, friendly and correct. Now the user says: which of these two apples looks riper?",
]

PATHS = [
    ("stepped", {"SML_SEQ_PROMPT": "1"}),
    ("exact",   {"SML_PROMPT_SB": "0"}),
    ("default", {}),
]


def sheep_logits(ssf, ids, extra):
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    env = dict(os.environ)
    for k in ENV_KEYS:
        env.pop(k, None)
    env.update(extra)
    try:
        if extra.get("GPU"):
            env["SML_LOGITS_OUT"] = path
            # the GPU binary runs from its own directory: it loads its PTX by name
            subprocess.run([GPU, os.path.abspath(ssf), ",".join(map(str, ids)), "0"],
                           env=env, check=True, cwd=os.path.dirname(GPU),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run([CLI, ssf, "--tokens", ",".join(map(str, ids)), "--gen", "0",
                            "--threads", "6", "--logits", path],
                           env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        meta = np.fromfile(path, dtype=np.int32, count=2)
        raw = np.fromfile(path, dtype=np.float32, offset=8)
    finally:
        os.remove(path)
    assert raw.size == int(meta[0]), (raw.size, meta)
    return raw.astype(np.float64)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    args = [a for a in sys.argv[1:] if a != "--gpu"]
    if len(args) != 2:
        sys.exit(__doc__)
    ssf, hf = args
    if "--gpu" in sys.argv:
        PATHS.append(("gpu", {"GPU": "1"}))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32)
    model.eval()
    torch.set_num_threads(6)

    to_fp = {n: [] for n, _ in PATHS}
    top1 = {n: 0 for n, _ in PATHS}
    lens = []
    for text in PROMPTS:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        lens.append(len(ids))
        with torch.no_grad():
            fp = model(torch.tensor([ids])).logits[0, -1].double().numpy()
        for name, extra in PATHS:
            v = sheep_logits(ssf, ids, extra)[: fp.size]
            to_fp[name].append(rel(v, fp))
            top1[name] += int(np.argmax(v) == np.argmax(fp))

    print(f"{os.path.basename(ssf)} vs {hf} (full precision)")
    print(f"{len(PROMPTS)} prompts, {min(lens)}-{max(lens)} tokens (median {int(np.median(lens))})")
    print(f"{'path':8s} {'rel-RMS to full precision':>28s} {'top-1 = full precision':>24s}")
    for name, _ in PATHS:
        f = np.array(to_fp[name])
        print(f"{name:8s} {f.mean():14.4f} (sd {f.std():.4f}) {top1[name]:16d}/{len(PROMPTS)}")


if __name__ == "__main__":
    main()
