#!/usr/bin/env python3
"""The orchestrator against the reference server, on the two things it adds.

    python tooling/sml_orch_race.py MODEL.ssf REFERENCE.gguf [--threads 6]

Both servers run the same model family on the same thread count, one request at a
time, greedy. Two races:

  repeat  -- each prompt is asked twice. The reference server keeps its prompt
             cache (it skips re-reading the prompt, then decodes the answer
             again); Sheep answers an exact repeat from its response cache.
             Scored: wall time of the second request, and that the second
             answer is identical to the first.
  arith   -- multiplication and addition questions with known answers. Sheep
             computes the value exactly and hands it to the model as a fact;
             the reference answers from the model alone. Scored: the exact
             value appears in the answer.

Neither race says the model is better. They measure what the serving layer does
with the model, which is the part the reference does not have.
"""
import argparse
import json
import os
import random
import re
import statistics as st
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SML_SERVER = os.path.join(HERE, "..", "sml", "sml-server" + (".exe" if os.name == "nt" else ""))
REF_SERVER = os.environ.get("REF_SERVER", r"C:\Users\earne\llama-cpu\llama-server.exe")

PROMPTS = [
    "Rewrite this more politely: send me the report now.",
    "Write a two-line thank-you note to my team.",
    "Explain in two sentences why the sky is blue.",
    "List three safe ways to store fresh milk without a fridge.",
    "Write a haiku about the rain in Nairobi.",
    "Summarize: the meeting moved from Tuesday to Thursday because the projector broke.",
    "Give me a short title for a talk about saving energy at home.",
    "Translate to Swahili: Good morning, how did you sleep?",
]


def chat(port, text, max_tokens):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": text}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return time.perf_counter() - t0, out["choices"][0]["message"]["content"]


def wait_up(port, proc, secs=180):
    t0 = time.time()
    while time.time() - t0 < secs:
        if proc.poll() is not None:
            sys.exit(f"server on {port} exited")
        try:
            chat(port, "Hi", 1)
            return
        except Exception:
            time.sleep(1)
    sys.exit(f"server on {port} did not come up")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ssf")
    ap.add_argument("gguf")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--arith", type=int, default=20)
    a = ap.parse_args()

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    ref = subprocess.Popen([REF_SERVER, "-m", a.gguf, "--port", "8091", "-t", str(a.threads),
                            "-c", "2048", "-dev", "none", "-ngl", "0"],
                           cwd=os.path.dirname(REF_SERVER), env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sml = subprocess.Popen([SML_SERVER, a.ssf, "--port", "8092", "--threads", str(a.threads),
                            "--ctx", "2048"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_up(8091, ref)
        wait_up(8092, sml)
        ports = {"sheep": 8092, "ref": 8091}

        print(f"== repeat: {len(PROMPTS)} prompts asked twice, max_tokens {a.max_tokens}, "
              f"{a.threads} threads each")
        second = {"sheep": [], "ref": []}
        same = {"sheep": 0, "ref": 0}
        for i, p in enumerate(PROMPTS):
            order = ("sheep", "ref") if i % 2 == 0 else ("ref", "sheep")
            for who in order:
                t1, x1 = chat(ports[who], p, a.max_tokens)
                t2, x2 = chat(ports[who], p, a.max_tokens)
                second[who].append(t2)
                same[who] += x1 == x2
            print(f"  prompt {i + 1}: second request sheep {second['sheep'][-1] * 1e3:8.1f} ms"
                  f"   ref {second['ref'][-1] * 1e3:8.1f} ms")
        ms, mr = st.median(second["sheep"]), st.median(second["ref"])
        print(f"  median second request: sheep {ms * 1e3:.1f} ms, ref {mr * 1e3:.1f} ms -> "
              f"{mr / ms:.0f}x faster; identical repeat answers sheep {same['sheep']}/{len(PROMPTS)}, "
              f"ref {same['ref']}/{len(PROMPTS)}")

        rng = random.Random(7)
        qs = []
        for k in range(a.arith):
            x, y = rng.randint(12, 989), rng.randint(12, 989)
            if k % 2 == 0:
                qs.append((f"What is {x} times {y}?", x * y))
            else:
                qs.append((f"What is {x} plus {y}?", x + y))
        print(f"== arith: {len(qs)} questions, max_tokens 96")
        right = {"sheep": 0, "ref": 0}
        for i, (q, v) in enumerate(qs):
            order = ("sheep", "ref") if i % 2 == 0 else ("ref", "sheep")
            got = {}
            for who in order:
                _, txt = chat(ports[who], q, 96)
                digits = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", txt)
                got[who] = re.search(r"(?<!\d)%d(?!\d)" % v, digits) is not None
                right[who] += got[who]
            print(f"  {q:<28} = {v:<7} sheep {'ok ' if got['sheep'] else 'NO '} ref {'ok' if got['ref'] else 'NO'}")
        print(f"  correct: sheep {right['sheep']}/{len(qs)}, ref {right['ref']}/{len(qs)}")
    finally:
        for p in (ref, sml):
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
