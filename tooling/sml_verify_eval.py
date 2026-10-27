#!/usr/bin/env python3
"""Grade exact-arithmetic verification against the model answering alone.

Run the SAME questions twice against sml-server -- once with the orchestrator
on, so ACT_VERIFY computes the value and hands it to the model as fact, and once
with it off, so the model answers unaided. Everything else is identical.

    python tooling/sml_verify_eval.py --port-on 8121 --port-off 8122

The grader looks for the exact integer anywhere in the answer, with thousands
separators tolerated. That is deliberately generous to the UNVERIFIED side: it
scores the number, never the sentence around it.
"""
import argparse
import json
import re
import time
import urllib.request

# Answers are computed here, not typed, so the expected column cannot drift.
QUESTIONS = [
    "What is 17 times 23?",
    "What is 1234 plus 5678?",
    "Compute 144 divided by 12.",
    "What is 2 to the power of 10?",
    "Calculate 987 minus 654.",
    "What is 47 times 89?",
    "Evaluate (3 + 4) * 5.",
    "What is 100 minus 37 minus 8?",
    "Compute 12 x 4.",
    "What is 25 modulo 7?",
    "What is 999 times 999?",
    "Calculate 7 squared.",
    "What is 384 plus 976 plus 145?",
    "Compute 6250 divided by 25.",
    "What is 31 times 31?",
    "Evaluate 2 ^ 16.",
    "What is 8888 minus 999?",
    "Compute 123 times 456.",
]

EXPECTED = [
    17 * 23, 1234 + 5678, 144 // 12, 2 ** 10, 987 - 654, 47 * 89,
    (3 + 4) * 5, 100 - 37 - 8, 12 * 4, 25 % 7, 999 * 999, 7 ** 2,
    384 + 976 + 145, 6250 // 25, 31 * 31, 2 ** 16, 8888 - 999, 123 * 456,
]


def ask(port, q, n=64):
    body = json.dumps({"messages": [{"role": "user", "content": q}],
                       "max_tokens": n}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return (time.perf_counter() - t,
            d["choices"][0]["message"]["content"],
            d.get("sml", {}).get("verified", False))


def hit(text, want):
    """Is the exact value present as a standalone number?"""
    for m in re.finditer(r"-?[\d][\d,]*", text):
        try:
            if int(m.group(0).replace(",", "")) == want:
                return True
        except ValueError:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-on", type=int, required=True)
    ap.add_argument("--port-off", type=int, required=True)
    args = ap.parse_args()

    rows = []
    for q, want in zip(QUESTIONS, EXPECTED):
        t_on, a_on, v = ask(args.port_on, q)
        t_off, a_off, _ = ask(args.port_off, q)
        rows.append((q, want, hit(a_on, want), hit(a_off, want), v, a_on, a_off))

    n = len(rows)
    on = sum(r[2] for r in rows)
    off = sum(r[3] for r in rows)
    fired = sum(r[4] for r in rows)

    print(f"\n  {'question':<34}{'want':>10}  verified  unaided")
    for q, want, o, f, v, _, _ in rows:
        print(f"  {q[:33]:<34}{want:>10}  "
              f"{'PASS' if o else 'fail':<9}{'PASS' if f else 'fail'}"
              f"{'' if v else '   (declined)'}")

    print(f"\n  verifier fired on   {fired}/{n}")
    print(f"  WITH verification   {on}/{n}  ({100.0 * on / n:.1f}%)")
    print(f"  WITHOUT             {off}/{n}  ({100.0 * off / n:.1f}%)")

    # A case the verifier declined is not evidence either way about the
    # verifier; report the subset where it actually fired, separately.
    sub = [r for r in rows if r[4]]
    if sub:
        so = sum(r[2] for r in sub)
        sf = sum(r[3] for r in sub)
        print(f"\n  on the {len(sub)} it fired on: {so}/{len(sub)} verified "
              f"vs {sf}/{len(sub)} unaided")

    print("\n  wrong when unaided, right when verified:")
    for q, want, o, f, v, a_on, a_off in rows:
        if o and not f:
            print(f"    {q}  want {want}")
            print(f"      unaided:  {a_off.strip()[:90]}")
    print("\n  regressions (right unaided, wrong verified):")
    bad = [r for r in rows if r[3] and not r[2]]
    if not bad:
        print("    none")
    for q, want, o, f, v, a_on, a_off in bad:
        print(f"    {q}  want {want}\n      verified: {a_on.strip()[:90]}")


if __name__ == "__main__":
    main()
