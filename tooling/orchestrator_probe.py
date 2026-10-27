#!/usr/bin/env python3
"""Does a logit-read classifier beat a keyword gate at routing requests?

Phase 3 of the orchestrator design rests on one assumption: that the resident
model can decide "answer directly" vs "use a tool" well enough to be worth
~100ms. An earlier test that let the model *generate* its answer scored 2/4 and
got the obvious cases backwards, so this measures the logit form instead - one
forward pass, reading P(DIRECT) vs P(TOOL) on the first token - and scores it
against a deterministic keyword gate on the same set.

If the classifier does not clearly beat the gate, Phase 3 is not worth building.

Usage:
    python tooling/orchestrator_probe.py [--host http://127.0.0.1:8080]
"""

import argparse
import json
import math
import re
import time
import urllib.request

# (request, ground truth) - TOOL means it genuinely needs live information
CASES = [
    ("What is the capital of France?",                                  "DIRECT"),
    ("Explain how a binary search works.",                              "DIRECT"),
    ("Write a haiku about rain.",                                       "DIRECT"),
    ("What is 17 * 23?",                                                "DIRECT"),
    ("Summarise the plot of Hamlet.",                                   "DIRECT"),
    ("Convert 100 fahrenheit to celsius.",                              "DIRECT"),
    ("Give me a regex for matching email addresses.",                   "DIRECT"),
    ("Why is the sky blue?",                                            "DIRECT"),
    ("Refactor this loop into a list comprehension.",                   "DIRECT"),
    ("What does the acronym HTTP stand for?",                           "DIRECT"),
    ("What is the latest version of Python?",                           "TOOL"),
    ("Who won the last Formula 1 race?",                                "TOOL"),
    ("What is the current price of bitcoin?",                           "TOOL"),
    ("What is the weather in Nairobi today?",                           "TOOL"),
    ("Summarise the news about AI regulation this week.",               "TOOL"),
    ("Read https://example.org and tell me what it says.",             "TOOL"),
    ("What did the Fed decide at its most recent meeting?",             "TOOL"),
    ("Has the Linux kernel had a release recently?",                     "TOOL"),
    ("What are the current top posts on Hacker News?",                  "TOOL"),
    ("Find the release date of the newest Qwen model.",                 "TOOL"),
]

PROMPT = (
    "You route user requests. Reply with exactly one word.\n"
    "TOOL   = needs live or current information from the web\n"
    "DIRECT = can be answered from existing knowledge\n\n"
    "Request: {req}\n"
    "Answer:"
)

LIVE = re.compile(
    r"\b(latest|current|currently|today|todays|now|recent|recently|this week|"
    r"this year|news|price|weather|who won|release date|newest|new version|"
    r"https?://)\b", re.I)


def deterministic_gate(request):
    return "TOOL" if LIVE.search(request) else "DIRECT"


def post(host, path, payload, timeout=600):
    req = urllib.request.Request(
        host + path, json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def classify(host, request):
    """One forward pass; read the probabilities rather than generating prose."""
    body = {
        "prompt": PROMPT.format(req=request),
        "n_predict": 1,
        "n_probs": 20,
        "temperature": 0.0,
        "cache_prompt": True,
    }
    started = time.time()
    out = post(host, "/completion", body)
    ms = (time.time() - started) * 1000

    probs = {}
    comp = out.get("completion_probabilities") or []
    if comp:
        # server returns top_logprobs: [{token, logprob}, ...]
        for entry in comp[0].get("top_logprobs", comp[0].get("probs", [])):
            tok = (entry.get("token") or entry.get("tok_str") or "").strip().upper()
            if tok not in ("DIRECT", "TOOL"):
                continue
            if "logprob" in entry:
                p = math.exp(entry["logprob"])
            else:
                p = entry.get("prob", 0.0)
            probs[tok] = max(probs.get(tok, 0.0), p)

    if not probs:
        # fall back to whatever it actually emitted
        text = (out.get("content") or "").strip().upper()
        return ("TOOL" if "TOOL" in text else "DIRECT"), 0.0, ms

    label = max(probs, key=probs.get)
    total = sum(probs.values()) or 1.0
    return label, probs[label] / total, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8080")
    args = ap.parse_args()

    gate_hits = llm_hits = 0
    latencies = []
    disagree = []

    print("%-52s %-7s %-7s %-7s %s" % ("request", "truth", "gate", "llm", "conf"))
    print("-" * 88)
    for request, truth in CASES:
        gate = deterministic_gate(request)
        llm, conf, ms = classify(args.host, request)
        latencies.append(ms)
        gate_ok, llm_ok = gate == truth, llm == truth
        gate_hits += gate_ok
        llm_hits += llm_ok
        if gate != llm:
            disagree.append((request, truth, gate, llm))
        print("%-52s %-7s %-7s%s %-7s%s %.2f"
              % (request[:52], truth,
                 gate, " " if gate_ok else "!",
                 llm, " " if llm_ok else "!", conf))

    n = len(CASES)
    print("-" * 88)
    print("keyword gate    : %2d/%d  (%.0f%%)" % (gate_hits, n, 100.0 * gate_hits / n))
    print("logit classifier: %2d/%d  (%.0f%%)" % (llm_hits, n, 100.0 * llm_hits / n))
    print("median latency  : %.0f ms" % sorted(latencies)[len(latencies) // 2])
    print("disagreements   : %d" % len(disagree))
    for request, truth, gate, llm in disagree:
        better = "llm right" if llm == truth else "gate right"
        print("   %-46s truth=%-6s gate=%-6s llm=%-6s  %s"
              % (request[:46], truth, gate, llm, better))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
