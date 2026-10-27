#!/usr/bin/env python3
"""End-to-end benchmark: sheep against a reference engine, across configurations.

Every number here is a median of N repeats, not a single sample. That is not
pedantry: an earlier one-sample run of this same long-context workload showed a
"15% regression" that vanished at n=5. Single samples on this machine
manufacture findings.

Configurations compared:
  reference out-of-box      upstream engine, its own defaults
  sheep out-of-box          autotune profile applies automatically
  sheep --no-tuning-profile control: proves the gain is the profile, not the fork
  sheep --orchestrator auto adds deterministic tool routing

Usage:
  python tooling/benchmark.py --sheep <sheep-server> --reference <reference-server> \
                              --model <model.ssf> [--reps 5]
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:8080"

LONG_TEXT = "The sea was calm and the ships sailed on. " * 400

WORKLOADS = [
    ("short-gen",   "Say hello and introduce yourself briefly.",       64),
    ("long-ctx",    "Summarise this text.\n\n" + LONG_TEXT,           48),
    ("code-gen",    "Write a Python function that reverses a linked "
                    "list, with comments.",                           128),
]

# tool-routing suite: does the model search when it should, and only then?
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]
NEEDS_NO_SEARCH = [
    "What is the capital of France?",
    "Explain how a binary search works.",
    "What is 17 * 23?",
    "Write a haiku about rain.",
    "Summarise the plot of Hamlet.",
    "Why is the sky blue?",
]
NEEDS_SEARCH = [
    "What is the current price of bitcoin?",
    "Who won the last Formula 1 race?",
    "What is the weather in Nairobi today?",
    "What is the latest version of Python?",
    "What are the current top posts on Hacker News?",
    "Summarise the news about AI this week.",
]


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        HOST + path, json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def wait_ready(proc, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited early (rc=%s)" % proc.returncode)
        try:
            urllib.request.urlopen(HOST + "/v1/models", timeout=5).read()
            time.sleep(1)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("server did not become ready")


def run_workloads(reps):
    out = {}
    for name, prompt, max_tokens in WORKLOADS:
        gen, pp = [], []
        for _ in range(reps):
            r = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "cache_prompt": False,
            })
            t = r["timings"]
            gen.append(t["predicted_per_second"])
            pp.append(t["prompt_per_second"])
        out[name] = {
            "gen": statistics.median(gen),
            "gen_spread": max(gen) - min(gen),
            "prompt": statistics.median(pp),
        }
    return out


def run_routing():
    def called(q):
        r = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": q}],
            "tools": TOOLS, "tool_choice": "auto",
            "max_tokens": 40, "temperature": 0.0,
        })
        return bool(r["choices"][0]["message"].get("tool_calls"))

    over = sum(called(q) for q in NEEDS_NO_SEARCH)
    under = sum(not called(q) for q in NEEDS_SEARCH)
    total = len(NEEDS_NO_SEARCH) + len(NEEDS_SEARCH)
    return {"over": over, "under": under, "correct": total - over - under, "total": total}


def measure(label, binary, extra, model, reps, routing):
    cmd = [binary, "-m", model, "--port", "8080", "--host", "127.0.0.1",
           "-c", "8192", "--jinja"] + extra
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        wait_ready(proc)
        result = run_workloads(reps)
        result["routing"] = run_routing() if routing else None
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheep", required=True)
    ap.add_argument("--reference", required=True,
                    help="upstream engine binary to compare against")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    configs = [
        ("reference out-of-box",      args.reference, [],                   True),
        ("sheep out-of-box",          args.sheep, [],                        True),
        ("sheep --no-tuning-profile", args.sheep, ["--no-tuning-profile"],   False),
        ("sheep --orchestrator auto", args.sheep, ["--orchestrator", "auto"], True),
    ]

    results = {}
    for label, binary, extra, routing in configs:
        print("running: %s ..." % label, flush=True)
        try:
            results[label] = measure(label, binary, extra, args.model,
                                     args.reps, routing)
        except Exception as exc:
            print("   failed: %s" % exc)
            results[label] = None

    print("\n" + "=" * 78)
    print("END-TO-END BENCHMARK  (median of %d, same model, same host)" % args.reps)
    print("=" * 78)
    header = "%-28s %10s %10s %10s" % ("config", "short gen", "long gen", "code gen")
    print(header)
    print("-" * 78)
    for label, _, _, _ in configs:
        r = results.get(label)
        if not r:
            print("%-28s %10s" % (label, "FAILED"))
            continue
        print("%-28s %10.2f %10.2f %10.2f" % (
            label, r["short-gen"]["gen"], r["long-ctx"]["gen"], r["code-gen"]["gen"]))

    print("\n%-28s %10s %10s" % ("config", "pp short", "pp long"))
    print("-" * 78)
    for label, _, _, _ in configs:
        r = results.get(label)
        if r:
            print("%-28s %10.2f %10.2f" % (
                label, r["short-gen"]["prompt"], r["long-ctx"]["prompt"]))

    print("\n%-28s %8s %8s %10s" % ("config", "over", "under", "routing"))
    print("-" * 78)
    for label, _, _, _ in configs:
        r = results.get(label)
        if r and r.get("routing"):
            rt = r["routing"]
            print("%-28s %8d %8d %7d/%d" % (
                label, rt["over"], rt["under"], rt["correct"], rt["total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
