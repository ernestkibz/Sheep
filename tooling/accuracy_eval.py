#!/usr/bin/env python3
"""Accuracy harness with enough headroom to measure a runtime change.

An earlier attempt compared greedy against temperature 0.8 on 12 easy factual
questions and got 11.0/12 both ways. That set was saturated: it could not have
shown an improvement even if one existed. So these cases are deliberately hard
enough that Qwen2.5-3B is expected to land somewhere in the 50-70% band, which
leaves room for a lever to move the number.

The cases were written before any accuracy lever was implemented, so the set
cannot have been shaped to flatter a particular change.

Four task classes, because the interesting hypothesis is that different classes
want different execution policies:
  math    - multi-step arithmetic word problems
  code    - predict the output of a short snippet
  factual - multi-hop, needs chaining two facts
  format  - the answer is trivial but the *format* is constrained

Usage:
  python tooling/accuracy_eval.py --server <sheep-server> --model <model.ssf>
  python tooling/accuracy_eval.py ... --extra --orchestrator auto
  python tooling/accuracy_eval.py ... --reps 5 --self-consistency 3
"""

import argparse
import collections
import json
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:8080"

# (class, question, accepted answers)
CASES = [
    # ---------------------------------------------------------------- math
    ("math", "A train travels 60 km in 45 minutes. At the same speed, how many km does it travel in 2 hours?", ["160"]),
    ("math", "A shop sells pens at 3 for $2. How much do 21 pens cost, in dollars?", ["14"]),
    ("math", "If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need to make 100 widgets?", ["5"]),
    ("math", "A book costs $12 after a 25% discount. What was the original price in dollars?", ["16"]),
    ("math", "The sum of three consecutive even numbers is 48. What is the largest of them?", ["18"]),
    ("math", "A tank fills in 6 hours using tap A alone, and 3 hours using tap B alone. How many hours using both together?", ["2"]),
    ("math", "12% of a number is 54. What is the number?", ["450"]),
    ("math", "A rectangle has perimeter 34 and length 10. What is its area?", ["70"]),
    ("math", "You buy 3 shirts costing $18 each and pay with $100. How much change do you get, in dollars?", ["46"]),
    ("math", "A number is tripled and then decreased by 7, giving 20. What was the number?", ["9"]),

    # ---------------------------------------------------------------- code
    ("code", 'What does this print?\nprint(len("hello world".split()))', ["2"]),
    ("code", "What does this print?\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))", ["4"]),
    ("code", "What does this print?\nprint(sum(range(1, 5)))", ["10"]),
    ("code", 'What does this print?\nprint("abc"[::-1])', ["cba"]),
    ("code", "What does this print?\nd = {'a': 1}\nd['b'] = 2\nprint(len(d))", ["2"]),
    ("code", "What does this print?\nprint(3 // 2 + 3 % 2)", ["2"]),
    ("code", "What does this print?\nprint(bool([]) or bool([0]))", ["true"]),
    ("code", 'How many elements does this list have?\nlist(range(0, 10, 3))', ["4"]),
    ("code", 'How many elements are in the result of "a,b,,c".split(",")?', ["4"]),
    ("code", "What does this print?\nx = 5\ndef f():\n    return x + 1\nprint(f())", ["6"]),

    # ------------------------------------------------------------- factual
    ("factual", "What is the capital of the country that hosted the 2016 Summer Olympics?", ["brasilia", "brasília"]),
    ("factual", "In which year was the author of 'Nineteen Eighty-Four' born?", ["1903"]),
    ("factual", "What is the chemical symbol of the element with atomic number 26?", ["fe"]),
    ("factual", "Who wrote the play in which Lady Macbeth appears?", ["shakespeare"]),
    ("factual", "Which ocean lies between Africa and Australia?", ["indian"]),
    ("factual", "What is the currency used in the country where the Eiffel Tower stands?", ["euro"]),
    ("factual", "How many bones are in the adult human body?", ["206"]),
    ("factual", "In which year did the Apollo 11 Moon landing take place?", ["1969"]),
    ("factual", "What language is most widely spoken in the largest country in South America?", ["portuguese"]),
    ("factual", "Which planet in our solar system is known as the Red Planet?", ["mars"]),

    # -------------------------------------------------------------- format
    ("format", "Is 17 a prime number? Answer yes or no.", ["yes"]),
    ("format", "Which is larger, 0.9 or 0.11? Reply with just the number.", ["0.9"]),
    ("format", "Is the Pacific Ocean larger than the Atlantic Ocean? Answer yes or no.", ["yes"]),
    ("format", "How many letters are in the word 'strawberry'? Reply with a number only.", ["10"]),
    ("format", "How many times does the letter r appear in 'strawberry'? Reply with a number only.", ["3"]),
    ("format", "Is 2024 a leap year? Answer yes or no.", ["yes"]),
    ("format", "Which comes first alphabetically, banana or apple? Reply with one word.", ["apple"]),
    ("format", "How many times does the letter r appear in 'raspberry'? Reply with a number only.", ["3"]),
    ("format", "What is 7 squared? Reply with a number only.", ["49"]),
    ("format", "Is the chemical formula for water H2O or CO2? Reply with the formula.", ["h2o"]),
]


def normalise(text):
    """Tolerant match: the model may pad an answer with words or punctuation."""
    t = text.strip().lower()
    t = t.replace("$", "").replace(",", "").replace("*", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .!:;\"'`\n")


def correct(reply, answers):
    got = normalise(reply)
    for want in answers:
        w = normalise(want)
        if got == w:
            return True
        # allow the answer inside a sentence, on a token boundary
        if re.search(r"(^|[^0-9a-z])" + re.escape(w) + r"([^0-9a-z]|$)", got):
            return True
    return False


def extract_code(question):
    """Pull the runnable snippet out of a code question.

    Sheep would do this server-side to decide whether a deterministic task has
    a checkable ground truth. Here it only has to be good enough to measure
    whether the loop is worth building.
    """
    lines = question.splitlines()
    if len(lines) > 1 and any(c in question for c in "()[]"):
        return "\n".join(lines[1:]).strip()
    # single-line questions with an inline expression, e.g. '"a,b,,c".split(",")'
    m = re.search(r'([\'"\w][\w\'"\.,\(\)\[\]: -]*\.\w+\([^\)]*\))', question)
    return m.group(1) if m else ""


def run_code(snippet, timeout=10):
    """Execute the snippet and return its output as ground truth.

    This is the whole point of the lever: the checker is *external*. The model
    is not being asked to grade itself - the interpreter decides.
    """
    if not snippet:
        return None
    src = snippet if "print(" in snippet else "print(%s)" % snippet
    try:
        out = subprocess.run(["python", "-c", src], capture_output=True,
                             text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def post(payload, timeout=1800):
    req = urllib.request.Request(
        HOST + "/v1/chat/completions", json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as exc:
        # surface the server's message; a bare "HTTP 500" hides the cause
        body = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError("HTTP %s: %s" % (exc.code, body)) from None


def ask(question, temperature, max_tokens=320):
    body = {
        "messages": [{"role": "user", "content":
                      question + "\n\nThink briefly, then end with the final answer on its own line."}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    r = post(body)
    content = r["choices"][0]["message"]["content"] or ""
    # the final answer is what we score; models pad with reasoning
    lines = [ln for ln in content.strip().splitlines() if ln.strip()]
    # the tail, not just the final line: models often put the answer on its own
    # line and then add a closing sentence after it
    return " ".join(lines[-2:]) if lines else "", content


def fetch_tools():
    """The tool definitions the server is offering, as a client would send them."""
    raw = json.load(urllib.request.urlopen(HOST + "/tools", timeout=60))
    return [t["definition"] for t in raw]


def repair_tool_args(raw):
    """Make model-emitted tool arguments valid JSON.

    Models put raw newlines inside the JSON string when the code is multi-line.
    Both sheep and the upstream engine hand those back unchanged, then refuse
    to re-parse them, so the control characters have to be escaped before the
    arguments are replayed or executed.
    """
    raw = raw or "{}"
    try:
        return json.loads(raw), raw
    except ValueError:
        pass
    # escape the raw control characters that make the string invalid, without
    # touching escapes the model already got right
    fixed = (raw.replace(chr(13) + chr(10), chr(92) + "n")
                .replace(chr(10), chr(92) + "n")
                .replace(chr(9), chr(92) + "t"))
    try:
        return json.loads(fixed), fixed
    except ValueError:
        return None, raw


def execute_tool_call(call):
    """Run what the model asked for, and hand back the real result.

    Only exec_shell_command is honoured - it is the one that produces ground
    truth for a computable question. The commands come from this eval's own
    questions, on a local box; this is a measurement harness, not a sandbox.
    """
    name = call["function"]["name"]
    args, _ = repair_tool_args(call["function"].get("arguments"))
    if args is None:
        return "error: malformed tool arguments"

    if name == "run_python":
        # argv, not a shell string - mirrors what the server tool does
        code = args.get("code", "")
        if not code:
            return "error: no code given"
        try:
            out = subprocess.run(["python", "-c", code], capture_output=True,
                                 text=True, timeout=15)
            return (out.stdout or out.stderr or "").strip()[:2000] or "(no output)"
        except Exception as exc:
            return "error: %s" % exc

    if name != "exec_shell_command":
        return "error: %s is not available here" % name

    command = args.get("command", "")
    if not command:
        return "error: no command given"
    try:
        out = subprocess.run(command, shell=True, capture_output=True,
                             text=True, timeout=15)
        return (out.stdout or out.stderr or "").strip()[:2000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "error: command timed out"
    except Exception as exc:
        return "error: %s" % exc


def ask_with_tools(question, temperature, tools, max_rounds=3):
    """The shipped path: offer tools, let the orchestrator decide, run what it asks.

    This is what actually ships, so it is what gets measured. It differs from
    --verify-exec: there the harness executed the snippet itself, here the model
    has to compose a working command, which is where it can still fail.
    """
    # identical wording to ask(); an earlier version dropped "Think briefly"
    # here, which removed the chain-of-thought cue and made the tools path look
    # 30 points worse on math than it actually was
    messages = [{"role": "user", "content":
                 question + "\n\nThink briefly, then end with the final answer on its own line."}]
    used_tool = False
    for _ in range(max_rounds):
        body = {"messages": messages, "tools": tools, "tool_choice": "auto",
                "max_tokens": 320}
        if temperature is not None:
            body["temperature"] = temperature
        msg = post(body)["choices"][0]["message"]

        calls = msg.get("tool_calls")
        if not calls:
            content = (msg.get("content") or "").strip()
            lines = [ln for ln in content.splitlines() if ln.strip()]
            return " ".join(lines[-2:]) if lines else "", used_tool

        used_tool = True
        # Models emit raw newlines inside the arguments JSON string, which both
        # sheep and the upstream engine accept on the way out, then refuse to
        # re-parse on replay (HTTP 500). Re-serialise so the history we send
        # back is always valid JSON.
        safe_calls = []
        for call in calls:
            fixed = json.loads(json.dumps(call))
            # NO repair here on purpose: the server now fixes malformed
            # arguments itself, so the harness must exercise that path rather
            # than hide it. Replay exactly what the model emitted.
            pass
            safe_calls.append(fixed)
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": safe_calls})
        for call in calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": execute_tool_call(call),
            })
    # rounds exhausted: ask once more without tools so there is a real answer
    # to score rather than an empty string
    body = {"messages": messages, "max_tokens": 200}
    if temperature is not None:
        body["temperature"] = temperature
    try:
        content = (post(body)["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        content = ""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    return " ".join(lines[-2:]) if lines else "", used_tool


def ask_voted(question, n, temperature):
    """Self-consistency: sample n times and take the most common answer."""
    finals = []
    for _ in range(n):
        final, _ = ask(question, temperature)
        finals.append(normalise(final))
    return collections.Counter(finals).most_common(1)[0][0], ""


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


def run_once(temperature, self_consistency, verify_exec=False, tools=None):
    per_class = collections.defaultdict(lambda: [0, 0])
    wrong = []
    verified = 0
    for cls, question, answers in CASES:
        if tools is not None:
            final, tool_used = ask_with_tools(question, temperature, tools)
            verified += tool_used
        elif self_consistency > 1:
            final, _ = ask_voted(question, self_consistency, temperature or 0.7)
        else:
            final, _ = ask(question, temperature)

        # Execution verification: for deterministic tasks there is a real
        # checker available, so the model does not have to be right first try.
        # Sheep runs the snippet and hands the model the ground truth.
        if verify_exec and tools is None and cls == "code":
            output = run_code(extract_code(question))
            if output is not None:
                verified += 1
                followup = ("%s\n\nThe code was executed. Its actual output was:\n%s\n\n"
                            "Using that, give the final answer on its own line."
                            % (question, output))
                final, _ = ask(followup, temperature)

        ok = correct(final, answers)
        per_class[cls][0] += ok
        per_class[cls][1] += 1
        if not ok:
            wrong.append((cls, question.splitlines()[0][:46], final[:34], answers[0]))
    return per_class, wrong, verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit to use the server's own default")
    ap.add_argument("--self-consistency", type=int, default=1)
    ap.add_argument("--show-wrong", action="store_true")
    ap.add_argument("--verify-exec", action="store_true",
                    help="run code snippets and feed the real output back")
    ap.add_argument("--use-server-tools", action="store_true",
                    help="measure the shipped path: offer the server's tools and "
                         "execute whatever the model calls")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra server args, e.g. --extra --orchestrator auto")
    args = ap.parse_args()

    cmd = [args.server, "-m", args.model, "--port", "8080", "--host",
           "127.0.0.1", "-c", "8192", "--jinja"] + args.extra
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        wait_ready(proc)
        totals, spread, last_wrong = collections.defaultdict(list), [], []
        for rep in range(args.reps):
            per_class, wrong, nverified = run_once(
                args.temperature, args.self_consistency, args.verify_exec,
                fetch_tools() if args.use_server_tools else None)
            overall_hits = sum(v[0] for v in per_class.values())
            overall_n = sum(v[1] for v in per_class.values())
            spread.append(100.0 * overall_hits / overall_n)
            for cls, (hits, n) in per_class.items():
                totals[cls].append(100.0 * hits / n)
            last_wrong = wrong
            print("  rep %d: %.1f%%" % (rep + 1, spread[-1]), flush=True)

        print("\n" + "=" * 58)
        print("ACCURACY  (median of %d reps, %d cases)" % (args.reps, len(CASES)))
        if args.extra:
            print("server args: %s" % " ".join(args.extra))
        if args.self_consistency > 1:
            print("self-consistency: n=%d" % args.self_consistency)
        if args.verify_exec:
            print("execution verification: on (%d snippets ran)" % nverified)
        if args.use_server_tools:
            print("server tools offered; %d requests used a tool" % nverified)
        print("=" * 58)
        for cls in ("math", "code", "factual", "format"):
            if cls in totals:
                print("  %-9s %5.1f%%" % (cls, statistics.median(totals[cls])))
        print("  %-9s %5.1f%%   (spread %.1f)"
              % ("OVERALL", statistics.median(spread), max(spread) - min(spread)))

        if args.show_wrong and last_wrong:
            print("\nmissed (last rep):")
            for cls, q, got, want in last_wrong:
                print("  %-8s %-46s got=%-34s want=%s" % (cls, q, got, want))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
