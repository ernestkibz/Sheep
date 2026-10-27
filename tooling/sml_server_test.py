"""sml-server protocol checks: tools, routing, slots, limits, stops.

Starts sml-server on the given model and checks each behaviour ML.M.I's agent
relies on, against the real engine rather than a mock:

  python tooling/sml_server_test.py C:/Users/earne/models/qwen3-0.6b-flat.ssf
  python tooling/sml_server_test.py C:/Users/earne/models/llama32-1b.ssf

Engine checks (routing, limits, slots, stops, the tool round trip's plumbing)
must pass on every model. Whether a small model CHOOSES to call a tool is the
model's skill, not the engine's; that line is printed as MODEL, not failed --
grading models is the ML.M.I checker's job.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "sml", "sml-server.exe" if os.name == "nt" else "sml-server")

WEATHER = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]

results = []


def check(name, ok, detail="", kind="ENGINE"):
    results.append((kind, name, ok))
    mark = "PASS" if ok else ("FAIL" if kind == "ENGINE" else "miss")
    print(f"  {kind:<6} {mark}  {name}{('  -- ' + detail) if detail else ''}", flush=True)


def post(port, body, stream=False, path="/v1/chat/completions"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, raw if stream else json.loads(raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())


def sse_events(raw):
    out = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            out.append(json.loads(line[6:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="cpu, gpu or hybrid")
    ap.add_argument("--gpu-layers", type=int, default=0)
    args = ap.parse_args()
    port = args.port
    srv = subprocess.Popen([EXE, args.model, "--port", str(port), "--ctx", str(args.ctx), "--slots", "2",
                            "--threads", str(args.threads), "--device", args.device]
                           + (["--gpu-layers", str(args.gpu_layers)] if args.gpu_layers else []), cwd=os.path.join(ROOT, "sml"),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(240):
            try:
                get(port, "/health")
                break
            except Exception:
                if srv.poll() is not None:
                    print(srv.stdout.read())
                    sys.exit("server exited")
                time.sleep(0.5)
        st = get(port, "/stats")
        print(f"model {st['model']}  template {st['template']}  slots {st['slots']}  device {st.get('device')}"
              f" ({st.get('gpu_layers')}/{st.get('gpu_of')} layers on the GPU) {st.get('device_why') or ''}")

        # 1. routing: a message that MENTIONS an endpoint is still a chat
        code, r = post(port, {"messages": [{"role": "user", "content": "What does GET /stats or /health return?"}],
                              "max_tokens": 8})
        check("chat mentioning /stats is answered as chat", code == 200 and "choices" in r)
        code, r = post(port, {"messages": [{"role": "user", "content": "hi"}]}, path="/v1/chat/completions/x")
        check("unknown path is 404", code == 404)

        # 2. content as an array of parts
        code, r = post(port, {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Reply with the single word: ready"}]}], "max_tokens": 8})
        check("content given as parts", code == 200 and r["choices"][0]["message"]["content"] is not None)

        # 3. exact arithmetic still bypasses the model
        code, r = post(port, {"messages": [{"role": "user", "content": "47*89"}], "max_tokens": 16,
                              "temperature": 0})
        check("orchestrator: pure arithmetic exact", code == 200 and "4183" in r["choices"][0]["message"]["content"])

        # 4. tool round trip
        msgs = [{"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
                {"role": "user", "content": "What is the weather in Paris right now?"}]
        code, r = post(port, {"messages": msgs, "tools": WEATHER, "max_tokens": 96, "enable_thinking": False})
        ch = r["choices"][0] if code == 200 else {}
        calls = (ch.get("message") or {}).get("tool_calls") or []
        check("tool request answered", code == 200, f"finish {ch.get('finish_reason')}")
        called = bool(calls) and calls[0]["function"]["name"] == "get_weather"
        check("model calls get_weather", called,
              json.dumps(calls[0]["function"]) if calls else repr((ch.get("message") or {}).get("content"))[:120],
              kind="MODEL")
        if calls:
            fa = calls[0]["function"]["arguments"]
            ok = isinstance(fa, str)
            try:
                json.loads(fa)
            except Exception:
                ok = False
            check("arguments are a JSON string", ok, fa)
            check("finish_reason is tool_calls", ch.get("finish_reason") == "tool_calls")
        call = calls[0] if calls else {"id": "call_0", "type": "function",
                                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
        msgs2 = msgs + [{"role": "assistant", "content": None, "tool_calls": [call]},
                        {"role": "tool", "tool_call_id": call.get("id", "call_0"),
                         "content": "{\"city\": \"Paris\", \"temp_c\": 18, \"sky\": \"light rain\"}"}]
        code, r = post(port, {"messages": msgs2, "tools": WEATHER, "max_tokens": 64, "enable_thinking": False})
        body = r["choices"][0]["message"] if code == 200 else {}
        check("tool result turn answered", code == 200, repr(body.get("content"))[:100])
        check("answer uses the tool result", "18" in (body.get("content") or ""), kind="MODEL")
        # the orchestrator must not re-answer an old user turn after a tool result
        msgs3 = [{"role": "user", "content": "2+2"},
                 {"role": "assistant", "content": None, "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp_c\": 18}"}]
        code, r = post(port, {"messages": msgs3, "tools": WEATHER, "max_tokens": 16, "enable_thinking": False})
        check("orchestrator stays out of a tool loop",
              code == 200 and (r.get("sml") or {}).get("action") != "VERIFY", json.dumps(r.get("sml")))

        # 5. streaming a tool turn: tool_calls arrive as a delta, then a finish chunk
        code, raw = post(port, {"messages": msgs, "tools": WEATHER, "max_tokens": 96, "stream": True,
                                "enable_thinking": False}, stream=True)
        ev = sse_events(raw) if code == 200 else []
        fins = [e["choices"][0].get("finish_reason") for e in ev if e["choices"][0].get("finish_reason")]
        check("stream ends with a finish_reason", bool(fins), str(fins))
        if called:
            check("stream carries tool_calls", any("tool_calls" in e["choices"][0]["delta"] for e in ev))

        # 6. stop strings
        code, r = post(port, {"messages": [{"role": "user", "content": "Count from 1 to 10, separated by commas."}],
                              "max_tokens": 48, "stop": ["5"]})
        txt = r["choices"][0]["message"]["content"] if code == 200 else ""
        check("stop string ends the reply before it", code == 200 and "5" not in txt, repr(txt)[:80])

        # 7. an over-long prompt is refused, not silently cut
        long = "word " * (args.ctx * 2)
        code, r = post(port, {"messages": [{"role": "user", "content": long}], "max_tokens": 4})
        check("over-long prompt refused with 400", code == 400 and "does not fit" in str(r), str(r)[:80])

        # 8. slots: two conversations interleaved each keep their prefix
        a = [{"role": "user", "content": "Conversation A. Tell me one fact about the Moon."}]
        b = [{"role": "user", "content": "Conversation B. Tell me one fact about the ocean floor."}]
        post(port, {"messages": a, "max_tokens": 16})
        post(port, {"messages": b, "max_tokens": 16})
        a2 = a + [{"role": "assistant", "content": "It has no air."}, {"role": "user", "content": "Another?"}]
        code, r = post(port, {"messages": a2, "max_tokens": 8})
        reused = (r.get("sml") or {}).get("reused", 0) if code == 200 else 0
        check("slots: A reuses its prefix after B ran", reused > 5, f"reused {reused} tokens")

        # 9. repeat is a cache hit
        t0 = time.time()
        code, r = post(port, {"messages": a2, "max_tokens": 8})
        check("repeat served from the response cache", code == 200 and (r.get("sml") or {}).get("cached") is True,
              f"{(time.time() - t0) * 1000:.1f} ms")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()
    eng = [r for r in results if r[0] == "ENGINE"]
    mod = [r for r in results if r[0] == "MODEL"]
    print(f"\nengine {sum(r[2] for r in eng)}/{len(eng)}   model {sum(r[2] for r in mod)}/{len(mod)}")
    sys.exit(0 if all(r[2] for r in eng) else 1)


if __name__ == "__main__":
    main()
