#!/usr/bin/env python3
"""The chat server on each device: speed, agreement with the CPU, and the fallback.

    python tooling/sml_device_check.py MODEL.ssf

Starts sml-server per device (cpu, gpu, hybrid with half the layers on the
card, and gpu with the PTX hidden, which must fall back to the CPU), sends the
same conversation (a prompt, then a follow-up that continues the cached prefix),
and reports prompt and decode speed and whether the greedy replies match the
CPU's. Exits non-zero if a device fails to answer or the fallback does not
fall back.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SML = os.path.join(HERE, "..", "sml")
EXE = os.path.join(SML, "sml-server.exe")


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.load(r)


def run(model, device, extra=(), env=None, port=8131):
    e = dict(os.environ)
    e.update(env or {})
    srv = subprocess.Popen([EXE, model, "--port", str(port), "--ctx", "2048", "--slots", "1", "--threads", "6",
                            "--device", device, *extra], cwd=SML, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=e)
    try:
        for _ in range(240):
            try:
                get(port, "/health")
                break
            except Exception:
                if srv.poll() is not None:
                    return {"error": srv.stdout.read()[-400:]}
                time.sleep(0.5)
        st = get(port, "/stats")
        msgs = [{"role": "user", "content": "Write three short sentences about why the sea is salty."}]
        r1 = post(port, {"messages": msgs, "max_tokens": 96, "temperature": 0})
        s1 = get(port, "/stats")
        msgs += [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
                 {"role": "user", "content": "Now say it in one sentence."}]
        r2 = post(port, {"messages": msgs, "max_tokens": 48, "temperature": 0})
        s2 = get(port, "/stats")
        return {"device": st.get("device"), "layers": f"{st.get('gpu_layers')}/{st.get('gpu_of')}", "why": st.get("device_why"),
                "reply1": r1["choices"][0]["message"]["content"], "reply2": r2["choices"][0]["message"]["content"],
                "pp": s1["last_prompt"] / max(1e-9, s1["last_prompt_ms"] / 1000), "tg": s1["last_gen"] / max(1e-9, s1["last_gen_ms"] / 1000),
                "reused2": s2["last_reused"], "prompt2": s2["last_prompt"]}
    finally:
        srv.kill()


def main():
    model = sys.argv[1]
    res = {}
    for name, dev, extra, env in (("cpu", "cpu", (), None), ("gpu", "gpu", (), None),
                                  ("hybrid", "hybrid", ("--gpu-layers", "14"), None),
                                  ("fallback", "gpu", (), {"SML_PTX": "no-such-file.ptx"})):
        r = run(model, dev, extra, env)
        res[name] = r
        if "error" in r:
            print(f"{name:9s} FAILED TO START: {r['error']}")
            continue
        print(f"{name:9s} device {r['device']:6s} layers {r['layers']:6s} prompt {r['pp']:6.1f} tok/s  decode {r['tg']:5.1f} tok/s"
              f"  follow-up reused {r['reused2']}/{r['prompt2']}  {r['why'] or ''}", flush=True)
    bad = []
    cpu = res.get("cpu", {})
    for name in ("gpu", "hybrid"):
        r = res.get(name, {})
        if "error" in r or r.get("device") == "cpu":
            bad.append(f"{name} did not run on the GPU")
            continue
        same1, same2 = r["reply1"] == cpu.get("reply1"), r["reply2"] == cpu.get("reply2")
        print(f"{name:9s} replies match the CPU: first {same1}, follow-up {same2}")
        if not same1:
            print("   cpu:", cpu.get("reply1", "")[:160].replace("\n", " "))
            print(f"   {name}:", r["reply1"][:160].replace("\n", " "))
        if r["reused2"] < 10:
            bad.append(f"{name}: the follow-up did not reuse the cached prefix")
    fb = res.get("fallback", {})
    if "error" in fb or fb.get("device") != "cpu" or not fb.get("why"):
        bad.append("the fallback did not fall back to the CPU with a reason")
    print("DEVICE CHECK PASS" if not bad else "DEVICE CHECK FAIL: " + "; ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
