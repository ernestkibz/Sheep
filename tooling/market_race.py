#!/usr/bin/env python3
"""Sheep against the inference engines on the market: one model, each engine's own 4-bit format.

    python tooling/market_race.py sheep:cpu llamacpp:cpu ollama:gpu ... [--runs 5] [--accuracy] [--out FILE]

Every engine gets the same chat: a ~512-token passage with a fresh first line each run (so no engine
answers from a prompt cache) and 128 new tokens at temperature 0. Speeds are the engine's OWN counters
(Sheep /stats, llama-server "timings", Ollama's eval counts, a streamer clock for the in-process ones);
load time, wall time, peak RAM of the engine's process tree and peak VRAM are measured from outside.
--accuracy also runs the 40 questions of accuracy_eval.py and Sheep's 8 canaries through the same chat.

Engines (each one's own format, made from the same original weights):
  sheep        SSF       sml-server, --orchestrator off, cache off
  llamacpp     GGUF      llama-server (REF_SERVER_CPU / REF_SERVER_CUDA)
  ollama       GGUF      ollama serve, a model created from the same GGUF (OLLAMA_MODEL)
  transformers safetensors bf16 on CPU, bitsandbytes 4-bit on GPU (run with the CUDA runtime's python)
  ortgenai     ONNX int4 onnxruntime-genai (ORT_MODEL_CPU / ORT_MODEL_GPU)
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

HOME = Path.home()
SSF = os.environ.get("SHEEP_SSF", str(HOME / ".mlmi" / "models" / "Qwen--Qwen2.5-3B-Instruct.ssf"))
GGUF = os.environ.get("RACE_GGUF", str(HOME / "models" / "race-b11901" / "qwen25-3b-q4km.gguf"))
ORIGINAL = os.environ.get("RACE_ORIGINAL", str(HOME / ".mlmi" / "originals" / "Qwen--Qwen2.5-3B-Instruct"))
SERVER_CPU = os.environ.get("REF_SERVER_CPU", str(HOME / "llama-cpu-b11901" / "llama-server.exe"))
SERVER_CUDA = os.environ.get("REF_SERVER_CUDA", str(HOME / "llama-cuda-b11901" / "llama-server.exe"))
OLLAMA = os.environ.get("OLLAMA_EXE", str(HOME / "ollama-race" / "ollama.exe"))
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", str(HOME / "ollama-race" / "models"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "race-qwen25-3b-q4km")
ORT_MODEL = {"cpu": os.environ.get("ORT_MODEL_CPU", ""), "gpu": os.environ.get("ORT_MODEL_GPU", "")}
THREADS, CTX, GEN = 6, 2048, 128

PASSAGE = (
    "The river town kept its market on the old stone bridge. Every morning the traders came before sunrise, "
    "carrying baskets of mangoes, dried fish, cloth dyed in deep blues, and clay pots still warm from the kiln. "
    "The bridge was narrow, so the town council had a rule: each trader could use two paces of stone, no more. "
    "Disputes were settled by the oldest weaver, who measured the paces with a knotted cord. "
)


def prompt(i):
    # a fresh first line per run: no engine may answer from a cached prefix
    text = f"Run {i}-{time.time_ns()}.\n" + PASSAGE * 7
    return [{"role": "user", "content": text + "\nRetell this story in detail, in at least 300 words."}]


# ------------------------------------------------------------------ measuring from outside
class Watch:
    """Peak RSS of a process tree and peak VRAM in use, sampled while a run goes; and how hard the
    hardware worked for it: the tree's CPU use (% of the whole CPU), the card's use and its power.
    vram0 is the card's use before the engine started (taken by race()), so the model's weights count."""

    def __init__(self, pid=None, vram0=None):
        self.pid, self.rss, self.vram, self.temp, self.go = pid, 0, 0, 0, True
        self.vram0 = gpu()[0] if vram0 is None else vram0
        self.cpu, self.util, self.watts = [], [], []
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def loop(self):
        import psutil
        procs = {}
        ncpu = psutil.cpu_count() or 1
        while self.go:
            try:
                p = psutil.Process(self.pid or os.getpid())
                tree = [p, *p.children(recursive=True)]
                self.rss = max(self.rss, sum(x.memory_info().rss for x in tree))
                busy = 0.0
                for x in tree:          # the first reading of a process is 0 by psutil's design
                    procs.setdefault(x.pid, x)
                    busy += procs[x.pid].cpu_percent(None)
                self.cpu.append(busy / ncpu)
            except Exception:
                pass
            used, temp, util, watts = gpu_full()
            self.vram, self.temp = max(self.vram, used - self.vram0), max(self.temp, temp)
            self.util.append(util)
            self.watts.append(watts)
            time.sleep(0.25)

    def stop(self):
        self.go = False
        self.t.join(2)
        avg = lambda xs: round(sum(xs[1:]) / max(1, len(xs) - 1), 1)
        return {"peak_ram_mb": round(self.rss / 2**20), "peak_vram_mb": max(0, round(self.vram)), "peak_gpu_c": self.temp,
                "avg_cpu_pct": avg(self.cpu), "avg_gpu_util": avg(self.util), "avg_gpu_w": avg(self.watts)}


def gpu_full():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        used, temp, util, watts = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else 0.0
        return num(used), int(num(temp)), num(util), num(watts)
    except Exception:
        return 0.0, 0, 0.0, 0.0


def gpu():
    used, temp, _, _ = gpu_full()
    return used, temp


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def wait_http(url, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"engine exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("engine did not come up")


# ------------------------------------------------------------------ the engines
class Http:
    """An engine behind an OpenAI-style chat endpoint, started as its own process."""
    name, fmt, file = "", "", ""
    model = None

    def __init__(self, device, port):
        self.device, self.port, self.proc = device, port, None
        self.url = f"http://127.0.0.1:{port}"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def pid(self):
        return self.proc.pid if self.proc else None

    def openai(self, messages, max_tokens, tools=None, temperature=0, extra=None):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": 1, **(extra or {})}
        if self.model:
            body["model"] = self.model
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        r = post(self.url + "/v1/chat/completions", body)
        return r, time.perf_counter() - t0


class Sheep(Http):
    name, fmt = "Sheep (SML)", "SSF"

    def start(self):
        self.file = SSF
        exe = ROOT / "sml" / "sml-server.exe"
        dev = {"cpu": "cpu", "gpu": "gpu", "auto": "gpu"}[self.device]
        self.proc = subprocess.Popen([str(exe), SSF, "--port", str(self.port), "--ctx", str(CTX), "--slots", "1",
                                      "--threads", os.environ.get("SHEEP_THREADS", str(THREADS)), "--device", dev,
                                      "--orchestrator", os.environ.get("SHEEP_ORCH", "off")],
                                     cwd=str(exe.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_http(self.url + "/health", self.proc)

    def chat(self, messages, max_tokens=GEN, tools=None, temperature=0):
        r, wall = self.openai(messages, max_tokens, tools, temperature, {"cache": False, "verify": False})
        s = get(self.url + "/stats")
        return r, wall, {"prompt_tokens": s.get("last_prompt"), "prompt_s": (s.get("last_prompt_ms") or 0) / 1000,
                         "gen_tokens": s.get("last_gen"), "gen_s": (s.get("last_gen_ms") or 0) / 1000}


class LlamaCpp(Http):
    name, fmt = "llama.cpp b11901", "GGUF Q4_K_M"

    def start(self):
        self.file = GGUF
        exe = SERVER_CUDA if self.device == "gpu" else SERVER_CPU
        args = [exe, "-m", GGUF, "--port", str(self.port), "-c", str(CTX), "-np", "1", "-t", str(THREADS), "--jinja"]
        args += {"gpu": ["-ngl", "99"], "auto": [], "cpu": ["-ngl", "0", "-dev", "none"]}[self.device]
        env = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": "-1"} if self.device == "cpu" else {}))
        self.proc = subprocess.Popen(args, cwd=os.path.dirname(exe), env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_http(self.url + "/health", self.proc)
        while True:                                   # /health answers 503 while the model loads
            try:
                if get(self.url + "/health").get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(0.3)

    def chat(self, messages, max_tokens=GEN, tools=None, temperature=0):
        r, wall = self.openai(messages, max_tokens, tools, temperature, {"cache_prompt": False})
        t = r.get("timings") or {}
        return r, wall, {"prompt_tokens": t.get("prompt_n"), "prompt_s": (t.get("prompt_ms") or 0) / 1000,
                         "gen_tokens": t.get("predicted_n"), "gen_s": (t.get("predicted_ms") or 0) / 1000}


class Ollama(Http):
    name, fmt = "Ollama", "GGUF Q4_K_M"
    model = OLLAMA_MODEL

    def start(self):
        self.file = GGUF
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{self.port}", OLLAMA_KEEP_ALIVE="30m", OLLAMA_MODELS=OLLAMA_MODELS,
                   **({"CUDA_VISIBLE_DEVICES": "-1"} if self.device == "cpu" else {}))
        self.proc = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_http(self.url + "/api/version", self.proc)
        self.native([{"role": "user", "content": "hi"}], 1)      # load the model now: load time belongs to start

    def options(self, max_tokens, temperature):
        gpu = {"gpu": {"num_gpu": 99}, "cpu": {"num_gpu": 0}, "auto": {}}[self.device]
        return {"num_ctx": CTX, "num_thread": THREADS, **gpu,
                "num_predict": max_tokens, "temperature": temperature, "seed": 1}

    def native(self, messages, max_tokens, temperature=0):
        t0 = time.perf_counter()
        r = post(self.url + "/api/chat", {"model": self.model, "messages": messages, "stream": False,
                                          "options": self.options(max_tokens, temperature)})
        return r, time.perf_counter() - t0

    def chat(self, messages, max_tokens=GEN, tools=None, temperature=0):
        if tools:                                   # tools through the OpenAI door, timings through the native one
            r, wall = self.openai(messages, max_tokens, tools, temperature)
            return r, wall, {}
        r, wall = self.native(messages, max_tokens, temperature)
        out = {"choices": [{"message": r.get("message", {}), "finish_reason": r.get("done_reason")}]}
        return out, wall, {"prompt_tokens": r.get("prompt_eval_count"), "prompt_s": (r.get("prompt_eval_duration") or 0) / 1e9,
                           "gen_tokens": r.get("eval_count"), "gen_s": (r.get("eval_duration") or 0) / 1e9}


class InProcess:
    """An engine driven from Python in this process; the clock is a streamer's first token."""
    model, proc = None, None

    def __init__(self, device, port):
        self.device = device

    def pid(self):
        return os.getpid()

    def stop(self):
        pass


class Transformers(InProcess):
    name = "HF transformers"

    def start(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ORIGINAL)
        if self.device == "gpu":
            from transformers import BitsAndBytesConfig
            self.fmt = "safetensors, bnb 4-bit"
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            self.m = AutoModelForCausalLM.from_pretrained(ORIGINAL, quantization_config=q, device_map="cuda:0")
        else:
            self.fmt = "safetensors bf16"
            torch.set_num_threads(THREADS)
            self.m = AutoModelForCausalLM.from_pretrained(ORIGINAL, torch_dtype=torch.bfloat16)
        self.file = ORIGINAL

    def chat(self, messages, max_tokens=GEN, tools=None, temperature=0):
        from transformers import TextIteratorStreamer
        ids = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self.m.device)
        st = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        out = {}
        t0 = time.perf_counter()
        th = threading.Thread(target=lambda: out.setdefault("ids", self.m.generate(
            ids, max_new_tokens=max_tokens, do_sample=False, streamer=st)))
        th.start()
        first, text = None, ""
        for piece in st:
            if first is None and piece:
                first = time.perf_counter()
            text += piece
        th.join()
        end = time.perf_counter()
        n = int(out["ids"].shape[1] - ids.shape[1])
        first = first or end
        r = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "length" if n >= max_tokens else "stop"}]}
        return r, end - t0, {"prompt_tokens": int(ids.shape[1]), "prompt_s": first - t0, "gen_tokens": n, "gen_s": end - first}


class OrtGenai(InProcess):
    name, fmt = "ONNX Runtime GenAI", "ONNX int4"

    def start(self):
        import onnxruntime_genai as og
        self.og = og
        self.file = ORT_MODEL[self.device]
        self.m = og.Model(self.file)
        self.tok = og.Tokenizer(self.m)
        from transformers import AutoTokenizer
        self.tmpl = AutoTokenizer.from_pretrained(ORIGINAL)

    def chat(self, messages, max_tokens=GEN, tools=None, temperature=0):
        og = self.og
        text_in = self.tmpl.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = self.tok.encode(text_in)
        p = og.GeneratorParams(self.m)
        p.set_search_options(max_length=len(ids) + max_tokens, do_sample=False)
        g = og.Generator(self.m, p)
        t0 = time.perf_counter()
        g.append_tokens(ids)
        first, out = None, []
        while not g.is_done() and len(out) < max_tokens:
            g.generate_next_token()
            if first is None:
                first = time.perf_counter()
            out.append(int(g.get_next_tokens()[0]))
        end = time.perf_counter()
        text = self.tok.decode(out)
        r = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "length" if len(out) >= max_tokens else "stop"}]}
        return r, end - t0, {"prompt_tokens": len(ids), "prompt_s": (first or end) - t0, "gen_tokens": len(out),
                             "gen_s": end - (first or end)}


ENGINES = {"sheep": Sheep, "llamacpp": LlamaCpp, "ollama": Ollama, "transformers": Transformers, "ortgenai": OrtGenai}


# ------------------------------------------------------------------ quality through the same chat
class AsLocal:
    """What run_canary expects of Sheep's Local client, on any engine."""

    def __init__(self, eng):
        self.eng = eng

    def chat(self, msgs, tools=None, max_tokens=256, **_):
        r, _, _ = self.eng.chat(msgs, max_tokens, tools)
        c = r["choices"][0]
        msg = c.get("message") or {}
        calls = msg.get("tool_calls") or []
        for x in calls:                              # some servers send arguments as an object
            f = x.get("function") or {}
            if isinstance(f.get("arguments"), dict):
                f["arguments"] = json.dumps(f["arguments"])
        return {"content": msg.get("content") or "", "tool_calls": calls}, {"finish": c.get("finish_reason", "")}


def accuracy(eng):
    import accuracy_eval as A
    right, by = 0, {}
    for kind, q, answers in A.CASES:
        r, _, _ = eng.chat([{"role": "user", "content": q + "\n\nThink briefly, then end with the final answer on its own line."}], 320)
        content = (r["choices"][0]["message"].get("content") or "").strip()
        lines = [ln for ln in content.splitlines() if ln.strip()]
        ok = A.correct(" ".join(lines[-2:]) if lines else "", answers)
        right += ok
        by.setdefault(kind, [0, 0])
        by[kind][0] += ok
        by[kind][1] += 1
    return {"right": right, "of": len(A.CASES), "by_class": by}


def canaries(eng):
    from mlmi.backend.orchestrator import CANARIES, run_canary
    rows = [run_canary(AsLocal(eng), *c) for c in CANARIES]
    return {"passed": sum(r["ok"] for r in rows), "of": len(rows),
            "failed": {r["name"]: r["why"] for r in rows if not r["ok"]}}


# ------------------------------------------------------------------ one engine, one device
def race(spec, runs, port, quality):
    kind, device = (spec.split(":") + ["cpu"])[:2]
    eng = ENGINES[kind](device, port)
    row = {"engine": eng.name, "kind": kind, "device": device, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    vram0 = gpu()[0]
    t0 = time.perf_counter()
    try:
        eng.start()
    except Exception as e:
        eng.stop()
        return {**row, "error": f"could not start: {str(e)[:300]}"}
    row["load_s"] = round(time.perf_counter() - t0, 2)
    row["format"] = eng.fmt
    f = Path(eng.file)
    row["size_mb"] = round((sum(x.stat().st_size for x in f.rglob("*") if x.is_file()) if f.is_dir() else f.stat().st_size) / 2**20)
    w = Watch(eng.pid(), vram0)
    try:
        r, _, _ = eng.chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], 16)
        row["sanity"] = (r["choices"][0]["message"].get("content") or "")[:60]
        eng.chat(prompt(-1), 16)                                  # warm-up, not counted
        pp, tg, wall, ptok, gtok = [], [], [], [], []
        for i in range(runs):
            _, w_s, t = eng.chat(prompt(i), GEN)
            if t.get("prompt_s"):
                pp.append(t["prompt_tokens"] / t["prompt_s"])
                ptok.append(t["prompt_tokens"])
            if t.get("gen_s") and t.get("gen_tokens", 0) > 1:
                tg.append(t["gen_tokens"] / t["gen_s"])
                gtok.append(t["gen_tokens"])
            wall.append(w_s)
        med = lambda xs: round(statistics.median(xs), 2) if xs else None
        row.update({"prompt_tok_s": med(pp), "gen_tok_s": med(tg), "wall_s": med(wall),
                    "prompt_tokens": med(ptok), "gen_tokens": med(gtok), "runs": runs,
                    "prompt_runs": [round(x, 1) for x in pp], "gen_runs": [round(x, 2) for x in tg]})
        if quality:
            row["accuracy"] = accuracy(eng)
            row["canaries"] = canaries(eng)
    except Exception as e:
        row["error"] = f"failed mid-run: {str(e)[:300]}"
    finally:
        row.update(w.stop())
        eng.stop()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="+", help="kind:device, e.g. sheep:cpu llamacpp:gpu")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--accuracy", action="store_true", help="also the 40 accuracy questions and the 8 canaries")
    ap.add_argument("--port", type=int, default=8171)
    ap.add_argument("--cool", type=int, default=0, help="before a GPU engine, wait until the GPU is at or below this")
    ap.add_argument("--out", default=str(HERE / "market_race.jsonl"))
    a = ap.parse_args()
    for spec in a.engines:
        if a.cool and spec.endswith(":gpu"):
            t0 = time.time()
            while gpu()[1] > a.cool and time.time() - t0 < 600:
                time.sleep(5)
        row = race(spec, a.runs, a.port, a.accuracy)
        print(json.dumps(row), flush=True)
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
