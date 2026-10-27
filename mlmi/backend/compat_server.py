"""Compatibility mode: the maker's own code (transformers) on the original weights,
behind the same HTTP API as sml-server, so the rest of Sheep cannot tell the two apart.

    <runtime python> -m mlmi.backend.compat_server <originals folder> --port 8099 [--trust-remote-code]

Used when the engine has no fit for a model (tooling/sml_autofit.py), and where the
orchestrator measured it faster on the NVIDIA card than the native file on the CPU.
Slower than a Sheep file on the same device, but it computes what the maker wrote.
Runs under a PyTorch runtime (gpu_runtime: cuda or cpu), one request at a time.
"""
import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"model": None, "tok": None, "device": "cpu", "src": "", "turns": 0, "tokens_total": 0,
         "last_prompt": 0, "last_gen": 0, "last_prompt_ms": 0.0, "last_gen_ms": 0.0}
LOCK = threading.Lock()


def _free_ram():
    """Bytes of RAM free now (0 when it cannot be read)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
        m = MS()
        m.len = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.avail)
    return 0


def load(src, trust_remote_code=False):
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    # on the CPU, fp32 is exact and fastest; bf16 when fp32 (twice the bf16 files) would not fit
    stored = sum(f.stat().st_size for f in Path(src).glob("*.safetensors"))
    free = _free_ram()
    dtype = torch.bfloat16 if cuda or (free and stored * 2 > 0.8 * free) else torch.float32
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, trust_remote_code=trust_remote_code)
    if cuda:
        model = model.to("cuda")
    model.eval()
    STATE.update(model=model, tok=tok, device="cuda" if cuda else "cpu", src=src,
                 device_name=torch.cuda.get_device_name(0) if cuda else "CPU")


def _json_end(s, i):
    """Index after the JSON object starting at s[i] ('{'), or -1."""
    depth, q, esc = 0, False, False
    for j in range(i, len(s)):
        c = s[j]
        if q:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                q = False
            continue
        if c == '"':
            q = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def _call(obj):
    try:
        o = json.loads(obj)
    except ValueError:
        return None
    if not isinstance(o, dict) or not o.get("name"):
        return None
    args = o.get("arguments", o.get("parameters", {}))
    return {"name": str(o["name"]), "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}


def parse_reply(text, tools):
    """(content, thinking, calls) as sml-server's parse_reply splits them: <tool_call>
    blocks whatever the name; a bare JSON object only when it names an offered tool."""
    think = ""
    m = re.search(r"<think>(.*?)(</think>|$)", text, re.S)
    if m:
        think, text = m.group(1), text[:m.start()] + text[m.end():]
    calls, content = [], text
    if tools:
        parts, pos = [], 0
        for b in re.finditer(r"<tool_call>(.*?)(</tool_call>|$)", text, re.S):
            parts.append(text[pos:b.start()])
            pos = b.end()
            i = b.group(1).find("{")
            j = _json_end(b.group(1), i) if i >= 0 else -1
            c = _call(b.group(1)[i:j]) if j > 0 else None
            if c:
                calls.append(c)
        if not calls:
            # NVIDIA Nemotron's own form: <TOOLCALL>[{...}, {...}]</TOOLCALL>
            b = re.search(r"<TOOLCALL>(.*?)(</TOOLCALL>|$)", text, re.S)
            if b:
                body, k = b.group(1), 0
                while True:
                    i = body.find("{", k)
                    j = _json_end(body, i) if i >= 0 else -1
                    c = _call(body[i:j]) if j > 0 else None
                    if not c:
                        break
                    calls.append(c)
                    k = j
                if calls:
                    parts, pos = [text[:b.start()]], b.end()
        if calls:
            content = "".join(parts) + text[pos:]
        else:
            names = {str((t.get("function") or t).get("name", "")) for t in tools}
            q = text.lstrip()
            q = q[len("<|python_tag|>"):] if q.startswith("<|python_tag|>") else q
            if q.startswith("```"):
                q = q[q.find("{"):] if "{" in q else q
            while q.startswith("{"):
                j = _json_end(q, 0)
                c = _call(q[:j]) if j > 0 else None
                if not c or c["name"] not in names:
                    break
                calls.append(c)
                q = q[j:].lstrip(" \n;,`")
            if calls:
                content = q
    return content.strip(), think, calls


def _template_takes_tools():
    return "tools" in (getattr(STATE["tok"], "chat_template", None) or "")


def tools_block(tools):
    """sml-server's tools_block (ChatML): for a template that has no place for tools."""
    return ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
            + "\n</tools>\n\nFor each function call, return a json object with function name and "
              "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
              "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")


def render(body):
    tok = STATE["tok"]
    if body.get("prompt"):
        return body["prompt"]
    kw = {"tools": body["tools"]} if body.get("tools") else {}
    if kw and not _template_takes_tools():
        # the template would drop the tools without a word: carried in the system turn instead
        msgs = [dict(m) for m in body.get("messages") or []]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{msgs[0].get('content') or ''}\n\n{tools_block(kw['tools'])}"
        else:
            msgs.insert(0, {"role": "system", "content": tools_block(kw["tools"])})
        body, kw = dict(body, messages=msgs), {}
    try:
        return tok.apply_chat_template(body.get("messages") or [], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=bool(body.get("enable_thinking", False)), **kw)
    except Exception:
        return tok.apply_chat_template(body.get("messages") or [], add_generation_prompt=True, tokenize=False, **kw)


def generate(body, on_piece=None):
    """(text, n_prompt, n_gen, finish_len)."""
    import torch
    from transformers import TextIteratorStreamer
    tok, model = STATE["tok"], STATE["model"]
    prompt = render(body)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    temp = float(body.get("temperature", 0) or 0)
    max_new = int(body.get("max_tokens") or 768)
    gen = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
               do_sample=temp > 0, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    if temp > 0:
        gen.update(temperature=temp, top_p=float(body.get("top_p", 1.0) or 1.0))
    else:
        gen.update(temperature=None, top_p=None, top_k=None)
    t0 = time.perf_counter()
    if on_piece:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        out = {}

        def run():
            with torch.no_grad():
                out["seq"] = model.generate(**gen, streamer=streamer)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        for piece in streamer:
            if piece:
                on_piece(piece)
        th.join()
        seq = out["seq"]
    else:
        with torch.no_grad():
            seq = model.generate(**gen)
    new = seq[0, ids.shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    ms = (time.perf_counter() - t0) * 1000
    STATE.update(turns=STATE["turns"] + 1, tokens_total=STATE["tokens_total"] + ids.shape[1] + len(new),
                 last_prompt=int(ids.shape[1]), last_gen=int(len(new)), last_gen_ms=ms)
    return text, int(ids.shape[1]), int(len(new)), len(new) >= max_new


def _calls_json(calls, stream):
    return [dict({"index": i} if stream else {}, id=f"call_{i}", type="function",
                 function={"name": c["name"], "arguments": c["arguments"]}) for i, c in enumerate(calls)]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "engine": "compat"})
        if self.path == "/v1/template":
            tok = STATE["tok"]
            # a template with no place for tools is left to render() here, which adds them
            return self._send(200, {"template": (getattr(tok, "chat_template", None) or "") if _template_takes_tools() else "",
                                    "bos_token": tok.bos_token or "", "eos_token": tok.eos_token or ""})
        if self.path == "/stats":
            return self._send(200, {"model": STATE["src"], "device": f"compat ({STATE['device']})",
                                    "device_name": STATE.get("device_name", ""), "engine": "compat", "slots": 1,
                                    "turns": STATE["turns"], "tokens_total": STATE["tokens_total"],
                                    "last_prompt": STATE["last_prompt"], "last_gen": STATE["last_gen"],
                                    "last_gen_ms": round(STATE["last_gen_ms"], 1), "template": "model's own"})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": "compat", "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        tools = body.get("tools") or []
        stream = bool(body.get("stream"))
        with LOCK:
            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def ev(obj):
                        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
                        self.wfile.flush()
                    live = not tools
                    text, n_p, n_g, cut = generate(body, (lambda p: ev({"choices": [{"index": 0, "delta": {"content": p}}]}))
                                                   if live else None)
                    content, _think, calls = parse_reply(text, tools)
                    if not live and content:
                        ev({"choices": [{"index": 0, "delta": {"content": content}}]})
                    if calls:
                        ev({"choices": [{"index": 0, "delta": {"tool_calls": _calls_json(calls, True)}}]})
                    ev({"choices": [{"index": 0, "delta": {}, "finish_reason":
                                     "tool_calls" if calls else "length" if cut else "stop"}]})
                    ev({"usage": {"prompt_tokens": n_p, "completion_tokens": n_g}, "sml": {"engine": "compat"}})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                    return
                text, n_p, n_g, cut = generate(body)
            except Exception as e:
                if stream:
                    self.close_connection = True
                    return
                return self._send(500, {"error": f"{type(e).__name__}: {e}"[:400]})
        content, think, calls = parse_reply(text, tools)
        msg = {"role": "assistant", "content": content if (content or not calls) else None}
        if think:
            msg["reasoning_content"] = think
        if calls:
            msg["tool_calls"] = _calls_json(calls, False)
        self._send(200, {"id": "compat", "object": "chat.completion", "model": "compat",
                         "choices": [{"index": 0, "message": msg,
                                      "finish_reason": "tool_calls" if calls else "length" if cut else "stop"}],
                         "usage": {"prompt_tokens": n_p, "completion_tokens": n_g}, "sml": {"engine": "compat"}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()
    print(f"compat: loading {a.src}", flush=True)
    load(a.src, a.trust_remote_code)
    print(f"compat: ready on {STATE['device']}, port {a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
