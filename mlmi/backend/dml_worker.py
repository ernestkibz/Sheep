"""Sheep models on AMD and Intel GPUs, through DirectML.

Runs in Sheep's DirectML runtime (torch-directml), a separate Python from the
app's. The weights are the engine's own: sml-export.dll reconstructs each one
from the Sheep file with the engine's dequantizers and original column order,
and hands over the norms, biases and rotary tables it built. This file holds
them on the GPU in fp16 and runs the forward pass of the dense pre-norm
families (Llama, Mistral, Qwen2, Qwen3); the DLL refuses anything else by name.

As a device of sml-server (--device ext): it connects to the port the server
gave it and answers prompt and step requests with next-token logits, so the
server keeps its templates, tokenizer, tools, guard and CPU fallback.

    python dml_worker.py MODEL.ssf --connect PORT [--adapter N | --adapters 0:16,1:28] [--ctx N]
    python dml_worker.py MODEL.ssf --grade [--adapter N | --adapters ...]     # against the CPU engine

Several GPUs at once, of any brand (--adapters A:END,B:END...): layers below
END on adapter A, the next ones on B, and so on; the hidden state moves card to
card and each layer's cache stays on its card, so the model may be larger than
any one card. The orchestrator sizes the ranges from each card's memory.

Wire format (little-endian int32 unless stated):
    hello    magic 'SXEF', status (0 ready, 1 refused), n_vocab, len, utf-8 text
    request  op (1 prompt, 2 step, 3 bye), pos0, n, ids[n]
    reply    status (0 ok), then n_vocab float32 logits
    middle   (--layers A:B) op 4, pos0, n, then n x n_embd float32 hidden rows;
             reply status, then the rows after layers A..B-1
    front    (--layers 0:B) op 5, pos0, n, ids[n]; reply status, then the n rows
             after layers 0..B-1 (the CPU runs the rest)
"""
import argparse
import ctypes
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

DLL = Path(__file__).resolve().parent.parent.parent / "sml" / "sml-export.dll"
MAGIC = 0x46455853            # 'SXEF'
CHUNK = 256                   # prompt tokens per pass: bounds the attention scores' memory


class Refused(Exception):
    pass


class Engine:
    def __init__(self, model, n_ctx, adapter=0, split=None, layers=None):
        """split: [(adapter, layer_end), ...] for several GPUs; else one adapter.
        layers: (A, B) to hold only layers A..B-1 as a middle stage (no embedding
        or head); the split's layer ends are absolute layer numbers (the last
        card's end is B)."""
        import torch
        import torch_directml
        self.torch = torch
        self.split = split or [(adapter, None)]
        for a, _ in self.split:
            if a >= torch_directml.device_count():
                raise Refused(f"DirectML adapter {a} not found")
        devs = [torch_directml.device(a) for a, _ in self.split]
        self.dev = devs[0]
        self.dev_name = " + ".join(str(torch_directml.device_name(a)).rstrip("\x00") for a, _ in self.split)
        self._dll_dir = os.add_dll_directory(str(DLL.parent)) if os.name == "nt" else None
        lib = ctypes.CDLL(str(DLL))
        self.lib = lib
        fp = ctypes.POINTER(ctypes.c_float)
        lib.sx_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.sx_tensor.argtypes = [ctypes.c_int, ctypes.c_int, fp]
        lib.sx_vec.argtypes = [ctypes.c_int, ctypes.c_int, fp]
        lib.sx_rope.argtypes = [fp, fp]
        lib.sx_cpu_logits.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, fp]
        why = ctypes.create_string_buffer(300)
        if not lib.sx_open(str(model).encode("utf-8"), n_ctx, why, 300):
            raise Refused(why.value.decode("utf-8", "replace"))
        iv, fv = (ctypes.c_int * 16)(), (ctypes.c_float * 8)()
        lib.sx_facts(iv, fv)
        (self.n_layer, self.n_embd, self.n_head, self.n_kv, self.hd, self.n_ff, self.n_vocab, self.n_ctx,
         self.interleaved, self.half, self.tied, self.qkv_bias, self.qk_norm) = list(iv)[:13]
        self.eps, self.inv, emb_scale, res_scale, logit_scale = list(fv)[:5]
        self.emb_scale = emb_scale or 1.0
        self.res_scale = res_scale or 1.0
        self.logit_scale = logit_scale or 1.0
        t0 = time.time()
        # which card holds each layer
        self.l0, self.l1 = layers if layers else (0, self.n_layer)
        if not (0 <= self.l0 < self.l1 <= self.n_layer):
            raise Refused(f"layers {self.l0}..{self.l1} are outside the model's {self.n_layer}")
        ends = [e if e is not None else self.l1 for _, e in self.split]
        ends[-1] = self.l1
        self.ldev, k = [None] * self.n_layer, 0
        for l in range(self.l0, self.l1):
            while l >= ends[k]:
                k += 1
            self.ldev[l] = devs[k]
        self.layer_cards = [ends[i] - (ends[i - 1] if i else self.l0) for i in range(len(ends))]
        self.layers = [None] * self.n_layer
        for l in range(self.l0, self.l1):
            d = self.ldev[l]
            L = {kk: self.weight(l, i, d) for i, kk in enumerate(("q", "k", "v", "o", "gate", "up", "down"))}
            L["attn_norm"], L["ffn_norm"] = self.vec(l, 0, d), self.vec(l, 1, d)
            if self.qkv_bias:
                L["qb"], L["kb"], L["vb"] = self.vec(l, 2, d), self.vec(l, 3, d), self.vec(l, 4, d)
            if self.qk_norm:
                L["qn"], L["kn"] = self.vec(l, 5, d), self.vec(l, 6, d)
            self.layers[l] = L
        self.last = self.ldev[self.l1 - 1]
        self.middle = bool(layers)
        if self.middle and self.l0 == 0:
            self.emb = self.weight(-1, 8, self.dev)                # the front stage embeds the ids itself
        if not self.middle:
            self.emb = self.weight(-1, 8, self.dev)                # the embedding with the first layer
            if self.tied:
                self.head = self.emb if self.last is self.dev else self.weight(-1, 8, self.last)
            else:
                self.head = self.weight(-1, 7, self.last)          # the head with the last layer
            self.out_norm = self.vec(-1, 7, self.last)
        cos = np.empty((self.n_ctx, self.half), np.float32)
        sin = np.empty((self.n_ctx, self.half), np.float32)
        lib.sx_rope(cos.ctypes.data_as(fp), sin.ctypes.data_as(fp))
        self.rope_on = {}
        for d in devs:
            self.rope_on[id(d)] = (torch.from_numpy(cos).to(d), torch.from_numpy(sin).to(d))
        # one cache tensor per layer, written along its first dimension: a slice
        # assigned after an integer index ([l, a:b] of one 4-D tensor) came out
        # wrong on DirectML (top-1 3/8 against the CPU engine, 2026-10-05)
        shape = (self.n_ctx, self.n_kv, self.hd)
        self.kc = [torch.zeros(shape, dtype=torch.float16, device=self.ldev[l]) if self.ldev[l] is not None else None
                   for l in range(self.n_layer)]
        self.vc = [torch.zeros(shape, dtype=torch.float16, device=self.ldev[l]) if self.ldev[l] is not None else None
                   for l in range(self.n_layer)]
        self.load_s = time.time() - t0

    def weight(self, layer, kind, dev=None):
        rows, cols = ctypes.c_int(), ctypes.c_int()
        if not self.lib.sx_shape(layer, kind, ctypes.byref(rows), ctypes.byref(cols)):
            raise Refused(f"tensor {kind} of layer {layer} is missing")
        buf = np.empty((rows.value, cols.value), np.float32)
        self.lib.sx_tensor(layer, kind, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        t = self.torch.from_numpy(buf).to(self.torch.float16)
        del buf
        return t.to(dev if dev is not None else self.dev)

    def vec(self, layer, kind, dev=None):
        n = self.lib.sx_vec(layer, kind, None)
        buf = np.empty(max(1, n), np.float32)
        self.lib.sx_vec(layer, kind, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return self.torch.from_numpy(buf[:n]).to(dev if dev is not None else self.dev)

    def rms(self, x, w):
        return x * self.torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, t, c, s):
        torch, h = self.torch, self.half
        c, s = c[:, None, :], s[:, None, :]
        if self.interleaved:
            x0, x1 = t[..., 0:2 * h:2], t[..., 1:2 * h:2]
            r = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)
        else:
            x0, x1 = t[..., :h], t[..., h:2 * h]
            r = torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], -1)
        return torch.cat([r, t[..., 2 * h:]], -1) if 2 * h < t.shape[-1] else r

    def _pass(self, ids, pos0, rows=None):
        """ids from the embedding, or rows (n x n_embd float32) from the stage
        before; returns the logits, or the rows after this stage's layers."""
        torch = self.torch
        F = torch.nn.functional
        n = len(ids) if rows is None else rows.shape[0]
        H, KV, hd = self.n_head, self.n_kv, self.hd
        rep = H // KV
        if rows is None:
            idx = torch.tensor(ids, dtype=torch.int64).to(self.dev)
            x = torch.index_select(self.emb, 0, idx).float()
            if self.emb_scale != 1.0:
                x = x * self.emb_scale
        else:
            x = torch.from_numpy(rows).to(self.dev)
        T = pos0 + n
        mask_cpu = None
        if n > 1:
            j = torch.arange(T).view(1, T)
            i = torch.arange(pos0, T).view(n, 1)
            mask_cpu = torch.where(j > i, torch.tensor(float("-inf")), torch.tensor(0.0))
        here = None
        for l in range(self.l0, self.l1):
            L = self.layers[l]
            if self.ldev[l] is not here:              # the next card: hand the hidden state over
                here = self.ldev[l]
                if x.device != here:
                    x = x.cpu().to(here)
                cs, sn = self.rope_on[id(here)]
                c, s = cs[pos0:pos0 + n], sn[pos0:pos0 + n]
                mask = mask_cpu.to(here) if mask_cpu is not None else None
            h = self.rms(x, L["attn_norm"]).half()
            q, k, v = F.linear(h, L["q"]).float(), F.linear(h, L["k"]).float(), F.linear(h, L["v"]).float()
            if self.qkv_bias:
                q, k, v = q + L["qb"], k + L["kb"], v + L["vb"]
            q, k, v = q.view(n, H, hd), k.view(n, KV, hd), v.view(n, KV, hd)
            if self.qk_norm:
                q, k = self.rms(q, L["qn"]), self.rms(k, L["kn"])
            q, k = self.rope(q, c, s), self.rope(k, c, s)
            kc, vc = self.kc[l], self.vc[l]
            kc[pos0:T] = k.half()
            vc[pos0:T] = v.half()
            K = kc[:T].float().permute(1, 2, 0).unsqueeze(1)      # [KV, 1, hd, T]
            V = vc[:T].float().permute(1, 0, 2).unsqueeze(1)      # [KV, 1, T, hd]
            qg = q.view(n, KV, rep, hd).permute(1, 2, 0, 3)               # [KV, rep, n, hd]
            sc = torch.matmul(qg, K) * self.inv                            # [KV, rep, n, T]
            if mask is not None:
                sc = sc + mask
            o = torch.matmul(torch.softmax(sc, -1), V)                     # [KV, rep, n, hd]
            o = o.permute(2, 0, 1, 3).reshape(n, H * hd)
            x = x + F.linear(o.half(), L["o"]).float() * self.res_scale
            h = self.rms(x, L["ffn_norm"]).half()
            g, u = F.linear(h, L["gate"]).float(), F.linear(h, L["up"]).float()
            x = x + F.linear((F.silu(g) * u).half(), L["down"]).float() * self.res_scale
        if self.middle:
            return x.cpu().numpy().astype(np.float32)
        last = self.rms(x[-1:], self.out_norm).half()
        lg = F.linear(last, self.head).float()
        if self.logit_scale != 1.0:
            lg = lg / self.logit_scale
        return lg

    def forward(self, ids, pos0):
        """Read ids at positions pos0.. and return the last one's logits (numpy)."""
        if pos0 + len(ids) > self.n_ctx:
            raise ValueError("past the context")
        lg = None
        for a in range(0, len(ids), CHUNK):
            lg = self._pass(ids[a:a + CHUNK], pos0 + a)
        return lg.cpu().numpy().reshape(-1).astype(np.float32)

    def rows(self, rows, pos0):
        """The middle stage: hidden rows in, the rows after this stage's layers out."""
        if pos0 + rows.shape[0] > self.n_ctx:
            raise ValueError("past the context")
        out = []
        for a in range(0, rows.shape[0], CHUNK):
            out.append(self._pass(None, pos0 + a, rows[a:a + CHUNK]))
        return np.concatenate(out, axis=0)

    def front(self, ids, pos0):
        """The front stage (--layers 0:B): ids in, the rows after layers 0..B-1 out;
        the CPU runs the rest of a model larger than the GPUs."""
        if pos0 + len(ids) > self.n_ctx:
            raise ValueError("past the context")
        return np.concatenate([self._pass(ids[a:a + CHUNK], pos0 + a) for a in range(0, len(ids), CHUNK)], axis=0)

    def cpu_logits(self, ids):
        out = np.empty(self.n_vocab, np.float32)
        arr = (ctypes.c_int * len(ids))(*ids)
        self.lib.sx_cpu_logits(arr, len(ids), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("the server closed the connection")
        buf += part
    return bytes(buf)


def parse_split(text):
    """'0:16,1:28' -> [(0, 16), (1, 28)]; empty -> None (one adapter)."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        a, _, e = part.partition(":")
        out.append((int(a), int(e) if e else None))
    return out


def parse_range(text):
    """'14:20' -> (14, 20); empty -> None (the whole model)."""
    if not text:
        return None
    a, _, b = text.partition(":")
    return int(a), int(b)


def serve(args):
    sock = socket.create_connection(("127.0.0.1", args.connect))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        eng = Engine(args.model, args.ctx, args.adapter, parse_split(args.adapters), parse_range(args.layers))
    except Exception as e:      # refused, out of memory, no DirectML: the server stays on the CPU
        msg = str(e).encode("utf-8")[:400]
        sock.sendall(struct.pack("<4i", MAGIC, 1, 0, len(msg)) + msg)
        return 1
    msg = f"{eng.dev_name}".encode("utf-8")
    sock.sendall(struct.pack("<4i", MAGIC, 0, eng.n_vocab, len(msg)) + msg)
    while True:
        op, pos0, n = struct.unpack("<3i", recv_exact(sock, 12))
        if op == 3:
            return 0
        if op == 4:
            rows = np.frombuffer(recv_exact(sock, 4 * n * eng.n_embd), dtype=np.float32).reshape(n, eng.n_embd).copy()
            try:
                out = eng.rows(rows, pos0)
                sock.sendall(struct.pack("<i", 0) + out.tobytes())
            except Exception as e:
                print(f"dml worker: {e}", file=sys.stderr, flush=True)
                sock.sendall(struct.pack("<i", 1))
            continue
        ids = list(struct.unpack(f"<{n}i", recv_exact(sock, 4 * n))) if n > 0 else []
        if op == 5:
            try:
                out = eng.front(ids, pos0)
                sock.sendall(struct.pack("<i", 0) + out.astype(np.float32).tobytes())
            except Exception as e:
                print(f"dml worker: {e}", file=sys.stderr, flush=True)
                sock.sendall(struct.pack("<i", 1))
            continue
        try:
            lg = eng.forward(ids, pos0)
            sock.sendall(struct.pack("<i", 0) + lg.tobytes())
        except Exception as e:
            print(f"dml worker: {e}", file=sys.stderr, flush=True)
            sock.sendall(struct.pack("<i", 1))


def grade(args):
    """The gate: this device's logits against the CPU engine's, by prompt and by
    stepping (so the cache is exercised), with the speed of each."""
    eng = Engine(args.model, args.ctx, args.adapter, parse_split(args.adapters))
    print(f"loaded on {eng.dev_name} in {eng.load_s:.1f} s, layers per card {eng.layer_cards}")
    ids = [128000, 791, 6864, 315, 9822, 374, 12366, 13, 578, 6864, 315, 9635, 374]
    worst, agree, steps = 1.0, 0, 0
    t0 = time.time()
    lg = eng.forward(ids, 0)
    t_pp = time.time() - t0
    seq = list(ids)
    t_step = 0.0
    for k in range(8):
        ref = eng.cpu_logits(seq)
        corr = float(np.corrcoef(lg, ref)[0, 1])
        worst = min(worst, corr)
        agree += int(lg.argmax() == ref.argmax())
        steps += 1
        tok = int(ref.argmax())
        seq.append(tok)
        t0 = time.time()
        lg = eng.forward([tok], len(seq) - 1)
        t_step += time.time() - t0
    print(f"worst corr {worst:.6f}  top-1 agree {agree}/{steps}  prompt {len(ids) / t_pp:.1f} tok/s  "
          f"step {8 / t_step:.2f} tok/s")
    ok = worst >= 0.999 and agree == steps
    print("DML GRADE PASS" if ok else "DML GRADE FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--connect", type=int, default=0)
    ap.add_argument("--adapter", type=int, default=0)
    ap.add_argument("--adapters", default="", help="several GPUs: ADAPTER:LAYER_END,... e.g. 0:16,1:28")
    ap.add_argument("--layers", default="", help="a middle stage: hold only layers A:B")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args()
    if a.grade:
        return grade(a)
    return serve(a)


if __name__ == "__main__":
    sys.exit(main())
