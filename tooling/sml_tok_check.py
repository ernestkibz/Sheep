"""Does Sheep cut text into the same token ids as the model's own tokenizer? (2026-10-14)

  python tooling/sml_tok_check.py DIR [--ssf MODEL.ssf]

DIR holds the checkpoint's tokenizer files. The reference is the library the model ships
for: `tokenizers` for a tokenizer.json, `sentencepiece` for a tokenizer.model (run with the
venv-tf5 python, which has both). Compared on ~200 lines (English, accents, CJK, Arabic,
code, numbers, odd spacing, emoji): the builder's Python cut (sml_unigram.Unigram or
sml_common.Bpe) and, with --ssf, the engine's own (sml-cli --tokenize). Exact ids or fail.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LINES = [
    "Hello world", "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "  two leading spaces", "trailing spaces   ", "many     inner      spaces", "tab\tseparated\tvalues",
    "line one\nline two", "para one\n\npara two", "\n\nleading newlines", "ends with newline\n",
    "Café, naïve, façade, coöperate, résumé", "Straße in München", "Ångström Øresund æble",
    "¿Dónde está la biblioteca?", "Ça va? Très bien!", "Zürich ist schön", "Ελληνικά γράμματα",
    "Русский текст для проверки", "Українська мова", "日本語のテキストです。", "中文文本测试，标点符号。",
    "한국어 문장입니다", "العربية لغة جميلة", "עברית זה כאן", "हिन्दी पाठ परीक्षण", "ภาษาไทย ทดสอบ",
    "Tiếng Việt có dấu", "Kiswahili ni lugha nzuri sana", "Oluganda lulimi lwange",
    "12345 67890 3.14159 -42 1,000,000", "2026-10-14T10:30:00Z", "$19.99 €5 £3 ¥100",
    "def add(a, b):\n    return a + b\n", "for (int i = 0; i < n; i++) { x[i] = 0; }",
    "if x == 1 and y != 2: print('ok')", "<div class=\"a\">text</div>", "https://example.com/a?b=c&d=e",
    "user@example.com", "C:\\Users\\name\\file.txt", "snake_case and camelCase and kebab-case",
    "ALL CAPS SENTENCE HERE", "MiXeD cAsE tExT", "emoji 🙂 🚀 👍🏽 ❤️", "fullwidth ＡＢＣ １２３",
    "ligature ﬁ ﬂ and ½ ² ™", "a", "I", " ", "!!!", "...", "a-b_c.d/e\\f",
    "Sheep makes documents, sheets and slides on this computer.",
    "Write a short report on climate change in Uganda.", "Can you make a video ad for my bakery?",
]


def corpus():
    out = list(LINES)
    for f in ("USER_GUIDE.md", "README.md"):
        p = os.path.join(HERE, "..", f)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out += [ln.rstrip("\n") for ln in fh if ln.strip()][:80]
    return out[:200]


def reference(d):
    tik = [f for f in os.listdir(d) if f.endswith(".tiktoken") or f == "tiktoken.model"]
    if tik and not os.path.exists(os.path.join(d, "tokenizer.json")):
        import tiktoken
        from sml_hf import tiktoken_vocab
        (tokens, _, _, _), pret = tiktoken_vocab(d, tik[0])
        import base64
        with open(os.path.join(d, tik[0]), "rb") as f:
            ranks = {base64.b64decode(t): int(r) for t, r in (ln.split() for ln in f if ln.strip())}
        from sml_hf import _code_pattern, QWEN_PAT
        code = [f for f in os.listdir(d) if f.startswith("tokenization_") and f.endswith(".py")]
        pat = (_code_pattern(open(os.path.join(d, code[0]), encoding="utf-8").read()) if code else None) or QWEN_PAT
        enc = tiktoken.Encoding("ref", pat_str=pat, mergeable_ranks=ranks,
                                special_tokens={tokens[i]: i for i in pret["added"]})
        return lambda s: enc.encode(s, disallowed_special=()), "tiktoken"
    if os.path.exists(os.path.join(d, "tokenizer.json")):
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        return lambda s: tk.encode(s, add_special_tokens=False).ids, "tokenizers"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "tokenizer.model"))
    return lambda s: sp.encode(s), "sentencepiece"


def ours(d):
    from sml_hf import spm_vocab
    if any(f.endswith(".tiktoken") or f == "tiktoken.model" for f in os.listdir(d)):
        return None
    if os.path.exists(os.path.join(d, "tokenizer.json")):
        with open(os.path.join(d, "tokenizer.json"), encoding="utf-8") as f:
            tj = json.load(f)
        if tj["model"].get("type") == "Unigram":
            from sml_unigram import Unigram, from_tokenizer_json
            tokens, info = from_tokenizer_json(tj)
            return Unigram(tokens, info).encode
        return None              # a BPE tokenizer.json: only the engine cuts it (PRET), not the builder
    tokens, merges, _, _, uni = spm_vocab(d)
    if uni:
        from sml_unigram import Unigram
        return Unigram(tokens, uni).encode
    from sml_common import Bpe
    return Bpe(tokens, merges).encode


def vocab_only(d, out):
    """A .ssf holding the tokenizer and nothing else, read by the builder's own readers
    (what sml_quantize writes), so the engine's cut is checked without any weights."""
    import struct
    from sml_format import HEADER_FMT, HEADER_BYTES, MAGIC, VERSION, build_vocab_blob
    from sml_hf import HfWeights
    W = HfWeights.__new__(HfWeights)
    W.path, W.t = d, {}
    tokens, merges, bos, eos = W._vocab()
    blob = build_vocab_blob(tokens, merges, bos, eos, getattr(W, "_uni", None), getattr(W, "_pret", None))
    js = b"{}"
    tok_off = HEADER_BYTES
    json_off = tok_off + len(blob)
    table_off = json_off + len(js)
    zero = list(struct.unpack(HEADER_FMT, bytes(struct.calcsize(HEADER_FMT))))
    zero[0], zero[1], zero[7] = MAGIC, VERSION, len(tokens)
    names = ["n_tensors", "table_off", "json_off", "json_n", "tok_off", "tok_n", "group_off"]
    vals = [0, table_off, json_off, len(js), tok_off, len(blob), 0]
    from sml_format import MAX_LEVELS, REC_FMT, REC_SIZE
    zero[11], zero[12], zero[13] = 32, 1, 4     # group 32, one level of 4 bits: what the open checks want
    at = 13 + MAX_LEVELS                        # after magic .. level_bits
    vals[0] = 1                                 # and one tensor: a 1x1 f32 placeholder
    zero[at:at + len(names)] = vals
    head = struct.pack(HEADER_FMT, *zero)
    rec = list(struct.unpack(REC_FMT, bytes(REC_SIZE)))
    rec[0], rec[2], rec[3], rec[5], rec[6] = b"placeholder", 1, 1, table_off + REC_SIZE, 4
    with open(out, "wb") as f:
        f.write(head + bytes(HEADER_BYTES - len(head)) + blob + js + struct.pack(REC_FMT, *rec) + bytes(4))
    return out


def engine(ssf, lines):
    exe = os.path.join(HERE, "..", "sml", "sml-cli.exe")
    blob = "\x1e".join(lines)
    r = subprocess.run([exe, ssf, "--tokenize", "-"], input=blob.encode("utf-8"), capture_output=True, timeout=600)
    if r.returncode:
        raise SystemExit("sml-cli --tokenize failed: " + r.stderr.decode("utf-8", "replace")[-300:])
    rows = r.stdout.decode("utf-8").strip("\n").split("\n")
    return [[int(x) for x in row.split()] for row in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ssf")
    ap.add_argument("--engine", action="store_true", help="check the engine's cut on a tokenizer-only .ssf")
    a = ap.parse_args()
    lines = corpus()
    ref, lib = reference(a.dir)
    want = [ref(s) for s in lines]
    enc = ours(a.dir)
    got = [enc(s) for s in lines] if enc else want
    bad = [(s, w, g) for s, w, g in zip(lines, want, got) if w != g]
    name = os.path.basename(os.path.normpath(a.dir))
    print(f"{name}: builder cut {len(lines) - len(bad)}/{len(lines)} lines match {lib}" if enc else
          f"{name}: reference {lib}")
    if not enc and not a.ssf:
        a.engine = True
    for s, w, g in bad[:6]:
        print("  ", repr(s)[:70], "\n     want", w[:16], "\n     got ", g[:16])
    ok = not bad
    if a.engine and not a.ssf:
        import tempfile
        a.ssf = vocab_only(a.dir, os.path.join(tempfile.mkdtemp(prefix="tokcheck-"), "vocab.ssf"))
    if a.ssf:
        eng = engine(a.ssf, lines)
        bad2 = [(s, w, g) for s, w, g in zip(lines, want, eng) if w != g]
        print(f"  engine cut {len(lines) - len(bad2)}/{len(lines)} lines match")
        for s, w, g in bad2[:6]:
            print("  ", repr(s)[:70], "\n     want", w[:16], "\n     got ", g[:16])
        ok = ok and not bad2
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
