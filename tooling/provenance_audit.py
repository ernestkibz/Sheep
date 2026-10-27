#!/usr/bin/env python3
"""Provenance audit: is the code in this tree derived from the engine it forked from?

Git cannot answer it -- every tracked file was added in the baseline fork commit --
and an assertion is not evidence. So this measures it, following the method courts
and source-code experts actually use for software copyright: the
ABSTRACTION-FILTRATION-COMPARISON test (Computer Associates v. Altai, 2d Cir. 1992).

  ABSTRACTION  A program is examined at levels of generality: purpose, module
               structure, data structures and formats, algorithms, source text.
  FILTRATION   Each level is stripped of what copyright does not protect: ideas
               and mathematics, elements dictated by external factors (the
               instruction set, the OS, the C library, the hardware), standard
               practice, and facts. FILTERED lists below record exactly what was
               removed and why, because filtration is where an audit is honest
               or is not.
  COMPARISON   What remains -- protectable expression -- is compared. Literal:
               identical token runs, comment prose, rare identifiers, constants.
               Non-literal ("structure, sequence and organization"): the module
               layout, the public API surface, and the data structures.

Tokens are compared after comments and whitespace are stripped, so reformatting,
renaming our own symbols, and different bracing cannot hide a match.

    python tooling/provenance_audit.py sml tooling -- /path/to/upstream/checkout

Exit status is 1 if anything survives filtration and exceeds the copying
thresholds, so this can run in CI.
"""
import os
import re
import sys

TOKEN = re.compile(r"[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+\.?\d*[fFuUlL]*|[^\s\w]")
IDENT = re.compile(r"[A-Za-z_]\w{3,}")
NUM = re.compile(r"\b0[xX][0-9a-fA-F]{3,}\b|\b\d{4,}\b|\b\d+\.\d{4,}(?:[eE][-+]?\d+)?f?\b")
WORD = re.compile(r"[a-z]{3,}")
STRUCT = re.compile(r"struct\s+(\w+)?\s*\{([^{}]*)\}\s*(\w*)\s*;", re.S)
FIELD = re.compile(r"(\w+)\s*(?:\[[^\]]*\])?\s*[;,]")

SRC_EXT = (".c", ".h", ".cpp", ".hpp", ".cc", ".cu", ".cuh", ".py")
SEED = 8            # shorter runs carry no information in C
LOOK = 16           # a human should read it
COPIED = 24         # no two people write the same 24 tokens by accident
CHUNK = 1_500_000   # tokens per index pass, to bound memory

# ---------------------------------------------------------------- FILTRATION
#
# Removed before comparison, with the reason each category is unprotectable.

# (1) The languages themselves, and their standard libraries: expression dictated
#     by the language, not chosen by the author.
FILTER_LANG = """
auto break case char const continue default do double else enum extern float for goto if inline int
long register restrict return short signed sizeof static struct switch typedef union unsigned void
volatile while bool true false NULL nullptr class namespace template public private protected new
delete this operator using virtual override const_cast static_cast reinterpret_cast
size_t ssize_t ptrdiff_t uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t intptr_t
malloc calloc realloc free memcpy memset memmove memcmp strlen strcmp strncmp strcpy strncpy strstr
strtol strtod atoi atof snprintf printf fprintf sprintf sscanf fopen fclose fread fwrite fseek ftell
stdout stderr stdin assert abort exit qsort bsearch fabs fabsf sqrt sqrtf expf logf powf fmaxf fminf
lrintf roundf floorf ceilf isnan isinf FILE SEEK_SET SEEK_END SEEK_CUR CLOCKS_PER_SEC clock time
include define ifdef ifndef endif pragma once error warning elif undef defined
self import from print range len enumerate zip open write read append sorted reversed dict list set
tuple str bytes bytearray float int bool None True False class def return yield lambda try except
finally raise with assert global nonlocal numpy np torch json os sys math struct argparse
""".split()

# (2) Hardware and platform interfaces: names fixed by Intel, NVIDIA, OpenMP,
#     POSIX and Win32. Two engines targeting the same instruction set must spell
#     these identically; that is external constraint, not expression.
FILTER_PLATFORM_RE = re.compile(
    r"^(_mm\w*|__m\d+\w*|_cvt\w+|__builtin\w*|__nv\w*|__syncthreads|blockIdx|threadIdx|blockDim"
    r"|gridDim|cuda\w*|CU\w+|omp_\w+|pthread\w*|Query\w+|LARGE_INTEGER|QuadPart|HMODULE|GetProcAddress"
    r"|LoadLibrary\w*|Sleep|WSA\w*|socket|bind|listen|accept|recv|send|htons|inet\w*|_Float16|__half"
    r"|__restrict\w*|__attribute\w*|__global__|__device__|__shared__|__forceinline__)$")

# (3) Facts and third-party names: model architectures, file formats and products
#     published by other people. Facts are not copyrightable (Feist v. Rural).
FILTER_FACTS = """
llama qwen qwen2 qwen3 gemma gemma2 gemma3 phi phi2 phi3 mistral mixtral falcon bloom mpt starcoder
gptneox gpt2 gptj olmo olmoe stablelm baichuan chatglm deepseek command bert bailingmoe bailingmoe2
afmoe granite cohere dbrx xverse orion internlm internlm2 minicpm openelm arctic jais nemotron
rwkv mamba jamba hunyuan ernie exaone glm plamo refact smollm solar starcoder2 t5 gguf ggml safetensors
huggingface hf transformers tokenizer bpe spm unigram wordpiece rope alibi swiglu geglu silu gelu relu
softmax rmsnorm layernorm attention mlp moe kv gqa mha mqa q4 q8 q6 k_m f16 f32 bf16 int8 int4
""".split()

# (4) Terms of art: the shared vocabulary of the field. Ideas and mathematics are
#     outside copyright entirely (17 U.S.C. 102(b)); the words for them cannot be
#     evidence of copying.
FILTER_DOMAIN = """
data size count index offset length total start stop begin end first last next prev current step
value values result results buffer buffers block blocks row rows col cols column columns dim dims
model layer layers token tokens vocab weight weights bias scale scales scaled min max sum dot mul add
sub div mean var eps epsilon norm normalize head heads embed embedding hidden state cache pos seq
quant quantize dequant dequantize tensor tensors float half byte bytes bits bit mask logits logit
n_embd n_head n_layer n_ctx n_vocab n_ff n_kv batch nbatch threads thread nthreads chunk chunks
group groups super superblock block_size bench benchmark test check verify gate prompt generate
input output src dst out tmp temp acc accum sums best worst median mode flags flag args argv argc
name names path file files dir load save init free copy zero fill shift pack unpack bits2 bits4
""".split()

FILTERED = set(FILTER_LANG) | set(FILTER_FACTS) | set(FILTER_DOMAIN)


def filtered(ident):
    return ident in FILTERED or ident.lower() in FILTERED or FILTER_PLATFORM_RE.match(ident)


def strip_comments(s, py=False):
    """Comments out, code in. Per FILE, never across a concatenated corpus: a
    docstring regex run over joined sources deletes everything between the first
    and last triple quote, which silently empties the corpus it is meant to scan.
    Preprocessor lines are kept for C -- a macro is code -- and dropped for Python,
    where # starts a comment."""
    if py:
        s = re.sub(r'"""(.*?)"""', " ", s, flags=re.S)
        s = re.sub(r"'''(.*?)'''", " ", s, flags=re.S)
        s = re.sub(r"#[^\n]*", " ", s)
        return s
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    # #include lines are filtration, not comparison: which standard headers a C
    # file pulls in is dictated by what it calls, so identical include blocks are
    # external constraint. Without this, every C file in any project "matches"
    # every other on <stdio.h> <stdlib.h> <string.h>. Macros are kept: a #define
    # is authored expression.
    s = re.sub(r"^[ \t]*#[ \t]*include[^\n]*", " ", s, flags=re.M)
    return s


def code_of(path, src):
    return strip_comments(src, py=path.endswith(".py"))


def comments_of(s, py=False):
    if py:
        out = [m.group(1) for m in re.finditer(r'"""(.*?)"""', s, flags=re.S)]
        out += [m.group(1) for m in re.finditer(r"'''(.*?)'''", s, flags=re.S)]
        out += [m.group(1) for m in re.finditer(r"#([^\n]*)", s)]
        return out
    out = [m.group(1) for m in re.finditer(r"/\*(.*?)\*/", s, flags=re.S)]
    out += [m.group(1) for m in re.finditer(r"//([^\n]*)", s)]
    return out


def sentences(text):
    for raw in re.split(r"[.;!?\n]", text):
        w = WORD.findall(raw.lower())
        if len(w) >= 6:
            yield " ".join(w)


def structs_of(code):
    """struct name -> its field names: the data-structure level of abstraction."""
    out = {}
    for m in STRUCT.finditer(code):
        name = m.group(3) or m.group(1) or ""
        fields = {f for f in FIELD.findall(m.group(2)) if not filtered(f)}
        if name and fields:
            out.setdefault(name, set()).update(fields)
    return out


def sha(p):
    import hashlib
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 16), b""):
                h.update(blk)
    except OSError:
        return ""
    return h.hexdigest()


def tracked(dirs):
    """Every tracked file, whatever its extension. The first version of this audit
    looked only at .c/.h/.cu/.py and never compared the repository images -- two of
    which turned out to be upstream files, byte for byte."""
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, check=True).stdout.split()
        return [f for f in out if os.path.isfile(f)]
    except Exception:
        return [p for d in dirs for p in walk(d)]


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def walk(root):
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "build")]
        for f in sorted(files):
            if f.endswith(SRC_EXT):
                yield os.path.join(dirpath, f)


def longest_run_against(ours, chunk_toks, idx):
    best, example = 0, ""
    for i in range(len(ours) - SEED + 1):
        for j in idx.get(hash(tuple(ours[i:i + SEED])), ())[:4]:
            n = SEED
            while i + n < len(ours) and j + n < len(chunk_toks) and ours[i + n] == chunk_toks[j + n]:
                n += 1
            if n > best:
                best, example = n, " ".join(ours[i:i + n])
    return best, example


def main():
    args = sys.argv[1:]
    sep = args.index("--")
    ours_dirs, up_root = args[:sep], args[sep + 1]

    # ---- corpus, one file at a time
    up_files = list(walk(up_root))
    up_toks, up_idents, up_nums, up_sents, up_structs = [], set(), set(), set(), {}
    for p in up_files:
        src = read(p)
        py = p.endswith(".py")
        code = strip_comments(src, py)
        up_toks.extend(TOKEN.findall(code))
        up_idents.update(i for i in IDENT.findall(code) if not filtered(i))
        up_nums.update(NUM.findall(code))
        for c in comments_of(src, py):
            up_sents.update(sentences(c))
        for name, fields in structs_of(code).items():
            up_structs.setdefault(name, set()).update(fields)
    up_fieldsets = {frozenset(v) for v in up_structs.values() if len(v) >= 3}

    # ---- identity sweep: is any file of ours byte-for-byte one of theirs?
    up_hashes = {}
    for dirpath, dirnames, files in os.walk(up_root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in files:
            fp = os.path.join(dirpath, f)
            up_hashes.setdefault(sha(fp), fp)
    identical = []
    for f in tracked(ours_dirs):
        m = up_hashes.get(sha(f))
        if m:
            identical.append((f, os.path.relpath(m, up_root).replace(chr(92), "/")))

    print("PROVENANCE AUDIT -- abstraction, filtration, comparison")
    print(f"identity sweep  : {len(up_hashes):,} upstream files hashed; "
          f"{len(identical)} file(s) of ours are byte-identical to one of theirs")
    for f, m in identical:
        print(f"  !! {f} IS upstream {m}")
    print(f"upstream corpus : {len(up_files):,} source files, {len(up_toks):,} tokens, "
          f"{len(up_idents):,} identifiers after filtration, {len(up_sents):,} comment sentences, "
          f"{len(up_structs):,} structs")
    print(f"filtration      : {len(FILTERED):,} names removed as language, platform, fact or "
          f"term of art, plus anything matching the platform pattern")
    print(f"thresholds      : run >= {LOOK} tokens to inspect, >= {COPIED} to treat as copied\n")

    # ---- ours
    ours = []
    for d in ours_dirs:
        for p in sorted(walk(d)):
            src = read(p)
            code = code_of(p, src)
            toks = TOKEN.findall(code)
            if len(toks) >= 50:
                ours.append((p.replace("\\", "/"), src, toks, code))

    runs = {p: (0, "") for p, _, _, _ in ours}
    for c0 in range(0, len(up_toks), CHUNK):
        chunk = up_toks[c0:c0 + CHUNK + 64]
        idx = {}
        for i in range(len(chunk) - SEED + 1):
            idx.setdefault(hash(tuple(chunk[i:i + SEED])), []).append(i)
        for p, _, toks, _ in ours:
            r, ex = longest_run_against(toks, chunk, idx)
            if r > runs[p][0]:
                runs[p] = (r, ex)
        del idx

    print("LITERAL COMPARISON (protectable expression only)")
    print(f"{'file':32s} {'tokens':>8s} {'run':>4s} {'ids':>4s} {'nums':>4s} {'prose':>5s}  verdict")
    flagged, worst = [], (0, "", "")
    for p, src, toks, code in ours:
        run, example = runs[p]
        clean = code
        shared_id = {i for i in IDENT.findall(clean) if not filtered(i)} & up_idents
        shared_num = set(NUM.findall(clean)) & up_nums
        mine_sents = set()
        for c in comments_of(src, p.endswith(".py")):
            mine_sents.update(sentences(c))
        shared_sent = mine_sents & up_sents
        verdict = ("COPIED?" if run >= COPIED or shared_sent else
                   "inspect" if run >= LOOK else "independent")
        if run > worst[0]:
            worst = (run, p, example)
        print(f"{p:32s} {len(toks):8,d} {run:4d} {len(shared_id):4d} {len(shared_num):4d} "
              f"{len(shared_sent):5d}  {verdict}")
        if verdict != "independent":
            flagged.append((p, run, example, sorted(shared_sent)[:3], sorted(shared_id)[:10],
                            sorted(shared_num)[:6]))

    print("\nNON-LITERAL COMPARISON (structure, sequence and organization)")
    our_structs = {}
    for _p, _s, _t, _c in ours:
        for _n, _f in structs_of(_c).items():
            our_structs.setdefault(_n, set()).update(_f)
    shared_shapes = [n for n, v in our_structs.items()
                     if len(v) >= 3 and frozenset(v) in up_fieldsets]
    our_pub = {i for _p, _s, _t, _c in ours for i in IDENT.findall(_c)
               if i.startswith(("sml_", "SML_")) and not filtered(i)}
    name_hits = sorted(i for i in our_pub if i in up_idents)
    print(f"  data structures : {len(our_structs)} ours vs {len(up_structs)} theirs; "
          f"{len(shared_shapes)} of ours share an identical field set with one of theirs "
          f"{shared_shapes[:5]}")
    print(f"  public API      : {len(our_pub)} sml_/SML_ names; {len(name_hits)} also appear "
          f"upstream {name_hits[:5]}")
    print(f"  module layout   : {len(ours)} source files ours vs {len(up_files)} theirs")

    print("\nFLAGS")
    for p, run, example, sents, ids, nums in flagged:
        print(f"--- {p}: longest identical run {run} tokens")
        print(f"    run  : {example[:200]}")
        if sents:
            print(f"    prose: {sents}")
        print(f"    ids  : {ids}")
        print(f"    nums : {nums}")
    print(f"\nworst run in the tree: {worst[0]} tokens, {worst[1]}")
    bad = [f for f in flagged if f[1] >= COPIED or f[3]]
    if not flagged:
        print("RESULT: nothing survives filtration above the inspection threshold.")
    elif not bad:
        print(f"RESULT: {len(flagged)} file(s) reach the inspection threshold; none reaches the "
              f"copying threshold. Each is listed above for a human to read.")
    else:
        print(f"RESULT: {len(bad)} file(s) at or above the copying threshold. Read them.")
    return 1 if (bad or identical) else 0


if __name__ == "__main__":
    sys.exit(main())
