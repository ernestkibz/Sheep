# Provenance audit

Sheep began as a fork of an MIT-licensed C/C++ inference engine. Every file that
originated there has been deleted (`NOTICE` records which, and when). This document
answers what deletion alone cannot: **is the code that remains original?**

It is measured, the method is the one used in software copyright practice, and the
measurement is reproducible by anyone with the two repositories.

*This is an engineering audit, not legal advice. It gives a lawyer measurements to
work from rather than an assurance to rely on.*

## Why the git history proves nothing

Every tracked file — `sml/sml-kernel.c` included — was added in the baseline commit
`2241a21`. Commit dates therefore cannot separate what was forked from what was
written. Only the content can.

## Method: abstraction, filtration, comparison

`tooling/provenance_audit.py` implements the test US courts apply to software
copyright, from *Computer Associates v. Altai* (2d Cir. 1992):

1. **Abstraction** — examine the work at levels of generality: purpose, module
   layout, data structures, algorithms, source text.
2. **Filtration** — remove from each level what copyright does not protect:
   ideas and mathematics (17 U.S.C. §102(b)), expression dictated by external
   factors, standard practice, and facts (*Feist v. Rural*).
3. **Comparison** — compare only what survives, both literally and non-literally.

### What was filtered, and why

| filtered | reason |
|---|---|
| C, C++ and Python keywords, and their standard libraries | dictated by the language, not chosen |
| Intel intrinsics (`_mm256_*`, `__m256i`), CUDA, OpenMP, POSIX sockets, Win32 | fixed by the instruction set, the toolkit and the OS: two AVX2 engines must spell these identically |
| `#include` directives | which standard headers a file pulls in follows from what it calls; without this every C file "matches" every other on `<stdio.h> <stdlib.h> <string.h>` |
| model architecture names, format names, publisher names | facts about other people's models |
| terms of art (`quantize`, `tensor`, `rope`, `n_embd`, `logits`, …) | the vocabulary of the field, describing ideas that copyright does not reach |

416 names are filtered explicitly, plus anything matching the platform pattern.
The list is in the script, so the filtration can be argued with rather than trusted.

### Corpus

The **entire** upstream repository at `master`, not a sample:

    1,641 source files · 5,808,680 tokens · 88,653 identifiers after filtration
    23,489 comment sentences · 955 struct definitions

against our 46 source files in `sml/` and `tooling/`.

### Step 0: the identity sweep

Before any clever comparison, the blunt one. Every **tracked file of ours**,
whatever its extension, is hashed and looked up against a hash of every one of the
**2,827 files** in the upstream checkout:

    identity sweep : 2,827 upstream files hashed; 0 file(s) of ours are
                     byte-identical to one of theirs

This check exists because the first version of this audit did not have it. It
scanned `.c`, `.h`, `.cu` and `.py` only, and so never looked at the repository's
images — two of which, `brand/matmul.svg` and `brand/matmul.png`, were upstream's
`media/matmul.svg` and `media/matmul.png`, **byte for byte**, tracked here and
referenced by nothing. They are untracked as of 2026-09-23. A source-only audit
would never have found them; a reviewer opening the repository might have found
them first.

### Reproducing it

    git clone --depth 1 <upstream repo> /tmp/upstream
    python tooling/provenance_audit.py sml tooling -- /tmp/upstream

Exit status is 1 if anything exceeds the copying thresholds, so it can gate CI.

## Result 1: non-literal comparison (structure, sequence, organization)

This is the prong that matters for a fork, because renaming defeats a text search
but cannot hide a copied design.

| | Sheep | upstream | shared |
|---|---|---|---|
| data structures | 17 | 955 | **0** share an identical field set |
| public API names (`sml_*`, `SML_*`) | 196 | — | **0** appear upstream |
| source files | 46 | 1,641 | — |

Nothing of their structure survives in ours. The engine is a fixed decoder loop
over a format we defined; theirs is a general tensor-graph runtime.

## Result 2: literal comparison

**No file in this tree shares a single comment sentence with upstream** — out of
23,489 of theirs. Copied code carries its prose; none of theirs is here.

Longest identical token run, by file class:

| | longest run | what it is |
|---|---|---|
| all C and CUDA engine files | ≤ 21 tokens | `for (int i = 0; i < n; i++)`, `sizeof(float))`, `#pragma unroll for (int g = 0; g < 4; g++)` |
| `tooling/sml_common.py` | 12 tokens | (after the rewrite below) |
| `tooling/benchmark.py` | 33 tokens | the OpenAI chat-completions request body |
| `tooling/sml_hf.py` | 41 tokens | tensor names inside third-party checkpoints |

### The three findings, and what was done about each

**1. `sml_common.py` — GPT-2 byte-to-unicode table. Fixed.**
The construction `list(range(ord("!"), ord("~")+1)) + list(range(0xA1,0xAD)) + …`
appears in OpenAI's GPT-2 encoder and in every tokenizer that reads its
vocabularies. The *values* are forced — change the set and the merge table stops
matching — but the *expression* had been transcribed. It is now derived by asking
which Latin-1 bytes are printable and non-whitespace, and a test asserts the
resulting table is identical to the old one, entry for entry. Longest run fell
from 29 tokens to 12.

**2. `tooling/sml_hf.py` — checkpoint tensor names. Reviewed, kept, measured.**
These strings (`transformer.blocks.{bid}.attn.Wqkv`, …) are the keys *inside other
people's checkpoint files*. Spell them differently and the file cannot be read, so
they are facts under external constraint. The question is whether the *compilation*
— the selection and arrangement, the only part of a list copyright protects — was
taken. Measured against their `gguf-py/gguf/tensor_mapping.py`:

| | value |
|---|---|
| our patterns | 234 unique |
| theirs | 1,092 unique |
| strings appearing in both | 227 (97% of ours) |
| **longest identical *ordered* run** | **2** |

The individual strings are shared because both programs read the same files; the
selection (234 of their 1,092) and the arrangement are not theirs.

**3. `tooling/benchmark.py` — the OpenAI request body. Reviewed, kept.**
`{"messages":[{"role":"user","content":…}]}` is the wire format of a third-party
API. Any client that speaks it must emit those bytes; upstream's own tests contain
the same string for the same reason.

## Result 3: the files that are not source

Config, build, documentation, scripts and images carry provenance too, and the
first audit never looked at them. Compared line by line, and for prose sentence by
sentence, against 435 upstream non-source files (35,795 distinct lines, 3,498 prose
sentences):

| file | lines shared with upstream | verdict |
|---|---|---|
| `brand/matmul.svg`, `brand/matmul.png` | **100%** | **their artwork — untracked 2026-09-23** |
| `.gitignore` | was 90% | **rewritten** for our own build output; what remains is `*.o`, `__pycache__/` and similar, which have one spelling |
| `.editorconfig` | was 78% | **rewritten**; what remains are the format's own keywords |
| `LICENSE` | was 80% | **replaced** with the AGPL-3.0 text on 2026-10-30; no upstream file remains that needs the MIT notice |
| `README.md`, `METHOD.md`, `RESULTS.md`, `FORMAT.md`, `NOTICE` | 0–2% | **0 shared prose sentences** of their 3,498 |
| `sml/Makefile`, `tooling/*.sh`, `sml/ui/index.html` | 2–10% | generic shell and markup lines |
| `brand/sheep-logo.svg`, `brand/sheep-mark.svg` | **0 shared lines** | ours |

## Result 4: constants and vendored helpers

Numeric literals are now compared too, floating point included, because copied
mathematics travels as magic numbers. Everything shared is published mathematics
or a power of two:

| constant | what it is |
|---|---|
| `0.7978845608f`, `0.044715f` | √(2/π) and the coefficient of the published GELU tanh approximation. In 14 of their files, and in PyTorch, TensorFlow and JAX, because it is the definition |
| `0x8000`, `0x800`, `1024`, `4096`, `8192` | the IEEE half sign bit, a UTF-8 boundary, powers of two |
| `0.17677669529663687f` | our Hadamard scale, 1/√32 — **in zero upstream files** |

Two hand-checked sites where copying usually hides:

- **half-to-float conversion** (`sml-kernel.c`): hardware `_cvtsh_ss` where the CPU
  has it, otherwise an IEEE-754 field decode whose constants are the half format's
  own layout. Upstream instead builds a 65,536-entry lookup table; the approaches
  are not related.
- **the reciprocal square root seed** (`sml-gpu-fwd.cu`): carried the 1999
  bit-trick constant `0x5f3759df` — folklore from a game engine rather than anyone's
  property here, but it also appears upstream, so it is gone: the seed is now the
  hardware single-precision square root refined by two Newton steps. All three
  models still gate 5/5 against the CPU engine, and the path that uses it holds
  correlation 0.9999.

## A method failure, disclosed

This audit has been wrong twice, and both are recorded because an audit that hides
its own failures is a testimonial rather than a measurement.

**First**, it reported a clean result that was worthless: the comment stripper ran a
Python docstring regex across the *concatenated* corpus, so everything between the
first and last triple-quote — most of the C code — was deleted before comparison,
and 1,641 files were measured as 402,619 tokens instead of 5.8 million. Caught
because the token count was implausible for the file count. Stripping now happens
per file, language by language.

**Second**, it only ever looked at `.c`, `.h`, `.cu` and `.py`, and reported the tree
clean while two upstream images sat tracked in `brand/`. Caught by going through the
tracked files one at a time instead of trusting the tool's own scope. The identity
sweep above now covers every tracked file regardless of type, and its finding sets
the exit status, so the same blind spot cannot return quietly.

An audit that cannot show its own failures is a testimonial, not a measurement.

## Scope

This covers the source in `sml/` and `tooling/` against that one upstream project.
It does not cover the model weights: an `.ssf` built from someone else's checkpoint
carries that checkpoint's licence. Llama-derived files keep "Llama" at the start of
their name and ship with the Llama licence and its "Built with Llama" notice;
NVIDIA and other publishers impose their own terms, which travel the same way.

Every `.ssf` built from now on carries a `provenance` block in its JSON section:
checkpoint, repository id, licence id, the licence files found beside the weights,
and the duties that licence imposes on whoever ships the file. The licence comes
from the model card or `--license`; when neither exists the build says so loudly
and records `UNKNOWN` with "do not distribute until it is" rather than guessing.
Files built before 2026-09-23 lack the block and are rebuilt before distribution.

One licence fact that shapes the demo: **Qwen2.5-3B-Instruct is released under
`qwen-research`**, not Apache-2.0 (verified on its model card). It is fine for
research and measurement, which is what RESULTS.md does with it; it is not the
model to ship in a product.

`LICENSE` carries both copyright lines and `NOTICE` states the lineage plainly. A
documented history with none of the code is a stronger position than silence.
