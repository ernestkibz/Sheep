# The engine: SML and the SSF format

Sheep runs local models in its own engine, **SML**, from its own file format, **SSF**. This page is the
summary; the full record is in [METHOD.md](../METHOD.md) (how each result was measured),
[sml/FORMAT.md](../sml/FORMAT.md) (the file format, byte by byte), [RESULTS.md](../RESULTS.md) (every
number, including what did not work) and [evidence/ENGINE_SPRINTS.md](evidence/ENGINE_SPRINTS.md) (the
improvement sprints).

## Races against llama.cpp

Raced against llama.cpp Q4_K_M on three models chosen to cover three distinct
architecture specs. Held-out tokens, every position scored, the reference decoded
from **their own file** — because comparing across token sets flatters whoever drew
the easier one. All races: equal threads, ABBA rounds, bootstrap 95% interval.

The CPU and GPU race tools append one JSONL record per workload to
`%USERPROFILE%\.mlmi\benchmark-runs.jsonl` (or `~/.mlmi/benchmark-runs.jsonl`
on Linux/macOS). Records include model/runtime file identity, host details,
settings, valid and excluded rounds, correctness checks, confidence intervals,
verdicts, and GPU/thermal metadata when available. Use `--output PATH` to select
another log and `--notes TEXT` to record workload or thermal conditions. File
identity uses path, size, and modification time; retain the exact artifacts for
stronger bit-for-bit provenance. These records supplement rather than rewrite
the checked-in historical results below.

### Quality, size, and decode (August 2026 build)

| spec exercised | model | quality | decode (Aug 2026 build) | file size |
|---|---|---|---|---|
| RMSNorm / halved rotary / GQA 16:2 / bias | Qwen2.5-3B | **23% better** | **1.144x** | 1.0225x |
| RMSNorm / halved rotary / GQA 32:8 / no bias | Llama-3.2-1B | **41% better** | **1.034x** | 1.0168x |
| LayerNorm / partial rotary / parallel residual | pythia-410m | **12.4% better** | **1.65x** | **0.9981x** |

### CPU and GPU races vs b11177 (Sep 2026)

**Quality, size, and decode on both CPU and GPU win on all three against the newest reference build.**
Raced on 2026-09-25 against llama.cpp b11177 (published that morning), same thread count
(6 each), warm, 128 generated tokens (64 on the 3B), rounds alternated ABBA, verdict from a
bootstrap 95% interval (tooling/sml_cpu_race_ci.py):

| CPU decode vs b11177 | ratio (median) | 95% CI | rounds won |
|---|---|---|---|
| Llama-3.2-1B | **1.056x** | [1.038, 1.080] | 8/8 |
| Qwen2.5-3B | **1.044x** | [1.026, 1.147] | 6/6 |
| pythia-410m | **1.164x** | [1.094, 1.275] | 8/8 |

Re-raced 2026-09-28 with the GPU powered off (a cooler laptop): 1B **1.058x**
[1.023, 1.096] 8/10, 3B **1.079x** [1.069, 1.094] 10/10, pythia **1.159x**
[1.119, 1.197] 6/6.

**GPU decode wins on all three against b11177 too** (sml_gpu_race_ci.py --decode, 64
greedy tokens, GPU top-1 checked against the CPU engine every round, both engines at
the laptop's thermal floor, 86-93 C):

| GPU decode vs b11177 | ratio (median) | 95% CI | rounds won |
|---|---|---|---|
| Llama-3.2-1B | **1.520x** | [1.517, 1.528] | 6/6 |
| Qwen2.5-3B | **1.469x** | [1.468, 1.473] | 6/6 |
| pythia-410m | **1.169x** | [1.157, 1.183] | 8/8 |

GPU prompt reading also wins against b11177, on all three models (sml_gpu_race_ci.py,
2026-09-26, after the kept-tensor matmul became a 128x128 cp.async tile):

| GPU prompt vs b11177 | 128 tokens | 512 tokens |
|---|---|---|
| Llama-3.2-1B | **1.479x** [1.471, 1.487], 6/6 | **1.289x** [1.282, 1.300], 9/10 |
| Qwen2.5-3B | **1.345x** [1.333, 1.354], 10/12 | **1.304x** [1.299, 1.306], 12/12 |
| pythia-410m | **1.513x** [1.413, 1.658], 6/6 | **1.365x** [1.358, 1.436], 6/6 |

**CPU prompt reading wins on all three against b11177** (sml_cpu_race_ci.py, same threads,
6 rounds each, every round won), after 6-bit tensors moved onto the rows-in-lanes kernel
(006966e; they had been read in two passes at a fifth of the 4-bit rate):

| CPU prompt vs b11177 | 28 tokens | 128 tokens |
|---|---|---|
| Llama-3.2-1B | **1.355x** [1.181, 1.386] | **1.519x** [1.422, 1.606] |
| Qwen2.5-3B | **1.510x** [1.418, 1.688] | **1.703x** [1.581, 1.873] |
| pythia-410m | **1.474x** [1.214, 1.690] | **1.677x** [1.620, 1.750] |

With bits re-placed by measured damage, all
three files are also **smaller** than the reference Q4_K_M files while still ahead on
quality -- 1B 0.9947x the size at 37% better, 3B 0.9986x at 24% better, pythia 0.9981x at
12.4% better. Against the August build the re-placed files decoded faster on CPU: 3B **1.20x** (95% CI 1.14-1.24), 1B **1.12x** (95% CI 1.07-1.15), pythia **1.65x** (95% CI 1.57-1.70).

*(History, 2026-09-18.)* GPU decode won on all three against the reference CUDA build -- 1.45x and 1.60x (1B), 1.45x and 1.44x (3B) in mirrored back-to-back rounds, and pythia 1.19x median winning 4 of 5 paired runs -- with both engines at the same 87-91 C, and every answer graded 5/5 top-1 against the CPU engine. Read the caveat with the number: that race was run at the laptop's thermal floor, where both engines are clamped, because it is the state this machine actually sits in.

**Prompt reading against the older build b10883 (kept as history; the b11177 figures above are current).** Every cell is the median of 8-10 interleaved rounds with a bootstrap 95% interval (tooling/sml_gpu_race_ci.py, sml_cpu_race_ci.py); "won" means the whole interval is above 1. An earlier CPU "win" gave this engine 12 threads to the reference's 6 -- at equal threads it is gone, and it is withdrawn here. Same session, mirrored, against reference build b10883 (the GPU rows re-raced against b11177 hold, above):

| prompt reading | model | 28 tokens | 128 tokens | 512 tokens |
|---|---|---|---|---|
| CPU (warm, same thread count, 95% CI) | Llama-3.2-1B | 1.02x, undecided | 0.94x, **lost** | |
| | Qwen2.5-3B | 1.19x, undecided | 1.00x, undecided | |
| GPU (int8 tensor cores, 95% CI) | Llama-3.2-1B | | **1.21x** [1.19, 1.32] | **1.10x** [1.05, 1.13] |
| | pythia-410m | | **1.39x** [1.26, 1.64] | **1.18x** [1.16, 1.27] |
| | Qwen2.5-3B | | 0.89x [0.88, 0.89], lost | 0.87x, lost |

The GPU path is graded against full precision like the CPU engine and matches it (Llama-1B 0.2381 vs 0.2373 rel-RMS; pythia 0.2127 vs 0.2087). The exception is the 3B on the GPU at 128 tokens: its int8 copy does not fit beside its 4-bit copy on this 4 GB card, so it is re-made on every call, and 128 tokens are too few to spread that over. Caveat for every GPU number: under load this laptop's GPU runs at 210 MHz (thermal slowdown), with both engines throttled alike.

**Batched serving: several conversations on one read of the weights.** Each conversation
keeps its own KV cache; every generation step reads the weights once for all of them
(sml_forward_multi, sml-batched). Raced against the reference's own llama-batched-bench,
Llama-3.2-1B, 128-token prompts and 128 generated tokens per conversation, 6 threads each,
10 interleaved rounds (tooling/sml_batched_race_ci.py):

| conversations | generation | total (prompts + generation) |
|---|---|---|
| 1 | **1.081x** [1.029, 1.125] | **1.168x** [1.109, 1.201] |
| 2 | **1.064x** [1.052, 1.087], 8/8 | **1.141x** [1.122, 1.166], 8/8 |
| 4 | **1.133x** [1.018, 1.145], 8/10 | **1.234x** [1.145, 1.257], 10/10 |
| 8 | **1.322x** [1.278, 1.363], 10/10 | **1.366x** [1.316, 1.402], 10/10 |

The row for two conversations is from 2026-09-26 (792867f): at two, each conversation
now runs through the decode kernel itself on 8-row tiles, the second from L1, which
streamed faster than the packed copy (it had been a tie, 1.039x) -- and is bit-identical
to decoding each conversation alone. From three up the batch uses a packed copy and the
prompt kernel's per-256 activation scaling (as the reference does for every token);
SML_MULTI=exact makes every batch size bit-identical.

**Four and eight conversations, 2026-09-28 (the rows above):** at four slots the packed kernel was short of instructions, not bandwidth (the 6-bit head ran at 84-87 GMAC/s, 30% of a step) and generation had slipped to 0.953x. 6-bit tiles in the packed copy now hold each code whole -- one load instead of a nibble and a 2-bit plane reassembled -- for 30% more bytes in a copy that exists only for batched serving; the sums are the same.

**The serving layer adds what the reference server does not have** (tooling/sml_orch_race.py,
the same 1B model on both sides, 6 threads, greedy). An exact repeat of a question is
answered from a response cache: the second request took **1.1 ms against 1721.5 ms**
for llama-server with its prompt cache (an earlier run: 12.4 vs 1698.6 ms), the same answer
on both. Arithmetic is computed, not generated: **20/20** against **11/20** on ten products
and ten sums. Neither makes the model better; they are what the server does around it.

**Vision runs locally.** A vision tower (SmolVLM-256M's) reproduces the reference's image features to 1.5e-6 relative, and the model answers "What color is the circle?" with "The circle is red." -- word for word what the full-precision reference says. CPU only, one 512 x 512 view per image, the tower in full precision.

### Scope of that claim, stated so nobody has to infer it

Three models, one laptop (Ryzen 5 5600H, DDR4-3200, NVIDIA GeForce RTX 3050 Laptop GPU), one
competitor, one of its quantization formats, **batch-1 decode** on its CPU and its
GPU, and prompt reading as tabled above. Not batched serving, not their other
formats, not their other backends.

**The speed result is AVX2-specific.** The fast kernels are x86 AVX2; there is
a portable scalar path that is correct (it agrees with the format spec at
~1e-6 and generates identical text) but measures **~2.5x slower** — 26.7 tok/s
against 67.5 on pythia. The reference engine ships hand-written NEON, Metal and
CUDA backends, so on anything that is not an AVX2 x86 part, assume we lose.
Nothing here has been measured on ARM at all.

The speed margin ranges **1.034x to 1.503x** and is a property of the model
shape, not of the engine — quote it per model, never averaged. On the largest
model raced it is 1.144x, and decode here is bandwidth-capped: 24.2 of 28.4
GB/s usable, so ~1.17x is the ceiling without reading less of the model.

Full numbers, including everything that did not work and the thirteen
measurements that faked a result, are in [RESULTS.md](../RESULTS.md).

## What is actually different

Quality is where the margin is largest, and it comes from three things the
reference format structurally cannot express:

1. **A Hessian.** Layer-sequential error feedback with `H = XᵀX` from real
   calibration activations, so each column's rounding is compensated in the ones
   that follow.
2. **An amplification pass.** Quantize one layer, hold the rest exact, measure
   the damage to the logits. Per-layer damage spreads **4.54x** — and the "every
   other layer" heuristic both engines shipped overlaps the measured top 18 by
   **3 of 18**.
3. **A per-tensor width field.** So the measurement can be acted on. Bits placed
   by measurement beat the fixed recipe by **1.28x at identical bytes**.

A hand-tuned recipe is the same for every model ever released. A measured
allocation is per-model, and it needs all three of the above to exist.

## Layout, and the ladder

SSF v8 makes storage layout a **per-tensor** choice:

- **flat** — the shipping default. One 32-byte load carries two whole groups.
- **ladder** — level-major 2-bit planes, so a *prefix* of the tensor is a coarser
  read of the whole thing and `--depth` becomes a runtime dial.

The ladder is kept because that dial is real, but it is not the default and the
reason is recorded: forcing reconstruction to be linear in the code cost
**1.552x of quality and ~2x of per-thread throughput**, on every tensor,
including the overwhelming majority never read at two depths. Sixteen earlier
mechanisms failed because they improved arithmetic sitting on top of that.

```
sml/        the engine: kernel, loader, forward pass, tokenizer, server, CLI
tooling/    build a model, and the gates that decide whether it is good
brand/      logo and marks
```

## Gates

Correctness is checked before speed, and a configuration that fails correctness
has its throughput reported as **void** rather than as a win. That rule exists
because thirteen separate measurements faked a result while this was being
built — a synthetic activation model, a group-size change scored on quality
alone, an error-feedback pass scored on its own training data, a quality win
measured on four tokens, three byte counters all wrong in our favour, and a
rotary table that corrupted both engines equally and therefore printed a *tie*.

The two checks that caught every one of them: grade against an independent
implementation, and disbelieve a result that is too good.

**The flattering number is the one that is wrong.**

## What can be claimed, with the number

| claim | measured |
|---|---|
| more accurate than the reference Q4_K_M files | 12.4% (pythia) to 41% (1B) lower residual error, held-out tokens |
| smaller files | 0.9947x (1B), 0.9986x (3B), 0.9981x (pythia) |
| faster decode on the GPU (vs llama.cpp b11177) | 1B 1.52x (95% CI 1.52-1.53), 3B 1.47x (1.47-1.47), pythia 1.17x (1.16-1.18); both engines throttled alike |
| faster decode on the CPU (vs b11177, 6 threads each) | 1B 1.025x (1.02-1.09), 3B 1.04x (1.03-1.15), pythia 1.16x (1.09-1.28) |
| faster GPU prompt reading (vs b11177) | 1B 1.48x at 128 tokens, 1.29x at 512; 3B 1.35x and 1.30x; pythia 1.51x and 1.37x (GPU at its thermal limit, both engines) |
| faster CPU prompt reading (vs b11177) | 1.36-1.70x on all three models, 28 and 128 tokens |
| repeat question from cache (vs llama-server) | 1.1 ms vs 1721.5 ms on the 1B (137-1540x across two runs), same answer |
| exact arithmetic (vs llama-server, same 1B) | 20/20 vs 11/20 |
| batched serving (vs llama-batched-bench, 1B) | total throughput 1.14-1.33x at 1, 2, 4 and 8 conversations; generation 1.05-1.24x at every size |
| original code | 0 of 17 data structures and 0 of 196 public names shared with upstream; 0 shared comment sentences; PROVENANCE.md |

## What must not be claimed

- **Not** "much faster on the CPU". CPU decode wins on all three, but on the 1B by
  5.8% (1.058x [1.023, 1.096], 2026-09-28) -- say "faster", never a big number, and
  quote the interval.

- **Not** "faster at everything". Nothing has been measured on ARM or Mac; 91
  architectures run as family heads against transformers (89 pass), and the newer ones
  are CPU-only. Say "faster decode and faster prompt
  reading, on CPU and GPU, on all three models raced".
- **Not** "it knows when it's unsure". It doesn't, and we measured that it doesn't —
  that's the honest story, and it is the better one.
- **Not** that Sheep's engine runs every model natively. Ten families have a grade on a
  real checkpoint (Llama, Qwen2, GPT-NeoX, Qwen3, Granite MoE, GPT-2, OLMo 2, BLOOM,
  SmolVLM, GPT-Neo); a model the engine cannot fit runs in compatibility mode instead.
- **Not** that it sees well. The local vision model is tiny (256M): it names
  colours and simple shapes, and it invents details in busy scenes -- pick a
  simple photo. Its vision tower matches the reference to 1.5e-6; its judgement
  is a 256M model's.
- Qwen2.5-3B's licence is `qwen-research`: use it to show Sheep, not as the product's model.

## Build and run the engine alone

```
cd sml && make
./sml-cli MODEL.ssf --prompt "The capital of France is" --gen 32
./sml-server MODEL.ssf --port 8080          # OpenAI-compatible
```

Needs a C compiler and OpenMP. `-march=native` picks the AVX2 kernels when the
CPU has them and the portable path otherwise; the portable path is correct but
about 2.5x slower, so treat it as a fallback rather than a build target.

`sml-cli` prints its own tok/s and GB/token and **no ratio**. A baseline belongs
to a model on a machine, not to a binary, so a comparison is opt-in:

```
./sml-cli MODEL.ssf --gen 128 --baseline TOKS_PER_S[,GB_PER_TOKEN]
```

Build a model from a Hugging Face checkpoint:

```
python tooling/sml_build.py HF_CHECKPOINT_DIR out.ssf
python tooling/sml_gates.py out.ssf         # correctness, then speed
python tooling/sml_hf_grade.py out.ssf HF_CHECKPOINT_DIR   # against full precision
```

Ask a vision model about a picture, on this machine:

```
python tooling/sml_image.py photo.jpg photo.f32          # resize + normalize
./sml-cli VISION.ssf --image photo.f32 --ask "What is in this picture?" --gen 64
python tooling/sml_bridge.py --local http://127.0.0.1:8099 --vision VISION.ssf
```

The bridge (OpenAI-compatible, `sml/ui` in front of it) answers image turns
with the local vision model when one is given, and otherwise sends them to the
cloud vision model named in `NEBIUS_VISION_MODEL`. `demo.ps1` starts all of it.
