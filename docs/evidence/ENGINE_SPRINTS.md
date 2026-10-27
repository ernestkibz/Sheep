# Pandora's box -- the hidden things, opened

A record of the improvement sprint that followed the first undeniable wins over
llama.cpp (2026-09-25). Each "box" is something that was not obvious from outside:
a hidden cost, a measurement trap, a correctness gap or an unused opportunity. For
each: what was found, how it was found, what changed, and what it measured. The
failures are here too, because a sprint record that only lists wins is the kind of
number this project has learned not to trust.

Every speed figure is a race against llama.cpp **b11177** (the newest build, published
that morning) on this laptop -- Ryzen 5 5600H, DDR4-3200, RTX 3050 Laptop 4 GB, Chrome
open -- with the CI tools: equal thread counts, warm, rounds alternated ABBA, verdict
from a bootstrap 95% interval, "won" only when the whole interval is above 1.

The results table is at the end.

---

## Box 1 -- the race harness can lie in either direction

**Found:** twice. A CPU prompt "win" came from giving Sheep 12 threads to the
reference's 6. Then GPU decode was reported *lost* (0.41-0.91x) because the engine's
defaults left the tuned GPU path (fused launches, graph replay) switched off -- every
earlier GPU win had raced it switched on.

**Changed:** race only with the CI tools (`sml_cpu_race_ci.py`, `sml_gpu_race_ci.py
--decode`, `sml_batched_race_ci.py`, `sml_orch_race.py`); the tuned GPU path is the
default (1da284e). Rule kept in memory: when a loss appears with no code change to
explain it, diff the configuration against the last win first.

**Measured:** GPU decode vs b11177 with plain defaults -- 1B **1.520x**, 3B **1.469x**,
pythia **1.169x**, every round.

## Box 2 -- the phase timers were sampling noise

**Found:** `omp_get_wtime` on this toolchain steps in whole milliseconds; a matvec is
~0.1 ms. Every per-phase profile before this was noise ("unaccounted" came out
negative).

**Changed:** the engine's timers use the performance counter.

## Box 3 -- forking OpenMP costs more than the work

**Found:** an OpenMP parallel region costs ~30 us to fork and join here, and a 1B decode
token opened ~65 of them. A synthetic token in isolation ran the kernel at 32 GB/s; the
engine at 24.

**Changed:** a resident worker pool (`sml-pool.c`) -- workers spin ~2 ms, then sleep --
for decode matvecs and attention, and later for batched serving.

## Box 4 -- five memory streams per row

**Found:** a flat tensor is read from five arrays (codes, two fp16 scales, two byte
fields): thirty open streams on six threads. The same bytes interleaved streamed 9%
faster in a bench.

**Changed:** decode repacks a private interleaved copy once (148-byte blocks per 256
weights), bit-identical kernels. Together with boxes 3 and 6: **CPU decode 1B
0.788x -> 1.025x** vs b11177.

## Box 5 -- 6-bit tensors ran in two passes

**Found:** every 6-bit tensor (the tied output head, and the measured 6-bit layers)
was read as a 2-bit pass plus a nibble pass -- in decode (head 30% of a token) and in
prompt reading, where profiling per call (`SML_PROF_BATCH=1`) showed 57-67 GMAC/s
against 270-320 for 4-bit tensors.

**Changed:** a one-pass 6-bit decode kernel, and 6-bit rows-in-lanes for prompts.

**Measured:** CPU prompt reading on all three models went from undecided/lost to
**1.36-1.70x** vs b11177, every round.

## Box 6 -- serial work and libm in hot loops

**Found:** decode attention was serial (6.5 ms of a 44 ms token at 256 tokens);
activation quantization called `lrintf` per element; GELU called `tanhf` per element
(98k per pythia token).

**Changed:** attention across the pool; vector quantization (bit-identical); vector
GELU. **pythia CPU decode ~1.2x faster.**

**Trap avoided:** the vector exp silu uses is only ~1.5e-4 relative; through 24 GELU
layers it moved pythia's logits by 2%. GELU got a float-accurate exp instead.

## Box 7 -- the orchestrator's "verified" flag sat on wrong answers

**Found:** the arithmetic verifier handed the exact value to the 1B as a fact; the
model ignored it, worked the sum out step by step and got it wrong -- 9 of 10 products
-- with the response still marked `verified`. And "what is -5 + 3" was read as
"5 + 3": the verified "fact" was 8.

**Changed:** a turn that is only arithmetic is answered exactly with no model call;
otherwise the exact value is appended when the answer omits it; signed runs decline.

**Measured vs llama-server, same 1B:** arithmetic **20/20 vs 11/20**; an exact repeat
answered from cache **1.1 ms vs 1721.5 ms**.

## Box 8 -- the GPU prompt on a 4 GB card

**Found:** the 3B's fast int8 copies need ~2.8 GB beside 2 GB of 4-bit weights; ~35%
fit. The unkept ones were re-made per call or read by tcm4, 1.67x slower per matmul.

**Changed:** tcm5 -- tcm3 with its weight tile re-made from 4-bit codes as it is
staged, bit-identical to the kept path. **0.868x -> 0.891x.**

**Failed:** tcm6 (32x128 tiles, one re-make per 128 tokens) measured *slower*
(266 vs 308 tok/s at the thermal floor), with or without a register cap; reverted.
At the floor the re-make is compute, so this one is VRAM-bound, not kernel-bound.

**Then won -- by the host (Ruth 2:3, Esther 4:14, Nehemiah 4:17).** The copies that do
not fit are made once, kept in pinned host memory, and streamed back over PCIe (6.5 GB/s
here) on the copy engine -- not the tensor cores -- while the matmuls run. Two traps on
the way: a "has the copy arrived?" test ran on the host, which enqueues a whole prompt in
milliseconds while the GPU takes ~400 ms, so it said "no" to ~95% of tensors (the GPU now
waits on the copy's event); and first-come keeping kept layers 0-12 and left the rest a
run of unkept tensors with no compute to hide the copies behind (kept tensors are now
spread evenly, and the granary is a ring of four). Bit-identical to keeping them.
**3B pp128 0.902x -> 1.101x [1.039, 1.259], 11/12; pp512 1.097x [1.091, 1.217], 12/12.**

## Box 9 -- batched serving did not exist

**Found:** the reference serves many conversations on one weight read; Sheep had one
context and no way to share a step.

**Changed:** `sml_forward_multi` -- each conversation keeps its own KV cache, every
step reads the weights once for all. Traps on the way: small batches borrowed the
prompt kernel, whose per-call repack only pays past ~64 tokens; OpenMP forks again;
`__m256` accumulator *arrays* made GCC spill every product (the first packed kernel
ran at ~12 GB/s); per-tile scalar field expansion cost as much as the multiplies. The
fix: a rows-in-lanes copy packed once (bit-identical to the prompt kernel), vector
field expansion, 1-4 slots sharing one unpack, and the pool. `SML_MULTI=exact` gives
every conversation bit-for-bit what it gets decoded alone.

**Measured vs llama-batched-bench (10 rounds):** total throughput **1.15-1.33x at 1, 2,
4 and 8 conversations**; generation 1.08x / 1.04x (undecided) / 1.05x / 1.24x.

## Box 10 -- pythia ran the wrong GELU

**Found:** pythia's config says `"gelu"` -- the exact erf form. The engine ran every
GELU model with the tanh approximation (right for GPT-2 and BLOOM). So does the
reference.

**Changed:** the file records it (`SML_ARCH_GELU_ERF`); CPU and GPU compute erf when it
is set; `sml_setarch.py` fixes existing files without touching weights.

**Measured -- honestly:** logits move 3.1%, but against full precision the change is
inside the noise (0.2091 -> 0.2095 stepped, 0.2106 -> 0.2086 batched; top-1 18/20 both).
A 3% component in quadrature with 21% quantization error. Correct, not a claimed gain.

## Box 11 -- v9 files were second-class

**Found:** SSF v9 packs group fields at 6 bits (2.7% fewer bytes), but the fast decode
layout required 8-bit fields and the GPU refused v9 outright -- so the smaller format
could not be the shipping one.

**Changed:** interleaved blocks carry 6-bit fields as they are (144/208 bytes, eight
unpacked by one variable shift, bit-identical); the GPU expands them at upload.

**Found later, and it changed the decision:** see box 12.

## Box 12 -- a 20-prompt grade can invent a regression

**Found:** a fresh 1B build with 6-bit fields graded 17% worse than the Sep 10 file, and
a control build *without* them graded 21% worse -- which read as a builder regression
since Sep 10 (float64 Hessian, encode refactor, calibration forward changes).

**How it was run down:** headers, every float tensor and the embedded tokenizer were
byte-identical; per-tensor weight error and per-tensor OUTPUT error on true activations
(Hugging Face hooks, layers 0-15) were equal to three digits. Then a paired grade on
100 held-out prefixes (`tooling/sml_grade_paired.py`, new):

| build | mean rel-RMS to full precision | vs Sep 10 |
|---|---|---|
| Sep 10 (shipping) | 0.1710 | |
| today's builder, same recipe | 0.1684 | -0.0026 [-0.0128, +0.0079] -- equal |
| today's builder, 6-bit fields (v9) | 0.1847 | **+0.0137 [+0.0046, +0.0235] -- 8% worse** |

**Result:** no regression -- the 20-prompt grade (sd ~0.09) compared one quantization
draw with another. But the same test found a real cost: 6-bit scale fields make the 1B
8% worse (the builder's help text said 0.31% of weight error). Raced, the v9 file
decoded 1.068x vs the shipping file's 1.056x -- inside each other's intervals. **8% of
quality for ~1% of speed: rejected; the shipping file stays.** The v9 support (box 11)
stays for the models where it pays.

## Box 13 -- two conversations are compute-bound under sustained heat

**Found:** a 2-conversation step streams its weights at 20-24 GB/s in the engine
against ~32-34 for one conversation. In isolation (best of ten bursts) both kernels
stream at ~35 GB/s at one and two slots; sustained, the laptop clocks down and two
slots carry ~1.5x the arithmetic per byte of one. The reference lands in the same
place (50-54 tok/s at B=2 against our 52.4): **a tie on this hardware, recorded as one.**

---

## Results

Against llama.cpp b11177, CI tools, this laptop. "Before" is the state when this
sprint began (after the first wins that day); "after" is its end.

| area | before | after |
|---|---|---|
| CPU decode, 1B | 1.025x [1.019, 1.088] | **1.056x [1.038, 1.080]**, 8/8 |
| CPU decode, 3B / pythia | 1.044x / 1.164x | unchanged |
| GPU decode, 1B / 3B / pythia | 1.520x / 1.469x / 1.169x | unchanged |
| CPU prompt, all three | 1.36-1.70x | unchanged |
| GPU prompt, 1B / pythia | won | unchanged |
| GPU prompt, 3B | 0.891x / 0.939x (lost) | **1.101x / 1.097x, both won** (box 8) |
| batched serving, total | won at 1, 2, 4, 8 | unchanged |
| batched serving, generation at 2 | 1.039x, undecided | a tie on this hardware (box 13) |
| orchestrator (arithmetic, repeats) | won | unchanged |
| pythia GELU | tanh (wrong form) | **exact erf, CPU and GPU** -- below the noise |
| v9 files | CPU arrays only, GPU refused | **fast decode path and GPU** |
| build comparison | 20-prompt grade | **paired 100-prefix grade with interval** |

## What the sprint added to the method

- A loss with no code change behind it: diff the configuration first (box 1).
- A "regression" between two builds: compare the files byte by byte, then per tensor
  on true activations, then a paired grade with an interval -- before bisecting (box 12).
- A kernel that is fast in a burst and slow in the engine: suspect sustained clocks
  before suspecting the engine (box 13).
- Arrays of vector accumulators spill; name them (box 9).

## Still open

- **Architectures:** 38 run against the reference's 100+.
- **Platforms:** nothing measured on ARM or Apple silicon; there is no such machine here.
- **Memory footprint:** decode keeps a private interleaved copy (and batched serving a
  second, packed copy) beside the mapped file.


---

# Sprint 2 -- a cool GPU (2026-09-26, morning)

The laptop had rested overnight, so the GPU started cool (1717 MHz) instead of at the
thermal floor (210 MHz) where every earlier GPU race had ended up. That one change of
conditions opened most of the boxes below.

## Box 14 -- the 3B's streamed copies were a floor-only win

**Found:** Ruth mode (box 8) streams every unkept int8 copy over PCIe. At the floor the
matmuls take ~415 ms of a 128-token 3B prompt and hide the ~290 ms of copies; cool, the
matmuls take ~130 ms and the prompt waited on the bus: **0.246x** the reference.

**Changed:** the first prompt after learning streams nothing and is timed; later prompts
stream the fraction 0.9 x compute / transfer, spread evenly, and re-make the rest
(bb53571). A fraction sweep inside one process (`SML_RUTH_FRACS`) confirmed the shape:
at 128 tokens 0-0.3 are equal once clocks settle, 0.45 and above are slower.

## Box 15 -- the re-make costs instructions, not waiting

**Found:** re-making int8 tiles from 4-bit codes inside the matmul (tcm5) took 1.41x the
kept path's matmul time on the 1B (24.7 vs 17.5 ms at 128 tokens).

**Tried, neutral:** packing the rounded bytes with `prmt` (3 instructions for 4 weights
instead of 12) -- ABBA in both orders, a 0.8% order effect each way and nothing else;
kept, since it is bit-identical. tcm7, which re-makes the next superblock into a
second buffer behind the current MMAs -- no faster: the re-make already overlapped the
other block on the SM. The cost is the instructions themselves.

**Changed:** tcm8 -- a 64 x 128 tile with eight warps, so each weight is re-made once per
128 tokens instead of per 64. 1B, all re-made: down projection 7.9-8.2 -> 6.6-6.7 ms,
prompt ~4600 -> ~5015 tok/s; 3B pp128 at 1717 MHz 1770 -> 1841 (94fd832).

**Failed:** tcm10 -- the re-made path on tcm9's frame (below) -- measured 3-4% slower
than tcm8 on the 3B; reverted.

## Box 16 -- the kept-tensor matmul ran at a third of peak

**Found:** tcm3 (64 x 64 tiles, four warps of 16 rows) read ~440 bytes of shared memory
per MMA -- every warp re-read all 64 tokens' fragments for its 16 rows -- and staged
each tile through registers. By this GPU's int8 rate it ran at roughly a third of peak.

**Changed:** tcm9 -- 128 x 128 tiles, eight warps of 32 x 64 (~190 bytes per MMA),
global -> shared by `cp.async`, double-buffered at half a superblock, 128-byte rows
XOR-swizzled. The same integer superblock sums and float epilogue: bit-identical to
tcm3 on all three models (ffa7a39).

**Measured:** 1B pp128 6700 -> 7970 tok/s, pp512 5850 -> 7600; 3B pp128 1833 -> 2038,
pp512 ~1300 -> ~1540 (interleaved, clocks printed alongside).

## Box 17 -- the driver's free-memory figure under Windows

**Found:** the keep budget was a fixed 3200 MB, and nvidia-smi showed ~570 MB of the
4 GB card unused at peak. `cuMemGetInfo` disagrees: under WDDM it reports ~950 MB taken
beside our 2161 MB, where nvidia-smi shows ~320 -- a budget from it is *smaller*
(2891 MB) than the fixed one every race has used safely.

**Changed:** opt-in only (`SML_KEEP_AUTO=1`); the keep spread uses the copy's true size
(1 + 1/64) instead of 5/4. A 3600 MB budget fits physically (3935 MiB peak) but the
clock swings are larger than its effect, so it is not the default.

## Box 18 -- the grade heated the race

**Found:** every prompt race run is gated on "next token agrees": a 40-token CPU prefill
compared with the GPU's logits. It ran just before the timed prompts, and this laptop's
CPU and GPU share one heatsink. In the mid-thermal rounds our timed prompts kept
landing in the GPU's throttle while the reference's did not (1B pp512 rounds of
0.45-0.71x between 1.28x rounds).

**Changed:** the CPU half of the grade runs after the timing; every run is still graded
(13ac7e4).

## Box 19 -- the mid-thermal state is a coin

**Found:** logging the clock every 100 ms across back-to-back runs: both engines are cut
by the same software thermal slowdown (reason 0x20), stepping between 1717 and 712 MHz
at 86-88 C. From a cool start, single rounds swing from 0.44x to 2.13x depending on
whose run lands in the drop. Those rounds say nothing about either engine.

**Consequence:** verdicts come from long races where the steady state dominates; no
cool-start GPU prompt verdict is claimed. Measured cool in one process, the 3B reads
2038 tok/s at 128 tokens against the reference's 1860-1888 cool (from race rounds) --
suggestive, not raced.

## Box 20 -- two conversations were on the wrong kernel

**Found:** box 13 recorded generation at two conversations as a tie, blamed on heat.
Profiled again, the packed rows-in-lanes copy streamed ~26-28 GB/s at two slots; the
decode kernel streams ~32-34 at one. The engine already had the other path
(`SML_MULTI=exact`: each slot through the decode kernel on 8-row tiles, the second slot
reading the tile from L1) and it was faster at two -- 58.8-61.2 vs 55.3-57.2 tok/s, four
alternating pairs -- while the packed copy still wins from four up (B=8 ~112 vs ~74).

**Changed:** two conversations run exact by default (792867f), which is also
bit-for-bit each conversation decoded alone.

**Measured vs llama-batched-bench:** B=2 generation **1.064x** [1.052, 1.087], 8/8 (it
read 0.969x, 1/6 that morning); total 1.141x [1.122, 1.166], 8/8. The heat story of
box 13 was true of the packed kernel, not of two conversations.

## Sprint 2 results

Against llama.cpp b11177, sml_gpu_race_ci.py, GPU at its thermal limit (both engines).

| GPU prompt | before sprint 2 | after |
|---|---|---|
| 1B, 128 / 512 tokens | 1.169x / 1.091x | **1.479x** [1.471, 1.487] / **1.289x** [1.282, 1.300] |
| 3B, 128 / 512 tokens | 1.101x / 1.097x | **1.345x** [1.333, 1.354] / **1.304x** [1.299, 1.306] |
| pythia, 128 / 512 tokens | 1.415x / 1.203x | **1.513x** [1.413, 1.658] / **1.365x** [1.358, 1.436] |
| 3B, cool start | 0.246x (not seen before) | ~1.09x in one process, not raced |
| batched generation, 2 conversations | a tie (0.969x-1.039x) | **1.064x** [1.052, 1.087] (box 20) |

## What sprint 2 added to the method

- A win measured in one thermal state is a claim about that state. Race cool and hot,
  or say which one was raced (box 14).
- When a kernel change is neutral, ask whether the cost is waiting or instructions
  before building a second overlap scheme (box 15).
- The race's own gate is part of the race: anything run before the timing has to cost
  both sides the same (box 18).

## Still open after sprint 2

- **A cool-start GPU prompt verdict** on the 3B (box 19).
- **Architectures:** 38 run against the reference's 100+.
- **Platforms:** nothing measured on ARM or Apple silicon.
- **Memory footprint:** as above.


---

# Sprint 3 -- one head per family (2026-09-28)

The question was the architecture count: 38 run against the reference's 100+. The
approach was Ernest's: find one head per family -- a model that behaves as every
member does -- run the heads, and leave each child's details to the checkpoints
people bring. The GPU dropped off the bus at the start of the day (Windows lists the
RTX 3050 as "Unknown"), so this sprint is CPU-side, plus the GPU keep budget written
and waiting for the card.

## Box 21 -- a family head does not need a download

**Found:** transformers 4.57.6 on this machine carries the modelling code for 139
causal-LM architectures. A head can be BUILT from that code -- a small random-weight
model, sized for 32-wide groups, with norm weights and biases pushed off their
defaults so every branch matters -- instead of downloaded. No checkpoint, no remote
code, a minute per family.

**Changed:** `tooling/sml_family_heads.py` builds each head, Sheep builds it at 6 bits
and runs it stepped and batched, and the last-position logits are compared with
transformers' in fp32. **The pass mark is the family's own 6-bit sensitivity**: the
same model in transformers with every matrix rounded to a 6-bit grid. Nemotron's (w+1)
norms and relu^2 make it 1.6x as sensitive as Llama, so a fixed band would have
failed a correct engine. `tooling/sml_head_dump.py` compares the residual stream
layer by layer when a head fails.

## Box 22 -- "ok from config" was mostly wrong

**Found:** the first sweep failed rows the roster called ok. Gemma (98% logit error),
Cohere (128%), Phi (38%), GPT-J (40%), ERNIE-4.5 (31%), Helium (30%), EXAONE 4 (66%),
Persimmon, Arcee, HunYuan (13%), BLOOM as transformers saves it (25%), Mistral past its
window (59%). Causes, each read from transformers' own source:

- seven families use **interleaved rotary** (Cohere, Cohere2, ERNIE, ERNIE-MoE, GLM,
  GLM-4, Helium); only GPT-J was listed
- Phi, GPT-J and Cohere run attention and FFN in parallel off **one** LayerNorm, and
  their configs never say so; Phi and GPT-J carry an **output-head bias** that was in the
  file and never read
- Gemma's GELU is **gated** (it ran as SiLU), its norms are (1+w), its embedding is
  scaled by sqrt(d)
- Mistral's `sliding_window` was ignored unless a `use_sliding_window` key said yes
- HunYuan normalizes Q and K **after** the rotary
- GPT-J rotates `rotary_dim`, a key nobody read

**All fixed; all pass.**

## Box 23 -- the builder dropped tensors silently

**Found:** HunYuan's `query_layernorm` / `key_layernorm` were not in the name table, so
the builder skipped them and the model ran without its Q/K norms. BLOOM's FFN norms,
named `transformer.h.N.post_attention_layernorm` in any BLOOM saved by transformers
(finetunes), were skipped the same way. The original BLOOM-560m names them without the
prefix and was fine.

**Changed:** the builder now **refuses** a checkpoint tensor it cannot place (mask
buffers and scalars excepted), naming the tensors. That one rule turned every later
unsupported family into a named error instead of a wrong answer.

## Box 24 -- what the heads unlocked

Per-layer windows and rotary (Gemma 2/3, Cohere2, SmolLM3's NoPE layers, EXAONE 4's
hybrid form, GPT-Neo, OLMo 3, Ministral), Gemma 3's second rotary base, the attention
softcap, relu^2 and LayerNorm Q/K norms (Nemotron, Persimmon, Arcee), Falcon's three
Q/K/V packings and its ALiBi-before-scale, MPT, StarCoder-1's shared K/V head, OLMo 1's
weightless norms and Q/K/V clamp, fused gate|up (Phi-3, GLM), the FFN width from the
tensors rather than config (transformers' MPT ignores expansion_ratio). Experts:
Mixtral, Qwen2-MoE and Qwen3-MoE (shared expert with a sigmoid gate), OLMoE,
HunYuan-MoE, Granite-MoE-shared, FlexOLMo, and DeepSeek-V3-style routing -- sigmoid
scores, a learned correction bias that chooses but does not weight, expert groups, a
routing scale -- for GLM-4.5-MoE, dots.llm1 and ERNIE-4.5-MoE. Also: dynamic-NTK
rotary no longer divides the frequencies (it changes nothing inside the trained
length), and HunYuan's NTK-alpha raises the base.

**Measured:** 52 of 65 heads pass. Roster: **62 run** (9 graded on real checkpoints,
42 as family heads, 11 from config only because their code is not in transformers),
37 gap with the feature named, 26 out of scope, 10 unknown. It was 38.

**Still gap, named:** hybrid state-space stacks (Jamba, Falcon-H1, Granite-4-H,
Nemotron-H, LFM2, Qwen3-Next), latent attention (DeepSeek-V2/V3, MiniCPM3), attention
sinks and clamped SwiGLU (gpt-oss), fused expert tensors (DBRX, Llama 4), sparsemixer
routing (PhiMoE), xIELU (Apertus), BitNet's sub-norms. Rows marked ok and disproved:
Apertus, MiniCPM3.

## Box 25 -- the GPU keep budget, written, not yet measured

The fixed 3200 MB left ~570 MB of the 4 GB card idle (peak 3524 of 4096 MiB) while
1.7 GB of the 3B's int8 copies were re-made or streamed; `cuMemGetInfo` under WDDM
under-reports (box 17). The budget now comes from NVML -- the card's physical free
memory, other programs included -- less 256 MB. The card went offline before it could
be raced; 3600 MB peaked at 3935 MiB earlier and fit.

**Measured when the card came back:** it picks 3315 MB with Chrome open, 3411 closed
(the 3B keeps 180 int8 tensors instead of 161). Old against new, interleaved in both
orders: whichever ran first won by 2-3% while the card warmed, and new was +3-5% at 128
tokens once hot; 512 tokens unchanged. Not slower, no memory trouble -- it stays the
default. Raced against b11177 with it: 3B pp128 **1.116x** [1.054, 1.161], pp512
**1.306x** [1.067, 1.310], 7/8 each.

## Box 26 -- the CPU cells, re-raced cooler

With the GPU off the laptop runs cooler. CPU decode: 1B **1.058x** [1.023, 1.096],
3B **1.079x** [1.069, 1.094], pythia **1.159x** [1.119, 1.197] -- the 1B and 3B came
out undecided at 6 rounds and won at 10; thin margins need the longer race. CPU prompt:
all six cells won (1.24-1.61x). Orchestrator: arithmetic 20/20 vs 11/20, repeats
1.0-1.6 ms vs 570-2370 ms. Batched serving: totals won at 1, 2, 4, 8; generation won at
1, 2, 8 -- and **lost at four, 0.953x [0.905, 1.027]** (it won 1.046x before). At four
slots the packed kernel is compute-bound (the 6-bit head: 84-87 GMAC/s, 30% of a step)
and the exact path is no faster there. Open.

## Sprint 3 results

| area | before | after |
|---|---|---|
| architectures run | 38 | **62** (51 run end to end against transformers) |
| rows marked ok and wrong | unknown | 14 found and fixed; 2 moved to gap |
| silent tensor drops | possible | refused by name |
| CPU decode 1B / 3B / pythia | 1.056x / 1.044x / 1.164x | 1.058x / **1.079x** / 1.159x |
| batched generation at 4 | 1.046x | **0.953x (open)** |
| GPU keep budget | fixed 3200 MB | NVML physical free - 256 MB: not slower, 3B still wins (1.116x / 1.306x) |

## Still open after sprint 3

- **Batched generation at four conversations** (box 26).
- **Architectures:** the 37 named gaps; 11 config-only rows need their remote code.
- **Platforms:** nothing measured on ARM or Apple silicon.
