# Measured results

All races run on one laptop: AMD Ryzen 5 5600H, 6 cores, 16 GB DDR4-3200,
NVIDIA RTX 3050 Laptop (4 GB, 210 MHz thermal floor under load). The reference
is llama.cpp, always the latest build, equal threads, ABBA rounds, bootstrap
95% interval. Read the caveats before quoting: all GPU numbers are at the
thermal limit; from a cold start the clock is higher and rounds can swing either way.

## Current standing (Oct 2026) — Llama 3.2 3B, the shipped model

Raced against llama.cpp b11177 (Sep 2026), 6 threads, warm, ABBA, 95% CI:

| measure | ratio |
|---|---|
| **bytes** | 2019 MB vs 2019 MB (tie) |
| **quality (KL)** | **1.21x better** (21% lower KL divergence) |
| **CPU prompt** | **1.48–1.55x** |
| **CPU decode** | 0.97–1.03x (undecided — interval crosses 1) |
| **GPU decode** | **1.37x** (measured hot) |
| **GPU prompt** | 1.19x hot, **0.86x cold boot** |

The GPU prompt at a cold boot lost because weights not kept in VRAM are copied
in 8-bit form; a direct 4-bit tensor-core kernel that skips that copy is the next
step. CPU decode is a tie and is stated as such.

Earlier (Sep 2026), three smaller models won every column against the same build:

| spec exercised | model | quality | decode (b11177) | file size |
|---|---|---|---|---|
| RMSNorm / halved rotary / GQA 16:2 / bias | Qwen2.5-3B | **23% better** | **1.144x** | 1.0225x |
| RMSNorm / halved rotary / GQA 32:8 / no bias | Llama-3.2-1B | **41% better** | **1.034x** | 1.0168x |
| LayerNorm / partial rotary / parallel residual | pythia-410m | **12.4% better** | **1.65x** | **0.9981x** |

GPU decode on all three also wins against b11177 at thermal floor: 1B 1.52x, 3B 1.47x, pythia 1.17x.

## Smaller than the reference on all three, and still better

Bytes were the one CPU axis still lost. They are not lost any more. The 6-bit
layers were re-placed by MEASURED logit damage and the least damaging ones were
dropped to 4 bits until each file was smaller than the reference Q4_K_M file --
then raced again on 512 held-out tokens, every position scored.

| model | SSF | reference Q4_K_M | size | quality, ours / theirs |
|---|---|---|---|---|
| pythia-410m | 266.07 MB | 266.58 MB | **0.9981x** | **0.876x** (12.4% better) |
| Llama-3.2-1B | 803.41 MB | 807.69 MB | **0.9947x** | **0.633x** (37% better) |
| Qwen2.5-3B | 1927.12 MB | 1929.90 MB | **0.9986x** | **0.761x** (24% better) |

How each file was built:

- **Llama-3.2-1B**: 6-bit `ffn_down` and `attn_v` on layers **1, 3, 13, 15** only.
  Those are the four most damaging layers by measurement, and under the
  every-other-layer recipe the original build used, all four were at 4 bits.
  Calibration matched the original (12288 samples).
- **Qwen2.5-3B**: kept 10 of the original 18 measured 6-bit layers and dropped the
  eight least damaging (3, 5, 13, 15, 23, 25, 27, 29). Calibration matched
  (16384 samples). The ranking measures each layer at its CURRENT width, which
  flatters layers already at 6 bits, so only existing 6-bit layers were
  candidates for demotion rather than re-ranking the whole model.

The 3B file is also better than the 46 MB larger file it replaces: residual
0.10778 against 0.10889. Top-1 against fp16 moved the other way, 91.6% to 90.4%,
and is reported for that reason. The 1B kept most of its lead (0.633x against the
larger file 0.592x).

Both files were checked for unwritten tensors before racing -- the builder
pre-allocates the whole file, so a full-size file is not evidence of a complete
one, and a partial file has fooled this project before.

**CPU decode, Qwen2.5-3B re-placed file (`qwen3b-meas.ssf`):** 12.99 tok/s median
against the reference 10.83 -- **1.1995x, 95% CI [1.1411, 1.2394]**, a real win,
and the ranges do not overlap: our p10 (12.54) is above their p90 (11.33). Twelve
reps in randomized order, the reference at its own best `-t 3`, ours at 6 threads.
It is not directly comparable with the older 1.144x figure, which came from a
different machine state; the ratio inside one run is what counts.

**CPU decode, Llama-3.2-1B re-placed file (`llama32-1b-meas.ssf`):** 30.34 tok/s median
against the reference 27.01 -- **1.1230x, 95% CI [1.0672, 1.1526]**, a real win
(our p10 29.02 against their p90 28.11). Same method, twelve randomized reps.

**CPU decode, pythia-410m (`pythia-flat.ssf`):** 72.62 tok/s median against 44.03 --
**1.6493x, 95% CI [1.5727, 1.6987]**, a real win. The earlier 1.503x came from four
order-balanced runs; this is the confidence-interval method, twelve randomized reps.

With these three races, **the CPU engine wins every measured axis on all three
architectures**: smaller files, better quality, faster decode.

## Head-to-head, each engine at its own best configuration

Interleaved, 5 reps, warm page cache, freshly booted machine. Thread count is
not a detail here: llama.cpp peaks at 3 threads (11.81) and collapses at 6
(9.66); this engine does the opposite (9.18 at 3, 11.69 at 6). Comparing both
at 6 threads reports **1.19x** for us and is meaningless -- it is our best
configuration against their worst.

| engine | tok/s | GB/token | correlation vs fp16 |
|---|---|---|---|
| llama.cpp Q4_K_M, `-t 3` | 11.42 | 1.920 | **0.99610** |
| Sheep depth 2, `-t 6` | **11.72** | **1.663** | 0.98837 |
| Sheep depth 3, `-t 6` | 8.84 | 2.556 | 0.99231 |

**Settled at 1.058x, 95% CI [1.035, 1.120].** Thread counts rotate through
3/4/5/6 and llama.cpp, so every configuration meets a similar thermal state
rather than the last one measured being the hottest. Freshly booted machine,
7 rounds, 128 tokens per run.

| config | median tok/s | drift early→late | vs llama.cpp |
|---|---|---|---|
| sheep `-t 6` | **12.28** | −3.3% | **1.058x** |
| sheep `-t 5` | 12.15 | −4.3% | 1.046x (CI contains 1.0) |
| sheep `-t 4` | 11.49 | −1.1% | 0.989x |
| sheep `-t 3` | 9.41 | −3.0% | 0.810x |
| llama.cpp `-t 3` | 11.61 | +4.4% | — |

Only `-t 6` clears 1.0 with its interval; `-t 5` does not. Six threads stays the
operating point.

**The condition matters and is part of the claim.** This is a burst-and-rest
pattern — one run, then others, then back — which resembles interactive use. A
*continuous* single-configuration load is a different regime and is not settled:
an earlier back-to-back run showed Sheep decaying 19.4% across it while
llama.cpp decayed 7.0%, because six threads heat the chip where three do not.
Under rotation that collapse does not reproduce (drift 1-4% everywhere), so it
belongs to sustained single-config load specifically.

Three numbers from identical code, all real:

| machine state | ratio |
|---|---|
| freshly booted, cold | 1.18x |
| rotating configs (this table) | **1.058x** |
| heat-soaked after an 80-minute build | 0.82x |

Report the middle one. The first flatters, the third slanders, and neither
describes use.

**Superseded:** 1.026x, measured with the run order alternated so neither
engine always inherits the hotter CPU — Sheep consistently ran second and
lost ~3% to that before the order was balanced. It clears the 1.02x gate but the
ranges overlap and the margin is close to the 2% noise floor, so this is a
narrow win, not a decisive one. **Q4_K_M is still ahead on quality**, so the
speed is bought at 0.98837 against 0.99610. Depth 3 remains strictly dominated.

**Shared-activation groups.** Tensors that read the same activation are issued
as one unit. Q/K/V are one group, ffn_gate/ffn_up are another. This is a
scheduling fact the file could state and a generic model file does not: three
matrices being adjacent is storage, three matrices sharing a flocking context is
something the runtime can act on.

The idea is falsifiable and was tested that way. Grouping Q/K/V gave 1.026x.
The principle then *predicted* a second group — gate and up also read the
post-`ffn_norm` vector — and predicted it would be worth less, because those two
already ran at 22 GB/s while K and V were stuck at 3.8. Measured: **1.030x**.
Directionally right and proportionally right, which is the stronger result.

The win came from one change: attn_q, attn_k and attn_v now share a single
dispatch. K and V are 256 rows against Q's 2048, and dispatched alone they were
too small to pay for a parallel region — 3.8 GB/s against 22 for the wide
tensors, and *worse* the more threads were added (6.6 on one thread, 3.8 on six)
while the kernel itself was fine. Folded into Q's dispatch they ride a team that
was already worth forming.

### Why we lose despite moving fewer bytes

Effective memory throughput, the two engines side by side:

| engine | GB/token | tok/s | achieved |
|---|---|---|---|
| llama.cpp Q4_K_M | 1.920 | 11.83 | **22.7 GB/s** |
| Sheep depth 2 | 1.663 | 11.62 | 19.3 GB/s |

**They extract ~18% more bandwidth from the same memory.** The format already
wins on bytes; the kernel loses more than that back on throughput. Per output
width:

| rows | calls/run | GB/s | what |
|---|---|---|---|
| 11008 | 3744 | 23.2 | ffn_up, ffn_gate |
| 151936 | 52 | 24.7 | output head |
| 2048 | 5616 | 19.6 | attn_q, attn_output, ffn_down |
| **256** | **3744** | **3.6** | **attn_k, attn_v** |

The narrow tensors are 7% of matvec time at a sixth of the achievable rate. That
was dismissed earlier as "worth 6%, not 26%" — true then, and the wrong call
now, because the engine gap is 2–4%.

Three attempts to close it all failed: NTA prefetch is inside noise at every
distance (the hardware prefetcher already covers a sequential stream), and
scaling the OpenMP team per call made things *worse* — a `num_threads` clause
that changes between calls forces team rebuilds, costing 18.9 → 14.3 GB/s on
tensors that were not supposed to change. Idling surplus threads instead
restored that and lifted the head to 24.7, with no end-to-end effect.

The one axis that is ours is bytes moved -- 1.663 GB against 1.920, 1.15x fewer
-- which matters for a memory-constrained machine but is not a speed win.

## Speed (historical, superseded by the table above)

| read depth | bits/weight | GB/token | correlation | tok/s | vs reference |
|---|---|---|---|---|---|
| 1 | 2.31 | 0.892 | 0.383 | 19.45 | unusable |
| **2** | **4.31** | **1.663** | **0.989** | **11.79** | **1.10x** |
| 3 | 6.31 | 2.435 | 0.987 | 8.47 | 0.79x |

Median of 5 on an idle machine; the margin held 1.03–1.12x across repeats.
Correlation is against an independent full-precision forward pass.

Depth 3 scores marginally *below* depth 2 because the error feedback was tuned
for the depth-2 view, so a full-depth read carries a compensation it does not
need.

## Accuracy

40-case harness, median of 3 reps, greedy (spread 0.0):

| class | SML depth 2 | recorded baseline |
|---|---|---|
| math | 100.0% | 90% |
| code | 90.0% | 30% |
| factual | 80.0% | 90% |
| format | 50.0% | 70% |
| **overall** | **80.0%** | **72.5%** |

**This is not a controlled comparison and the difference is probably not the
engine.** SML runs at 4.31 b/w built from the fp16 checkpoint; the baseline ran
a 4-bit file that is itself 5.31% from fp16 before anything else happens. The
servers also differ in sampling and template defaults. The honest claim is
"80.0% on this harness under these settings", not "7.5 points more accurate".

Remaining failures are model-level, not engine-level: counting letters in
"strawberry" (9, want 10), r's in it (4, want 3), 17 answered as not prime. A 3B
model missing those is expected. The format sub-score is the model ignoring
"answer yes or no" and being verbose — what the orchestrator's grammar path
existed to fix, and which the server does not yet carry.

## Bandwidth

| working set | kernel | engine |
|---|---|---|
| 192 MB | 21.7 – 26.1 GB/s | — |
| 1700 MB (realistic) | 18.6 – 21.7 GB/s | 20.4 – 21.0 GB/s |

The engine is at the kernel's achievable bandwidth for a real working set, and
at the same figure the reference engine achieves. **There is no bandwidth
headroom left**; speed can only come from moving fewer bytes.

An earlier 24.4 GB/s figure came from the 192 MB bench and does not survive at
model scale. A day of scheduler work was nearly spent closing a gap that a warm
cache had invented.

## Batched verification, and what it sizes speculative decoding at

One weight read serving k activation vectors (`sml_matvec_batch`), measured at a
realistic 1.7 GB working set and reproducible across runs:

| k | vs scoring one at a time |
|---|---|
| 2 | 1.43x |
| 4 | 1.84x |
| 8 | 2.24x |

**Sublinear.** The weight read is shared but the activation loads and per-group
arithmetic scale with k, and at 4 bytes of weight per 32 activation values that
arithmetic is not free. "The FLOPs are idle so batching is free" was too glib.

Sizing speculative decoding from those numbers, with a 0.5B draft against a 3B
target — cost per round is `k·T/speedup + k·T·(0.5/3)`, yielding `a+1` tokens:

| acceptance | k=2 | k=4 | k=8 |
|---|---|---|---|
| 50% | 1.15x | 1.06x | 1.02x |
| 60% | **1.27x** | 1.20x | 1.18x |
| 70% | **1.39x** | 1.34x | 1.35x |
| 80% | 1.50x | 1.48x | 1.51x |

**Small k wins**, which is the opposite of common practice. Because the batch
speedup is sublinear, drafting 2 tokens beats drafting 4 or 8 at any realistic
acceptance rate — the extra candidates cost nearly full price while adding
progressively less. Worth knowing before writing the loop rather than after.

Expected end-to-end: ~1.3x on top of the current 1.10x, so ~1.4x total. Real,
and the largest lever the SiLU finding does not block, but not transformative.

## Speculative decoding: implemented, sound, and a wash

Draft model proposes k tokens; the target verifies them against one read of its
weights. Interleaved against plain decoding, 3 runs each:

| | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| plain | 10.13 | 10.52 | 10.91 | **10.52** |
| speculative k=3 | 11.04 | 10.30 | 9.48 | **10.30** |

Acceptance is 77-85%, which is high. It still does not pay, because the
per-round work *outside* the shared weight read — k gathers, k activation
quantizations, attention walked serially over k positions — scales with k and
eats the saving. That work was ~5% of runtime at batch 1.

**The check that mattered:** making the target draft for *itself*, which must
accept 100% by construction. It accepted 83%, exposing two cache bugs — a
re-prime writing one position early, and the draft never running on its own last
proposal, leaving a hole in its cache whenever every proposal was accepted. With
both fixed, self-draft hits 100% and the real draft went from 34% to 77-85%.
Without that oracle the 34% would have been filed as "weak draft model" and the
bugs would have shipped.

## Architecture breadth: 3 families, verified against their own reference

Graded against each checkpoint's reference implementation, not against our numpy
pass -- ours is our reading of the architecture, so it agrees with our engine
when both are wrong the same way.

| model | family | correlation (depth 3) | top-1 | exercises |
|---|---|---|---|---|
| Qwen2.5-0.5B | Qwen2 | 0.978 | match | RMSNorm, halved RoPE, SwiGLU, QKV bias, tied head |
| Pythia-410m | GPT-NeoX | 0.977 | match | LayerNorm, partial rotary 16/64, parallel residual, GELU FFN, fused QKV, untied head |
| TinyLlama-1.1B | Llama | 0.991 | match | SPM tokenizer, untied head |

Pythia is the useful one: five paths that had been written and never run.

**Three bugs it exposed, all producing fluent output from wrong maths:**

1. **The RoPE table was built with stride and frequency base `head_dim/2` while
   the reader indexed it with `n_rot/2`.** Wrong entries at wrong frequencies,
   and only on partial-rotary models. One stored `rope_half` now feeds both.
2. **`rope_type` was inverted for five architectures.** The reference engine
   calls the *halved* convention "NEOX rope", after the family it first met it
   in. Reading their source and carrying the name across gives exactly the
   wrong mapping. Only GPT-J interleaves — checked against each architecture's
   own reference implementation rather than reasoned about.
3. **`attn_output`, `ffn_up` and `ffn_down` biases were never read.** Only QKV
   bias existed, so GPT-NeoX ran without three of its four bias terms.

**Naming is a second axis of breadth, separate from the maths.** Every variant
can be implemented correctly and a model still not load because its tensors are
called something else. That is now one regex table — 288 patterns over thirteen
roles — carrying the reference converter's knowledge but not its shape: theirs
is a class per architecture, ours is a table, so GPT-2 and MPT names already
resolve with nothing written for them.

Fused QKV is packed per-head by GPT-NeoX and as three contiguous blocks by
GPT-2/MPT/Phi-3. The layout is declared per family and **refuses when unknown**,
because reading one as the other produces a model that still speaks.

**And the calibration walk was Llama-shaped.** It generates the activations GPTQ
fits its error feedback against, so a wrong forward pass there degrades the
built model silently, with nothing downstream to catch it. It now follows the
same architecture flags the engine does.

### A seventh measurement that faked a result

A throwaway oracle read the engine's logits file as raw float32, past its 8-byte
`{n_vocab, depth}` header. A correct engine scored **0.10**, its top-1 landing
exactly 2 tokens off — and the layer-by-layer dump that finally caught it showed
all 24 layers matching at 1.00000. The instrument was wrong, not the engine.
Nearly bought a rewrite of working code.

The lesson is the one this file keeps repeating, now pointed at tooling: the
check is as likely to be broken as the thing it checks, and "everything matches
until the last step" is the signature of a reader bug, not a maths bug.

## What was tried and did not work

| idea | result |
|---|---|
| Contextual sparsity (skip inactive FFN rows) | **Fails at the oracle.** 40% keep leaves 13–20% error. No predictor can fix what perfect knowledge cannot. |
| Mixed-depth rows (read unimportant rows coarsely) | At 3.11 b/w, 4–6x worse than uniform depth 2 at 4.31. |
| Stacked residual quantizers | Composite got *worse* as the base improved; replaced by bit-slicing one integer. |
| Scaling a group by `amax` | Step lands ~1.8σ when MSE-optimal is ~1.0σ. |
| Asymmetric quantization | Better absolutely, worse per bit (1.0 b/w vs 0.3125). |
| Permuting by activation × \|w\| | Much worse than by \|w\| alone. Importance belongs in the objective, not the ordering. |
| Group 32 chosen on quality alone | 1.10x quality, 2x bandwidth loss. |
| Speculative decoding | Sound and correct, but a wash — per-round overhead scales with k. |
| Token-adaptive precision (cheap path + escalate) | Depth 1 agrees with depth 2 on top-1 **9.4%** of the time, and its top-2 margin carries no signal. Every threshold loses to uniform depth 2. |
| Large pages (2 MB) for the weight stream | **0.968x** — 3% slower than mmap, median of 5 interleaved reps on a freshly booted machine. TLB misses hurt random access; weight streaming is sequential, where the page-walker and prefetcher already hide them. |
| Retargeting GPTQ at the depth a tensor is read | **A 16x win in isolation that reverses on the model.** One tensor, fixed Hessian: `ffn_down` targeted at 4 bits and read at 6 gives 7.461%, targeted at 6 and read at 6 gives 0.460%. End to end the same change is *worse* — 0.98579 against 0.99151 for the mistargeted build. The build is layer-sequential, so changing one tensor's objective changes every downstream layer's calibration, which an isolated tensor cannot show. Third instance of extrapolating a model-wide decision from one sample. |
| Mixed allocation, measured properly | The same run found `attn_v`+`ffn_down` at depth 3 worth **+0.005** on Qwen2.5-0.5B (0.98650 → 0.99151) for 9% more bytes — where the 3B measured +0.0009. ggml spends its extra bits on exactly these two tensors. The allocation is right; our attempt to improve its objective was not. |
| Cross-tensor error compensation (quantize Q, let K absorb its error) | The one idea the Shepherd frame produced rather than the quantization literature: Q and K's errors multiply into the same score, so a declared group could quantize them jointly. **Dead twice over.** The errors are already independent (5.821% measured against 5.839% root-sum-square), so nothing adds constructively; and even with K *entirely unquantized* the ceiling is 1.42x, because `W_k` must be a linear map of X and cannot reach the compensating target. Retargeting K then spending its real bit budget gives **6.122% against 5.821%** — worse than doing nothing, since the compensating matrix has a wider dynamic range and costs more to represent. |
| Resident low-rank logit-correction head (`z_q + ABh_q`) | Held-out residual 98–100% at every rank; best gain +0.001, inside noise. Not a sample-size artifact — **in-sample** rank 1 explains only 13% of the error, so `h_q` does not determine the logit error at all. |
| Output-side (Jacobian) weighting, mode-coupled rounding | **Gate passed, mechanism works, does not transfer.** `M = JᵀJ` is 99.3% off-diagonal, λ₁/λ₆₄ = 218x. Coupled rounding cut the objective **6.12x on calibration** and **0.55x held out** — worse than nearest rounding on unseen text, degrading as modes are added. `H = XᵀX` is a converging second moment; `JᵀJ` depends on attention patterns and gates, so it is input-specific and cannot be fitted once. |
| Hot-precision (keep upgraded groups resident in LLC so their bytes stop recurring) | Bounded above by promoting everything, which reaches 0.99304 against a 0.99610 target -- the whole depth-2→3 budget is worth +0.0009 where +0.0027 is needed. Separately, value concentration is 7.7x at the top 1% of bytes where ~90x would be required to fit 772 MB of upgrade into 16 MB of L3. |
| Dropping the ladder (flat 4-bit instead of a 6-bit prefix) | Worth only **1.08x** once GPTQ runs on both sides: 3.242% nested against 2.989% flat. The earlier 1.5x figure compared grids with NO error feedback; the feedback absorbs nearly the whole nesting penalty. The progressive property is close to free. |
| Learned orthogonal change of basis (blocks of 32, Cayley-parameterised) | 1.10x, against 1.05x for fixed Hadamard and 1.60x for plain GPTQ. Predicted beforehand at <=1.28x: `P` acts on the input dimension so all 4864 rows share it, and it can only reshape the average distribution, which was already near-Gaussian. |
| Quantization-aware weight shaping (straight-through, real grid) | 6.08% against GPTQ's 5.73% at 4 bits -- Adam plus STE loses to second-order error feedback. Dual-depth shaping is the one useful variant: 6-bit goes 7.83% -> 3.92% for 1.2x worse at 4-bit, making every rung usable instead of just the tuned one. |
| Hadamard incoherence processing (QuaRot/QuIP# style) | 1.06x at 4 bits, 1.05x at 6. Group-of-32 weights are already near-Gaussian — outlier ratio 2.50 against ~2.2 for pure Gaussian — so the rotation has no outliers to spread. |
| Certified block skipping (don't execute provably negligible blocks) | Oracle skips only 10% of blocks for free; the implementable bound `Σ\|x\|·max\|w\|` loses 20% at 90% kept, against the oracle's 3%. Triangle inequality over 32 signed terms is ~6x too loose. |
| Low-rank correction of the quantization error | One 2-bit plane removes **97.8%** of the activation-weighted error for 1.09 MB. A rank-188 int8 correction at the same bytes leaves 1.8x more; matching the plane needs rank ~310 at 1.7x the bytes. |

> **WARNING: much of the table above was judged on the wrong metric.**
>
> Nearly every rejection here is a PER-TENSOR number. Error compounds through
> 36 layers, and today measured that compounding three independent times:
>
> | mechanism | per-tensor | end-to-end | implied exponent |
> |---|---|---|---|
> | dropping the ladder | 1.080x | **1.552x** | 5.7 |
> | allocation by amplification | 1.014x | 1.118x | 8.0 |
> | grid weighted by 1/[H^-1]jj | 1.015x | 1.073x | 4.7 |
>
> **"Dropping the ladder" is in this rejected table at 1.08x** -- and end to end
> it is the change that put SML ahead of Q4_K_M on quality and speed. It sat
> here filed as a failure while being the largest single win the project has had.
>
> So a mechanism dismissed at 1.06x per-tensor may be worth 1.19-1.34x end to
> end, and one at 1.10x may be worth 1.33-1.95x. On that basis at least these
> deserve re-measuring END TO END before they stay rejected:
>
> - Hadamard incoherence processing, 1.06x at 4 bits and 1.05x at 6
> - learned orthogonal change of basis, 1.10x
> - quantization-aware weight shaping, whose dual-depth variant took 6-bit from
>   7.83% to 3.92% -- rejected when nothing was 6-bit, and the mixed allocation
>   now puts the head, half the ffn_down layers and half the attn_v layers there
>
> **First re-test done. Hadamard pays.** Held out, end to end, flat 4-bit,
> Qwen2.5-0.5B:
>
> | config | logit err | top-1 | |
> |---|---|---|---|
> | flat 4-bit | 12.796% | 85.2% | 1.000x |
> | flat 4-bit + Hadamard | **11.605%** | **90.0%** | **1.103x** |
>
> Rejected at 1.06x per-tensor; worth **1.103x** by the metric that decides, at
> **zero bytes** -- the rotation is arithmetic, and decode leaves 75-90% of the
> FLOPs idle. Top-1 rises 4.8 points.
>
> The implied exponent here is 1.68, well below the 4.7-8.0 the other three
> showed, so the compounding factor is clearly not a constant and must not be
> used as one. It is a reason to RE-MEASURE, never a multiplier to apply.
>
> To ship it the codes stay in the rotated basis and the ACTIVATION is rotated
> at runtime, blockwise over the group of 32. The probe above rotates, quantizes
> and then un-rotates, which measures the quality honestly but is not a storable
> format -- a butterfly on the gathered activation is ~5 multiplies per element,
> under 0.5% of a token.

> **And the re-test reversed on the model.** Built into the real toolchain and
> run on the 3B, Hadamard is WORSE, not better:
>
> | build | bytes/token | correlation |
> |---|---|---|
> | v2, no rotation | 1.966 | **0.99660** |
> | v3, leaner allocation | 1.915 | 0.99505 |
> | v4, leaner + Hadamard | 1.923 | **0.98966** |
> | llama.cpp Q4_K_M | 1.920 | 0.99610 |
>
> Engine and the numpy harness agree on it (both put top-1 at 264 where the
> reference says 4460), so it is the model and not the instrument.
>
> The reason is in the original rejection and I read past it: *group-of-32
> weights are already near-Gaussian, so the rotation has no outliers to spread.*
> It is worse than neutral, because rotating MIXES 32 COLUMNS TOGETHER and the
> per-column importance is exactly what act-order permutation and the
> 1/[H^-1]jj grid weighting both exploit. The rotation averages away the two
> gains added today.
>
> The 0.5B probe that read 1.103x never quantized `token_embd` at all and used
> an exact head, so it did not contain the configuration that ships. Same shape
> as "retargeting GPTQ at the read depth": a clean win in isolation that
> reverses on the model.
>
> So the re-test was right to run and the answer was no. What survives is the
> METHOD -- judge by the end-to-end metric -- not this particular stone.

> The rule this breaks is not a new one. Rule 6 says never extrapolate a
> model-wide cost from one sample. The inverse holds and was never written down:
> **never dismiss a model-wide gain from one sample either.**

Five mechanisms have now failed, and the last two explain the first three.

**Adaptive precision fails at every granularity** — rows, tensor classes,
layers, tokens — for one reason that is not about granularity at all: the 2-bit
rung is noise, not a coarse answer. A depth-1 read picks the same top token as
depth 2 only 9.4% of the time. No allocation policy can be built on a cheap path
that is wrong 90% of the time, so a three-rung ladder has two usable rungs.

**And the marginal plane is very hard to beat.** Measured on one tensor in the
metric the model actually feels, `‖X·Eᵀ‖`: one extra 2-bit plane removes 97.8%
of the depth-1 error for 1.09 MB. The quantization error IS structured in
activation space — rank 1 captures 27.5%, where incoherent noise would give
~0.02% — but structure is not the point. Two bits per weight is *targeted*
information; a low-rank factor is a *global* approximation, and against even a
mildly structured residual the targeted representation wins. That closes
codebooks, learned correction operators and error dictionaries together: they
all compete against 2 bits/weight and lose on the same arithmetic.

The original byte-reduction mechanisms fail for one architectural reason: **SiLU is
small-but-nonzero everywhere**, so there is no negligible tail to skip or
coarsen. Published sparsity results come from ReLU-family models, which have
genuine zeros. That precondition is worth checking before betting on it —
`tooling/sml_sparsity_probe.py` tests any checkpoint unchanged.

## Weight error is the wrong objective (measured)

The quantizer spends the same effort on every layer, and the record shows it:
quantize one layer at a time with everything else exact, Qwen2.5-0.5B, depth 2.

| | spread across the 24 layers |
|---|---|
| weight error | 11.47% - 12.14%, **1.06x** |
| logit error | 3.14% - 9.63%, **3.07x** |
| correlation between them | **0.177** |

Weight error is flat because the quantizer makes it flat. Logit damage is not,
and the two barely relate. Layers 21, 23, 2 and 3 amplify roughly three times
harder than layers 8, 10, 12 and 13, and nothing in the pipeline knows it.

This is the single most useful thing measured in this pass. Every mechanism in
this file was graded on weight error or an activation-weighted proxy for it, and
that number explains 3% of the variance in what actually reaches the logits.

**The lever is narrower than the finding.** Read depth moves in 2-bit steps, so
bits cannot be shifted between layers that way (see below). Group size can:
two-level scale and min cost `2*(16/256 + 8/G)` b/w, so sixteen layers at G=64
plus eight at G=16 is exactly the byte cost of twenty-four at G=32.

| plan | b/w overhead | logit error | |
|---|---|---|---|
| uniform G=32 | 0.625 | 34.774% | |
| allocated by measured amplification | 0.625 | 34.298% | **1.014x** |

Byte-matched and real, but small, because group size is a weak dial: halving it
buys about 1.10x locally and only a third of the model moves in each direction.
The finding deserves a better lever than the one we have.

## Non-uniform levels: the one gain that survived (1.056x, free)

Levels are placed uniformly across each group's range. Weight distributions are
not uniform -- measured kurtosis 2.45-2.51, flatter than Gaussian -- so a
Lloyd-Max table placed on the actual density fits the population instead of the
extremes.

Scored after error feedback, which is what matters; both paths identical except
which levels the rounding snaps to:

| tensor | uniform | Lloyd-Max | |
|---|---|---|---|
| attn_q | 3.352% | 3.173% | 1.056x |
| attn_v | 4.731% | 4.439% | 1.066x |
| ffn_gate | 2.184% | 2.106% | 1.037x |
| ffn_up | 2.912% | 2.758% | 1.056x |

On raw weights it measures 1.07-1.13x; feedback was already taking about half of
it. **1.056x is the honest number.**

It costs **zero bytes per weight** and 16 bytes per tensor. It is NOT free in
the kernel, and the first version of this entry said it was, by reading
`FORMAT.md` instead of `sml-kernel.c` -- rule 4, committed one day after writing
rule 4.

The kernel is group-planar: `bit-plane g IS group g`, and each ladder level is a
separate pass contributing `4^level * sum(code * x)` as an integer dot product,
combined by scalar multipliers in the epilogue. The reconstruction is therefore
**linear in the code by construction, and the 4-bit code is never assembled**. A
non-uniform table `T[code]` is nonlinear and does not decompose into per-plane
dot products.

Applying it means fusing two planes into a 4-bit index (`srli`+`and` twice, then
`slli`+`or`), one `pshufb`, then a single `maddubs` where there were two. That is
roughly instruction-neutral at depth 2 and strictly worse at depth 3, where a
64-entry lookup needs four `pshufb` and blends. So the mechanism survives, at the
price of an inner-loop rewrite rather than an added instruction.

What still separates it from the codebook family is the absence of gathers: the
table is 16 bytes and lives in a register.

**It saturates immediately.** Selecting one of K tables per row costs
`rows * log2(K)` bits, about 0.0004 b/w -- effectively free -- and buys nothing:

| tensor | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| attn_q | 3.173% | 3.167% | 3.154% | 3.156% |
| ffn_up | 2.758% | 2.750% | 2.751% | 2.749% |

1.056x at K=1, 1.063x at K=8. Rows inside a tensor share a shape. Ship K=1.

## The grid search is weighted by the wrong Hessian term (1.015x, free)

`_choose_grid` weights its objective by `imp**2`, the column norm, which is
`diag(H)`. Error feedback then compensates part of each column's error from the
columns after it, so the cost of an error in column *j* is not `(dw)^2 * H_jj`
but `(dw)^2 / [H^-1]_jj` -- the uncompensable share. The engine already computes
that number: it is `U[j,j]` in the Cholesky factor, used for propagation and not
for choosing the grid.

| tensor | unweighted | diag(H) (shipped) | 1/[H^-1]jj | |
|---|---|---|---|---|
| attn_q | 3.238% | 2.912% | 2.866% | 1.016x |
| attn_v | 4.537% | 4.385% | 4.360% | 1.006x |
| ffn_gate | 2.054% | 2.019% | 1.991% | 1.014x |
| ffn_up | 2.703% | 2.626% | 2.549% | 1.030x |

Median **1.015x**. Same cost -- a different weight vector, not extra work. All
four move the same way, so the algebra is right and the magnitude is just small;
after the column permutation the two weightings are strongly correlated.

## Per-tensor read depth: uniform is already optimal

llama.cpp's K-quants assign different bit widths per tensor and we assign one
depth to everything, which looked like a structural gap. It is not one.

| tensor | depth 1 | depth 2 | depth 3 | share of bytes |
|---|---|---|---|---|
| attn_q | 26.91% | 4.52% | 5.34% | 5.4% |
| attn_k | 24.35% | 4.09% | 4.51% | 0.8% |
| attn_v | 31.05% | 5.72% | 5.31% | 0.8% |
| attn_output | 42.31% | 7.03% | 11.04% | 5.4% |
| ffn_gate | 23.55% | 4.04% | 4.83% | 29.2% |
| ffn_up | 32.46% | 5.44% | 7.00% | 29.2% |
| ffn_down | 32.84% | 7.64% | 2.41% | 29.2% |

Searching every assignment at or below the bytes of uniform depth 2 returns
uniform depth 2, **ratio 1.000x**. Depth 1 costs 24-42% on every tensor, so
there is no cheap tensor to demote and nothing can pay for a promotion.

Two things fall out. Depth 3 is *worse* than depth 2 for six of seven tensors --
the feedback is tuned to the depth-2 view and a fuller read carries a
compensation it no longer needs. `ffn_down` is the exception, 7.64% -> 2.41%,
and it is 29.2% of the bytes.

## Sparse outlier extraction: the loss is diffuse

The residual family failed three times as a dense correction, and a 2-bit plane
beat a rank-188 factor at the same bytes. The sparse version is a different
mechanism -- keep the few worst weights exactly, let the grid re-fit on the rest
-- and it costs ~0.006 b/w at 0.1%, with the correction being idle FLOPs rather
than moved bytes.

Activation-weighted, 4-bit asymmetric grid:

| | base | 0.05% | 0.10% | 0.50% | 1.00% |
|---|---|---|---|---|---|
| attn_q, oracle | 6.88% | 6.55% | 6.36% | 5.71% | 5.34% |
| attn_q, realistic | 6.88% | 6.78% | 6.75% | 6.40% | 6.03% |
| ffn_gate, realistic | 5.59% | 5.53% | 5.45% | 5.07% | 4.84% |
| ffn_up, realistic | 8.11% | 7.97% | 7.89% | 7.56% | 7.27% |

The oracle knows exactly which entries hurt and still removes only 22% of the
error by zeroing the worst 1%. Realistic gain at 0.1% is 1.02-1.03x.

**The error is spread evenly, not concentrated in outliers.** That is the same
shape as the SML gap itself -- 1.43-1.70x behind Q4_K_M on every weight class at
matched bits rather than on any one of them -- and it is why targeted mechanisms
keep failing here.

## Entropy coding is closed: the codes are 96.9% incompressible

Never tried, and worth one measurement because bytes are the only lever left.
Read the shipping file's 2-bit planes directly, Qwen2.5-0.5B depth 2.

| tensor | plane 0 | plane 1 | joint 4-bit | order-1 |
|---|---|---|---|---|
| attn_q | 1.906 | 1.998 | 3.880 | 3.868 |
| attn_output | 1.896 | 1.998 | 3.870 | 3.861 |
| ffn_gate | 1.903 | 1.998 | 3.879 | 3.873 |
| ffn_up | 1.901 | 1.998 | 3.876 | 3.871 |
| ffn_down | 1.900 | 1.999 | 3.877 | 3.870 |

**3.876 bits of entropy in 4 stored bits.** Ceiling for Huffman, rANS or zstd on
the code payload is **1.03x**, bought at the price of variable-length decoding,
which ends random access and the fixed-stride streaming the kernel is built on.
Closed on the ceiling alone.

It also explains the previous section. The MSE grid search spreads codes to
near-maximum entropy -- a nearly flat code histogram -- which is precisely why
non-uniform levels can only recover 1.056x. The two results are one fact seen
from two sides: **there is almost no redundancy left in the code stream, so
neither a better table nor a better coder has much to take.**

## What SML actually has to beat, stated once

Two figures for "the gap" appear above and they are consistent, but they are
easy to confuse, so the arithmetic is written out here.

| | correlation | residual error |
|---|---|---|
| SML depth 2 | 0.98837 | 0.15207 |
| llama.cpp Q4_K_M | 0.99610 | 0.08823 |

`residual = sqrt(1 - corr^2)`, so **SML needs a 1.724x reduction in end-to-end
output error to match Q4_K_M** -- while already spending 13% fewer bytes per
token (1.663 vs 1.920 GB). The "1.43-1.70x behind on every weight class" figure
is the same gap measured on weights rather than logits.

1.724x is the number any proposal has to be sized against. The three mechanisms
found in the Beatitudes pass multiply to 1.087x **if they compose perfectly**,
which is not yet measured and is not the way to bet.

## The gap is structural, not budgetary

We spend 1.663 GB/token where llama.cpp spends 1.920 -- a 13% surplus, unspent.
The obvious move is to spend it on the layers that hurt, which the ladder can do
per layer and GGUF cannot do at all. Sized before building, per rule 1:

| promoted to depth 3 | GB/token | vs llama | correlation |
|---|---|---|---|
| 0 layers | 1.663 | 0.87x | 0.98837 |
| 6 layers | 1.886 | 0.98x | 0.99050 |
| all 24 (= full depth 3) | 2.556 | 1.33x | 0.99231 |
| **llama.cpp Q4_K_M** | **1.920** | 1.00x | **0.99610** |

**Q4_K_M beats our depth 3 while reading 25% fewer bytes.** Spending the surplus
cannot close the gap, because no point on our ladder reaches Q4_K_M at any
price. Depth 3 is 6.31 b/w against their ~4.5 and still loses.

So the deficit is not a budget we mis-allocated. It is quality per bit, and the
bit-sliced ladder is where it comes from: levels are 2 bits each and the
reconstruction is forced to be *linear in the code* so that a prefix of the code
is a coarser read of the same number. That constraint is what buys `--depth
auto`; it is also what costs the quality.

That reframes sixteen closed mechanisms. They were all improving the arithmetic
sitting on top of the structure that is itself the deficit -- which is why each
one returned a few percent and the total never moved. **The next real attempt
has to change the ladder, not the quantizer.**

## What the ladder actually costs, and what their format does not buy

Ernest's question: build our own version of what the reference format solved,
and start from there. Measured before building. Representation only, no error
feedback on either side, activation-weighted, medians over three layers.

| scheme | b/w | attn_q | attn_v | attn_output | ffn_gate | ffn_up |
|---|---|---|---|---|---|---|
| LADDER (ships) | 4.625 | 7.184% | 8.129% | 8.293% | 5.872% | 8.406% |
| FLAT, no prefix constraint | 4.625 | 6.708% | 7.549% | 7.760% | 5.454% | 7.842% |
| HIER, clean-room K-style | 4.500 | 6.835% | 7.894% | 7.832% | 5.542% | 7.953% |
| EXACT scales (control) | 4.000 | 6.681% | 7.539% | 7.742% | 5.457% | 7.851% |

**The ladder tax is 1.072x at identical bits.** That is the price of forcing
reconstruction to be linear in the code so a prefix is a coarser read. It buys
`--depth auto`, which nothing else in this space has. Now it has a number.

**Their arrangement is not the advantage.** A clean-room hierarchical
scale-and-min -- 8 sub-blocks of 32, 6-bit sub-scale and 6-bit sub-min against
fp16 super d/dmin -- lands *at* our two-level scheme, not above it (1.057x vs
the ladder at 0.125 fewer b/w, i.e. level with FLAT). We already had the
equivalent, independently. Copying the block structure buys nothing.

**Scale quantization is nearly free.** EXACT (unquantized scales, 4.000 b/w)
scores within 0.5% of FLAT (4.625 b/w). The 0.625 b/w we spend on scales is not
where the loss is.

So the reference lead decomposes as: ~7% is our own ladder tax, and most of the
remainder is that Q4_K_M simply spends more bits -- 1.920 GB/token against our
1.663, because it promotes selected tensors to 6-bit.

The actionable form is not a redesign. It is **dropping the prefix constraint on
tensors that never need a runtime depth change**, which is worth 1.072x at zero
byte cost, and keeping the ladder only where `--depth auto` is actually used.

### One instrument bug, caught by rule 5

The first run of this read 70% error on every scheme. Group mins are SIGNED and
the two-level quantizer clipped them to [0, 255], destroying every negative min.
A 4-bit quantizer cannot produce 70% error; the number was impossible before it
was wrong, which is what flagged it.

## Held out: the Beatitudes stack is 1.124x, not 1.194x

Four configurations, one code path, layer-sequential, graded on the final
logits. Built on calibration tokens 0-511, graded on 2048-2559.

| config | logit err | top-1 | vs base |
|---|---|---|---|
| baseline (shipping) | 19.917% | 83.6% | 1.000x |
| + Lloyd-Max levels | 20.063% | 80.5% | 0.993x |
| + 1/[H^-1]jj weighting | 18.562% | 83.6% | 1.073x |
| + amplification allocation | **17.722%** | 83.8% | **1.124x** |

In-sample the same four read 1.000 / 0.979 / 1.068 / **1.194x**. The error gain
survives; the top-1 gain did not (in-sample +5.1 points, held out +0.2).

Lloyd-Max alone is NEGATIVE end-to-end (0.993x) despite being 1.056x better per
tensor, and only turns positive once the grid weighting changes. The three do not
decompose; they interact, and only the stack is worth quoting.

## Gather the fragments: one was empty, one was free

| | held out |
|---|---|
| per-row bias absorbing the uncompensated mean error | **1.001x** |
| column order: most important FIRST, not last | **1.028x** |

The bias is dead and consistent with an earlier result: stochastic rounding also
removed a bias that was already zero (+4.04e-07). The MSE grid search centres the
error, so there is no mean left to gather.

The order is real and costs nothing. We permuted by `max|w|` **ascending**, which
puts the largest-magnitude columns LAST -- where no later columns remain to
absorb their error. Reversing it to most-important-first is a one-line change
worth 1.028x at zero bytes.

## Mixed bit widths: the first path whose arithmetic reaches the target

Per-tensor bit allocation returned exactly 1.000x this morning. That was a fact
about the ladder, not about allocation: depth moves in 2-bit steps, so the only
trade available was demoting a tensor to depth 1 at 24-42% error.

Drop the prefix constraint and the step size disappears. Held-out error by flat
width, error feedback and act-order on:

| tensor | 4-bit | 5-bit | 6-bit | share of bytes |
|---|---|---|---|---|
| attn_q | 5.937% | 2.923% | 1.434% | 5.4% |
| attn_k | 5.050% | 2.491% | 1.227% | 0.8% |
| attn_v | 7.043% | 3.429% | 1.702% | 0.8% |
| attn_output | 6.762% | 3.298% | 1.627% | 5.4% |
| ffn_gate | 4.783% | 2.344% | 1.178% | 29.2% |
| ffn_up | 7.475% | 3.665% | 1.811% | 29.2% |
| ffn_down | 3.755% | 1.849% | 0.910% | 29.2% |

Their byte count is 4.98 b/w in our terms (1.920/1.663 x 4.31). The best
allocation inside it lands at 4.97:

```
attn_q:4  attn_k:5  attn_v:6  attn_output:5  ffn_gate:5  ffn_up:5  ffn_down:4
```

**1.621x over uniform 4-bit flat**, and 1.621 x 1.072 (the ladder tax) = **1.738x
over what ships**, against the 1.724x needed to match Q4_K_M.

This is the first route all day whose arithmetic reaches the target, and it wins
nothing by cleverness: it concedes that spending the bits was correct, and
allocates them by measured amplification rather than by a fixed rule.

**Not yet a claim.** That figure chains ratios from two harnesses across two
probe layers, which is rule 6 -- the mistake made four times already in this
file. The end-to-end build is what decides it.

## Dropping the ladder: 1.552x at identical bytes, end to end

The single most valuable measurement in this file. Held out, layer-sequential,
final logits, baseline reproduced four times at exactly 19.917%.

| config | b/w | logit err | top-1 | vs shipping |
|---|---|---|---|---|
| ladder, as shipped | 4.31 | 19.917% | 83.6% | 1.000x |
| **flat uniform 4-bit** | **4.31** | **12.834%** | **86.9%** | **1.552x** |
| flat mixed widths | 4.97 | 8.760% | 91.6% | 2.274x |

The middle row costs **nothing**: same bit width, same bytes, same bandwidth, so
the same 1.058x decode. It is the ladder tax plus the column order, compounded
through 24 layers. Per tensor those measured 1.072x and 1.028x; end to end they
are worth 1.552x, which is the compounding this file has now seen four times.

In the terms that decide it:

| | correlation | b/w | decode |
|---|---|---|---|
| SML as shipped | 0.98837 | 4.31 | 1.058x |
| **SML flat uniform-4** | **0.99519** | 4.31 | 1.058x |
| llama.cpp Q4_K_M | 0.99610 | 4.98 | 1.000x |

**The gap went from 1.724x to 1.111x for free.**

Closing the rest costs +0.18 b/w (interpolating the measured 4.31 -> 4.97 =
1.465x), landing at 4.49 b/w = 1.733 GB/token against their 1.920:

| achieved bandwidth | decode vs llama.cpp | quality |
|---|---|---|
| our 19.5 GB/s today | 0.985x | matched |
| llama-class 21.9 GB/s | **1.107x** | matched |

So the whole remaining question is **bandwidth efficiency, not quantization**.
We achieve 19.5 GB/s where they achieve 21.9 with half the threads, and the
ladder is a plausible cause: depth 2 is two passes over two separate level
arrays with two epilogues, where a flat 4-bit code is one pass with one. A
single-threaded microbenchmark of exactly that change reads 1.293x (7.2 -> 9.3
GB/s), though single-threaded is compute-bound and the engine at 6 threads is
not, so that figure is an upper bound rather than a prediction.

### What the ladder is actually worth now

`--depth auto` is not free and never was; it now has a price: **1.552x of
end-to-end quality**. The capability should therefore be *per tensor*, not
global -- flat codes for everything, and the prefix-readable ladder only where a
runtime depth change is actually used.

### A near miss worth recording

The first uniform-4 run reported numbers identical to the mixed run to three
decimals. Two different bit allocations cannot produce identical logits; the
edit setting the allocation had silently failed because the file had CRLF line
endings and the search pattern had LF. Same bug class as the heredoc that once
ate a CLI fix and let a commit claim a repair that never applied.

> Identical results from configurations that differ is never a coincidence.
> Verify the edit landed in the file, not that the script ran.

## The flat path, end to end on the real engine

Not a numpy simulation any more. SSF v8 carries a per-tensor layout, the builder
emits it, the kernel reads it, and the whole stack runs.

Qwen2.5-0.5B, identical bytes per token (0.287 GB), 128 tokens, 6 threads,
configurations rotated so each meets a similar thermal state:

| round | flat | ladder |
|---|---|---|
| 1 | 48.91 | 41.59 |
| 2 | 47.65 | 41.99 |
| 3 | 48.63 | 41.50 |
| 4 | 48.47 | 41.68 |
| 5 | 49.34 | 42.52 |
| **median** | **48.63** | **41.68** |

**1.167x**, and the two do not overlap at all -- the slowest flat run beats the
fastest ladder run. Same bytes, and the quality is better too: the numpy
measurement put the same change at 1.552x on end-to-end logit error, held out.

Generation, same prompt:

```
flat    " a 16 year old male. I have a 1.5 inch thick scar on my left hand..."
ladder  " a 17 year old, and I am a male. I am 18 years old, and I am a female."
```

### The bug that made it look broken, and what found it

The first flat model emitted nothing but newlines, and gave the SAME top token
for two completely different prompts. Identical output from different input is
not a quality problem -- it means the input never arrived.

The matvec was correct. The kernel had been graded against an independent scalar
reference at 3.275e-07, the packing round-tripped exactly, the file reconstructed
against the HF checkpoint at 8-15% on every tensor class, and the dispatch was
instrumented and confirmed taking the flat branch with `layout=1 nbits=4`.

The embedding lookup was the one path that reads `token_embd` as a ROW rather
than a dot product, through `sml_dequant_row`, which knows only the ladder's
2-bit planes. Every token's input vector was misread.

What isolated it was running a numpy forward pass **from the file's own weights**
and comparing against the engine on the SAME tokens: numpy 2130, engine 198. That
splits builder from engine in one measurement. After the fix both read 2130.

> A kernel that passes its unit test can still leave the model broken, because
> the unit test grades the path you thought about. The paths you did not think
> about are the ones that read the same tensor a different way.

There were two such call sites, not one. The first patch fixed the batched path,
the output did not change, and the second site -- the single-token generate path
-- was the one actually running.

## Head to head on the 3B: faster, smaller, still 1.165x less accurate

The whole stack, flat with mixed 4/6 widths, against the reference at 3B.

**Speed**, order-balanced so each engine sits in each position equally -- running
last every round is worth ~3% and would have flattered us. Heat-soaked machine,
which our own notes call the realistic sustained condition.

| config | median tok/s | vs reference |
|---|---|---|
| **Sheep, flat mixed 4/6** | **11.10** | **1.136x** |
| Sheep, ladder depth 2 | 9.59 | 0.982x |
| llama.cpp Q4_K_M `-t 3` | 9.77 | 1.000x |

The ladder falls *below* parity here at 0.982x, against 1.058x recorded on a
cold machine -- six threads throttle where three do not, exactly as recorded.

**Bytes**: 1.871 GB/token against 1.920, 1.026x fewer.

**Quality**, and this is the one that decides it. Measured from THEIR file: the
reference format decoded directly and run through the SAME numpy forward on the
SAME tokens, because our ladder reads 0.98837 on the recorded token set and
0.99223 on this one -- gaps compress on easy tokens, and quoting across sets
would flatter whichever was measured on the easier one.

| engine | correlation | rel-RMS | top-1 | top-5 |
|---|---|---|---|---|
| llama.cpp Q4_K_M | **0.99610** | 0.08980 | yes | 5/5 |
| SML mixed 4/6 | 0.99471 | 0.10551 | yes | 4/5 |

Their 0.99610 reproduces the recorded figure exactly, which validates the chain.

**Residual error 0.10272 against 0.08821 -- we are 1.165x behind.** Faster,
smaller, less accurate. Not a win.

For the record the deficit was **1.724x** at the start of the day.

### Decoding their format to measure it

`scratchpad/refq.py`, deliberately outside the repo so the tree stays clean of
their names. Verified before being trusted, against the fp16 checkpoint:

| block type | error vs fp16 |
|---|---|
| their 4-bit blocks | 7.54% |
| their 6-bit blocks | 1.79-1.86% |

The 6-bit decode read **103%** on the first attempt -- noise that still looks
like weights. Within a 128-weight half the four quarters draw from scale slots
`is+0, is+2, is+4, is+6`; getting that interleave wrong produces a plausible
tensor rather than an error. Same failure shape as everything else this week.

**They promote `ffn_down` to 6 bits too.** We chose that allocation from our own
per-tensor measurement before ever reading their file, and arrived at the same
answer independently.

### Two byte counters were wrong, both in our favour

`sml_bytes_at_depth` counted only plane 0, reading a 6-bit tensor as its 2-bit
plane alone. `matvecN` and the batch path still used the ladder formula
`depth * level_bytes`, which under-counts a flat nibble plane by 2x -- and Q/K/V
and gate/up all issue through there.

Together they reported **1.413 GB/token** where the truth is **1.871**, and would
have claimed "1.36x fewer bytes" instead of 1.026x. Bytes/token is the
denominator of every bandwidth claim, so it is the number that must never
flatter us. Caught because 1.413 disagreed with a 1.85 GB estimate done on
paper, and because a mixed model cannot possibly read fewer bytes than a uniform
4-bit one.

## MAX_H is a regulariser, not a memory compromise -- and the Hessian has been rank-deficient all along

`MAX_H = 4096` caps the Hessian and runs `gptq_block` on each chunk
independently, so `ffn_down` at 11008 columns gets three separate sweeps and
error never propagates between them. That looked like a quality bug worth
fixing. It is the opposite.

blk.12.ffn_down, Qwen2.5-0.5B, 4864 columns, 512 calibration samples,
activation-weighted:

| MAX_H | chunks | weighted error | |
|---|---|---|---|
| 4096 | 2 | **1.080%** | |
| 4864 | 1 | 7.730% | **7.2x WORSE** |

Removing the cap makes it seven times worse. With T samples and n columns the
Hessian has rank <= T, so at 512 samples and 4864 columns it is rank-deficient
tenfold: the inverse is mostly damping, `U` carries huge off-diagonal entries,
and error propagates wildly across the sweep. **Blocking contains the damage.**

The real variable is therefore SAMPLE COUNT, and every model built in this
project has been on the wrong side of it:

| build | samples | ffn_down columns | Hessian |
|---|---|---|---|
| v2 and everything before it | 3,072 | 11,008 | rank-deficient 3.6x |
| v5 | 16,384 | 11,008 | **full rank** |

`--seqs 64 --seq-len 256` is the first build whose calibration exceeds the
widest tensor's column count. The record already said the held-out GPTQ gain
went 0.87x at 96 tokens to 2.11x at 800 and nobody asked where it stopped.

If that holds, `MAX_H` should then be raised as well -- it is compensating for
a rank deficiency that more samples remove -- but that is a second experiment
and it must not be run until the first one has an answer.

## Quality against Q4_K_M is a dead heat, measured properly

Three sample sizes, same files, same forward pass. The headline moved every
time the sample grew, which is the definition of a number that was not measured:

| tokens | ours vs theirs, residual error |
|---|---|
| 4 | 0.935x -- "ours better by 6.5%" |
| 128 | 0.978x -- "ours better by 2.2%" |
| **512** | **1.008x -- theirs better by 0.8%** |

At 512 held-out tokens, every position scored:

| | llama.cpp Q4_K_M | SML v2 |
|---|---|---|
| correlation | 0.98992 | 0.98976 |
| top-1 agreement | 88.9% | 87.9% |
| top-5 overlap | 4.42/5 | **4.47/5** |

Top-1 differs by 1.0 point against a standard error of 1.4%. **Neither engine is
ahead.** Quality is parity.

### The oracle that proved it was noise

Swapping in an EXACT output head made *llama.cpp worse* at 128 tokens, 89.8% ->
87.5%. A perfect head cannot genuinely hurt; quantization error in the head
partially cancels error from earlier layers, and at that sample size the
cancellation is luck. That single impossible result is what showed the
comparison was under-powered, before any bits were spent chasing a phantom gap.

> A difference smaller than the standard error is not a small win. It is not a
> measurement.

## CORRECTION: on a proper sample, quality is a TIE, not a win

The section below reports quality from **four tokens, scoring only the final
position**. Ernest asked whether the project was therefore done, and the honest
answer required widening the sample first.

128 held-out tokens, every position scored, same forward pass, same files:

| | llama.cpp Q4_K_M | SML v2 | |
|---|---|---|---|
| correlation | 0.98806 | **0.98859** | ours by 2.2% |
| **top-1 agreement** | **89.8%** | 87.5% | **theirs by 2.3 points** |
| top-5 overlap | 4.39/5 | 4.41/5 | tie |
| residual error | 0.15407 | 0.15063 | 0.978x |

**The 4-token figure (0.99660 vs 0.99610, "6.5% better") does not survive.** At
128 tokens the margin collapses to 2.2% on correlation, and top-1 -- the metric
that decides which word comes out -- goes the other way.

So the standing claim is:

| axis | verdict |
|---|---|
| decode speed | **ours, 1.108x**, six order-balanced runs, no overlap |
| quality | **parity** -- ahead on correlation, behind on top-1 |
| bytes/token | theirs, we read 2.4% more |

Faster, tied on quality, slightly larger. That is a real result and it is not
"we beat them".

A single prompt scored at one position is a sample of size one dressed up as a
measurement. It belongs with the other entries in "measurements that faked a
result", and it nearly became a headline.

## SML beats Q4_K_M on quality and speed, at 2.4% more bytes

The corrected allocation, 3B, everything measured on one machine on one day.

| | SML flat v2 | llama.cpp Q4_K_M | |
|---|---|---|---|
| correlation vs fp16 | **0.99660** | 0.99610 | **ours better** |
| residual error | **0.08239** | 0.08823 | **0.934x** |
| top-1 / top-5 | yes, 5/5 | yes, 5/5 | equal |
| decode, order-balanced | **12.33 tok/s** | 11.13 | **1.108x** |
| bytes/token | 1.966 GB | 1.920 GB | 1.024x MORE |
| bits/weight | 5.09 | 4.97 | |
| achieved bandwidth | **24.2 GB/s** | 21.4 GB/s | **1.134x** |

Speed medians do not overlap: our slowest run (12.16) beats their fastest
(11.30) across six order-balanced runs.

**Faster and more accurate, reading slightly more bytes.** Not a clean sweep,
and the honest way to say it is that the win is on the two axes a user feels,
with a 2.4% byte cost. At matched bits the quality edge would be smaller; 2.4%
more bits buying 6.6% less error is a good trade but not a large one.

The engine number is the strong one: **24.2 GB/s against their 21.4**. We move
MORE bytes and still finish first, which is a property of the flat kernel rather
than of the quantizer.

### What fixed the quality, and it was reading their file

Their per-role allocation, which we had been guessing at:

| role | theirs |
|---|---|
| `token_embd` | 6-bit, every one |
| `attn_v` | 6-bit on 18 of 36 layers |
| `ffn_down` | 6-bit on 18 of 36 layers |
| everything else | 4-bit |

We had it backwards on both counts -- 6 bits on `ffn_down` in ALL 36 layers
where they use 18, and the head pinned at 4 where they use 6. On a tied model
`token_embd` IS the output head, so it produces the logits being graded. We were
paying more than they do on the wrong tensor while starving the one that
decides the answer. Correcting it moved top-1 from disagreeing to agreeing and
took correlation 0.99471 -> 0.99660.

### Buying the bytes back does not pay

The obvious follow-up: spend some of the quality surplus to get under their byte
budget and win all three axes. Built it -- `ffn_down` at 6 bits on every 4th
layer instead of every 2nd.

| | bytes/token | correlation |
|---|---|---|
| v2 | 1.966 | **0.99660** |
| v3, leaner | **1.915** | 0.99505 |
| llama.cpp Q4_K_M | 1.920 | 0.99610 |

It lands under their bytes and falls below their quality. Neither configuration
dominates, and the curve around `ffn_down` is steep: 51 MB costs 0.99660 ->
0.99505, which is more quality than the bytes are worth.

**v2 is the better trade** -- 2.4% more bytes for 6.6% less error and 1.108x
decode. Recorded so the next person does not spend another hour rediscovering
that the leaner point is worse.

### A third byte counter was wrong, and this one cost us the byte win

`sml_scale_bytes` counts the fp16 super-scale and the uint8 per-group scale --
and NOT the per-group MIN, which the asymmetric grid has stored the same way
since v6 and which is the same size again.

So **every byte figure this project has published since v6 was low by ~0.6
bits/weight.** The mixed model's "1.871 GB/token, 1.026x fewer bytes" was really
~1.99 and MORE than theirs. Corrected, this model reads 1.966 against their
1.920.

Caught because the moved counter (1.845) disagreed with the static one (1.966)
by 121 MB, and a hand calculation -- 1723 MB of codes plus 241 MB of scales and
mins -- landed on 1.964, agreeing with the static figure. The number that
flattered us was the wrong one.

That is three byte-counting bugs in one day, all three in our favour. Bytes are
the denominator of every bandwidth claim, so they are exactly where a wrong
number does the most damage.

## Measured bit placement beats the fixed recipe by 1.28x, at identical bytes

The one thing a hand-tuned quantizer structurally cannot do. Quantize one layer
at a time on the 3B, everything else exact, and measure the logit damage; then
give the 6-bit slots to the layers that actually hurt.

Measured spread across the 36 layers: **4.54x**. Layer 0 costs 1.2332% and layer
1 costs 3.9760%. "Every other layer" -- which is what the reference mix uses and
what we shipped -- takes 0 and skips 1, and overlaps the measured top 18 by
**3 of 18**.

512 held-out tokens, every position scored, same forward pass, same files:

| | correlation | rel-RMS | top-1 | top-5 |
|---|---|---|---|---|
| llama.cpp Q4_K_M | 0.98992 | 0.13877 | 88.9% | 4.42/5 |
| **SML v7, measured placement** | **0.99405** | **0.10603** | **91.6%** | **4.58/5** |

Residual error **0.10889 against 0.14160 -- 0.769x, 23% better.**

Against our own previous build, which differs in NOTHING except which 18 layers
get the 6-bit slots:

| build | placement | residual vs theirs |
|---|---|---|
| v5 | every other layer | 0.983x |
| **v7** | **measured top 18** | **0.769x** |

**1.28x from placement alone, at identical bytes, identical bit budget and
identical calibration.**

This is the differentiator working as designed. The reference engine applies one
recipe to every model ever released; it has no Hessian, no amplification pass
and no per-tensor width field to act on one with. A measured allocation is a
property of the model in front of you, and it has to be re-measured per model --
which is precisely why a fixed recipe cannot reach it.

## Batch verification is compute-bound, and retiling it made it worse

The speculative ceiling rests on one number: what verifying `nb` tokens costs
against verifying one. Ideal is ~1.0 -- a single weight read serving `nb`
candidates. Measured on the 3B, idle machine:

| nb | vs nb=1 | per token |
|---|---|---|
| 1 | 1.000x | 0.1010 |
| 2 | 1.297x | 0.0655 |
| 3 | 1.703x | 0.0573 |
| 4 | 1.970x | 0.0497 |
| 8 | 3.584x | 0.0452 |

Decomposed: ~0.064 of nb=1 is memory and ~0.037 is compute, with each extra
batch element adding ~0.037. So the weight stream IS amortised correctly and
the cost is per-batch arithmetic plus epilogue.

**Hypothesis: register pressure.** At nb=8 the kernel holds `vsum[8]` plus
`c0..c3` plus `acc[4]` live across the quad loop -- 16 vector registers before
constants, on a machine with 16. Tiling the batch loop to 4 should have bought
them back at the price of one extra pass over the row's codes, which are in L1.

**Wrong, and measurably so:**

| nb | before | after tiling |
|---|---|---|
| 2 | 1.297x | 1.654x |
| 3 | 1.703x | **2.087x** |
| 8 | 3.584x | 3.788x |

Every point worse. The tell is that nb<=4 uses a single tile, so tiling cannot
explain a 27% regression there -- it came from indexing `vsum[b - bt]` with a
computed offset, which stopped the compiler keeping the accumulators in
registers at all. The change intended to relieve register pressure created it.

Reverted. `--verify-batch` re-run and passing on both sides of the revert.

**What the ceiling actually is.** With k=2 at 83% acceptance the expected yield
is 1 + 0.83 + 0.83^2 = 2.52 tokens, costing 1.70 units of verification plus
~0.33 for the 0.5B draft: 2.52/2.03 = **1.24x** over our own non-speculative
speed, which is what the sweep measured. Even with verification made FREE the
same arithmetic gives 2.52/1.33 = 1.89x. So perfecting this kernel is worth
about 1.5x on speculation and no more; it is a real project, not a quick win,
and the honest ceiling should be quoted before anyone spends a week on it.

## Three specs raced, not one

The claim rested on a single model, which is what a reviewer kills it on. It now
rests on three, chosen as **family specs** rather than as three models: each one
exercises a distinct combination of norm, rotary and residual wiring, so a family
that shares the spec is covered by the race that exercised it. 256-512 held-out
tokens, every position scored, reference decoded from THEIR file and run through
the SAME forward pass.

| spec exercised | model raced | quality | decode | file size |
|---|---|---|---|---|
| RMSNorm / halved rotary / GQA 16:2 / QKV bias | Qwen2.5-3B | **0.769x -- 23% better** | **1.144x** | 1.0225x |
| RMSNorm / halved rotary / GQA 32:8 / no bias | Llama-3.2-1B | **0.592x -- 41% better** | **1.034x** | 1.0168x |
| LayerNorm / partial rotary / parallel residual / GELU / biases | pythia-410m | **0.876x -- 12.4% better** | **1.503x** | **0.9981x** |

| | llama.cpp Q4_K_M | SML |
|---|---|---|
| Qwen2.5-3B correlation | 0.98992 | **0.99405** |
| top-1 | 88.9% | **91.6%** |
| Llama-3.2-1B correlation | 0.97902 | **0.99270** |
| top-1 | 85.0% | **89.6%** |
| pythia-410m correlation | 0.97656 | **0.98205** |
| top-1 | 51.6% | **59.8%** |
| top-5 | 3.18/5 | **3.54/5** |

**Quality wins on all three. Speed wins on all three. Size loses on two of three**
-- and pythia is the first model where the SSF file is *smaller* than the GGUF,
by 0.19%, which is a tie rather than a win.

Three things to read honestly off that table:

- **The speed margin is not stable across models**: 1.034x, 1.144x, 1.503x. It is
  a property of the model shape, not of the engine, and must be quoted per model
  rather than averaged. The 1.503x on pythia is the widest and the least
  representative -- a 410m model fits far better in cache.
- **The quality margin does not track size either** (23%, 41%, 12.4%). Error
  feedback buys most where round-to-nearest costs most.
- **pythia's top-1 is 51.6% / 59.8% for BOTH engines**, because a 410m model's
  own next-token distribution is flat. The *ratio* is the result there; the
  absolute is not comparable to the 3B's 91.6%.

Only the 3B reference existed, so their `llama-quantize` was built from source to
produce the Llama and pythia Q4_K_M files. Their format is touched ONLY in
`scratchpad/refq.py`, never in the tree.

### The pythia race was invalid for a day, and the fix names the bug class

The first GPT-NeoX run scored both engines at correlation ~0.873 and top-1
**21-26%** against fp16. Four-bit quantization does not do that, so the "1.007x
tie" it printed was discarded rather than reported.

The cause was **the rotary frequency table built at the wrong width**. GPT-NeoX
applies rotary to only the first `n_rot` of each head's dimensions, so the table
needs `n_rot / 2` entries; the harness built `head_dim / 2` -- 32 where 8 was
correct -- and rotated the un-rotated tail with frequencies belonging to
positions that do not exist:

```python
_half = (fl["n_rot"] // 2) if fl["n_rot"] else (hd // 2)
fr = rope_freqs(th, _half, cfg.get("rope_scaling"))
```

It corrupted BOTH sides equally, which is exactly why it read as a tie and not as
a failure. **A bug that damages both engines equally hides inside a ratio.** That
is a third distinct way for this project to be fooled, alongside a too-small
sample and a mis-counted byte.

## The whole forward pass runs on the GPU, and agrees with the CPU token for token

Every op of `sml_forward` now has a GPU kernel -- activation quantize, 4- and
6-bit matvecs, RMSNorm and LayerNorm, RoPE, attention over a KV cache, SiLU and
GELU, argmax -- built by clang as NVPTX with no CUDA toolkit and loaded through
the driver `nvcuda.dll`. A token crosses PCIe as one embedding row in and one
token id out. There is no libm on this machine, so sqrt, exp and tanh are built
from instructions (`sml/sml-gpu-fwd.cu`, `sml/sml-gpu-fwd.c`, `make gpu`).

Model parsing is not reimplemented: the host reuses `sml_ctx_init`, which
already resolves every tensor, bias, norm and RoPE table for three validated
architectures, and uploads what it found.

Graded against the CPU engine at every position, teacher-forced by the CPU
choice so a single disagreement cannot cascade:

| model | architecture exercised | mean logit corr | worst | top-1 |
|---|---|---|---|---|
| pythia-410m | LayerNorm, GELU, parallel residual, partial rotary, biases | 0.999615 | 0.998638 | 29/29 |
| Llama-3.2-1B | RMSNorm, SiLU, GQA 32:8 | 0.999847 | 0.999731 | 29/29 |
| Qwen2.5-3B | RMSNorm, SiLU, GQA 16:2, QKV bias | 0.999784 | 0.999566 | 29/29 |

The remaining difference is deliberate: the GPU keeps its KV cache in float32
where the CPU stores fp16, and its softmax uses the polynomial exp where the CPU
calls libm.

### Launches were measured before anything was fused

One kernel launch costs 13.4 us here. The first forward issued about 800 per 3B
token -- 10.7 ms of submission alone -- and a synced profile showed the small
kernels (quantize, add, norm, RoPE, copy) costing more as launches than as
arithmetic. Folding quantize and bias into the matvec, the residual add into the
following norm, and RoPE into the KV write took it to 435 launches per 3B token
(195 on the 1B, 267 on pythia), with agreement unchanged.

### Float activations were tried and are not the default

A matvec dotting 4-bit codes against FLOAT activations removes the int8
quantization error (4.1e-03 down to about 1e-06 against an exact reference) but
ran about 3x slower than the int8 kernel even after staging fixes: 34.5 against
109.8 GB/s. The reference engine CUDA decode also quantizes activations to 8
bits, so int8 concedes nothing to it. Float stays available as a quality mode.

### GPU decode against the reference engine: we lose on every model, by 2x to 4.3x

Measured on an idle machine against the reference engine official Windows CUDA
build (b10883, cuda 13.3, `-ngl 99 -p 0 -n 64 -r 2`), interleaved, with the order
reversed in round 2. Ours is 64 greedy tokens after 8 warm-up tokens.

| model | round | SML GPU | reference CUDA | ratio |
|---|---|---|---|---|
| pythia-410m | 1 | 58.58 | 252.12 +- 0.29 | 0.23x |
| pythia-410m | 2 | 50.76 | 224.12 +- 7.27 | 0.23x |
| Llama-3.2-1B | 1 | 51.69 | 106.29 +- 16.26 | 0.49x |
| Llama-3.2-1B | 2 | 28.73 | 76.23 +- 5.56 | 0.38x |
| Qwen2.5-3B | 1 | 16.87 | 38.02 +- 2.14 | 0.44x |
| Qwen2.5-3B | 2 | 9.93 | 29.43 +- 5.98 | 0.34x |

The laptop GPU started round 1 at 79 C and round 2 at 86 C, and BOTH engines
slowed in round 2 -- thermal throttling, which the interleaving spreads across
both rather than charging to one. The verdict does not depend on it: we lose in
both rounds on all three models.

The smallest model loses worst, which points at fixed per-launch and per-block
costs rather than at the weight read. But the 3B, where the memory bus should
dominate, still loses 2.3x: the reference moves about 73 GB/s there and we move
about 33, although our matvec kernel alone reaches 110 GB/s on a big tensor. The
time is being lost outside the part that was optimised.

**Standing on GPU: correct, and not competitive.** Any claim of beating the
reference engine applies to CPU only.

### Gated GPU race: both engines cooled to 70 C before EVERY run

The earlier interleaved races were decided by heat as much as by code: the laptop
3050 loses about two thirds of its speed when hot and the reference throttles
with it. This race waited for the GPU to cool before each run, ours and theirs
alike, and printed the start temperature beside every number.

| model | round | fused | shared | dp4a | reference CUDA | dp4a / ref |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | 1 | 19.32 | 19.70 | **20.16** | 61.61 | 0.327x |
| Qwen2.5-3B | 2 | 19.27 | 19.74 | **20.16** | 61.55 | 0.328x |
| Llama-3.2-1B | 1 | 52.63 | 54.75 | **56.90** | 139.29 | 0.408x |
| Llama-3.2-1B | 2 | 52.63 | 54.45 | **56.28** | 132.95 | 0.423x |
| pythia-410m | 1 | 58.88 | 59.22 | **59.53** | 231.33 | 0.257x |
| pythia-410m | 2 | 59.05 | 58.78 | **61.17** | 241.06 | 0.254x |

- **dp4a is the fastest path on all three models in both rounds**, and becomes the
  default. Shared beats fused in five of six.
- **Cool, the reference does 61.6 tok/s on the 3B.** The 38 measured earlier was a
  warm number, so the true gap is larger than it looked: 0.33x, 0.42x, 0.26x.
- **The matvec arithmetic is not the bottleneck.** Replacing eight extractions and
  four multiplies with dp4a moved the forward only 1-8%, and pythia barely changes
  across all three paths. The cost that dominates is fixed per token, not per weight.

### Host wait mode is not the stall, and heat collapses the GPU sixfold

dp4a is now the default path. With it, the host was tested three ways --
asynchronous, synchronizing after every layer, and after every launch -- in
mirror order. At equal temperature they tie: Qwen-3B 10.64 tok/s async against
10.62 per layer, and per launch slightly slower at 9.9. The earlier 54 vs 39
ms/token gap between the async decode and the synced profile was heat drifting
between the two phases, not an async stall. `SML_SYNC` stays at 0.

The same runs show what heat does without a cooldown gate, with Chrome back on the
GPU: pythia fell from 31.9 to 10.3 tok/s within a dozen runs as the GPU reached
89 C, against 59.5 cool. Every GPU number needs the gate.

**Next hypothesis: per-block overhead.** A pythia token issues 364 launches of up
to 16 blocks each, about 5800 blocks. Its bytes take about 2 ms at 110 GB/s and
launch submission about 4.9 ms, which leaves roughly 10 ms that neither explains --
exactly what a microsecond or two of per-block setup, staging and barrier would
cost. A gated sweep of 1, 2, 4, 8 and 16 blocks per matvec tests it.

## The orchestrator now does something: 100% vs 55.6% on exact arithmetic

Having established that `ACT_VERIFY` was a label that changed nothing, the fix
was to make it true rather than to delete it -- because external verification of
computable answers is the **one accuracy lever this project ever measured as
working** (v1: math 90% -> 100%, overall 72.5% -> 81.2%).

`sml/sml-verify.h` evaluates integer arithmetic exactly, in process. Where it
succeeds the value is handed to the model as fact before it generates; where it
cannot evaluate exactly it **declines**, and a decline leaves the model's own
answer untouched.

18 arithmetic questions, Qwen2.5-3B, same server binary, orchestrator on vs off:

| | correct |
|---|---|
| **with verification** | **18/18 (100.0%)** |
| unaided | 10/18 (55.6%) |
| regressions | **none** |

The eight it turned from wrong to right are ordinary: `47 x 89`, `123 x 456`,
`2^16`, `6250/25`, `384+976+145`, `100-37-8`, `987-654`, `(3+4)*5`. The model
writes a correct-looking derivation and lands on the wrong number; it is not
short of method, it is short of arithmetic.

### Two bugs found by measuring instead of assuming

**The verifier extracted a fragment of a decimal and answered confidently.**
`"3.5 + 1"` returned **6** -- it stopped at the `.`, picked up `5 + 1`, and
computed. That is the exact failure a verifier must not have, since it would
overwrite a correct answer with a wrong one. It now declines any prompt
containing a decimal outright. It also declines non-integer division, division
by zero, overflow, and bounded-exponent blowups like `9^9^9`; 17 cases, 17 pass.

**Gating on keywords meant it fired on 8 of 18.** `orchestrate()` tested for
"compute", "calculate", "evaluate" and code cues, so *"What is 47 times 89?"* --
precisely the case this exists for -- never reached the verifier. The cue for
arithmetic is that it **parses**, not that the prompt used a particular verb.
Asking the verifier IS the test, because it already declines what it cannot do.
Firing went 8/18 -> 18/18 and the result went 83.3% -> 100%.

### What this is and is not

- It is **integer arithmetic only**, and narrow on purpose.
- It evaluates **in process** rather than shelling out to python as v1 did. A
  server that spawns an interpreter on request text is a remote code execution
  surface, and this one binds a socket.
- The reference engine's server has no equivalent, but this is **thirty lines of
  parser**, not an architectural moat. What is defensible is that it ships in the
  engine and is measured.

## The orchestrator's "1326x" is dead: the code cannot produce it

Carried unexamined across many sessions and flagged twice as unverified. Read
the code and it is worse than unverified -- **it is impossible**.

`orchestrate()` is a keyword classifier. It scans the prompt for execution cues
and live-information cues and returns `ACT_VERIFY`, `ACT_TOOL` or `ACT_DIRECT`.
That value is then used in exactly one place: **the `action` field of the
response JSON**. It routes nothing, calls no tool, verifies nothing, and changes
not one token of generation. A label cannot make anything faster.

So the number never came from the orchestrator. It could only have come from the
**response cache**, which is a real mechanism and is now measured.

### What the response cache is actually worth

Qwen2.5-3B through `sml-server`, six distinct questions, 32 tokens each, then the
same six again:

| pass | total | median/turn | cache |
|---|---|---|---|
| cold | 17.51 s | 3.303 s | 0/6 |
| repeat | **0.057 s** | 0.012 s | 6/6 |

**308x on an exact-repeat workload.** Answers byte-identical, and a hit reads
**zero** model bytes.

That figure is honest and nearly useless on its own, because it describes a
workload of pure repeats. What it is worth depends entirely on the repeat rate:

| exact-repeat rate | mixed workload |
|---|---|
| 10% | **1.11x** |
| 30% | **1.43x** |
| 50% | **1.99x** |

Report the mixed number, or report the repeat rate beside the 308x. Quoting
"308x" for an assistant is the same error as quoting a quality win measured on
four tokens.

Three things this does NOT permit saying:

- It is **not inference speed**, and must never be blended with decode tok/s.
- It is **not an advantage over the reference engine's architecture**. A response
  cache is thirty lines; they could add one this afternoon. What is ours is the
  decode underneath it.
- The cache is keyed on an **exact hash** of prompt, token budget and depth. One
  different character misses. There is no semantic matching.

**"1326x" is withdrawn permanently.** It is not a number that was measured badly;
it is a number the code has no path to produce.

## SML runs on the GPU, and the win came from the activation, not the weights

The machine has an RTX 3050 Laptop that had been disabled in BIOS. Turned on:
**176.0 GB/s** theoretical peak against the CPU's 28.4, 4.29 GB of VRAM, 16 SMs.
All three raced models fit.

There is **no CUDA toolkit on this machine** -- no `nvcc`, no import library, no
headers; the "CUDA v11.8/v12.0/v13.0" directories are fragments and VS2022 is
empty. The path used instead needs none of it:

- **clang emits PTX** (`-x cuda --cuda-device-only -nocudainc -nocudalib`), so
  `threadIdx` and friends are NVVM intrinsics and no `cuda_runtime.h` is read.
- **`nvcuda.dll` ships with the DISPLAY DRIVER**, so the dozen driver-API entry
  points are bound with `GetProcAddress` at runtime.

`make gpu` builds it. It is deliberately not part of `make all`.

### Four kernels, and only the fourth was asking the right question

| | design | GB/s |
|---|---|---|
| v1 | thread per group, block per row, hand-rolled half decode | 22.6 |
| v2 | thread per group-PAIR, so each byte is read once | **17.0** |
| v3 | warp per row, shuffle reduction, native `cvt.f32.f16` | 18.3 |
| v4 | **activation staged in shared memory** | **109.8** |

v1 read every byte twice: groups 4q and 4q+1 are the low and high nibbles of the
SAME 32 bytes. v2 fixed that and got **worse**, which was the useful result --
halving the traffic and losing means traffic was not the constraint. v3 removed
2048 block-wide reductions and a data-dependent `while` loop in the half decode,
and barely moved, so neither of those was the constraint either.

What all three missed: the weights are 13 MB, but **every warp re-read the whole
activation, one byte at a time**. 2048 rows x 11 KB is 22 MB of activation
traffic against 13 MB of weights -- the thing being streamed was never the model.
Staging it once per block took the kernel from 18.3 to **109.8 GB/s**, 62% of
peak.

That is the ladder lesson in a different place: **three rounds of tuning sat on
top of the wrong bottleneck**, and the measurement that identified it was the one
where the obvious fix made things worse.

### What it measures, and what it does not

| tensor | shape | CPU AVX2 | GPU | vs our CPU |
|---|---|---|---|---|
| `ffn_down` | 2048 x 11008 | 13.5 GB/s | **109.9** | **8.17x** |
| `ffn_up` | 11008 x 2048 | 14.9 | 109.8 | 7.35x |
| `ffn_gate` (blk 10) | 11008 x 2048 | 15.2 | 108.6 | 6.60x |
| `attn_q` | 2048 x 2048 | 15.8 | 108.4 | 5.34x |

Every one graded against the CPU kernel's own output at **rel-RMS 3.2e-07**,
because a kernel that reads the packing wrong still produces confident numbers.

**Three things this is NOT, stated so the number is not over-read:**

1. **It is a KERNEL, not a decode path.** Attention, norms, sampling and the KV
   cache are all still on the CPU. End-to-end tok/s on GPU is **not measured**.
2. **It has not been raced against their CUDA backend.** Building that needs
   `nvcc`, which this machine does not have. Their Q4_K CUDA kernels are mature;
   assume they are in the same 60-70%-of-peak band until measured.
3. **8.17x is against our own CPU kernel**, not against them. It is the right
   number for "does the format work on a GPU" and the wrong one for any public
   comparison.

What this does establish: **the SSF format needs no change to run on a GPU.** The
same bytes, the same group-planar nibble layout, the same two-level scales and
per-group min, read by a different kernel. And the quality advantage is in the
format, so it transfers unchanged.

### The engine did not build on any machine without AVX2

Found while checking what a stranger would hit on release day. Three separate
reasons, none of which could show up here, because the only machine this has
ever been built on has AVX2 and therefore never selected the other branch:

- `sml-run.c` included `<immintrin.h>` unconditionally, which does not exist
  off x86 -- Apple Silicon fails before reaching any of the portable code.
- The portable arm of `sml-kernel.c` **had not compiled since the per-group min
  landed**: it called `sml_matvec_rows` with ten arguments where twelve were
  expected and accumulated into an undeclared `acc`. Dead code that no build
  ever selected.
- The five FLAT functions -- the shipping layout -- were defined only inside the
  AVX2 arm. Three of them carry no intrinsics at all and were simply in the
  wrong place; two needed scalar twins.

Fixed. `sml_matvec_rows_flat4` and its batched twin now have scalar
implementations, and the portable path agrees with the format spec at
**7.2e-07** (4-bit) and **2.6e-06** (6-bit) and generates byte-identical text.

**It costs 2.5x:** 26.7 tok/s against 67.5 on pythia, six threads. So the speed
result in this file is an **AVX2 result**. The reference engine ships NEON,
Metal and CUDA backends; on anything that is not an AVX2 x86 part, assume we
lose. Nothing here has been measured on ARM.

### The CLI could no longer print a byte ratio it had not measured

`sml-cli` had `11.53 tok/s` and `1.920 GB/token` compiled in as "the baseline".
Those are the **Qwen2.5-3B** figures. Run on pythia, the binary printed
**"8.72x fewer bytes"** -- comparing a 410m model against a 3B model's baseline,
a number with no meaning that looked like the best result in the project.

A baseline belongs to a model on a machine, never to a binary. It is now
`--baseline TOKS_PER_S[,GB_PER_TOKEN]`, and with nothing supplied the CLI prints
its own absolute figures and no ratio at all. This was found while writing the
pythia row of the table above -- one day before a public release that would have
carried it.

### Three post-conversion artifacts, found in one session

Reading their format to get a reference exposed three places their stored
representation differs from the HF checkpoint. Each produced statistically
PERFECT values in the WRONG ORDER -- matching rms and absmax, correlation near
zero -- which is the signature to watch for:

| artifact | symptom | fix |
|---|---|---|
| Q/K permuted for the llama family, so interleaved RoPE reproduces `rotate_half` | `attn_q` 139.90%, `ffn_down` 1.82% | invert the permutation -> 7.44% |
| GPT-NeoX QKV stored as plain concat `[Q;K;V]`, where HF stores it per head | 138.88% head-layout vs 3.65% concat | split as concat |
| their rope-type table lists LLAMA as interleaved | would invert 31 architectures | do not copy; read the checkpoint's config |

All three are artifacts of THEIR converter. None transplants to a reader of HF
checkpoints, and each would have produced fluent nonsense rather than an error.
Also added Q5_K to the reference decoder, which their Q4_K_M mix uses for fused
QKV.

### The first attempt said we won by 7x, and was discarded

It read llama.cpp at correlation **0.379** and top-1 **4.5%**. Their quantizer
does not destroy models; the reader was wrong.

Diagnosis, and the numbers are the whole lesson: decoded rms **0.03606** against
fp16 **0.03606**, absmax 0.6686 against 0.6758 -- and correlation **-0.04**.
Right values, wrong ORDER. And only `attn_q` and `attn_k` were affected, while
`ffn_down` and `token_embd` read 1.8%.

That is their converter's **Q/K permutation**, which exists so that interleaved
("NORM") RoPE reproduces HF's `rotate_half`. Qwen2 is converted for halved
("NEOX") RoPE and is NOT permuted -- which is why the 3B decoded correctly from
the first attempt and nothing looked wrong. Inverting it took `attn_q` from
139.90% to **7.44%**, exactly the expected Q4_K error.

Two independent findings converged on one fact within an hour: their rope-type
table and their stored weight layout are both **post-conversion artifacts**.
Neither transplants to a reader of HF checkpoints, and copying either would have
inverted the convention for 31 architectures -- the same bug this project already
had once, in the opposite direction.

## Thirteen measurements that faked a result

Recorded because each looked fine in isolation, and because the discipline that
caught them is the reason to trust anything above.

1. **Synthetic activations** put importance weighting at 2.35x; real calibration gave 1.07x.
2. **A group-size change scored on quality only** — better quality, 2x bandwidth loss, unnoticed for hours.
3. **A verifier comparing permuted against unpermuted columns** — yields ~√2 and looks exactly like a packing bug.
4. **Error feedback scored on its own training data** showed 4.80x; held out, the same run was 0.87x.
5. **A benchmark on a loaded machine** read 5.39 ± 0.62 against a true 10.71 ± 0.22. Wide variance was the tell.
6. **A kernel bench at 192 MB** showed 24.4 GB/s; at realistic scale it is ~20.
7. **An oracle that skipped a file header** scored a correct engine at 0.10, with every layer matching at 1.00000.
8. **Mode-coupled rounding** scored 6.12x on its calibration set and 0.55x held out — a regression that would have shipped as a 6x win.
9. **Large pages measured against a cold page cache** read +15%; the large-page path itself warms the cache for the mmap runs that follow. Warm on both sides it is 0.968x.
10. **A quality win measured on four tokens** read 6.5%; at 128 tokens 2.2%, and at 512 we were 0.8% behind. Published, then withdrawn.
11. **Three byte counters all wrong in our favour** reported 1.413 GB/token against a true 1.871 — `sml_scale_bytes` omitted the per-group min, so every figure since v6 was low by ~0.6 b/w.
12. **A rotary table built at `head_dim/2` where partial rotary needs `n_rot/2`** corrupted BOTH engines equally and so printed a 1.007x *tie* rather than a failure. A bug that damages both sides hides inside a ratio.
13. **A baseline compiled into the binary** — 11.53 tok/s and 1.920 GB/token, the 3B's figures — made `sml-cli` print "8.72x fewer bytes" for a 410m model. A baseline belongs to a model on a machine, never to a binary.

Grade the final answer, never the mechanism. Correctness is checked before
speed, and a read depth that fails correctness reports VOID rather than a
number.
