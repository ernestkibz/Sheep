# How the findings were actually found

Not a style guide. These are the specific moves that produced results in this
project, each with the measurement that earned it a place. Most were learned by
being wrong first.

Written down because the SML quality gap is the one problem still open, and the
next attempt on it should start from here rather than from a new idea.

---

## 1. Ask the ceiling before building anything

> **What does this reach if it works perfectly, and have I already measured
> that number?**

One minute of arithmetic, and it killed more proposals than any experiment.

- *Plane-3 selection* is bounded by buying plane 3 always. Measured: 0.99231
  against a 0.99610 target. Dead before a line was written.
- *Hot-precision in LLC* is bounded by promoting everything. Same bound, same
  answer.
- *Certified block skipping* was bounded by its own oracle: 10% of blocks free.

A mechanism whose perfect case is already in `RESULTS.md` needs no experiment.
Ask this first, every time.

## 2. Oracle before predictor

Measure the upper bound with **perfect knowledge**, then decide whether a real
predictor is worth building.

Contextual sparsity died this way: with an oracle that knew the true
magnitudes, keeping 40% of rows still left 13–20% error. No predictor can beat
knowing the answer, so no predictor was written.

The inverse also holds. The self-draft oracle — making the target model draft
for *itself*, which must accept 100% by construction — accepted 83%, and that
gap exposed two cache bugs. Without it the 34% acceptance would have been filed
as "weak draft model" and shipped.

## 3. Hold out, always, when anything is fitted to data

The single highest-yield check in the project.

- GPTQ scored **4.80x** on its own calibration and **0.87x** held out.
- Mode-coupled rounding scored **6.12x** on calibration and **0.55x** held out —
  *worse than doing nothing*, and it degraded as capacity grew, which is the
  textbook overfitting signature.

Without the second number, the second one ships as a 6x win and is a 1.8x
regression.

**And ask whether the thing being fitted is even stable.** `H = XᵀX` is a second
moment that converges with more data. `JᵀJ` is a property of the model *running
one input* — attention patterns and gates change it. Fitting it once cannot
work, and no amount of calibration fixes that. Structure existing is not the
same as structure transferring.

## 4. Grade against the reference implementation, never your own

Our numpy pass and our C engine agreed with each other while both were wrong,
because they encode the same reading of the architecture. Grading against
`transformers` — the definition — found what agreement could not.

Same rule for names: the RoPE convention was inverted for five architectures
because ggml calls the *halved* layout "NEOX rope" after the family it met it
in. Reading their source and carrying the **name** across gives the wrong
mapping. Check each architecture's own reference implementation.

## 5. Suspect the instrument before the system

A correct engine scored **0.10** because a throwaway oracle read past an 8-byte
header. Every layer matched at 1.00000 and only the final logits were wrong.

> **"Everything matches until the last step" is the signature of a reader bug,
> not a maths bug.**

Learn the signatures. Comparing permuted against unpermuted columns yields
**√2** every time and looks exactly like a packing bug — that one has now
appeared twice.

## 6. Never extrapolate a whole-model cost from one sample

The recurring error, three times over:

- One tensor said the column permutation was worth 0.6%. The full 3B said
  **0.0038 correlation** — larger than the entire gap to Q4_K_M.
- The GPTQ probe sampled layers 0/5/10 and predicted 2.11x. Delivered **1.23x**,
  because the gain falls with depth and layer 35 sits near 1.1x.
- Random-direction probing measured anisotropy at **2.65x**; the true eigenvalue
  spread was **218x**. In 896 dimensions random directions concentrate, so
  sampling systematically understates.

If the decision is model-wide, measure model-wide, or sample the *deep* end.

## 7. Predict the magnitude, not just the direction

A theory that says "this should help" is cheap. One that says "this should help
*less than the last one, because those kernels were already efficient*" is a
rule.

Shared-activation grouping was tested that way: Q/K/V gave 1.026x, and gate/up
was predicted to be worth less **before** it was built. Measured 1.030x. That is
what promoted it from an anecdote to the format's organising principle.

## 8. When a fix degrades something it never touched, the fix is the bug

An OpenMP `num_threads` clause aimed at 256-row tensors cost **18.9 → 14.3
GB/s** on the wide tensors it was not meant to affect. The clause was making the
runtime rebuild its team on every call. The unrelated regression was the whole
diagnosis.

## 9. Vary the confound on purpose, then vary it again

Decode ratio, measured four ways on the same code:

| condition | ratio |
|---|---|
| both engines at 6 threads | 1.19x |
| each at its own best thread count | 0.985x |
| freshly booted | 1.18x |
| after sustained load | **0.99x** |

Every one of those is a real measurement and only the last describes real use.
The confounds were **thread count**, **run order** (we always ran second and
inherited a hotter CPU, worth ~3%), and **thermal state**.

> A single number is a claim about a condition you did not vary.

## 10. Report what decided it, not what you concluded

`groups: 48 declared by the file, 0 inferred by the engine` — because a file can
declare relationships while the runtime quietly ignores them, and the difference
*was* the claim. Assertions about your own system are the ones to instrument.

---

# Where the ideas came from

The rules above stop you being wrong. These are the moves that produced
something. Also earned, also with the finding attached.

## A. Re-open dismissed findings when the target moves

`attn_k`/`attn_v` ran at 3.8 GB/s against 22 for everything else. That sat in
`RESULTS.md` for weeks, correctly dismissed: *"they are 6% of the time -- fixing
them is worth 6%, not 26%."*

True when the gap was 26%. **We were losing by 2%.** The same 6% was suddenly
three times the size of the problem.

> A finding filed as too small is filed against a target. When the target moves,
> re-read the file.

That single re-read produced the fusion, which produced the only engine win.

## B. Test how a thing responds to a resource, not how fast it is

Instead of "is this kernel fast", ask "what happens when I give it more".

```
threads      1      2      3      6
256 rows    6.6    8.0    4.7    3.8   <- worse with more
2048 rows   6.5   11.3   17.0   20.7   <- normal
```

**Performance falling as resources rise is diagnostic gold.** It said the kernel
was fine and the coordination was the cost — which no absolute measurement of
either would have revealed. Two minutes, no code, complete diagnosis.

## C. When two of your own facts contradict, that is the finding

Asymmetric quantization produced *much better weights* and *worse output*:

```
weight error   0.146 -> 0.088   better
output corr    0.974 -> 0.892   worse
```

The instinct is to call it noise or a bug. It was neither. It meant fidelity and
error *placement* are orthogonal — the grid decides how much error a group
carries, the feedback decides which columns absorb it. That distinction directed
everything after it.

> Contradictions between two things you measured are information. Only
> contradictions with things you assumed are mistakes.

## D. Decompose the objective until the parts disagree

`tr(EᵀMEH)` — writing it out showed `H` weights **columns** and `M` weights
**rows**. GPTQ solves rows independently, so a diagonal `M` changes nothing and
an off-diagonal one breaks the solver.

That killed the entire Jacobian family analytically, in ten minutes, before an
experiment. It also said exactly which sub-case *could* work (low-rank coupling)
and what to measure to check (the off-diagonal share: 99.3%).

> Algebra is the cheapest experiment available.

## E. Ask what the other system structurally cannot do

Not "how do we go faster" but **"what must they do that we don't, and what can
we express that they can't?"**

- They cut precision at conversion and can never revisit it → `--depth auto`.
- Their file says where bytes are; it cannot say which tensors belong together →
  shared-activation groups.

This is the only line of questioning that produced *capabilities* rather than
percentages. Percentages get matched. Capabilities have to be designed for.

## F. Make the theory predict a second case, with a size

After Q/K/V grouping gave 1.026x, the principle was made to predict *before*
building: gate/up should also group, and should be worth **less**, because those
kernels already ran at 22 GB/s and had little dispatch overhead left. Measured
1.030x.

> Predicting the direction is a hunch. Predicting the magnitude is a rule.

This is also what makes a negative safe to trust: by the thirteenth mechanism
the theory predicted its own failure first, which is how you know a search is
finished rather than merely unlucky.

## G. Ask what a number is a function of

The decode ratio was never a property of the code. It was a function of thermal
state:

```
freshly booted     1.18x
partly warm        1.03x
heat-soaked        0.82x
sustained          0.99x
```

Once the question becomes *"what is this a function of"*, you vary that thing
instead of averaging over it — and the variation is the result. Here it produced
a better claim than any single ratio: the advantage is real and thermally
limited, because 6 threads throttle where 3 do not.

## H. Spend the cheapest decisive experiment first

Ranked by information per minute, tonight:

| minutes | what it settled |
|---|---|
| 2 | thread sweep — diagnosed the 3.8 GB/s completely |
| 10 | algebra on `tr(EᵀMEH)` — killed a whole family |
| 15 | ceiling arithmetic — killed three mechanisms |
| 80 | a model build — produced one reverted change |

The ordering is not an accident. **Reach for arithmetic, then a probe, then a
build.** Most ideas die before the build if you let them.

---

## Tomorrow

Three components are marked ✅ or ❌ and none of those marks means *closed*.
A ✅ means measured; a ❌ means measured too. Apply rule 1 to each candidate
before building it, and rule A to each component before believing it is done.

### Orchestrator — re-open speculative decoding (highest value)

Filed as *"sound, 85% acceptance, but a wash"*, with the reason recorded
exactly:

> the per-round work outside the shared weight read — k gathers, k activation
> quantizations, attention walked serially — scales with k. That work was ~5% of
> runtime at batch 1.

**That constant moved tonight.** Fusion collapsed six dispatches per layer into
four and removed redundant gathers; the attention walk is now SIMD. Acceptance
was never the problem — 77–85% is healthy. The old sizing put k=2 at 1.27–1.39x
if the overhead were gone.

This is rule A exactly: a finding filed against conditions that have since
changed. Ceiling ≈1.4x, and the failure was implementation cost rather than
anything structural, so it is a genuine re-open rather than a retry.

### SSF — put the quality ladder in the file

`--depth auto` fits by **bytes**, and then a floor of depth 2 is hardcoded in
the engine because depth 1 agrees with depth 2 on the top token only 9.4% of the
time.

That floor is an engine assumption about a file property. The file should carry
its own measured correlation per depth, written at build time, so the runtime
picks *the cheapest depth meeting a quality target* rather than *the deepest that
fits* — and a model whose depth 1 happens to be usable can say so.

Same principle as groups: measured guidance in the file, not assumptions in the
engine. Ceiling on speed ≈0; this is a capability and a correctness fix, so do
not let it compete with work that moves a number.

### Sheep — sustained, not peak

Tonight's real finding: the decode advantage is a function of thermal headroom,
because 6 threads throttle where llama.cpp's 3 do not.

```
freshly booted   1.18x      sustained (reps 21-24)   0.99x
```

The thread count was chosen on a cold machine by measuring peak. The open
question is which count maximises **sustained** throughput — measurement in
`scratchpad/sustained.sh`, rotating counts so each meets a similar thermal
state. Report sustained medians, and never a cold-machine number alone again.

### SML — the gap, restated

0.0027 correlation at matched traffic. Thirteen mechanisms closed, most by
rule 1, and by the last few the theory predicted its own failure first — which
is how a finished search differs from an unlucky one.

**It is missing information, not misapplied information.** Scheduling guidance
cannot add bits, which is why the Shepherd primitives paid for Sheep and did
nothing here.

Before any fourteenth idea, on paper:

1. What is its ceiling, and is that number already in `RESULTS.md`?
2. What is it fitted to, and does that thing transfer to an input it has not
   seen?

Question 2 killed the two strongest candidates. Both had real structure —
`JᵀJ` is 99.3% off-diagonal with a 218x eigenvalue spread, which is as
exploitable as structure gets. Neither had structure that was the same for the
next sentence.
