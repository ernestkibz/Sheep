# Models: what Sheep runs, and how any model gets there

## Any model, with nothing to set

A person gives Sheep a model the way they would name it: a Hugging Face ID, a folder, or a `.gguf` file.
Sheep then does every step itself and shows each one in **Settings › Activity › What Sheep did for you**:

1. **Find the original.** A GGUF, GPTQ, AWQ, EXL2, MLX or ONNX page names the model it was made from
   (`base_model` in its card), or carries that model's name with a format suffix. Sheep follows it to the
   original weights and builds from those, because a copy compressed by someone else has already lost
   quality (`mlmi/backend/hf_source.py`).
2. **Convert when there is no original.** A fine-tune published only as GGUF, a GPTQ or AWQ copy, or Meta's
   raw `.pth` checkpoint is converted to standard weights first (`tooling/sml_gguf.py`,
   `tooling/sml_unpack.py`). The result keeps that copy's quality, and Sheep says so.
3. **Fit or fall back.** An architecture Sheep's engine has not met is fitted to it from its own config and
   checked against the maker's code (`tooling/sml_autofit.py`). If nothing fits, the model runs in
   **compatibility mode** with the maker's own code (transformers), slower but correct, and Sheep moves it to
   its own engine later when a fit is found (`mlmi/backend/compat_server.py`).
4. **Build for this computer.** The level (Q2 to Q8, or the weights as published) is chosen from this
   machine's memory and devices; the person may pick another (`mlmi/backend/levels.py`).
5. **Place and check.** The orchestrator places the model on the graphics card, the processor or both, then
   checks it: a short first exam, check questions with known answers, and real jobs in Sheep's apps, each
   checked by opening what the model made (`mlmi/backend/orchestrator.py`).
6. **Repair.** A model that starts to misbehave is repaired by itself (a shorter context, another device,
   another chat frame), and the repair is shown.

Refusals are in words and say what to do: a LoRA adapter (not a whole model), a page with no weights, a format
with no original named, a model made for speech or pictures offered as a chat model, a disk too full to
finish the download.

## Formats

| Given | What Sheep does |
|---|---|
| An original Hugging Face checkpoint (safetensors or PyTorch .bin) | Builds it |
| An F16, BF16 or F32 GGUF; Meta's raw .pth | Converts it, then builds at any level |
| An already-quantized GGUF (Q2 to Q8, IQ), GPTQ or AWQ copy | Packs it into SSF as it is, at its own width (Q4_K_M at 4/6-bit, Q8_0 at 8-bit); the other levels are locked, and the original is named for anyone who wants to choose one. The level is read from the file name, so keep it in the name (model-Q8_0.gguf); a GGUF whose name gives none is treated as 4-bit |
| An EXL2, MLX, ONNX or bitsandbytes copy | Refused with the name of its original, which Sheep builds at any level |
| A ready Sheep file (.ssf) on a Hugging Face page | Downloads it; no build |
| An architecture the engine has not met | Fits it, or runs it in compatibility mode |
| A LoRA adapter; an EXL2, ONNX or Core ML copy with no original named | Refused, in words |

## Publishing

A model built in Sheep can be published to the person's Hugging Face account (**Settings › Models › Publish**),
private first, with a model card that names the original and its licence, and the original's LICENSE file.
Only licences that allow sharing are published (`mlmi/backend/publish.py`). Publishing needs a Hugging Face
token of type **Write**. Anyone can then add that page in Sheep, and it downloads ready to run, with no build.

## Architecture support

`tooling/sml_archs.py` rosters **140 architectures** (the reference's list plus
transformers' names for them). The status counts below are computed from the
current roster; an `ok` status is a build-policy classification, not proof that
every checkpoint in that row has been independently tested.

| | count | meaning |
|---|---|---|
| **raced** | **3** | measured head-to-head against the reference engine |
| graded on a real checkpoint | 9 | against full precision in transformers |
| **run as a family head** | **91** | random-weight models built from transformers' own code for those architectures, compared logit for logit (below); 89 pass |
| supported from config only | 11 | settings read from config, never run: their code is not in transformers (remote code), so no head was built |
| **roster marked `ok`** | **91** | allowed by the build gate; this status count is not the same as a validated-checkpoint count |
| gap | 17 | the missing engine feature is **named** (hybrid state-space, latent attention, attention sinks, fused experts...) |
| out of scope | 24 | encoder-only, pure state-space, diffusion, vision-only |
| unknown | 8 | newer than transformers 4.57 here |

The three races are included in the ten real-checkpoint-grade families. Evidence groups
overlap, so do not add their counts to derive the roster's `ok` count.

**Family heads (2026-09-28, `tooling/sml_family_heads.py`).** Most architectures
differ in a handful of behaviours -- norm, rotary convention and width, positions,
activation, gating, residual wiring, windows, Q/K norms, expert routing. For each
model_type the harness builds a small random-weight model from transformers' own
modelling code (no download, no remote code), Sheep builds and runs it stepped and
batched, and the last-position logits are compared with transformers' in fp32. A
head passes when its error is what transformers itself loses on 6-bit weights for
that family. 89 of 91 pass in the small shape, and 178 of 182 counting the shape of a big checkpoint too (`tooling/family_heads.json`); the two that do not are BitNet (its sub-norms) and Llama 4 (fused experts), and the build refuses both by name.

The heads found that most rows once marked supported from config were wrong:
interleaved rotary in seven families (Cohere, ERNIE, GLM, Helium...), Gemma's
GeGLU / (1+w) norms / embedding scale, Phi's and GPT-J's shared LayerNorm and head
bias, Mistral's ignored sliding window, HunYuan's Q/K norm order -- and two tensors
the builder dropped **silently** (HunYuan's Q/K norms; BLOOM's FFN norms in a
transformers-saved checkpoint). The builder now refuses any tensor it cannot place.
All are fixed, and the same pass added per-layer windows and rotary (Gemma 2/3,
Cohere2, SmolLM3, EXAONE 4, GPT-Neo), attention softcap, relu^2, LayerNorm Q/K norms,
Falcon's three Q/K/V packings, MPT, StarCoder-1, OLMo 1, and expert routing for
Mixtral, Qwen2/3-MoE, OLMoE, HunYuan-MoE, Granite-MoE-shared and the
DeepSeek-V3 style (GLM-4.5-MoE, dots.llm1, ERNIE-4.5-MoE). New architectures run
on the CPU; the GPU path refuses them rather than approximate.

**The build refuses an unsupported model by name rather than running it.** That
is deliberate: every architecture bug in this project produced fluent nonsense
rather than an error, so a wrong answer is more expensive here than a refusal.

Ten families now have at least one recorded real-checkpoint grade: Llama,
Qwen2, GPT-NeoX, Qwen3 (top token matches full precision on 17 of 20 prompts),
Granite MoE (19 of 20), GPT-2 (19 of 20), OLMo 2 (full precision matches HF
20 of 20; quantized weights match 10 of 20), BLOOM, SmolVLM, and GPT-Neo. The
one-hour audit pinned `EleutherAI/gpt-neo-125m` at revision
`21def0189f5705e2521767faed922f1f15e7d7db`, converted it in 133 seconds, and
compared 20 prompts: the exact/stepped path matched the full-precision top
token on 6/20 prompts (mean relative RMS 0.1522). This is converter/parity
evidence, not a broad behavior-quality claim. Its temporary 529 MB source and
110 MB `.ssf` were hashed and removed after grading. The existing Llama grade
has local content hashes but no upstream revision. Eight older family grades
remain without retained revision/digest pins. Family-head and config-only
results are separate evidence and do not establish real-checkpoint support for
other roster entries. The earlier GPU grades cover Llama, Qwen2, and GPT-NeoX;
the other seven family grades are CPU-only.
