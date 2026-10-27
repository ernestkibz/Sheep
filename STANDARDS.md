# Sheep / ML.M.I — standards and process flows

One page per part of the system: what it owns, what goes in and out, and the
gate it must pass before the next step may use its output. New code follows
these flows. A change that needs a new flow adds it here first.

## 1. The layers and who owns what

| Layer | Owns | Lives in | Never does |
|---|---|---|---|
| **SSF** (format) | how a model is stored: tensors, widths, scales, vocabulary, template, stops, provenance | `sml/FORMAT.md`, `tooling/sml_format.py` | runtime decisions |
| **Builder** | original checkpoint → SSF (the recipe) | `tooling/sml_build.py`, `sml_hf.py`, `sml_archs.py` | guess an architecture it has not been graded on |
| **Engine (SML)** | running an SSF on CPU and GPU, correctly and fast | `sml/sml-run.c`, `sml-kernel.c`, `sml-gpu-*` | choose placement or modes (it offers mechanisms) |
| **Orchestrator** | every runtime decision: placement, split, modes, context, threads, local vs Nebius, guards | `sml/sml-server.c` (+ `tooling/sml_bridge.py` for Nebius) | expose a knob to the user that it can decide itself |
| **ML.M.I backend** | agent, tools, workspace, model registry, the app's API | `mlmi/backend/` | talk to model files directly (it goes through the orchestrator) |
| **ML.M.I app** | everything the user sees | `mlmi/desktop/` | hold state the backend does not know about |
| **Proof** | races against llama.cpp, grades against originals | `tooling/sml_*_race*.py`, `sml_hf_grade.py`, `sml_family_heads.py` | change what it measures to win (see §9) |

## 2. Flow: a model is added (intake)

```
source (HF id or folder) -> licence -> architecture -> tokenizer & template -> BUILD -> GRADE -> REGISTER -> PROFILE -> ready
```

| Step | Input | Output | Gate (stop if it fails) |
|---|---|---|---|
| Licence | model card | licence id + duties (e.g. "Built with Llama") | unknown licence: flagged, never demoed |
| Architecture | `config.json` | roster row (`sml_archs.check`) | not "ok": refuse with the reason |
| Tokenizer | tokenizer files | vocab size, special-token ids | vocab size ≠ embedding rows, or special ids outside the vocab: refuse |
| Template | `chat_template(.jinja)` | the model's own template | missing: plain-text frame, flagged amber |
| Stops | generation/tokenizer config | EOS, EOT, end-of-turn markers | none found: flagged amber |
| Build | the above | `.ssf` (§3) | builder exits non-zero or a tensor is all-zero: refuse |
| Grade | `.ssf` + original | correlation, top-1 (§4) | below threshold: never registered |
| Register | all of the above | one registry entry (§5) | — |
| Profile | registry entry + this machine | the orchestrator's plan (§6) | canary fails: red, offer rebuild or Nebius |

## 3. Flow: build (the recipe)

- **Input:** the original checkpoint (safetensors or torch .bin); a GGUF, GPTQ, AWQ or MLX page leads to the
  original it was made from. Only when no original exists is the copy converted first: a GGUF (C1), a GPTQ or
  AWQ copy unpacked to float16, or Meta's raw .pth (C6); the quality is then the copy's. EXL2, ONNX and Core ML
  are refused by name.
- **Recipe, one for every model:**
  - flat layout, 6-bit group fields;
  - Q4_K_M-shaped allocation with the 6 bits on the embedding;
  - a byte budget (≤ the reference file when racing);
  - field search radius 2.
- **Writes:** tensors; the model's chat template and stop tokens; provenance
  (source, licence, recipe, calibration).
- **Gate:** every quantized tensor has non-zero scales; the file opens in the
  engine; the size is ≤ the budget.

## 4. Flow: grade (before anything is offered)

- **Reference:** the original weights (transformers fp32, or llama.cpp's f16
  when racing).
- **Measures:** logit correlation and top-1 over fixed prompts. For races, KL
  with a paired CI.
- **Pass:** within the family's 6-bit sensitivity band (`sml_family_heads.py`
  rule), and top-1 agreeing on the canary prompts.
- A model that has not passed its grade is never shown as available.

## 5. The model registry (one entry per model)

`~/.mlmi/models.json`, one record each:
- ssf path; source; licence and duties;
- template id and stops;
- grade (numbers and date);
- build recipe;
- the orchestrator's last plan per machine;
- history (loads, speeds, faults), which feeds the dashboard's model log.

## 6. Flow: orchestrate (the admin)

**On load:**
1. Probe the device: free VRAM (NVML), RAM, AC or battery, heat.
2. Plan the placement: all on the GPU, a GPU/CPU split (k layers), or CPU only
   if the GPU refuses a layer type.
3. Choose the modes: resident / streamed / re-made weights, context size,
   prompt batch, threads.
4. Run the canary prompt through the degeneration guard (§7).
5. Lamp: amber while planning, green when in sync, red on a failed guard.

**While running:**
- Watch replies (§7).
- Re-plan when VRAM, power or heat changes.
- Route a turn to Nebius only in Sheep + Nebius mode, and only by the routing
  rules.

**Always:** every decision and its reason goes to the log; nothing here is a
user setting.

## 7. Flow: serve a request

```
request -> render the model's OWN template -> engine (streaming) -> guard -> stops -> reply
```

- **Guard trips:**
  - the same n-gram repeating past a limit;
  - runs of punctuation or replacement characters;
  - an entropy collapse.
- **On a trip:** stop the stream, re-check template and stops, retry once,
  then mark red and offer a rebuild or Nebius.
- **Tools:** the model's tool calls are parsed per template; argument types
  are coerced to the schema.

## 7b. Flow: an agent job

```
request -> route tools -> render -> engine -> mend the call -> scope guard -> tool -> done?
```

| Step | Rule | Why (measured on Llama 3.2 3B, 2026-09-30) |
|---|---|---|
| Route tools | each app's core (6 or fewer, fixed order) plus the extras the request names | with 24 tools it wandered; prompt 3.4K -> 1.2K tokens |
| Render | the model's own template; tool JSON compact when the indented list passes 1/4 of the context | indent-4 JSON did not fit 8K |
| Warm-up | an app's core prompt is read when the app opens | first step 91 s -> 5 s |
| Mend the call | missing closing braces; arguments nested under "properties" | both seen on the 3B |
| Scope guard | writes only to the open file, files the request names, this job's files, and one new file | it wrote into the person's own document |
| Done | an out-of-scope write, a repeated call, or content already added ends the job with a plain summary | it re-added a list and kept going |
| Canaries | a fix passes only at all-but-one of 8; replies are never served from cache on a re-check | babble reads as "wrong answer" |

## 8. The app (UI standards)

- **Look:**
  - Sheep's telemetry style (`sml/ui/index.html`): dark, dense, instrument
    panel, custom components;
  - no stock kits, no Microsoft/Google/Adobe names, layouts or fonts;
  - open-licence fonts only.
- **Motion:**
  - every screen change transitions smoothly;
  - KPIs grow in (rings fill, bars rise) on open and refresh;
  - nothing jumps.
- **KPIs:** speed, quality, GPU/CPU. Red, amber and green first, a number
  second, detail only when a section is expanded.
- **Every view** is driven by the backend API: no state only the page knows.
- **Done means screenshotted:** each UI step is checked with a screenshot of
  the Electron app before it is called done.

## 9. Proof (claims we make)

- **What a claim needs:** a same-session race against llama.cpp's own tools
  (converter, Q4_K_M, bench, perplexity), with equal threads, ABBA rounds and
  a CI.
- **Verdicts:** WON only when the whole CI clears 1 (speed) or 0 (KL).
- **Reporting:**
  - ties and losses are reported as measured;
  - heat and clock are printed beside every GPU number;
  - claims to clients come from real checkpoints; random-weight family heads
    prove architectures only.
- **Frozen during a run:** binaries are frozen (hashed) and the machine does
  nothing else.

## 10. Engineering rules

- **One source of truth per fact.**
  - Structure is read from the tensors, behaviour from the model's config,
    templates and stops from the model's own files.
  - Nothing is declared twice.
- **Mechanisms in the engine, decisions in the orchestrator.**
- **Refuse, don't guess.** An unsupported feature is refused by name. Every
  past architecture bug produced fluent nonsense rather than an error.
- **Commits:** one step per commit, with the measured numbers in the message.
  A gate re-runs after every change it covers.
- **Knobs:** environment knobs (`SML_*`) are for development and tuning only;
  anything a user would need becomes an orchestrator decision.
- **Never rebuild a binary a running race uses:** use `make gpu-dev`.
- **The product carries no llama.cpp code or names.** LICENSE and NOTICE keep
  the lineage.
