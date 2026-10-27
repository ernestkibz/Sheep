# Development history

Every stage of Sheep, in order: what was built, why, and how it was proven. Dates are commit dates.
The commit messages and the dated comments in the code hold the detail. This page is the map.

## 1. The fork baseline (2026-08-29)
- **Built:** the repository started as a fork of an MIT-licensed C/C++ inference engine
  (commit `2241a21`). Its docs, CI and platform scripts were removed the same day.
- **Why:** to start from a working engine and replace it piece by piece with our own format and kernels.
- **Proof:** NOTICE and PROVENANCE.md record which upstream files were deleted and when. None remain in the
  tree.

## 2. Our own file format and engine: SSF and SML (2026-08-29 → 09-10)
- **Built:**
  - A layer-sequential SSF build that reads Hugging Face checkpoints directly, with no GGUF or safetensors
    library.
  - `sml-server`, an OpenAI-style HTTP server.
  - An asymmetric quantization grid, fused Q/K/V and a flat 4-bit kernel.
  - Measured bit placement, and calibration that puts SML ahead on quality.
  - GPU decode via PTX and dp4a.
- **Why:** to be smaller, faster and more accurate than the reference on the same models. Each idea was
  measured, and the losing ones were recorded as losses: entropy coding, large pages, certified block
  skipping.
- **Proof:** "FIRST WIN: 1.10x at usable quality" (`fa38860`). The 3B head to head (`f82ba5f`).
  "Smaller than the reference Q4_K_M on all three models, and still better" (`f6365e2`).

## 3. Races against llama.cpp (2026-09-18 → 09-28)
- **Built:**
  - CPU and GPU race tools with equal threads, ABBA rounds and bootstrap 95% intervals.
  - GPU prompt kernels with an integer epilogue.
  - A tuned GPU decode path.
- **Why:** claims had to hold against the reference's newest build, not an old one.
- **Proof:** CPU decode wins on all three against b11177 (1B 1.025x, 3B 1.044x, pythia 1.164x, CIs above 1).
  GPU decode wins on all three. CPU prompt reading wins 1.36–1.70x (RESULTS.md).

## 4. Breadth: model families (2026-09-23 → 10-19)
- **Built:**
  - Family "heads" checked against transformers.
  - Qwen3; MoE (Granite, Qwen 3.5-MoE, LFM2-MoE and more); Mamba; LFM2 hybrids; local vision.
  - Multi-head latent attention (DeepSeek, Kimi K2), Nemotron-H and Nemotron 3 Nano, PhiMoE, Unigram
    tokenizers, Qwen v1 tiktoken.
- **Why:** "any model" means the families people actually download.
- **Proof:** the roster grew from 38 to 62 architectures (`ffdb311`, `181b23a`), then to 178 of 182 family
  heads passing.

## 5. The apps and the agent (2026-09-29 → 10-14)
- **Built:**
  - `sml-server` speaks tools; the agent.
  - Words, Sheets, Slides, Studio (video), Text & Code and the Work Canvas.
  - Attachments, web search with sources, speech, pictures.
  - "The model writes; Sheep operates": the shepherd turns a small model's markdown into real files.
  - The first end-to-end .docx.
- **Why:** small models write well but fumble multi-step tool calls.
- **Proof:** PARITY.md, a ✅/❌ table per app, each ✅ backed by a test.

## 6. The orchestrator (2026-09-30 → 10-14)
- **Built:**
  - Device planning (GPU, CPU+GPU, CPU, several GPUs, AMD and Intel via Vulkan/DirectML).
  - Quantization levels sized for this machine.
  - Startup checks and canaries.
  - Orientation: real jobs in the apps.
  - Automatic repair of a model that misbehaves.
  - A heat governor and a power guard.
- **Why:** the user should never configure anything. The laptop shut down from heat once (2026-10-12).
- **Proof:** commits `7b9877b`, `ff20e10`, `ad763ff`, `7af01cb`, `c7b1ca3`, `bf24f77`.

## 7. Sheep + Nebius (2026-09-30 →)
- **Built:**
  - NVIDIA Nemotron on Nebius plans multi-step jobs as exact tool calls, which Sheep runs locally.
  - Requests go through OpenRouter pinned to the Nebius provider, because Token Factory does not yet accept
    accounts from Uganda.
- **Why:** strong planning when it helps, with nothing on the cloud that the laptop can do.
- **Proof:** docs/evidence/NEBIUS_ROUTING_EVIDENCE.md: every call logged with the provider that answered.

## 8. Any model, set up by itself (2026-10-19 → 10-20)
- **Built:**
  - Any Hugging Face page builds from its original weights (GGUF, GPTQ, AWQ, MLX and bitsandbytes pages
    resolved).
  - Auto-fit for architectures nobody listed.
  - Compatibility mode (the maker's own code) when the engine cannot run the maths.
  - A speed trial.
- **Why:** Ernest's core intent: any model works with no configuration.
- **Proof:** Youtu-LLM-2B added end to end through the app (`2a1d6c9`, `f9f2501`, `3c7dc61`, `fdb5908`).

## 9. Honest checks and readable Settings (2026-10-20)
- **Built:**
  - The app checks fail only for real.
  - Files first ("powerpoint about X" makes a deck).
  - A closable checks card and the Activity page.
  - Custom tools.
  - Plain words throughout.
- **Why:** the checks failed "as a habit", and Settings spoke engineer.
- **Proof:** 14/14 app checks on the 3B. The 7-turn chat replay.

## 10. The hackathon entry (2026-10-20 → 10-21)
- **Built:**
  - Keys in Settings, and a first run that answers at once with Nemotron.
  - Nemotron checking the local model's drafts.
  - Research decks with Tavily sources.
  - An installer that works on a PC with no Python.
  - The README, Devpost text, video script and feedback.
- **Why:** the Nebius × NVIDIA hackathon rules and the judges' criteria.
- **Proof:** the clean-profile install test. The B10 proof pass.

## 11. Any format in, Sheep files out (2026-10-21)
- **Built:**
  - GGUF, GPTQ, AWQ and Meta `.pth` converted when no original exists. EXL2, ONNX and Core ML pages are traced
    to their original by card or by name, and refused in words when there is none.
  - One "Add any model" box.
  - Publishing to Hugging Face.
  - Compatibility models rebuilt natively once the engine can run them.
- **Proof:** a downloaded Qwen2.5-0.5B GGUF converted, built (408 MB), green, chatting. A GPTQ folder built to
  331 MB.

## 12. Big hardware (2026-10-21 →)
- **Built:** the biggest NVIDIA card is picked and pinned, and big cards get a longer context.
- **Still to do:** the run on a Nebius AI Cloud GPU.

## 13. Polish (2026-10-21 →)
- **Built:**
  - One name, Sheep, everywhere a person looks (`ML.M.I` stays only in internal names).
  - The agent split into a package by job; the UI photo tools, journeys and proof pass in `tooling/`.
  - Plain words on every screen, guarded by a test; checks that count as they go.
  - Activity tiles from the same records as the checks card; "What Sheep did for you", a model's setup step by
    step, also live in the add-model dialog.
  - Models as cards with one "Add any model" box; file cards in chat with Open and the Office copy.
  - Failures in words with a way back: no network, a refused key, an engine crash, a full disk, a battery.
  - The docs reorganised: architecture, models, engine, building, testing, evidence.
- **Why:** a judge or a new person should understand every screen without asking.
- **Proof:** `tooling/ui_audit/photos.py` and `journeys.py` (J1-J7), and the words and docs tests.
  The old status, audit and session notes were folded in here and removed; git history keeps them.
