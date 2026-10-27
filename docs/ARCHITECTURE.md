# Architecture

Sheep is one desktop app over one local server. The server runs a local model in Sheep's own engine, calls
NVIDIA Nemotron on Nebius when the person allows it, and makes real files in Sheep's own apps.

```
 Electron window (mlmi/desktop)                     the person
   chat · Words · Sheets · Slides · Studio · Text · Work Canvas · Settings
        │  HTTP + server-sent events, localhost only
        ▼
 server.py ── modes.py ── who answers each step: the local model, or Nemotron on Nebius
    │
    ├── agent/ ── a request becomes tool calls ─┐
    │     shepherd.py: the model writes, Sheep operates
    ├── tools/ ── Words, Sheets, Slides, Studio, files, web, mail, media
    │                   (the editors call the very same tools: one operation, one undo)
    ├── workspace.py ── every file Sheep makes, and its undo history
    │
    └── orchestrator.py ── the model's admin
          plans the device · starts the engine · checks · watches heat · repairs
             │
             ├── sml-server (C, OpenMP, CUDA PTX) on an .ssf file        native
             ├── compat_server.py (transformers) on the original weights  compatibility mode
             └── dml_worker.py (DirectML) on AMD and Intel graphics
```

## One request, end to end

1. The person types "powerpoint about the history of coffee". The window posts it to `/api/agent`.
2. `agent/routing.py` reads the words, the open app and the model's abilities, and decides which tools the
   turn may use. A deck request gets the Slides tools.
3. In **Sheep + Nebius** mode, Nemotron plans the job as exact tool calls (`agent/plan_calls.py`); in
   **Sheep** mode the local model writes the content in plain markdown and the **shepherd** turns it into
   tool calls (`shepherd.py`). Small models write well but fumble multi-step tool calls; the shepherd is why
   they still make real files.
4. Each tool call runs in `tools/` against the workspace. Every change is recorded, so Undo works across
   chat and editors.
5. `verify.py` opens what was made and checks it is what was asked (the slide count, the named rows). A short
   deck is topped up (`agent/slides_fill.py`).
6. With Nebius on, Nemotron reads the local model's draft against the request before the file is made
   ("Checked by Nemotron" in the chat).
7. The reply streams back with the file as a card: Open, the Office copy (.pptx, .docx, .xlsx) and Canvas.

## One model, from a name to "Ready"

`hf_source.resolve` finds the original weights → `quantizer.py` downloads, converts (GGUF, GPTQ, AWQ, .pth)
or fits an unfamiliar architecture (`tooling/sml_autofit.py`), then builds the `.ssf` at the level
`levels.py` chose → `orchestrator.py` places it (graphics card, processor or both), starts the engine and
runs the checks → `story.py` tells the person what happened. [MODELS.md](MODELS.md) has every step.

The orchestrator's checks, in order:

| Check (as the window names it) | What it does |
|---|---|
| Model ready | Loads the model and asks a short first exam |
| Answers checked | Check questions with known answers; each answer read for structure, never for opinion |
| Apps checked | Real jobs in Sheep's apps, each checked by opening the file the model made |

Checks give way to the person's own requests, pause when the laptop is hot or on battery (`thermal.py`),
and are reused while the model, engine, settings and device stay the same. A model that misbehaves later is
repaired (`orchestrator.py`, "repair"), and the repair is shown in Settings › Activity.

## Code map

| Path | What it holds |
|---|---|
| `mlmi/desktop/main.js`, `preload.js` | The Electron window, the tray, the file pickers |
| `mlmi/desktop/ui/app.js` | The chat, the readiness card, the add-model dialog |
| `mlmi/desktop/ui/views.js` | Settings: General, Activity, Orchestrator, This computer, Models, Custom tools, Saved files, Sent out |
| `mlmi/desktop/ui/{words,sheets,slides,studio,text,work}.js` | The apps |
| `mlmi/backend/server.py` | The local server and its routes |
| `mlmi/backend/agent/` | The agent: prompts, routing, planner calls, scope, media, research, slide top-up |
| `mlmi/backend/shepherd.py` | Markdown from a model into real files |
| `mlmi/backend/tools/` | Every operation the model and the editors share |
| `mlmi/backend/orchestrator.py` | Planning, the engine, checks, heat, repair, the Activity tiles |
| `mlmi/backend/story.py` | "What Sheep did for you" |
| `mlmi/backend/hf_source.py`, `quantizer.py`, `levels.py`, `model_slots.py` | Any model in: find, convert, fit, build |
| `mlmi/backend/compat_server.py` | Compatibility mode |
| `mlmi/backend/modes.py` | Local or Nebius for each step; the cloud call log |
| `mlmi/backend/publish.py` | Publishing a built model to Hugging Face |
| `sml/` | The engine in C: kernels, loader, forward pass, tokenizer, `sml-server` |
| `tooling/` | The builder, the converters, the race and grading tools, the UI audit |

## Internal names

The app was called ML.M.I until 2026-10-21. The Python package is still `mlmi`, the home folder is
`~/.mlmi` and the environment variable is `MLMI_HOME`, so existing installs keep their models and settings.
Custom tools are "herders" in the code. The repair step was "Egypt" in older code and history entries. The
local file formats are `.ssf` (models), `.sdoc`, `.ssheet`, `.sdeck` and `.sstudio` (the apps).
