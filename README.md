# Sheep

**A personal AI that makes real files on your own computer.** Ask in plain words for a slide deck, a
document, a spreadsheet or a video: Sheep makes it in its own apps and exports it to PowerPoint, Word, Excel
or MP4. The AI model runs on your laptop in Sheep's own engine, and **NVIDIA Nemotron on Nebius** plans the
big jobs, checks the local model's drafts, and answers while the local model loads. **Tavily** brings in
current facts, with their sources.

Sheep is for people with an ordinary laptop, a slow or costly connection, or work that should stay on their
own machine. It was built in Uganda, where Nebius Token Factory itself is not yet open to accounts.

![Sheep: a deck made from one line, with the Work Canvas beside the chat](mlmi/media/gallery/1-chat.png)

## For judges: try it in two minutes

1. On Windows 10 or 11 (64-bit), download `Sheep-Setup.exe` from this repository's **Releases** page and
   install it. No NVIDIA GPU is needed; Sheep uses the processor when there is none.
2. Open **Settings › Accounts**, paste the judges' keys from the Devpost
   testing instructions, and press **Save**.
3. Close Settings and click **Start now with Nemotron on Nebius** on the welcome screen. Ask
   `powerpoint about the history of coffee`; when the deck is made, type `pptx`.
4. Ask `make a presentation about the latest NVIDIA Nemotron models`: Sheep searches with Tavily first, and
   the deck ends with a Sources slide.
5. Add a local model: **Settings › Models › Add any model**, for example `Qwen/Qwen2.5-1.5B-Instruct`, then
   **Check** and **Build**. Watch **Settings › Activity › What Sheep did for you**: Sheep finds the
   original weights, builds them for your computer, places the model, and checks it on real app jobs.

## Any model, with nothing to set

Give Sheep a model the way you would name it: a Hugging Face ID, a folder, or a `.gguf` file. A GGUF, GPTQ,
AWQ, EXL2 or MLX page is traced back to the original it was made from, and built from that. A fine-tune that
exists only as a GGUF or GPTQ copy is converted first. An architecture Sheep's engine has not met is fitted to
it, or runs in compatibility mode with the maker's own code. The level is chosen for your computer's memory,
the model is placed on the graphics card, the processor or both, and it is checked before you rely on it. You
set nothing. [docs/MODELS.md](docs/MODELS.md) has every step and every format.

![What Sheep did for you: a model's setup, step by step](mlmi/media/gallery/2-what-sheep-did.png)

## How Sheep uses Nebius, NVIDIA Nemotron and Tavily

| What | Where it runs | What it does in Sheep |
|---|---|---|
| **NVIDIA Nemotron 3 Nano 30B-A3B** | Nebius. Requests go through OpenRouter pinned to the Nebius provider with fallbacks off, because Token Factory does not yet accept accounts from Uganda; Nebius Support confirmed this route in writing ([evidence](docs/evidence/NEBIUS_ROUTING_EVIDENCE.md)) | Plans a multi-step job as exact tool calls that Sheep runs on the computer; checks the local model's draft against the request before Sheep makes the file; answers while the local model loads or is being repaired |
| **A local model** (Qwen, Llama, Gemma, Phi, Mistral, NVIDIA Nemotron-H and [many more](docs/MODELS.md#architecture-support)) | This computer, in Sheep's own engine (SML + SSF) | Writes the content; Sheep makes the files with its apps' own tools |
| **Tavily** | Web search API | Current facts for questions, and for decks and documents, which end with their sources |

Every request that leaves the computer is listed in **Settings › Advanced › Sent out**, with the provider that answered
it and that provider's id for the call. Keys go in Settings › Accounts, or in the environment variables
`NEBIUS_API_KEY`, `NEBIUS_BASE_URL` (the default is Token Factory, `https://api.tokenfactory.nebius.com/v1`),
`NEBIUS_MODEL` and `TAVILY_API_KEY`. In **Sheep** mode nothing leaves the computer; without a Tavily key, web
search uses a keyless public search.

## What is inside

| App | Format | Exports |
|---|---|---|
| **Words** | .sdoc | .docx, .md, .pdf |
| **Sheets** | .ssheet | .xlsx, .csv, .pdf |
| **Slides** | .sdeck | .pptx, .pdf |
| **Studio** | .sstudio | .mp4 |
| **Text** | .txt, .md, .py, .js, … | |
| **Work Canvas** | blocks: pin, drag, run code | |

Every editor runs the same tools the model calls: a click and a model's tool call are the same operation, land
in the same file, and share one undo. Also: attachments (pictures, documents, sheets, slides, PDFs), web search
with sources, voice and captions in Studio, pictures read on this computer, email over IMAP/SMTP and calendar
as .ics, and custom tools the model writes for itself.

**The orchestrator** is the model's admin. It finds your hardware, plans where the model runs, checks it (a
first exam, check questions with known answers, then real jobs in the apps, each checked by opening what was
made), paces itself when the laptop is hot or on battery, and repairs a model that starts to misbehave. Every
check and repair is shown in **Settings › Activity**.

## The engine

Sheep's engine (**SML**) and file format (**SSF**) were built during the hackathon. On the three models raced
against llama.cpp's Q4_K_M (Qwen2.5-3B, Llama-3.2-1B, pythia-410m) the SSF files are smaller and 12-41% more
accurate, and decode is faster on both CPU and GPU against llama.cpp b11177, with 95% intervals above 1. The
claims, their limits and every number are in [docs/ENGINE.md](docs/ENGINE.md) and [RESULTS.md](RESULTS.md).

## Built during the hackathon

This repository began on 2026-08-29 as a fork of an MIT-licensed C/C++ inference engine. The submission
period opened on 2026-08-26. Every file that came from upstream has since been deleted and replaced
([NOTICE](NOTICE), [PROVENANCE.md](PROVENANCE.md)), and the dated commits record what was built since then.
[docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md) walks through every stage.

## Honest limits

- The desktop app and installer are Windows only. The engine builds on Linux.
- Nebius is reached through OpenRouter, pinned to Nebius, until Token Factory opens to accounts in Uganda.
- The engine's speed wins are measured on one laptop with AVX2; on other CPUs, ARM or Mac, assume llama.cpp is
  faster.
- Compatibility mode (the maker's own code) is correct but slower than Sheep's engine.
- Small local models make mistakes; in Sheep + Nebius mode Nemotron checks their work, and every check is shown.
- The local vision model is tiny (256M): it names colours and simple shapes.
- Qwen2.5-3B, used in the tests, has a research licence; pick an Apache-2.0 model for real work.

## Docs

| Page | For |
|---|---|
| [User guide](docs/USER_GUIDE.md) | Using Sheep |
| [Models](docs/MODELS.md) | Which models work, formats in and refused, publishing |
| [Architecture](docs/ARCHITECTURE.md) | The parts, one request's path, the code map |
| [Engine](docs/ENGINE.md) | SML and SSF, the races, what can and cannot be claimed |
| [Building](docs/BUILDING.md) | The engine, the app from source, the installer |
| [Testing](docs/TESTING.md) | Unit tests, photos, journeys, the proof pass |
| [Development history](docs/DEVELOPMENT_HISTORY.md) | Every stage, with its proof |
| [Evidence](docs/evidence/README.md) | Routing evidence, results, parity, sprint records |
| [Changelog](CHANGELOG.md) | What changed, by date |

## From source

```
python -m pip install .
cd mlmi/desktop && npm install && cd ../..
Sheep.cmd
```

Needs Windows 10 or later, Python 3.9+ and Node.js. An NVIDIA GPU with 4 GB or more helps; the
processor alone works. [docs/BUILDING.md](docs/BUILDING.md) builds the engine and the installer.

## Licence

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE): this began as a fork and the attribution stands.
