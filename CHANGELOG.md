# Changelog

Grouped by date; the commit messages hold the detail. [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)
tells the same story by stage, with the proof for each.

## 2026-10-30: the agent keeps the thread
- A turn belongs to the conversation: "just do it", "you decide", an answer to Sheep's question, "try again" or a
  short change to a picture all carry the request before them, for every tool and every model.
- The model chooses the tools unless the person is only talking: a greeting, thanks, a question or "no file".
- "Can you make ..." is a request, not a question, and gets its tools.
- One request can make several files, and a job can change a file it found and read.
- A model asked to make something and answering with a question is told once to choose for itself and do the job.
- A new file holding only its title is not done: the model is told to finish it, and `words_create` takes the
  whole document in one call.

## 2026-10-21: polish
- One name, Sheep, in the window, installer, launcher and the model's own instructions.
- Plain words on every screen, with a test that fails on engineer's words.
- Activity: the checks under the same names as the checks card, read from the same records; speed in words.
- Orchestrator: "What Sheep did for you", each model's setup step by step; the same steps live while adding a
  model.
- Models: your models as cards; one "Add any model" box; extras grouped.
- Chat: made files as cards with Open, .docx/.xlsx/.pptx and Canvas.
- Failures said in words with a way back: no network, a refused key, an engine crash, a full disk, a battery.
- EXL2 and other copies traced to their original by name; a Hub hiccup no longer sizes a model at 0 GB.
- Publishing asks for a Write token and names a read-only one as the cause.
- Docs reorganised: architecture, models, engine, building, testing, evidence, this changelog.

## 2026-10-19 → 10-21: any model, any format
- Any Hugging Face page builds from its original weights; GGUF, GPTQ, AWQ and Meta `.pth` converted when no
  original exists.
- Auto-fit for architectures nobody listed; compatibility mode with the maker's own code; compatibility models
  rebuilt natively later.
- Publishing a built model to Hugging Face, and adding a published Sheep model with no build.

## 2026-10-20 → 10-21: the hackathon entry
- Keys in Settings; a first run that answers at once with Nemotron on Nebius.
- Nemotron checks the local model's drafts; research decks with Tavily sources.
- An installer that works on a PC with no Python.

## 2026-09-30 → 10-14: the orchestrator and Sheep + Nebius
- Device planning across NVIDIA, AMD/Intel and the processor; levels sized for this computer.
- Startup exam, check questions and real app jobs; automatic repair; heat and battery pacing.
- Nemotron on Nebius plans multi-step jobs as exact tool calls; every call logged.

## 2026-09-29 → 10-14: the apps and the agent
- Words, Sheets, Slides, Studio, Text and the Work Canvas, sharing their tools with the model.
- The shepherd: the model writes, Sheep makes the files.

## 2026-08-29 → 09-28: the engine
- SSF format and SML engine, measured bit placement, GPU kernels.
- Races against llama.cpp b11177: smaller and more accurate files, faster decode on CPU and GPU.
- Family heads checked against transformers.
