# Testing Sheep

Five layers, from seconds to an hour. Run the first two before every commit and all five before a release.

## 1. Unit tests (seconds)

```
python -m pytest mlmi/tests -q
```

About 630 tests: the agent's routing and scope, the shepherd, every app's tools, formulas, exports, the
orchestrator's plans and checks, heat and battery pacing, Hugging Face resolving (with a fake Hub), the
model check, publishing, and the timeline. They need no model and no network.

## 2. The words and docs tests (part of the unit tests)

- `mlmi/tests/test_ui_words.py` collects the text a person sees in the window and fails on engineer's words
  (slots, canaries, quantizing, layers, CUDA, file extensions...). Technical lines live inside an element with
  the class `tech` (the Details parts), which the test skips.
- `mlmi/tests/test_docs.py` checks that every link in the docs resolves and that old names (ML.M.I, Dashboard,
  Herders, This machine) are gone.

## 3. Photos of every screen (a few minutes)

```
python tooling/ui_audit/photos.py          # a returning person, with a model
python tooling/ui_audit/photos.py new      # first launch
```

Each run gets its own home (`MLMI_HOME`) and window data, so it opens beside a Sheep that is already running
and never touches the real `~/.mlmi`. Every Settings page is photographed, dark and light, at two widths,
top to bottom. Read the photos; they catch what tests cannot (a cut-off button, a sentence said twice).

## 4. The journeys (an hour; needs the network)

```
python tooling/ui_audit/journeys.py            # J1 to J7
python tooling/ui_audit/journeys.py j3 j6      # some
```

| Journey | What a new person does |
|---|---|
| J1 | No model, a Nebius key: the first answer, "powerpoint about …", then "pptx" |
| J2 | No key: the suggested model downloads and builds; the first local answer; the timeline |
| J3 | GGUF, GPTQ and EXL2 pages lead to their original; wrong pages refused in words; the original builds |
| J4 | A research deck that ends with a Sources slide |
| J5 | A local model's work checked by Nemotron |
| J6 | No network, a wrong key, an engine crash, a nearly full disk: each in words, each with a way back |
| J7 | On battery the checks wait and say why; on the charger they go on |

J6 and J7 use test seams: `HTTPS_PROXY` at a closed port for no network, `MLMI_FAKE_FREE_GB` for the disk and
`MLMI_FAKE_POWER` (`battery:60`, `ac`, or `@file`) for the battery.

## 5. The proof pass (before a release; the charger in)

```
python tooling/proof/proof_pass.py
```

A fresh home on the 3B: the app checks row by row, a seven-turn replay of a real person's chat, and the
video's prompts twice, each timed. On battery the background checks wait by design, so plug in first.

## The engine

The engine has its own gates (correctness before speed) and race tools; see [ENGINE.md](ENGINE.md) and
[METHOD.md](../METHOD.md).
