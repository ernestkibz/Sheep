# Release checklist

Each gate, with the evidence for it. A gate that is not met says so, and why.

| # | Gate | State | Evidence |
|---|---|---|---|
| 1 | Every screen photographed, dark and light, at 1920, 1280 and 900 wide, and reviewed; no jargon; every icon button labelled | Met | `tooling/ui_audit/photos.py` (returning and `new`); `mlmi/tests/test_ui_words.py` passes with no exceptions left |
| 2 | The journeys pass as a new person on fresh homes, with no dead ends | Met | J1–J7 results below. Fresh reruns: J2 7/7, J3 10/10, J4 3/3, J5 3/3 and J7 4/4. |
| 3 | No backend function over ~150 lines; one name, Sheep; no clutter; tests clean up; the suite green | Partly | One name: met (`test_docs.py`, `test_ui_words.py`). Full suite (2026-10-23): 623 passed, 11 skipped, 194 subtests passed. Local refactors extracted the model-slot media checks (`check()` 134 lines), reduced the trainer, training worker, shepherd job and orchestrator start path below 150 lines, and split the pure planner into helpers; focused planner tests passed (44 passed, 1 skipped). **Not met:** four functions remain over 150 lines: `agent.run` (1266), `server.make_handler` (826), its `do_POST` (485) and `do_GET` (285). `agent.run` shares mutable closure state; splitting the server routes is the larger deferred refactor. |
| 4 | Docs organised and current, with a development history covering every stage; a test fails on stale names and broken links | Met | `docs/`, `CHANGELOG.md`, `mlmi/tests/test_docs.py` |
| 5 | The proof pass green on the final code; the installer on a stripped profile | Partly | `tooling/proof/proof_pass.py` passed: 7-turn replay, video prompts twice, and 14/14 app checks on CPU. Evidence: `docs/evidence/PROOF_PASS_2026-10-22.md` and `scratchpad/proof-final-20261022/`. Rebuilt the SML engine from current sources with w64devkit GCC/OpenMP; `submission/BUILD_INSTALLER.ps1 -OutputDir build/Sheep-final-20261022` staged 360.7 MB; Inno Setup 6.7.3 compiled `build/Sheep-Setup-final-20261022.exe`, and its silent per-user install exited 0. A stripped-PATH launch downloaded Python 3.12.10 but did not reach `/health`: this Windows profile has a stale Python 3.12 registration pointing to a missing earlier scratchpad install, so the installer treated Python as already installed and applied no packages. Evidence: `docs/evidence/STRIPPED_PROFILE_2026-10-22.md`; raw logs in `build/Sheep-stripped-profile-20261022/`. Repeat on a genuinely fresh Windows profile with no Python registrations. |
| 6 | Hugging Face publishing live on Ernest's account | Waits on Ernest | Needs a Hugging Face token of type Write |
| 7 | Ernest's outward steps confirmed by him | Waits on Ernest | Repo public, judges' keys, release, video, Devpost |

## Post-push app relaunch

The 2026-10-22 workspace Electron relaunch was at `2ef97b0`; the follow-up evidence is pushed on `origin/master` through `175b60b`. The visible window title was Sheep; `/health` and `/` both returned 200 on port 8080. Runtime evidence: `docs/evidence/APP_RELAUNCH_2026-10-22.md` and `scratchpad/relaunch-20261022.json`.

The existing Qwen2.5 3B GPU selection loaded 3,916 MiB of 4,096 MiB and reached 95°C at 100% utilization. The saved thermal guard had been off, so its first status said `cool` at that temperature. The SML engine was stopped at 95°C; the backend restarted it. I enabled the thermal guard, which then reported `critical` at 91°C GPU and a 101°C ACPI CPU reading. The follow-up check showed startup passed, canaries 8/8 passed, and orientation failed at 13/14 after the engine interruption. No model answer was attempted after the guard was enabled.

Latest source relaunch (2026-10-23, `dcfb46f`): Electron started after clearing inherited `ELECTRON_RUN_AS_NODE=1`, and `/health` returned 200 on port 8080. Startup restored Qwen2.5 3B; the GPU reached 91°C at 100% utilization. I stopped the Sheep process tree to let it cool. The app is currently closed, and no final GPU proof was run on this commit.

## Journeys

| Journey | Result |
|---|---|
| J1 no model, a key: first answer, deck, `pptx` | 4/4 in the earlier run |
| J2 no key: suggested model builds, first local answer | 7/7 on 2026-10-22. Qwen3 0.6B built at Q6 High, answered “Nairobi” locally on the RTX 3050, and showed the setup timeline. The quick exam recorded 4/5 (follow-up failed); Sheep accepted it and the first answer worked. Evidence: `scratchpad/ui_audit/new-20261022-223530/`. |
| J3 any model: GGUF, GPTQ, EXL2 to the original; refusals in words; the original builds | 10/10 on 2026-10-22. Evidence: `scratchpad/ui_audit/new-20261022-214808/`. Two console HTTP 400 entries occurred on the intended refused-page checks. |
| J4 a research deck with Sources | 3/3 on 2026-10-22 using the Nebius route and CPU model placement after the 3B GPU attempt reached 99°C. The five-slide deck has a Sources slide linking to Wikipedia's Solar power in Kenya page; this proves the source-slide flow, not the prompt's “latest news” recency. Evidence: `scratchpad/ui_audit/returning-20261022-223326/`. |
| J5 a local model's work checked by Nemotron | 3/3 on 2026-10-22 using CPU model placement: Nemotron reviewed the local model's slides. Evidence: `scratchpad/ui_audit/returning-20261022-230638/`. |
| J6 no network, a wrong key, an engine crash, a full disk | All four cases passed: offline and engine crash in the earlier run; wrong key 2/2 and low disk 2/2 on 2026-10-22. Evidence: `scratchpad/ui_audit/j6-tail-20261022/`. |
| J7 battery waits, the charger resumes | 4/4 on 2026-10-22 using CPU placement. The battery tile said checks wait; after AC resumed, it said checks run slower. Evidence: `scratchpad/ui_audit/returning-20261022-230724/`. |

Latest hardware evidence (2026-10-22): Windows PnP reports the RTX 3050 as **OK**, and AC power is connected. J2 completed on the GPU. The separate Qwen2.5 3B GPU J4 attempt reached 99°C and was stopped before creating a deck; J4, J5, J7, and the final B10 proof pass completed with CPU placement. The GPU stayed below its slowdown threshold during those CPU runs.

