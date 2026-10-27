# B10 proof pass — 2026-10-22

Command: `python tooling/proof/proof_pass.py scratchpad/proof-final-20261022`

The run used the isolated Qwen2.5 3B profile with `device: cpu`; the proof script configures this deliberately. AC power was connected. The RTX 3050 remained around 90–91°C.

## Results

- **Replay:** all 7 turns completed. The run created and exported the Legend of the Seeker deck, added follow-up slides, searched the web, made a second deck, listed the files, and exported the second deck.
- **Video prompts:** both fresh-home runs completed, in 71 seconds and 65 seconds. Each sent 3 calls to Nebius. In both runs, the first `web_fetch` attempt failed and the retry succeeded; the decks and follow-up slides were created.
- **App checks:** 14/14 passed, each on its first try: `sl000`, `sl008`, `wd000`, `wd009`, `sh001`, `sh006`, `fl005`, `ch002`, `cd000`, `fu004`, `fu008`, `pi000`, `vi000`, `sp000`.
- The runner ended with `B10 DONE`.

The isolated homes, workspaces, server logs, and Nebius call logs are under `scratchpad/proof-final-20261022/`.
