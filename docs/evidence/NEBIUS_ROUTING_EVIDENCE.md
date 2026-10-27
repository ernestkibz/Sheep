# Nebius routing evidence

Sheep's cloud model is NVIDIA Nemotron on Nebius. Nebius Token Factory is not
open to accounts in Uganda (2026-09-30), so Sheep reaches Nebius through
OpenRouter. On 2026-10-13 Nebius Support confirmed in writing that this works
when every request is pinned to the Nebius provider with fallbacks off:

> OpenRouter can be used to call Nebius Token Factory models when the request is
> explicitly pinned to the Nebius provider and fallbacks are disabled. Please make
> sure your OpenRouter configuration uses provider.only = nebius and
> allow_fallbacks = false, and keep that configuration as evidence for your
> submission.

## Configuration

| Setting | Value |
|---|---|
| Endpoint (`NEBIUS_BASE_URL`) | `https://openrouter.ai/api/v1` |
| Model (`NEBIUS_MODEL`) | `nvidia/nemotron-3-nano-30b-a3b` |
| Provider block sent with every request | `{"only": ["nebius"], "allow_fallbacks": false}` |

The provider block is added in code, not left to configuration. Both cloud call
sites go through `pin_route()`:

- `tooling/sml_bridge.py`: `OPENROUTER_PIN`, `pin_route()`, and `Bridge.ask_cloud` (chat, streamed)
- `mlmi/backend/modes.py`: `Cloud.chat` (job planning and the cloud stand-in)

The model is Nemotron 3 Nano 30B because, on OpenRouter, it is the NVIDIA model
that Nebius serves. We checked this with OpenRouter's public endpoints API
(`/api/v1/models/<model>/endpoints`) on 2026-10-13: Nemotron 3 Super 120B is
served there only by DeepInfra and DekaLLM, and its free version only by NVIDIA.

## Log of every call

Each cloud call adds one line to `%LOCALAPPDATA%\Sheep\routing_evidence.jsonl`.
A line holds the endpoint, the model asked for, the provider block sent,
OpenRouter's generation ID, and the provider OpenRouter reports as having
answered. It does not hold the messages.

## Test calls, 2026-10-13

| Time (EAT) | Model asked for | Answered by | OpenRouter generation ID |
|---|---|---|---|
| 18:19:28 | nemotron-3-super-120b-a12b:free | refused (404) | none |
| 18:19:31 | nemotron-3-nano-30b-a3b | Nebius | gen-1791904768-hrejDtA1S4XZu5ARKkNH |
| 18:19:38 | nemotron-3-nano-30b-a3b (streamed) | Nebius | gen-1791904777-d6LN2igNWqKKE1R78pZ6 |
| 18:21:38 | nemotron-3-nano-30b-a3b | Nebius | gen-1791904897-IP4ReFRGOYYoPwXNyME9 |
| 18:21:41 | nemotron-3-nano-30b-a3b | Nebius | gen-1791904898-9qgZm8XcXcCjllpbXxzu |
| 18:22:44 | nemotron-3-nano-30b-a3b | Nebius | gen-1791904914-OMllK2TrhI7rInX6impt |
| 18:22:57 | nemotron-3-nano-30b-a3b | Nebius | gen-1791904964-JR4AC5K36uP6Jm0fpGKd |
| 18:23:24 | nemotron-3-nano-30b-a3b | Nebius | gen-1791904992-tdSaEltbw4ylroLxA1xk |
| 18:23:27 | nemotron-3-nano-30b-a3b | Nebius | gen-1791905004-KpPoxCiYdbpZ60Rtez9R |
| 18:23:30 | nemotron-3-nano-30b-a3b | Nebius | gen-1791905007-tEJl2S0q4JOKbAlkPYci |

The first row shows the pin working. Asked for a model Nebius does not serve,
OpenRouter refused the request ("your request's provider.only preference permits
only: nebius") instead of sending it to another provider.

## B10 proof pass, 2026-10-21 (EAT)

The video's prompts, run twice on a fresh home. Every call was pinned to Nebius, and Nebius answered each one.

| Time | Model asked | Answered by | OpenRouter generation ID |
|---|---|---|---|
| 18:03:13 | nemotron-3-nano-30b-a3b | Nebius | gen-1792594990-JtcPJmPVdhmV0UzaCQkI |
| 18:03:16 | nemotron-3-nano-30b-a3b | Nebius | gen-1792594997-WBisYG9vHKa1ARDRpRyb |
| 18:03:29 | nemotron-3-nano-30b-a3b | Nebius | gen-1792595003-AjxD5imITuTnfiw1nLGO |
| 18:04:03 | nemotron-3-nano-30b-a3b | Nebius | gen-1792595041-S4U935zt44XhklAuiVyf |
| 18:04:06 | nemotron-3-nano-30b-a3b | Nebius | gen-1792595048-rrJ8cAV1Bx6sxbCgtVyV |
| 18:04:15 | nemotron-3-nano-30b-a3b | Nebius | gen-1792595051-ElrSoCzASI6xRoqXSXWQ |
| 18:31:51 | nemotron-3-nano-30b-a3b | Nebius | gen-1792596676-OiIGBy7lQvyjY3N6OT06 |
| 18:31:56 | nemotron-3-nano-30b-a3b | Nebius | gen-1792596715-GzpKp9XFSCU9VTeETswy |
| 18:33:02 | nemotron-3-nano-30b-a3b | Nebius | gen-1792596726-zbYCjY6rWs8Evg4c843R |
| 18:33:36 | nemotron-3-nano-30b-a3b | Nebius | gen-1792596813-raMYot5ilVYNn4uhWpXo |
| 18:33:53 | nemotron-3-nano-30b-a3b | Nebius | gen-1792596835-DoqtbB5sFP4EZ76wanzs |
| 18:34:38 | nemotron-3-nano-30b-a3b | Nebius | gen-1792596838-fotauWUWMCFnAvluX1ry |
