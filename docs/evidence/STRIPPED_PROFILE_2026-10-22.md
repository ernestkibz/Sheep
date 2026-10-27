# Stripped profile launch check — 2026-10-22

The packaged `build/Sheep/Sheep.cmd` was launched with a Windows-only `PATH`, a fresh `LOCALAPPDATA`, `%TEMP%` and `MLMI_HOME`, no Hugging Face key, and port 8141. This excluded the installed Python directories and Windows Store aliases from command lookup.

The launcher downloaded Python 3.12.10, then failed to find its private interpreter and exited with code 1. The Python setup log shows that this Windows account already has a per-user Python 3.12 registration from an earlier temporary Sheep run, but that registration points to a missing `python.exe`. Setup detected its Python packages as present and applied no package changes, so the new isolated `LOCALAPPDATA` received neither `Sheep/python/python.exe` nor `Sheep/venv/Scripts/python.exe`. Sheep therefore did not start and neither `/health` nor the UI route was reached.

The setup log shows that it applied no packages. The existing registered Python 3.14 executable remained available afterward and reported version 3.14.3. No Electron process or backend was started by this failed run.

Raw evidence is under the ignored build directory: `build/Sheep-stripped-profile-20261022/launcher.stdout.log`, `launcher.stderr.log`, and `temp/Python 3.12.10 (64-bit)_20261022231352.log`. The installer artifact is `temp/sheep-python.exe`.

## Installer artifact

`build/Sheep-Setup-final-20261022.exe` is 105,063,665 bytes. SHA-256: `5D99DBE3AE4AE77971BAE32CF7EBC73E244ACA45E02E647EB63DFCC91CE1BC8F`. Windows reports `NotSigned` for Authenticode. The installer compiled and its earlier silent per-user installation exited 0.

The release gate remains partly met. A fresh Windows profile with no Python registrations is still required to verify the installer's no-Python bootstrap through to app health.
