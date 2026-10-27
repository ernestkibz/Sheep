# App relaunch - 2026-10-22

After pushing `master` through `2ef97b0`, I relaunched the workspace Electron app with `ELECTRON_RUN_AS_NODE`, `NODE_ENV`, and `MLMI_HOME` cleared. The window title is Sheep. The local backend health route and UI route both returned HTTP 200 on port 8080. The launcher recorded a starting GPU temperature of 81 C and a temperature of 80 C when the health/UI checks first passed; details are in `scratchpad/relaunch-20261022.json`.

The existing user model preference was GPU with Qwen2.5 3B. Automatic startup work later loaded 3,916 MiB of the GPU's 4,096 MiB and reached 95 C at 100% utilization. The system ACPI CPU thermal-zone reading was 101 C. The saved thermal guard setting was `false`, so Sheep reported the heat level as `cool` despite the high readings.

I stopped the SML engine process at 95 C. The backend restarted it, while the app window and `/health` remained available. I enabled `thermal_guard` through Sheep's settings route; the setting is now saved as `true`, and the governor reported `critical` at a 91 C GPU reading and 101 C CPU thermal-zone reading. The selected model preference remains GPU. At the follow-up status check, startup was passed, canaries were 8/8 passed, and model orientation was failed at 13/14 after the engine interruption. No further model request was sent.

The app is relaunched and available for the user. The last observed GPU reading was 90 C at 27% utilization with 3,928 MiB allocated. The thermal guard is on and background/new GPU work should wait until the device cools. The orientation result is not a complete pass; repeat it only after the temperatures are safe.
