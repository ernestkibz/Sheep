# The manual's screens, photographed by Sheep itself (MLMI_SHOTS, mlmi/desktop/main.js):
# each view opens, settles, and is saved to docs/shots/<name>.png; then Sheep quits.
#   powershell -File tooling/manual_shots.ps1        (close Sheep first)
$root = Split-Path -Parent $PSScriptRoot
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
Remove-Item Env:NODE_ENV -ErrorAction SilentlyContinue
$env:MLMI_SHOTS = Join-Path $root "docs\shots"
$env:MLMI_WAIT = "3500"
$env:MLMI_VIEWS = "home,guide:MLMI.openGuide(),basics:MLMI.closeGuide(),models,accounts,activity,studio:document.querySelector('#closeSettings')&&document.querySelector('#closeSettings').click()"
Push-Location (Join-Path $root "mlmi\desktop")
& ".\node_modules\electron\dist\electron.exe" . | Out-Null
Pop-Location
# a view with a script is saved as <name>-<index>.png; the manual wants <name>.png
Get-ChildItem $env:MLMI_SHOTS -Filter "*-*.png" | ForEach-Object {
    Move-Item $_.FullName (Join-Path $env:MLMI_SHOTS (($_.BaseName -replace '-\d+$', '') + ".png")) -Force
}
Get-ChildItem $env:MLMI_SHOTS
