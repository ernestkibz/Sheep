#!/usr/bin/env pwsh
param(
    [string]$InnoSetupPath
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($InnoSetupPath)) {
    $InnoSetupPath = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6"),
        (Join-Path $env:LOCALAPPDATA "Programs\SheepBuildTools\Inno")
    ) | Where-Object { Test-Path (Join-Path $_ "ISCC.exe") } | Select-Object -First 1
}
if (!$InnoSetupPath -or !(Test-Path (Join-Path $InnoSetupPath "ISCC.exe"))) {
    throw "Inno Setup 6 was not found. Install it or pass -InnoSetupPath."
}

Write-Host "Staging the canonical desktop layout..." -ForegroundColor Cyan
& (Join-Path $repo "installer\BUILD_INSTALLER.ps1")

$output = Join-Path $repo "build\Sheep"
$iss = Join-Path $repo "installer\sheep.iss"
Write-Host "Compiling the canonical Inno Setup script..." -ForegroundColor Cyan
& (Join-Path $InnoSetupPath "ISCC.exe") "/O$output" $iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed to compile the installer." }

$exe = Join-Path $output "Sheep-Setup.exe"
if (!(Test-Path $exe)) { throw "Inno Setup completed without producing $exe." }
$size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "Installer ready: $exe ($size MB)" -ForegroundColor Green
