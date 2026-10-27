# Sheep demo: the engine, the bridge in front of it, and the interface in a browser.
#
#   .\demo.ps1                                   # Llama-3.2-1B, cloud if NEBIUS_API_KEY is set
#   .\demo.ps1 -Model C:\models\other.ssf
#   .\demo.ps1 -Check                            # start, prove both answer, stop
#
# The cloud key is read from the environment only:
#   $env:NEBIUS_API_KEY = "..."    (never put it in this file)

param(
    [string]$Model = "C:\Users\earne\models\llama32-1b-meas.ssf",
    # a Sheep vision model: photos are answered on this machine when it exists
    [string]$Vision = "C:\Users\earne\models\smolvlm-256m-q6.ssf",
    [int]$Port = 8080,
    [int]$EnginePort = 8099,
    [int]$Threads = 6,
    [switch]$Check
)

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$server = Join-Path $root "sml\sml-server.exe"
$bridge = Join-Path $root "tooling\sml_bridge.py"

if (-not (Test-Path $server)) { Write-Host "sml-server.exe not built. Run: cd sml; make"; exit 1 }
if (-not (Test-Path $Model))  { Write-Host "model not found: $Model"; exit 1 }

if ($env:NEBIUS_API_KEY) {
    Write-Host "cloud ON  -- facts, code and long reasoning go to Nemotron on Nebius"
} else {
    Write-Host "cloud OFF -- NEBIUS_API_KEY is not set, so every turn stays on this machine"
}

# Start-Process joins arguments with spaces and does not quote them, and this
# repository lives under a path with a space in it -- so every path is quoted here.
$engine = Start-Process -FilePath $server -PassThru -WindowStyle Hidden -ArgumentList `
    "`"$Model`"", "--port", "$EnginePort", "--threads", "$Threads", "--ctx", "2048"
$bargs = @("`"$bridge`"", "--local", "http://127.0.0.1:$EnginePort", "--port", "$Port")
if ($Vision -and (Test-Path $Vision)) {
    $bargs += @("--vision", "`"$Vision`"")
    Write-Host "vision LOCAL -- photos are answered on this machine ($(Split-Path -Leaf $Vision))"
} else {
    Write-Host "vision: no local model -- photos go to NEBIUS_VISION_MODEL if it is set"
}
$front = Start-Process -FilePath "python" -PassThru -WindowStyle Hidden -ArgumentList $bargs

function Wait-Up([string]$url, [int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-RestMethod -Uri $url -TimeoutSec 2 | Out-Null; return $true } catch { Start-Sleep -Milliseconds 400 }
    }
    return $false
}

$ok = (Wait-Up "http://127.0.0.1:$EnginePort/health" 60) -and (Wait-Up "http://127.0.0.1:$Port/health" 20)
if (-not $ok) {
    Write-Host "did not come up -- stopping"
    Stop-Process -Id $engine.Id, $front.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$stats = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/stats"
Write-Host ("Sheep is up: {0}, {1} threads, cloud configured: {2}" -f $stats.model, $stats.threads, $stats.cloud_configured)

if ($Check) {
    Stop-Process -Id $engine.Id, $front.Id -Force -ErrorAction SilentlyContinue
    Write-Host "check passed; stopped"
    exit 0
}

Start-Process "http://127.0.0.1:$Port"
Write-Host "open http://127.0.0.1:$Port   -- press Enter here to stop Sheep"
[void][Console]::ReadLine()
Stop-Process -Id $engine.Id, $front.Id -Force -ErrorAction SilentlyContinue
Write-Host "stopped"
