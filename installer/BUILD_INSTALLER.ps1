# Build a self-contained Sheep directory that runs with one double-click.
# Output: build\Sheep\ - copy this folder anywhere and run Sheep.cmd.
#
# Requirements on the build machine:
#   - Python 3.9+ with pip (the launcher creates a per-user venv on first run; on a PC with no
#     Python it fetches Python 3.12 from python.org into %LOCALAPPDATA%\Sheep\python, once)
#   - Node.js (Electron is expected in mlmi/desktop/node_modules)
#   - The SML engine built (sml/sml-server.exe, sml/sml-quant.dll, sml/*.cu, sml/*.ptx)
#   - Inno Setup 6 (https://jrsoftware.org/isinfo.php) for the .exe installer
#
# The shipped 3B model is NOT bundled - it downloads on first run from
# SHEEP_MODEL_URL (set by Ernest). The DirectML runtime downloads only when
# an AMD or Intel GPU is found.

param([string]$OutputDir)

$ErrorActionPreference = "Stop"
$REPO = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$engine = Join-Path $REPO "sml"
$desktop = Join-Path $REPO "mlmi\desktop"
$engineInputs = @("sml-model.c", "sml-model.h", "sml-run.c", "sml-run.h",
                  "sml-kernel.c", "sml-kernel.h", "sml-pool.c", "sml-pool.h",
                  "sml-vocab.c", "sml-vocab.h", "sml-server.c", "sml-cli.c")
$latestEngineInput = ($engineInputs | ForEach-Object {
    Get-Item (Join-Path $engine $_)
} | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc
foreach ($name in @("sml-server.exe", "sml-cli.exe")) {
    $binary = Join-Path $engine $name
    if (!(Test-Path $binary)) { throw "Required engine binary is missing: $binary" }
    if ((Get-Item $binary).LastWriteTimeUtc -lt $latestEngineInput) {
        throw "$binary is older than the current engine sources. Rebuild the SML engine before packaging."
    }
    # A build without OpenMP runs on one core (2.5-3x slower); never ship one.
    $text = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($binary))
    if ($text -notmatch "GOMP_") {
        throw "$binary was built without OpenMP. Rebuild with gcc -fopenmp (e.g. w64devkit) before packaging."
    }
}
if (!(Test-Path "$desktop\node_modules\electron\dist\electron.exe")) {
    throw "Electron is missing from $desktop\node_modules. Run npm ci before staging the release."
}
$buildRoot = Join-Path $REPO "build"
New-Item -ItemType Directory -Force $buildRoot | Out-Null
$stage = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $buildRoot "Sheep"
} else {
    $outputPath = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
        $OutputDir
    } else {
        Join-Path $REPO $OutputDir
    }
    [System.IO.Path]::GetFullPath($outputPath)
}
$stagePrefix = [System.IO.Path]::GetFullPath($buildRoot).TrimEnd('\') + '\'
if (!$stage.StartsWith($stagePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must be a child directory of $buildRoot."
}

# ---------- stage directory ----------
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$stage\app\mlmi\backend" | Out-Null
New-Item -ItemType Directory -Force "$stage\app\mlmi\herders" | Out-Null
New-Item -ItemType Directory -Force "$stage\app\mlmi\desktop" | Out-Null
New-Item -ItemType Directory -Force "$stage\app\tooling" | Out-Null
New-Item -ItemType Directory -Force "$stage\app\sml" | Out-Null

# ---------- Python backend ----------
Copy-Item -Recurse "$REPO\mlmi\backend\*" "$stage\app\mlmi\backend"
Copy-Item "$REPO\mlmi\__init__.py" "$stage\app\mlmi\__init__.py"
Copy-Item -Recurse "$REPO\mlmi\herders\*" "$stage\app\mlmi\herders"
Copy-Item -Recurse "$REPO\tooling\*" "$stage\app\tooling"
Copy-Item "$REPO\pyproject.toml" "$stage\app\pyproject.toml"
Copy-Item "$REPO\README.md","$REPO\LICENSE","$REPO\NOTICE" "$stage\app"
# the Guide's footer sends people to docs\USER_GUIDE.md: it ships, with the manual
New-Item -ItemType Directory -Force "$stage\app\docs" | Out-Null
Copy-Item "$REPO\docs\USER_GUIDE.md","$REPO\docs\Sheep-Manual.pdf" "$stage\app\docs"

# ---------- Engine ----------
Write-Host "Copying engine binaries..."
foreach ($name in @("sml-server.exe", "sml-cli.exe")) {
    $source = Join-Path $engine $name
    Copy-Item $source "$stage\app\sml"
}
# both GPU kernel builds: the standard one, and the wide one for models wider than 16384
# (Qwen2.5-7B and larger). Without the wide one those models stay off the GPU.
foreach ($ptx in @("sml-gpu-fwd.ptx", "sml-gpu-fwd-wide.ptx")) {
    if (-not (Test-Path (Join-Path $engine $ptx))) {
        Write-Warning "$ptx is missing: build it in sml with: make $ptx CLANG=clang"
    }
}
Get-ChildItem "$engine\*.dll","$engine\sml-gpu-fwd.ptx","$engine\sml-gpu-fwd-wide.ptx" -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-Item $_.FullName "$stage\app\sml" }
# the Vulkan engine for AMD, Intel and any other GPU, and its kernels (build: make vk)
if ((Test-Path "$engine\sml-vk.exe") -and (Test-Path "$engine\vk\mv4.spv")) {
    Copy-Item "$engine\sml-vk.exe" "$stage\app\sml"
    New-Item -ItemType Directory -Force "$stage\app\sml\vk" | Out-Null
    Copy-Item "$engine\vk\*.spv" "$stage\app\sml\vk"
} else {
    Write-Warning "sml-vk.exe or sml\vk\*.spv is missing: other GPUs fall back to DirectML. Build in sml: make vk"
}

# ---------- Electron desktop ----------
Write-Host "Copying Electron app..."
Copy-Item "$desktop\main.js","$desktop\preload.js","$desktop\package.json" "$stage\app\mlmi\desktop"
Copy-Item "$desktop\package-lock.json" "$stage\app\mlmi\desktop" -ErrorAction SilentlyContinue
Copy-Item -Recurse "$desktop\ui" "$stage\app\mlmi\desktop\ui"
if (Test-Path "$desktop\node_modules") {
    Copy-Item -Recurse "$desktop\node_modules" "$stage\app\mlmi\desktop\node_modules"
}
Copy-Item "$desktop\*.ico" "$stage\app\mlmi\desktop" -ErrorAction SilentlyContinue
Copy-Item "$desktop\*.png" "$stage\app\mlmi\desktop" -ErrorAction SilentlyContinue
Copy-Item "$desktop\*.ico" "$stage\" -ErrorAction SilentlyContinue
Copy-Item "$desktop\*.png" "$stage\" -ErrorAction SilentlyContinue
if (!(Test-Path "$stage\app\mlmi\desktop\sheep.png") -or !(Test-Path "$stage\app\mlmi\desktop\icon.ico")) {
    throw "The staged Sheep window icon files are missing."
}

# ---------- Launcher ----------
@"
@echo off
set "MLMI_ROOT=%~dp0app"
set "SHEEP_ENV=%LOCALAPPDATA%\Sheep\venv"
set "SHEEP_PY=%LOCALAPPDATA%\Sheep\python\python.exe"
rem A Python that really runs (Windows' Store alias answers to "python" but is not one), else
rem Sheep's own copy, fetched once from python.org into Sheep's folder: judges' PCs have none.
set "PY="
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if not defined PY if exist "%SHEEP_PY%" set "PY=%SHEEP_PY%"
if not defined PY (
  echo Getting Python 3.12 for Sheep, once ^(about 25 MB^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "`$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile (Join-Path `$env:TEMP 'sheep-python.exe')"
  start "" /wait "%TEMP%\sheep-python.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0 AssociateFiles=0 TargetDir="%LOCALAPPDATA%\Sheep\python"
  if exist "%SHEEP_PY%" set "PY=%SHEEP_PY%"
)
if not defined PY (
  echo Could not get Python for Sheep. Check the internet connection and run Sheep again.
  pause
  exit /b 1
)
if not exist "%SHEEP_ENV%\Scripts\python.exe" (
  echo Preparing the private Python environment, once...
  "%PY%" -m venv "%SHEEP_ENV%"
  if errorlevel 1 (
    echo Could not create Sheep's Python environment.
    pause
    exit /b 1
  )
)
set "MLMI_PYTHON=%SHEEP_ENV%\Scripts\python.exe"
rem Install once, and again only when the dependency list changes, so later
rem launches are quick and work offline.
fc /b "%MLMI_ROOT%\pyproject.toml" "%SHEEP_ENV%\installed-pyproject.toml" >nul 2>&1
if errorlevel 1 (
  echo Installing Sheep's backend dependencies, once...
  "%MLMI_PYTHON%" -m pip install --disable-pip-version-check "%MLMI_ROOT%"
  if errorlevel 1 (
    echo Could not install backend dependencies. Check your internet connection and retry.
    pause
    exit /b 1
  )
  copy /y "%MLMI_ROOT%\pyproject.toml" "%SHEEP_ENV%\installed-pyproject.toml" >nul
)
rem NVIDIA: pictures and video need CUDA PyTorch (DirectML does not fit a 4 GB card).
rem It installs once, in the background, while Sheep opens; Settings shows its progress.
where nvidia-smi >nul 2>&1
if not errorlevel 1 if not exist "%LOCALAPPDATA%\Sheep\runtimes\cuda-state.json" (
  start "" /b cmd /c ""%MLMI_PYTHON%" -m mlmi.backend.gpu_runtime install --if-nvidia > "%LOCALAPPDATA%\Sheep\cuda-install.log" 2>&1"
)
cd /d "%MLMI_ROOT%\mlmi\desktop"
rem Do not inherit a shell's Electron-as-Node setting into the desktop app.
set "ELECTRON_RUN_AS_NODE="
start "" "%MLMI_ROOT%\mlmi\desktop\node_modules\electron\dist\electron.exe" "%MLMI_ROOT%\mlmi\desktop"
"@ | Out-File "$stage\Sheep.cmd" -Encoding ascii

# ---------- a README for the build ----------
@"
Sheep - a local AI with its own apps.

What you have:
  Sheep.cmd     - double-click to start the app (Electron + backend + engine)
  app/sml/      - the SML inference engine (sml-server.exe and its files)
  app/          - Python package, tooling, UI and Electron frame

First run:
  Choose a model using a Hugging Face ID, an original checkpoint folder, or an
  existing Sheep .ssf file. Sheep checks architecture and hardware fit before
  it builds; unsupported architectures are refused rather than guessed.

What it needs:
  - Python 3.9 or later on PATH, or none: the launcher then fetches Python 3.12 once; it creates a private environment
    and installs app/pyproject.toml dependencies on first run
  - An NVIDIA GPU with 4 GB or more for best performance; CPU-only works too
  - The model download on first run (about 2 GB, once)

The DirectML runtime for AMD/Intel GPUs downloads automatically when such a
GPU is found (about 1.5 GB, once, optional).
An internet connection is needed on first launch for Python packages and model
downloads.

License:
  MIT. See LICENSE and NOTICE in the Sheep repository. Model weights retain
  their own upstream licenses.
"@ | Out-File "$stage\README.txt" -Encoding utf8

Write-Host "Build is in $stage"
Write-Host "Size: $([math]::Round(((Get-ChildItem $stage -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB), 1)) MB"
Write-Host ""
Write-Host "To make an installer, install Inno Setup 6 and run:"
Write-Host "  iscc installer\sheep.iss"
