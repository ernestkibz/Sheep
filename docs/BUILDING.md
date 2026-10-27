# Building Sheep

## The engine (Windows)

The engine is C with OpenMP. Build it with **w64devkit**'s gcc: the llvm-mingw toolchain links without
OpenMP and runs on one core.

```
cd sml
$env:PATH = "C:\path\to\w64devkit\bin;$env:PATH"
make CC=gcc LDFLAGS="-lm -lpthread -lws2_32 -static" sml-server.exe
```

`make` (no target) builds `sml-server` and `sml-cli`. The NVIDIA kernels are PTX built by clang
(`make gpu`, `sml-gpu-fwd.ptx` and `sml-gpu-fwd-wide.ptx`); there is no nvcc step, because the display
driver compiles PTX when it loads it. The Vulkan path for AMD and Intel is `make vk` (set `GLSLANG` to your
glslang). `-march=native` picks the AVX2 kernels when the CPU has them; the portable path is correct but
about 2.5x slower.

## The engine (Linux)

```
cd sml && make
./sml-cli MODEL.ssf --prompt "The capital of France is" --gen 32
./sml-server MODEL.ssf --port 8080
```

Needs gcc or clang with OpenMP. The desktop app and the installer are Windows only today.

## The app, from source

```
python -m pip install .            # the backend and its app dependencies
cd mlmi/desktop && npm install     # Electron
cd ../.. && Sheep.cmd              # starts the backend and the window
```

`python -m pip install ".[audio]"` adds speech to text. Picture, video and speech models download on demand,
from inside the app, with their progress shown.

## The installer

```
powershell -File submission/BUILD_INSTALLER.ps1
```

It checks the engine binaries are newer than their sources, gathers the backend, the desktop app and the
engine into `build\Sheep\`, and, with Inno Setup 6 installed, makes `Sheep-Setup.exe` from
`submission/sheep.iss`. On a computer with no Python, the launcher fetches Python 3.12 from python.org into
`%LOCALAPPDATA%\Sheep\python` once, then makes a private environment for Sheep.

No model is bundled. On first run the person adds one (or starts with Nemotron on Nebius), and Sheep builds it
for that computer.
