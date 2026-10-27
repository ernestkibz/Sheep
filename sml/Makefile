# SML build. Replaces the CMake tree, which existed to configure a dozen
# backends and a model zoo we no longer carry.
#
#   make            build everything
#   make bench      kernel bandwidth in isolation (gate G2)
#   make run        generate from a model:  make run MODEL=path.ssf
#   make serve      OpenAI-compatible endpoint on :8080

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm -lpthread
ifeq ($(OS),Windows_NT)
  LDFLAGS += -lws2_32
  EXT = .exe
  EXTRA = sml-export.dll sml-quant.dll
endif

# Headers are dependencies too. Without this, changing sml-model.h or
# sml-kernel.h rebuilt NOTHING, and the stale binary then reported a v9
# file as unreadable and its kernel as broken -- a format change looked
# like a wiring bug for as long as it took to notice.
HDRS = sml-kernel.h sml-pool.h sml-model.h sml-run.h sml-vocab.h sml-verify.h sml-vision.h

CORE = sml-kernel.c sml-pool.c sml-model.c
ENG  = $(CORE) sml-run.c sml-vocab.c

BINS = sml-cli$(EXT) sml-server$(EXT) sml-check$(EXT) sml-kernel-bench$(EXT) sml-batched$(EXT)

# FIRST target, so a bare `make` builds the engine. The GPU rules below are
# opt-in and must not become the default by being written above this line.
all: $(BINS) $(EXTRA)

# ---------------------------------------------------------------- GPU (opt-in)
#
# Deliberately NOT part of `all`. The GPU path needs clang (for NVPTX) and an
# NVIDIA driver, and it is a KERNEL, not a decode path -- see RESULTS.md.
#
# It needs no CUDA toolkit: clang emits PTX and nvcuda.dll, which ships with the
# display driver, loads it. That is why there is no nvcc anywhere in this file.
CLANG   ?= clang
GPU_ARCH ?= sm_86
PTXFLAGS = -x cuda --cuda-device-only -nocudainc -nocudalib            --offload-arch=$(GPU_ARCH) -Xclang -target-feature -Xclang +ptx75 -O3

sml-gpu.ptx: sml-gpu.cu
	$(CLANG) $(PTXFLAGS) -S -o $@ $<

sml-gpu-bench$(EXT): sml-gpu-bench.c $(CORE) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

# clang 18 cannot emit dp4a, so the dp4a kernels call a placeholder that
# ptx_dp4a.py replaces with the instruction after compilation.
sml-gpu-fwd.ptx: sml-gpu-fwd.cu sml-gpu-fast.cu ../tooling/ptx_dp4a.py
	$(CLANG) $(PTXFLAGS) -S -o sml-gpu-fwd.raw.ptx sml-gpu-fwd.cu
	python ../tooling/ptx_dp4a.py sml-gpu-fwd.raw.ptx $@

# the same kernels with room for activations up to 32768 wide (FFN 18944 on Qwen2.5-7B);
# slower on small models (occupancy), so the host loads it only for a model that needs it
sml-gpu-fwd-wide.ptx: sml-gpu-fwd.cu sml-gpu-fast.cu ../tooling/ptx_dp4a.py
	$(CLANG) $(PTXFLAGS) -DMAX_W=8192 -DMAX_G=1024 -S -o sml-gpu-fwd-wide.raw.ptx sml-gpu-fwd.cu
	python ../tooling/ptx_dp4a.py sml-gpu-fwd-wide.raw.ptx $@

# ---------------------------------------------------------------- Vulkan (AMD, Intel, any GPU)
# sml-vk runs the file's own 4/6-bit codes on any Vulkan GPU; vulkan-1.dll ships with every
# driver and is loaded at run time, so only the headers and a GLSL compiler are needed here.
VKINC   ?= C:/Users/earne/vk-tools/headers/Vulkan-Headers-1.3.290/include
# the Vulkan shader compiler: set GLSLANG to its path, or put glslang on PATH
GLSLANG ?= glslang
VK_SPV  = $(patsubst %.comp,%.spv,$(wildcard vk/*.comp))

vk/%.spv: vk/%.comp vk/common.glsl
	$(GLSLANG) -V --target-env vulkan1.1 -o $@ $<

vk: sml-vk$(EXT) $(VK_SPV)

sml-vk$(EXT): sml-vk.c $(ENG) $(HDRS)
	$(CC) $(CFLAGS) -I$(VKINC) -o $@ sml-vk.c $(ENG) $(LDFLAGS) -lws2_32

sml-gpu-fwd$(EXT): sml-gpu-fwd.c $(ENG) $(HDRS) sml-gpu-drv.h
	$(CC) $(CFLAGS) -Wno-unused-function -o $@ sml-gpu-fwd.c $(ENG) $(LDFLAGS)

# Development copies: build and test GPU changes without replacing the binary a
# running race is timing. Run with SML_PTX=sml-gpu-fwd-dev.ptx.
gpu-dev: sml-gpu-fwd-dev.ptx sml-gpu-fwd-dev$(EXT)

sml-gpu-fwd-dev.ptx: sml-gpu-fwd.cu sml-gpu-fast.cu ../tooling/ptx_dp4a.py
	$(CLANG) $(PTXFLAGS) -S -o sml-gpu-fwd-dev.raw.ptx sml-gpu-fwd.cu
	python ../tooling/ptx_dp4a.py sml-gpu-fwd-dev.raw.ptx $@

sml-gpu-fwd-dev$(EXT): sml-gpu-fwd.c $(ENG) $(HDRS) sml-gpu-drv.h
	$(CC) $(CFLAGS) -Wno-unused-function -o $@ sml-gpu-fwd.c $(ENG) $(LDFLAGS)

gpu: sml-gpu.ptx sml-gpu-bench$(EXT) sml-gpu-fwd.ptx sml-gpu-fwd$(EXT)
	@echo "built. run: ./sml-gpu-bench MODEL.ssf TENSOR [reps] [sml-gpu.ptx]" 

sml-cli$(EXT): sml-cli.c $(ENG) sml-vision.c sml-json.c $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

sml-batched$(EXT): sml-batched.c $(ENG) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

# The engine's view of a model for a forward pass written elsewhere (the
# DirectML worker for AMD and Intel GPUs): weights, norms, rotary tables, grades.
sml-export.dll: sml-export.c $(ENG) $(HDRS)
	$(CC) $(CFLAGS) -shared -static -o $@ sml-export.c $(ENG) $(LDFLAGS)

# The Sheep Quantizer's searches, fused (tooling/sml_grid_c.py loads it): about
# 36x the numpy grid search with the same fields.
sml-quant.dll: sml-quant.c
	$(CC) -O3 -march=native -fopenmp -shared -static -o $@ sml-quant.c -lm

# The GPU forward pass, linked into the server as a library (sml-gpu.h).
sml-gpu-lib.o: sml-gpu-fwd.c sml-gpu.h sml-gpu-drv.h $(HDRS)
	$(CC) $(CFLAGS) -Wno-unused-function -DSML_GPU_LIB -c -o $@ sml-gpu-fwd.c

sml-server$(EXT): sml-server.c $(ENG) sml-json.c sml-gpu-lib.o sml-gpu.h $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS)

# no OpenMP: this is a correctness check, and determinism matters more than speed
sml-check$(EXT): sml-check.c $(CORE) $(HDRS)
	$(CC) -O3 -march=native -o $@ $^ -lm

sml-model-test$(EXT): sml-model-test.c sml-model.c sml-kernel.c $(HDRS)
	$(CC) -O2 -o $@ $(filter %.c,$^) -lm

test-model: sml-model-test$(EXT)
	./sml-model-test$(EXT)

sml-kernel-bench$(EXT): sml-kernel-bench.c sml-kernel.c $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

MODEL ?= $(HOME)/models/qwen2.5-3b.ssf
DEPTH ?= 3
THREADS ?= 6

bench: sml-kernel-bench$(EXT)
	./sml-kernel-bench$(EXT) $(THREADS) 30

run: sml-cli$(EXT)
	./sml-cli$(EXT) "$(MODEL)" --prompt "The capital of France is" \
	  --depth $(DEPTH) --gen 32 --threads $(THREADS)

serve: sml-server$(EXT)
	./sml-server$(EXT) "$(MODEL)" --port 8080 --depth $(DEPTH) --threads $(THREADS)

clean:
	rm -f $(BINS) sml-model-test$(EXT)

.PHONY: all bench run serve test-model clean
