// The GPU forward pass as a library, for sml-server (sml-gpu-fwd.c built with
// -DSML_GPU_LIB). One model and one GPU KV cache per process. Positions are
// absolute: a prompt may continue a prefix the cache already holds. Every call
// returns NULL (or 0) on failure instead of ending the process; the caller then
// serves on the CPU. No windows.h here, so it mixes with winsock code.
#pragma once

// Load the model on the GPU. gpu_layers > 0 puts that many leading layers on the
// card and runs the rest on the CPU (hybrid); <= 0 lets free VRAM decide.
// Returns 1, or 0 with the reason in why.
int sml_gpu_open(const char * path, int n_ctx, int threads, int gpu_layers, char * why, int nwhy);

// Read ids[0..n) at positions pos0..pos0+n-1; logits of the last one (host memory).
const float * sml_gpu_prefill(const int * ids, int pos0, int n);

// One token at position pos; its logits (host memory).
const float * sml_gpu_step(int tok, int pos);

// Layers on the card, and of how many.
int sml_gpu_layers(int * of);

// How a prompt is read: 2 every batched kernel (the fastest), 1 without the
// tensor-core one, 0 a token at a time, as decode does. The caller measures
// which of them writes this model's own words (sml-server does it at load) and
// never goes above what the model was opened able to do.
void sml_gpu_prompt_mode(int level);

// A middle stage: after the GPU's layers, fn(rows, pos0, n) runs the next
// `layers` layers on the n hidden rows (n x n_embd floats, in place) on another
// device -- an AMD or Intel GPU through the server's DirectML worker -- and the
// CPU continues after them. fn returns 1, or 0 on failure (the GPU path retires).
void sml_gpu_set_middle(int layers, int (*fn)(float * rows, int pos0, int n));
