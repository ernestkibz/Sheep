#pragma once

// A resident worker team for decode.
//
// A decode token on the 1B is ~65 small matvecs of 0.1-1 ms each, and an OpenMP
// parallel region costs ~30 us to fork and join on this toolchain -- ~2 ms of a
// 30 ms token, measured (scratchpad kbd3: one region per token instead of one
// per matvec was 1.08-1.09x, and spinning OpenMP threads did not recover it).
// These workers stay awake between dispatches and claim chunks from one atomic
// counter, so a dispatch costs a store and a few cache-line transfers.
//
// Workers spin for ~2 ms after their last job, then sleep, so a prompt read by
// the OpenMP paths after decode does not share its cores with spinning threads.

// Start n_threads - 1 workers (the caller is the n-th). Safe to call again; a
// changed count restarts the team. n_threads <= 1 runs everything inline.
void sml_pool_init(int n_threads);
void sml_pool_free(void);

// Run fn(ctx, k) for k in [0, n) across the team, the caller included; returns
// when every k has run AND every worker has left the job.
void sml_pool_for(int n, void (*fn)(void * ctx, int k), void * ctx);

int sml_pool_threads(void);
