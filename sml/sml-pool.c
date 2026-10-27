#include "sml-pool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#include <x86intrin.h>
#define POOL_PAUSE() _mm_pause()
static inline unsigned long long pool_ticks(void) { return __rdtsc(); }
// ~2 ms at the 3 GHz-class TSC of the parts this runs on; exactness does not matter
#define POOL_SPIN_TICKS 6000000ULL
#else
#define POOL_PAUSE() ((void) 0)
static unsigned long long pool_ticks_n;
static inline unsigned long long pool_ticks(void) { return ++pool_ticks_n; }
#define POOL_SPIN_TICKS 2000000ULL
#endif

// One job at a time. Main writes fn/ctx/n and resets the counters, then bumps
// gen; a worker that sees a new gen claims chunks from `next` until they run
// out and checks out through `done`. Main returns only when all workers have
// checked out, so no worker can still be reading a job main is about to rewrite.
static struct {
    _Alignas(64) atomic_uint gen;
    _Alignas(64) atomic_int  next;
    _Alignas(64) atomic_int  done;
    _Alignas(64) atomic_int  sleepers;
    atomic_int quit;
    void (*fn)(void *, int);
    void * ctx;
    int n;
    int nworkers;
    pthread_t * th;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
} P;

static int pool_up = 0;

static void run_chunks(void) {
    for (;;) {
        const int k = atomic_fetch_add_explicit(&P.next, 1, memory_order_relaxed);
        if (k >= P.n) break;
        P.fn(P.ctx, k);
    }
}

static void * worker(void * unused) {
    (void) unused;
    // gen is 0 when the team starts; reading it here instead could skip a job
    // dispatched before this thread first ran, and main would wait for it forever
    unsigned seen = 0;
    for (;;) {
        unsigned g = atomic_load_explicit(&P.gen, memory_order_acquire);
        if (g == seen) {
            const unsigned long long t0 = pool_ticks();
            while ((g = atomic_load_explicit(&P.gen, memory_order_acquire)) == seen) {
                if (atomic_load_explicit(&P.quit, memory_order_relaxed)) return NULL;
                if (pool_ticks() - t0 > POOL_SPIN_TICKS) {
                    // sleepers is raised BEFORE gen is re-read under the lock, and
                    // main bumps gen BEFORE reading sleepers (both seq_cst), so a
                    // wake-up cannot fall between them
                    atomic_fetch_add(&P.sleepers, 1);
                    pthread_mutex_lock(&P.mu);
                    while (atomic_load(&P.gen) == seen && !atomic_load(&P.quit))
                        pthread_cond_wait(&P.cv, &P.mu);
                    pthread_mutex_unlock(&P.mu);
                    atomic_fetch_sub(&P.sleepers, 1);
                    if (atomic_load(&P.quit)) return NULL;
                    break;
                }
                POOL_PAUSE();
            }
            g = atomic_load_explicit(&P.gen, memory_order_acquire);
        }
        seen = g;
        run_chunks();
        atomic_fetch_add_explicit(&P.done, 1, memory_order_release);
    }
}

void sml_pool_free(void) {
    if (!pool_up) return;
    atomic_store(&P.quit, 1);
    pthread_mutex_lock(&P.mu);
    pthread_cond_broadcast(&P.cv);
    pthread_mutex_unlock(&P.mu);
    for (int i = 0; i < P.nworkers; i++) pthread_join(P.th[i], NULL);
    free(P.th);
    pthread_mutex_destroy(&P.mu);
    pthread_cond_destroy(&P.cv);
    pool_up = 0;
}

void sml_pool_init(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (pool_up && P.nworkers == n_threads - 1) return;
    sml_pool_free();
    atomic_store(&P.gen, 0);
    atomic_store(&P.next, 0);
    atomic_store(&P.done, 0);
    atomic_store(&P.sleepers, 0);
    atomic_store(&P.quit, 0);
    P.nworkers = n_threads - 1;
    P.th = (pthread_t *) calloc((size_t) (P.nworkers > 0 ? P.nworkers : 1), sizeof(pthread_t));
    pthread_mutex_init(&P.mu, NULL);
    pthread_cond_init(&P.cv, NULL);
    for (int i = 0; i < P.nworkers; i++) pthread_create(&P.th[i], NULL, worker, NULL);
    pool_up = 1;
}

int sml_pool_threads(void) { return pool_up ? P.nworkers + 1 : 1; }

void sml_pool_for(int n, void (*fn)(void * ctx, int k), void * ctx) {
    if (n <= 0) return;
    if (!pool_up || P.nworkers == 0 || n == 1) {
        for (int k = 0; k < n; k++) fn(ctx, k);
        return;
    }
    P.fn = fn;
    P.ctx = ctx;
    P.n = n;
    atomic_store_explicit(&P.next, 0, memory_order_relaxed);
    atomic_store_explicit(&P.done, 0, memory_order_relaxed);
    atomic_fetch_add(&P.gen, 1);                      // seq_cst: publishes the job
    if (atomic_load(&P.sleepers) > 0) {
        pthread_mutex_lock(&P.mu);
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.mu);
    }
    run_chunks();
    while (atomic_load_explicit(&P.done, memory_order_acquire) < P.nworkers) POOL_PAUSE();
}
