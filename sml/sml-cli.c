// sml-cli: run the SML engine and report what it moved.
//
// Reports bytes/token alongside tok/s on purpose. The architecture optimises
// bytes across the memory bus, so a tok/s figure alone cannot distinguish a real
// win from a lucky one -- and this project has been burned before by numbers
// that graded the mechanism instead of the outcome.
//
//   sml-cli MODEL.ssf --tokens 1,2,3 --depth 2 --gen 32 --threads 4
//   sml-cli MODEL.ssf --tokens-file ids.txt --logits out.bin

#include "sml-run.h"
#include "sml-vocab.h"
#include "sml-vision.h"

#include <stdio.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
static double now_s(void) { return omp_get_wtime(); }
#else
#include <time.h>
static double now_s(void) { return (double) clock() / CLOCKS_PER_SEC; }
#endif

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s MODEL.ssf [--tokens a,b,c] [--depth N] [--gen N]\n"
                "          [--threads N] [--ctx N] [--logits out.bin]\n"
                "          [--baseline TOKS_PER_S[,GB_PER_TOKEN]]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const char * ids_arg = "9707,11,358,1079";   // arbitrary but deterministic
    const char * logits_out = NULL;
    const char * image_path = NULL;    // raw f32, 3 x image x image, normalized
    const char * vision_out = NULL;    // dump the tower's rows (grading)
    const char * ask = NULL;           // a question about --image, framed as a chat turn
    int warmup = 0;                    // read the prompt once, untimed, before the timed read
    const char * tokenize = NULL;      // "-": stdin, texts split by 0x1E, one line of ids each
    const char * prompt = NULL;
    int verify_batch = 0;
    int batch_scaling = 0;
    const char * draft_path = NULL;
    int spec_k = 2;   // measured: small k wins, see RESULTS.md
    int depth = 0, gen = 0, threads = 4, n_ctx = 512;
    // A baseline is a property of the MODEL and the MACHINE, never of this
    // binary. It was compiled in as 11.53 tok/s and 1.920 GB/token -- the
    // Qwen2.5-3B figures -- so every other model printed a ratio against a
    // number that had nothing to do with it. Unset means no ratio is printed.
    double base_tps = 0.0, base_gbt = 0.0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens")  && i + 1 < argc) ids_arg   = argv[++i];
        else if (!strcmp(argv[i], "--prompt")  && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--depth")   && i + 1 < argc) {
            const char * v = argv[++i];
            // "auto" fits the read to the machine the file landed on.
            depth = strcmp(v, "auto") ? atoi(v) : -1;
        }
        else if (!strcmp(argv[i], "--gen")     && i + 1 < argc) gen   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx")     && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits")  && i + 1 < argc) logits_out = argv[++i];
        else if (!strcmp(argv[i], "--image")   && i + 1 < argc) image_path = argv[++i];
        else if (!strcmp(argv[i], "--vision-out") && i + 1 < argc) vision_out = argv[++i];
        else if (!strcmp(argv[i], "--ask")     && i + 1 < argc) ask = argv[++i];
        else if (!strcmp(argv[i], "--warmup")) warmup = 1;
        else if (!strcmp(argv[i], "--tokenize") && i + 1 < argc) tokenize = argv[++i];
        else if (!strcmp(argv[i], "--verify-batch")) verify_batch = 1;
        else if (!strcmp(argv[i], "--batch-scaling")) batch_scaling = 1;
        else if (!strcmp(argv[i], "--draft") && i + 1 < argc) draft_path = argv[++i];
        else if (!strcmp(argv[i], "--spec-k") && i + 1 < argc) spec_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--baseline") && i + 1 < argc) {
            // "TOKS_PER_S[,GB_PER_TOKEN]" measured on THIS machine for THIS model.
            const char * a = argv[++i];
            base_tps = atof(a);
            const char * c = strchr(a, ',');
            if (c) base_gbt = atof(c + 1);
        }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    sml_model m;
    if (!sml_model_open(&m, path)) return 1;
    const sml_hdr * h = m.hdr;

    if (tokenize) {
        // the tokenizer alone, for tooling/sml_tok_check.py: no weights are read
        sml_vocab tv;
        if (!sml_vocab_init(&tv, &m)) return 1;
        FILE * in = strcmp(tokenize, "-") ? fopen(tokenize, "rb") : stdin;
        if (!in) return 1;
#ifdef _WIN32
        if (in == stdin) _setmode(_fileno(stdin), _O_BINARY);
#endif
        size_t cap = 1 << 16, n = 0;
        char * all = malloc(cap);
        for (size_t k; (k = fread(all + n, 1, cap - n - 1, in)) > 0; ) {
            n += k;
            if (n + 1 >= cap) all = realloc(all, cap *= 2);
        }
        all[n] = 0;
        int32_t * ids = malloc(sizeof(int32_t) * (n + 16) * 4);
        for (char * t = all; t; ) {
            char * e = strchr(t, 0x1e);
            if (e) *e = 0;
            const int k = sml_encode(&tv, t, ids, (int) (n + 16) * 4);
            for (int q = 0; q < k; q++) printf(q ? " %d" : "%d", ids[q]);
            printf("\n");
            t = e ? e + 1 : NULL;
        }
        return 0;
    }

    printf("%s\n", path);
    printf("  arch layers=%u embd=%u head=%u kv=%u ff=%u vocab=%u\n",
           h->n_layer, h->n_embd, h->n_head, h->n_head_kv, h->n_ff, h->n_vocab);
    printf("  ladder %u levels x %u bits, group %u\n",
           h->n_levels, h->level_bits[0], h->group);
    for (uint32_t d = 1; d <= h->n_levels; d++)
        printf("  depth %u: %.3f GB/token\n", d,
               (double) sml_bytes_at_depth(&m, (int) d) / 1e9);
    // Printed only when supplied. A baseline belongs to a MODEL on a MACHINE,
    // so measure it interleaved with this run, idle, at the reference engine's
    // OWN best thread count -- an earlier fixed -t 3 was never checked against
    // their optimum and every margin quoted against it was too generous.
    if (base_tps > 0.0)
        printf("  baseline: %.2f tok/s, supplied for this model on this machine\n",
               base_tps);

    sml_ctx c;
    if (!sml_ctx_init(&c, &m, n_ctx, threads)) return 1;
    // Whether the schedule came from the file or from the engine guessing. The
    // difference is the whole claim, so it is reported rather than assumed.
    printf("  groups: %d declared by the file, %d inferred by the engine\n\n",
           c.grp_from_file, c.grp_inferred);
    if (depth > 0) c.depth = depth;
    else if (depth < 0) {
        const uint64_t freeb = sml_free_ram();
        c.depth = sml_fit_depth(&m, 0);
        printf("  depth auto: %.1f GB free -> depth %d (%.3f GB/token)\n",
               (double) freeb / 1e9, c.depth,
               (double) sml_bytes_at_depth(&m, c.depth) / 1e9);
    }

    sml_vocab vocab;
    const int have_vocab = sml_vocab_init(&vocab, &m);

    int32_t ids[512];
    int n_ids = 0;
    int stop_id = -1;   // a vision turn ends at the model's end-of-utterance
    if (ask) {
        sml_vcfg av;
        if (!have_vocab || !sml_vision_cfg(&m, &av)) {
            fprintf(stderr, "--ask needs a vision model with a vocabulary\n");
            return 1;
        }
        n_ids = sml_vision_prompt(&av, &vocab, ask, ids, 512);
        if (n_ids <= 0) { fprintf(stderr, "this file carries no chat frame for images\n"); return 1; }
        stop_id = av.tok_end;
        printf("ask: %s\n  -> %d tokens\n\n", ask, n_ids);
    } else if (prompt) {
        if (!have_vocab) { fprintf(stderr, "no vocab in model\n"); return 1; }
        n_ids = sml_encode(&vocab, prompt, ids, 512);
        if (n_ids <= 0) { fprintf(stderr, "tokenizer produced nothing\n"); return 1; }
        printf("prompt: %s\n  -> %d tokens:", prompt, n_ids);
        for (int i = 0; i < n_ids && i < 16; i++) printf(" %d", ids[i]);
        printf("\n\n");
    } else {
        char * s = strdup(ids_arg);
        for (char * t = strtok(s, ","); t && n_ids < 512; t = strtok(NULL, ","))
            ids[n_ids++] = atoi(t);
        free(s);
    }

    // How much does verifying nb tokens actually cost, relative to one step?
    // The whole speculative argument rests on this being near 1.0 -- one weight
    // read serving nb candidates. If it climbs linearly, the batch path is
    // compute-bound and speculation cannot pay however good the draft is.
    if (batch_scaling) {
        float * bl = malloc((size_t) SML_MAX_BATCH * h->n_vocab * sizeof(float));
        int32_t tk[SML_MAX_BATCH];
        for (int i = 0; i < SML_MAX_BATCH; i++) tk[i] = ids[i % n_ids];
        // warm the page cache so the first nb is not paying for the mmap
        sml_forward_batch(&c, tk, 0, 1, c.depth, bl);
        printf("batch scaling, %d reps each (nb=1 is the unit):\n", 3);
        printf("  %3s %10s %10s %10s\n", "nb", "sec", "vs nb=1", "per token");
        double unit = 0.0;
        for (int nb = 1; nb <= SML_MAX_BATCH; nb++) {
            double best = 1e18;
            for (int rep = 0; rep < 3; rep++) {
                const double t0 = now_s();
                sml_forward_batch(&c, tk, 0, nb, c.depth, bl);
                const double dt = now_s() - t0;
                if (dt < best) best = dt;
            }
            if (nb == 1) unit = best;
            printf("  %3d %10.4f %10.3fx %10.4f\n",
                   nb, best, best / unit, best / nb);
        }
        free(bl);
        sml_ctx_free(&c);
        sml_model_close(&m);
        return 0;
    }

    // Batched verification must produce exactly what stepping one token at a
    // time produces, or speculative decoding silently accepts wrong tokens.
    // Checked here before anything is built on it.
    if (verify_batch) {
        const int nb = n_ids < SML_MAX_BATCH ? n_ids : SML_MAX_BATCH;
        float * seq = malloc((size_t) nb * h->n_vocab * sizeof(float));
        for (int i = 0; i < nb; i++) {
            const float * l = sml_forward(&c, ids[i], i, c.depth);
            memcpy(seq + (size_t) i * h->n_vocab, l,
                   (size_t) h->n_vocab * sizeof(float));
        }
        // The KV cache is fp16 (uint16_t), not float. Sizing this wipe with
        // sizeof(float) overran the allocation by exactly 2x and took the
        // process down every time -- which is why this gate has never once run
        // to completion, and why the batch path was believed sound without ever
        // having been checked.
        const size_t kvspan = (size_t) h->n_layer * c.n_ctx
                            * (h->n_head_kv * (h->n_embd / h->n_head));
        memset(c.k_cache, 0, kvspan * sizeof(uint16_t));
        memset(c.v_cache, 0, kvspan * sizeof(uint16_t));

        float * bat = malloc((size_t) nb * h->n_vocab * sizeof(float));
        sml_forward_batch(&c, ids, 0, nb, c.depth, bat);

        printf("batched vs sequential, %d tokens:\n", nb);
        printf("  %5s %12s %12s %9s\n", "pos", "seq argmax", "batch argmax", "rel-RMS");
        int allsame = 1;
        for (int i = 0; i < nb; i++) {
            const float * a = seq + (size_t) i * h->n_vocab;
            const float * b = bat + (size_t) i * h->n_vocab;
            double num = 0.0, den = 0.0;
            for (uint32_t j = 0; j < h->n_vocab; j++) {
                const double d = (double) a[j] - b[j];
                num += d * d;
                den += (double) a[j] * a[j];
            }
            const int ia = sml_sample_greedy(a, (int) h->n_vocab);
            const int ib = sml_sample_greedy(b, (int) h->n_vocab);
            allsame &= (ia == ib);
            printf("  %5d %12d %12d %9.2e%s\n", i, ia, ib,
                   sqrt(num / (den + 1e-12)), ia == ib ? "" : "   MISMATCH");
        }
        printf("\n%s\n", allsame
               ? "PASS - batched scoring matches stepping one at a time"
               : "FAIL - batch and sequential disagree; speculation would be unsound");
        free(seq); free(bat);
        sml_ctx_free(&c);
        sml_model_close(&m);
        return allsame ? 0 : 1;
    }

    // ---------------------------------------------------------- speculative
    //
    // A small model proposes k tokens; the big one checks all of them against a
    // SINGLE read of its weights. Every token the big model would have produced
    // anyway is free. Rejected ones cost nothing but the draft time, because
    // the verification pass had to happen regardless.
    //
    // Measured batch scaling says small k wins here: k=2 costs 1.43x one step,
    // k=4 costs 1.84x. Since the extra candidates are nearly full price, a low
    // acceptance rate punishes large k much harder than the usual analysis
    // suggests.
    if (draft_path) {
        sml_model dm;
        if (!sml_model_open(&dm, draft_path)) return 1;
        sml_ctx dc;
        if (!sml_ctx_init(&dc, &dm, n_ctx, threads)) return 1;
        dc.depth = c.depth;
        if (dm.hdr->n_vocab != h->n_vocab) {
            fprintf(stderr, "draft vocab %u != target %u; they must share a tokenizer\n",
                    dm.hdr->n_vocab, h->n_vocab);
            return 2;
        }
        printf("draft: %s (%u layers, %u embd)\n\n",
               draft_path, dm.hdr->n_layer, dm.hdr->n_embd);

        // prime both models on the prompt
        const float * tl = NULL;
        for (int i = 0; i < n_ids; i++) {
            tl = sml_forward(&c,  ids[i], i, c.depth);
                 sml_forward(&dc, ids[i], i, dc.depth);
        }
        int pos = n_ids;
        int cur = sml_sample_greedy(tl, (int) h->n_vocab);

        int32_t dtok[SML_MAX_BATCH], vin[SML_MAX_BATCH];
        float * vlog = malloc((size_t) SML_MAX_BATCH * h->n_vocab * sizeof(float));
        long produced = 0, accepted = 0, rounds = 0;
        const uint64_t b0 = c.bytes_moved + dc.bytes_moved;
        const double g0 = now_s();
        if (have_vocab) printf("  \"");

        while (produced < gen && pos + spec_k + 1 < n_ctx) {
            // draft k tokens, each conditioned on the last
            int prev = cur;
            for (int j = 0; j < spec_k; j++) {
                const float * dl = sml_forward(&dc, prev, pos + j, dc.depth);
                dtok[j] = sml_sample_greedy(dl, (int) h->n_vocab);
                prev = dtok[j];
            }

            // target scores positions pos..pos+k in one pass: input i is the
            // token AT that position, so row i predicts position i+1
            vin[0] = cur;
            for (int j = 0; j < spec_k; j++) vin[j + 1] = dtok[j];
            sml_forward_batch(&c, vin, pos, spec_k + 1, c.depth, vlog);

            // accept the longest prefix the target agrees with
            int acc = 0;
            while (acc < spec_k) {
                const int want = sml_sample_greedy(
                    vlog + (size_t) acc * h->n_vocab, (int) h->n_vocab);
                if (want != dtok[acc]) break;
                acc++;
            }
            // the target's own token at the first disagreement is always valid
            const int bonus = sml_sample_greedy(
                vlog + (size_t) acc * h->n_vocab, (int) h->n_vocab);

            for (int j = 0; j < acc && produced < gen; j++, produced++) {
                if (have_vocab) {
                    char txt[64];
                    const int nb = sml_decode(&vocab, dtok[j], txt, sizeof txt);
                    fwrite(txt, 1, (size_t) nb, stdout);
                }
            }
            if (produced < gen) {
                if (have_vocab) {
                    char txt[64];
                    const int nb = sml_decode(&vocab, bonus, txt, sizeof txt);
                    fwrite(txt, 1, (size_t) nb, stdout);
                }
                produced++;
            }
            fflush(stdout);

            // Rewind to just past what was accepted. Both caches are indexed by
            // position and overwritten in place, so rewinding is only a matter
            // of where the next write lands -- rejected entries are harmless.
            // The draft wrote cache for cur and dtok[0..k-2] -- it never runs on
            // its own LAST proposal. If every proposal was accepted, that token
            // is part of the real sequence and the draft's cache has a hole at
            // its position, so every later round drafts from a wrong history.
            // This is what made a model reject its own drafts (83%, not 100%).
            if (acc == spec_k)
                sml_forward(&dc, dtok[spec_k - 1], pos + spec_k, dc.depth);

            pos += acc + 1;
            cur = bonus;
            accepted += acc;
            rounds++;
        }
        if (have_vocab) printf("\"\n");

        const double dt = now_s() - g0;
        const uint64_t moved = (c.bytes_moved + dc.bytes_moved) - b0;
        printf("\nspeculative k=%d, %ld tokens in %.2fs\n", spec_k, produced, dt);
        if (base_tps > 0.0)
            printf("  %.2f tok/s   vs baseline %.2f  ->  %.2fx\n",
                   produced / dt, base_tps, (produced / dt) / base_tps);
        else
            printf("  %.2f tok/s\n", produced / dt);
        printf("  acceptance %.2f of %d drafted per round (%.0f%%)\n",
               rounds ? (double) accepted / rounds : 0.0, spec_k,
               rounds ? 100.0 * accepted / (rounds * spec_k) : 0.0);
        printf("  %.3f GB/token moved (target + draft)\n",
               produced ? (double) moved / produced / 1e9 : 0.0);

        free(vlog);
        sml_ctx_free(&dc);
        sml_model_close(&dm);
        sml_ctx_free(&c);
        sml_model_close(&m);
        return 0;
    }

    // Batched prompt reading must land where stepping lands, across chunk
    // boundaries, or every answer starts from a wrong state. SML_CHECK_PREFILL=1
    // runs that gate; SML_SEQ_PROMPT=1 reads the prompt the old way.
    if (getenv("SML_CHECK_PREFILL")) {
        float * seq = malloc((size_t) h->n_vocab * sizeof(float));
        const float * l = NULL;
        for (int i = 0; i < n_ids; i++) l = sml_forward(&c, ids[i], i, c.depth);
        memcpy(seq, l, (size_t) h->n_vocab * sizeof(float));
        const float * b = sml_prefill(&c, ids, 0, n_ids, c.depth);
        double num = 0.0, den = 0.0;
        for (uint32_t j = 0; j < h->n_vocab; j++) {
            const double d = (double) seq[j] - b[j];
            num += d * d;
            den += (double) seq[j] * seq[j];
        }
        const int ia = sml_sample_greedy(seq, (int) h->n_vocab);
        const int ib = sml_sample_greedy(b, (int) h->n_vocab);
        printf("prefill vs sequential, %d tokens: argmax %d / %d, rel-RMS %.2e\n%s\n",
               n_ids, ia, ib, sqrt(num / (den + 1e-12)),
               ia == ib ? "PASS" : "FAIL - batched prompt reading disagrees with stepping");
        free(seq);
        sml_ctx_free(&c);
        sml_model_close(&m);
        return ia == ib ? 0 : 1;
    }

    FILE * lf = logits_out ? fopen(logits_out, "wb") : NULL;

    // An image: run the tower once, then each image token in the prompt
    // reads the next row of its output instead of its own embedding.
    float * vis = NULL;
    sml_vcfg vc;
    if (image_path) {
        if (!sml_vision_cfg(&m, &vc)) {
            fprintf(stderr, "%s has no vision tower\n", path);
            return 1;
        }
        const size_t npx = (size_t) 3 * vc.image * vc.image;
        float * px = malloc(npx * sizeof(float));
        vis = malloc((size_t) vc.n_tok * vc.out_dim * sizeof(float));
        FILE * pf = fopen(image_path, "rb");
        if (!px || !vis || !pf || fread(px, sizeof(float), npx, pf) != npx) {
            fprintf(stderr, "cannot read %zu floats of pixels from %s\n", npx, image_path);
            return 1;
        }
        fclose(pf);
        const double tv = now_s();
        if (sml_vision_encode(&m, &vc, px, vis, threads) != 0) return 1;
        printf("image %dx%d -> %d rows in %.2fs\n", vc.image, vc.image, vc.n_tok, now_s() - tv);
        free(px);
        if (vision_out) {
            FILE * vf = fopen(vision_out, "wb");
            if (vf) { fwrite(vis, sizeof(float), (size_t) vc.n_tok * vc.out_dim, vf); fclose(vf); }
        }
    }

    // --warmup: the first read of a freshly mapped file pays for touching every
    // page of the model (~0.8 GB on the 1B -- a fifth of a 28-token prompt), which
    // a running server pays once. The reference's bench runs a warm-up first too.
    // The same positions are rewritten by the timed read, so the result is equal.
    if (warmup && !vis) sml_prefill(&c, ids, 0, n_ids, c.depth);

    const double t0 = now_s();
    const float * logits = NULL;
    int pos = 0;
    if (vis) {
        // stepped: the batched body reads token ids, not rows
        int used = 0;
        for (; pos < n_ids; pos++) {
            c.emb_in = (ids[pos] == vc.image_token_id && used < vc.n_tok)
                       ? vis + (size_t) (used++) * vc.out_dim : NULL;
            logits = sml_forward(&c, ids[pos], pos, c.depth);
        }
        c.emb_in = NULL;
        if (used != vc.n_tok)
            fprintf(stderr, "warning: prompt has %d image tokens, the image gives %d rows\n",
                    used, vc.n_tok);
    } else if (getenv("SML_ALL_LOGITS") && lf) {
        // Every position's logits from index N on, for a quality race scored
        // position by position against another engine's saved distributions:
        // [n_vocab, count] then count rows of n_vocab floats.
        const int from = atoi(getenv("SML_ALL_LOGITS"));
        int32_t meta[2] = { (int32_t) h->n_vocab, n_ids - from > 0 ? n_ids - from : 0 };
        fwrite(meta, sizeof(int32_t), 2, lf);
        for (; pos < n_ids; pos++) {
            logits = sml_forward(&c, ids[pos], pos, c.depth);
            if (pos >= from) fwrite(logits, sizeof(float), h->n_vocab, lf);
        }
        fclose(lf);
        lf = NULL;
    } else if (getenv("SML_SEQ_PROMPT"))
        for (; pos < n_ids; pos++) logits = sml_forward(&c, ids[pos], pos, c.depth);
    else {
        logits = sml_prefill(&c, ids, 0, n_ids, c.depth);
        pos = n_ids;
    }
    const double t_prompt = now_s() - t0;

    if (lf) {
        int32_t meta[2] = { (int32_t) h->n_vocab, c.depth };
        fwrite(meta, sizeof(int32_t), 2, lf);
        fwrite(logits, sizeof(float), h->n_vocab, lf);
    }

    printf("prompt %d tokens in %.2fs (%.1f tok/s)\n",
           n_ids, t_prompt, n_ids / t_prompt);
    printf("  top1=%d  top2-margin=%.3f\n",
           sml_sample_greedy(logits, (int) h->n_vocab),
           sml_top2_margin(logits, (int) h->n_vocab));

    if (gen > 0) {
        const uint64_t b0 = c.bytes_moved;
        const double g0 = now_s();
        int tok = sml_sample_greedy(logits, (int) h->n_vocab);
        if (have_vocab) printf("  \"");
        for (int i = 0; i < gen && pos < n_ctx; i++, pos++) {
            if (tok == stop_id) break;
            if (have_vocab) {
                char txt[64];
                const int nb = sml_decode(&vocab, tok, txt, sizeof txt);
                fwrite(txt, 1, (size_t) nb, stdout);
                fflush(stdout);
            }
            logits = sml_forward(&c, tok, pos, c.depth);
            tok = sml_sample_greedy(logits, (int) h->n_vocab);
        }
        if (have_vocab) printf("\"\n");
        const double dt = now_s() - g0;
        const uint64_t moved = c.bytes_moved - b0;
        printf("\ngenerate %d tokens in %.2fs\n", gen, dt);
        if (base_tps > 0.0)
            printf("  %.2f tok/s   vs baseline %.2f  ->  %.2fx\n",
                   gen / dt, base_tps, (gen / dt) / base_tps);
        else
            printf("  %.2f tok/s\n", gen / dt);
        if (base_gbt > 0.0)
            printf("  %.3f GB/token moved   vs baseline %.3f  ->  %.2fx fewer bytes\n",
                   (double) moved / gen / 1e9, base_gbt,
                   base_gbt * 1e9 / ((double) moved / gen));
        else
            printf("  %.3f GB/token moved\n", (double) moved / gen / 1e9);
        printf("  achieved %.1f GB/s\n", (double) moved / dt / 1e9);

        // The engine sustains well under the kernel's standalone bandwidth.
        // One hypothesis (OpenMP fork/join) was already tested and worth only
        // ~6%, so the rest gets measured rather than guessed at.
        {
            extern double sml_t_matvec, sml_t_prep, sml_t_rope, sml_t_attn;
            const double acc = sml_t_matvec + sml_t_prep + sml_t_rope + sml_t_attn;
            printf("\n  where the time goes (whole run, incl. prompt):\n");
            printf("    matvec (kernel)  %7.3fs\n", sml_t_matvec);
            printf("    gather+quantize  %7.3fs\n", sml_t_prep);
            printf("    rope             %7.3fs\n", sml_t_rope);
            printf("    attention        %7.3fs\n", sml_t_attn);
            printf("    unaccounted      %7.3fs\n", t_prompt + dt - acc);
        }

        // The engine sustains under the kernel's standalone bandwidth and the
        // phase timers put ~95% of the time inside matvec, so the gap is INSIDE
        // the kernel calls. attn_k and attn_v are 256 rows, which across 6
        // threads is 42 rows each -- mostly parallel overhead. This says whether
        // that is actually where it goes, before anything is rebuilt to fix it.
        {
            extern double   sml_shape_time[];
            extern uint64_t sml_shape_bytes[];
            extern uint64_t sml_shape_calls[];
            extern int      sml_shape_rows[];
            printf("\n  matvec by output rows:\n");
            printf("    %8s %8s %9s %9s\n", "rows", "calls", "secs", "GB/s");
            for (int i = 0; i < 8 && sml_shape_rows[i]; i++)
                printf("    %8d %8llu %9.3f %9.1f\n", sml_shape_rows[i],
                       (unsigned long long) sml_shape_calls[i], sml_shape_time[i],
                       sml_shape_time[i] > 0
                         ? (double) sml_shape_bytes[i] / sml_shape_time[i] / 1e9 : 0.0);
        }
    }

    if (lf) fclose(lf);
    if (have_vocab) sml_vocab_free(&vocab);
    sml_ctx_free(&c);
    sml_model_close(&m);
    return 0;
}
