#include "sml-vocab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------- hashing

static uint32_t fnv(const uint8_t * p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static uint32_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t) x;
}

static uint32_t next_pow2(uint32_t n) {
    uint32_t p = 8;
    while (p < n) p <<= 1;
    return p;
}

// --------------------------------------------------------------------- setup

static const uint8_t * tok_ptr(const sml_vocab * v, uint32_t id, uint32_t * len) {
    *len = v->offsets[id + 1] - v->offsets[id];
    return v->bytes + v->offsets[id];
}

static uint32_t tok_lookup_h(const sml_vocab * v, const uint8_t * s, uint32_t n, uint32_t h) {
    uint32_t i = h & v->tok_mask;
    for (;;) {
        const uint32_t slot = v->tok_slots[i];
        if (slot == UINT32_MAX) return UINT32_MAX;
        uint32_t l;
        const uint8_t * t = tok_ptr(v, slot, &l);
        if (l == n && memcmp(t, s, n) == 0) return slot;
        i = (i + 1) & v->tok_mask;
    }
}

static uint32_t tok_lookup(const sml_vocab * v, const uint8_t * s, uint32_t n) {
    return tok_lookup_h(v, s, n, fnv(s, n));
}

static uint32_t mrg_lookup(const sml_vocab * v, uint32_t a, uint32_t b) {
    const uint64_t key = ((uint64_t) a << 32) | b;
    uint32_t i = mix64(key) & v->mrg_mask;
    for (;;) {
        if (v->mrg_rank[i] == UINT32_MAX) return UINT32_MAX;
        if (v->mrg_keys[i] == key) return v->mrg_rank[i];
        i = (i + 1) & v->mrg_mask;
    }
}

static int pret_init(sml_vocab * v, const uint8_t * q, const uint8_t * end);
static void pret_free(sml_vocab * v);

int sml_vocab_init(sml_vocab * v, const sml_model * m) {
    memset(v, 0, sizeof *v);
    const uint8_t * p = sml_ptr(m, m->hdr->tok_off);
    if (m->hdr->tok_n < 16) {
        fprintf(stderr, "sml: model has no binary vocab\n");
        return 0;
    }
    const uint32_t * u = (const uint32_t *) p;
    v->n_tokens = u[0];
    v->n_merges = u[1];
    v->bos = u[2];
    v->eos = u[3];
    v->offsets = u + 4;
    v->bytes   = (const uint8_t *)(v->offsets + v->n_tokens + 1);
    v->merges  = (const uint32_t *)(v->bytes + v->offsets[v->n_tokens]);

    v->tok_mask = next_pow2(v->n_tokens * 2) - 1;
    v->tok_slots = malloc((size_t)(v->tok_mask + 1) * sizeof(uint32_t));
    if (!v->tok_slots) return 0;
    memset(v->tok_slots, 0xff, (size_t)(v->tok_mask + 1) * sizeof(uint32_t));
    for (uint32_t id = 0; id < v->n_tokens; id++) {
        uint32_t l;
        const uint8_t * s = tok_ptr(v, id, &l);
        uint32_t i = fnv(s, l) & v->tok_mask;
        while (v->tok_slots[i] != UINT32_MAX) i = (i + 1) & v->tok_mask;
        v->tok_slots[i] = id;
    }

    // Collect specials: "<|...|>" is the convention every ChatML-family vocab
    // uses, and they sit at the end of the table.
    v->n_special = 0;
    v->spec_ids = malloc((size_t) 1024 * sizeof(uint32_t));
    if (v->spec_ids) {
        for (uint32_t id = 0; id < v->n_tokens && v->n_special < 1024; id++) {
            uint32_t l;
            const uint8_t * s = tok_ptr(v, id, &l);
            if (l >= 4 && s[0] == '<' && s[1] == '|' && s[l - 2] == '|' && s[l - 1] == '>')
                v->spec_ids[v->n_special++] = id;
        }
        // Qwen's tool and thinking markers are single tokens without the "<|"
        // convention. Split into "<", "tool", "_call", ">" the model reads its own
        // tool history in a spelling it never saw in training. Only exact
        // single-token matches are taken, so a vocab without them is unchanged.
        static const char * const MARKS[] = {
            "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
            "<think>", "</think>", NULL
        };
        for (int k = 0; MARKS[k] && v->spec_ids; k++) {
            const uint32_t ml = (uint32_t) strlen(MARKS[k]);
            for (uint32_t id = 0; id < v->n_tokens && v->n_special < 1024; id++) {
                uint32_t l;
                const uint8_t * s = tok_ptr(v, id, &l);
                if (l == ml && memcmp(s, MARKS[k], ml) == 0) { v->spec_ids[v->n_special++] = id; break; }
            }
        }
    }

    // Detect the byte-encoding family from the vocabulary itself: SPM spells
    // raw bytes as "<0xNN>" tokens, BPE maps them onto printable code points.
    for (int b = 0; b < 256; b++) v->byte_tok[b] = UINT32_MAX;
    for (int b = 0; b < 256; b++) {
        char lit[8];
        snprintf(lit, sizeof lit, "<0x%02X>", b);
        v->byte_tok[b] = tok_lookup(v, (const uint8_t *) lit, 6);
    }
    v->spm = (v->byte_tok[0] != UINT32_MAX);

    // The Unigram trailer, when the file has one (see sml-vocab.h)
    {
        const uint8_t * end = p + m->hdr->tok_n;
        const uint8_t * t = (const uint8_t *)(v->merges + 2 * (size_t) v->n_merges);
        const uint8_t * after = t;           // where a following section (PRET) would start
        if (t + 32 <= end && memcmp(t, "UNIG", 4) == 0) {
            const uint32_t * h = (const uint32_t *)(t + 4);
            v->unigram = 1;
            v->unk = h[1];
            v->uflags = h[2];
            memcpy(&v->unk_score, h + 3, 4);
            memcpy(&v->max_score, h + 4, 4);
            const uint32_t n_norm = h[5], norm_bytes = h[6];
            v->scores = (const float *)(t + 32);
            v->kinds  = (const uint8_t *)(v->scores + v->n_tokens);
            const uint8_t * rec = v->kinds + ((v->n_tokens + 3) & ~3u);
            if (rec + norm_bytes > end) { fprintf(stderr, "sml: unigram trailer is cut short\n"); return 0; }
            after = rec + norm_bytes;
            v->spm = 1;                       // pieces spell a space as U+2581, bytes as <0xNN>
            for (uint32_t id = 0; id < v->n_tokens; id++) {
                uint32_t l;
                tok_ptr(v, id, &l);
                if (v->kinds[id] != SML_PIECE_SKIP && l > v->max_tok) v->max_tok = l;
                if (v->kinds[id] == SML_PIECE_ADDED && l && v->spec_ids && v->n_special < 1024) {
                    uint32_t k = 0;
                    while (k < v->n_special && v->spec_ids[k] != id) k++;
                    if (k == v->n_special) v->spec_ids[v->n_special++] = id;
                }
            }
            v->norm_mask = next_pow2(n_norm * 2 + 8) - 1;
            v->norm_rec = calloc((size_t) v->norm_mask + 1, sizeof *v->norm_rec);
            if (!v->norm_rec) return 0;
            for (uint32_t k = 0; k < n_norm; k++) {
                const uint8_t sl = rec[0];
                if (sl > v->norm_max) v->norm_max = sl;
                uint32_t i = fnv(rec + 2, sl) & v->norm_mask;
                while (v->norm_rec[i]) i = (i + 1) & v->norm_mask;
                v->norm_rec[i] = rec;
                rec += 2 + rec[0] + rec[1];
            }
        }
        // a BPE tokenizer's own pipeline (sml_pretok.py): normalizer, splits, merge options
        if (after + 8 <= end && memcmp(after, "PRET", 4) == 0 && !pret_init(v, after, end)) {
            sml_vocab_free(v);
            return 0;
        }
        for (uint32_t k = 0; k < v->n_special; k++) {
            uint32_t l;
            const uint8_t * s0 = tok_ptr(v, v->spec_ids[k], &l);
            if (l) v->first[s0[0] >> 3] |= (uint8_t)(1u << (s0[0] & 7));
        }
    }

    v->mrg_mask = next_pow2(v->n_merges * 2 + 8) - 1;
    v->mrg_keys = malloc((size_t)(v->mrg_mask + 1) * sizeof(uint64_t));
    v->mrg_rank = malloc((size_t)(v->mrg_mask + 1) * sizeof(uint32_t));
    if (!v->mrg_keys || !v->mrg_rank) { sml_vocab_free(v); return 0; }
    memset(v->mrg_rank, 0xff, (size_t)(v->mrg_mask + 1) * sizeof(uint32_t));
    for (uint32_t r = 0; r < v->n_merges; r++) {
        const uint64_t key = ((uint64_t) v->merges[2 * r] << 32) | v->merges[2 * r + 1];
        uint32_t i = mix64(key) & v->mrg_mask;
        while (v->mrg_rank[i] != UINT32_MAX) i = (i + 1) & v->mrg_mask;
        v->mrg_keys[i] = key;
        v->mrg_rank[i] = r;
    }
    return 1;
}

void sml_vocab_free(sml_vocab * v) {
    pret_free(v);
    free(v->tok_slots); free(v->mrg_keys); free(v->mrg_rank); free(v->spec_ids); free(v->norm_rec);
    memset(v, 0, sizeof *v);
}

// ------------------------------------------------------------------ encoding

// GPT-2 byte<->symbol map: every byte gets a printable code point so the merge
// table can be plain text. Bytes that are already printable map to themselves;
// the rest are pushed above 0x100.
// SPM: space is U+2581. Everything else is its own UTF-8 bytes, and any byte
// with no token falls back to "<0xNN>" (handled by the caller, which needs the
// vocabulary to know).
static void byte_to_spm(int b, char * out, int * n) {
    if (b == ' ') { out[0] = (char) 0xE2; out[1] = (char) 0x96; out[2] = (char) 0x81; *n = 3; return; }
    out[0] = (char) b;
    *n = 1;
}

static void byte_to_utf8(int b, char * out, int * n) {
    int cp;
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        cp = b;
    } else {
        // the k-th "unprintable" byte maps to 0x100 + k
        int k = 0;
        for (int i = 0; i < b; i++) {
            if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255)))
                k++;
        }
        cp = 256 + k;
    }
    if (cp < 0x80) { out[0] = (char) cp; *n = 1; }
    else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        *n = 2;
    } else {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        *n = 3;
    }
}

#define MAX_SYM 4096

// Merge one pre-token's symbols by repeatedly applying the lowest-ranked pair.
static int merge_word(const sml_vocab * v, uint32_t * sym, int n,
                      int32_t * out, int max, int * written) {
    while (n > 1) {
        uint32_t best = UINT32_MAX;
        int bi = -1;
        for (int i = 0; i + 1 < n; i++) {
            const uint32_t r = mrg_lookup(v, sym[i], sym[i + 1]);
            if (r < best) { best = r; bi = i; }
        }
        if (bi < 0) break;

        // the merged pair must itself be a token
        uint32_t la, lb;
        const uint8_t * a = tok_ptr(v, sym[bi], &la);
        const uint8_t * b = tok_ptr(v, sym[bi + 1], &lb);
        uint8_t buf[512];
        if (la + lb > sizeof buf) break;
        memcpy(buf, a, la);
        memcpy(buf + la, b, lb);
        const uint32_t id = tok_lookup(v, buf, la + lb);
        if (id == UINT32_MAX) break;

        sym[bi] = id;
        memmove(sym + bi + 1, sym + bi + 2, (size_t)(n - bi - 2) * sizeof(uint32_t));
        n--;
    }
    if (*written + n > max) return -1;
    for (int i = 0; i < n; i++) out[(*written)++] = (int32_t) sym[i];
    return 0;
}

// The special token starting at text[i], longest first; UINT32_MAX if none.
static uint32_t special_at(const sml_vocab * v, const char * text, size_t i, size_t len, uint32_t * blen) {
    uint32_t best = UINT32_MAX;
    *blen = 0;
    const uint8_t c0 = (uint8_t) text[i];
    if ((v->unigram || v->pret) ? !(v->first[c0 >> 3] & (1u << (c0 & 7))) : (text[i] != '<' || i + 1 >= len))
        return best;
    for (uint32_t k = 0; k < v->n_special; k++) {
        uint32_t l;
        const uint8_t * s = tok_ptr(v, v->spec_ids[k], &l);
        if (l > *blen && i + l <= len && memcmp(text + i, s, l) == 0) {
            best = v->spec_ids[k];
            *blen = l;
        }
    }
    return best;
}

// ------------------------------------------------------------------ unigram

typedef struct { uint8_t * p; size_t n, cap; } ubuf;
static int ub_put(ubuf * b, const void * s, size_t n) {
    if (b->n + n > b->cap) {
        size_t c = b->cap ? b->cap * 2 : 256;
        while (c < b->n + n) c *= 2;
        uint8_t * q = realloc(b->p, c);
        if (!q) return 0;
        b->p = q; b->cap = c;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    return 1;
}
static const uint8_t SP[3] = { 0xE2, 0x96, 0x81 };     // U+2581
static int is_sp(const uint8_t * s, size_t at, size_t n) {
    return at + 3 <= n && s[at] == SP[0] && s[at + 1] == SP[1] && s[at + 2] == SP[2];
}
static size_t u8len(uint8_t c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }

// The best-scoring cut of one piece, as sml_unigram.Unigram._viterbi: starts in order,
// ends shortest first, a strictly better score replaces; a character no piece starts
// with is one unknown, a run of unknowns is one, spelled as bytes with byte fallback.
static int uni_cut(const sml_vocab * v, const uint8_t * b, size_t n, int32_t * out, int max, int * w) {
    double  * best = malloc((n + 1) * sizeof *best);
    int32_t * from = malloc((n + 1) * sizeof *from);
    int32_t * tid  = malloc((n + 1) * sizeof *tid);
    int ok = best && from && tid;
    if (ok) {
        // SentencePiece keeps path scores as float and adds a piece in double (its
        // EncodeOptimized): ties then fall by rounding, which this reproduces
        const int spm = (v->uflags & SML_UNI_SPM_MATH) != 0;
        for (size_t k = 0; k <= n; k++) from[k] = -2;
        best[0] = 0; from[0] = -1;
        for (size_t i = 0; i < n; i++) {
            if (from[i] == -2 || (b[i] & 0xC0) == 0x80) continue;
            const double base = best[i];
            int single = 0;
            uint32_t h = 2166136261u;
            size_t j = i, chars = 0;
            while (j < n && j - i < v->max_tok) {
                const size_t e = j + u8len(b[j]) > n ? n : j + u8len(b[j]);
                if (e - i > v->max_tok) break;
                for (size_t k = j; k < e; k++) { h ^= b[k]; h *= 16777619u; }
                j = e; chars++;
                const uint32_t id = tok_lookup_h(v, b + i, (uint32_t)(j - i), h);
                if (id == UINT32_MAX || v->kinds[id] == SML_PIECE_SKIP) continue;
                const double sc = v->kinds[id] == SML_PIECE_USER ? (double) chars * v->max_score - 0.1 : v->scores[id];
                if (chars == 1) single = 1;
                if (from[j] == -2 || base + sc > best[j]) { best[j] = spm ? (double)(float)(base + sc) : base + sc; from[j] = (int32_t) i; tid[j] = (int32_t) id; }
            }
            if (!single) {
                const size_t e = i + u8len(b[i]) > n ? n : i + u8len(b[i]);
                const double cand = spm ? (double)(v->unk_score + (float) base) : base + v->unk_score;
                if (from[e] == -2 || cand > best[e]) { best[e] = cand; from[e] = (int32_t) i; tid[e] = -1; }
            }
        }
        // walk back from the end, then emit forwards
        size_t m = 0;
        for (size_t j = n; j; j = (size_t) from[j]) m++;
        int32_t * ends = malloc((m + 1) * sizeof *ends);
        ok = ends != NULL;
        if (ok) {
            size_t k = m;
            for (size_t j = n; j; j = (size_t) from[j]) ends[--k] = (int32_t) j;
            for (size_t q = 0; q < m && ok; q++) {
                size_t e = (size_t) ends[q];
                const size_t s0 = (size_t) from[e];
                if (tid[e] >= 0) {
                    if (*w >= max) ok = 0; else out[(*w)++] = tid[e];
                    continue;
                }
                while (q + 1 < m && tid[ends[q + 1]] < 0) e = (size_t) ends[++q];
                int bytes = (v->uflags & SML_UNI_BYTE_FALLBACK) != 0;
                for (size_t c = s0; c < e && bytes; c++) if (v->byte_tok[b[c]] == UINT32_MAX) bytes = 0;
                if (bytes) {
                    for (size_t c = s0; c < e && ok; c++) { if (*w >= max) ok = 0; else out[(*w)++] = (int32_t) v->byte_tok[b[c]]; }
                } else if (*w >= max) ok = 0;
                else out[(*w)++] = (int32_t) v->unk;
            }
            free(ends);
        }
    }
    free(best); free(from); free(tid);
    return ok ? 0 : -1;
}

// One stretch of text between specials: normalized, prefixed, split, cut.
static int uni_segment(const sml_vocab * v, const uint8_t * s, size_t n, int32_t * out, int max, int * w) {
    ubuf a = { 0 }, b = { 0 };
    int ok = 1;
    for (size_t i = 0; i < n && ok; ) {                  // the charsmap, longest source first
        size_t took = 0;
        for (uint32_t L = v->norm_max; L && !took; L--) {
            if (i + L > n) continue;
            uint32_t k = fnv(s + i, L) & v->norm_mask;
            for (const uint8_t * r; (r = v->norm_rec[k]); k = (k + 1) & v->norm_mask)
                if (r[0] == L && memcmp(r + 2, s + i, L) == 0) { ok = ub_put(&a, r + 2 + L, r[1]); took = L; break; }
        }
        if (!took) { ok = ub_put(&a, s + i, 1); took = 1; }
        i += took;
    }
    // whitespace, then spaces as U+2581, then the prefix
    size_t lo = 0, hi = a.n;
    if (v->uflags & SML_UNI_STRIP) {
        while (lo < hi && a.p[lo] == ' ') lo++;
        while (hi > lo && a.p[hi - 1] == ' ') hi--;
    }
    const int collapse = (v->uflags & (SML_UNI_STRIP | SML_UNI_COLLAPSE)) != 0;
    const int lead_sp = lo < hi && a.p[lo] == ' ';
    if (ok && ((v->uflags & SML_UNI_PREFIX_ALWAYS) || ((v->uflags & SML_UNI_PREFIX_MISSING) && !lead_sp))) ok = ub_put(&b, SP, 3);
    for (size_t i = lo; i < hi && ok; i++) {
        if (a.p[i] == ' ') {
            if (collapse && i > lo && a.p[i - 1] == ' ') continue;
            ok = ub_put(&b, SP, 3);
        } else ok = ub_put(&b, a.p + i, 1);
    }
    // pieces: newline runs alone, then words cut before each U+2581
    for (size_t i = 0; i < b.n && ok; ) {
        size_t j = i;
        if (v->uflags & SML_UNI_SPLIT_NEWLINES) {
            const int nl = b.p[i] == '\n';
            while (j < b.n && (b.p[j] == '\n') == nl) j++;
        } else j = b.n;
        for (size_t p0 = i; p0 < j && ok; ) {
            size_t q = j;
            if (v->uflags & SML_UNI_SPLIT_WORDS) {
                q = p0 + (is_sp(b.p, p0, j) ? 3 : u8len(b.p[p0]));
                while (q < j && !is_sp(b.p, q, j)) q += u8len(b.p[q]);
                if (q > j) q = j;
            }
            ok = uni_cut(v, b.p + p0, q - p0, out, max, w) == 0;
            p0 = q;
        }
        i = j;
    }
    free(a.p); free(b.p);
    return ok ? 0 : -1;
}

static int uni_encode(const sml_vocab * v, const char * text, int32_t * out, int max) {
    int written = 0;
    const size_t len = strlen(text);
    size_t i = 0, seg = 0;
    while (i < len) {
        uint32_t blen;
        const uint32_t sp = special_at(v, text, i, len, &blen);
        if (sp == UINT32_MAX) { i++; continue; }
        if (i > seg && uni_segment(v, (const uint8_t *) text + seg, i - seg, out, max, &written) < 0) return -1;
        if (written >= max) return -1;
        out[written++] = (int32_t) sp;
        i += blen;
        seg = i;
    }
    if (len > seg && uni_segment(v, (const uint8_t *) text + seg, len - seg, out, max, &written) < 0) return -1;
    return written;
}

// ------------------------------------------------------------------ PRET
//
// A BPE tokenizer's own pipeline (sml_pretok.py writes it from tokenizer.json): the
// normalizer, then pre-tokenizer splits, then the merges per piece. The splits are
// regexes in tokenizers' (Oniguruma's) syntax over code points; the Unicode classes
// they name (\p{L}, \p{N} ...) arrive as code point ranges in the file, so the engine
// carries no Unicode tables of its own. A small backtracking matcher covers what these
// patterns use: alternation, (?: ) (?i: ) groups, classes with ranges and \p{..},
// \s \d \w, quantifiers ? * + {m,n} (greedy, lazy), and (?! ) (?= ) lookaheads.

static uint32_t rd32(const uint8_t ** q) { uint32_t x; memcpy(&x, *q, 4); *q += 4; return x; }

static uint32_t dec_cp(const uint8_t * s, size_t n, size_t * len) {
    const uint8_t c = s[0];
    if (c >= 0xC0 && c < 0xE0 && n >= 2) { *len = 2; return ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F); }
    if (c >= 0xE0 && c < 0xF0 && n >= 3) {
        *len = 3; return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if (c >= 0xF0 && n >= 4) {
        *len = 4;
        return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) | ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    *len = 1; return c;
}

typedef struct { const uint32_t * r; uint32_t n; } crange;     // (lo, hi) pairs

static crange cat_get(const sml_vocab * v, const char * name) {
    crange c = { NULL, 0 };
    const uint8_t * q = v->pcats;
    for (uint32_t k = 0; q && k < v->n_cats; k++) {
        char nm[5] = { 0 };
        memcpy(nm, q, 4);
        q += 4;
        const uint32_t n = rd32(&q);
        if (!strcmp(nm, name)) { c.r = (const uint32_t *) q; c.n = n; return c; }
        q += (size_t) n * 8;
    }
    return c;
}

static int in_cat(crange c, uint32_t cp) {
    uint32_t lo = 0, hi = c.n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        uint32_t a, b;
        memcpy(&a, c.r + 2 * mid, 4); memcpy(&b, c.r + 2 * mid + 1, 4);
        if (cp < a) hi = mid; else if (cp > b) lo = mid + 1; else return 1;
    }
    return 0;
}

// ---- the matcher
enum { RX_CHAR, RX_ANY, RX_SET, RX_GROUP, RX_NLOOK, RX_PLOOK, RX_BOL, RX_EOL };
enum { RI_RANGE, RI_CAT, RI_SPACE, RI_DIGIT, RI_WORD };
typedef struct { uint8_t kind, neg; uint32_t lo, hi; crange cat; } rx_item;
struct rx_alt;
typedef struct {
    int kind, ci, min, max, lazy, neg, n_items, n_alts;
    uint32_t cp;
    rx_item * items;
    struct rx_alt * alts;
} rx_node;
typedef struct rx_alt { rx_node * nodes; int n; } rx_alt;
typedef struct { rx_alt * alts; int n_alts; crange sp, L, N, Nd; void ** mem; int n_mem, cap_mem; } rx;

static void * rx_new(rx * R, size_t bytes) {
    if (R->n_mem == R->cap_mem) {
        R->cap_mem = R->cap_mem ? R->cap_mem * 2 : 64;
        R->mem = realloc(R->mem, (size_t) R->cap_mem * sizeof *R->mem);
    }
    void * p = calloc(1, bytes ? bytes : 1);
    R->mem[R->n_mem++] = p;
    return p;
}
static void rx_free(rx * R) {
    for (int i = 1; i < R->n_mem; i++) free(R->mem[i]);       // slot 0 is the vocabulary
    free(R->mem);
    memset(R, 0, sizeof *R);
}

typedef struct { const uint32_t * p; size_t n, i; rx * R; int err; } rxp;

static rx_alt * rx_alts(rxp * P, int ci, int * n_alts);

static int rx_hex(rxp * P, int digits, uint32_t * out) {
    uint32_t v = 0;
    for (int k = 0; k < digits; k++) {
        if (P->i >= P->n) return 0;
        const uint32_t c = P->p[P->i++];
        v = v * 16 + (c >= '0' && c <= '9' ? c - '0' : (c | 32) >= 'a' && (c | 32) <= 'f' ? (c | 32) - 'a' + 10 : 99);
        if (v > 0x10FFFF) return 0;
    }
    *out = v;
    return 1;
}

// an escape after the backslash: a class item (returns 1, fills *it) or a literal (returns 0, *cp)
static int rx_escape(rxp * P, rx_item * it, uint32_t * cp) {
    if (P->i >= P->n) { P->err = 1; return 0; }
    const uint32_t c = P->p[P->i++];
    memset(it, 0, sizeof *it);
    switch (c) {
    case 's': it->kind = RI_SPACE; return 1;
    case 'S': it->kind = RI_SPACE; it->neg = 1; return 1;
    case 'd': it->kind = RI_DIGIT; return 1;
    case 'D': it->kind = RI_DIGIT; it->neg = 1; return 1;
    case 'w': it->kind = RI_WORD; return 1;
    case 'W': it->kind = RI_WORD; it->neg = 1; return 1;
    case 'p': case 'P': {
        char name[8] = { 0 };
        int k = 0;
        if (P->i < P->n && P->p[P->i] == '{') {
            P->i++;
            while (P->i < P->n && P->p[P->i] != '}' && k < 7) name[k++] = (char) P->p[P->i++];
            if (P->i < P->n && P->p[P->i] == '}') P->i++; else P->err = 1;
        } else if (P->i < P->n) name[k++] = (char) P->p[P->i++];
        it->kind = RI_CAT; it->neg = c == 'P';
        it->cat = cat_get((const sml_vocab *) P->R->mem[0], name);
        if (!it->cat.r) P->err = 1;          // a class the file did not carry
        return 1;
    }
    case 'r': *cp = '\r'; return 0;
    case 'n': *cp = '\n'; return 0;
    case 't': *cp = '\t'; return 0;
    case 'f': *cp = '\f'; return 0;
    case 'v': *cp = '\v'; return 0;
    case '0': *cp = 0; return 0;
    case 'x': if (!rx_hex(P, 2, cp)) P->err = 1; return 0;
    case 'u': if (!rx_hex(P, 4, cp)) P->err = 1; return 0;
    default: *cp = c; return 0;
    }
}

static rx_node rx_set(rxp * P) {
    rx_node nd = { 0 };
    nd.kind = RX_SET; nd.min = nd.max = 1;
    if (P->i < P->n && P->p[P->i] == '^') { nd.neg = 1; P->i++; }
    rx_item tmp[96];
    int n = 0, firstc = 1;
    while (P->i < P->n && (P->p[P->i] != ']' || firstc) && n < 96) {
        firstc = 0;
        uint32_t lo = P->p[P->i++];
        rx_item it;
        if (lo == '[') { P->err = 1; break; }              // nested classes are not followed
        if (lo == '\\') {
            if (rx_escape(P, &it, &lo)) { tmp[n++] = it; continue; }
        }
        uint32_t hi = lo;
        if (P->i + 1 < P->n && P->p[P->i] == '-' && P->p[P->i + 1] != ']') {
            P->i++;
            hi = P->p[P->i++];
            if (hi == '\\' && rx_escape(P, &it, &hi)) { P->err = 1; break; }
        }
        memset(&it, 0, sizeof it);
        it.kind = RI_RANGE; it.lo = lo; it.hi = hi;
        tmp[n++] = it;
    }
    if (P->i < P->n && P->p[P->i] == ']') P->i++; else P->err = 1;
    nd.items = rx_new(P->R, sizeof(rx_item) * (size_t) n);
    memcpy(nd.items, tmp, sizeof(rx_item) * (size_t) n);
    nd.n_items = n;
    return nd;
}

static int rx_num(rxp * P, int * v) {
    int d = 0;
    *v = 0;
    while (P->i < P->n && P->p[P->i] >= '0' && P->p[P->i] <= '9') { *v = *v * 10 + (int)(P->p[P->i++] - '0'); d++; }
    return d;
}

static rx_alt rx_seq(rxp * P, int ci) {
    rx_node tmp[128];
    int n = 0;
    while (P->i < P->n && P->p[P->i] != '|' && P->p[P->i] != ')' && n < 128 && !P->err) {
        const uint32_t c = P->p[P->i++];
        rx_node nd = { 0 };
        nd.min = nd.max = 1; nd.ci = ci;
        if (c == '(') {
            int ci2 = ci;
            nd.kind = RX_GROUP;
            if (P->i < P->n && P->p[P->i] == '?') {
                P->i++;
                const uint32_t f = P->i < P->n ? P->p[P->i++] : 0;
                if (f == ':') {}
                else if (f == '!') nd.kind = RX_NLOOK;
                else if (f == '=') nd.kind = RX_PLOOK;
                else if (f == 'i' && P->i < P->n && P->p[P->i] == ':') { P->i++; ci2 = 1; }
                else { P->err = 1; break; }
            }
            nd.alts = rx_alts(P, ci2, &nd.n_alts);
            if (P->i < P->n && P->p[P->i] == ')') P->i++; else { P->err = 1; break; }
        } else if (c == '[') nd = rx_set(P), nd.ci = ci;
        else if (c == '.') nd.kind = RX_ANY;
        else if (c == '^') nd.kind = RX_BOL;
        else if (c == '$') nd.kind = RX_EOL;
        else if (c == '\\') {
            rx_item it;
            uint32_t cp = 0;
            if (rx_escape(P, &it, &cp)) {
                nd.kind = RX_SET; nd.items = rx_new(P->R, sizeof it); nd.items[0] = it; nd.n_items = 1;
            } else { nd.kind = RX_CHAR; nd.cp = cp; }
        } else { nd.kind = RX_CHAR; nd.cp = c; }
        // the quantifier, then a lazy '?' (a possessive '+' is read as greedy)
        int quant = 0;
        if (P->i < P->n) {
            const uint32_t q = P->p[P->i];
            if (q == '*') { nd.min = 0; nd.max = INT32_MAX; P->i++; quant = 1; }
            else if (q == '+') { nd.min = 1; nd.max = INT32_MAX; P->i++; quant = 1; }
            else if (q == '?') { nd.min = 0; nd.max = 1; P->i++; quant = 1; }
            else if (q == '{') {
                const size_t save = P->i++;
                int a, b;
                if (rx_num(P, &a)) {
                    b = a;
                    if (P->i < P->n && P->p[P->i] == ',') { P->i++; if (!rx_num(P, &b)) b = INT32_MAX; }
                    if (P->i < P->n && P->p[P->i] == '}') { P->i++; nd.min = a; nd.max = b; quant = 1; }
                    else P->i = save;
                } else P->i = save;
                if (P->i == save) { nd.kind = RX_CHAR; nd.cp = '{'; P->i++; }
            }
            if (quant && P->i < P->n && (P->p[P->i] == '?' || P->p[P->i] == '+')) {
                if (P->p[P->i] == '?') nd.lazy = 1;
                P->i++;
            }
        }
        tmp[n++] = nd;
    }
    rx_alt a;
    a.n = n;
    a.nodes = rx_new(P->R, sizeof(rx_node) * (size_t) n);
    memcpy(a.nodes, tmp, sizeof(rx_node) * (size_t) n);
    return a;
}

static rx_alt * rx_alts(rxp * P, int ci, int * n_alts) {
    rx_alt tmp[64];
    int n = 0;
    for (;;) {
        if (n == 64) { P->err = 1; break; }
        tmp[n++] = rx_seq(P, ci);
        if (P->i < P->n && P->p[P->i] == '|') { P->i++; continue; }
        break;
    }
    rx_alt * a = rx_new(P->R, sizeof(rx_alt) * (size_t) n);
    memcpy(a, tmp, sizeof(rx_alt) * (size_t) n);
    *n_alts = n;
    return a;
}

static int rx_compile(rx * R, const sml_vocab * v, const uint8_t * pat, size_t len) {
    memset(R, 0, sizeof *R);
    R->cap_mem = 64;
    R->mem = malloc((size_t) R->cap_mem * sizeof *R->mem);
    if (!R->mem) return 0;
    R->mem[0] = (void *) v;                 // slot 0: the vocabulary, for \p{..} lookups (not freed)
    R->n_mem = 1;
    uint32_t * cps = malloc((len + 1) * sizeof *cps);
    size_t n = 0;
    for (size_t i = 0; i < len; ) { size_t l; cps[n++] = dec_cp(pat + i, len - i, &l); i += l; }
    rxp P = { cps, n, 0, R, 0 };
    R->alts = rx_alts(&P, 0, &R->n_alts);
    if (P.i != P.n) P.err = 1;
    free(cps);
    R->sp = cat_get(v, "s"); R->L = cat_get(v, "L"); R->N = cat_get(v, "N"); R->Nd = cat_get(v, "Nd");
    return !P.err;
}

typedef struct K {
    int kind;                    // 0: continue a sequence; 1: one more turn of a repeated group
    const rx_alt * seq;
    int idx, count;
    const rx_node * node;
    size_t start;
    const struct K * next;
} K;
typedef struct { const rx * R; const uint32_t * s; size_t n, end; long steps; } rxm;

static uint32_t lc(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

static int rx_one(const rxm * M, const rx_node * nd, uint32_t cp) {
    if (nd->kind == RX_CHAR) return nd->ci ? lc(cp) == lc(nd->cp) : cp == nd->cp;
    if (nd->kind == RX_ANY) return cp != '\n';
    int hit = 0;
    for (int k = 0; k < nd->n_items && !hit; k++) {
        const rx_item * it = &nd->items[k];
        int h;
        switch (it->kind) {
        case RI_RANGE: h = (cp >= it->lo && cp <= it->hi) ||
                           (nd->ci && lc(cp) >= lc(it->lo) && lc(cp) <= lc(it->hi)); break;
        case RI_CAT: h = in_cat(it->cat, cp); break;
        case RI_SPACE: h = M->R->sp.r ? in_cat(M->R->sp, cp) : (cp == ' ' || (cp >= 9 && cp <= 13)); break;
        case RI_DIGIT: h = (cp >= '0' && cp <= '9') || (M->R->Nd.r && in_cat(M->R->Nd, cp)); break;
        default: h = cp == '_' || (cp >= '0' && cp <= '9') || ((cp | 32) >= 'a' && (cp | 32) <= 'z') ||
                     (M->R->L.r && in_cat(M->R->L, cp)) || (M->R->N.r && in_cat(M->R->N, cp)); break;
        }
        hit = it->neg ? !h : h;
    }
    return nd->neg ? !hit : hit;
}

static int rx_k(rxm * M, const K * k, size_t pos);

static int rx_rep(rxm * M, const rx_node * nd, int count, size_t pos, const K * cont) {
    if (!nd->lazy) {
        if (count < nd->max)
            for (int a = 0; a < nd->n_alts; a++) {
                const K rk = { 1, NULL, 0, count + 1, nd, pos, cont };
                const K sk = { 0, &nd->alts[a], 0, 0, NULL, 0, &rk };
                if (rx_k(M, &sk, pos)) return 1;
            }
        return count >= nd->min && rx_k(M, cont, pos);
    }
    if (count >= nd->min && rx_k(M, cont, pos)) return 1;
    if (count < nd->max)
        for (int a = 0; a < nd->n_alts; a++) {
            const K rk = { 1, NULL, 0, count + 1, nd, pos, cont };
            const K sk = { 0, &nd->alts[a], 0, 0, NULL, 0, &rk };
            if (rx_k(M, &sk, pos)) return 1;
        }
    return 0;
}

static int rx_node_at(rxm * M, const rx_node * nd, size_t pos, const K * cont) {
    if (++M->steps > 2000000) return 0;           // a runaway pattern fails instead of hanging
    switch (nd->kind) {
    case RX_CHAR: case RX_ANY: case RX_SET: {
        size_t m = 0;
        while ((long long) m < nd->max && pos + m < M->n && rx_one(M, nd, M->s[pos + m])) m++;
        if ((long long) m < nd->min) return 0;
        if (!nd->lazy) {
            for (size_t c = m + 1; c-- > (size_t) nd->min; ) if (rx_k(M, cont, pos + c)) return 1;
        } else {
            for (size_t c = (size_t) nd->min; c <= m; c++) if (rx_k(M, cont, pos + c)) return 1;
        }
        return 0;
    }
    case RX_BOL: return pos == 0 && rx_k(M, cont, pos);
    case RX_EOL: return pos == M->n && rx_k(M, cont, pos);
    case RX_NLOOK: case RX_PLOOK: {
        const size_t save = M->end;
        int hit = 0;
        for (int a = 0; a < nd->n_alts && !hit; a++) {
            const K sk = { 0, &nd->alts[a], 0, 0, NULL, 0, NULL };
            hit = rx_k(M, &sk, pos);
        }
        M->end = save;
        if (nd->kind == RX_NLOOK ? hit : !hit) return 0;
        return rx_k(M, cont, pos);
    }
    default: return rx_rep(M, nd, 0, pos, cont);
    }
}

static int rx_k(rxm * M, const K * k, size_t pos) {
    if (!k) { M->end = pos; return 1; }
    if (k->kind == 1) {
        // one turn of a repeated group ended here: an empty turn does not loop
        if (pos == k->start) return rx_k(M, k->next, pos);
        return rx_rep(M, k->node, k->count, pos, k->next);
    }
    if (k->idx == k->seq->n) return rx_k(M, k->next, pos);
    K kk = *k;
    kk.idx++;
    return rx_node_at(M, &k->seq->nodes[k->idx], pos, &kk);
}

// the leftmost match at or after `from` in s[0..n): 1 and [*a, *b), or 0
static int rx_find(const rx * R, const uint32_t * s, size_t n, size_t from, size_t * a, size_t * b) {
    rxm M = { R, s, n, 0, 0 };
    for (size_t st = from; st < n; st++) {
        for (int k = 0; k < R->n_alts; k++) {
            const K sk = { 0, &R->alts[k], 0, 0, NULL, 0, NULL };
            M.steps = 0;
            if (rx_k(&M, &sk, st)) {
                if (M.end > st) { *a = st; *b = M.end; return 1; }
                goto next;                       // an empty match: the next position
            }
        }
    next:;
    }
    return 0;
}

// ---- pieces
typedef struct { uint8_t * p; size_t n; int first; } piece;
typedef struct { piece * v; size_t n, cap; } pieces;

static int pc_push(pieces * L, const uint8_t * p, size_t n, int first) {
    if (!n) return 1;
    if (L->n == L->cap) {
        L->cap = L->cap ? L->cap * 2 : 16;
        piece * q = realloc(L->v, L->cap * sizeof *q);
        if (!q) return 0;
        L->v = q;
    }
    uint8_t * b = malloc(n);
    if (!b) return 0;
    memcpy(b, p, n);
    L->v[L->n++] = (piece) { b, n, first };
    return 1;
}
static void pc_free(pieces * L) {
    for (size_t i = 0; i < L->n; i++) free(L->v[i].p);
    free(L->v);
    memset(L, 0, sizeof *L);
}

// one piece cut at its delimiter spans (byte offsets), by tokenizers' behaviours
static int pc_split(pieces * out, const piece * pc, const size_t * sa, const size_t * sb, size_t ns, int beh, int inv) {
    // the parts in order: (start, end, is_delimiter)
    size_t cap = 2 * ns + 2, np = 0, at = 0;
    size_t * ps = malloc(cap * sizeof *ps), * pe = malloc(cap * sizeof *pe);
    int * pd = malloc(cap * sizeof *pd), ok = ps && pe && pd;
    for (size_t k = 0; ok && k < ns; k++) {
        if (sa[k] > at) { ps[np] = at; pe[np] = sa[k]; pd[np++] = inv; }
        ps[np] = sa[k]; pe[np] = sb[k]; pd[np++] = !inv;
        at = sb[k];
    }
    if (ok && at < pc->n) { ps[np] = at; pe[np] = pc->n; pd[np++] = inv; }
    const uint8_t * p = pc->p;
    for (size_t k = 0; ok && k < np; k++) {
        const int first = pc->first && ps[k] == 0;
        if (beh == 0) { if (!pd[k]) ok = pc_push(out, p + ps[k], pe[k] - ps[k], first); }        // removed
        else if (beh == 1) ok = pc_push(out, p + ps[k], pe[k] - ps[k], first);                   // isolated
        else if (beh == 2) {                                                                     // with previous
            if (!pd[k] && k + 1 < np && pd[k + 1]) { ok = pc_push(out, p + ps[k], pe[k + 1] - ps[k], first); k++; }
            else ok = pc_push(out, p + ps[k], pe[k] - ps[k], first);
        } else if (beh == 3) {                                                                   // with next
            if (pd[k] && k + 1 < np && !pd[k + 1]) { ok = pc_push(out, p + ps[k], pe[k + 1] - ps[k], first); k++; }
            else ok = pc_push(out, p + ps[k], pe[k] - ps[k], first);
        } else {                                                                                 // contiguous
            size_t e = k;
            if (pd[k]) while (e + 1 < np && pd[e + 1]) e++;
            ok = pc_push(out, p + ps[k], pe[e] - ps[k], first);
            k = e;
        }
    }
    free(ps); free(pe); free(pd);
    return ok;
}

// the delimiter spans of a regex in a piece, in byte offsets
static int pc_regex_spans(const rx * R, const piece * pc, size_t ** sa, size_t ** sb, size_t * ns) {
    uint32_t * cps = malloc((pc->n + 1) * sizeof *cps);
    size_t * off = malloc((pc->n + 2) * sizeof *off);
    if (!cps || !off) { free(cps); free(off); return 0; }
    size_t n = 0;
    for (size_t i = 0; i < pc->n; ) { size_t l; off[n] = i; cps[n++] = dec_cp(pc->p + i, pc->n - i, &l); i += l; }
    off[n] = pc->n;
    size_t cap = 16, k = 0, from = 0, a, b;
    *sa = malloc(cap * sizeof **sa); *sb = malloc(cap * sizeof **sb);
    while (*sa && *sb && from < n && rx_find(R, cps, n, from, &a, &b)) {
        if (k == cap) { cap *= 2; *sa = realloc(*sa, cap * sizeof **sa); *sb = realloc(*sb, cap * sizeof **sb); if (!*sa || !*sb) break; }
        (*sa)[k] = off[a]; (*sb)[k] = off[b]; k++;
        from = b;
    }
    *ns = k;
    free(cps); free(off);
    return *sa && *sb;
}

static int pc_string_spans(const piece * pc, const uint8_t * d, size_t dn, size_t ** sa, size_t ** sb, size_t * ns) {
    size_t cap = 16, k = 0;
    *sa = malloc(cap * sizeof **sa); *sb = malloc(cap * sizeof **sb);
    for (size_t i = 0; dn && i + dn <= pc->n && *sa && *sb; ) {
        if (memcmp(pc->p + i, d, dn) == 0) {
            if (k == cap) { cap *= 2; *sa = realloc(*sa, cap * sizeof **sa); *sb = realloc(*sb, cap * sizeof **sb); if (!*sa || !*sb) break; }
            (*sa)[k] = i; (*sb)[k] = i + dn; k++;
            i += dn;
        } else i++;
    }
    *ns = k;
    return *sa && *sb;
}

// a byte string with every `a` replaced by `b`
static int bs_replace(ubuf * o, const uint8_t * s, size_t n, const uint8_t * a, size_t an, const uint8_t * b, size_t bn) {
    for (size_t i = 0; i < n; ) {
        if (an && i + an <= n && memcmp(s + i, a, an) == 0) { if (!ub_put(o, b, bn)) return 0; i += an; }
        else { if (!ub_put(o, s + i, 1)) return 0; i++; }
    }
    return 1;
}

typedef struct { uint8_t kind, a, b, c; const uint8_t * s1; uint32_t n1; const uint8_t * s2; uint32_t n2; } pret_op;

static const uint8_t * op_read(const uint8_t * q, pret_op * op) {
    op->kind = q[0]; op->a = q[1]; op->b = q[2]; op->c = q[3];
    q += 4;
    op->n1 = rd32(&q); op->s1 = q; q += op->n1;
    op->n2 = rd32(&q); op->s2 = q; q += op->n2;
    return q;
}

// ---- the merges over one piece (a heap of (rank, left), lowest rank first, leftmost on ties)
typedef struct { uint64_t * h; size_t n, cap; } heap;
static int hp_push(heap * H, uint64_t x) {
    if (H->n == H->cap) { H->cap = H->cap ? H->cap * 2 : 64; uint64_t * q = realloc(H->h, H->cap * 8); if (!q) return 0; H->h = q; }
    size_t i = H->n++;
    H->h[i] = x;
    while (i && H->h[(i - 1) / 2] > H->h[i]) { uint64_t t = H->h[i]; H->h[i] = H->h[(i - 1) / 2]; H->h[(i - 1) / 2] = t; i = (i - 1) / 2; }
    return 1;
}
static uint64_t hp_pop(heap * H) {
    const uint64_t top = H->h[0];
    H->h[0] = H->h[--H->n];
    for (size_t i = 0;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < H->n && H->h[l] < H->h[m]) m = l;
        if (r < H->n && H->h[r] < H->h[m]) m = r;
        if (m == i) break;
        uint64_t t = H->h[i]; H->h[i] = H->h[m]; H->h[m] = t; i = m;
    }
    return top;
}

static int pret_bpe(const sml_vocab * v, const uint8_t * p, size_t n, int32_t * out, int max, int * w) {
    ubuf m = { 0 };
    int ok = 1;
    // the symbols: GPT-2's printable code point per byte, or the piece's own characters
    size_t cnt = 0;
    size_t * so = malloc((n + 1) * sizeof *so), * sl = malloc((n + 1) * sizeof *sl), * src = malloc((n + 1) * sizeof *src),
           * srl = malloc((n + 1) * sizeof *srl);
    if (!so || !sl || !src || !srl) ok = 0;
    for (size_t i = 0; ok && i < n; ) {
        size_t l = 1;
        if (v->pflags & SML_PRET_BYTE_LEVEL) {
            char u[4]; int ul;
            byte_to_utf8(p[i], u, &ul);
            so[cnt] = m.n; sl[cnt] = (size_t) ul; src[cnt] = i; srl[cnt] = 1;
            ok = ub_put(&m, u, (size_t) ul);
        } else {
            l = u8len(p[i]); if (i + l > n) l = n - i;
            so[cnt] = m.n; sl[cnt] = l; src[cnt] = i; srl[cnt] = l;
            ok = ub_put(&m, p + i, l);
        }
        cnt++;
        i += l;
    }
    if (ok && (v->pflags & SML_PRET_IGNORE_MERGES)) {
        const uint32_t id = tok_lookup(v, m.p, (uint32_t) m.n);
        if (id != UINT32_MAX) {
            if (*w >= max) ok = 0; else out[(*w)++] = (int32_t) id;
            free(m.p); free(so); free(sl); free(src); free(srl);
            return ok ? 0 : -1;
        }
    }
    uint32_t * id = malloc((cnt + 1) * sizeof *id);
    int * nx = malloc((cnt + 1) * sizeof *nx), * pv = malloc((cnt + 1) * sizeof *pv);
    uint8_t * dead = calloc(cnt + 1, 1);
    heap H = { 0 };
    if (!id || !nx || !pv || !dead) ok = 0;
    for (size_t i = 0; ok && i < cnt; i++) {
        id[i] = tok_lookup(v, m.p + so[i], (uint32_t) sl[i]);
        nx[i] = i + 1 < cnt ? (int) i + 1 : -1;
        pv[i] = (int) i - 1;
    }
    for (size_t i = 0; ok && i + 1 < cnt; i++) {
        if (id[i] == UINT32_MAX || id[i + 1] == UINT32_MAX) continue;
        const uint32_t r = mrg_lookup(v, id[i], id[i + 1]);
        if (r != UINT32_MAX) ok = hp_push(&H, ((uint64_t) r << 32) | i);
    }
    while (ok && H.n) {
        const uint64_t top = hp_pop(&H);
        const uint32_t r = (uint32_t) (top >> 32);
        const int a = (int) (top & 0xffffffffu), b = nx[a];
        if (dead[a] || b < 0 || id[a] == UINT32_MAX || id[b] == UINT32_MAX || mrg_lookup(v, id[a], id[b]) != r) continue;
        const uint32_t mid = tok_lookup(v, m.p + so[a], (uint32_t) (sl[a] + sl[b]));
        if (mid == UINT32_MAX) continue;
        id[a] = mid; sl[a] += sl[b]; srl[a] += srl[b];
        dead[b] = 1; nx[a] = nx[b];
        if (nx[b] >= 0) pv[nx[b]] = a;
        if (pv[a] >= 0 && id[pv[a]] != UINT32_MAX) {
            const uint32_t r2 = mrg_lookup(v, id[pv[a]], id[a]);
            if (r2 != UINT32_MAX) ok = hp_push(&H, ((uint64_t) r2 << 32) | (uint32_t) pv[a]);
        }
        if (ok && nx[a] >= 0 && id[nx[a]] != UINT32_MAX) {
            const uint32_t r3 = mrg_lookup(v, id[a], id[nx[a]]);
            if (r3 != UINT32_MAX) ok = hp_push(&H, ((uint64_t) r3 << 32) | (uint32_t) a);
        }
    }
    // out, in order; an unknown character as its bytes (byte fallback) or <unk>, a run fused
    int last_unk = 0;
    for (int i = cnt ? 0 : -1; ok && i >= 0; i = nx[i]) {
        if (id[i] != UINT32_MAX) {
            if (*w >= max) ok = 0; else out[(*w)++] = (int32_t) id[i];
            last_unk = 0;
            continue;
        }
        int bytes = (v->pflags & SML_PRET_BYTE_FALLBACK) != 0;
        for (size_t c = 0; c < srl[i] && bytes; c++) if (v->byte_tok[p[src[i] + c]] == UINT32_MAX) bytes = 0;
        if (bytes) {
            for (size_t c = 0; c < srl[i] && ok; c++) { if (*w >= max) ok = 0; else out[(*w)++] = (int32_t) v->byte_tok[p[src[i] + c]]; }
            last_unk = 0;
        } else if (v->punk != UINT32_MAX && !(last_unk && (v->pflags & SML_PRET_FUSE_UNK))) {
            if (*w >= max) ok = 0; else out[(*w)++] = (int32_t) v->punk;
            last_unk = 1;
        }
    }
    free(H.h); free(id); free(nx); free(pv); free(dead);
    free(m.p); free(so); free(sl); free(src); free(srl);
    return ok ? 0 : -1;
}

// GPT-2's printable stand-in for a byte: what later pre-tokenizer steps see after ByteLevel
static uint32_t bl_cp(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    uint32_t k = 0;
    for (int i = 0; i < b; i++) if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174)) k++;
    return 256 + k;
}

// the character at byte j of a piece and its length: UTF-8, or after ByteLevel one byte's stand-in
static uint32_t pc_cp(const piece * pc, size_t j, int bl, size_t * l) {
    if (bl) { *l = 1; return bl_cp(pc->p[j]); }
    const uint32_t c = dec_cp(pc->p + j, pc->n - j, l);
    if (!*l) *l = 1;
    return c;
}

// tokenizers' Digits (char::is_numeric: Nd, Nl, No) and Punctuation (ASCII punctuation or P*)
static int is_num(crange N, uint32_t c) { return (c >= '0' && c <= '9') || (N.r && c > 127 && in_cat(N, c)); }
static int is_punct(crange P, uint32_t c) {
    if (c < 128) return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
    return P.r && in_cat(P, c);
}

// one stretch between added tokens: normalized, cut by the pre-tokenizer, merged piece by piece
static int pret_segment(const sml_vocab * v, const uint8_t * s, size_t n, int first, int32_t * out, int max, int * w) {
    ubuf a = { 0 };
    int ok = ub_put(&a, s, n);
    const uint8_t * q = v->pnorm;
    for (uint32_t k = 0; ok && k < v->n_norm; k++) {
        pret_op op;
        q = op_read(q, &op);
        ubuf b = { 0 };
        if (op.kind == 1) { ok = a.n ? (ub_put(&b, op.s1, op.n1) && ub_put(&b, a.p, a.n)) : 1; if (!a.n) { free(b.p); continue; } }
        else if (op.kind == 2) ok = bs_replace(&b, a.p, a.n, op.s1, op.n1, op.s2, op.n2);
        else if (op.kind == 4) { ok = ub_put(&b, a.p, a.n); for (size_t i = 0; ok && i < b.n; i++) if (b.p[i] >= 'A' && b.p[i] <= 'Z') b.p[i] += 32; }
        else if (op.kind == 5) {
            size_t lo = 0, hi = a.n;
            if (op.a) while (lo < hi && (a.p[lo] == ' ' || (a.p[lo] >= 9 && a.p[lo] <= 13))) lo++;
            if (op.b) while (hi > lo && (a.p[hi - 1] == ' ' || (a.p[hi - 1] >= 9 && a.p[hi - 1] <= 13))) hi--;
            ok = ub_put(&b, a.p + lo, hi - lo);
        } else { free(b.p); continue; }                 // NFC: text arrives composed in practice
        free(a.p);
        a = b;
    }
    pieces L = { 0 };
    if (ok) ok = pc_push(&L, a.p, a.n, first);
    free(a.p);
    q = v->ppre;
    const rx * res = (const rx *) v->pres;
    const crange Nc = cat_get(v, "N"), Pc = cat_get(v, "P");
    int bl = 0;                                          // a ByteLevel step has run
    for (uint32_t k = 0; ok && k < v->n_pre; k++) {
        pret_op op;
        q = op_read(q, &op);
        pieces O = { 0 };
        for (size_t i = 0; ok && i < L.n; i++) {
            piece pc = L.v[i];
            size_t * sa = NULL, * sb = NULL, ns = 0;
            ubuf t = { 0 };
            if (op.kind == 1) {
                ok = pc_regex_spans(&res[k], &pc, &sa, &sb, &ns) && pc_split(&O, &pc, sa, sb, ns, op.a, op.b);
            } else if (op.kind == 2) {
                ok = pc_string_spans(&pc, op.s1, op.n1, &sa, &sb, &ns) && pc_split(&O, &pc, sa, sb, ns, op.a, op.b);
            } else if (op.kind == 3) {                       // byte level: the prefix space, then GPT-2's split
                if (op.a && !(pc.n && pc.p[0] == ' ')) { ok = ub_put(&t, " ", 1) && ub_put(&t, pc.p, pc.n); pc.p = t.p; pc.n = t.n; }
                if (ok && op.b) ok = pc_regex_spans(&res[k], &pc, &sa, &sb, &ns) && pc_split(&O, &pc, sa, sb, ns, 1, 0);
                else if (ok) ok = pc_push(&O, pc.p, pc.n, pc.first);
            } else if (op.kind == 4) {                       // Metaspace
                ok = bs_replace(&t, pc.p, pc.n, (const uint8_t *) " ", 1, op.s1, op.n1);
                if (ok && (op.a == 2 || (op.a == 1 && pc.first)) && !(t.n >= op.n1 && memcmp(t.p, op.s1, op.n1) == 0)) {
                    ubuf u = { 0 };
                    ok = ub_put(&u, op.s1, op.n1) && ub_put(&u, t.p, t.n);
                    free(t.p);
                    t = u;
                }
                piece np = { t.p, t.n, pc.first };
                if (ok && op.b) ok = pc_string_spans(&np, op.s1, op.n1, &sa, &sb, &ns) && pc_split(&O, &np, sa, sb, ns, 3, 0);
                else if (ok) ok = pc_push(&O, t.p, t.n, pc.first);
            } else if (op.kind == 5) {                       // digits, alone or as runs
                size_t cap = 16, n2 = 0;
                sa = malloc(cap * sizeof *sa); sb = malloc(cap * sizeof *sb);
                // a digit is any numeric character, as Rust's char::is_numeric (Nd, Nl, No):
                // tokenizers' Digits cuts "½" and "²" out like "5" (SmolLM2, StarCoder2)
                for (size_t j = 0; sa && sb && j < pc.n; ) {
                    size_t l, l2;
                    if (is_num(Nc, pc_cp(&pc, j, bl, &l))) {
                        size_t e = j + l;
                        if (!op.a) while (e < pc.n && is_num(Nc, pc_cp(&pc, e, bl, &l2))) e += l2;
                        if (n2 == cap) { cap *= 2; sa = realloc(sa, cap * sizeof *sa); sb = realloc(sb, cap * sizeof *sb); if (!sa || !sb) break; }
                        sa[n2] = j; sb[n2] = e; n2++;
                        j = e;
                    } else j += l;
                }
                ns = n2;
                ok = sa && sb && pc_split(&O, &pc, sa, sb, ns, 1, 0);
            } else if (op.kind == 6) {                       // punctuation, each mark a delimiter (Falcon)
                size_t cap = 16, n2 = 0;
                sa = malloc(cap * sizeof *sa); sb = malloc(cap * sizeof *sb);
                for (size_t j = 0; sa && sb && j < pc.n; ) {
                    size_t l;
                    if (is_punct(Pc, pc_cp(&pc, j, bl, &l))) {
                        if (n2 == cap) { cap *= 2; sa = realloc(sa, cap * sizeof *sa); sb = realloc(sb, cap * sizeof *sb); if (!sa || !sb) break; }
                        sa[n2] = j; sb[n2] = j + l; n2++;
                    }
                    j += l;
                }
                ns = n2;
                ok = sa && sb && pc_split(&O, &pc, sa, sb, ns, op.a, 0);
            } else ok = pc_push(&O, pc.p, pc.n, pc.first);
            free(sa); free(sb); free(t.p);
        }
        pc_free(&L);
        L = O;
        if (op.kind == 3) bl = 1;
    }
    for (size_t i = 0; ok && i < L.n; i++) ok = pret_bpe(v, L.v[i].p, L.v[i].n, out, max, w) == 0;
    pc_free(&L);
    return ok ? 0 : -1;
}

static int pret_encode(const sml_vocab * v, const char * text, int32_t * out, int max) {
    int written = 0;
    const size_t len = strlen(text);
    size_t i = 0, seg = 0;
    while (i < len) {
        uint32_t blen;
        const uint32_t sp = special_at(v, text, i, len, &blen);
        if (sp == UINT32_MAX) { i++; continue; }
        if (i > seg && pret_segment(v, (const uint8_t *) text + seg, i - seg, seg == 0, out, max, &written) < 0) return -1;
        if (written >= max) return -1;
        out[written++] = (int32_t) sp;
        i += blen;
        seg = i;
    }
    if (len > seg && pret_segment(v, (const uint8_t *) text + seg, len - seg, seg == 0, out, max, &written) < 0) return -1;
    return written;
}

// the PRET section at q: fills v, compiles the regexes; 0 on a section the engine cannot follow
static int pret_init(sml_vocab * v, const uint8_t * q, const uint8_t * end) {
    if (q + 8 > end) return 0;
    q += 4;
    const uint32_t len = rd32(&q);
    if (q + len > end) return 0;
    const uint32_t ver = rd32(&q);
    if (ver != 1) { fprintf(stderr, "sml: tokenizer pipeline v%u is newer than this engine\n", ver); return 0; }
    v->pflags = rd32(&q);
    v->punk = rd32(&q);
    v->n_norm = rd32(&q);
    v->pnorm = q;
    for (uint32_t k = 0; k < v->n_norm; k++) { pret_op op; q = op_read(q, &op); }
    v->n_pre = rd32(&q);
    v->ppre = q;
    for (uint32_t k = 0; k < v->n_pre; k++) {
        pret_op op;
        q = op_read(q, &op);
        if (op.kind < 1 || op.kind > 6) { fprintf(stderr, "sml: tokenizer step %u is newer than this engine\n", op.kind); return 0; }
    }
    const uint32_t n_added = rd32(&q);
    for (uint32_t k = 0; k < n_added; k++) {
        const uint32_t id = rd32(&q);
        if (id < v->n_tokens && v->spec_ids && v->n_special < 1024) {
            uint32_t j = 0;
            while (j < v->n_special && v->spec_ids[j] != id) j++;
            if (j == v->n_special) v->spec_ids[v->n_special++] = id;
        }
    }
    v->n_cats = rd32(&q);
    v->pcats = q;
    rx * res = calloc(v->n_pre ? v->n_pre : 1, sizeof *res);
    if (!res) return 0;
    v->pres = res;
    const uint8_t * o = v->ppre;
    for (uint32_t k = 0; k < v->n_pre; k++) {
        pret_op op;
        o = op_read(o, &op);
        if ((op.kind == 1 || (op.kind == 3 && op.b)) && !rx_compile(&res[k], v, op.s1, op.n1)) {
            fprintf(stderr, "sml: the tokenizer's split pattern is beyond this engine: %.*s\n", (int) op.n1, op.s1);
            return 0;
        }
    }
    v->pret = 1;
    return 1;
}

static void pret_free(sml_vocab * v) {
    rx * res = (rx *) v->pres;
    for (uint32_t k = 0; res && k < v->n_pre; k++) if (res[k].mem) { res[k].mem[0] = NULL; rx_free(&res[k]); }
    free(res);
    v->pres = NULL;
}

int sml_encode(const sml_vocab * v, const char * text, int32_t * out, int max) {
    if (v->unigram) return uni_encode(v, text, out, max);
    if (v->pret) return pret_encode(v, text, out, max);
    int written = 0;
    const size_t len = strlen(text);
    size_t i = 0;

    while (i < len) {
        // Specials first, matched longest-first as whole units.
        uint32_t blen;
        const uint32_t best = special_at(v, text, i, len, &blen);
        if (best != UINT32_MAX) {
            if (written >= max) return -1;
            out[written++] = (int32_t) best;
            i += blen;
            continue;
        }

        // Pre-tokenize: a chunk starts at a space and runs to just before the
        // next one, so the leading space stays attached to the word -- which is
        // what the byte map turns into the familiar space-prefixed token.
        // It also ends where a special begins: "\n\n</think>" read as one chunk
        // split Qwen's marker into ordinary text.
        size_t j = i + 1;
        while (j < len && text[j] != ' ' && !(text[j] == '<' && j + 1 < len
                                              && (text[j + 1] == '|'
                                                  || special_at(v, text, j, len, &blen) != UINT32_MAX))) j++;

        uint32_t sym[MAX_SYM];
        int n = 0;
        for (size_t k = i; k < j && n < MAX_SYM; k++) {
            const int b = (unsigned char) text[k];
            char u[4];
            int ul;
            if (v->spm) byte_to_spm(b, u, &ul);
            else        byte_to_utf8(b, u, &ul);
            uint32_t id = tok_lookup(v, (const uint8_t *) u, (uint32_t) ul);
            // SPM falls back to the explicit byte token when the character has
            // no entry of its own -- that is what byte fallback means
            if (id == UINT32_MAX && v->spm) id = v->byte_tok[b];
            if (id != UINT32_MAX) sym[n++] = id;
        }
        if (n && merge_word(v, sym, n, out, max, &written) < 0) return -1;
        i = j;
    }
    return written;
}

// ------------------------------------------------------------------ decoding

// Inverse of byte_to_utf8: recover the original byte from its symbol.
static int utf8_to_byte(const uint8_t * s, int n, int * used) {
    int cp;
    if (s[0] < 0x80)             { cp = s[0];                       *used = 1; }
    else if ((s[0] & 0xE0) == 0xC0 && n >= 2) {
        cp = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);                  *used = 2;
    } else if ((s[0] & 0xF0) == 0xE0 && n >= 3) {
        cp = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        *used = 3;
    } else { *used = 1; return -1; }

    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255))
        return cp;
    if (cp >= 256 && cp < 256 + 68) {
        int k = cp - 256, b = 0;
        for (int i = 0; i < 256; i++) {
            if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))) {
                if (b++ == k) return i;
            }
        }
    }
    return -1;
}

int sml_decode(const sml_vocab * v, int32_t id, char * buf, int max) {
    if (id < 0 || (uint32_t) id >= v->n_tokens) return 0;
    uint32_t l;
    const uint8_t * s = tok_ptr(v, (uint32_t) id, &l);
    int w = 0;

    if (v->spm) {
        // "<0xNN>" is one raw byte; U+2581 is a space; everything else is
        // already UTF-8 and passes through untouched.
        if (l == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            char hx[3] = { (char) s[3], (char) s[4], 0 };
            if (max > 0) buf[0] = (char) strtoul(hx, NULL, 16);
            return max > 0 ? 1 : 0;
        }
        for (uint32_t i = 0; i < l && w < max; ) {
            if (i + 2 < l && s[i] == 0xE2 && s[i+1] == 0x96 && s[i+2] == 0x81) {
                buf[w++] = ' ';
                i += 3;
            } else {
                buf[w++] = (char) s[i++];
            }
        }
        return w;
    }

    for (uint32_t i = 0; i < l && w < max; ) {
        int used;
        const int b = utf8_to_byte(s + i, (int)(l - i), &used);
        i += used;
        if (b >= 0) buf[w++] = (char) b;
    }
    return w;
}
