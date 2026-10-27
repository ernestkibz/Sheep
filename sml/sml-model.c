#include "sml-model.h"
#include "sml-kernel.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static int range_inside(size_t size, uint64_t off, uint64_t n) {
    return off <= (uint64_t) size && n <= (uint64_t) size - off;
}

static int payload_inside(size_t size, uint64_t off, uint64_t n) {
    return off >= SML_FILE_HEADER_BYTES && range_inside(size, off, n);
}

static int validate_tensor_ranges(const sml_model * m, const char * path) {
    const int exact = (m->hdr->flags & SML_MODEL_EXACT) != 0;
    for (uint64_t i = 0; i < m->hdr->n_tensors; i++) {
        const sml_rec * r = &m->recs[i];
        const char * issue = NULL;
        if (!r->rows || !r->cols || r->rows > INT_MAX || r->cols > INT_MAX)
            issue = "invalid tensor dimensions";
        else if (r->kind == SML_KIND_F32) {
            const uint64_t count = (uint64_t) r->rows * r->cols;
            if (count > UINT64_MAX / sizeof(float) ||
                r->nbytes != count * sizeof(float) ||
                !payload_inside(m->size, r->off, r->nbytes))
                issue = "invalid f32 tensor payload";
        } else if (r->kind != SML_KIND_LADDER) {
            issue = "unknown tensor kind";
        } else if (exact) {
            issue = "exact-weight file contains quantized tensors";
        } else if (r->layout > SML_LAYOUT_FLAT ||
                   (r->layout == SML_LAYOUT_FLAT && (r->nbits < 2 || r->nbits > 8))) {
            issue = "invalid tensor code layout";
        }
        if (!issue && r->kind == SML_KIND_LADDER) {
            const uint64_t offsets[] = {r->sd_off, r->sq_off, r->perm_off,
                                        r->md_off, r->mq_off};
            const uint64_t lengths[] = {r->sd_n, r->sq_n, r->perm_n,
                                        r->md_n, r->mq_n};
            for (size_t j = 0; j < sizeof offsets / sizeof offsets[0]; j++) {
                if (lengths[j] && !payload_inside(m->size, offsets[j], lengths[j])) {
                    issue = "tensor metadata exceeds file bounds";
                    break;
                }
            }
            for (int j = 0; !issue && j < SML_MAX_LEVELS; j++) {
                if (r->codes_n[j] && !payload_inside(m->size, r->codes_off[j], r->codes_n[j]))
                    issue = "tensor codes exceed file bounds";
            }
        }
        if (!issue && exact && r->kind != SML_KIND_F32)
            issue = "exact-weight file contains quantized tensors";
        if (issue) {
            fprintf(stderr, "sml: %s tensor %llu (%.*s): %s\n", path,
                    (unsigned long long) i, SML_NAME_LEN, r->name, issue);
            return 0;
        }
    }
    return 1;
}

int sml_model_open(sml_model * m, const char * path) {
    memset(m, 0, sizeof *m);

#if defined(_WIN32)
    // SML_LARGE_PAGES=1 loads the file into 2 MB pages instead of mapping it.
    //
    // A 2.5 GB model at 4 KB pages needs 640k page-table entries and the TLB
    // holds a few thousand, so a decode that streams the whole model touches a
    // new page every few hundred bytes and misses the TLB nearly every time.
    // 2 MB pages cut the entry count by 512x.
    //
    // The cost is real and worth stating: this COPIES the file into committed
    // memory, so start-up pays a full read and the model can no longer be
    // shared between processes or evicted under pressure. It is a speed/memory
    // trade, not a free win, which is why it is opt-in.
    // SML_COPY=1 does the same copy into ordinary 4 KB private pages, which
    // needs no privilege: a measurement of mapped-file versus private memory.
    const char * lp_env = getenv("SML_LARGE_PAGES");
    const char * cp_env = getenv("SML_COPY");
    const int want_lp = lp_env && *lp_env == '1';
    if (want_lp || (cp_env && *cp_env == '1')) {
        HANDLE lf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (lf != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER lsz;
            SIZE_T gran = want_lp ? GetLargePageMinimum() : 4096;
            if (gran && GetFileSizeEx(lf, &lsz)) {
                HANDLE tk;
                if (OpenProcessToken(GetCurrentProcess(),
                                     TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tk)) {
                    TOKEN_PRIVILEGES tp;
                    tp.PrivilegeCount = 1;
                    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME,
                                             &tp.Privileges[0].Luid))
                        AdjustTokenPrivileges(tk, FALSE, &tp, sizeof tp, NULL, NULL);
                    CloseHandle(tk);
                }
                const size_t want = (((size_t) lsz.QuadPart + gran - 1) / gran) * gran;
                uint8_t * buf = (uint8_t *) VirtualAlloc(
                    NULL, want, MEM_COMMIT | MEM_RESERVE | (want_lp ? MEM_LARGE_PAGES : 0),
                    PAGE_READWRITE);
                if (buf) {
                    size_t got = 0;
                    int ok = 1;
                    while (got < (size_t) lsz.QuadPart) {
                        DWORD chunk = (DWORD) ((size_t) lsz.QuadPart - got > (1u << 24)
                                               ? (1u << 24) : (size_t) lsz.QuadPart - got);
                        DWORD rd = 0;
                        if (!ReadFile(lf, buf + got, chunk, &rd, NULL) || !rd) { ok = 0; break; }
                        got += rd;
                    }
                    if (ok) {
                        m->base = buf;
                        m->size = (size_t) lsz.QuadPart;
                        m->handle = NULL;
                        m->owns_pages = 1;
                    } else {
                        VirtualFree(buf, 0, MEM_RELEASE);
                    }
                }
            }
            CloseHandle(lf);
        }
        if (!m->base)
            fprintf(stderr, "sml: large pages unavailable, falling back to mmap\n");
    }

    if (!m->base) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "sml: cannot open %s\n", path);
        return 0;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz)) { CloseHandle(f); return 0; }
    HANDLE map = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(f);
    if (!map) { fprintf(stderr, "sml: cannot map %s\n", path); return 0; }
    const uint8_t * base = (const uint8_t *) MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(map); return 0; }
    m->base = base;
    m->size = (size_t) sz.QuadPart;
    m->handle = map;
    }
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "sml: cannot open %s\n", path); return 0; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 0; }
    void * base = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { fprintf(stderr, "sml: cannot map %s\n", path); return 0; }
    m->base = (const uint8_t *) base;
    m->size = (size_t) st.st_size;
    m->handle = NULL;
#endif

    if (m->size < SML_FILE_HEADER_BYTES) {
        fprintf(stderr, "sml: %s is truncated (file is smaller than the SSF header)\n", path);
        sml_model_close(m);
        return 0;
    }
    m->hdr = (const sml_hdr *) m->base;
    if (memcmp(m->hdr->magic, "SSF ", 4) != 0) {
        fprintf(stderr, "sml: %s is not an .ssf file\n", path);
        sml_model_close(m);
        return 0;
    }
    // A newer file has a LARGER tensor record, so reading it with this struct
    // misparses the whole table and crashes somewhere unrelated. Refuse by
    // version instead: the failure should name the cause, not land as a
    // segfault three functions away.
    if (m->hdr->version > SML_MAX_VERSION) {
        fprintf(stderr, "sml: %s is format v%u, this engine reads up to v%d\n",
                path, m->hdr->version, SML_MAX_VERSION);
        sml_model_close(m);
        return 0;
    }
    if (!m->hdr->n_levels || m->hdr->n_levels > SML_MAX_LEVELS || !m->hdr->n_tensors) {
        fprintf(stderr, "sml: %s has invalid level or tensor counts\n", path);
        sml_model_close(m);
        return 0;
    }
    if ((m->hdr->flags & ~(SML_MODEL_HADAMARD | SML_MODEL_SCALE6 |
                            SML_MODEL_MINNEG | SML_MODEL_EXACT)) ||
        ((m->hdr->flags & SML_MODEL_MINNEG) && !(m->hdr->flags & SML_MODEL_SCALE6))) {
        fprintf(stderr, "sml: %s has unsupported model flags\n", path);
        sml_model_close(m);
        return 0;
    }
    const size_t rec_size = m->hdr->version >= 8 ? sizeof(sml_rec) :
                            m->hdr->version >= 6 ? sizeof(sml_rec_v7) : sizeof(sml_rec_v5);
    if (m->hdr->table_off < SML_FILE_HEADER_BYTES || m->hdr->table_off > m->size ||
        m->hdr->n_tensors > ((uint64_t) m->size - m->hdr->table_off) / rec_size ||
        m->hdr->n_tensors > SIZE_MAX / sizeof(sml_rec)) {
        fprintf(stderr, "sml: %s has a tensor table outside file bounds\n", path);
        sml_model_close(m);
        return 0;
    }
    if ((m->hdr->json_n && (m->hdr->json_off < SML_FILE_HEADER_BYTES ||
                            !range_inside(m->size, m->hdr->json_off, m->hdr->json_n))) ||
        (m->hdr->tok_n && (m->hdr->tok_off < SML_FILE_HEADER_BYTES ||
                           !range_inside(m->size, m->hdr->tok_off, m->hdr->tok_n))) ||
        (m->hdr->group_off && (m->hdr->group_off < SML_FILE_HEADER_BYTES ||
                               !range_inside(m->size, m->hdr->group_off, sizeof(sml_group)))) ||
        ((m->hdr->flags & SML_MODEL_EXACT) &&
         (m->hdr->n_levels != 1 || (m->hdr->flags & (SML_MODEL_HADAMARD | SML_MODEL_SCALE6 |
                                                     SML_MODEL_MINNEG))))) {
        fprintf(stderr, "sml: %s has invalid metadata ranges or exact-weight flags\n", path);
        sml_model_close(m);
        return 0;
    }
    // v9 packs per-group scales and mins at 6 bits. The kernels read that as a
    // file property rather than a per-call argument, so a process holding two
    // models -- speculative decoding opens a draft and a target -- requires
    // both to agree. Refuse rather than silently decode one of them wrong.
    {
        const int six = (m->hdr->flags & SML_MODEL_SCALE6) ? 1 : 0;
        static int have = 0, first = 0;
        if (have && first != six) {
            fprintf(stderr, "sml: %s uses %d-bit group scales but a model "
                    "already open uses %d-bit; they cannot share a process\n",
                    path, six ? 6 : 8, first ? 6 : 8);
            sml_model_close(m);
            return 0;
        }
        have = 1; first = six;
        sml_kernel_scale6(six);
        // negative-only mins are a property of the file too, and just as global
        static int have_n = 0, first_n = 0;
        const int mn = six && (m->hdr->flags & SML_MODEL_MINNEG) != 0;
        if (have_n && first_n != mn) {
            fprintf(stderr, "sml: %s reads its 6-bit mins differently from a model already open\n", path);
            sml_model_close(m);
            return 0;
        }
        have_n = 1; first_n = mn;
        sml_kernel_minneg(mn);
    }
    if (m->hdr->group != SML_GROUP) {
        fprintf(stderr, "sml: file group=%u, kernel built for %d\n",
                m->hdr->group, SML_GROUP);
        sml_model_close(m);
        return 0;
    }
    // A v8 FLAT build is a single 4-bit plane, which the flat kernel reads and
    // the ladder kernel cannot. Admit exactly that signature and nothing else --
    // the guard's purpose is to refuse what is not implemented rather than run
    // it wrong, and every bug that cost real time here produced plausible output
    // from a wrong reading rather than an error.
    const int flat_hdr = (m->hdr->version >= 8 && m->hdr->n_levels == 1 &&
                          m->hdr->level_bits[0] == 4);
    if (!flat_hdr) {
        for (uint32_t i = 0; i < m->hdr->n_levels; i++) {
            if (m->hdr->level_bits[i] != SML_LEVEL_BITS) {
                fprintf(stderr, "sml: level %u is %u bits, kernel requires %d\n",
                        i, m->hdr->level_bits[i], SML_LEVEL_BITS);
                sml_model_close(m);
                return 0;
            }
        }
    }
    // Refuse what is not implemented rather than running it wrong. Every bug
    // that cost real time on this project produced plausible output from a
    // wrong computation; a model whose maths differs must fail loudly.
    if (m->hdr->version >= 4) {
        // Implemented: RMSNorm and LayerNorm, halved and interleaved RoPE,
        // custom attention scale, logit soft-capping, sliding-window attention.
        // Anything still unimplemented must refuse rather than run wrong maths.
        const char * why = NULL;
        if (m->hdr->norm_type > 1) why = "an unknown normalisation";
        else if (m->hdr->rope_type > 1) why = "an unknown RoPE layout";
        if (why) {
            fprintf(stderr, "sml: %s needs %s, which this engine does not do yet\n",
                    path, why);
            sml_model_close(m);
            return 0;
        }
    }

    if (m->hdr->version >= 8) {
        m->recs = (const sml_rec *) (m->base + m->hdr->table_off);
    } else if (m->hdr->version >= 6) {
        // v6/v7: same record minus the trailing layout and nbits. Everything
        // built before v8 is a ladder by construction, so that is the default.
        const sml_rec_v7 * old = (const sml_rec_v7 *) (m->base + m->hdr->table_off);
        m->owned = calloc((size_t) m->hdr->n_tensors, sizeof(sml_rec));
        if (!m->owned) { sml_model_close(m); return 0; }
        for (uint64_t i = 0; i < m->hdr->n_tensors; i++) {
            memcpy(&m->owned[i], &old[i], sizeof(sml_rec_v7));
            m->owned[i].layout = SML_LAYOUT_LADDER;
            m->owned[i].nbits  = 0;
        }
        m->recs = m->owned;
    } else {
        const sml_rec_v5 * old = (const sml_rec_v5 *) (m->base + m->hdr->table_off);
        m->owned = calloc((size_t) m->hdr->n_tensors, sizeof(sml_rec));
        if (!m->owned) { sml_model_close(m); return 0; }
        for (uint64_t i = 0; i < m->hdr->n_tensors; i++) {
            sml_rec * d = &m->owned[i];
            const sml_rec_v5 * o = &old[i];
            memcpy(d->name, o->name, SML_NAME_LEN);
            d->kind = o->kind; d->rows = o->rows; d->cols = o->cols;
            d->group = o->group;
            d->off = o->off; d->nbytes = o->nbytes;
            d->sd_off = o->sd_off; d->sd_n = o->sd_n;
            d->sq_off = o->sq_off; d->sq_n = o->sq_n;
            d->perm_off = o->perm_off; d->perm_n = o->perm_n;
            d->md_off = d->md_n = d->mq_off = d->mq_n = 0;   // symmetric
            for (int l = 0; l < SML_MAX_LEVELS; l++) {
                d->codes_off[l] = o->codes_off[l];
                d->codes_n[l]   = o->codes_n[l];
            }
        }
        m->recs = m->owned;
    }
    if (!validate_tensor_ranges(m, path)) {
        sml_model_close(m);
        return 0;
    }

    // v7 group table: appended after the JSON block, terminated by a zero kind
    // or the end of the file. Older files declare none and the engine falls back
    // to what it can work out from the tensor names.
    m->groups = NULL;
    m->n_groups = 0;
    if (m->hdr->version >= 7 && m->hdr->group_off &&
        range_inside(m->size, m->hdr->group_off, sizeof(sml_group))) {
        const sml_group * g = (const sml_group *) (m->base + m->hdr->group_off);
        uint32_t n = 0;
        const uint64_t max_groups = ((uint64_t) m->size - m->hdr->group_off) / sizeof(sml_group);
        while ((uint64_t) n < max_groups
               && g[n].kind != 0 && g[n].n > 0 && g[n].n <= SML_GROUP_MAX) {
            n++;
        }
        if (n) { m->groups = g; m->n_groups = n; }
    }
    return 1;
}

void sml_model_close(sml_model * m) {
    free(m->owned);
    m->owned = NULL;
    if (!m->base) return;
#if defined(_WIN32)
    if (m->owns_pages) {
        VirtualFree((LPVOID) m->base, 0, MEM_RELEASE);
    } else {
        UnmapViewOfFile((LPCVOID) m->base);
        if (m->handle) CloseHandle((HANDLE) m->handle);
    }
#else
    munmap((void *) m->base, m->size);
#endif
    memset(m, 0, sizeof *m);
}

const sml_rec * sml_find(const sml_model * m, const char * name) {
    for (uint64_t i = 0; i < m->hdr->n_tensors; i++) {
        if (strncmp(m->recs[i].name, name, SML_NAME_LEN) == 0) return &m->recs[i];
    }
    return NULL;
}

const sml_rec * sml_findf(const sml_model * m, const char * fmt, int layer) {
    char buf[SML_NAME_LEN];
    snprintf(buf, sizeof buf, fmt, layer);
    return sml_find(m, buf);
}

uint64_t sml_free_ram(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof st;
    if (!GlobalMemoryStatusEx(&st)) return 0;
    return (uint64_t) st.ullAvailPhys;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), sz = sysconf(_SC_PAGESIZE);
    if (pages < 0 || sz < 0) return 0;
    return (uint64_t) pages * (uint64_t) sz;
#endif
}

int sml_fit_depth(const sml_model * m, uint64_t budget) {
    // Pick the deepest read whose working set fits the room we are actually in.
    //
    // This is the one thing the format can do that a fixed-precision file
    // cannot, and it went unused for the whole project: a GGUF-style file has
    // its precision cut at conversion time, so a machine that cannot hold it
    // has no move except to download a different file. The ladder means the
    // SAME file serves every budget, decided when it lands on the hardware.
    //
    // Loose fit, long life: the file is not optimised for one machine, it
    // accommodates whichever one it meets.
    if (!budget) budget = sml_free_ram();
    if (!budget) return (int) m->hdr->n_levels;
    // Leave a margin -- the KV cache, activations and the OS all want room, and
    // a model that fits exactly is a model that swaps.
    const uint64_t usable = budget - budget / 5;
    // Never auto-select below depth 2. Measured: a depth-1 read agrees with
    // depth 2 on the top token 9.4% of the time -- it is noise, not a coarse
    // answer, and its top-2 margin carries no signal about when to trust it.
    // Fitting the machine is worth nothing if the fit produces nonsense, so the
    // floor is a usable read and running out of room is the caller's problem.
    const int floor_d = m->hdr->n_levels >= 2 ? 2 : 1;
    for (int d = (int) m->hdr->n_levels; d > floor_d; d--)
        if (sml_bytes_at_depth(m, d) <= usable) return d;
    return floor_d;
}

uint64_t sml_bytes_at_depth(const sml_model * m, int depth) {
    uint64_t n = 0;
    for (uint64_t i = 0; i < m->hdr->n_tensors; i++) {
        const sml_rec * r = &m->recs[i];
        if (r->kind != SML_KIND_LADDER) { n += r->nbytes; continue; }
        n += r->sd_n + r->sq_n + r->md_n + r->mq_n;
        if (r->layout == SML_LAYOUT_FLAT) {
            // No partial read exists for a flat tensor, so `depth` does not
            // apply: every plane it has is streamed. Counting only plane 0 read
            // a 6-bit tensor as its 2-bit plane alone and UNDER-reported
            // bytes/token -- the one number that must never flatter us,
            // because it is the denominator of every bandwidth claim.
            for (int l = 0; l < SML_MAX_LEVELS; l++) n += r->codes_n[l];
            continue;
        }
        for (int l = 0; l < depth && l < (int) m->hdr->n_levels; l++) n += r->codes_n[l];
    }
    return n;
}
