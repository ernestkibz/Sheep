#include "sml-model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

typedef struct {
    const char * name;
    int (*mutate)(sml_hdr * hdr, sml_rec * rec, uint8_t * payload);
    int valid;
} fixture_case;

static int no_change(sml_hdr * hdr, sml_rec * rec, uint8_t * payload) {
    (void) hdr; (void) rec; (void) payload;
    return 1;
}

static int valid_quantized(sml_hdr * hdr, sml_rec * rec, uint8_t * payload) {
    (void) payload;
    hdr->flags = 0;
    rec->kind = SML_KIND_LADDER;
    rec->rows = 1;
    rec->cols = SML_GROUP;
    rec->group = SML_GROUP;
    rec->nbytes = 0;
    rec->codes_off[0] = hdr->table_off + sizeof *rec;
    rec->codes_n[0] = 64;
    return 1;
}

#define BAD_HEADER(fn) \
    static int fn(sml_hdr * hdr, sml_rec * rec, uint8_t * payload) { \
        (void) rec; (void) payload;
#define BAD_RECORD(fn) \
    static int fn(sml_hdr * hdr, sml_rec * rec, uint8_t * payload) { \
        (void) hdr; (void) payload;
#define BAD_BOTH(fn) \
    static int fn(sml_hdr * hdr, sml_rec * rec, uint8_t * payload) {
#define MUTATE_END return 1; }

BAD_HEADER(bad_magic)
    hdr->magic[0] = 'X';
MUTATE_END

BAD_HEADER(newer_version)
    hdr->version = SML_MAX_VERSION + 1;
MUTATE_END

BAD_HEADER(bad_level_count)
    hdr->n_levels = SML_MAX_LEVELS + 1;
MUTATE_END

BAD_HEADER(bad_flags)
    hdr->flags = 0x80000000u;
MUTATE_END

BAD_HEADER(bad_exact_flags)
    hdr->flags |= SML_MODEL_HADAMARD;
MUTATE_END

BAD_HEADER(bad_table_range)
    hdr->table_off = UINT64_MAX;
MUTATE_END

BAD_HEADER(bad_json_range)
    hdr->json_off = UINT64_MAX;
    hdr->json_n = 1;
MUTATE_END

BAD_HEADER(bad_group_range)
    hdr->group_off = UINT64_MAX;
MUTATE_END

BAD_RECORD(zero_dimensions)
    rec->rows = 0;
MUTATE_END

BAD_RECORD(unknown_kind)
    rec->kind = 99;
MUTATE_END

BAD_RECORD(bad_f32_payload_size)
    rec->nbytes--;
MUTATE_END

BAD_RECORD(bad_f32_payload_range)
    rec->off = UINT64_MAX;
MUTATE_END

BAD_RECORD(exact_with_quantized_tensor)
    rec->kind = SML_KIND_LADDER;
MUTATE_END

BAD_BOTH(bad_flat_layout)
    (void) payload;
    hdr->flags = 0;
    rec->kind = SML_KIND_LADDER;
    rec->layout = SML_LAYOUT_FLAT;
    rec->nbits = 1;
MUTATE_END

BAD_BOTH(bad_code_range)
    (void) payload;
    hdr->flags = 0;
    rec->kind = SML_KIND_LADDER;
    rec->layout = SML_LAYOUT_LADDER;
    rec->codes_off[0] = UINT64_MAX;
    rec->codes_n[0] = 1;
MUTATE_END

static int write_fixture(const char * path, const fixture_case * test,
                         int truncate_file) {
    sml_hdr hdr;
    sml_rec rec;
    uint8_t payload[64] = {0};
    memset(&hdr, 0, sizeof hdr);
    memset(&rec, 0, sizeof rec);
    memcpy(hdr.magic, "SSF ", 4);
    hdr.version = SML_MAX_VERSION;
    hdr.n_layer = hdr.n_embd = hdr.n_head = hdr.n_head_kv = 1;
    hdr.n_ff = hdr.n_vocab = hdr.n_ctx_train = 1;
    hdr.group = SML_GROUP;
    hdr.n_levels = 1;
    hdr.level_bits[0] = SML_LEVEL_BITS;
    hdr.n_tensors = 1;
    hdr.table_off = SML_FILE_HEADER_BYTES;
    hdr.flags = SML_MODEL_EXACT;
    rec.kind = SML_KIND_F32;
    rec.rows = rec.cols = 1;
    rec.off = hdr.table_off + sizeof rec;
    rec.nbytes = sizeof(float);
    rec.layout = SML_LAYOUT_LADDER;
    if (!test->mutate(&hdr, &rec, payload)) return 0;

    FILE * f = fopen(path, "wb");
    if (!f) return 0;
    int ok = 1;
    if (truncate_file) {
        ok = fwrite(&hdr, 1, 64, f) == 64;
    } else {
        uint8_t padding[SML_FILE_HEADER_BYTES] = {0};
        memcpy(padding, &hdr, sizeof hdr);
        ok = fwrite(padding, 1, sizeof padding, f) == sizeof padding &&
             fwrite(&rec, 1, sizeof rec, f) == sizeof rec &&
             fwrite(payload, 1, sizeof payload, f) == sizeof payload;
    }
    if (fclose(f) != 0) ok = 0;
    return ok;
}

int main(void) {
    static const fixture_case cases[] = {
        {"valid exact f32", no_change, 1},
        {"valid quantized ladder", valid_quantized, 1},
        {"truncated header", no_change, 0},
        {"bad magic", bad_magic, 0},
        {"newer format version", newer_version, 0},
        {"invalid level count", bad_level_count, 0},
        {"unsupported flags", bad_flags, 0},
        {"invalid exact flags", bad_exact_flags, 0},
        {"tensor table out of bounds", bad_table_range, 0},
        {"JSON metadata out of bounds", bad_json_range, 0},
        {"group table out of bounds", bad_group_range, 0},
        {"zero tensor dimensions", zero_dimensions, 0},
        {"unknown tensor kind", unknown_kind, 0},
        {"incorrect F32 payload size", bad_f32_payload_size, 0},
        {"F32 payload out of bounds", bad_f32_payload_range, 0},
        {"quantized tensor in exact file", exact_with_quantized_tensor, 0},
        {"invalid flat code width", bad_flat_layout, 0},
        {"quantized codes out of bounds", bad_code_range, 0},
    };
    unsigned failures = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        char path[160];
        snprintf(path, sizeof path, "sml-model-test-%ld-%u.ssf",
                 (long) getpid(), (unsigned) i);
        if (!write_fixture(path, &cases[i],
                           strcmp(cases[i].name, "truncated header") == 0)) {
            fprintf(stderr, "FAIL: could not write fixture: %s\n", cases[i].name);
            failures++;
            remove(path);
            continue;
        }
        sml_model model;
        int opened = sml_model_open(&model, path);
        if (opened) sml_model_close(&model);
        remove(path);
        if (opened != cases[i].valid) {
            fprintf(stderr, "FAIL: %s: expected %s, loader %s\n", cases[i].name,
                    cases[i].valid ? "acceptance" : "rejection",
                    opened ? "accepted" : "rejected");
            failures++;
        } else {
            printf("PASS: %s\n", cases[i].name);
        }
    }
    printf("%u cases, %u failures\n",
           (unsigned) (sizeof cases / sizeof cases[0]), failures);
    return failures ? 1 : 0;
}
