// Minimal JSON reader/writer for the server.
//
// The ENGINE parses no JSON -- that is a deliberate property of SSF, and it is
// why the model file carries a binary header, a binary tensor table and a binary
// vocab. But an OpenAI-compatible endpoint speaks JSON by definition, so the
// server needs just enough to walk a request and emit a response.
//
// Scanner, not a DOM: it finds values by path and hands back slices of the input
// buffer. No allocation, no tree, and nothing to free.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char * p;   // start of the value (points into the caller's buffer)
    size_t       n;   // length
    int          kind; // 's' string, 'n' number, 'b' bool, 'o' object, 'a' array, 0 none
} sml_json;

// Find `key` at the top level of the object starting at `obj`.
sml_json sml_json_get(const char * obj, size_t len, const char * key);

// Walk an array: call with idx = 0, 1, ... until kind == 0.
sml_json sml_json_at(const char * arr, size_t len, int idx);

// Decode a JSON string value into `out` (unescaped, NUL-terminated).
// Returns bytes written.
size_t sml_json_str(sml_json v, char * out, size_t max);

long   sml_json_int(sml_json v, long fallback);
double sml_json_num(sml_json v, double fallback);

// Append `s` to `out` as an escaped JSON string body (no surrounding quotes).
size_t sml_json_escape(const char * s, size_t n, char * out, size_t max);

#ifdef __cplusplus
}
#endif
