#include "sml-json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char * skip_ws(const char * p, const char * e) {
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

// Advance past one complete value, honouring nesting and string escapes.
static const char * skip_value(const char * p, const char * e) {
    p = skip_ws(p, e);
    if (p >= e) return e;
    if (*p == '"') {
        p++;
        while (p < e && *p != '"') p += (*p == '\\' && p + 1 < e) ? 2 : 1;
        return p < e ? p + 1 : e;
    }
    if (*p == '{' || *p == '[') {
        const char open = *p, close = (open == '{') ? '}' : ']';
        int d = 0;
        while (p < e) {
            if (*p == '"') {                      // strings can contain braces
                p++;
                while (p < e && *p != '"') p += (*p == '\\' && p + 1 < e) ? 2 : 1;
                if (p < e) p++;
                continue;
            }
            if (*p == open) d++;
            else if (*p == close && --d == 0) return p + 1;
            p++;
        }
        return e;
    }
    while (p < e && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') p++;
    return p;
}

static int kind_of(const char * p) {
    switch (*p) {
        case '"': return 's';
        case '{': return 'o';
        case '[': return 'a';
        case 't': case 'f': return 'b';
        default:  return 'n';
    }
}

static sml_json make(const char * p, const char * end) {
    sml_json v;
    v.p = p;
    v.n = (size_t)(end - p);
    v.kind = (v.n && p) ? kind_of(p) : 0;
    if (v.kind == 's') { v.p = p + 1; v.n = v.n >= 2 ? v.n - 2 : 0; }
    return v;
}

sml_json sml_json_get(const char * obj, size_t len, const char * key) {
    sml_json none = { NULL, 0, 0 };
    const char * e = obj + len;
    const char * p = skip_ws(obj, e);
    if (p >= e || *p != '{') return none;
    p++;
    const size_t klen = strlen(key);

    while (p < e) {
        p = skip_ws(p, e);
        if (p >= e || *p == '}') break;
        if (*p != '"') { p = skip_value(p, e); continue; }
        const char * ks = p + 1;
        const char * ke = skip_value(p, e);       // past the closing quote
        const size_t n = (size_t)(ke - ks - 1);
        p = skip_ws(ke, e);
        if (p < e && *p == ':') p++;
        const char * vs = skip_ws(p, e);
        const char * ve = skip_value(vs, e);
        if (n == klen && memcmp(ks, key, klen) == 0) return make(vs, ve);
        p = skip_ws(ve, e);
        if (p < e && *p == ',') p++;
    }
    return none;
}

sml_json sml_json_at(const char * arr, size_t len, int idx) {
    sml_json none = { NULL, 0, 0 };
    const char * e = arr + len;
    const char * p = skip_ws(arr, e);
    if (p >= e || *p != '[') return none;
    p++;
    for (int i = 0; p < e; i++) {
        p = skip_ws(p, e);
        if (p >= e || *p == ']') break;
        const char * vs = p;
        const char * ve = skip_value(vs, e);
        if (i == idx) return make(vs, ve);
        p = skip_ws(ve, e);
        if (p < e && *p == ',') p++;
    }
    return none;
}

size_t sml_json_str(sml_json v, char * out, size_t max) {
    size_t w = 0;
    if (v.kind != 's' || !max) { if (max) out[0] = 0; return 0; }
    for (size_t i = 0; i < v.n && w + 1 < max; i++) {
        char c = v.p[i];
        if (c == '\\' && i + 1 < v.n) {
            const char n = v.p[++i];
            switch (n) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'u': {
                    // Basic-plane only; enough for chat text. Surrogate pairs
                    // fall back to the replacement character.
                    if (i + 4 < v.n) {
                        char hex[5] = { v.p[i+1], v.p[i+2], v.p[i+3], v.p[i+4], 0 };
                        const unsigned cp = (unsigned) strtoul(hex, NULL, 16);
                        i += 4;
                        if (cp < 0x80) { out[w++] = (char) cp; }
                        else if (cp < 0x800 && w + 2 < max) {
                            out[w++] = (char)(0xC0 | (cp >> 6));
                            out[w++] = (char)(0x80 | (cp & 0x3F));
                        } else if (w + 3 < max) {
                            out[w++] = (char)(0xE0 | (cp >> 12));
                            out[w++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            out[w++] = (char)(0x80 | (cp & 0x3F));
                        }
                        continue;
                    }
                    c = '?';
                    break;
                }
                default: c = n; break;
            }
        }
        out[w++] = c;
    }
    out[w] = 0;
    return w;
}

long sml_json_int(sml_json v, long fallback) {
    if (v.kind != 'n' || !v.n) return fallback;
    char buf[32];
    const size_t n = v.n < sizeof buf - 1 ? v.n : sizeof buf - 1;
    memcpy(buf, v.p, n);
    buf[n] = 0;
    return strtol(buf, NULL, 10);
}

double sml_json_num(sml_json v, double fallback) {
    if (v.kind != 'n' || !v.n) return fallback;
    char buf[48];
    const size_t n = v.n < sizeof buf - 1 ? v.n : sizeof buf - 1;
    memcpy(buf, v.p, n);
    buf[n] = 0;
    return strtod(buf, NULL);
}

size_t sml_json_escape(const char * s, size_t n, char * out, size_t max) {
    size_t w = 0;
    for (size_t i = 0; i < n && w + 8 < max; i++) {
        const unsigned char c = (unsigned char) s[i];
        switch (c) {
            case '"':  out[w++] = '\\'; out[w++] = '"';  break;
            case '\\': out[w++] = '\\'; out[w++] = '\\'; break;
            case '\n': out[w++] = '\\'; out[w++] = 'n';  break;
            case '\t': out[w++] = '\\'; out[w++] = 't';  break;
            case '\r': out[w++] = '\\'; out[w++] = 'r';  break;
            default:
                if (c < 0x20) w += (size_t) snprintf(out + w, max - w, "\\u%04x", c);
                else out[w++] = (char) c;
        }
    }
    out[w] = 0;
    return w;
}
