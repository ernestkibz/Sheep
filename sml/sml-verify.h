// Exact arithmetic verification for the server's orchestrator.
//
// Why this exists: `orchestrate()` classified a prompt as ACT_VERIFY and then
// did nothing with it -- the label reached the response JSON and changed not one
// token. Meanwhile the one accuracy lever this project ever measured as working
// was exactly this: verify computable answers EXTERNALLY rather than asking the
// model to self-correct. On the v1 harness that took math from 90% to 100% and
// overall accuracy from 72.5% to 81.2%.
//
// Why it evaluates arithmetic in-process instead of shelling out to python, as
// v1 did: a server that spawns an interpreter on request text is a remote code
// execution surface, and this one binds a socket. Integer arithmetic covers the
// case that actually moved the number and needs no subprocess, no shell quoting
// and no sandbox. Anything it cannot parse it declines, and declining leaves the
// model's own answer untouched.
//
// It is deliberately EXACT: 64-bit integers, no floating point, no silent
// overflow. A verifier that is occasionally wrong is worse than none, because it
// would overwrite a correct answer with a confident wrong one.
#ifndef SML_VERIFY_H
#define SML_VERIFY_H

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char * p;
    int          bad;      // set on overflow, divide-by-zero or a parse failure
} sv_scan;

static long long sv_expr(sv_scan * s);

static void sv_ws(sv_scan * s) {
    while (*s->p == ' ' || *s->p == '\t' || *s->p == ',') s->p++;
}

// Overflow-checked helpers. Returning a wrong number confidently is the one
// failure mode this must not have.
static long long sv_mul(sv_scan * s, long long a, long long b) {
    if (a != 0 && (b > INT64_MAX / a || b < INT64_MIN / a)) { s->bad = 1; return 0; }
    return a * b;
}
static long long sv_add(sv_scan * s, long long a, long long b) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) { s->bad = 1; return 0; }
    return a + b;
}

static long long sv_atom(sv_scan * s) {
    sv_ws(s);
    if (*s->p == '(') {
        s->p++;
        const long long v = sv_expr(s);
        sv_ws(s);
        if (*s->p == ')') s->p++; else s->bad = 1;
        return v;
    }
    if (*s->p == '-') { s->p++; const long long v = sv_atom(s); return sv_mul(s, v, -1); }
    if (*s->p == '+') { s->p++; return sv_atom(s); }
    if (!isdigit((unsigned char) *s->p)) { s->bad = 1; return 0; }
    long long v = 0;
    int digits = 0;
    while (isdigit((unsigned char) *s->p)) {
        v = sv_add(s, sv_mul(s, v, 10), *s->p - '0');
        if (s->bad) return 0;
        s->p++;
        if (++digits > 18) { s->bad = 1; return 0; }
    }
    // A decimal point means this is not integer arithmetic; decline rather than
    // truncate, because truncating produces a confident wrong answer.
    if (*s->p == '.' && isdigit((unsigned char) s->p[1])) { s->bad = 1; return 0; }
    return v;
}

// "to the power of" is written ^ here; it is right-associative and bounded,
// because a prompt can otherwise ask for 9^9^9 and hang the server.
static long long sv_pow(sv_scan * s) {
    long long base = sv_atom(s);
    sv_ws(s);
    if (*s->p == '^') {
        s->p++;
        const long long e = sv_pow(s);
        if (e < 0 || e > 63) { s->bad = 1; return 0; }
        long long r = 1;
        for (long long i = 0; i < e; i++) { r = sv_mul(s, r, base); if (s->bad) return 0; }
        return r;
    }
    return base;
}

static long long sv_term(sv_scan * s) {
    long long v = sv_pow(s);
    for (;;) {
        sv_ws(s);
        const char c = *s->p;
        if (c == '*' || c == 'x' || c == 'X') {
            // 'x' is only multiplication between two numbers ("12 x 4"), never
            // inside a word.
            if ((c == 'x' || c == 'X') && isalnum((unsigned char) s->p[1])) return v;
            s->p++;
            v = sv_mul(s, v, sv_pow(s));
        } else if (c == '/') {
            s->p++;
            const long long d = sv_pow(s);
            if (d == 0) { s->bad = 1; return 0; }
            if (v % d != 0) { s->bad = 1; return 0; }   // not an integer answer
            v /= d;
        } else if (c == '%') {
            s->p++;
            const long long d = sv_pow(s);
            if (d == 0) { s->bad = 1; return 0; }
            v %= d;
        } else return v;
        if (s->bad) return 0;
    }
}

static long long sv_expr(sv_scan * s) {
    long long v = sv_term(s);
    for (;;) {
        sv_ws(s);
        if (*s->p == '+')      { s->p++; v = sv_add(s, v, sv_term(s)); }
        else if (*s->p == '-') { s->p++; v = sv_add(s, v, sv_mul(s, sv_term(s), -1)); }
        else return v;
        if (s->bad) return 0;
    }
}

// Rewrite the English of an arithmetic question into symbols, in place.
// "17 times 23" -> "17 * 23". Only whole words are replaced.
static void sv_normalise(const char * in, char * out, size_t max) {
    static const char * WORD[][2] = {
        {"multiplied by", "*"}, {"divided by", "/"}, {"to the power of", "^"},
        {"times", "*"}, {"plus", "+"}, {"minus", "-"}, {"modulo", "%"},
        {"mod", "%"}, {"squared", "^2"}, {"cubed", "^3"}, {NULL, NULL}
    };
    size_t w = 0;
    for (size_t i = 0; in[i] && w + 1 < max; ) {
        int hit = 0;
        for (int k = 0; WORD[k][0]; k++) {
            const size_t n = strlen(WORD[k][0]);
            if (strncmp(in + i, WORD[k][0], n) == 0) {
                const int lb = (i == 0) || !isalnum((unsigned char) in[i - 1]);
                const int rb = !isalnum((unsigned char) in[i + n]);
                if (lb && rb) {
                    const size_t rn = strlen(WORD[k][1]);
                    if (w + rn + 1 >= max) { out[w] = 0; return; }
                    memcpy(out + w, WORD[k][1], rn);
                    w += rn; i += n; hit = 1;
                    break;
                }
            }
        }
        if (!hit) out[w++] = in[i++];
    }
    out[w] = 0;
}

// Find and evaluate the arithmetic in `lower`. Returns 1 and fills `val` only
// when the whole expression parsed cleanly and at least one operator was
// present -- a bare number is not a computation.
static int sml_verify_arith_expr(const char * lower, long long * val, char * shown, size_t smax) {
    char norm[2048];
    sv_normalise(lower, norm, sizeof norm);

    // A decimal anywhere means this is not integer arithmetic. Decline the WHOLE
    // prompt rather than scanning past it: an earlier version stopped at the '.'
    // in "3.5 + 1", picked up the fragment "5 + 1" and answered 6. Extracting a
    // piece of a number and computing confidently with it is precisely the
    // failure this verifier exists to avoid.
    for (const char * d = norm; *d; d++)
        if (*d == '.' && d != norm && isdigit((unsigned char) d[-1])
            && isdigit((unsigned char) d[1])) return 0;

    // Scan for the longest run that is entirely arithmetic. A run may open on a
    // parenthesis, so "(3 + 4) * 5" is not missed for want of a leading digit.
    const char * best = NULL;
    size_t bestlen = 0;
    for (const char * p = norm; *p; p++) {
        if (!isdigit((unsigned char) *p) && *p != '(') continue;
        if (p != norm && (isalnum((unsigned char) p[-1]) || p[-1] == '_')) continue;
        const char * q = p;
        int ops = 0, depth = 0;
        while (*q) {
            const char c = *q;
            if (isdigit((unsigned char) c) || c == ' ' || c == ',') { q++; continue; }
            if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == '^') {
                ops++; q++; continue;
            }
            // "12 x 4" is multiplication; the x in "x2" or "box" is not.
            if ((c == 'x' || c == 'X')
                && !isalnum((unsigned char) q[1]) && q != p
                && !isalnum((unsigned char) q[-1])) { ops++; q++; continue; }
            if (c == '(') { depth++; q++; continue; }
            if (c == ')') { if (--depth < 0) break; q++; continue; }
            break;
        }
        while (q > p && (q[-1] == ' ' || q[-1] == ',')) q--;
        if (ops && depth == 0 && (size_t)(q - p) > bestlen) { best = p; bestlen = (size_t)(q - p); }
        p = q > p ? q - 1 : p;
    }
    if (!best || bestlen < 3 || bestlen >= 512) return 0;
    // A run starts at a digit, so in "what is -5 + 3" it is "5 + 3", and the
    // value (8) contradicts the question (-2). A sign in front of the run means
    // the run is not the whole expression: decline.
    {
        const char * b = best;
        while (b > norm && b[-1] == ' ') b--;
        if (b > norm && (b[-1] == '-' || b[-1] == '+')) return 0;
    }

    char expr[512];
    memcpy(expr, best, bestlen);
    expr[bestlen] = 0;

    // A date, a phone number or a year range is not a sum: "add Dentist on
    // 2026-11-03" was answered "Checked by exact computation: 2026-11-03 = 2012"
    // (2026-10-12 simulation). Digit groups joined by '-' or '/' with no spaces:
    // three groups (2026-11-03, 3/4/2026) or two long ones (555-1234, 2020-2024)
    // are declined; "12-5" and "10/2" are still arithmetic.
    {
        int groups = 1, glen[4] = {0, 0, 0, 0}, plain = 1;
        for (const char * q = expr; *q && plain; q++) {
            if (isdigit((unsigned char) *q)) { if (groups <= 4) glen[groups - 1]++; }
            else if ((*q == '-' || *q == '/') && q > expr && isdigit((unsigned char) q[-1])
                     && isdigit((unsigned char) q[1])) groups++;
            else plain = 0;
        }
        if (plain && (groups == 3 || (groups == 2 && glen[0] >= 3 && glen[1] >= 3))) return 0;
    }

    sv_scan s = { expr, 0 };
    const long long v = sv_expr(&s);
    sv_ws(&s);
    if (s.bad || *s.p) return 0;      // trailing junk means we did not understand it
    *val = v;
    if (shown && smax) {
        size_t w = 0;
        for (const char * q = expr; *q && w + 1 < smax; q++) shown[w++] = *q;
        shown[w] = 0;
    }
    return 1;
}

static int sml_verify_arith(const char * lower, long long * val) {
    return sml_verify_arith_expr(lower, val, NULL, 0);
}

// Is this turn NOTHING BUT an arithmetic question? "what is 343 times 982?",
// "calculate (3 + 4) * 5", "12 x 4 =". Then the server answers with the exact
// value itself: handing the value to a 1B model as a fact did not stop it
// working the sum out step by step and getting it wrong (9 of 10 products
// wrong, 2026-09-25, with the response still marked verified). Anything with
// other words in it -- a word problem, a question about a number -- is not
// pure, and goes to the model as before. `shown` gets the expression as asked.
static int sml_verify_pure(const char * lower_turn, long long * val, char * shown, size_t max) {
    char norm[2048];
    sv_normalise(lower_turn, norm, sizeof norm);
    char * p = norm;
    size_t n = strlen(p);
    for (;;) {                                         // trailing ? ! . = and filler
        while (n && strchr(" \t\r\n?!.=", p[n - 1])) n--;
        if (n >= 7 && !strncmp(p + n - 7, " equals", 7)) { n -= 7; continue; }
        if (n >= 6 && !strncmp(p + n - 6, " equal", 6))  { n -= 6; continue; }
        if (n >= 7 && !strncmp(p + n - 7, " please", 7)) { n -= 7; continue; }
        break;
    }
    p[n] = 0;
    static const char * LEAD[] = { "what is ", "what's ", "whats ", "how much is ",
                                   "calculate ", "compute ", "evaluate ", "solve ",
                                   "please ", "work out ", NULL };
    for (int again = 1; again; ) {
        again = 0;
        while (*p == ' ' || *p == '\t') p++;
        for (int k = 0; LEAD[k]; k++) {
            const size_t ln = strlen(LEAD[k]);
            if (!strncmp(p, LEAD[k], ln)) { p += ln; again = 1; break; }
        }
    }
    // The arithmetic scan starts runs at a digit or '(', so "-5 + 3" would be
    // read as "5 + 3": a pure question must start where the scan starts, and
    // balanced parentheses keep the run from ending before the text does.
    if (!isdigit((unsigned char) *p) && *p != '(') return 0;
    int bal = 0;
    for (const char * q = p; *q; q++) {
        if (!(isdigit((unsigned char) *q) || strchr(" ,+-*/%^()xX", *q))) return 0;
        bal += (*q == '(') - (*q == ')');
        if (bal < 0) return 0;
    }
    if (bal) return 0;
    if (!sml_verify_arith(p, val)) return 0;
    size_t w = 0;
    for (const char * q = p; *q && w + 1 < max; q++) shown[w++] = *q;
    while (w && shown[w - 1] == ' ') w--;
    shown[w] = 0;
    return 1;
}

// Does `text` state `val`? Thousands separators are ignored; the number must
// stand alone (336826 is not found inside 3368261).
static int sml_verify_states(const char * text, long long val) {
    char want[32], flat[8192];
    snprintf(want, sizeof want, "%lld", val);
    size_t w = 0;
    for (size_t i = 0; text[i] && w + 1 < sizeof flat; i++) {
        if (text[i] == ',' && i && isdigit((unsigned char) text[i - 1])
            && isdigit((unsigned char) text[i + 1])) continue;
        flat[w++] = text[i];
    }
    flat[w] = 0;
    const size_t wl = strlen(want);
    for (const char * h = strstr(flat, want); h; h = strstr(h + 1, want)) {
        const int lb = h == flat || !isdigit((unsigned char) h[-1]);
        const int rb = !isdigit((unsigned char) h[wl]);
        if (lb && rb) return 1;
    }
    return 0;
}

#endif
