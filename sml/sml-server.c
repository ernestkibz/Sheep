// sml-server: OpenAI-compatible endpoint on the SML engine.
//
// This is what let the forked applications, runtime and tensor trees be deleted.
// It also carries forward the only original work v1 had: the orchestrator's
// deterministic pre-model routing, and the measured-profile idea behind tuning.
//
// The orchestrator rules are ported verbatim in intent, including the finding
// that matters most about them: the intervention is to SUPPRESS tools, not to
// force them. Qwen2.5-3B at temperature 0, offered a web tool, over-called on
// 5 of 6 questions needing no search and under-called on 0 of 6 that did. The
// bias is entirely toward calling, so forcing was worthless and clearing the
// tool list is what helped.
//
// Tool calling (ML.M.I, 2026-09-30): a request's `tools` are written into the
// prompt in the model family's own convention, and what the model writes back
// is returned as OpenAI `tool_calls`. The agent that executes them lives in
// front of this server; the engine only speaks the protocol.
//
//   sml-server MODEL.ssf --port 8080 --threads 6 --depth 2 [--slots 4]

#include "sml-run.h"
#include "sml-gpu.h"

#include "sml-verify.h"
#include "sml-vocab.h"
#include "sml-json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define CLOSESOCK close
#endif

#define REQ_MAX   (1 << 20)
#define GEN_MAX   4096
#define SLOTS_MAX 8
#define CALLS_MAX 16
#define STOPS_MAX 4

// Where the model runs: "cpu", "gpu" (every layer on the card) or "hybrid"
// (the leading layers on the card, the rest on the CPU). The orchestrator asks
// with --device; the GPU may say no, and any failure later finishes the reply on
// the CPU. /stats tells the app which it is and why.
static int  g_gpu = 0;
static char g_dev[16] = "cpu", g_dev_req[16] = "cpu", g_dev_why[200] = "";
static int  g_gpu_layers = 0, g_gpu_of = 0;

// ---- an external device: a worker process that runs the forward pass
// elsewhere (mlmi/backend/dml_worker.py: AMD and Intel GPUs through DirectML).
// The server starts it with --connect PORT and talks over a local socket:
//   hello    'SXEF', status (0 ready / 1 refused), n_vocab, len, text
//   request  op (1 prompt, 2 step, 3 bye), pos0, n, ids[n]
//   reply    status (0 ok), then n_vocab float32 logits
// Everything else (templates, tokenizer, tools, the guard) stays here, and a
// failed request finishes the reply on the CPU, as the GPU does.
static int g_ext = 0;
static char g_ext_cmd[2048] = "", g_dev_name[160] = "";
#if defined(_WIN32)
static SOCKET g_ext_sock = INVALID_SOCKET;
static float * g_ext_lg = NULL;
static int g_ext_vocab = 0;

static int ext_io(char * p, int n, int send_) {
    while (n > 0) {
        const int k = send_ ? send(g_ext_sock, p, n, 0) : recv(g_ext_sock, p, n, 0);
        if (k <= 0) return 0;
        p += k; n -= k;
    }
    return 1;
}

static int ext_open(const char * cmd, int n_vocab, char * why, int nwhy) {
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    int alen = sizeof a;
    if (ls == INVALID_SOCKET || bind(ls, (struct sockaddr *) &a, sizeof a) != 0 || listen(ls, 1) != 0
        || getsockname(ls, (struct sockaddr *) &a, &alen) != 0) {
        snprintf(why, (size_t) nwhy, "cannot open a local port for the device worker");
        return 0;
    }
    char line[2200];
    snprintf(line, sizeof line, "%s --connect %d", cmd, (int) ntohs(a.sin_port));
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si); si.cb = sizeof si;
    memset(&pi, 0, sizeof pi);
    if (!CreateProcessA(NULL, line, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        snprintf(why, (size_t) nwhy, "the device worker did not start");
        closesocket(ls);
        return 0;
    }
    CloseHandle(pi.hThread);
    // the worker loads the model before it connects: wait while it is alive
    for (int waited = 0;; waited++) {
        fd_set rs;
        FD_ZERO(&rs); FD_SET(ls, &rs);
        struct timeval tv = { 1, 0 };
        if (select(0, &rs, NULL, NULL, &tv) > 0) break;
        if (WaitForSingleObject(pi.hProcess, 0) == WAIT_OBJECT_0 || waited > 900) {
            snprintf(why, (size_t) nwhy, "the device worker stopped before it was ready");
            CloseHandle(pi.hProcess);
            closesocket(ls);
            return 0;
        }
    }
    CloseHandle(pi.hProcess);
    g_ext_sock = accept(ls, NULL, NULL);
    closesocket(ls);
    int one = 1;
    setsockopt(g_ext_sock, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof one);
    int hello[4];
    if (g_ext_sock == INVALID_SOCKET || !ext_io((char *) hello, sizeof hello, 0) || hello[0] != 0x46455853) {
        snprintf(why, (size_t) nwhy, "the device worker did not answer");
        return 0;
    }
    char msg[512] = "";
    const int ml = hello[3] < (int) sizeof msg - 1 ? hello[3] : (int) sizeof msg - 1;
    if (ml > 0 && !ext_io(msg, ml, 0)) { snprintf(why, (size_t) nwhy, "the device worker did not answer"); return 0; }
    msg[ml > 0 ? ml : 0] = 0;
    if (hello[1] != 0) {
        snprintf(why, (size_t) nwhy, "%s", msg[0] ? msg : "the device refused the model");
        closesocket(g_ext_sock); g_ext_sock = INVALID_SOCKET;
        return 0;
    }
    if (hello[2] != n_vocab) {
        snprintf(why, (size_t) nwhy, "the device worker read another vocabulary (%d, not %d)", hello[2], n_vocab);
        closesocket(g_ext_sock); g_ext_sock = INVALID_SOCKET;
        return 0;
    }
    g_ext_vocab = n_vocab;
    g_ext_lg = malloc((size_t) n_vocab * sizeof(float));
    snprintf(g_dev_name, sizeof g_dev_name, "%s", msg);
    return g_ext_lg != NULL;
}

// The middle stage (--device multi): n hidden rows to the worker and back, in place.
static int g_mid_layers = 0, g_n_embd = 0;
static int ext_mid(float * rows, int pos0, int n) {
    int head[3] = { 4, pos0, n };
    int st = 1;
    const int bytes = n * g_n_embd * 4;
    return ext_io((char *) head, sizeof head, 1) && ext_io((char *) rows, bytes, 1)
        && ext_io((char *) &st, sizeof st, 0) && st == 0 && ext_io((char *) rows, bytes, 0);
}

// The front stage (--device amd --mid-layers M): the worker embeds ids and runs
// layers 0..M-1 on its GPUs; the CPU runs the rest (a model larger than the cards).
static int g_front = 0;
static float * g_front_rows = NULL;
static int ext_front(const int32_t * ids, int pos0, int n, float * rows) {
    int head[3] = { 5, pos0, n };
    int st = 1;
    return ext_io((char *) head, sizeof head, 1) && ext_io((char *) ids, n * 4, 1)
        && ext_io((char *) &st, sizeof st, 0) && st == 0 && ext_io((char *) rows, n * g_n_embd * 4, 0);
}

static const float * ext_fwd(int op, const int32_t * ids, int pos0, int n) {
    int head[3] = { op, pos0, n };
    int st = 1;
    if (!ext_io((char *) head, sizeof head, 1) || !ext_io((char *) ids, n * 4, 1)
        || !ext_io((char *) &st, sizeof st, 0) || st != 0
        || !ext_io((char *) g_ext_lg, g_ext_vocab * 4, 0)) return NULL;
    return g_ext_lg;
}
#else
static int g_mid_layers = 0, g_n_embd = 0;
static int ext_mid(float * rows, int pos0, int n) { return 0; }
static int g_front = 0;
static float * g_front_rows = NULL;
static int ext_front(const int32_t * ids, int pos0, int n, float * rows) { return 0; }
static int ext_open(const char * cmd, int n_vocab, char * why, int nwhy) {
    snprintf(why, (size_t) nwhy, "external devices run on Windows only for now");
    return 0;
}
static const float * ext_fwd(int op, const int32_t * ids, int pos0, int n) { return NULL; }
#endif

// The CPU contexts that stand by behind the GPU, without decode's copy until needed.
static sml_ctx * g_standby = NULL;
static int g_n_standby = 0;

static void gpu_retire(const char * why) {
    g_gpu = 0;
    g_ext = 0;
    g_front = 0;
    for (int i = 0; i < g_n_standby; i++) sml_ctx_repack(&g_standby[i]);
    g_n_standby = 0;
    snprintf(g_dev, sizeof g_dev, "cpu");
    snprintf(g_dev_why, sizeof g_dev_why, "%s", why);
    fprintf(stderr, "gpu: %s; serving on the CPU\n", why);
}

// The prompt: ids[pos0..pos0+n). On a GPU failure the CPU reads the whole
// conversation from the start, since its own cache never held it.
static const float * fwd_prefill(sml_ctx * c, const int32_t * ids, int pos0, int n, int depth) {
    if (g_front) {
        float * rows = realloc(g_front_rows, (size_t) n * g_n_embd * sizeof(float));
        if (rows) g_front_rows = rows;
        if (rows && ext_front(ids + pos0, pos0, n, rows))
            return sml_prefill_from(c, rows, pos0, n, g_mid_layers, depth);
        gpu_retire("the device worker failed while reading a prompt");
        return sml_prefill(c, ids, 0, pos0 + n, depth);
    }
    if (g_ext) {
        const float * l = ext_fwd(1, ids + pos0, pos0, n);
        if (l) return l;
        gpu_retire("the device worker failed while reading a prompt");
        return sml_prefill(c, ids, 0, pos0 + n, depth);
    }
    if (g_gpu) {
        const float * l = sml_gpu_prefill(ids + pos0, pos0, n);
        if (l) return l;
        gpu_retire("the GPU failed while reading a prompt");
        return sml_prefill(c, ids, 0, pos0 + n, depth);
    }
    return sml_prefill(c, ids + pos0, pos0, n, depth);
}

// One generated token at pos (ids[pos] == tok).
static const float * fwd_step(sml_ctx * c, const int32_t * ids, int tok, int pos, int depth) {
    if (g_front) {
        const int32_t t1 = tok;
        float * rows = realloc(g_front_rows, (size_t) g_n_embd * sizeof(float));
        if (rows) g_front_rows = rows;
        if (rows && ext_front(&t1, pos, 1, rows)) return sml_forward_from(c, rows, pos, g_mid_layers, depth);
        gpu_retire("the device worker failed while writing a reply");
        return sml_prefill(c, ids, 0, pos + 1, depth);
    }
    if (g_ext) {
        const int32_t t1 = tok;
        const float * l = ext_fwd(2, &t1, pos, 1);
        if (l) return l;
        gpu_retire("the device worker failed while writing a reply");
        return sml_prefill(c, ids, 0, pos + 1, depth);
    }
    if (g_gpu) {
        const float * l = sml_gpu_step(tok, pos);
        if (l) return l;
        gpu_retire("the GPU failed while writing a reply");
        return sml_prefill(c, ids, 0, pos + 1, depth);
    }
    return sml_forward(c, tok, pos, depth);
}


// ------------------------------------------------------------- orchestrator

// Ported from v1's own server orchestrator. Deterministic rules scored
// 20/20 on the routing probe at microsecond cost; a 3B model asked to make the
// same call scored 10/20, answering DIRECT to everything with high confidence.
static const char * LIVE_CUES[] = {
    "latest", "current", "today", "right now", "this week", "price", "weather",
    "news", "stock", "score", "http://", "https://", "www.", NULL
};
static const char * EXEC_CUES[] = {
    "what is the output", "what does this print", "print(", "```python",
    "compute", "calculate", "evaluate", NULL
};

typedef enum { ACT_DIRECT, ACT_TOOL, ACT_VERIFY } sml_action;

static int mentions(const char * hay, const char * needle) {
    const size_t n = strlen(needle);
    for (const char * p = hay; (p = strstr(p, needle)) != NULL; p++) {
        // word-boundary aware, so "score" does not fire inside "scoreboard"
        const int lb = (p == hay) || !isalnum((unsigned char) p[-1]);
        const int rb = !isalnum((unsigned char) p[n]);
        if (lb && rb) return 1;
    }
    return 0;
}

static sml_action orchestrate(const char * lower, const char ** why) {
    for (int i = 0; EXEC_CUES[i]; i++)
        if (strstr(lower, EXEC_CUES[i])) { *why = "computable ground truth"; return ACT_VERIFY; }
    for (int i = 0; LIVE_CUES[i]; i++)
        if (mentions(lower, LIVE_CUES[i])) { *why = "needs live information"; return ACT_TOOL; }
    *why = "answerable from the model";
    return ACT_DIRECT;
}

// ------------------------------------------------------------ string builder

// Replies now carry tool calls and conversations run to 8K tokens, so the
// fixed buffers that sized the old answers by guess are gone from the reply path.
typedef struct { char * p; size_t n, cap; } sbuf;

static int sb_room(sbuf * b, size_t more) {
    if (b->n + more + 1 <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->n + more + 1) cap *= 2;
    char * p = (char *) realloc(b->p, cap);
    if (!p) return 0;
    b->p = p;
    b->cap = cap;
    return 1;
}

static void sb_put(sbuf * b, const char * s, size_t n) {
    if (!sb_room(b, n)) return;
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void sb_str(sbuf * b, const char * s) { sb_put(b, s, strlen(s)); }

static void sb_esc(sbuf * b, const char * s, size_t n) {
    if (!sb_room(b, 6 * n + 16)) return;
    b->n += sml_json_escape(s, n, b->p + b->n, b->cap - b->n);
}

static void sb_fmt(sbuf * b, const char * fmt, ...) {
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    const int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (k > 0) sb_put(b, tmp, (size_t) k < sizeof tmp ? (size_t) k : sizeof tmp - 1);
}

// ----------------------------------------------------------- chat template

// The template is decided from the VOCAB, not assumed. Wrapping a model in
// another family markers tokenizes them as ordinary text: the model echoes
// them, there is no stop token to end the turn, and the whole thing reads as a
// broken engine. That is exactly what pythia did through this server until
// someone actually called it, and what Llama-3.2 did here -- it answered, then
// invented its own next question and looped, because plain text gives it no
// turn to end. Both families tokenize their markers as single tokens
// (<|im_start|> 151644 in Qwen, <|start_header_id|> 128006 in Llama-3), which
// is what makes a fixed string enough and a template engine unnecessary.
enum { TMPL_PLAIN = 0, TMPL_CHATML = 1, TMPL_LLAMA3 = 2 };

static const char * role_of(sml_json m, char * role, size_t max) {
    sml_json_str(sml_json_get(m.p, m.n, "role"), role, max);
    if (!role[0]) snprintf(role, max, "user");
    return role;
}

// A message's text. OpenAI clients send a string, an array of parts
// ({"type":"text","text":...}), or null for an assistant turn that only called
// tools. All three arrive here as plain text.
static void msg_text(sml_json m, sbuf * out) {
    out->n = 0;
    sb_put(out, "", 0);
    const sml_json c = sml_json_get(m.p, m.n, "content");
    if (c.kind == 's') {
        if (!sb_room(out, c.n)) return;
        out->n = sml_json_str(c, out->p, c.n + 1);
        return;
    }
    if (c.kind != 'a') return;
    for (int i = 0; ; i++) {
        const sml_json part = sml_json_at(c.p, c.n, i);
        if (part.kind == 0) break;
        const sml_json t = sml_json_get(part.p, part.n, "text");
        if (t.kind != 's') continue;
        if (out->n) sb_put(out, "\n", 1);
        if (!sb_room(out, t.n)) return;
        out->n += sml_json_str(t, out->p + out->n, t.n + 1);
    }
}

static int has_tools(sml_json tools) {
    return tools.kind == 'a' && sml_json_at(tools.p, tools.n, 0).kind == 'o';
}

// Each family was trained on one way of being shown tools. Qwen (ChatML) reads
// them inside <tools></tools> in the system turn and answers in <tool_call>
// tags; Llama 3.x reads them in the first user turn and answers with a bare
// {"name", "parameters"} object. Showing a model the other family's convention
// is the tool-calling version of the wrong chat template.
static void tools_block(sbuf * o, sml_json tools, int tmpl) {
    if (tmpl == TMPL_LLAMA3) {
        sb_str(o, "Given the following functions, please respond with a JSON for a function call "
                  "with its proper arguments that best answers the given prompt.\n\n"
                  "Respond in the format {\"name\": function name, \"parameters\": dictionary of "
                  "argument name and its value}. Do not use variables.\n\n");
        for (int i = 0; ; i++) {
            const sml_json t = sml_json_at(tools.p, tools.n, i);
            if (t.kind == 0) break;
            sb_put(o, t.p, t.n);
            sb_str(o, "\n\n");
        }
        return;
    }
    sb_str(o, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    for (int i = 0; ; i++) {
        const sml_json t = sml_json_at(tools.p, tools.n, i);
        if (t.kind == 0) break;
        sb_str(o, "\n");
        sb_put(o, t.p, t.n);
    }
    sb_str(o, "\n</tools>\n\nFor each function call, return a json object with function name and "
              "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
              "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
}

// An earlier assistant turn's calls, written back the way the model wrote them.
static void render_calls(sbuf * o, sml_json calls, int tmpl, int after_text) {
    for (int i = 0; ; i++) {
        const sml_json c = sml_json_at(calls.p, calls.n, i);
        if (c.kind == 0) break;
        const sml_json fn = sml_json_get(c.p, c.n, "function");
        const sml_json nm = sml_json_get(fn.p, fn.n, "name");
        const sml_json ar = sml_json_get(fn.p, fn.n, "arguments");
        char name[256];
        sml_json_str(nm, name, sizeof name);
        sbuf args = { 0 };
        if (ar.kind == 's') {                        // OpenAI sends arguments as a JSON string
            if (sb_room(&args, ar.n)) args.n = sml_json_str(ar, args.p, ar.n + 1);
        } else if (ar.kind == 'o') {
            sb_put(&args, ar.p, ar.n);
        }
        if (!args.n) sb_str(&args, "{}");
        if (tmpl == TMPL_LLAMA3) {
            if (i) sb_str(o, "; ");
            sb_fmt(o, "{\"name\": \"%s\", \"parameters\": ", name);
        } else {
            if (i || after_text) sb_str(o, "\n");
            sb_fmt(o, "<tool_call>\n{\"name\": \"%s\", \"arguments\": ", name);
        }
        sb_put(o, args.p, args.n);
        sb_str(o, tmpl == TMPL_LLAMA3 ? "}" : "}\n</tool_call>");
        free(args.p);
    }
}

// `think_off` closes Qwen3's thinking block before it opens: its own template
// does exactly this for enable_thinking=false. An agent turn that thinks first
// spends hundreds of tokens before its first tool call.
static size_t build_prompt(const char * body, size_t blen, sbuf * out, int tmpl,
                           const char * fact, int think_off) {
    const sml_json msgs = sml_json_get(body, blen, "messages");
    const sml_json tools = sml_json_get(body, blen, "tools");
    const int nt = has_tools(tools);
    out->n = 0;
    sb_put(out, "", 0);
    if (msgs.kind != 'a' || sml_json_at(msgs.p, msgs.n, 0).kind == 0) return 0;
    sbuf content = { 0 };
    char role[32], prev[32] = "", next[32];
    if (tmpl == TMPL_LLAMA3) sb_str(out, "<|begin_of_text|>");

    // A tool list with no system message still needs a system turn to carry it.
    const sml_json m0 = sml_json_at(msgs.p, msgs.n, 0);
    const int sys0 = !strcmp(role_of(m0, role, sizeof role), "system");
    if (nt && !sys0 && tmpl == TMPL_CHATML) {
        sb_str(out, "<|im_start|>system\n");
        tools_block(out, tools, tmpl);
        sb_str(out, "<|im_end|>\n");
    } else if (nt && tmpl == TMPL_PLAIN) {
        tools_block(out, tools, tmpl);
        sb_str(out, "\n");
    }
    int tools_given = 0;
    for (int i = 0; ; i++) {
        const sml_json m = sml_json_at(msgs.p, msgs.n, i);
        if (m.kind == 0) break;
        role_of(m, role, sizeof role);
        const sml_json mn = sml_json_at(msgs.p, msgs.n, i + 1);
        next[0] = 0;
        if (mn.kind) role_of(mn, next, sizeof next);
        msg_text(m, &content);
        const char * text = content.p ? content.p : "";
        const sml_json calls = sml_json_get(m.p, m.n, "tool_calls");
        const int is_tool = !strcmp(role, "tool");
        if (tmpl == TMPL_CHATML) {
            if (is_tool) {
                // Qwen's own convention: a run of tool results shares ONE user turn
                if (strcmp(prev, "tool")) sb_str(out, "<|im_start|>user");
                sb_str(out, "\n<tool_response>\n");
                sb_str(out, text);
                sb_str(out, "\n</tool_response>");
                if (strcmp(next, "tool")) sb_str(out, "<|im_end|>\n");
            } else {
                sb_str(out, "<|im_start|>");
                sb_str(out, role);
                sb_str(out, "\n");
                sb_str(out, text);
                if (nt && i == 0 && sys0) {
                    if (*text) sb_str(out, "\n\n");
                    tools_block(out, tools, tmpl);
                }
                if (calls.kind == 'a') render_calls(out, calls, tmpl, *text != 0);
                sb_str(out, "<|im_end|>\n");
            }
        } else if (tmpl == TMPL_LLAMA3) {
            sb_str(out, "<|start_header_id|>");
            sb_str(out, is_tool ? "ipython" : role);
            sb_str(out, "<|end_header_id|>\n\n");
            if (nt && !tools_given && !strcmp(role, "user")) {
                tools_block(out, tools, tmpl);
                tools_given = 1;
            }
            sb_str(out, text);
            if (calls.kind == 'a') render_calls(out, calls, tmpl, 0);
            sb_str(out, "<|eot_id|>");
        } else {
            // Base model: no turn markers exist in its vocabulary, so the
            // honest prompt is the text itself and nothing else.
            if (out->n) sb_str(out, "\n");
            if (is_tool) sb_str(out, "Tool result: ");
            sb_str(out, text);
            if (calls.kind == 'a') render_calls(out, calls, tmpl, *text != 0);
        }
        snprintf(prev, sizeof prev, "%s", role);
    }
    free(content.p);
    // Ground truth goes in BEFORE the assistant turn, as a fact the model is
    // answering with rather than a correction applied afterwards. The model
    // still writes the sentence; it is simply no longer guessing the number.
    if (fact && *fact) {
        if (tmpl == TMPL_CHATML) {
            sb_str(out, "<|im_start|>system\n");
            sb_str(out, fact);
            sb_str(out, "<|im_end|>\n");
        } else if (tmpl == TMPL_LLAMA3) {
            sb_str(out, "<|start_header_id|>system<|end_header_id|>\n\n");
            sb_str(out, fact);
            sb_str(out, "<|eot_id|>");
        } else {
            sb_str(out, "\n");
            sb_str(out, fact);
            sb_str(out, "\n");
        }
    }
    if (tmpl == TMPL_CHATML) {
        sb_str(out, "<|im_start|>assistant\n");
        if (think_off) sb_str(out, "<think>\n\n</think>\n\n");
    } else if (tmpl == TMPL_LLAMA3) {
        sb_str(out, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    return out->n;
}

// -------------------------------------------------------------- tool calls

typedef struct {
    sbuf name, args;      // unescaped name; arguments as raw JSON text
} tcall;

// End of the JSON value starting at p (an object here), honouring strings.
static const char * json_end(const char * p, const char * e) {
    int d = 0;
    for (; p < e; p++) {
        if (*p == '"') {
            for (p++; p < e && *p != '"'; p += (*p == '\\') ? 2 : 1) { }
            continue;
        }
        if (*p == '{' || *p == '[') d++;
        else if ((*p == '}' || *p == ']') && --d == 0) return p + 1;
    }
    return NULL;
}

static int tool_named(sml_json tools, const char * name) {
    for (int i = 0; ; i++) {
        const sml_json t = sml_json_at(tools.p, tools.n, i);
        if (t.kind == 0) return 0;
        const sml_json fn = sml_json_get(t.p, t.n, "function");
        char nm[256];
        sml_json_str(sml_json_get(fn.p, fn.n, "name"), nm, sizeof nm);
        if (!strcmp(nm, name)) return 1;
    }
}

// One {"name", "arguments"|"parameters"} object into a call. Returns 1 if it is one.
static int take_call(const char * p, size_t n, tcall * c) {
    const sml_json nm = sml_json_get(p, n, "name");
    if (nm.kind != 's' || !nm.n) return 0;
    sml_json ar = sml_json_get(p, n, "arguments");
    if (ar.kind == 0) ar = sml_json_get(p, n, "parameters");
    memset(c, 0, sizeof *c);
    if (sb_room(&c->name, nm.n)) c->name.n = sml_json_str(nm, c->name.p, nm.n + 1);
    if (ar.kind == 'o') sb_put(&c->args, ar.p, ar.n);
    else if (ar.kind == 's' && sb_room(&c->args, ar.n)) c->args.n = sml_json_str(ar, c->args.p, ar.n + 1);
    if (!c->args.n) sb_str(&c->args, "{}");
    return 1;
}

// Split what the model wrote into prose, thinking, and tool calls.
//
// <tool_call> blocks are the ChatML convention and are taken whatever the name,
// so a call to a tool that does not exist reaches the agent, which can say so
// and let the model correct itself. A bare JSON object -- Llama's convention, and
// what small models of every family fall back to -- is taken only when its name
// is one of the offered tools, because an answer that merely contains JSON is
// not a call.
static int parse_reply(const char * text, size_t n, sml_json tools, int want_calls,
                       sbuf * content, sbuf * think, tcall * calls) {
    const char * p = text, * e = text + n;
    content->n = 0; sb_put(content, "", 0);
    think->n = 0;   sb_put(think, "", 0);
    int nc = 0;
    // thinking first: Qwen3 writes it before anything else
    const char * ts = strstr(p, "<think>");
    if (ts) {
        const char * te = strstr(ts, "</think>");
        sb_put(content, p, (size_t) (ts - p));
        const char * body = ts + 7;
        sb_put(think, body, (size_t) ((te ? te : e) - body));
        p = te ? te + 8 : e;
    }
    while (want_calls && p < e && nc < CALLS_MAX) {
        const char * s = strstr(p, "<tool_call>");
        if (!s) break;
        sb_put(content, p, (size_t) (s - p));
        const char * be = strstr(s, "</tool_call>");
        const char * ob = s + 11;
        const char * oe = be ? be : e;
        while (ob < oe && *ob != '{') ob++;
        const char * oj = ob < oe ? json_end(ob, oe) : NULL;
        if (oj && take_call(ob, (size_t) (oj - ob), &calls[nc])) nc++;
        p = be ? be + 12 : e;
    }
    if (want_calls && !nc) {
        // NVIDIA Nemotron's own form: <TOOLCALL>[{"name": ..., "arguments": {...}}, ...]</TOOLCALL>.
        // Llama-3.1-Nemotron-Nano-4B called the right tool this way and was graded wrong (2026-10-21)
        const char * s = strstr(p, "<TOOLCALL>");
        if (s) {
            const char * be = strstr(s, "</TOOLCALL>");
            const char * oe = be ? be : e;
            const char * q = s + 10;
            while (q < oe && *q != '[' && *q != '{') q++;
            if (q < oe && *q == '[') q++;
            while (q < oe && nc < CALLS_MAX) {
                while (q < oe && (isspace((unsigned char) *q) || *q == ',')) q++;
                if (q >= oe || *q != '{') break;
                const char * oj = json_end(q, oe);
                if (!oj) break;
                if (take_call(q, (size_t) (oj - q), &calls[nc])) nc++;
                q = oj;
            }
            if (nc) {
                sb_put(content, p, (size_t) (s - p));
                p = be ? be + 11 : e;
            }
        }
    }
    if (want_calls && !nc) {
        // bare JSON: after optional whitespace, Llama's <|python_tag|>, or a ``` fence
        const char * q = p;
        while (q < e && isspace((unsigned char) *q)) q++;
        if (!strncmp(q, "<|python_tag|>", 14)) q += 14;
        if (!strncmp(q, "```", 3)) { q += 3; while (q < e && *q != '\n' && *q != '{') q++; }
        while (q < e && isspace((unsigned char) *q)) q++;
        while (q < e && *q == '{' && nc < CALLS_MAX) {
            const char * oj = json_end(q, e);
            tcall c;
            if (!oj || !take_call(q, (size_t) (oj - q), &c)) break;
            if (!tool_named(tools, c.name.p)) { free(c.name.p); free(c.args.p); break; }
            calls[nc++] = c;
            q = oj;
            while (q < e && (isspace((unsigned char) *q) || *q == ';' || *q == ',' || *q == '`')) q++;
        }
        if (nc) p = q;
    }
    sb_put(content, p, (size_t) (e - p));
    // trim the prose: tags leave blank lines behind
    size_t a = 0, b = content->n;
    while (a < b && isspace((unsigned char) content->p[a])) a++;
    while (b > a && isspace((unsigned char) content->p[b - 1])) b--;
    memmove(content->p, content->p + a, b - a);
    content->n = b - a;
    content->p[content->n] = 0;
    return nc;
}

// ------------------------------------------------------------------ serving

static void send_all(sock_t s, const char * p, size_t n) {
    while (n) {
        const int k = send(s, p, (int) n, 0);
        if (k <= 0) return;
        p += k;
        n -= (size_t) k;
    }
}

// Wall clock. clock() sums every OpenMP thread and would report six times
// the truth for a six-thread decode.
static double now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return 1000.0 * (double) t.QuadPart / (double) f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1e6;
#endif
}

static void respond(sock_t s, int code, const char * ctype, const char * body) {
    char head[512];
    const int n = snprintf(head, sizeof head,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n",
        code, code == 200 ? "OK" : "Error", ctype, strlen(body));
    send_all(s, head, (size_t) n);
    send_all(s, body, strlen(body));
}

// Our interface, read from disk per request so it can be edited while the
// server runs -- which is what you want the evening before a demo.
static const char * g_ui = NULL;

static void serve_ui(sock_t s) {
    const char * cands[3] = { g_ui, "ui/index.html", "sml/ui/index.html" };
    for (int i = 0; i < 3; i++) {
        if (!cands[i]) continue;
        FILE * f = fopen(cands[i], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        const long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        char * buf = n > 0 ? (char *) malloc((size_t) n + 1) : NULL;
        if (!buf) { fclose(f); break; }
        const size_t got = fread(buf, 1, (size_t) n, f);
        fclose(f);
        buf[got] = 0;
        respond(s, 200, "text/html; charset=utf-8", buf);
        free(buf);
        return;
    }
    respond(s, 404, "text/plain", "ui/index.html not found -- pass --ui PATH");
}

// A spoken reply does not need the last token before it can start: the first
// sentence is enough. With "stream": true each piece leaves as it is decoded,
// which is worth about a second per turn out loud.
static void send_sse_head(sock_t s) {
    const char * h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     "Cache-Control: no-cache\r\n"
                     "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    send_all(s, h, strlen(h));
}

static void send_sse_piece(sock_t s, const char * piece, size_t n) {
    sbuf b = { 0 };
    sb_str(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
    sb_esc(&b, piece, n);
    sb_str(&b, "\"}}]}\n\n");
    if (b.p) send_all(s, b.p, b.n);
    free(b.p);
}

// A whole answer as SSE, cut at UTF-8 character boundaries so no piece ends
// inside a multi-byte character.
static void send_sse_text(sock_t s, const char * text, size_t n) {
    while (n) {
        size_t k = n < 1000 ? n : 1000;
        while (k < n && k > 0 && ((unsigned char) text[k] & 0xC0) == 0x80) k--;
        send_sse_piece(s, text, k);
        text += k;
        n -= k;
    }
}

// Every reply leaves through here: live-streamed, streamed after the fact
// (tool turns, cache hits, exact arithmetic), or as one JSON body. `streamed`
// says the prose already went out piece by piece.
static void send_reply(sock_t s, int stream, int streamed, const char * text, size_t tw,
                       sml_json tools, int finish_len, int n_prompt, int n_gen,
                       const char * sml_extra) {
    sbuf content = { 0 }, think = { 0 }, b = { 0 };
    tcall calls[CALLS_MAX];
    const int nc = parse_reply(text, tw, tools, has_tools(tools), &content, &think, calls);
    const char * finish = nc ? "tool_calls" : finish_len ? "length" : "stop";
    if (stream) {
        if (!streamed && content.n) send_sse_text(s, content.p, content.n);
        if (nc) {
            sb_str(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[");
            for (int i = 0; i < nc; i++) {
                sb_fmt(&b, "%s{\"index\":%d,\"id\":\"call_%d\",\"type\":\"function\",\"function\":{\"name\":\"",
                       i ? "," : "", i, i);
                sb_esc(&b, calls[i].name.p, calls[i].name.n);
                sb_str(&b, "\",\"arguments\":\"");
                sb_esc(&b, calls[i].args.p, calls[i].args.n);
                sb_str(&b, "\"}}");
            }
            sb_str(&b, "]}}]}\n\n");
        }
        sb_fmt(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n", finish);
        // the counts and the engine's own notes (the guard, reuse) ride at the end,
        // so a streamed reply loses nothing a whole one would carry
        sb_fmt(&b, "data: {\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}", n_prompt, n_gen);
        if (sml_extra && *sml_extra) { sb_str(&b, ",\"sml\":"); sb_str(&b, sml_extra); }
        sb_str(&b, "}\n\n");
        sb_str(&b, "data: [DONE]\n\n");
        send_all(s, b.p, b.n);
    } else {
        sb_str(&b, "{\"id\":\"sml\",\"object\":\"chat.completion\",\"model\":\"sml\","
                   "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (nc && !content.n) sb_str(&b, "null");
        else { sb_str(&b, "\""); sb_esc(&b, content.p, content.n); sb_str(&b, "\""); }
        if (think.n) { sb_str(&b, ",\"reasoning_content\":\""); sb_esc(&b, think.p, think.n); sb_str(&b, "\""); }
        if (nc) {
            sb_str(&b, ",\"tool_calls\":[");
            for (int i = 0; i < nc; i++) {
                sb_fmt(&b, "%s{\"id\":\"call_%d\",\"type\":\"function\",\"function\":{\"name\":\"",
                       i ? "," : "", i);
                sb_esc(&b, calls[i].name.p, calls[i].name.n);
                sb_str(&b, "\",\"arguments\":\"");
                sb_esc(&b, calls[i].args.p, calls[i].args.n);
                sb_str(&b, "\"}}");
            }
            sb_str(&b, "]");
        }
        sb_fmt(&b, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}",
               finish, n_prompt, n_gen);
        if (sml_extra && *sml_extra) { sb_str(&b, ",\"sml\":"); sb_str(&b, sml_extra); }
        sb_str(&b, "}");
        respond(s, 200, "application/json", b.p ? b.p : "{}");
    }
    for (int i = 0; i < nc; i++) { free(calls[i].name.p); free(calls[i].args.p); }
    free(content.p); free(think.p); free(b.p);
}

typedef struct {
    // Several conversations each keep their own KV cache: an agent's tool loop
    // and a chat beside it no longer evict each other's prompt. The weights --
    // and decode's repacked copy of them -- are shared; a slot costs its KV only.
    sml_ctx   * slot[SLOTS_MAX];
    long long   used[SLOTS_MAX];
    int         n_slots;
    long long   clock;
    sml_vocab * vocab;
    int         depth;
    int         orchestrator;
    int32_t     im_end;   // turn terminator; distinct from the file's eos
    int32_t     eom;      // Llama's end-of-message after a tool call, -1 if none
    int         tmpl;     // TMPL_PLAIN, TMPL_CHATML or TMPL_LLAMA3
    int         can_think_off;   // the vocab has <think>: Qwen3's switch works
    // The model's OWN chat frame, from the SSF's "chat" block (the builder reads
    // it from the checkpoint): its template for the backend to render, and every
    // end-of-turn id. STANDARDS.md section 7.
    const char * chat_json;   size_t chat_n;
    int32_t     stop_ids[16]; int n_stop_ids;
    long        degenerate;   // replies the guard cut off

    // What /stats reports. These are the engine's own counters -- bytes it
    // actually moved, tokens the cache actually reused -- not a stopwatch
    // wrapped around the request, which would measure the socket as much as
    // the model.
    const char * model;
    int         threads, ctx_n;
    double      l_prompt_ms, l_gen_ms, l_bpt;
    double      l_margin, l_margin_min;   // top-2 logit gap: mean and worst over the turn
    int         l_cached;                 // last turn came from the response cache
    int         l_prompt, l_gen, l_reused, l_slot, l_calls;
    long        turns;
    long long   tokens_total;
} server_state;

// Qwen2.5 ends an assistant turn with <|im_end|>, which is not the same token as
// the vocab's eos. Stopping only on eos let the model run past the turn boundary
// and start hallucinating the next one.
static int32_t find_token(const sml_vocab * v, const char * lit) {
    const size_t n = strlen(lit);
    for (uint32_t id = 0; id < v->n_tokens; id++) {
        const uint32_t a = v->offsets[id], b = v->offsets[id + 1];
        if (b - a == n && memcmp(v->bytes + a, lit, n) == 0) return (int32_t) id;
    }
    return -1;
}

// Response cache. Prefix reuse skips recomputing a shared prefix; this skips
// the model entirely when the same question comes back.
//
// It is the only lever in this project with no ceiling. Decode throughput is
// bounded by DRAM bandwidth and the engine already sits at the machine's limit,
// so per-token work cannot improve much. A cache hit moves zero weight bytes
// and takes no forward pass at all, which is not a faster answer -- it is the
// absence of the work.
//
// Exact match only. A near-miss returning the wrong answer would be far worse
// than a miss, and "these two prompts mean the same thing" is a judgement this
// layer has no business making. Sampled replies are not cached: the same
// prompt is SUPPOSED to give a different answer.
#define RCACHE_N 128

typedef struct {
    uint64_t key;
    uint32_t hits;
    int      n_prompt, n_gen, finish_len;
    char     text[4096];
} rcache_entry;

static rcache_entry g_rcache[RCACHE_N];
static uint64_t g_rc_hits, g_rc_misses;

static uint64_t rc_hash(const char * p, size_t n, long want, int depth, const char * stops) {
    uint64_t h = 1469598103934665603ULL;          // FNV-1a
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char) p[i];
        h *= 1099511628211ULL;
    }
    for (const char * q = stops; *q; q++) {
        h ^= (unsigned char) *q;
        h *= 1099511628211ULL;
    }
    // Sampling settings change the answer, so they are part of the identity.
    h ^= (uint64_t) want * 2654435761ULL;
    h *= 1099511628211ULL;
    h ^= (uint64_t) depth * 40503ULL;
    return h ? h : 1;                              // 0 marks an empty slot
}

// ----------------------------------------------------------------- sampling

static uint64_t g_rng = 88172645463325252ULL;
static double rng_unit(void) {                     // xorshift64*
    g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
    return (double) ((g_rng * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
}

// Temperature and nucleus sampling over the 64 likeliest tokens. Past the top
// 64, a vocab of 150K holds nothing a top_p below 1 would keep, and a full sort
// per token would cost more than the rest of the sampling put together.
#define TOPK 64
static int sample(const float * logits, int n, double temp, double top_p) {
    if (temp <= 0.0) return sml_sample_greedy(logits, n);
    int id[TOPK];
    float lv[TOPK];
    int k = 0;
    for (int i = 0; i < n; i++) {
        const float x = logits[i];
        if (k < TOPK) {
            int j = k++;
            while (j > 0 && lv[j - 1] < x) { lv[j] = lv[j - 1]; id[j] = id[j - 1]; j--; }
            lv[j] = x; id[j] = i;
        } else if (x > lv[TOPK - 1]) {
            int j = TOPK - 1;
            while (j > 0 && lv[j - 1] < x) { lv[j] = lv[j - 1]; id[j] = id[j - 1]; j--; }
            lv[j] = x; id[j] = i;
        }
    }
    double p[TOPK], sum = 0.0;
    for (int i = 0; i < k; i++) { p[i] = exp((lv[i] - lv[0]) / temp); sum += p[i]; }
    int keep = k;
    if (top_p > 0.0 && top_p < 1.0) {
        double acc = 0.0;
        for (int i = 0; i < k; i++) { acc += p[i] / sum; if (acc >= top_p) { keep = i + 1; break; } }
    }
    double tot = 0.0;
    for (int i = 0; i < keep; i++) tot += p[i];
    double r = rng_unit() * tot;
    for (int i = 0; i < keep; i++) { r -= p[i]; if (r <= 0.0) return id[i]; }
    return id[keep - 1];
}

// "stop": a string or up to four strings; the reply ends before the first one.
static int read_stops(const char * body, size_t blen, char stops[STOPS_MAX][64], char * flat, size_t fmax) {
    const sml_json sv = sml_json_get(body, blen, "stop");
    int ns = 0;
    flat[0] = 0;
    if (sv.kind == 's') {
        sml_json_str(sv, stops[0], 64);
        if (stops[0][0]) ns = 1;
    } else if (sv.kind == 'a') {
        for (int i = 0; ns < STOPS_MAX; i++) {
            const sml_json x = sml_json_at(sv.p, sv.n, i);
            if (x.kind == 0) break;
            sml_json_str(x, stops[ns], 64);
            if (stops[ns][0]) ns++;
        }
    }
    for (int i = 0; i < ns; i++) {
        strncat(flat, stops[i], fmax - strlen(flat) - 1);
        strncat(flat, "\x1f", fmax - strlen(flat) - 1);
    }
    return ns;
}

static int think_off_requested(const char * body, size_t blen) {
    sml_json v = sml_json_get(body, blen, "enable_thinking");
    if (v.kind == 0) {
        const sml_json k = sml_json_get(body, blen, "chat_template_kwargs");
        if (k.kind == 'o') v = sml_json_get(k.p, k.n, "enable_thinking");
    }
    return v.kind == 'b' && v.p[0] == 'f';
}

// The degeneration guard: a reply that has fallen into a loop or into noise is
// cut off rather than shown. Trips on (1) one token repeated 16 times running,
// (2) a block of 2-32 tokens repeated back to back over at least 32 tokens
// (and 4 times at the least): a short block needs more repeats, so JSON arrays
// and table rules, which repeat a few tokens a few times, are not loops. The
// first rule (4 repeats of any block in a reply of 24+ characters) cut a
// Llama 3.2 3B tool call on its first agent run, 2026-09-30. Or
// (3) 48 bytes of punctuation, symbols with no letter or digit.
// Returns a reason, or NULL.
static const char * degenerate(const int32_t * ids, int n, const char * text, size_t tw, int asked_repeat) {
    if (n >= 16 && !asked_repeat) {
        int same = 1;
        for (int i = n - 16; i < n - 1; i++) same &= ids[i] == ids[n - 1];
        if (same) return "one token repeated";
    }
    for (int per = 2; per <= 32 && !asked_repeat; per++) {
        const int need = per >= 8 ? 4 : (32 + per - 1) / per;
        if (per * need > n) continue;
        int rep = 1;
        for (int i = n - 2 * per; i >= 0 && rep < need; i -= per) {
            if (memcmp(ids + i, ids + n - per, (size_t) per * sizeof(int32_t))) break;
            rep++;
        }
        if (rep >= need) return "a looping block";
    }
    if (tw >= 48) {
        int noise = 1;
        for (size_t i = tw - 48; i < tw && noise; i++) {
            const unsigned char ch = (unsigned char) text[i];
            if ((ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch >= 0xC0)
                noise = 0;
        }
        if (noise) return "punctuation noise";
    }
    return NULL;
}

// The batched prompt path is sound when its reply starts as reading a token at a time does (the first
// 4 words) and does not get stuck: no word more than 4 times in the n, and at least 5 different ones.
static int prompt_path_sound(const int * want, const int * got, int n) {
    for (int k = 0; k < 4 && k < n; k++) if (want[k] != got[k]) return 0;
    int distinct = 0;
    for (int i = 0; i < n; i++) {
        int seen = 0, same = 0;
        for (int j = 0; j < n; j++) {
            if (got[j] == got[i]) same++;
            if (j < i && got[j] == got[i]) seen = 1;
        }
        if (same > 4) return 0;
        if (!seen) distinct++;
    }
    return distinct >= 5;
}

// The prompt check's verdict is the same on every start of the same file, engine build and layer
// split, and it cost ~1.5 s of every GPU start (3.9 s against llama.cpp's 2.4, 2026-10-27). So it is
// kept beside the model, keyed by all three; anything different checks again.
static void prompt_key(char * key, int n, const char * path, int layers, int of) {
    long long size = 0, mtime = 0;
    FILE * f = fopen(path, "rb");
    if (f) {
#ifdef _WIN32
        _fseeki64(f, 0, SEEK_END); size = _ftelli64(f);
        struct __stat64 st; if (_stat64(path, &st) == 0) mtime = (long long) st.st_mtime;
#else
        fseeko(f, 0, SEEK_END); size = (long long) ftello(f);
#endif
        fclose(f);
    }
    // the kernel settings an engine tune may set change which prompt path is sound, so they are in the key
    extern char ** environ;
    unsigned h = 2166136261u;
    for (char ** e = environ; e && *e; e++)
        if (!strncmp(*e, "SML_", 4) && strncmp(*e, "SML_PROMPT_RECHECK", 18))
            for (const char * c = *e; *c; c++) h = (h ^ (unsigned char) *c) * 16777619u;
    snprintf(key, n, "%s %s|%lld|%lld|%d/%d|%08x", __DATE__, __TIME__, size, mtime, layers, of, h);
}

static int prompt_cached(const char * path, const char * key) {
    char p[1100], line[300];
    snprintf(p, sizeof p, "%s.prompt", path);
    FILE * f = fopen(p, "r");
    if (!f) return -1;
    int lvl = -1;
    if (fgets(line, sizeof line, f)) {
        char * bar = strrchr(line, ' ');
        if (bar && !strncmp(line, key, (size_t) (bar - line)) && strlen(key) == (size_t) (bar - line))
            lvl = atoi(bar + 1);
    }
    fclose(f);
    return lvl >= 0 && lvl <= 2 ? lvl : -1;
}

static void prompt_remember(const char * path, const char * key, int lvl) {
    char p[1100];
    snprintf(p, sizeof p, "%s.prompt", path);
    FILE * f = fopen(p, "w");
    if (f) { fprintf(f, "%s %d\n", key, lvl); fclose(f); }
}

static void pace_rest(int ms) {
#if defined(_WIN32)
    Sleep((DWORD) ms);
#else
    struct timespec ts = { ms / 1000, (long) (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
#endif
}

static void handle_chat(server_state * st, sock_t s, const char * body, size_t blen) {
    const sml_json tools = sml_json_get(body, blen, "tools");
    const int nt = has_tools(tools);
    const int think_off = st->can_think_off && think_off_requested(body, blen);
    sbuf prompt = { 0 };
    // "prompt": rendered by the backend with the model's OWN template (the
    // file's chat block); taken as is. Without it the built-in frames apply.
    const sml_json rendered = sml_json_get(body, blen, "prompt");
    size_t plen = 0;
    if (rendered.kind == 's' && rendered.n > 0) {
        if (sb_room(&prompt, rendered.n + 1)) {
            plen = sml_json_str(rendered, prompt.p, rendered.n + 1);
            prompt.n = plen;
        }
    } else {
        plen = build_prompt(body, blen, &prompt, st->tmpl, NULL, think_off);
    }
    if (!plen) {
        free(prompt.p);
        respond(s, 400, "application/json", "{\"error\":\"no messages\"}");
        return;
    }
    const sml_json sv = sml_json_get(body, blen, "stream");
    const int stream = (sv.kind == 0x62 && sv.n >= 4 && sv.p[0] == 0x74);   // b, t

    // The orchestrator answers the person, so it reads the last USER turn and
    // acts only when that turn is the one being answered. Inside an agent loop
    // the last message is a tool result: re-answering an old arithmetic
    // question there would hijack the loop.
    char turn[8192], last_role[32] = "";
    turn[0] = 0;
    {
        const sml_json msgs = sml_json_get(body, blen, "messages");
        for (int i = 0; msgs.kind == 'a'; i++) {
            const sml_json m = sml_json_at(msgs.p, msgs.n, i);
            if (m.kind == 0) break;
            role_of(m, last_role, sizeof last_role);
            if (!strcmp(last_role, "user")) {
                sbuf t = { 0 };
                msg_text(m, &t);
                snprintf(turn, sizeof turn, "%s", t.p ? t.p : "");
                free(t.p);
            }
        }
        for (char * q = turn; *q; q++) *q = (char) tolower((unsigned char) *q);
    }
    // "verify": false -- the caller's turn quotes text it did not write (web pages
    // for a cited answer): a score like "59 - 28" in a source is not a sum asked for
    const sml_json verify_opt = sml_json_get(body, blen, "verify");
    const int orch = st->orchestrator && !strcmp(last_role, "user")
                     && !(verify_opt.kind == 'b' && verify_opt.p[0] == 'f');

    char lower[8192];
    size_t lo = 0;
    for (size_t i = 0; i < plen && lo + 1 < sizeof lower; i++)
        lower[lo++] = (char) tolower((unsigned char) prompt.p[i]);
    lower[lo] = 0;
    const char * why = "";
    // what is asked NOW: the whole prompt holds earlier turns, so "Are you back?"
    // was answered "... Checked by exact computation: 17 * 23 = 391" from a
    // question three turns before (2026-10-13 simulation)
    const char * asked_now = (nt || turn[0]) ? turn : lower;
    sml_action act = orch ? orchestrate(asked_now, &why) : ACT_DIRECT;

    // ACT_VERIFY used to be a label and nothing more. Where the answer is
    // exactly computable, compute it and hand it to the model as fact. The
    // verifier declines anything it cannot evaluate exactly, and a decline
    // leaves the model's own answer untouched.
    long long vval = 0;
    int verified = 0;
    char fact[128];
    fact[0] = 0;
    // The CUE for arithmetic is that it PARSES, not that the prompt happened to
    // contain a word like "compute". Gating on EXEC_CUES first meant "What is 47
    // times 89?" -- exactly the case this is for -- never reached the verifier,
    // and it fired on 8 of 18 questions instead of all 18. The verifier already
    // declines everything it cannot evaluate exactly, so asking it IS the test.
    // A turn that is nothing but arithmetic is answered with the exact value,
    // before any model runs: no weights read, nothing for a small model to get
    // wrong on the way (sml_verify_pure says why this replaced the fact alone).
    if (orch) {
        char shown[512];
        if (sml_verify_pure(turn, &vval, shown, sizeof shown)) {
            char ans[1024];
            snprintf(ans, sizeof ans, "%s = %lld", shown, vval);
            st->turns++;
            st->l_prompt = st->l_gen = st->l_reused = st->l_calls = 0;
            st->l_prompt_ms = st->l_gen_ms = st->l_bpt = 0.0;
            st->l_margin = st->l_margin_min = -1.0;
            st->l_cached = 0;
            fprintf(stderr, "orchestrator: exact arithmetic answered directly (%s)\n", ans);
            if (stream) send_sse_head(s);
            const sml_json none = { NULL, 0, 0 };
            send_reply(s, stream, 0, ans, strlen(ans), none, 0, 0, 0,
                       "{\"action\":\"VERIFY\",\"reason\":\"exact arithmetic, computed\","
                       "\"verified\":true,\"bytes_moved\":0}");
            free(prompt.p);
            return;
        }
    }
    // A tool prompt starts with the tool schemas; scanning it whole would find
    // "arithmetic" in their descriptions. The user's turn is what is asked.
    char vexpr[512];
    vexpr[0] = 0;
    if (orch && sml_verify_arith_expr(asked_now, &vval, vexpr, sizeof vexpr)) {
        act = ACT_VERIFY;
        verified = 1;
        snprintf(fact, sizeof fact,
                 "Verified by exact computation: the value is %lld. "
                 "State this value.", vval);
        if (rendered.kind != 's') plen = build_prompt(body, blen, &prompt, st->tmpl, fact, think_off);
        why = "exact arithmetic, verified";
    }

    const int n_ctx = st->slot[0]->n_ctx;
    int32_t * ids = (int32_t *) malloc(((size_t) n_ctx + 1) * sizeof(int32_t));
    int n_ids = ids ? sml_encode(st->vocab, prompt.p, ids, n_ctx + 1) : 0;
    if (n_ids < 0 || n_ids >= n_ctx - 1) {
        // An honest refusal: silently dropping the end of an agent's transcript
        // drops the tool result it is waiting on.
        char err[160];
        snprintf(err, sizeof err, "{\"error\":\"prompt does not fit: the context holds %d tokens\"}", n_ctx);
        respond(s, 400, "application/json", err);
        free(ids); free(prompt.p);
        return;
    }
    if (n_ids == 0) {
        respond(s, 400, "application/json", "{\"error\":\"tokenize\"}");
        free(ids); free(prompt.p);
        return;
    }

    long want = sml_json_int(sml_json_get(body, blen, "max_tokens"), 128);
    if (want > GEN_MAX) want = GEN_MAX;
    const double temp = sml_json_num(sml_json_get(body, blen, "temperature"), 0.0);
    const double top_p = sml_json_num(sml_json_get(body, blen, "top_p"), 1.0);
    const sml_json seed = sml_json_get(body, blen, "seed");
    // "pace_ms": a rest after each written token, set by the backend's heat guard when the
    // machine runs hot. The reply keeps flowing, the GPU does less work per second and cools,
    // and nothing stops (2026-10-26: a hard pause at the danger line was the only tool).
    int pace_ms = (int) sml_json_num(sml_json_get(body, blen, "pace_ms"), 0.0);
    if (pace_ms < 0) pace_ms = 0;
    if (pace_ms > 250) pace_ms = 250;
    if (seed.kind == 'n') g_rng = (uint64_t) sml_json_int(seed, 1) * 2654435761ULL + 1;
    else if (temp > 0.0) g_rng ^= (uint64_t) time(NULL) * 2654435761ULL + (uint64_t) now_ms();
    char stops[STOPS_MAX][64], stop_flat[300];
    const int ns = read_stops(body, blen, stops, stop_flat, sizeof stop_flat);
    // Tool turns are parsed as a whole before anything is sent; streaming the
    // raw call text would hand the client JSON it then has to un-see.
    const int live = stream && !nt;

    const uint64_t rkey = rc_hash(prompt.p, plen, want, st->depth, stop_flat);
    rcache_entry * slot = &g_rcache[rkey % RCACHE_N];
    const sml_json cache_opt = sml_json_get(body, blen, "cache");
    const int use_rcache = !(cache_opt.kind == 'b' && cache_opt.p[0] == 'f');
    if (temp <= 0.0 && use_rcache && slot->key == rkey) {
        slot->hits++;
        g_rc_hits++;
        // A cache hit is a turn too, and it moved no weights at all. Its
        // confidence is "not measured" (-1), not the previous turn's figure:
        // a stale number here once sent the wrong turn to the cloud.
        st->turns++;
        st->l_prompt = slot->n_prompt;
        st->l_gen = slot->n_gen;
        st->l_reused = slot->n_prompt;
        st->l_prompt_ms = st->l_gen_ms = st->l_bpt = 0.0;
        st->l_margin = st->l_margin_min = -1.0;
        st->l_cached = 1;
        // A streaming client reads SSE and nothing else. Answering a cached
        // turn with a plain JSON body made every repeated question arrive
        // EMPTY through the bridge and the web interface.
        fprintf(stderr, "cache: response HIT%s (%llu hit / %llu miss)\n", stream ? ", streamed" : "",
                (unsigned long long) g_rc_hits, (unsigned long long) g_rc_misses);
        if (stream) send_sse_head(s);
        send_reply(s, stream, 0, slot->text, strlen(slot->text), tools, slot->finish_len,
                   slot->n_prompt, slot->n_gen, "{\"cached\":true,\"bytes_moved\":0}");
        free(ids); free(prompt.p);
        return;
    }
    g_rc_misses++;

    // A request that CONTINUES a slot's conversation -- shares at least half of
    // what that slot holds -- goes back to it. Anything else takes the slot used
    // longest ago. Choosing by longest shared prefix alone let conversation B,
    // sharing only "<|im_start|>user\nConversation" with A, take A's slot and
    // throw A's cache away.
    int si = -1, best = 0;
    for (int i = 0; i < st->n_slots; i++) {
        const int L = sml_cache_prefix(st->slot[i], ids, n_ids);
        if (L > 0 && 2 * L >= st->slot[i]->n_hist && L > best) { best = L; si = i; }
    }
    if (si < 0) {
        si = 0;
        for (int i = 1; i < st->n_slots; i++) if (st->used[i] < st->used[si]) si = i;
    }
    st->used[si] = ++st->clock;
    sml_ctx * c = st->slot[si];
    const float * logits = NULL;
    // The generation loop appends each new token to `ids`, so n_ids grows in
    // lockstep with `pos`. Every figure derived as (pos - n_ids) was therefore
    // structurally ZERO -- completion_tokens and bytes_per_token both reported
    // 0 on every uncached response. Snapshot the prompt length instead.
    const int n_prompt = n_ids;
    int n_gen = 0;
    // Reuse whatever leading positions the cache already holds. A chat resends
    // the whole transcript every turn, so this is the difference between
    // prefill cost growing with the conversation and it staying flat.
    int pos = sml_cache_prefix(c, ids, n_ids);
    const int reused = pos;
    const int end = n_ids;
    const double t0_read = now_ms();
    if (end > pos) {
        logits = fwd_prefill(c, ids, pos, end - pos, st->depth);
        pos = end;
    }
    // count generation traffic only; the prompt is a different access pattern
    const double t0_write = now_ms();
    const uint64_t bytes0 = c->bytes_moved;

    static char text[1 << 18];
    size_t tw = 0;
    int finish_len = 1;
    const char * degen = NULL;
    if (stream) send_sse_head(s);
    const int nv = (int) c->model->hdr->n_vocab;
    int tok = sample(logits, nv, temp, top_p);
    // How sure the model was, token by token: the gap between its top two
    // logits. A small gap means the next token was nearly a coin toss. The
    // bridge in front of this server sends a turn to a larger model when the
    // local one was unsure -- the model's own evidence, not a guess about it.
    double msum = sml_top2_margin(logits, nv), mmin = msum;
    int mcount = 1;
    for (long i = 0; i < want && pos < c->n_ctx - 1; i++, pos++) {
        int end_turn = (uint32_t) tok == st->vocab->eos || tok == st->im_end || (st->eom >= 0 && tok == st->eom);
        for (int k = 0; k < st->n_stop_ids && !end_turn; k++) end_turn = tok == st->stop_ids[k];
        if (end_turn) {
            finish_len = 0;
            break;
        }
        char piece[64];
        const int nb = sml_decode(st->vocab, tok, piece, sizeof piece);
        if (tw + (size_t) nb + 1 >= sizeof text) break;
        memcpy(text + tw, piece, (size_t) nb);
        tw += (size_t) nb;
        text[tw] = 0;
        // a stop string ends the reply before it; only the part before it is sent
        int stopped = 0;
        for (int k = 0; k < ns && !stopped; k++) {
            const size_t sl = strlen(stops[k]);
            const size_t from = tw > (size_t) nb + sl ? tw - (size_t) nb - sl : 0;
            char * hit = strstr(text + from, stops[k]);
            if (hit) {
                const size_t keep = (size_t) (hit - text);
                if (live && keep > tw - (size_t) nb) send_sse_piece(s, text + tw - nb, keep - (tw - (size_t) nb));
                tw = keep;
                text[tw] = 0;
                stopped = 1;
            }
        }
        if (stopped) { finish_len = 0; n_gen++; break; }
        if (live) send_sse_piece(s, piece, (size_t) nb);
        if (n_ids < n_ctx + 1) ids[n_ids++] = tok;
        n_gen++;
        {
            // a loop the user ASKED for ("repeat", "N times") is not a fault;
            // noise always is
            const int asked = mentions(turn, "repeat") || mentions(turn, "times");
            const char * bad = n_gen >= 16 ? degenerate(ids + n_prompt, n_gen, text, tw, asked) : NULL;
            if (bad) {
                degen = bad;
                st->degenerate++;
                {
                    // what was cut, for the orchestrator's log (the tail, one line)
                    char tail[121];
                    const size_t from = tw > 120 ? tw - 120 : 0;
                    size_t k = 0;
                    for (size_t j = from; j < tw && k < 120; j++) tail[k++] = (text[j] == '\n' || text[j] == '\r') ? ' ' : text[j];
                    tail[k] = 0;
                    fprintf(stderr, "guard: reply cut off after %d tokens (%s): ...%s\n", n_gen, bad, tail);
                }
                if (n_ids > n_prompt && ids[n_ids - 1] == tok) n_ids--;
                finish_len = 0;
                break;
            }
        }
        logits = fwd_step(c, ids, tok, pos, st->depth);
        tok = sample(logits, nv, temp, top_p);
        if (pace_ms > 0) pace_rest(pace_ms);
        const double mg = sml_top2_margin(logits, nv);
        msum += mg;
        mcount++;
        if (mg < mmin) mmin = mg;
    }
    st->l_prompt_ms = t0_write - t0_read;
    st->l_gen_ms = now_ms() - t0_write;
    st->l_prompt = n_prompt;
    st->l_gen = n_gen;
    st->l_reused = reused;
    st->l_slot = si;
    st->l_bpt = n_gen ? (double)(c->bytes_moved - bytes0) / n_gen : 0.0;
    st->l_margin = msum / mcount;
    st->l_cached = 0;
    st->l_margin_min = mmin;
    st->turns++;
    st->tokens_total += n_gen;

    // "verified" is a claim about the ANSWER. A 1B model handed the exact value
    // as a fact still worked sums out itself and stated a wrong one; when the
    // answer does not state the value, the value is added to it, so the flag
    // is never on an answer that contradicts it.
    if (verified && !sml_verify_states(text, vval)) {
        char add[640];
        const int an = snprintf(add, sizeof add, "\n\nChecked by exact computation: %s = %lld.", vexpr, vval);
        if (an > 0 && tw + (size_t) an + 1 < sizeof text) {
            memcpy(text + tw, add, (size_t) an);
            tw += (size_t) an;
            text[tw] = 0;
            if (live) send_sse_piece(s, add, (size_t) an);
            fprintf(stderr, "orchestrator: the answer did not state the verified value; appended\n");
        }
    }

    // The cache now holds the prompt AND what we just generated, so the next
    // turn of the same conversation reuses both.
    sml_cache_commit(c, ids, n_ids);
    if (temp <= 0.0 && tw < sizeof slot->text && !degen) {
        slot->key = rkey;
        slot->hits = 0;
        // tw is a BYTE count of the decoded text, not a token count; storing
        // it here made every cache hit report bytes where tokens belong.
        slot->n_prompt = n_prompt;
        slot->n_gen = n_gen;
        slot->finish_len = finish_len;
        memcpy(slot->text, text, tw);
        slot->text[tw] = 0;
    }
    fprintf(stderr, "cache: slot %d reused %d of %d tokens (%.0f%%)\n",
            si, reused, n_prompt, n_prompt ? 100.0 * reused / n_prompt : 0.0);

    char extra[512];
    snprintf(extra, sizeof extra,
             "{\"depth\":%d,\"slot\":%d,\"reused\":%d,\"bytes_per_token\":%.0f,\"action\":\"%s\","
             "\"reason\":\"%s\",\"verified\":%s,\"degenerate\":%s%s%s}",
             st->depth, si, reused, st->l_bpt,
             act == ACT_VERIFY ? "VERIFY" : act == ACT_TOOL ? "TOOL" : "DIRECT", why,
             verified ? "true" : "false", degen ? "\"" : "", degen ? degen : "false", degen ? "\"" : "");
    send_reply(s, stream, live, text, tw, tools, finish_len, n_prompt, n_gen, extra);
    free(ids);
    free(prompt.p);
}

// The request line decides the endpoint. Matching the path anywhere in the
// request sent a chat message that merely MENTIONED "/stats" to the stats page.
static int path_is(const char * req, const char * method, const char * path) {
    const size_t ml = strlen(method), pl = strlen(path);
    if (strncmp(req, method, ml) || req[ml] != ' ') return 0;
    const char * p = req + ml + 1;
    if (strncmp(p, path, pl)) return 0;
    return p[pl] == ' ' || p[pl] == '?';
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.ssf [--port N] [--threads N] "
                        "[--depth N] [--ctx N] [--slots N] [--orchestrator on|off] "
                        "[--ui PATH]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1][0] == '-' ? NULL : argv[1];
    int port = 8080, threads = 6, depth = 0, n_ctx = 2048, orch = 1, n_slots = 1;
    for (int i = path ? 2 : 1; i < argc; i++) {
        // Accept the reference engine's flag spelling too, so the accuracy
        // harness can drive this unchanged. The harness is the measuring
        // instrument -- editing it to fit a new server is how a comparison
        // quietly stops comparing the same thing.
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) i++;   // loopback only
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--jinja")) { }   // ChatML is applied always
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) snprintf(g_dev_req, sizeof g_dev_req, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--gpu-layers") && i + 1 < argc) g_gpu_layers = atoi(argv[++i]);
        // --device NAME --ext-cmd "worker command": the forward pass runs in that worker
        else if (!strcmp(argv[i], "--ext-cmd") && i + 1 < argc) snprintf(g_ext_cmd, sizeof g_ext_cmd, "%s", argv[++i]);
        // --device multi --gpu-layers K --mid-layers M --ext-cmd "worker --layers K:K+M"
        else if (!strcmp(argv[i], "--mid-layers") && i + 1 < argc) g_mid_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--slots") || !strcmp(argv[i], "-np")) && i + 1 < argc) n_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ui") && i + 1 < argc) g_ui = argv[++i];
        else if (!strcmp(argv[i], "--orchestrator") && i + 1 < argc)
            orch = strcmp(argv[++i], "off") != 0;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (n_slots < 1) n_slots = 1;
    if (n_slots > SLOTS_MAX) n_slots = SLOTS_MAX;
    // one GPU cache holds one conversation: on the card, one slot
    if (strcmp(g_dev_req, "cpu") && n_slots > 1) n_slots = 1;

#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { fprintf(stderr, "winsock\n"); return 1; }
#endif

    if (!path) { fprintf(stderr, "no model given\n"); return 2; }
    sml_model m;
    if (!sml_model_open(&m, path)) return 1;
    if ((m.hdr->flags & SML_MODEL_EXACT) && strcmp(g_dev_req, "cpu")) {
        fprintf(stderr, "exact-weight SSF uses the CPU dense path; ignoring requested device %s\n", g_dev_req);
        snprintf(g_dev_req, sizeof g_dev_req, "cpu");
        g_ext_cmd[0] = 0;
        g_gpu_layers = g_mid_layers = 0;
    }
    static sml_ctx ctxs[SLOTS_MAX];
    server_state st;
    memset(&st, 0, sizeof st);
    // on the GPU the CPU context only stands by: no 2 GB decode copy unless called on
    if (strcmp(g_dev_req, "cpu")) sml_set_repack(0);
    for (int i = 0; i < n_slots; i++) {
        if (!sml_ctx_init(&ctxs[i], &m, n_ctx, threads)) {
            if (!i) return 1;
            fprintf(stderr, "slot %d: out of memory; serving with %d\n", i, i);
            n_slots = i;
            break;
        }
        if (depth > 0) ctxs[i].depth = depth;
        st.slot[i] = &ctxs[i];
    }
    st.n_slots = n_slots;
    g_n_embd = (int) m.hdr->n_embd;
    if (!strcmp(g_dev_req, "multi") && g_ext_cmd[0] && g_mid_layers > 0 && g_gpu_layers > 0) {
        // the GPU's layers first, then the worker's, then the CPU's
        char why[200] = "";
        if (sml_gpu_open(path, n_ctx, threads, g_gpu_layers, why, (int) sizeof why)
            && ext_open(g_ext_cmd, (int) m.hdr->n_vocab, why, (int) sizeof why)) {
            g_gpu = 1;
            g_gpu_layers = sml_gpu_layers(&g_gpu_of);
            sml_gpu_set_middle(g_mid_layers, ext_mid);
            snprintf(g_dev, sizeof g_dev, "multi");
            printf("  serving on several devices: layers 0-%d on the NVIDIA GPU, %d-%d on %s, the rest on the CPU\n",
                   g_gpu_layers - 1, g_gpu_layers, g_gpu_layers + g_mid_layers - 1, g_dev_name);
            g_standby = ctxs; g_n_standby = n_slots;
        } else {
            snprintf(g_dev_why, sizeof g_dev_why, "%s", why);
            printf("  several devices are not available (%s): serving on the CPU\n", why);
            for (int i = 0; i < n_slots; i++) sml_ctx_repack(&ctxs[i]);
        }
        sml_set_repack(1);
    } else if (!strcmp(g_dev_req, "amd") && g_ext_cmd[0] && g_mid_layers > 0 && g_mid_layers < (int) m.hdr->n_layer) {
        // no NVIDIA room: the other GPUs take the first layers, the CPU the rest
        char why[200] = "";
        for (int i = 0; i < n_slots; i++) sml_ctx_repack(&ctxs[i]);
        if (ext_open(g_ext_cmd, (int) m.hdr->n_vocab, why, (int) sizeof why)) {
            g_front = 1;
            snprintf(g_dev, sizeof g_dev, "amd");
            printf("  serving on several devices: layers 0-%d on %s, the rest on the CPU\n",
                   g_mid_layers - 1, g_dev_name);
        } else {
            snprintf(g_dev_why, sizeof g_dev_why, "%s", why);
            printf("  %s is not available (%s): serving on the CPU\n", g_dev_req, why);
        }
        sml_set_repack(1);
    } else if (strcmp(g_dev_req, "cpu") && g_ext_cmd[0]) {
        char why[200] = "";
        if (ext_open(g_ext_cmd, (int) m.hdr->n_vocab, why, (int) sizeof why)) {
            g_ext = 1;
            snprintf(g_dev, sizeof g_dev, "%s", g_dev_req);
            printf("  serving on %s (%s)\n", g_dev_name, g_dev_req);
            g_standby = ctxs; g_n_standby = n_slots;
        } else {
            snprintf(g_dev_why, sizeof g_dev_why, "%s", why);
            printf("  %s is not available (%s): serving on the CPU\n", g_dev_req, why);
            for (int i = 0; i < n_slots; i++) sml_ctx_repack(&ctxs[i]);
        }
        sml_set_repack(1);
    } else if (strcmp(g_dev_req, "cpu")) {
        char why[200] = "";
        const int want = !strcmp(g_dev_req, "hybrid") && g_gpu_layers > 0 ? g_gpu_layers : (g_gpu_layers > 0 ? g_gpu_layers : -1);
        if (sml_gpu_open(path, n_ctx, threads, want, why, (int) sizeof why)) {
            g_gpu = 1;
            g_gpu_layers = sml_gpu_layers(&g_gpu_of);
            snprintf(g_dev, sizeof g_dev, "%s", g_gpu_layers < g_gpu_of ? "hybrid" : "gpu");
            printf("  serving on the GPU: %d of %d layers on the card%s\n", g_gpu_layers, g_gpu_of,
                   g_gpu_layers < g_gpu_of ? ", the rest on the CPU" : "");
        } else {
            snprintf(g_dev_why, sizeof g_dev_why, "%s", why);
            printf("  the GPU is not available (%s): serving on the CPU\n", why);
            for (int i = 0; i < n_slots; i++) sml_ctx_repack(&ctxs[i]);
        }
        if (g_gpu) { g_standby = ctxs; g_n_standby = n_slots; }
        sml_set_repack(1);
    }
    sml_vocab v;
    if (!sml_vocab_init(&v, &m)) { fprintf(stderr, "no vocab in model\n"); return 1; }

    // BOTH markers of a family must exist, or that template is not this model's.
    const int32_t im_s = find_token(&v, "<|im_start|>");
    const int32_t im_e = find_token(&v, "<|im_end|>");
    const int32_t l3_s = find_token(&v, "<|start_header_id|>");
    const int32_t l3_e = find_token(&v, "<|eot_id|>");
    int tmpl = TMPL_PLAIN;
    int32_t stop = -1;
    if (im_s >= 0 && im_e >= 0)      { tmpl = TMPL_CHATML; stop = im_e; }
    else if (l3_s >= 0 && l3_e >= 0) { tmpl = TMPL_LLAMA3; stop = l3_e; }
    st.vocab = &v; st.depth = ctxs[0].depth; st.orchestrator = orch;
    // How this model's prompts are read on the card, measured on a real turn.
    //
    // The batched prompt kernels are worth 2.9x of the 3B's prompt speed, and on
    // Qwen2.5-1.5B the tensor-core one (int8 weights, one scale per 256) left the
    // prompt's cache subtly wrong: the reply began correctly and then repeated one
    // word for ever, while every per-token grade passed (2026-10-07). The error
    // follows the activations, so it shows on a real turn in the model's own frame
    // and not on arbitrary ids -- hence here, after the template is known.
    //
    // Each level must write what reading a token at a time writes, or it is not
    // used. The probe writes the cache positions it then reads, so the slots are
    // untouched; it costs about a second.
    if (g_gpu && !g_front) {
        char probe[600];
        const char * q = "Name three rivers in Africa, one per line.";
        if (tmpl == TMPL_CHATML)
            snprintf(probe, sizeof probe, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", q);
        else if (tmpl == TMPL_LLAMA3)
            snprintf(probe, sizeof probe, "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                     "<|start_header_id|>assistant<|end_header_id|>\n\n", q);
        else
            snprintf(probe, sizeof probe, "%s\n", q);
        int pid[256];
        const int np = sml_encode(&v, probe, pid, 256);
        int want[12], got[12], lvl = 2, np_fail = 0;
        char pkey[300];
        prompt_key(pkey, sizeof pkey, path, g_gpu_layers, g_gpu_of);
        const int known = getenv("SML_PROMPT_RECHECK") ? -1 : prompt_cached(path, pkey);
        if (known >= 0) lvl = known;
        else if (np > 4) {
            for (int mode = 0; mode <= 2; mode++) {
                const int want_lvl = mode == 0 ? 0 : (mode == 1 ? 2 : 1);
                sml_gpu_prompt_mode(want_lvl);
                int * out = mode == 0 ? want : got;
                const float * lg = NULL;
                if (mode == 0) { for (int i = 0; i < np; i++) lg = sml_gpu_step(pid[i], i); }
                else lg = sml_gpu_prefill(pid, 0, np);
                for (int k = 0; k < 12 && lg; k++) {
                    out[k] = sml_sample_greedy(lg, (int) m.hdr->n_vocab);
                    lg = sml_gpu_step(out[k], np + k);
                }
                if (!lg) { lvl = 0; np_fail = 1; break; }
                if (mode == 0) continue;
                // Word for word over 12 tokens was too strict: rounding in the batched sums can flip
                // one late greedy choice, and the 3B then read every prompt a token at a time on the
                // card (13.8 tok/s against llama.cpp's 380, 2026-10-27). What the check is for is a
                // broken cache, which starts right and then repeats one word for ever: so the first
                // 4 words must match and the 12 must not be stuck on one word.
                if (!memcmp(want, got, sizeof want) || prompt_path_sound(want, got, 12)) { lvl = want_lvl; break; }
                lvl = 0;
            }
            if (!np_fail) prompt_remember(path, pkey, lvl);
        }
        sml_gpu_prompt_mode(lvl);
        if (sml_gpu_decode_cpu(1)) printf("  answers: written on the CPU, prompts read on the card\n");
        if (lvl < 2)
            printf("  prompts: %s does not write this model's own words; reading them %s\n",
                   lvl == 1 ? "the tensor-core prompt kernel" : "the batched prompt path",
                   lvl == 1 ? "without it" : "a token at a time");
    }
    st.im_end = stop; st.tmpl = tmpl;
    st.eom = tmpl == TMPL_LLAMA3 ? find_token(&v, "<|eom_id|>") : -1;
    st.can_think_off = tmpl == TMPL_CHATML && find_token(&v, "<think>") >= 0;
    st.model = path; st.threads = threads; st.ctx_n = n_ctx;
    {
        const char * js = (const char *) sml_ptr(&m, m.hdr->json_off);
        const sml_json ch = sml_json_get(js, (size_t) m.hdr->json_n, "chat");
        if (ch.kind == 'o') {
            st.chat_json = ch.p; st.chat_n = ch.n;
            const sml_json ids = sml_json_get(ch.p, ch.n, "stop_ids");
            for (int i = 0; ids.kind == 'a' && st.n_stop_ids < 16; i++) {
                const sml_json v = sml_json_at(ids.p, ids.n, i);
                if (v.kind == 0) break;
                st.stop_ids[st.n_stop_ids++] = (int32_t) sml_json_int(v, -1);
            }
            printf("  the model's own chat frame is in the file (%d end-of-turn ids)\n", st.n_stop_ids);
        }
    }
    printf("  chat template: %s (turn terminator token %d)%s\n",
           tmpl == TMPL_CHATML ? "ChatML" :
           tmpl == TMPL_LLAMA3 ? "Llama-3" : "plain text, base model",
           (int) stop, st.can_think_off ? "; thinking can be switched off" : "");

    sock_t ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof one);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // localhost only
    addr.sin_port = htons((unsigned short) port);
    if (bind(ls, (struct sockaddr *) &addr, sizeof addr) != 0 || listen(ls, 16) != 0) {
        fprintf(stderr, "cannot listen on port %d\n", port);
        return 1;
    }

    printf("sml-server on http://127.0.0.1:%d  depth=%d threads=%d ctx=%d slots=%d\n",
           port, st.depth, threads, n_ctx, n_slots);
    printf("  %.3f GB/token at this depth; orchestrator %s\n",
           (double) sml_bytes_at_depth(&m, st.depth) / 1e9, orch ? "on" : "off");
    fflush(stdout);

    static char req[REQ_MAX];
    for (;;) {
        sock_t s = accept(ls, NULL, NULL);
        if (s == INVALID_SOCKET) continue;

        size_t n = 0;
        char * body = NULL;
        size_t need = 0;
        while (n + 1 < REQ_MAX) {
            const int k = recv(s, req + n, (int)(REQ_MAX - n - 1), 0);
            if (k <= 0) break;
            n += (size_t) k;
            req[n] = 0;
            if (!body) {
                char * hdr_end = strstr(req, "\r\n\r\n");
                if (hdr_end) {
                    body = hdr_end + 4;
                    const char * cl = strstr(req, "Content-Length:");
                    if (!cl) cl = strstr(req, "content-length:");
                    need = cl && cl < hdr_end ? (size_t) atol(cl + 15) : 0;
                }
            }
            if (body && (size_t)(n - (size_t)(body - req)) >= need) break;
        }
        if (!body) { CLOSESOCK(s); continue; }

        if (!strncmp(req, "OPTIONS", 7)) {
            respond(s, 200, "text/plain", "");
        } else if (path_is(req, "GET", "/") || path_is(req, "GET", "/index.html")) {
            serve_ui(s);
        } else if (path_is(req, "GET", "/stats")) {
            const char * base = st.model ? st.model : "";
            for (const char * p = base; *p; p++) if (*p == '/' || *p == 92) base = p + 1;
            char js[1400];
            snprintf(js, sizeof js,
                "{\"model\":\"%s\",\"threads\":%d,\"ctx\":%d,\"depth\":%d,\"slots\":%d,"
                "\"device\":\"%s\",\"device_requested\":\"%s\",\"gpu_layers\":%d,\"gpu_of\":%d,\"device_why\":\"%s\","
                "\"device_name\":\"%s\",\"mid_layers\":%d,"
                "\"turns\":%ld,\"tokens_total\":%lld,"
                "\"last_prompt\":%d,\"last_gen\":%d,\"last_reused\":%d,\"last_slot\":%d,"
                "\"last_prompt_ms\":%.1f,\"last_gen_ms\":%.1f,"
                "\"last_bytes_per_token\":%.0f,"
                "\"last_margin\":%.3f,\"last_margin_min\":%.3f,\"last_cached\":%s,"
                "\"template\":\"%s\"}",
                base, st.threads, st.ctx_n, st.depth, st.n_slots, g_dev, g_dev_req, g_gpu ? g_gpu_layers : 0, g_gpu_of, g_dev_why, (g_ext || g_front || !strcmp(g_dev, "multi")) ? g_dev_name : "", (g_front || !strcmp(g_dev, "multi")) ? g_mid_layers : 0,
                st.turns, st.tokens_total,
                st.l_prompt, st.l_gen, st.l_reused, st.l_slot,
                st.l_prompt_ms, st.l_gen_ms, st.l_bpt, st.l_margin, st.l_margin_min,
                st.l_cached ? "true" : "false",
                st.tmpl == TMPL_CHATML ? "chatml" : st.tmpl == TMPL_LLAMA3 ? "llama3" : "plain");
            respond(s, 200, "application/json", js);
        } else if (path_is(req, "GET", "/v1/template")) {
            // the model's own chat block, for the backend to render exactly
            if (st.chat_json) {
                char * t = malloc(st.chat_n + 1);
                if (t) { memcpy(t, st.chat_json, st.chat_n); t[st.chat_n] = 0; respond(s, 200, "application/json", t); free(t); }
            } else {
                respond(s, 200, "application/json", "{\"template\":\"\"}");
            }
        } else if (path_is(req, "GET", "/health")) {
            respond(s, 200, "application/json", "{\"status\":\"ok\",\"engine\":\"sml\"}");
        } else if (path_is(req, "GET", "/v1/models")) {
            respond(s, 200, "application/json",
                    "{\"object\":\"list\",\"data\":[{\"id\":\"sml\",\"object\":\"model\"}]}");
        } else if (path_is(req, "POST", "/v1/chat/completions")) {
            handle_chat(&st, s, body, need ? need : strlen(body));
        } else {
            respond(s, 404, "application/json", "{\"error\":\"not found\"}");
        }
        CLOSESOCK(s);
    }
}
