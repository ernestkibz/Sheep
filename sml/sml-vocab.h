// SSF tokenizer: byte-level BPE, read from the model file.
//
// Replaces the reference engine's vocab handling. The engine takes text in and gives text
// out without any part of the old tree, which is what lets core/ be deleted.
//
// The vocab is stored BINARY in the .ssf file, not JSON -- the engine parses no
// JSON anywhere, and a tokenizer that needed a JSON parser would drag one in for
// the sake of a table of strings.
//
//   u32 n_tokens
//   u32 n_merges
//   u32 bos, eos
//   u32 offsets[n_tokens + 1]      into the byte blob
//   u8  bytes[]                    token strings, concatenated
//   u32 merges[n_merges * 2]       token id pairs, in rank order
//
// Optional Unigram trailer (2026-10-14; Helium, XGLM, Persimmon), after the merges.
// Older files end at the merges and read exactly as before:
//
//   "UNIG", u32 version (1), u32 unk, u32 flags (SML_UNI_*), f32 unk_score, f32 max_score
//   u32 n_norm, u32 norm_bytes
//   f32 scores[n_tokens]
//   u8  kinds[n_tokens], padded to 4    (SML_PIECE_*)
//   norm records: u8 src_len, u8 dst_len, src, dst   (the precompiled charsmap, unpacked)
//
// tooling/sml_unigram.py writes it and encodes the same way in Python.
//
// Merges are resolved from strings to token ids at build time, so the encoder
// never does string comparison in its inner loop.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include "sml-model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SML_UNI_PREFIX_ALWAYS   1u   // U+2581 before the text (SentencePiece's dummy prefix)
#define SML_UNI_PREFIX_MISSING  2u   // ... unless it starts with one (tokenizers' Metaspace)
#define SML_UNI_BYTE_FALLBACK   4u   // an unknown character is spelled as its <0xNN> bytes
#define SML_UNI_SPLIT_WORDS     8u   // cut before every U+2581, each word alone
#define SML_UNI_SPLIT_NEWLINES 16u   // a run of newlines is its own piece
#define SML_UNI_COLLAPSE       32u   // two or more spaces become one
#define SML_UNI_STRIP          64u   // collapse, and strip both ends
#define SML_UNI_SPM_MATH      128u   // SentencePiece's arithmetic: float32 path scores, double candidates

#define SML_PIECE_NORMAL 0
#define SML_PIECE_USER   1           // user defined: always chosen where it fits
#define SML_PIECE_SKIP   2           // never chosen by the cut (control, byte, unused)
#define SML_PIECE_ADDED  3           // cut out whole before normalization

typedef struct {
    uint32_t n_tokens, n_merges, bos, eos;
    const uint32_t * offsets;
    const uint8_t  * bytes;
    const uint32_t * merges;      // pairs (a, b) in rank order

    // token string -> id, open addressing
    uint32_t * tok_slots;
    uint32_t   tok_mask;

    // (a, b) -> rank, open addressing
    uint64_t * mrg_keys;
    uint32_t * mrg_rank;
    uint32_t   mrg_mask;

    // Special tokens (<|im_start|>, <|im_end|>, ...). These must be matched as
    // whole units BEFORE byte-level BPE: run through BPE they decompose into
    // ordinary pieces, the model never sees the chat structure it was trained
    // on, and it emits turn delimiters as literal text instead of stopping.
    uint32_t * spec_ids;
    uint32_t   n_special;

    // Byte-encoding family. Both are byte-level BPE with merges; they differ
    // only in how a raw byte is spelled in the vocabulary:
    //
    //   BPE (GPT-2, Qwen, Llama-3): bytes map to printable code points, space
    //                               becomes U+0120
    //   SPM (Llama-1/2, Mistral):   space becomes U+2581, and any byte with no
    //                               token of its own is spelled "<0xNN>"
    //
    // Detected from the vocabulary rather than declared, since the file format
    // that carried the declaration is gone.
    int spm;
    uint32_t byte_tok[256];   // SPM only: id of "<0xNN>", or UINT32_MAX

    // Unigram: a score per piece and no merges; text is cut along the best-scoring
    // path (Viterbi), after the model's own normalization
    int unigram;
    uint32_t uflags, unk, max_tok;
    float unk_score, max_score;
    const float   * scores;
    const uint8_t * kinds;
    const uint8_t ** norm_rec;   // records hashed by source bytes
    uint32_t norm_mask, norm_max;
    uint8_t  first[32];          // bitset: bytes a special can start with

    // PRET: a BPE tokenizer's own text pipeline, from its tokenizer.json (sml_pretok.py,
    // 2026-10-19): normalizer, pre-tokenizer splits (regex, string, byte-level, Metaspace,
    // digits), ignore_merges, byte fallback. Files without it are cut the old way.
    int pret;
    uint32_t pflags, punk, n_norm, n_pre, n_cats;
    const uint8_t * pnorm;       // op streams, as packed
    const uint8_t * ppre;
    const uint8_t * pcats;       // Unicode class table
    void * pres;                 // the pre ops' compiled regexes (owned)
} sml_vocab;

#define SML_PRET_BYTE_LEVEL    1u
#define SML_PRET_IGNORE_MERGES 2u
#define SML_PRET_BYTE_FALLBACK 4u
#define SML_PRET_FUSE_UNK      8u

int  sml_vocab_init(sml_vocab * v, const sml_model * m);
void sml_vocab_free(sml_vocab * v);

// Encode UTF-8 text. Returns the number of tokens written, or -1 if `max` is
// too small.
int sml_encode(const sml_vocab * v, const char * text, int32_t * out, int max);

// Decode one token into `buf`; returns bytes written (not NUL terminated).
int sml_decode(const sml_vocab * v, int32_t id, char * buf, int max);

#ifdef __cplusplus
}
#endif
