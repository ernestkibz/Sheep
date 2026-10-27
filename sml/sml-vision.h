// The vision tower: an image in, a few dozen text-width vectors out, which the
// text model reads in place of its image tokens. Idefics3 / SmolVLM shape: a ViT
// over square patches, a pixel shuffle that folds scale x scale neighbouring
// patches into one token, and a linear projection to the text model's width.
#ifndef SML_VISION_H
#define SML_VISION_H

#include "sml-model.h"

typedef struct {
    int   n_embd, n_layer, n_head, n_ff;   // the tower's own transformer
    int   image, patch, scale;             // input side, patch side, shuffle factor
    int   image_token_id;                  // the text token each output row replaces
    int   n_patch;                         // (image / patch)^2
    int   n_tok;                           // n_patch / scale^2: rows handed to the text model
    int   out_dim;                         // the text model's width
    float eps;
    // The chat frame for an image turn (Idefics3), -1 / "" when the file has
    // none: <start> USER <fake><global> image x n_tok <fake> QUESTION <end> ASSISTANT
    int   tok_start, tok_fake, tok_global, tok_end;
    char  user[32], assistant[32];
} sml_vcfg;

// 1 and fills *v if the file carries a vision tower, 0 if it does not.
int sml_vision_cfg(const sml_model * m, sml_vcfg * v);

// Token ids for one image turn: the frame, the image block, the question.
// `vocab` is the model's sml_vocab. Returns the count, or -1 if the file
// carries no chat frame or `max` is too short.
int sml_vision_prompt(const sml_vcfg * v, const void * vocab, const char * question,
                      int32_t * ids, int max);

// pixels: 3 x image x image floats, channel-major, already resized and
// normalized as the model expects. out: n_tok x out_dim floats.
// Returns 0 on success, -1 if a tensor is missing or memory runs out.
int sml_vision_encode(const sml_model * m, const sml_vcfg * v, const float * pixels,
                      float * out, int n_threads);

#endif
