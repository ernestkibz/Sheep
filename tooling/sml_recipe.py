"""THE build recipe's allocation rule, in one place: the races, the pilot and
the ML.M.I quantizer all build with it (STANDARDS.md section 3). Pure Python,
no heavy imports, so the app can use it without loading the race harness."""


def sheep_alloc(n_layer):
    """THE allocation (commandment II), one rule for every family: Q4_K_M's own
    6-bit attn_v / ffn_down layers, but the 6 bits it spends on the OUTPUT head
    spent on the EMBEDDING instead -- the same bytes. Measured on the Qwen3 race
    head (2026-09-30): the 4-bit embedding carried a third of all KL; with the
    swap (and the byte budget) KL 0.0198 against llama.cpp's 0.0292, where the
    head-6 allocation gave 0.0252 -- and decode reads 13 MB less per token, since
    the head is read in full every token and the embedding one row. A tied model
    has one tensor for both, and it stays 6-bit."""
    return q4km_alloc(n_layer).replace("output=6", "token_embd=6")


def q4km_alloc(n_layer):
    """Sheep's bits placed where Q4_K_M places its own: a 6-bit head (their
    output.weight is Q6_K) and 6-bit attn_v / ffn_down in the layers llama.cpp's
    use_more_bits picks (the first and last eighth, and every third between).
    The race is then quantizer against quantizer at the same structure."""
    more = [i for i in range(n_layer)
            if i < n_layer // 8 or i >= 7 * n_layer // 8 or (i - n_layer // 8) % 3 == 2]
    lst = ";".join(map(str, more))
    return f"output=6,attn_v=6@{lst},ffn_down=6@{lst}"
