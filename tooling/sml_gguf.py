"""A GGUF file back to an original-format folder (safetensors, config, tokenizer), so Sheep's
normal build can read it.

    python tooling/sml_gguf.py <file.gguf> <out folder>

Run by a PyTorch runtime: transformers reads the GGUF (its own gguf loader, which needs the
`gguf` package) and writes the weights as float16. The quality is the GGUF's: a Q4 file stays
a Q4 model, now held in float16. Used only when there is no original checkpoint to build from,
such as a fine-tune published only as GGUF, or a .gguf file already on this computer.
Prints one line "RESULT {json}".
"""
import json
import os
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        d, f = os.path.dirname(os.path.abspath(src)), os.path.basename(src)
        print(f"converting: reading {f}", flush=True)
        tok = AutoTokenizer.from_pretrained(d, gguf_file=f)
        model = AutoModelForCausalLM.from_pretrained(d, gguf_file=f, torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True)
        os.makedirs(out, exist_ok=True)
        print("converting: writing the original-format folder", flush=True)
        model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
        tok.save_pretrained(out)
        print("RESULT " + json.dumps({"ok": True, "model_type": model.config.model_type}), flush=True)
    except Exception as e:      # an architecture transformers' GGUF loader does not know, and the like
        print("RESULT " + json.dumps({"ok": False, "why": f"{type(e).__name__}: {str(e)[:300]}"}), flush=True)


if __name__ == "__main__":
    main()
