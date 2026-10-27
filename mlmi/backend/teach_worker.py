"""Train a Checker model on Sheep's lessons: LoRA on the model's original weights,
then merged into a full checkpoint the Sheep Quantizer can read.

    python teach_worker.py --src ORIGINAL_DIR --lessons lessons.jsonl --out MERGED_DIR [--device cuda|cpu]

On a CUDA card the base model is held in 4 bits (QLoRA), so a 1.5B model trains
in about 2 GB. Each lesson's loss counts only the answer (the last message), in
the model's own chat template. One JSON object per line on stdout:
{"stage"}, {"step", "of", "loss"}, then {"ok": true, "out"} or {"error"}.
"""
import argparse
from collections import defaultdict
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path


def say(**kw):
    print(json.dumps(kw), flush=True)


def encode(tok, msgs, max_len):
    """(input ids, labels): labels only on the answer's tokens."""
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    labels = [-100] * min(n, len(ids)) + ids[n:]
    return ids, labels[:len(ids)]


def training_step_count(examples, epochs, batch, max_steps=0):
    """Return the calculated optimizer steps, optionally capped for a short trial."""
    if batch < 1 or epochs <= 0:
        raise ValueError("training batch and epochs must be positive")
    calculated = math.ceil(examples * epochs / batch)
    return min(calculated, max_steps) if max_steps > 0 else calculated


def _latest_checkpoint(checkpoint_root):
    if not checkpoint_root:
        return None
    try:
        name = (checkpoint_root / "latest.txt").read_text(encoding="utf-8").strip()
        path = checkpoint_root / name
        return path if path.is_dir() else None
    except OSError:
        return None


def _save_checkpoint(model, checkpoint_root, step, order, opt, sched, rnd, torch, dev):
    if not checkpoint_root:
        return
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{step:06d}"
    tmp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoint_root))
    try:
        model.save_pretrained(tmp / "adapter", safe_serialization=True)
        payload = {"step": step, "order": order, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                   "python_rng": rnd.getstate(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if dev == "cuda" else []}
        torch.save(payload, tmp / "state.pt")
        final = checkpoint_root / name
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        marker = checkpoint_root / "latest.txt.tmp"
        marker.write_text(name, encoding="utf-8")
        os.replace(marker, checkpoint_root / "latest.txt")
        # Keep the latest two complete checkpoints so an interrupted save has a predecessor.
        complete = sorted((p for p in checkpoint_root.glob("checkpoint-*" ) if p.is_dir()),
                          key=lambda p: p.name)
        for old in complete[:-2]:
            shutil.rmtree(old, ignore_errors=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _current_gpu_temp():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from mlmi.backend.training_policy import nvidia_thermal
        return nvidia_thermal().get("current_c")
    except Exception:
        return None


def _merge_saved_adapter(a, tok, torch, AutoModelForCausalLM, PeftModel, adapter):
    say(stage="merging the lessons into the weights")
    base = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    for f in Path(a.src).iterdir():                 # licence, generation config, template: the model's own papers
        if f.is_file() and not (out / f.name).exists() and not f.name.endswith((".safetensors", ".bin")):
            shutil.copyfile(f, out / f.name)
    return out


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--lessons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap optimizer steps for a bounded Checker trial; 0 keeps the full calculated schedule")
    ap.add_argument("--pause-at-c", type=int, default=0)
    ap.add_argument("--max-chunk-seconds", type=int, default=0)
    ap.add_argument("--stop-file", default="")
    return ap.parse_args()


def main():
    total_started = time.monotonic()
    a = _parse_args()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    load_started = time.monotonic()
    say(stage="loading the original weights", device=dev)
    tok = AutoTokenizer.from_pretrained(a.src)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if dev == "cuda":
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(a.src, quantization_config=q, device_map={"": 0},
                                                     dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.float32)
        model.gradient_checkpointing_enable()
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    checkpoint_root = Path(a.checkpoint_dir) if a.checkpoint_dir else None
    existing = _latest_checkpoint(checkpoint_root) if a.resume else None
    if existing and (existing / "adapter").is_dir():
        model = PeftModel.from_pretrained(model, existing / "adapter", is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    load_seconds = round(time.monotonic() - load_started, 1)
    prepare_started = time.monotonic()
    rows = [json.loads(l) for l in open(a.lessons, encoding="utf-8") if l.strip()]
    data = [(str(row.get("kind") or "other"), encode(tok, row["messages"], a.max_len)) for row in rows]
    data = [(kind, encoded) for kind, encoded in data if any(x != -100 for x in encoded[1])]
    calculated_steps = math.ceil(len(data) * a.epochs / a.batch)
    steps = training_step_count(len(data), a.epochs, a.batch, a.max_steps)
    if steps < 1:
        raise RuntimeError("the selected Checker training set contains no usable examples")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / steps))
    model.train()
    rnd = random.Random(1)
    groups = defaultdict(list)
    for index, (kind, _) in enumerate(data):
        groups[kind].append(index)
    for indexes in groups.values():
        rnd.shuffle(indexes)
    order = []
    # Interleave lesson types so every bounded chunk sees arithmetic, facts,
    # translations, open requests, and Checker fault examples where available.
    while any(groups.values()):
        for kind in sorted(groups):
            if groups[kind]:
                order.append(groups[kind].pop())
    first_pass = list(order)
    while len(order) < steps * a.batch:
        order.extend(first_pass)
    order = order[:steps * a.batch]
    start_step = 0
    if existing and (existing / "state.pt").is_file():
        try:
            try:
                saved = torch.load(existing / "state.pt", map_location="cpu", weights_only=False)
            except TypeError:
                saved = torch.load(existing / "state.pt", map_location="cpu")
            start_step = min(int(saved.get("step", 0)), steps)
            order = saved.get("order", order)
            if any(index < 0 or index >= len(data) for index in order):
                raise RuntimeError("the saved checkpoint does not match these training lessons")
            if int(saved.get("step", 0)) > steps:
                raise RuntimeError("the saved run is longer than the new short trial; start a fresh trial to avoid skipping its exam")
            opt.load_state_dict(saved["optimizer"])
            sched.load_state_dict(saved["scheduler"])
            rnd.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if dev == "cuda" and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            say(stage="resumed from a safe checkpoint", step=start_step, of=steps)
        except Exception as exc:
            raise RuntimeError(f"could not resume the last safe training checkpoint: {exc}") from exc
    prepare_seconds = round(time.monotonic() - prepare_started, 1)
    say(stage="training", lessons=len(data), steps=steps, calculated_steps=calculated_steps,
         effective_batch=a.batch, micro_batch=1 if dev == "cuda" else a.batch,
         timings={"load_seconds": load_seconds, "prepare_seconds": prepare_seconds})
    train_started = time.monotonic()
    t0, avg = time.time(), None
    micro = 1 if dev == "cuda" else a.batch   # one CUDA example at a time; gradients accumulate to the same batch

    next_thermal_check = 0.0
    chunk_started = time.monotonic()
    for s in range(start_step, steps):
        batch = [data[i][1] for i in order[s * a.batch:(s + 1) * a.batch]]
        opt.zero_grad()
        total = 0.0
        for k in range(0, len(batch), micro):
            part = batch[k:k + micro]
            L = max(len(x[0]) for x in part)
            ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in part])
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in part])
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in part])
            out = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device), labels=lab.to(model.device))
            (out.loss * len(part) / len(batch)).backward()
            total += out.loss.item() * len(part) / len(batch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        avg = total if avg is None else 0.9 * avg + 0.1 * total
        elapsed = time.monotonic() - chunk_started
        temp = None
        thermal_checked = dev == "cuda" and bool(a.pause_at_c) and time.monotonic() >= next_thermal_check
        if thermal_checked:
            temp = _current_gpu_temp()
            next_thermal_check = time.monotonic() + 10
        stop_file = Path(a.stop_file) if a.stop_file else None
        should_stop = bool(stop_file and stop_file.exists())
        thermal_pause = temp is not None and temp >= a.pause_at_c
        telemetry_pause = thermal_checked and temp is None
        chunk_pause = dev == "cuda" and a.max_chunk_seconds and elapsed >= a.max_chunk_seconds
        if (s + 1) % max(1, a.checkpoint_every) == 0 or s == steps - 1 or should_stop or thermal_pause or chunk_pause:
            _save_checkpoint(model, checkpoint_root, s + 1, order, opt, sched, rnd, torch, dev)
        if s % 5 == 0 or s == steps - 1 or thermal_pause or telemetry_pause or chunk_pause:
            say(step=s + 1, chunk_step=s + 1 - start_step, of=steps, loss=round(avg, 4), s=round(time.time() - t0),
                timings={"load_seconds": load_seconds, "prepare_seconds": prepare_seconds,
                         "train_seconds": round(time.monotonic() - train_started, 1)})
        if should_stop or thermal_pause or telemetry_pause or chunk_pause:
            why = ("stopped at a safe checkpoint" if should_stop else
                   f"temperature reached {temp} °C; checkpoint saved" if thermal_pause else
                   "GPU temperature sensor stopped responding; checkpoint saved and training paused" if telemetry_pause else
                   f"GPU chunk reached {a.max_chunk_seconds} seconds; checkpoint saved")
            say(paused=not should_stop, stopped=should_stop, why=why, step=s + 1, of=steps,
                temperature_c=temp, checkpoint=str(checkpoint_root) if checkpoint_root else "",
                timings={"load_seconds": load_seconds, "prepare_seconds": prepare_seconds,
                         "train_seconds": round(time.monotonic() - train_started, 1)})
            return
    train_seconds = round(time.monotonic() - train_started, 1)
    merge_started = time.monotonic()
    adapter = Path(a.out + "-adapter")
    model.save_pretrained(adapter)
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    out = _merge_saved_adapter(a, tok, torch, AutoModelForCausalLM, PeftModel, adapter)
    merge_seconds = round(time.monotonic() - merge_started, 1)
    say(ok=True, out=str(out), s=round(time.time() - t0),
        timings={"load_seconds": load_seconds, "prepare_seconds": prepare_seconds,
                 "train_seconds": train_seconds, "merge_seconds": merge_seconds,
                 "total_seconds": round(time.monotonic() - total_started, 1)})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        say(error=f"{type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
