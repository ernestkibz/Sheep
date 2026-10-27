#!/usr/bin/env python3
"""Grade a Sheep vision model against the full-precision original.

    python tooling/sml_vlm_grade.py MODEL.ssf HF_CHECKPOINT_DIR

Two checks, kept apart because a slightly wrong tower still hands the text model
plausible vectors:
  tower   -- the rows the vision tower gives the text model, against HF's own
             image features for the same pixels (relative error, correlation);
  answer  -- the next-token logits for an image + question, against HF's
             (correlation, and whether the top token matches).

Images are drawn here, in code: a fixed, reproducible set with nothing borrowed.
The processor runs with image splitting OFF, one 512 x 512 view per image, which
is the configuration the engine implements.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "sml", "sml-cli" + (".exe" if os.name == "nt" else ""))


def images():
    from PIL import Image, ImageDraw
    out = []
    im = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(im).ellipse((128, 128, 384, 384), fill=(220, 20, 20))
    out.append(("red circle on white", im, "What color is the circle?"))
    im = Image.new("RGB", (640, 480), (250, 230, 40))
    ImageDraw.Draw(im).rectangle((200, 120, 440, 360), fill=(20, 40, 200))
    out.append(("blue square on yellow", im, "What shape is in the middle of the picture?"))
    im = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(im)
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            if (i // 64 + j // 64) % 2 == 0:
                d.rectangle((i, j, i + 63, j + 63), fill="black")
    out.append(("checkerboard", im, "Describe this image in one sentence."))
    im = Image.new("RGB", (512, 384), (30, 140, 60))
    ImageDraw.Draw(im).polygon([(256, 40), (60, 340), (452, 340)], fill=(255, 255, 255))
    out.append(("white triangle on green", im, "How many triangles are there?"))
    g = np.linspace(0, 255, 512, dtype=np.float32)
    arr = np.stack([np.tile(g, (512, 1)), np.tile(g[:, None], (1, 512)), np.full((512, 512), 128.0)], -1)
    out.append(("gradient", Image.fromarray(arr.astype(np.uint8)), "What colors do you see?"))
    return out


def sheep(ssf, ids, px):
    d = tempfile.mkdtemp()
    pf, vf, lf = (os.path.join(d, n) for n in ("px.f32", "vis.f32", "logits.bin"))
    px.astype(np.float32).tofile(pf)
    env = dict(os.environ, SML_RL=os.environ.get("SML_RL", "0"))
    subprocess.run([CLI, ssf, "--tokens", ",".join(map(str, ids)), "--image", pf,
                    "--vision-out", vf, "--gen", "0", "--threads", "6", "--logits", lf],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vis = np.fromfile(vf, dtype=np.float32)
    logits = np.fromfile(lf, dtype=np.float32, offset=8)
    for f in (pf, vf, lf):
        os.remove(f)
    os.rmdir(d)
    return vis, logits.astype(np.float64)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ssf, hf = sys.argv[1], sys.argv[2]
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(hf)
    proc.image_processor.do_image_splitting = False   # a call argument is ignored
    model = AutoModelForImageTextToText.from_pretrained(hf, dtype=torch.float32).eval()
    torch.set_num_threads(6)
    rows = []
    for name, im, q in images():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        enc = proc(text=text, images=[im], return_tensors="pt")
        pv = enc["pixel_values"]
        assert pv.shape[1] == 1, f"expected one view per image, got {tuple(pv.shape)}"
        with torch.no_grad():
            feats = model.model.get_image_features(pv, enc.get("pixel_attention_mask"))
            feats = (feats[0] if isinstance(feats, (list, tuple)) else feats).reshape(-1).double().numpy()
            fp = model(**enc).logits[0, -1].double().numpy()
        ids = enc["input_ids"][0].tolist()
        vis, lg = sheep(ssf, ids, pv[0, 0].numpy())
        lg = lg[: fp.size]
        tv = np.linalg.norm(vis - feats) / np.linalg.norm(feats)
        rows.append((name, tv, np.corrcoef(vis, feats)[0, 1], np.corrcoef(lg, fp)[0, 1],
                     int(lg.argmax() == fp.argmax()), proc.tokenizer.decode([int(fp.argmax())]),
                     proc.tokenizer.decode([int(lg.argmax())])))
    print(f"{os.path.basename(ssf)} vs {hf} (full precision), {len(rows)} images, no splitting")
    print(f"{'image':24s} {'tower rel':>9s} {'tower corr':>10s} {'answer corr':>11s}  top-1 (HF / Sheep)")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:9.4f} {r[2]:10.5f} {r[3]:11.4f}  {'Y' if r[4] else '-'} {r[5]!r} / {r[6]!r}")
    a = np.array([r[1:5] for r in rows], dtype=np.float64)
    print(f"{'MEAN':24s} {a[:, 0].mean():9.4f} {a[:, 1].mean():10.5f} {a[:, 2].mean():11.4f}  "
          f"{int(a[:, 3].sum())}/{len(rows)}")


if __name__ == "__main__":
    main()
