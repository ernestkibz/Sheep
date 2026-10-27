#!/usr/bin/env python3
"""Turn an image into the pixels a Sheep vision tower reads.

    python tooling/sml_image.py PHOTO.jpg OUT.f32 [--size 512]

Writes 3 x size x size float32, channel-major, scaled to [-1, 1]. The recipe was
matched against the reference processor (splitting off) by comparing pixels, not
by reading its source: longest edge to 2048 with Lanczos, then 512 x 512 with
Lanczos, then (x / 255 - 0.5) / 0.5. The same two resizes give a max difference
of 0.0000; one direct resize to 512 differs by up to 0.17, which is a different
image as far as the tower is concerned.
"""
import argparse

import numpy as np


def prepare(image, size=512, first_edge=2048):
    from PIL import Image
    im = image if isinstance(image, Image.Image) else Image.open(image)
    im = im.convert("RGB")
    w, h = im.size
    s = first_edge / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((a - 0.5) / 0.5).transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    prepare(a.image, a.size).tofile(a.out)


if __name__ == "__main__":
    main()
