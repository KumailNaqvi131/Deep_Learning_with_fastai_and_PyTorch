"""
Chapter 2 — From Model to Production
Topic 7: How to avoid disaster: out-of-domain data and domain shift (book section "How to Avoid
Disaster" and "Unforeseen Consequences and Feedback Loops")

A model only knows the kind of data it was trained on. Shown something different (out-of-domain
data), it doesn't say "I don't know": it still picks one of its classes, often with high
confidence. Over time, the data the model sees in production can also drift away from the
training data (domain shift).

This script uses the bear model from 05_export_and_inference.py and shows it:
  * normal validation images,
  * out-of-domain images: cartoon cars, buses, chairs and pure noise,
  * a domain shift: the same bears, but much darker (photos taken at night).

The book's advice: deploy carefully. Run the model in parallel with the existing (human) process
at first, roll out in a limited scope, and monitor its predictions over time.

Run:
    python 07_how_to_avoid_disaster.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance  # noqa: E402

from common import *  # noqa: F401,F403,E402
from common.synth_vision import _icon, background  # noqa: E402

OUT = outputs_dir(__file__)
model_file = OUT / "export.pkl"
if not model_file.exists():
    sys.exit("No exported model yet: run  python 05_export_and_inference.py  first.")
learn_inf = load_learner(model_file)
path = untar_data(URLs.BEARS)
rng = np.random.default_rng(0)

in_domain = [Image.open(f) for f in get_image_files(path)[::50][:8]]
ood = []
for name in ("car", "bus", "chair", "tvmonitor"):
    img = background(rng, 400, 300)
    _icon(ImageDraw.Draw(img), name, 200, 150, 90, rng)
    ood.append(img)
ood += [Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)) for _ in range(2)]
night = [ImageEnhance.Brightness(im).enhance(0.25) for im in in_domain[:6]]


def report(name, images):
    results = [learn_inf.predict(im) for im in images]
    conf = [float(p.max()) for _, _, p in results]
    print(f"\n{name}: average confidence {np.mean(conf):.2f}")
    for (pred, _, p), c in zip(results, conf):
        print(f"   predicted {pred:8s} with probability {c:.2f}")
    show_images(images, [f"{r[0]} {c:.2f}" for r, c in zip(results, conf)],
                OUT / f"disaster_{name.split()[0].lower()}.png", ncols=4)


report("In-domain validation images", in_domain)
report("Out-of-domain images (no bears!)", ood)
report("Domain shift (night photos)", night)
print("\nThe model happily calls cars and noise 'bears'. A softmax always sums to 1: there is no "
      "'none of these' answer unless you train for it (e.g. with multi-label classification, Ch. 6).")
