"""
Chapter 7 — Training a State-of-the-Art Model
Topic 1: Imagenette and a baseline trained from scratch (book section "Imagenette")

Imagenette: 10 very different classes picked from ImageNet (tench, English springer, cassette
player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute). It's small
enough to experiment quickly, yet results transfer to the full ImageNet: a good place to try ideas.

The baseline: an xResNet-50 trained from scratch (no pretrained weights):

    model = xresnet50(n_out=dls.c)
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    learn.fit_one_cycle(5, 3e-3)

The next scripts each add one technique and compare with this baseline.

Run:
    python 01_imagenette_baseline.py            # add --real for the real imagenette2-160 (95 MB)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (CrossEntropyLossFlat, Learner, accuracy, outputs_dir, set_seed,  # noqa: E402
                    xresnet50)
from imagenette_data import NAMES, get_dls, path  # noqa: E402

set_seed(42)
print("classes (WordNet ids):", {k: NAMES.get(k, k) for k in sorted(p.name for p in (path / "train").iterdir())})
dls = get_dls(64, 128, normalize=False)
dls.show_batch(max_n=9, fname=outputs_dir(__file__) / "imagenette_batch.png")

model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
