"""
Chapter 7 — Training a State-of-the-Art Model
Topic 2: Normalization (book section "Normalization")

Models train better when the input data has a mean of 0 and a standard deviation of 1. Our image
pixels are between 0 and 1, with a mean around 0.4-0.5. Normalize each channel with the
statistics of ImageNet (also used when training the pretrained models, which is why it matters
even more for transfer learning: the pretrained weights expect normalised inputs).

    x,y = dls.one_batch()
    x.mean(dim=[0,2,3]), x.std(dim=[0,2,3])
    batch_tfms=[*aug_transforms(size=size, min_scale=0.75), Normalize.from_stats(*imagenet_stats)]

Run:
    python 02_normalization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CrossEntropyLossFlat, Learner, accuracy, imagenet_stats, set_seed, xresnet50  # noqa: E402
from imagenette_data import get_dls  # noqa: E402

set_seed(42)
x, y = get_dls(64, 128, normalize=False).one_batch()
print("without Normalize: mean per channel", [round(v, 3) for v in x.mean(dim=[0, 2, 3]).tolist()],
      " std", [round(v, 3) for v in x.std(dim=[0, 2, 3]).tolist()])
print("imagenet_stats:", imagenet_stats)

dls = get_dls(64, 128)
x, y = dls.one_batch()
print("with Normalize:    mean per channel", [round(v, 3) for v in x.mean(dim=[0, 2, 3]).tolist()],
      " std", [round(v, 3) for v in x.std(dim=[0, 2, 3]).tolist()])

learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
print("Compare with 01_imagenette_baseline.py.")
