"""
Chapter 7 — Training a State-of-the-Art Model
Topic 4: Test time augmentation (book section "Test Time Augmentation")

On the validation set, images are centre-cropped: parts of the image are never seen. TTA makes
predictions on several augmented versions of each validation image and averages them (plus the
normal prediction). No retraining needed: slower inference, often better accuracy.

    preds,targs = learn.tta()
    accuracy(preds, targs).item()

Run:
    python 04_test_time_augmentation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CrossEntropyLossFlat, Learner, accuracy, set_seed, xresnet50  # noqa: E402
from imagenette_data import get_dls  # noqa: E402

set_seed(42)
dls = get_dls(64, 128)
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)

preds, targs = learn.get_preds()
print(f"\naccuracy, normal validation:           {accuracy(preds, targs).item():.4f}")
preds, targs = learn.tta()
print(f"accuracy with test time augmentation:  {accuracy(preds, targs).item():.4f}")
