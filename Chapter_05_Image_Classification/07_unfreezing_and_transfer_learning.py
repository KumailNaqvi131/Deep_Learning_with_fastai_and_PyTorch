"""
Chapter 5 — Image Classification
Topic 7: Unfreezing and transfer learning (book section of the same name)

A pretrained model's last layer was made for the 1,000 ImageNet classes: cnn_learner throws it
away and adds a new, random head for our 37 breeds. fine_tune(n) does:
  1. freeze the pretrained body and train only the new head for 1 epoch (freeze_epochs),
  2. unfreeze, and train everything for n epochs.

The book does the same by hand:
    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.fit_one_cycle(3, 3e-3)          # the body is frozen
    learn.unfreeze()
    learn.lr_find()                       # the curve looks different now: the model is trained
    learn.fit_one_cycle(6, lr_max=1e-5)

Run:
    python 07_unfreezing_and_transfer_learning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cnn_learner, error_rate, outputs_dir, resnet34, set_seed  # noqa: E402
from pets_data import get_dls  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)
learn = cnn_learner(get_dls(), resnet34, metrics=error_rate)


def trainable():
    return sum(p.numel() for p in learn.model.parameters() if p.requires_grad)


total = sum(p.numel() for p in learn.model.parameters())
print(f"Frozen: training {trainable():,} of {total:,} parameters (the head + BatchNorm layers)")
learn.fit_one_cycle(3, 3e-3)

learn.unfreeze()
print(f"\nUnfrozen: training {trainable():,} of {total:,} parameters")
learn.lr_find(fname=OUT / "lr_find_unfrozen.png")
learn.fit_one_cycle(6, lr_max=1e-5)
