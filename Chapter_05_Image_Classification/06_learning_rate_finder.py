"""
Chapter 5 — Image Classification
Topic 6: The learning rate finder (book section "The Learning Rate Finder")

Too high a learning rate and training diverges; too low and it takes forever (and overfits,
because it sees the data many times). Leslie Smith's learning rate finder: train one mini-batch
at a very small lr, then increase the lr exponentially batch after batch, recording the loss,
until the loss explodes. Then pick a lr where the loss is still clearly decreasing: the minimum
divided by 10, or the steepest point.

    learn.fine_tune(1, base_lr=0.1)            # way too high: look at the losses
    lr_min, lr_steep = learn.lr_find()
    learn.fine_tune(2, base_lr=3e-3)

Run:
    python 06_learning_rate_finder.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cnn_learner, error_rate, outputs_dir, resnet34, set_seed  # noqa: E402
from pets_data import get_dls  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)
dls = get_dls()

print("=== base_lr=0.1: too high ===")
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1, base_lr=0.1)

print("\n=== Using the learning rate finder ===")
learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min, lr_steep = learn.lr_find(fname=OUT / "lr_find.png")
print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
print("Losses vs learning rates (every 10th batch):")
for lr, loss in list(zip(learn.recorder.lrs, learn.recorder.losses))[::10]:
    print(f"   lr {lr:9.2e}  loss {loss:7.3f}  " + "#" * min(60, int(loss * 10)))

learn.fine_tune(2, base_lr=3e-3)
