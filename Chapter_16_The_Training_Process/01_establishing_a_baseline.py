"""
Chapter 16 — The Training Process
Topic 1: Establishing a baseline (book section of the same name)

To compare optimizers, train the same model (a ResNet-34 from scratch) on Imagenette, first with
fastai's default (Adam with 1cycle), then with plain SGD. With SGD the learning-rate finder points
to a higher lr, and with moms=(0,0,0) (no momentum) training is much slower.

    learn = get_learner()
    learn.fit_one_cycle(3, 0.003)
    learn = get_learner(opt_func=SGD)
    learn.lr_find()
    learn.fit_one_cycle(3, 0.03, moms=(0,0,0))

Run:
    python 01_establishing_a_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import SGD, outputs_dir, set_seed  # noqa: E402
from training_setup import get_learner  # noqa: E402

set_seed(42)
print("Adam (fastai's default):")
learn = get_learner()
learn.fit_one_cycle(3, 0.003)

print("\nplain SGD, no momentum:")
set_seed(42)
learn = get_learner(opt_func=SGD)
learn.lr_find(fname=outputs_dir(__file__) / "sgd_lr_find.png")
learn.fit_one_cycle(3, 0.03, moms=(0, 0, 0))
