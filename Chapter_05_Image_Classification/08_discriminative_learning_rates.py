"""
Chapter 5 — Image Classification
Topic 8: Discriminative learning rates (book section of the same name)

The first layers of a pretrained model learned general features (edges, gradients, textures)
that are useful for any image task; the last layers learned features specific to ImageNet. So
use a lower learning rate for the early layers and a higher one for the late layers:

    learn.fit_one_cycle(3, 3e-3)
    learn.unfreeze()
    learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))

slice(1e-6, 1e-4): the first parameter group gets 1e-6, the last 1e-4, the ones in between are
spread geometrically. Here the model has 3 groups: early body, late body, head.

Run:
    python 08_discriminative_learning_rates.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cnn_learner, error_rate, outputs_dir, resnet34, set_seed  # noqa: E402
from pets_data import get_dls  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)
learn = cnn_learner(get_dls(), resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
print("learning rates for the 3 parameter groups with lr_max=slice(1e-6, 1e-4):",
      [f"{lr:.0e}" for lr in learn._lrs(slice(1e-6, 1e-4))])
learn.fit_one_cycle(12, lr_max=slice(1e-6, 1e-4))
learn.recorder.plot_loss(fname=OUT / "plot_loss.png")
print("Look at outputs/plot_loss.png: the validation loss stops improving while the training loss keeps "
      "going down. Watch the *metric* (error rate), not just the loss, to decide when it's overfitting.")
