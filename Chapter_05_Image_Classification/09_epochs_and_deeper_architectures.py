"""
Chapter 5 — Image Classification
Topic 9: Selecting the number of epochs; deeper architectures; mixed precision (book sections of
the same names)

Number of epochs: pick how long you are happy to wait, train, and look at the metrics. If the
metric is still improving at the end, train longer. With 1cycle training, retrain from scratch with
the right number of epochs rather than using early stopping (the best results come at the end,
when the learning rate is small).

Deeper architectures (ResNet-18, 34, 50, 101, 152) can fit more complex functions, but take
longer, need more memory and overfit more easily. Mixed precision training (to_fp16) uses
half-precision floats where it's safe: faster on modern GPUs and less memory.

    learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
    learn.fine_tune(6, freeze_epochs=3)

Run:
    python 09_epochs_and_deeper_architectures.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cnn_learner, error_rate, get_device, resnet50, set_seed  # noqa: E402
from pets_data import get_dls  # noqa: E402

set_seed(42)
learn = cnn_learner(get_dls(), resnet50, metrics=error_rate).to_fp16()
print("mixed precision:", "on" if learn.use_amp else f"off (needs an NVIDIA GPU, found {get_device()})")
t0 = time.time()
learn.fine_tune(6, freeze_epochs=3)
print(f"ResNet-50, 3 + 6 epochs: {time.time() - t0:.0f}s. Compare its error rate with the ResNet-34 runs.")
