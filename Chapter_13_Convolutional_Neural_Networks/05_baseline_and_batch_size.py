"""
Chapter 13 — Convolutional Neural Networks
Topic 5: Improving training stability: a simple baseline; increase batch size (book sections
"Improving Training Stability", "A Simple Baseline", "Increase Batch Size")

Now all ten digits. A CNN like the previous one, a bit bigger, trained at a high learning rate
(0.06) for speed... and it trains badly (in the book it doesn't train at all: 11% accuracy, a
random guess; how badly varies from run to run). To see why, the ActivationStats callback
records the activations of every layer at every batch: most activations of the last layers end
up near zero ("dead"), so little gradient flows.

A first remedy: a larger batch size. Gradients are averaged over more items, so they're less
noisy... but there are fewer batches (fewer weight updates) per epoch.

Run:
    python 05_baseline_and_batch_size.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402

from common import ActivationStats, Learner, accuracy, outputs_dir, set_seed  # noqa: E402
from mnist_full import get_dls, simple_cnn  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)
dls = get_dls()
dls.show_batch(max_n=9, fname=OUT / "mnist_batch.png")


def fit(dls, epochs=1, lr=0.06):
    learn = Learner(dls, simple_cnn(), loss_func=F.cross_entropy, metrics=accuracy,
                    cbs=[ActivationStats(with_hist=True)])
    learn.fit(epochs, lr)
    return learn


print("batch size 64, lr 0.06:")
learn = fit(dls)
learn.activation_stats.plot_layer_stats(0, fname=OUT / "stats_first_layer.png")
learn.activation_stats.plot_layer_stats(-2, fname=OUT / "stats_penultimate_layer.png")

print("\nbatch size 512, lr 0.06:")
learn = fit(get_dls(512))
learn.activation_stats.plot_layer_stats(-2, fname=OUT / "stats_penultimate_layer_bs512.png")
