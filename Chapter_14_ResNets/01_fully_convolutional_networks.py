"""
Chapter 14 — ResNets
Topic 1: Going back to Imagenette: fully convolutional networks (book section "Going Back to
Imagenette")

For MNIST we used enough stride-2 convolutions to shrink the image to 1x1. That ties the model to
one image size. Instead: a few stride-2 convs, then take the *average* of the final grid of
activations (adaptive average pooling), whatever its size. A fully convolutional network:
it works on images of any size, and each final activation says "how much of this feature is in
the image, on average".

    def avg_pool(x): return x.mean((2,3))
    def block(ni, nf): return ConvLayer(ni, nf, stride=2)
    def get_model():
        return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128),
                             block(128, 256), nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(256, dls.c))

Run:
    python 01_fully_convolutional_networks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import ConvLayer, Flatten, outputs_dir, set_seed  # noqa: E402
from resnet_data import dls, get_learner  # noqa: E402


def avg_pool(x):
    return x.mean((2, 3))


x = torch.randn(2, 256, 4, 4)
print("avg_pool == nn.AdaptiveAvgPool2d(1):", torch.allclose(avg_pool(x), nn.AdaptiveAvgPool2d(1)(x).flatten(1)))


def block(ni, nf):
    return ConvLayer(ni, nf, stride=2)      # conv (stride 2) -> BatchNorm -> ReLU


def get_model():
    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128), block(128, 256),
                         nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(256, dls.c))


m = get_model()
for size in (128, 160, 224):
    print(f"input {size}x{size} -> output {tuple(m(torch.randn(1, 3, size, size)).shape)}  (any size works)")

set_seed(42)
learn = get_learner(get_model())
learn.lr_find(fname=outputs_dir(__file__) / "lr_find.png")
learn.fit_one_cycle(5, 3e-3)
