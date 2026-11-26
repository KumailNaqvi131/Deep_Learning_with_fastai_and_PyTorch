"""
Chapter 13 — Convolutional Neural Networks
Topic 3: Our first convolutional neural network; convolution arithmetic; receptive fields (book
sections "Our First Convolutional Neural Network", "Creating the CNN", "Understanding Convolution
Arithmetic", "Receptive Fields")

Replace the linear layers of Chapter 4 with convolutions. Each stride-2 conv halves the grid,
so after 4 of them a 28x28 image becomes 2x2, and one more gives 1x1 with 2 channels: the two
class scores. We double the number of filters at each stride-2 layer so the amount of computation
per layer stays roughly the same.

    def conv(ni, nf, ks=3, act=True):
        res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
        if act: res = nn.Sequential(res, nn.ReLU())
        return res
    simple_cnn = sequential(conv(1,4), conv(4,8), conv(8,16), conv(16,32), conv(32,2,act=False), Flatten())

Receptive field: the area of the input image that influences one activation. It grows with
every layer: deeper layers "see" more of the image and can detect larger, more complex features.

Run:
    python 03_first_cnn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import Flatten, ImageBlock, Learner, PILImageBW, URLs, accuracy, set_seed, untar_data  # noqa: E402
from common.vision import CategoryBlock, DataBlock, GrandparentSplitter, get_image_files, parent_label  # noqa: E402

set_seed(42)
path = untar_data(URLs.MNIST_SAMPLE)
dls = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), get_items=get_image_files,
                splitter=GrandparentSplitter(), get_y=parent_label).dataloaders(path)


def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks // 2)
    if act:
        res = nn.Sequential(res, nn.ReLU())
    return res


simple_cnn = nn.Sequential(
    conv(1, 4),               # 14x14
    conv(4, 8),               # 7x7
    conv(8, 16),              # 4x4
    conv(16, 32),             # 2x2
    conv(32, 2, act=False),   # 1x1
    Flatten(),
)
learn = Learner(dls, simple_cnn, loss_func=F.cross_entropy, metrics=accuracy)
learn.summary()
print("\nConv arithmetic: the first layer has 4 kernels of 1x3x3 weights + 4 biases =", 4 * 9 + 4, "parameters,"
      "\nbut it is applied at 14x14 positions: ~", 14 * 14 * 4 * 9, "multiply-adds per image.")

rf = 1
for i in range(5):
    rf = rf + 2 * (2 ** i)                    # each 3x3 stride-2 layer adds 2 * (product of earlier strides)
    print(f"receptive field after layer {i + 1}: {rf}x{rf} input pixels")

learn.fit_one_cycle(2, 0.01)
