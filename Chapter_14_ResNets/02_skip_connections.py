"""
Chapter 14 — ResNets
Topic 2: Building a modern CNN: skip connections (book sections "Building a Modern CNN: ResNet",
"Skip Connections")

He et al. (2015) noticed that a deeper network (56 layers) trained *worse* than a shallower one
(20 layers), even on the training set, although the deep one could in theory copy the shallow
one and make its extra layers do nothing. Their fix: make "doing nothing" easy.
A residual block computes  x + convs(x)  (the "skip connection"): if convs(x) learns to output 0,
the block is the identity. Initialising the last BatchNorm of each block to 0 (zero_bn) means
every block *starts* as the identity. The convs only have to learn the residual: the difference
between the identity and the desired output. When the shape changes (stride 2 or more channels),
the identity path uses an average pool and a 1x1 conv.

    class ResBlock(Module):
        def __init__(self, ni, nf, stride=1):
            self.convs = _conv_block(ni,nf,stride)
            self.idconv = noop if ni==nf else ConvLayer(ni, nf, 1, act_cls=None)
            self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
        def forward(self, x):
            return F.relu(self.convs(x) + self.idconv(self.pool(x)))

(Here ConvLayer(..., act=False) is fastai's act_cls=None and zero_bn=True its norm_type=NormType.BatchZero.)

What you'll see: with the zero-initialised BatchNorm, our 5-block network starts as a stack of
1x1 convs and pools and needs many steps before its 3x3 convs "switch on": on this small
dataset and a 5-epoch schedule it ends up *behind* the plain CNN of 01_*.py. With the usual
BatchNorm init (gamma=1) the same ResNet beats the plain CNN clearly. Zero-init pays off for very
deep networks (ResNet-50 and beyond, 03_* and 04_*), where it makes the start of training stable.

Run:
    python 02_skip_connections.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import ConvLayer, Flatten, set_seed  # noqa: E402
from resnet_data import dls, get_learner  # noqa: E402


def _conv_block(ni, nf, stride, zero_bn=True):
    return nn.Sequential(ConvLayer(ni, nf, stride=stride), ConvLayer(nf, nf, act=False, zero_bn=zero_bn))


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, zero_bn=True):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride, zero_bn)
        self.idconv = nn.Identity() if ni == nf else ConvLayer(ni, nf, 1, act=False)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


# Thanks to zero_bn, a new block (with ni == nf, stride 1) is the identity (followed by a ReLU)
b = ResBlock(8, 8).eval()
x = torch.randn(2, 8, 16, 16).relu()
print("a freshly initialised ResBlock is the identity:", torch.allclose(b(x), x))


def get_model(block):
    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128), block(128, 256),
                         nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(256, dls.c))


results = {}
for name, block in [
        ("5 ResBlocks, zero-init BatchNorm (fastai's default)", lambda ni, nf: ResBlock(ni, nf, stride=2)),
        ("5 ResBlocks, normal BatchNorm init", lambda ni, nf: ResBlock(ni, nf, stride=2, zero_bn=False)),
        ("5 x (stride-2 ResBlock + stride-1 ResBlock), normal init: twice as deep",
         lambda ni, nf: nn.Sequential(ResBlock(ni, nf, stride=2, zero_bn=False), ResBlock(nf, nf, zero_bn=False)))]:
    print(f"\n{name}:")
    set_seed(42)
    learn = get_learner(get_model(block))
    learn.fit_one_cycle(5, 3e-3)
    results[name] = learn.recorder.values[-1][3]
print("\nfinal accuracies (compare with the plain CNN of 01_fully_convolutional_networks.py):")
for name, acc in results.items():
    print(f"   {acc:.3f}  {name}")
