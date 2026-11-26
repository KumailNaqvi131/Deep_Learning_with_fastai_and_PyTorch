"""
Chapter 14 — ResNets
Topic 3: A state-of-the-art ResNet (book section of the same name)

The "bag of tricks" ResNet (He et al., 2018):
  * a stem: the first few layers do most of the computation (big images, few channels), so use
    plain convs there (3x3 convs with 32, 32, 64 channels, the first with stride 2) and a max pool,
  * then 4 groups of ResBlocks with 64, 128, 256, 512 channels; the first block of each group
    (except the first) has stride 2,
  * layers=[2,2,2,2] gives ResNet-18, [3,4,6,3] ResNet-34.

    def _resnet_stem(*sizes):
        return [ConvLayer(sizes[i], sizes[i+1], 3, stride = 2 if i==0 else 1)
                for i in range(len(sizes)-1)] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

Run:
    python 03_state_of_the_art_resnet.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import ConvLayer, Flatten, count_parameters, set_seed  # noqa: E402
from resnet_data import dls, get_learner  # noqa: E402


def _conv_block(ni, nf, stride):
    return nn.Sequential(ConvLayer(ni, nf, stride=stride), ConvLayer(nf, nf, act=False, zero_bn=True))


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride)
        self.idconv = nn.Identity() if ni == nf else ConvLayer(ni, nf, 1, act=False)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


def _resnet_stem(*sizes):
    return [ConvLayer(sizes[i], sizes[i + 1], 3, stride=2 if i == 0 else 1)
            for i in range(len(sizes) - 1)] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]


class ResNet(nn.Sequential):
    def __init__(self, n_out, layers, expansion=1):
        stem = _resnet_stem(3, 32, 32, 64)
        self.block_szs = [64, 64, 128, 256, 512]
        for i in range(1, 5):
            self.block_szs[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]
        super().__init__(*stem, *blocks, nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(self.block_szs[-1], n_out))

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_szs[idx:idx + 2]
        return nn.Sequential(*[ResBlock(ch_in if i == 0 else ch_out, ch_out, stride if i == 0 else 1)
                               for i in range(n_layers)])


print("the stem:", nn.Sequential(*_resnet_stem(3, 32, 32, 64)))
rn = ResNet(dls.c, [2, 2, 2, 2])
print(f"\nResNet-18: {count_parameters(rn):,} parameters")
set_seed(42)
learn = get_learner(rn)
learn.fit_one_cycle(5, 3e-3)
