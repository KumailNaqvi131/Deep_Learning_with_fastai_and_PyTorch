"""
Chapter 14 — ResNets
Topic 4: Bottleneck layers (book section of the same name)

For deeper ResNets (50 layers and more), each block is a "bottleneck": a 1x1 conv that reduces
the number of channels by 4, a 3x3 conv on that small number of channels, and a 1x1 conv that
expands it back. 1x1 convs are very fast, so we can afford many more channels (here 256..2048)
for about the same cost as the basic block. With layers=[3,4,6,3] and expansion=4: ResNet-50.

    def _conv_block(ni,nf,stride):
        return nn.Sequential(ConvLayer(ni, nf//4, 1),
                             ConvLayer(nf//4, nf//4, stride=stride),
                             ConvLayer(nf//4, nf, 1, act_cls=None, norm_type=NormType.BatchZero))
    rn = ResNet(dls.c, [3,4,6,3], 4)

(The book trains it 20 epochs on 224px images for 92.7%. Here: 5 epochs on 128px images, to keep it
quick; deeper models need longer training to shine.)

Run:
    python 04_bottleneck_layers.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import ConvLayer, Flatten, count_parameters, set_seed  # noqa: E402
from resnet_data import dls, get_learner  # noqa: E402


def _conv_block(ni, nf, stride):
    return nn.Sequential(ConvLayer(ni, nf // 4, 1), ConvLayer(nf // 4, nf // 4, stride=stride),
                         ConvLayer(nf // 4, nf, 1, act=False, zero_bn=True))


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


rn = ResNet(dls.c, [3, 4, 6, 3], 4)
print(f"ResNet-50 (bottleneck blocks): {count_parameters(rn):,} parameters; block sizes {rn.block_szs}")
print("one bottleneck block:", rn[5][0].convs)
set_seed(42)
t0 = time.time()
learn = get_learner(rn)
learn.fit_one_cycle(5, 3e-3)
print(f"{time.time() - t0:.0f}s")
