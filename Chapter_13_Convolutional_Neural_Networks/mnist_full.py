"""Shared by Chapter 13 scripts 05-07: the full MNIST (all ten digits) and the simple CNN.

    def get_dls(bs=64):
        return DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                         get_items=get_image_files,
                         splitter=GrandparentSplitter('training','testing'),
                         get_y=parent_label,
                         batch_tfms=Normalize()).dataloaders(path, bs=bs)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch import nn  # noqa: E402

from common import (CategoryBlock, DataBlock, Flatten, GrandparentSplitter, ImageBlock, Normalize,  # noqa: E402
                    PILImageBW, URLs, get_image_files, parent_label, untar_data)

path = untar_data(URLs.MNIST)


def get_dls(bs=64):
    return DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), get_items=get_image_files,
                     splitter=GrandparentSplitter("training", "testing"), get_y=parent_label,
                     batch_tfms=Normalize()).dataloaders(path, bs=bs)


def conv(ni, nf, ks=3, act=True, bn=False):
    layers = [nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks // 2)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    if act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def simple_cnn(bn=False):
    return nn.Sequential(
        conv(1, 8, ks=5, bn=bn),        # 14x14  (a 5x5 kernel: 25 pixels -> 8 features forces it to learn)
        conv(8, 16, bn=bn),             # 7x7
        conv(16, 32, bn=bn),            # 4x4
        conv(32, 64, bn=bn),            # 2x2
        conv(64, 10, act=False, bn=bn),  # 1x1
        Flatten(),
    )
