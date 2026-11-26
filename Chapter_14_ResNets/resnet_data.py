"""Shared by the Chapter 14 scripts: Imagenette (160px, presized to 160, trained at 128) and a Learner.

    def get_data(url, presize, resize):
        path = untar_data(url)
        return DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
                         splitter=GrandparentSplitter(valid_name='val'), get_y=parent_label,
                         item_tfms=Resize(presize),
                         batch_tfms=[*aug_transforms(min_scale=0.5, size=resize),
                                     Normalize.from_stats(*imagenet_stats)],
                         ).dataloaders(path, bs=128)
    def get_learner(m):
        return Learner(dls, m, loss_func=nn.CrossEntropyLoss(), metrics=accuracy).to_fp16()

We use bs=64: our synthetic Imagenette has 2,600 training images instead of 9,469, so a smaller batch
size gives a number of optimizer steps per epoch closer to the book's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch import nn  # noqa: E402

from common import (CategoryBlock, DataBlock, GrandparentSplitter, ImageBlock, Learner, Normalize, Resize,  # noqa: E402
                    URLs, accuracy, aug_transforms, get_image_files, imagenet_stats, parent_label, untar_data)


def get_data(url=URLs.IMAGENETTE_160, presize=160, resize=128):
    path = untar_data(url)
    return DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
                     splitter=GrandparentSplitter(valid_name="val"), get_y=parent_label,
                     item_tfms=Resize(presize),
                     batch_tfms=[*aug_transforms(min_scale=0.5, size=resize), Normalize.from_stats(*imagenet_stats)],
                     ).dataloaders(path, bs=64)


dls = get_data()


def get_learner(m):
    return Learner(dls, m, loss_func=nn.CrossEntropyLoss(), metrics=accuracy).to_fp16()
