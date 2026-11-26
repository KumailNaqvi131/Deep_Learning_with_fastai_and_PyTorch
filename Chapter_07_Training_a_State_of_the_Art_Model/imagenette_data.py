"""Shared by the Chapter 7 scripts: Imagenette DataLoaders at a given batch size and image size.

    def get_dls(bs, size):
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           get_y=parent_label,
                           item_tfms=Resize(460),
                           batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                                       Normalize.from_stats(*imagenet_stats)])
        return dblock.dataloaders(path, bs=bs)

The book uses the full-size Imagenette (images ~500px, presized to 460, trained at 224). We use the
160px version (generated offline, or `--real` to download imagenette2-160): presize to 160, train at 128.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (CategoryBlock, DataBlock, GrandparentSplitter, ImageBlock, Normalize, Resize,  # noqa: E402
                    URLs, aug_transforms, get_image_files, imagenet_stats, parent_label, untar_data)

path = untar_data(URLs.IMAGENETTE)
NAMES = {"n01440764": "tench", "n02102040": "English springer", "n03000684": "chain saw",
         "n02979186": "cassette player", "n03028079": "church", "n03394916": "French horn",
         "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball", "n03888605": "parachute"}


def get_dls(bs=64, size=128, normalize=True, presize=160):
    tfms = [*aug_transforms(size=size, min_scale=0.75)]
    if normalize:
        tfms.append(Normalize.from_stats(*imagenet_stats))
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                       splitter=GrandparentSplitter(valid_name="val"),
                       item_tfms=Resize(presize), batch_tfms=tfms)
    return dblock.dataloaders(path, bs=bs)
