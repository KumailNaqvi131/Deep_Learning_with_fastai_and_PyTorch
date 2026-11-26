"""Shared by the Chapter 5 scripts: the pet-breeds DataBlock from the book.

    pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                     get_items=get_image_files,
                     splitter=RandomSplitter(seed=42),
                     get_y=using_attr(RegexLabeller(r'(.+)_\\d+.jpg$'), 'name'),
                     item_tfms=Resize(460),
                     batch_tfms=aug_transforms(size=224, min_scale=0.75))
    dls = pets.dataloaders(path/"images")
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (CategoryBlock, DataBlock, ImageBlock, RandomSplitter, RegexLabeller, Resize,  # noqa: E402
                    URLs, aug_transforms, get_image_files, untar_data, using_attr)

path = untar_data(URLs.PETS)

pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r"(.+)_\d+.jpg$"), "name"),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))


def get_dls(bs=64):
    return pets.dataloaders(path / "images", bs=bs)
