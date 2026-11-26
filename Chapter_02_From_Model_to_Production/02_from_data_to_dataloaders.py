"""
Chapter 2 — From Model to Production
Topic 2: From data to DataLoaders: the DataBlock API and resizing (book section "From Data to DataLoaders")

To turn files into DataLoaders we tell fastai four things:
  * what kinds of data we have                  blocks=(ImageBlock, CategoryBlock)
  * how to get the list of items                get_items=get_image_files
  * how to label them                           get_y=parent_label        (the folder name)
  * how to create the validation set            splitter=RandomSplitter(valid_pct=0.2, seed=42)
plus item transforms (run on each image) so all images in a batch have the same size.

Resize options (look at the saved pictures):
  * crop    : crop to a square, then resize       (may lose important details at the edges)
  * squish  : squash/stretch to a square           (distorts shapes)
  * pad     : add zeros/reflection around          (wastes computation on empty pixels)
  * RandomResizedCrop: a different random part of the image every epoch (what we use in practice)

Run:
    python 02_from_data_to_dataloaders.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403,E402

OUT = outputs_dir(__file__)
path = untar_data(URLs.BEARS)
for f in verify_images(get_image_files(path)):          # (in case 01_gathering_data.py wasn't run)
    f.unlink()

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

dls = bears.dataloaders(path)
print("classes:", dls.vocab, "| train:", len(dls.train_ds), "valid:", len(dls.valid_ds))
xb, yb = dls.one_batch()
print("one batch:", tuple(xb.shape), tuple(yb.shape), "labels:", yb[:8].tolist())
dls.show_batch(max_n=4, ds_idx=1, fname=OUT / "resize_crop.png")

bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
bears.dataloaders(path).show_batch(max_n=4, ds_idx=1, fname=OUT / "resize_squish.png")

bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode="zeros"))
bears.dataloaders(path).show_batch(max_n=4, ds_idx=1, fname=OUT / "resize_pad.png")

# RandomResizedCrop: each time we look at the same image, we see a different part of it
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
bears.dataloaders(path).show_batch(max_n=4, unique=True, fname=OUT / "random_resized_crop.png")
print("Compare the four pictures in outputs/.")
