"""
Chapter 2 — From Model to Production
Topic 3: Data augmentation (book section "Data Augmentation")

Data augmentation creates random variations of the training images (rotation, flipping,
perspective warping, brightness and contrast changes, ...) that still show the same thing, so the
model learns what matters and not, say, which way the bear is facing.

Because all the images in a batch have the same size after the item transforms, the augmentations
run as *batch transforms*, on the GPU, on the whole batch at once:

    bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))

mult=2 doubles the amount of augmentation, so it's easy to see. In practice:

    bears = bears.new(item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms())

Run:
    python 03_data_augmentation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403,E402

OUT = outputs_dir(__file__)
path = untar_data(URLs.BEARS)
for f in verify_images(get_image_files(path)):
    f.unlink()

bears = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42), get_y=parent_label)

bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.show_batch(max_n=8, unique=True, fname=OUT / "augmentation_mult2.png")
print("aug_transforms gives:", dls.train.after_batch[1:])

bears = bears.new(item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms())
dls = bears.dataloaders(path)
dls.show_batch(max_n=8, unique=True, fname=OUT / "augmentation_default.png")

# The validation set is NOT augmented (we want to measure performance on normal images):
dls.show_batch(max_n=4, ds_idx=1, fname=OUT / "validation_not_augmented.png")
