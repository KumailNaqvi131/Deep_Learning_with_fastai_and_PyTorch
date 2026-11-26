"""
Chapter 5 — Image Classification
Topic 3: Checking and debugging a DataBlock (book section of the same name)

Always look at your data before training: dls.show_batch(). If something is wrong,
DataBlock.summary(source) walks through every step for one item and tells you where it failed.
The book's example forgets the Resize, so images of different sizes can't be collated into a batch:

    pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(seed=42),
                      get_y=using_attr(RegexLabeller(r'(.+)_\\d+.jpg$'), 'name'))
    pets1.summary(path/"images")

Then train a quick *baseline* model right away: it tells you if the task is feasible and
where the difficulties are, before you spend time on details.

Run:
    python 03_checking_the_datablock.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cnn_learner, error_rate, outputs_dir, resnet34, set_seed  # noqa: E402
from pets_data import get_dls, path, pets  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)

dls = get_dls()
dls.show_batch(max_n=9, fname=OUT / "pets_show_batch.png")
print("dls.vocab has", len(dls.vocab), "breeds\n")

print("=== The DataBlock without Resize: summary() shows the problem ===")
pets1 = pets.new(item_tfms=None, batch_tfms=None)
pets1.summary(path / "images")

print("\n=== A quick baseline ===")
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
