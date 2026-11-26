"""
Chapter 6 — Other Computer Vision Problems
Topic 1: Multi-label classification: the data, pandas, and constructing a DataBlock (book sections
"Multi-Label Classification", "The Data", "Constructing a DataBlock")

Multi-label: each image can have zero, one or several labels (PASCAL 2007: person, car, dog, ...).
That's also the honest answer to "what if there's no bear in the picture?" from Chapter 2: with
multi-label classification the model can predict *none* of the classes.

The labels are in a CSV file, read with pandas: fname | labels (space-separated) | is_valid.
The DataBlock is built step by step:

    def get_x(r): return path/'train'/r['fname']
    def get_y(r): return r['labels'].split(' ')
    def splitter(df):
        train = df.index[~df['is_valid']].tolist()
        valid = df.index[df['is_valid']].tolist()
        return train,valid
    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                       splitter=splitter, get_x=get_x, get_y=get_y,
                       item_tfms = RandomResizedCrop(128, min_scale=0.35))
    dls = dblock.dataloaders(df)

Run:
    python 01_multilabel_data_and_datablock.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import *  # noqa: F401,F403,E402

OUT = outputs_dir(__file__)
path = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(path / "train.csv")

# ---- pandas basics -----------------------------------------------------------------------------------
print(df.head(), "\n")
print("df.iloc[:,0] (first column):", df.iloc[:, 0].head(3).tolist())
print("df.iloc[0,:] (first row):", df.iloc[0, :].tolist())
print("df['fname'] (a column by name):", df["fname"].head(3).tolist())
tmp_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
tmp_df["c"] = tmp_df["a"] + tmp_df["b"]                 # a new column, computed on whole columns at once
print("a new column:\n", tmp_df)


# ---- Constructing the DataBlock step by step ------------------------------------------------------------
def get_x(r):
    return path / "train" / r["fname"]


def get_y(r):
    return r["labels"].split(" ")


def splitter(items):
    """Use the dataset's own is_valid column (items are the DataFrame's rows)."""
    flags = [bool(r["is_valid"]) for r in items]
    return [i for i, f in enumerate(flags) if not f], [i for i, f in enumerate(flags) if f]


dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock), splitter=splitter, get_x=get_x, get_y=get_y,
                   item_tfms=RandomResizedCrop(128, min_scale=0.35))
train_ds, valid_ds = dblock.datasets(df)
print(f"\ntrain: {len(train_ds)} images, valid: {len(valid_ds)} images")
x, y = train_ds[0]
print("first item: image", tuple(x.shape), "| labels", get_y(df.iloc[0]), "| one-hot encoded:", y.int().tolist())
print("vocab:", dblock.vocab)
print("decoding the one-hot vector:", [dblock.vocab[i] for i in torch.where(y == 1)[0]])

dls = dblock.dataloaders(df)
dls.show_batch(max_n=9, fname=OUT / "pascal_batch.png")
