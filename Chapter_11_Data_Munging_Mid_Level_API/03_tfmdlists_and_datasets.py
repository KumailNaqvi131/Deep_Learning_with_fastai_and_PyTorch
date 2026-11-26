"""
Chapter 11 — Data Munging with fastai's Mid-Level API
Topic 3: TfmdLists and Datasets: transformed collections (book section of the same name)

  * TfmdLists(items, tfms, splits): a Pipeline applied lazily to a list of items. With splits,
    the transforms are set up on the training items only (e.g. no vocab from validation texts),
    and .train / .valid give the two subsets.
  * Datasets(items, [x_tfms, y_tfms], splits): several pipelines in parallel on the same items,
    so dsets[i] is a tuple (x, y).
  * .dataloaders(bs, before_batch=pad_input): batches; texts of different lengths are padded.

    x_tfms = [Tokenizer.from_folder(path), Numericalize]
    y_tfms = [parent_label, Categorize()]
    dsets = Datasets(files, [x_tfms, y_tfms], splits=splits)
    dls = dsets.dataloaders(bs=64, before_batch=pad_input)

Run:
    python 03_tfmdlists_and_datasets.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import URLs, parent_label, untar_data  # noqa: E402
from common.midlevel import Categorize, Datasets, TfmdLists, pad_input  # noqa: E402
from common.text import get_text_files  # noqa: E402
from text_transforms import Numericalize, Tokenizer  # noqa: E402

path = untar_data(URLs.IMDB)
files = get_text_files(path, folders=["train", "test"])
random.Random(0).shuffle(files)
files = files[:3000]

tls = TfmdLists(files, [Tokenizer(), Numericalize()])
t = tls[0]
print("tls[0][:20] =", t[:20].tolist())
print("tls.decode(t)[:100] =", repr(tls.decode(t)[:100]))

cut = int(len(files) * 0.8)
splits = [list(range(cut)), list(range(cut, len(files)))]
tls = TfmdLists(files, [Tokenizer(), Numericalize()], splits=splits)
print(f"\nwith splits: train {len(tls.train)}, valid {len(tls.valid)}; tls.valid[0][:10] =", tls.valid[0][:10].tolist())

lbls = [parent_label(f) for f in files]
cat = Categorize()
cat.setup(lbls)
print("\nCategorize: vocab", cat.vocab, "| cat(lbls[0]) =", cat(lbls[0]), "| decode:", cat.decode(cat(lbls[0])))
tls_y = TfmdLists(files, [parent_label, Categorize()])
print("TfmdLists for the labels: tls_y[0] =", tls_y[0])

x_tfms = [Tokenizer(), Numericalize()]
y_tfms = [parent_label, Categorize()]
dsets = Datasets(files, [x_tfms, y_tfms], splits=splits)
x, y = dsets.valid[0]
print("\nDatasets: x[:10] =", x[:10].tolist(), " y =", y)
print("decoded:", tuple(str(o)[:60] for o in dsets.decode((x, y))))

dls = dsets.dataloaders(bs=64, before_batch=pad_input)
xb, yb = dls.one_batch()
print(f"\nOne padded batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}; padding tokens in it: {(xb == 1).sum().item()}")
