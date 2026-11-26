"""
Chapter 19 — A fastai Learner from Scratch
Topic 1: Data, Dataset, DataLoader, normalization (book sections "Data", "Dataset")

Everything from scratch, using only PyTorch tensors (and F.conv2d later). The data: Imagenette,
images resized to 64x64.
  * Dataset: __len__ and __getitem__ (an image tensor and its label index)
  * collate: turn a list of indices into a batch (stack the tensors)
  * DataLoader: shuffle the indices, cut them into batches, collate each batch, several batches
    in parallel. The book uses a ProcessPoolExecutor (separate processes). Forking processes after
    PyTorch has started its own CPU threads can deadlock on some systems, and Windows/macOS can't
    fork at all, so we use a ThreadPoolExecutor: Pillow releases Python's GIL while it decodes and
    resizes images, so threads run in parallel here too.
  * Normalize with statistics computed on a batch, and permute to channels-first [bs, 3, 64, 64]

The other scripts of this chapter import what this one defines (that's why the demo is inside
`if __name__ == "__main__":`).

Run:
    python 01_data_and_dataset.py
"""

import random
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import URLs, get_device, get_image_files, untar_data  # noqa: E402

path = untar_data(URLs.IMAGENETTE_160)
files = get_image_files(path)
lbls = sorted({f.parent.name for f in files})
v2i = {v: i for i, v in enumerate(lbls)}


class Dataset:
    def __init__(self, fns):
        self.fns = fns

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, i):
        im = Image.open(self.fns[i]).resize((64, 64)).convert("RGB")
        y = v2i[self.fns[i].parent.name]
        return torch.tensor(np.array(im)).float() / 255, torch.tensor(y)


train = [f for f in files if f.parent.parent.name == "train"]
valid = [f for f in files if f.parent.parent.name != "train"]
train_ds, valid_ds = Dataset(train), Dataset(valid)


def collate(idxs, ds):
    xb, yb = zip(*[ds[i] for i in idxs])
    return torch.stack(xb), torch.stack(yb)


class DataLoader:
    def __init__(self, ds, bs=128, shuffle=False, n_workers=1):
        self.ds, self.bs, self.shuffle, self.n_workers = ds, bs, shuffle, n_workers

    def __len__(self):
        return (len(self.ds) - 1) // self.bs + 1

    def __iter__(self):
        idxs = list(range(len(self.ds)))
        if self.shuffle:
            random.shuffle(idxs)
        chunks = [idxs[n:n + self.bs] for n in range(0, len(self.ds), self.bs)]
        if self.n_workers <= 1:
            yield from map(partial(collate, ds=self.ds), chunks)
        else:                                            # several batches prepared in parallel threads
            with ThreadPoolExecutor(self.n_workers) as ex:
                yield from ex.map(partial(collate, ds=self.ds), chunks)


n_workers = min(8, torch.multiprocessing.cpu_count())
train_dl = DataLoader(train_ds, bs=32, shuffle=True, n_workers=n_workers)   # (book: bs=128 for 3.6x more images)
valid_dl = DataLoader(valid_ds, bs=256, shuffle=False, n_workers=n_workers)


class Normalize:
    def __init__(self, stats):
        self.stats = stats

    def __call__(self, x):
        if x.device != self.stats[0].device:
            self.stats = [s.to(x.device) for s in self.stats]
        return (x - self.stats[0]) / self.stats[1]


_xb, _ = collate(list(range(0, len(train_ds), max(1, len(train_ds) // 256)))[:256], train_ds)
stats = [_xb.mean((0, 1, 2)), _xb.std((0, 1, 2))]
norm = Normalize(stats)


def tfm_x(x):
    return norm(x).permute((0, 3, 1, 2))           # [bs, 64, 64, 3] -> [bs, 3, 64, 64]


device = get_device()

if __name__ == "__main__":
    print(f"{len(files)} files, {len(lbls)} classes: {lbls}")
    print(f"train: {len(train_ds)}, valid: {len(valid_ds)}")
    x, y = train_ds[0]
    print("one item:", tuple(x.shape), y, "-> class", lbls[int(y)])
    xb, yb = next(iter(train_dl))
    print(f"one batch from our DataLoader ({n_workers} worker threads):", tuple(xb.shape), tuple(yb.shape))
    print("per-channel mean/std of a sample:", [round(v, 3) for v in stats[0].tolist()], [round(v, 3) for v in stats[1].tolist()])
    t = tfm_x(xb)
    print("after tfm_x:", tuple(t.shape), "mean", [round(v, 2) for v in t.mean((0, 2, 3)).tolist()],
          "std", [round(v, 2) for v in t.std((0, 2, 3)).tolist()])
