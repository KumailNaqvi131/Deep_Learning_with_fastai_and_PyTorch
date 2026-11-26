"""
Chapter 11 — Data Munging with fastai's Mid-Level API
Topic 4: Applying the mid-level API: SiamesePair (book section of the same name)

A task the data block API can't express easily: given TWO pet images, are they the same breed?
The mid-level API makes it a few lines:
  * SiameseImage: a tuple (img1, img2, same_breed) that knows how to show itself,
  * SiameseTransform: for each file, draw a second file (same breed with probability 0.5) and
    build a SiameseImage; for the validation set the draw is fixed once, so validation is stable,
  * TfmdLists(files, tfm, splits) + dataloaders(after_item=[resize, to tensor], after_batch=[...]).
Chapter 15 trains a model on it.

Run:
    python 04_siamese_pair.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import (Normalize, PILImage, RandomSplitter, RegexLabeller, Resize, URLs, get_image_files,  # noqa: E402
                    imagenet_stats, outputs_dir, show_images, untar_data)
from common.midlevel import TfmdLists, Transform  # noqa: E402
from common.vision import _to_tensor  # noqa: E402

OUT = outputs_dir(__file__)
path = untar_data(URLs.PETS)
files = get_image_files(path / "images")
labeller = RegexLabeller(r"^(.*)_\d+.jpg$")


def label_func(fname):
    return labeller(fname.name)


class SiameseImage(tuple):
    """(img1, img2, same) that can show itself."""

    def show(self, fname):
        img1, img2, same = self
        return show_images([img1, img2], [f"same breed: {bool(same)}", ""], fname, ncols=2)


class SiameseTransform(Transform):
    def __init__(self, files, label_func, splits):
        super().__init__()
        self.labels = sorted(set(label_func(f) for f in files))
        self.lbl2files = {lbl: [f for f in files if label_func(f) == lbl] for lbl in self.labels}
        self.label_func = label_func
        self.valid = {f: self._draw(f) for f in [files[i] for i in splits[1]]}   # fixed for validation

    def encodes(self, f: Path):
        f2, t = self.valid.get(f, None) or self._draw(f)
        img1, img2 = PILImage.create(f), PILImage.create(f2)
        return SiameseImage((img1, img2, t))

    def _draw(self, f):
        same = random.random() < 0.5
        cls = self.label_func(f)
        if not same:
            cls = random.choice([lbl for lbl in self.labels if lbl != cls])
        return random.choice(self.lbl2files[cls]), same


img = PILImage.create(files[0])
SiameseImage((img, img, True)).show(OUT / "siamese_same.png")

splits = RandomSplitter(seed=42)(files)
tfm = SiameseTransform(files, label_func, splits)
tls = TfmdLists(files, tfm, splits=splits)
for i in range(3):
    tls.valid[i].show(OUT / f"siamese_valid_{i}.png")


# after_item: resize both images and turn everything into tensors; after_batch: normalise on the GPU
class SiameseToTensor(Transform):
    whole_tuple = True

    def encodes(self, s: SiameseImage):
        img1, img2, same = s
        r = Resize(224)
        a = _to_tensor(r([img1], ["image"], False), ["image"])[0]
        b = _to_tensor(r([img2], ["image"], False), ["image"])[0]
        return a, b, torch.tensor(int(same))


norm = Normalize.from_stats(*imagenet_stats)


class SiameseNormalize(Transform):
    whole_tuple = True

    def encodes(self, b: tuple):
        x1, x2, y = b
        x1, x2 = norm((x1.float() / 255,), ["image"], False)[0], norm((x2.float() / 255,), ["image"], False)[0]
        return x1, x2, y


dls = tls.dataloaders(bs=32, after_item=[SiameseToTensor()], after_batch=[SiameseNormalize()], n_inp=2)
x1, x2, y = dls.one_batch()
print(f"one batch: two image tensors {tuple(x1.shape)} {tuple(x2.shape)} and targets {y[:10].tolist()}")
print(f"{len(tls.train)} training pairs, {len(tls.valid)} validation pairs. Pictures saved in outputs/.")
