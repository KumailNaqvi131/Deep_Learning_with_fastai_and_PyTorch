"""
Chapter 11 — Data Munging with fastai's Mid-Level API
Topic 1: Transforms; writing your own Transform (book sections "Going Deeper into fastai's Layered
API", "Transforms", "Writing Your Own Transform")

The data block API is the high level. Underneath, everything is made of Transforms: objects that
  * encode an item (encodes),
  * can be set up on the training data first (setups: e.g. build a vocab, compute a mean),
  * can decode the result back to something you can read or show (decodes).
Applied to a tuple (x, y), a Transform transforms each element. With a type annotation
(def encodes(self, x: int)) it only applies to that type and leaves everything else alone.

    class NormalizeMean(Transform):
        def setups(self, items): self.mean = sum(items)/len(items)
        def encodes(self, x): return x-self.mean
        def decodes(self, x): return x+self.mean

Run:
    python 01_transforms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import URLs, untar_data  # noqa: E402
from common.midlevel import Transform  # noqa: E402
from common.text import get_text_files  # noqa: E402
from text_transforms import Numericalize, Tokenizer  # noqa: E402

path = untar_data(URLs.IMDB)
files = get_text_files(path, folders=["train", "test"])
txts = [o.read_text() for o in files[:2000]]

tok = Tokenizer()
toks = [tok(t) for t in txts]
print("tok(txts[0])[:15]:", toks[0][:15])
num = Numericalize()
num.setup(toks)                               # setups: build the vocab
nums = [num(t) for t in toks]
print("num(...)[:10]:", nums[0][:10].tolist())
nums_dec = num.decode(nums[0][:10])
print("num.decode:", nums_dec, "\ntok.decode:", repr(tok.decode(nums_dec)))
print("applied to a tuple, each element is tokenized:", [t[:4] for t in tok((txts[0], txts[1]))])


# A Transform from a function, with type dispatch
def f(x: int):
    return x + 1


tfm = Transform(f)
print("\nTransform(f)(2) =", tfm(2), "   Transform(f)(2.0) =", tfm(2.0), " (not an int: unchanged)")


class NormalizeMean(Transform):
    def setups(self, items):
        self.mean = sum(items) / len(items)

    def encodes(self, x):
        return x - self.mean

    def decodes(self, x):
        return x + self.mean


tfm = NormalizeMean()
tfm.setup([1, 2, 3, 4, 5])
start = 2
y = tfm(start)
z = tfm.decode(y)
print(f"NormalizeMean: mean={tfm.mean}, encode(2)={y}, decode back={z}")
