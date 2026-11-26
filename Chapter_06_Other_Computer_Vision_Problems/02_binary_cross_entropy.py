"""
Chapter 6 — Other Computer Vision Problems
Topic 2: Binary cross-entropy and multi-label accuracy (book section "Binary Cross Entropy")

Softmax + cross-entropy wants exactly one class per image. For multi-label we need each of the
20 activations to say yes/no *independently*: apply a sigmoid to each, and use binary
cross-entropy (the mnist_loss of Chapter 4, made logarithmic) on each, then average.

    def binary_cross_entropy(inputs, targets):
        inputs = inputs.sigmoid()
        return -torch.where(targets==1, inputs, 1-inputs).log().mean()

PyTorch: F.binary_cross_entropy_with_logits / nn.BCEWithLogitsLoss (sigmoid included, more
numerically stable). The accuracy must also change: compare each sigmoid output to a threshold.

    def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
        if sigmoid: inp = inp.sigmoid()
        return ((inp>thresh)==targ.bool()).float().mean()

Run:
    python 02_binary_cross_entropy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

from common import *  # noqa: F401,F403,E402

set_seed(42)
path = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(path / "train.csv")
dls = DataBlock(blocks=(ImageBlock, MultiCategoryBlock), splitter=ColSplitter("is_valid"),
                get_x=ColReader("fname", pref=path / "train"), get_y=ColReader("labels", label_delim=" "),
                item_tfms=RandomResizedCrop(128, min_scale=0.35)).dataloaders(df)

learn = cnn_learner(dls, resnet18)
x, y = dls.train.one_batch()
activs = learn.model(x)
print("activations for one batch:", tuple(activs.shape), "(64 images x 20 classes)")
print("activations of the first image:", [round(v, 2) for v in activs[0][:6].tolist()], "...")


def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets == 1, inputs, 1 - inputs).log().mean()


print("\nour binary_cross_entropy:          ", binary_cross_entropy(activs, y).item())
print("F.binary_cross_entropy_with_logits:", F.binary_cross_entropy_with_logits(activs, y).item())
print("nn.BCEWithLogitsLoss():            ", nn.BCEWithLogitsLoss()(activs, y).item())
print("the DataLoaders' default loss:      ", dls.loss_func)


# partial: fix some arguments of a function
def say_hello(name, say_what="Hello"):
    return f"{say_what} {name}."


f = partial(say_hello, say_what="Bonjour")
print("\npartial:", say_hello("Jeremy"), "|", f("Jeremy"), "|", f("Sylvain"))

print("accuracy_multi on random activations, thresh=0.5:", accuracy_multi(activs, y).item())
print("accuracy_multi with partial(accuracy_multi, thresh=0.2):", partial(accuracy_multi, thresh=0.2)(activs, y).item())
