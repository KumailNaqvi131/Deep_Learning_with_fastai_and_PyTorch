"""
Chapter 4 — Under the Hood: Training a Digit Classifier
Topic 7: Adding a nonlinearity; going deeper (book sections "Adding a Nonlinearity" and "Going Deeper")

Two linear layers in a row are still just one linear function. Put a nonlinearity between them
(ReLU: replace every negative number by 0) and you get a neural network that, given enough
parameters, can approximate any function (the universal approximation theorem):

    def simple_net(xb):
        res = xb@w1 + b1
        res = res.max(tensor(0.0))
        res = res@w2 + b2
        return res

    simple_net = nn.Sequential(nn.Linear(28*28,30), nn.ReLU(), nn.Linear(30,1))
    learn = Learner(dls, simple_net, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
    learn.fit(40, 0.1)

Then "going deeper": an 18-layer ResNet (trained from scratch on the images) gets close to 100%.

Run:
    python 07_adding_a_nonlinearity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import (SGD, DataLoaders, ImageDataLoaders, Learner, accuracy, cnn_learner,  # noqa: E402
                    outputs_dir, resnet18, try_import_matplotlib)
from mnist_data import load_xy, path  # noqa: E402

torch.manual_seed(42)
OUT = outputs_dir(__file__)
train_x, train_y = load_xy("train")
valid_x, valid_y = load_xy("valid")
dls = DataLoaders(DataLoader(list(zip(train_x, train_y)), batch_size=256, shuffle=True),
                  DataLoader(list(zip(valid_x, valid_y)), batch_size=256))


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def batch_accuracy(xb, yb):
    return ((xb.sigmoid() > 0.5) == yb).float().mean()


# ReLU = max(x, 0)
print("F.relu([-2,-1,0,1,2]) =", F.relu(torch.tensor([-2.0, -1, 0, 1, 2])).tolist())

# The same network written by hand and with nn.Sequential
w1, b1 = torch.randn(28 * 28, 30) * 0.05, torch.zeros(30)
w2, b2 = torch.randn(30, 1) * 0.2, torch.zeros(1)


def simple_net_by_hand(xb):
    res = xb @ w1 + b1
    res = res.max(torch.tensor(0.0))
    res = res @ w2 + b2
    return res


print("output of the hand-written net for 3 images:", simple_net_by_hand(train_x[:3]).squeeze().tolist())

simple_net = nn.Sequential(nn.Linear(28 * 28, 30), nn.ReLU(), nn.Linear(30, 1))
learn = Learner(dls, simple_net, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(40, 0.1)
print("final accuracy:", round(learn.recorder.values[-1][3], 4))

plt = try_import_matplotlib()
if plt is not None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([v[3] for v in learn.recorder.values])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    fig.savefig(OUT / "simple_net_accuracy.png", bbox_inches="tight")
    print(f"saved {OUT / 'simple_net_accuracy.png'}")

# What did the first layer learn? Each of the 30 neurons has 784 weights = a 28x28 picture
from common import show_images  # noqa: E402

w = simple_net[0].weight.detach()
show_images([((r - r.min()) / (r.max() - r.min())).view(28, 28) for r in w[:10]], [f"neuron {i}" for i in range(10)],
            OUT / "first_layer_features.png", ncols=5, cell=90)

# ---- Going deeper: an 18-layer CNN on the images ------------------------------------------------------
print("\nGoing deeper: ResNet-18 (not pretrained) on the 28x28 images, 1 epoch:")
dls_img = ImageDataLoaders.from_folder(path)
learn = cnn_learner(dls_img, resnet18, pretrained=False, loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
