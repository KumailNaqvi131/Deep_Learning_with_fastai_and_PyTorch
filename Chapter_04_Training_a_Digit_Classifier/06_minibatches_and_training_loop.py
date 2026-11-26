"""
Chapter 4 — Under the Hood: Training a Digit Classifier
Topic 6: SGD and mini-batches, putting it all together, creating an optimizer (book sections
"SGD and Mini-Batches", "Putting It All Together", "Creating an Optimizer")

Computing the loss on the whole dataset for each step is slow; on one item it is very noisy.
So we use mini-batches: a few hundred items at a time, shuffled every epoch (a DataLoader).

We build the training loop three times, each time replacing hand-written code by PyTorch/fastai:
  1. by hand: init_params, linear1, calc_grad, train_epoch, validate_epoch
  2. nn.Linear (holds the parameters) + our own BasicOptim (does the step and zero_grad)
  3. the Learner: Learner(dls, nn.Linear(28*28,1), opt_func=SGD, loss_func=mnist_loss,
     metrics=batch_accuracy).fit(10, lr=1.0)

Run:
    python 06_minibatches_and_training_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import SGD, DataLoaders, Learner  # noqa: E402
from mnist_data import load_xy  # noqa: E402

torch.manual_seed(42)
train_x, train_y = load_xy("train")
valid_x, valid_y = load_xy("valid")
dset, valid_dset = list(zip(train_x, train_y)), list(zip(valid_x, valid_y))

# ---- Mini-batches ------------------------------------------------------------------------------------
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
print("A DataLoader over 0..14, batch size 5, shuffled:", [b.tolist() for b in dl])

dl = DataLoader(dset, batch_size=256, shuffle=True)
valid_dl = DataLoader(valid_dset, batch_size=256)
xb, yb = next(iter(dl))
print("one batch:", tuple(xb.shape), tuple(yb.shape))


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


# ---- 1. Everything by hand ----------------------------------------------------------------------------
def init_params(size, std=1.0):
    return (torch.randn(size) * std).requires_grad_()


weights, bias = init_params((28 * 28, 1)), init_params(1)
params = weights, bias


def linear1(xb):
    return xb @ weights + bias


def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()


def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


print("\n1. By hand: validation accuracy after each epoch")
lr = 1.0
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=" ")

# ---- 2. nn.Linear and an optimizer ---------------------------------------------------------------------
linear_model = nn.Linear(28 * 28, 1)
w, b = linear_model.parameters()
print(f"\n\n2. nn.Linear has the same parameters: {tuple(w.shape)}, {tuple(b.shape)}")


class BasicOptim:
    def __init__(self, params, lr):
        self.params, self.lr = list(params), lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None


opt = BasicOptim(linear_model.parameters(), lr)


def train_epoch2(model):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


def train_model(model, epochs):
    for i in range(epochs):
        train_epoch2(model)
        print(validate_epoch(model), end=" ")


train_model(linear_model, 20)

# ---- 3. The Learner does all of this for us -------------------------------------------------------------
print("\n\n3. With a Learner (fastai's SGD = PyTorch's SGD without momentum):")
dls = DataLoaders(dl, valid_dl)
learn = Learner(dls, nn.Linear(28 * 28, 1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(10, lr=lr)
