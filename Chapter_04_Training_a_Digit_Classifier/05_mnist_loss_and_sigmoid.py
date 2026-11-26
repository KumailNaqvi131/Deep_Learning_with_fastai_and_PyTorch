"""
Chapter 4 — Under the Hood: Training a Digit Classifier
Topic 5: The MNIST loss function and sigmoid (book sections "The MNIST Loss Function" and "Sigmoid")

The model: one weight per pixel and a bias, prediction = x @ weights + bias (a matrix
multiplication for a whole batch). To learn, we need a loss whose gradient tells each weight how
to change. Why not use the accuracy? Because the accuracy only changes when a prediction flips
from 3 to 7: a tiny change of a weight almost never changes it, so its gradient is 0 almost
everywhere and nothing gets learned.

Instead, a smooth loss:
    def mnist_loss(predictions, targets):
        predictions = predictions.sigmoid()                      # squash into 0..1
        return torch.where(targets==1, 1-predictions, predictions).mean()
(how far each prediction is from the right answer).

Run:
    python 05_mnist_loss_and_sigmoid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from mnist_data import load_xy  # noqa: E402

train_x, train_y = load_xy("train")
print("train_x:", tuple(train_x.shape), " train_y:", tuple(train_y.shape), "(1 = three, 0 = seven)")
dset = list(zip(train_x, train_y))                     # a Dataset: (x, y) pairs
x, y = dset[0]
print("one item:", tuple(x.shape), y)


def init_params(size, std=1.0):
    return (torch.randn(size) * std).requires_grad_()


torch.manual_seed(0)
weights = init_params((28 * 28, 1))
bias = init_params(1)
print("\nprediction for one image (weights * pixels, summed, + bias):", ((train_x[0] * weights.T).sum() + bias).item())


def linear1(xb):
    return xb @ weights + bias                           # @ = matrix multiplication, a whole batch at once


preds = linear1(train_x)
print("predictions for all images:", tuple(preds.shape))
corrects = (preds > 0.0).float() == train_y
print(f"accuracy with random weights: {corrects.float().mean().item():.4f}")

# Why accuracy can't be the loss: nudge one weight a little
with torch.no_grad():
    weights[0] *= 1.0001
preds2 = linear1(train_x)
print("accuracy after nudging a weight:", ((preds2 > 0.0).float() == train_y).float().mean().item(),
      "-> unchanged, so its gradient would be zero")

# ---- A smooth loss ---------------------------------------------------------------------------------
trgts = torch.tensor([1, 0, 1])
prds = torch.tensor([0.9, 0.4, 0.2])
print("\ntorch.where(trgts==1, 1-prds, prds) =", torch.where(trgts == 1, 1 - prds, prds).tolist(),
      "-> mean", torch.where(trgts == 1, 1 - prds, prds).mean().item())


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


print("sigmoid of [-5, -1, 0, 1, 5] =", [round(v, 4) for v in sigmoid(torch.tensor([-5.0, -1, 0, 1, 5])).tolist()],
      "(always between 0 and 1)")


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


loss = mnist_loss(linear1(train_x), train_y)
loss.backward()
print(f"\nmnist_loss with random weights: {loss.item():.4f}")
print("gradient shapes:", tuple(weights.grad.shape), tuple(bias.grad.shape),
      "| mean |gradient| of the weights:", weights.grad.abs().mean().item(), "(not zero: we can learn!)")
print("\nLoss vs metric: the LOSS is for the computer (smooth, used for gradients), the METRIC (accuracy) is for us.")
