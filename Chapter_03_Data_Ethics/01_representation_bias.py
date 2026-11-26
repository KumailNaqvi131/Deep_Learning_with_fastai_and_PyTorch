"""
Chapter 3 — Data Ethics
Topic 1: Bias: representation bias and aggregate metrics (book section "Bias")

The book describes facial-recognition systems with error rates near 0% for lighter-skinned men but
up to 35% for darker-skinned women (Gender Shades): the training data contained few examples of
some groups, and the error was hidden by a good *overall* accuracy.

The same effect, simulated: a classifier is trained on data where group B makes up only 5% of
the examples, and where the relationship between features and label is a little different for
group B. The overall accuracy looks fine. Only when we break the metric down by group do we see
that the model fails group B. Then we retrain with a balanced dataset.

Lesson: always evaluate your model separately on every subgroup that matters.

Run:
    python 01_representation_bias.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import set_seed  # noqa: E402

set_seed(0)


def make_group(n, group):
    """2 features -> label. For group B the decision boundary is rotated (a different relationship)."""
    x = torch.randn(n, 2)
    w = torch.tensor([1.0, 1.0]) if group == "A" else torch.tensor([1.0, -1.0])
    y = ((x @ w) > 0).float()
    g = torch.full((n,), 0 if group == "A" else 1)
    return x, y, g


def dataset(n_a, n_b):
    xa, ya, ga = make_group(n_a, "A")
    xb, yb, gb = make_group(n_b, "B")
    x = torch.cat([xa, xb])
    g = torch.cat([ga, gb]).float()
    return torch.cat([x, g.unsqueeze(1)], 1), torch.cat([ya, yb]), g   # the model also sees the group


def train(x, y):
    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(300):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def report(name, model):
    x, y, g = dataset(5000, 5000)                       # a test set with BOTH groups well represented
    pred = (model(x).squeeze(1) > 0).float()
    acc = lambda m: (pred[m] == y[m]).float().mean().item()  # noqa: E731
    pop = 0.95 * acc(g == 0) + 0.05 * acc(g == 1)      # accuracy on a population that is 95% group A
    print(f"{name}\n   accuracy on the 95%/5% population: {pop:.3f}   group A: {acc(g == 0):.3f}   group B: {acc(g == 1):.3f}")


x, y, g = dataset(1900, 100)                             # group B = 5% of the training data
report("Trained on 95% A / 5% B:", train(x, y))
x, y, g = dataset(1000, 1000)
report("Trained on a balanced dataset:", train(x, y))
print("\nThe headline number hid the failure on group B. Collecting representative data fixed it.")
