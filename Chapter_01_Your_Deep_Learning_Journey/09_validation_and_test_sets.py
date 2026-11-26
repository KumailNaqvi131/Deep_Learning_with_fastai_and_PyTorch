"""
Chapter 1 — Your Deep Learning Journey
Topic 9: Validation sets and test sets (book sections "Validation Sets and Test Sets" and
"Use Judgment in Defining Test Sets")

  * Training set:   used to fit the parameters.
  * Validation set: used to see how well the model generalises, and to choose hyperparameters
                    (model size, epochs, learning rate, ...). Because WE use it to make choices,
                    we can overfit to it too.
  * Test set:       locked away and used once, at the very end.

And the book's key warning: a *random* split is not always right. For time series, the
validation set should be the most recent period, because that is what the model will face in
production (predicting the future from the past).

This script shows (1) overfitting: training error keeps falling as the model gets more flexible,
validation error doesn't; (2) a random split of a time series gives an over-optimistic error,
a time-based split gives the honest one.

Run:
    python 09_validation_and_test_sets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import outputs_dir, set_seed, try_import_matplotlib  # noqa: E402

set_seed(1)
OUT = outputs_dir(__file__)

# ---- 1. Overfitting: fit polynomials of increasing degree --------------------------------------
x = torch.rand(40) * 2 - 1
y = torch.sin(3 * x) + 0.25 * torch.randn(40)
x_tr, y_tr, x_va, y_va = x[:25], y[:25], x[25:], y[25:]


def fit_poly(degree):
    X = torch.stack([x_tr ** k for k in range(degree + 1)], 1)
    coef = torch.linalg.lstsq(X, y_tr.unsqueeze(1)).solution
    pred = lambda xs: torch.stack([xs ** k for k in range(degree + 1)], 1) @ coef  # noqa: E731
    return pred, ((pred(x_tr).squeeze() - y_tr) ** 2).mean().item(), ((pred(x_va).squeeze() - y_va) ** 2).mean().item()


print("degree  train_mse  valid_mse")
results = {}
for d in (1, 3, 5, 9, 15):
    pred, tr, va = fit_poly(d)
    results[d] = pred
    print(f"{d:6d}  {tr:9.3f}  {va:9.3f}" + ("   <- overfitting: memorised the training points" if d == 15 else ""))

# ---- 2. Time series: random split vs. time split --------------------------------------------------
days = torch.arange(1000).float()
sales = 50 + 0.05 * days + 10 * torch.sin(days / 30) + 3 * torch.randn(1000)


def nearest_neighbour_error(train_idx, valid_idx):
    """Predict each validation day with the sales of the closest training day (a model that 'memorises')."""
    tr_days = days[train_idx]
    nearest = (days[valid_idx].unsqueeze(1) - tr_days).abs().argmin(1)
    return (sales[train_idx][nearest] - sales[valid_idx]).abs().mean().item()


perm = torch.randperm(1000)
random_err = nearest_neighbour_error(perm[:800], perm[800:])
time_err = nearest_neighbour_error(torch.arange(800), torch.arange(800, 1000))
print(f"\nTime series, same model:\n  random split error: {random_err:.2f}   (looks great, but it's 'interpolating')"
      f"\n  time split error:   {time_err:.2f}   (honest: this is what predicting the future is like)")

plt = try_import_matplotlib()
if plt is not None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    xs = torch.linspace(-1, 1, 200)
    axs[0].scatter(x_tr, y_tr, label="train")
    axs[0].scatter(x_va, y_va, label="valid")
    for d in (1, 5, 15):
        axs[0].plot(xs, results[d](xs).squeeze().clamp(-3, 3), label=f"degree {d}")
    axs[0].legend()
    axs[0].set_title("Under- and over-fitting")
    axs[1].plot(days[:800], sales[:800], label="training data")
    axs[1].plot(days[800:], sales[800:], label="validation data (the future)")
    axs[1].legend()
    axs[1].set_title("Time series: validate on the most recent period")
    fig.savefig(OUT / "validation_sets.png", bbox_inches="tight")
    print(f"saved {OUT / 'validation_sets.png'}")
