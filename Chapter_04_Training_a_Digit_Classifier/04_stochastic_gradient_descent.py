"""
Chapter 4 — Under the Hood: Training a Digit Classifier
Topic 4: Stochastic gradient descent (book sections "Stochastic Gradient Descent", "Calculating
Gradients", "Stepping with a Learning Rate", "An End-to-End SGD Example")

The seven steps of learning:
    1. Initialize the weights.            5. Step: change the weights based on the gradients.
    2. Predict.                           6. Repeat from step 2.
    3. Calculate the loss.                7. Stop (after enough epochs, or when it's good enough).
    4. Calculate the gradients.

First a toy: find the minimum of f(x) = x**2 by following the gradient. Then the book's
end-to-end example: someone measured a roller coaster's speed every second as it went over a
hump; find the quadratic a*t**2 + b*t + c that best fits these noisy measurements.

Run:
    python 04_stochastic_gradient_descent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import outputs_dir, try_import_matplotlib  # noqa: E402

torch.manual_seed(42)
OUT = outputs_dir(__file__)


# ---- Calculating gradients ----------------------------------------------------------------------
def f(x):
    return x ** 2


xt = torch.tensor(3.0).requires_grad_()       # "track the gradient of this value"
yt = f(xt)
yt.backward()                                 # backpropagation: compute d(yt)/d(xt)
print("f(3) =", yt.item(), "  gradient at 3 =", xt.grad.item(), "  (the derivative of x**2 is 2x)")

xt = torch.tensor([3.0, 4.0, 10.0]).requires_grad_()
(xt ** 2).sum().backward()
print("gradients of sum(x**2) at [3,4,10]:", xt.grad.tolist())

# ---- Stepping with a learning rate: walk down f(x) = x**2 ----------------------------------------
x = torch.tensor(-1.5, requires_grad=True)
for lr in (0.1,):
    for step in range(10):
        loss = f(x)
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad          # w -= gradient(w) * lr
        x.grad = None
    print(f"after 10 steps with lr={lr}: x = {x.item():.4f} (the minimum is at 0)")

# ---- An end-to-end SGD example: the roller coaster -----------------------------------------------
time = torch.arange(0, 20).float()
speed = torch.randn(20) * 3 + 0.75 * (time - 9.5) ** 2 + 1


def model(t, params):
    a, b, c = params
    return a * (t ** 2) + (b * t) + c


def mse(preds, targets):
    return ((preds - targets) ** 2).mean()


params = torch.randn(3).requires_grad_()                 # step 1: initialize
orig_params = params.clone()
lr = 1e-5
snapshots = []


def apply_step(params, prn=True):
    preds = model(time, params)                          # step 2: predict
    loss = mse(preds, speed)                             # step 3: loss
    loss.backward()                                      # step 4: gradients
    params.data -= lr * params.grad.data                 # step 5: step
    params.grad = None
    if prn:
        print(f"  loss {loss.item():10.2f}")
    return preds


print("\nRoller coaster, 10 steps (step 6: repeat):")
for i in range(10):
    snapshots.append(apply_step(params).detach())
print("parameters a, b, c went from", [round(v, 3) for v in orig_params.tolist()],
      "to", [round(v, 3) for v in params.tolist()])

plt = try_import_matplotlib()
if plt is not None:
    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    for ax, i in zip(axs, (0, 3, 6, 9)):
        ax.scatter(time, speed)
        ax.scatter(time, snapshots[i], color="red")
        ax.set_ylim(-300, 100)
        ax.set_title(f"after {i + 1} steps")
    fig.savefig(OUT / "roller_coaster_sgd.png", bbox_inches="tight")
    print(f"saved {OUT / 'roller_coaster_sgd.png'}  (blue: data, red: predictions)")
