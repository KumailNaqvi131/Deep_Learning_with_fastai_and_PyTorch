"""
Chapter 16 — The Training Process
Topic 3: Momentum (book section "Momentum")

SGD with momentum steps along a moving average of the gradients rather than the current gradient:
    weight.avg = beta * weight.avg + (1-beta) * weight.grad
    new_weight = weight - lr * weight.avg
Like a ball rolling down a slope: it keeps going in a consistent direction and is less shaken by
noisy gradients. beta=0.9 is the usual value (fastai varies it 0.95 -> 0.85 -> 0.95 with 1cycle).

    def average_grad(p, mom, grad_avg=None, **kwargs):
        if grad_avg is None: grad_avg = torch.zeros_like(p.grad.data)
        return {'grad_avg': grad_avg*mom + p.grad.data}
    def momentum_step(p, lr, grad_avg, **kwargs): p.data.add_(-lr, grad_avg)
    opt_func = partial(Optimizer, cbs=[average_grad,momentum_step], mom=0.9)

Run:
    python 03_momentum.py
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import outputs_dir, set_seed, try_import_matplotlib  # noqa: E402
from training_setup import Optimizer, average_grad, get_learner, momentum_step  # noqa: E402

# What a moving average does to noisy data
torch.manual_seed(0)
x = torch.linspace(-4, 4, 100)
y = 1 - (x / 3) ** 2 + torch.randn(100) * 0.1
plt = try_import_matplotlib()
if plt is not None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for beta, ax in zip((0.5, 0.7, 0.9, 0.99), axs.flatten()):
        avg, res = 0, []
        for yi in y:
            avg = beta * avg + (1 - beta) * yi
            res.append(avg)
        ax.scatter(x, y, s=8)
        ax.plot(x, res, color="red")
        ax.set_title(f"beta={beta}")
    fig.savefig(outputs_dir(__file__) / "moving_averages.png", bbox_inches="tight")
    print("saved outputs/moving_averages.png: a higher beta is smoother but lags behind")

set_seed(42)
opt_func = partial(Optimizer, cbs=[average_grad, momentum_step], mom=0.9)
learn = get_learner(opt_func=opt_func)
learn.fit_one_cycle(3, 0.03)
