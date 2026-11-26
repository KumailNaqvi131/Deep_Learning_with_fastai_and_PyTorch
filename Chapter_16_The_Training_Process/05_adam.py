"""
Chapter 16 — The Training Process
Topic 5: Adam (book section "Adam")

Adam = momentum (a dampened moving average of gradients) + RMSProp (divide by the root of the
moving average of squared gradients), with "debiasing": both averages start at zero, so early
on they are divided by (1 - beta**step) to correct for that.
    w.avg = beta1 * w.avg + (1-beta1) * w.grad
    unbias_avg = w.avg / (1 - (beta1**(i+1)))
    w.sqr_avg = beta2 * w.sqr_avg + (1-beta2) * (w.grad ** 2)
    new_w = w - lr * unbias_avg / sqrt(w.sqr_avg + eps)
fastai's defaults: beta1 (mom) = 0.9, beta2 (sqr_mom) = 0.99, eps = 1e-5.

Run:
    python 05_adam.py
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import set_seed  # noqa: E402
from training_setup import (Optimizer, adam_step, average_grad_dampened, average_sqr_grad, get_learner,  # noqa: E402
                            step_stat, weight_decay)

# Our Adam gives the same steps as PyTorch's AdamW
torch.manual_seed(0)
w1 = torch.nn.Parameter(torch.randn(5))
w2 = torch.nn.Parameter(w1.detach().clone())
ours = Optimizer([w1], cbs=[weight_decay, average_grad_dampened, average_sqr_grad, step_stat, adam_step],
                 lr=0.01, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01)
torchs = torch.optim.AdamW([w2], lr=0.01, betas=(0.9, 0.99), eps=1e-5, weight_decay=0.01)
for _ in range(10):
    g = torch.randn(5)
    w1.grad, w2.grad = g.clone(), g.clone()
    ours.step()
    torchs.step()
print("our Adam == torch.optim.AdamW after 10 steps:", torch.allclose(w1, w2, atol=1e-6))

set_seed(42)
opt_func = partial(Optimizer, cbs=[weight_decay, average_grad_dampened, average_sqr_grad, step_stat, adam_step],
                   mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01)
learn = get_learner(opt_func=opt_func)
learn.fit_one_cycle(3, 0.003)
