"""
Chapter 16 — The Training Process
Topic 6: Decoupled weight decay (book section of the same name)

Two ways to keep weights small:
  * L2 regularization: add wd * w to the gradient       (def l2_reg(p, lr, wd): p.grad.data.add_(wd, p.data))
  * weight decay: shrink the weight directly             (def weight_decay(p, lr, wd): p.data.mul_(1 - lr*wd))
For plain SGD they're the same thing. With momentum, and especially with Adam, they differ: with L2
the decay term goes into the moving averages and gets divided by sqrt(sqr_avg), so it's weak for
parameters with large gradients. Loshchilov & Hutter showed decoupled weight decay (AdamW) works
better, and fastai uses it by default.

Run:
    python 06_decoupled_weight_decay.py
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import set_seed  # noqa: E402
from training_setup import (Optimizer, adam_step, average_grad_dampened, average_sqr_grad, get_learner, l2_reg,  # noqa: E402
                            sgd_cb, step_stat, weight_decay)

# With plain SGD both give exactly the same update
for cb in (weight_decay, l2_reg):
    w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    w.grad = torch.tensor([0.1, 0.1])
    Optimizer([w], cbs=[cb, sgd_cb], lr=0.1, wd=0.5).step()
    print(f"SGD + {cb.__name__:12s}: {[round(v, 4) for v in w.data.tolist()]}")

adam_cbs = [average_grad_dampened, average_sqr_grad, step_stat, adam_step]
for name, cbs in [("Adam + L2 regularization", [l2_reg] + adam_cbs), ("Adam + decoupled weight decay (AdamW)", [weight_decay] + adam_cbs)]:
    print(f"\n{name}, wd=0.1:")
    set_seed(42)
    learn = get_learner(opt_func=partial(Optimizer, cbs=cbs, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.1))
    learn.fit_one_cycle(3, 0.003)
