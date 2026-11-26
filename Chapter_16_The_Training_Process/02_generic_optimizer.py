"""
Chapter 16 — The Training Process
Topic 2: A generic optimizer (book section of the same name)

fastai's Optimizer is a loop over parameters that calls a list of small "step callbacks". Plain
SGD is one line:

    def sgd_cb(p, lr, **kwargs): p.data.add_(-lr, p.grad.data)
    opt_func = partial(Optimizer, cbs=[sgd_cb])

The class and all the callbacks of this chapter are in training_setup.py.

Run:
    python 02_generic_optimizer.py
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import set_seed  # noqa: E402
from training_setup import Optimizer, get_learner, sgd_cb  # noqa: E402

# One step by hand
w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
w.grad = torch.tensor([0.5, -1.0])
opt = Optimizer([w], cbs=[sgd_cb], lr=0.1)
opt.step()
print("after one SGD step with lr=0.1:", w.data.tolist(), "(= w - lr * grad)")

set_seed(42)
opt_func = partial(Optimizer, cbs=[sgd_cb])
learn = get_learner(opt_func=opt_func)
learn.fit(3, 0.03)
print("Same results as PyTorch's SGD: 'an optimizer' is just how the weights are updated from the gradients.")
