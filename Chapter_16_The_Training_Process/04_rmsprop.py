"""
Chapter 16 — The Training Process
Topic 4: RMSProp (book section "RMSProp")

RMSProp (Hinton) gives each parameter its own learning rate: divide the step by the root of a
moving average of the squared gradients. Parameters whose gradients are consistently small get
bigger steps; noisy parameters with big gradients get smaller ones.
    w.square_avg = alpha * w.square_avg + (1-alpha) * (w.grad ** 2)
    new_w = w - lr * w.grad / math.sqrt(w.square_avg + eps)

    opt_func = partial(Optimizer, cbs=[average_sqr_grad,rms_prop_step], sqr_mom=0.99, eps=1e-7)

Run:
    python 04_rmsprop.py
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import set_seed  # noqa: E402
from training_setup import Optimizer, average_sqr_grad, get_learner, rms_prop_step  # noqa: E402

set_seed(42)
opt_func = partial(Optimizer, cbs=[average_sqr_grad, rms_prop_step], sqr_mom=0.99, eps=1e-7)
learn = get_learner(opt_func=opt_func)
learn.fit_one_cycle(3, 0.003)
