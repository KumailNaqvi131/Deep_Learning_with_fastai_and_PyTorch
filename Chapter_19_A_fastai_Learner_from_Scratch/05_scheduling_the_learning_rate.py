"""
Chapter 19 — A fastai Learner from Scratch
Topic 5: Scheduling the learning rate (book section of the same name)

Two more callbacks, and our from-scratch Learner has fastai's two key training tools:
  * LRFinder: multiply the lr by 1.2 every batch, record the losses, stop when lr > 10 or NaN
  * OneCycle: linear warm-up from base_lr/10 to base_lr for the first 25% of training, then a
    linear decrease to 0 (fastai uses cosine curves, but the idea is the same)

Run:
    python 05_scheduling_the_learning_rate.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import outputs_dir, try_import_matplotlib  # noqa: E402

models = importlib.import_module("02_module_and_parameter")
losses = importlib.import_module("03_loss")
lrn = importlib.import_module("04_learner_and_callbacks")
OUT = outputs_dir(__file__)


class LRFinder(lrn.Callback):
    def before_fit(self):
        self.losses, self.lrs = [], []
        self.learner.lr = 1e-6

    def before_batch(self):
        if not self.model.training:
            return
        self.opt.lr *= 1.2

    def after_batch(self):
        if not self.model.training:
            return
        if self.opt.lr > 10 or torch.isnan(self.loss):
            raise lrn.CancelFitException
        self.losses.append(self.loss.item())
        self.lrs.append(self.opt.lr)


class OneCycle(lrn.Callback):
    def __init__(self, base_lr):
        self.base_lr = base_lr

    def before_fit(self):
        self.lrs = []

    def before_batch(self):
        if not self.model.training:
            return
        n = len(self.dls.train)
        bn = self.epoch * n + self.num
        mn = self.n_epochs * n
        pct = bn / mn
        pct_start, div_start = 0.25, 10
        if pct < pct_start:
            pct /= pct_start
            lr = (1 - pct) * self.base_lr / div_start + pct * self.base_lr
        else:
            pct = (pct - pct_start) / (1 - pct_start)
            lr = (1 - pct) * self.base_lr
        self.opt.lr = lr
        self.lrs.append(lr)


if __name__ == "__main__":
    torch.manual_seed(42)
    cbs = [lrn.SetupLearnerCB(), lrn.TrackResults()]
    print("Learning rate finder:")
    lrfind = LRFinder()
    learn = lrn.Learner(models.simple_cnn(), lrn.dls, losses.cross_entropy, lr=0.1, cbs=cbs + [lrfind])
    learn.fit(2)
    best = min(range(len(lrfind.losses)), key=lambda i: lrfind.losses[i])
    print(f"   {len(lrfind.lrs)} batches; lowest loss {lrfind.losses[best]:.3f} at lr {lrfind.lrs[best]:.3g}")
    plt = try_import_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(lrfind.lrs[:-2], lrfind.losses[:-2])
        ax.set_xscale("log")
        fig.savefig(OUT / "lr_finder_from_scratch.png", bbox_inches="tight")
        print(f"   saved {OUT / 'lr_finder_from_scratch.png'}")

    print("\n1cycle training, 8 epochs, base lr 0.1:")
    torch.manual_seed(42)
    onecyc = OneCycle(0.1)
    learn = lrn.Learner(models.simple_cnn(), lrn.dls, losses.cross_entropy, lr=0.1, cbs=cbs + [onecyc])
    learn.fit(8)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(onecyc.lrs)
        ax.set_title("our 1cycle learning rate schedule")
        fig.savefig(OUT / "one_cycle_from_scratch.png", bbox_inches="tight")
        print(f"   saved {OUT / 'one_cycle_from_scratch.png'}")
