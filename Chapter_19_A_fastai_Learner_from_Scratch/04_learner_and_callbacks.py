"""
Chapter 19 — A fastai Learner from Scratch
Topic 4: Learner and callbacks (book sections "Learner", "Callbacks")

A minimal Learner: an optimizer (SGD with weight decay), a training loop over epochs and batches,
and callbacks called at each step: before_fit, before_epoch, before_batch, after_batch,
after_epoch, after_fit. Two callbacks:
  * SetupLearnerCB: move the model and each batch to the GPU, and normalise the batch
  * TrackResults: compute and print the loss and accuracy of each epoch

Run:
    python 04_learner_and_callbacks.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

data = importlib.import_module("01_data_and_dataset")
models = importlib.import_module("02_module_and_parameter")
losses = importlib.import_module("03_loss")


class CancelFitException(Exception):
    pass


class SGD:
    def __init__(self, params, lr, wd=0.0):
        self.params, self.lr, self.wd = params, lr, wd

    def step(self):
        for p in self.params:
            p.data -= (p.grad.data + p.data * self.wd) * self.lr
            p.grad.data.zero_()


class DataLoaders:
    def __init__(self, *dls):
        self.train, self.valid = dls


class Learner:
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=SGD):
        self.model, self.dls, self.loss_func, self.lr, self.cbs, self.opt_func = model, dls, loss_func, lr, cbs, opt_func
        for cb in cbs:
            cb.learner = self

    def one_batch(self):
        self("before_batch")
        xb, yb = self.batch
        self.preds = self.model(xb)
        self.loss = self.loss_func(self.preds, yb)
        if self.model.training:
            self.loss.backward()
            self.opt.step()
        self("after_batch")

    def one_epoch(self, train):
        self.model.training = train
        self("before_epoch")
        dl = self.dls.train if train else self.dls.valid
        for self.num, self.batch in enumerate(dl):
            if train:
                self.one_batch()
            else:
                with torch.no_grad():
                    self.one_batch()
        self("after_epoch")

    def fit(self, n_epochs):
        self("before_fit")
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self.n_epochs = n_epochs
        try:
            for self.epoch in range(n_epochs):
                self.one_epoch(True)
                self.one_epoch(False)
        except CancelFitException:
            pass
        self("after_fit")

    def __call__(self, name):
        for cb in self.cbs:
            getattr(cb, name, lambda: None)()


class Callback:
    """Attributes not found on the callback are looked up on the learner (fastai's GetAttr)."""

    def __getattr__(self, k):
        if k == "learner":
            raise AttributeError(k)
        return getattr(self.learner, k)


class SetupLearnerCB(Callback):
    def before_batch(self):
        xb, yb = self.batch
        self.learner.batch = data.tfm_x(xb.to(data.device)), yb.to(data.device)

    def before_fit(self):
        self.model.to(data.device)


class TrackResults(Callback):
    def before_epoch(self):
        self.accs, self.losses, self.ns = [], [], []

    def after_epoch(self):
        n = sum(self.ns)
        print(f"   epoch {self.epoch} {'train' if self.model.training else 'valid'}: "
              f"loss {sum(self.losses).item() / n:.4f}, accuracy {sum(self.accs).item() / n:.4f}")

    def after_batch(self):
        xb, yb = self.batch
        acc = (self.preds.argmax(dim=1) == yb).float().sum()
        self.accs.append(acc)
        n = len(xb)
        self.losses.append(self.loss.detach() * n)
        self.ns.append(n)


dls = DataLoaders(data.train_dl, data.valid_dl)

if __name__ == "__main__":
    torch.manual_seed(42)
    cbs = [SetupLearnerCB(), TrackResults()]
    learn = Learner(models.simple_cnn(), dls, losses.cross_entropy, lr=0.1, cbs=cbs)
    learn.fit(1)
