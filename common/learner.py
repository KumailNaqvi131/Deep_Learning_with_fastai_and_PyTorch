"""A small, readable version of fastai's training loop: Learner, callbacks, fit_one_cycle,
fine_tune, lr_find, freezing, discriminative learning rates, mixed precision and metrics.

It follows fastai's names and defaults, so the book's code maps one-to-one:

    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    learn.lr_find()
    learn.fit_one_cycle(5, 3e-3)
    learn.fine_tune(2)                         # freeze -> 1 epoch -> unfreeze -> 2 epochs
    preds, targs = learn.get_preds()

Chapter 16 explains optimizers and callbacks and Chapter 19 rebuilds this from scratch.
"""

import copy
import math
import time
from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utils import get_device, try_import_matplotlib


# ==================================================================================================
# Data: move batches to the GPU and run batch transforms there
# ==================================================================================================
def to_device(b, device):
    if isinstance(b, (list, tuple)):
        return type(b)(to_device(o, device) for o in b)
    if isinstance(b, dict):
        return {k: to_device(v, device) for k, v in b.items()}
    return b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b


class DeviceDataLoader:
    """Wraps a PyTorch DataLoader: moves each batch to `device`, then applies the batch transforms
    (data augmentation on the GPU, normalisation, ...). `train=True` turns on training-only transforms."""

    def __init__(self, dl, device=None, after_batch=(), kinds=None, train=False, n_inp=1):
        self.dl, self.device, self.after_batch = dl, device or get_device(), list(after_batch)
        self.kinds, self.train, self.n_inp = kinds, train, n_inp

    def __iter__(self):
        for b in self.dl:
            b = to_device(tuple(b), self.device)
            for tfm in self.after_batch:
                b = tfm(b, kinds=self.kinds, train=self.train)
            yield b

    def __len__(self):
        return len(self.dl)

    @property
    def dataset(self):
        return self.dl.dataset

    @property
    def bs(self):
        return self.dl.batch_size

    def one_batch(self):
        return next(iter(self))

    def new(self, dl=None, train=None):
        """Same transforms, different DataLoader (or train/valid behaviour)."""
        return DeviceDataLoader(dl or self.dl, self.device, self.after_batch, self.kinds,
                                self.train if train is None else train, self.n_inp)


class DataLoaders:
    """A train and a valid DataLoader plus what the Learner needs to know about the data
    (`vocab`, number of outputs `c`, default `loss_func`). Same idea as fastai's DataLoaders."""

    def __init__(self, train, valid, vocab=None, c=None, loss_func=None, device=None, n_inp=1):
        device = device or get_device()
        wrap = lambda dl, tr: dl if isinstance(dl, DeviceDataLoader) else DeviceDataLoader(dl, device, train=tr, n_inp=n_inp)  # noqa: E731
        self.loaders = [wrap(train, True), wrap(valid, False)]
        self.vocab, self.loss_func, self.device, self.n_inp = vocab, loss_func, device, n_inp
        self.c = c if c is not None else (len(vocab) if vocab is not None else None)

    train = property(lambda self: self.loaders[0])
    valid = property(lambda self: self.loaders[1])

    def __getitem__(self, i):
        return self.loaders[i]

    def one_batch(self):
        return self.train.one_batch()

    def decode_x(self, x):
        """Undo normalisation (for showing images)."""
        for tfm in reversed(self.train.after_batch):
            if hasattr(tfm, "decode"):
                x = tfm.decode(x)
        return x


# ==================================================================================================
# Losses (with the activation and decoding that go with them) and metrics
# ==================================================================================================
class BaseLossFlat(nn.Module):
    """Flattens predictions and targets before calling the loss, like fastai's `*LossFlat`.
    `axis` is the class dimension (1 for segmentation masks [bs, classes, h, w])."""

    def __init__(self, loss_cls, axis=-1, flatten=True, floatify=False, is_2d=True, **kwargs):
        super().__init__()
        self.func, self.axis, self.flatten, self.floatify, self.is_2d = loss_cls(**kwargs), axis, flatten, floatify, is_2d

    def forward(self, inp, targ):
        inp = inp.movedim(self.axis, -1).contiguous()          # e.g. [bs, classes, h, w] -> [bs, h, w, classes]
        targ = targ.movedim(self.axis, -1).contiguous() if targ.ndim == inp.ndim else targ
        if self.floatify:
            targ = targ.float()
        if self.flatten:
            inp = inp.view(-1, inp.shape[-1]) if self.is_2d else inp.view(-1)
            targ = targ.view(-1)
        return self.func(inp, targ)

    def __repr__(self):
        return f"FlattenedLoss of {self.func}"


class CrossEntropyLossFlat(BaseLossFlat):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(nn.CrossEntropyLoss, axis=axis, **kwargs)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)

    def decodes(self, x):
        return x.argmax(dim=self.axis)


class BCEWithLogitsLossFlat(BaseLossFlat):
    def __init__(self, axis=-1, thresh=0.5, **kwargs):
        super().__init__(nn.BCEWithLogitsLoss, axis=axis, floatify=True, is_2d=False, **kwargs)
        self.thresh = thresh

    def activation(self, x):
        return torch.sigmoid(x)

    def decodes(self, x):
        return x > self.thresh


class MSELossFlat(BaseLossFlat):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(nn.MSELoss, axis=axis, floatify=True, is_2d=False, **kwargs)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy against 'soft' targets: (1-eps) on the right class, eps/N spread everywhere (Ch. 7)."""

    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        c = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)
        smooth = -log_preds.sum(dim=-1).mean()                 # the "spread" part
        nll = F.nll_loss(log_preds, target)                     # the usual part
        return smooth * self.eps / c + (1 - self.eps) * nll

    activation = staticmethod(lambda x: F.softmax(x, dim=-1))
    decodes = staticmethod(lambda x: x.argmax(dim=-1))


def accuracy(inp, targ, axis=-1):
    """Fraction of items where the highest activation is the right class."""
    return (inp.argmax(dim=axis) == targ).float().mean()


def error_rate(inp, targ, axis=-1):
    return 1 - accuracy(inp, targ, axis)


def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    """Multi-label accuracy: each (item, label) pair counts once (Ch. 6)."""
    if sigmoid:
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()


def rmse(inp, targ):
    return torch.sqrt(F.mse_loss(inp.view(-1).float(), targ.view(-1).float()))


def mse(inp, targ):
    return F.mse_loss(inp.view(-1).float(), targ.view(-1).float())


def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    """Pixel accuracy ignoring background pixels (segmentation)."""
    targ = targ.squeeze(1) if targ.ndim == inp.ndim else targ
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


def metric_name(m):
    if isinstance(m, partial):
        return m.func.__name__
    return getattr(m, "__name__", type(m).__name__)


# ==================================================================================================
# Callbacks
# ==================================================================================================
class CancelBatchException(Exception): pass      # noqa: E701  skip the rest of this batch
class CancelTrainException(Exception): pass      # noqa: E701  skip the rest of the training phase
class CancelValidException(Exception): pass      # noqa: E701  skip validation
class CancelEpochException(Exception): pass      # noqa: E701  skip the rest of this epoch
class CancelFitException(Exception): pass        # noqa: E701  stop training


EVENTS = ["before_fit", "before_epoch", "before_train", "before_batch", "after_pred", "after_loss",
          "before_backward", "after_backward", "after_step", "after_cancel_batch", "after_batch",
          "after_train", "before_validate", "after_validate", "after_epoch", "after_fit"]


class Callback:
    """Override any of the EVENTS methods. `self.learn` is the Learner (whose attributes,
    such as `self.learn.loss` or `self.learn.pred`, you may read and change)."""
    order = 0
    learn = None

    @property
    def name(self):
        n = type(self).__name__
        return "".join("_" + c.lower() if c.isupper() else c for c in n).lstrip("_").replace("_callback", "")


class Recorder(Callback):
    """Records losses, learning rates and metrics, and prints the progress table."""
    order = 50

    def __init__(self, train_metrics=False, beta=0.98):
        self.beta = beta

    def before_fit(self):
        self.losses, self.lrs, self.values, self._avg = [], [], [], 0.0
        self._n = 0
        self.metric_names = ["epoch", "train_loss", "valid_loss"] + [metric_name(m) for m in self.learn.metrics] + ["time"]
        if not getattr(self.learn, "silent", False):
            print("  ".join(f"{n:>11}" if i else f"{n:>5}" for i, n in enumerate(self.metric_names)))

    def before_epoch(self):
        self.start, self.train_tot, self.train_cnt = time.time(), 0.0, 0

    def after_batch(self):
        if not self.learn.training:
            return
        loss = self.learn.loss.item()
        self._n += 1
        self._avg = self._avg * self.beta + loss * (1 - self.beta)          # exponentially smoothed loss
        self.losses.append(self._avg / (1 - self.beta ** self._n))
        self.lrs.append(self.learn.opt.param_groups[-1]["lr"])
        bs = self.learn.batch_size
        self.train_tot += loss * bs
        self.train_cnt += bs

    def after_epoch(self):
        train_loss = self.train_tot / max(self.train_cnt, 1)
        vals = [self.learn.epoch, train_loss, *self.learn.valid_results, time.time() - self.start]
        self.values.append(vals)
        if not getattr(self.learn, "silent", False):
            out = [f"{vals[0]:>5}"] + [f"{v:>11.6f}" if v == v else f"{'nan':>11}" for v in vals[1:-1]]
            m, s = divmod(int(vals[-1]), 60)
            print("  ".join(out) + f"  {m:>8}:{s:02d}", flush=True)

    def plot_loss(self, fname="loss.png", skip_start=5):
        plt = try_import_matplotlib()
        if plt is None:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(skip_start, len(self.losses)), self.losses[skip_start:], label="train (smoothed)")
        n_iter = len(self.losses) / max(len(self.values), 1)
        ax.plot([(i + 1) * n_iter for i in range(len(self.values))], [v[2] for v in self.values], "o-", label="valid")
        ax.set_xlabel("batches")
        ax.set_ylabel("loss")
        ax.legend()
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fname}")


def combined_cos(pct, start, middle, end):
    """Cosine from start to middle during the first `pct` of training, then middle to end (1cycle)."""
    def sched(pos):
        if pos < pct:
            p = pos / pct
            return start + (middle - start) * (1 - math.cos(math.pi * p)) / 2
        p = (pos - pct) / (1 - pct)
        return middle + (end - middle) * (1 - math.cos(math.pi * p)) / 2
    return sched


class ParamScheduler(Callback):
    """Changes hyper-parameters (per parameter group) at every training batch.
    `scheds` maps a name ('lr', 'mom') to a list of functions of the training progress (0..1)."""
    order = 60

    def __init__(self, scheds):
        self.scheds = scheds

    def before_fit(self):
        self.hps = {k: [] for k in self.scheds}

    def before_batch(self):
        if not self.learn.training:
            return
        pos = self.learn.pct_train
        for name, fs in self.scheds.items():
            self.learn.set_hyper(name, [f(pos) for f in fs])
            self.hps[name].append([f(pos) for f in fs])


class LRFinder(Callback):
    """Raise the learning rate exponentially every batch, record the loss, stop when it explodes (Ch. 5)."""
    order = 60

    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        self.start_lr, self.end_lr, self.num_it, self.stop_div = start_lr, end_lr, num_it, stop_div

    def before_fit(self):
        self.best, self.it = float("inf"), 0

    def before_batch(self):
        if self.learn.training:
            lr = self.start_lr * (self.end_lr / self.start_lr) ** (self.it / self.num_it)
            self.learn.set_hyper("lr", [lr] * len(self.learn.opt.param_groups))

    def after_batch(self):
        if not self.learn.training:
            return
        self.it += 1
        smooth = self.learn.recorder.losses[-1]
        self.best = min(self.best, smooth)
        if (self.stop_div and (smooth > 4 * self.best or math.isnan(smooth))) or self.it >= self.num_it:
            raise CancelFitException()

    def before_validate(self):
        raise CancelValidException()


class MixedPrecision(Callback):
    """Marker: `learn.to_fp16()` adds it, and the training loop then uses torch.autocast + GradScaler."""


SuggestedLRs = namedtuple("SuggestedLRs", ["lr_min", "lr_steep"])


# ==================================================================================================
# Optimizers (fastai defaults: Adam with decoupled weight decay)
# ==================================================================================================
def Adam(params, lr=1e-3, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01):
    return torch.optim.AdamW(params, lr=lr, betas=(mom, sqr_mom), eps=eps, weight_decay=wd)


def SGD(params, lr=1e-3, mom=0.0, wd=0.0):
    return torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)


def trainable_params(m):
    return [p for p in m.parameters() if p.requires_grad]


# ==================================================================================================
# The Learner
# ==================================================================================================
class Learner:
    def __init__(self, dls, model, loss_func=None, opt_func=Adam, lr=1e-3, splitter=None, cbs=None,
                 metrics=None, wd=None, moms=(0.95, 0.85, 0.95), path=".", train_bn=True):
        self.dls, self.model = dls, model.to(dls.device)
        self.loss_func = loss_func if loss_func is not None else dls.loss_func
        assert self.loss_func is not None, "Pass loss_func= (the DataLoaders don't define a default one)"
        self.opt_func, self.lr, self.wd, self.moms, self.path = opt_func, lr, wd, moms, Path(path)
        self.splitter = splitter or (lambda m: [list(m.parameters())])
        self.metrics = list(metrics) if isinstance(metrics, (list, tuple)) else ([metrics] if metrics else [])
        self.train_bn, self.opt, self.use_amp, self.silent = train_bn, None, False, False
        self.cbs = []
        self.add_cbs([Recorder()] + list(cbs or []))

    # ---- callbacks ----
    def add_cbs(self, cbs):
        for cb in cbs:
            cb.learn = self
            setattr(self, cb.name, cb)
            self.cbs.append(cb)
        self.cbs.sort(key=lambda c: c.order)

    def remove_cbs(self, cbs):
        for cb in cbs:
            self.cbs.remove(cb)

    def __call__(self, event):
        for cb in self.cbs:
            f = getattr(cb, event, None)
            if f is not None:
                f()

    # ---- optimizer, hyper-parameters, freezing ----
    def create_opt(self):
        groups = [{"params": g} for g in self.splitter(self.model)]
        kw = {} if self.wd is None else {"wd": self.wd}
        self.opt = self.opt_func(groups, lr=self.lr, **kw)

    def set_hyper(self, name, values):
        for g, v in zip(self.opt.param_groups, values):
            if name == "mom":                               # momentum has different names in different optimizers
                if "betas" in g:
                    g["betas"] = (v, g["betas"][1])
                elif "momentum" in g:
                    g["momentum"] = v
                elif "mom" in g:
                    g["mom"] = v
            elif name == "weight_decay" and "weight_decay" not in g and "wd" in g:
                g["wd"] = v
            else:
                g[name] = v

    def _lrs(self, lr):
        """A number -> same lr for every group. slice(a, b) -> spread geometrically from a (first group)
        to b (last group). slice(b) -> b/10 for all groups but the last (fastai's convention)."""
        n = len(self.splitter(self.model))
        if isinstance(lr, slice):
            if lr.start is None:
                return [lr.stop / 10] * (n - 1) + [lr.stop]
            return list(np.geomspace(lr.start, lr.stop, n)) if n > 1 else [lr.stop]
        return list(lr) if isinstance(lr, (list, tuple, np.ndarray)) else [lr] * n

    def freeze_to(self, n):
        """Train only parameter groups n.. ; earlier groups are frozen (except BatchNorm layers)."""
        groups = self.splitter(self.model)
        frozen = {id(p) for g in groups[:n] for p in g}
        bn_params = {id(p) for m in self.model.modules()
                     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for p in m.parameters()}
        for g in groups:
            for p in g:
                p.requires_grad_(id(p) not in frozen or (self.train_bn and id(p) in bn_params))

    def freeze(self):
        self.freeze_to(-1)

    def unfreeze(self):
        self.freeze_to(0)

    def to_fp16(self):
        self.use_amp = get_device().type == "cuda"
        return self

    # ---- the training loop ----
    def _with_events(self, f, name, exc):
        try:
            self(f"before_{name}")
            f()
        except exc:
            self(f"after_cancel_{name}")
        self(f"after_{name}")

    def _do_one_batch(self):
        xb, yb = self.batch[:self.dls.n_inp], self.batch[self.dls.n_inp:]
        self.xb, self.yb = xb, yb
        with torch.autocast(self.dls.device.type, enabled=self.use_amp):
            self.pred = self.model(*xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, *yb)
        self("after_loss")
        if not self.training or not len(yb):
            return
        self("before_backward")
        self.scaler.scale(self.loss).backward()
        self("after_backward")
        self.scaler.step(self.opt)
        self.scaler.update()
        self("after_step")
        self.opt.zero_grad(set_to_none=True)

    def one_batch(self, i, b):
        self.iter, self.batch = i, b
        self.batch_size = len(b[0]) if not isinstance(b[0], (list, tuple)) else len(b[0][0])
        try:
            self("before_batch")
            self._do_one_batch()
        except CancelBatchException:
            self("after_cancel_batch")
        self("after_batch")

    def _all_batches(self):
        self.n_iter = len(self.dl)
        for i, b in enumerate(self.dl):
            if self.training:
                self.pct_train = (self.epoch + i / self.n_iter) / self.n_epoch
            self.one_batch(i, b)

    def _do_epoch_train(self):
        self.model.train()
        self.training, self.dl = True, self.dls.train
        self._with_events(self._all_batches, "train", CancelTrainException)

    def _do_epoch_validate(self, dl=None):
        self.model.eval()
        self.training, self.dl = False, dl or self.dls.valid
        self._val_preds, self._val_targs, self._val_losses = [], [], []
        with torch.no_grad():
            self._with_events(self._all_batches_collect, "validate", CancelValidException)
        if self._val_preds:
            preds, targs = torch.cat(self._val_preds), torch.cat(self._val_targs)
            lb = torch.tensor(self._val_losses)                 # (loss, batch size) per batch
            loss = (lb[:, 0] * lb[:, 1]).sum() / lb[:, 1].sum()
            self.valid_results = [loss.item()] + [float(m(preds, targs)) for m in self.metrics]
        else:
            self.valid_results = [float("nan")] * (1 + len(self.metrics))

    def _all_batches_collect(self):
        self.n_iter = len(self.dl)
        for i, b in enumerate(self.dl):
            self.one_batch(i, b)
            if self.dls.n_inp < len(b):
                p, y = self.pred.detach().float().cpu(), b[self.dls.n_inp].cpu()
                if p.ndim == 3 and y.ndim == 2:              # language models: one prediction per token
                    p, y = p.reshape(-1, p.shape[-1]), y.reshape(-1)
                self._val_preds.append(p)
                self._val_targs.append(y)
                self._val_losses.append((self.loss.item(), self.batch_size))

    def _do_epoch(self):
        self._do_epoch_train()
        self._do_epoch_validate()

    def _do_fit(self):
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            self._with_events(self._do_epoch, "epoch", CancelEpochException)

    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False):
        if reset_opt or self.opt is None:
            if wd is not None:
                self.wd = wd
            self.create_opt()
        if wd is not None:
            self.set_hyper("weight_decay", [wd] * len(self.opt.param_groups))
        self.set_hyper("lr", self._lrs(self.lr if lr is None else lr))
        self.n_epoch, self.epoch, self.pct_train = n_epoch, 0, 0.0
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        cbs = list(cbs or [])
        self.add_cbs(cbs)
        try:
            self._with_events(self._do_fit, "fit", CancelFitException)
        finally:
            self.remove_cbs(cbs)

    def fit_one_cycle(self, n_epoch, lr_max=None, div=25.0, div_final=1e5, pct_start=0.25, wd=None,
                      moms=None, cbs=None):
        """1cycle: lr warms up from lr_max/div to lr_max (cosine), then anneals to lr_max/div_final;
        momentum does the opposite (0.95 -> 0.85 -> 0.95). See Chapter 13."""
        if self.opt is None:
            self.create_opt()
        lr_max = self._lrs(self.lr if lr_max is None else lr_max)
        moms = moms or self.moms
        scheds = {"lr": [combined_cos(pct_start, lr / div, lr, lr / div_final) for lr in lr_max],
                  "mom": [combined_cos(pct_start, *moms)] * len(lr_max)}
        self.fit(n_epoch, lr=lr_max, wd=wd, cbs=[ParamScheduler(scheds)] + list(cbs or []))

    def fine_tune(self, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100, pct_start=0.3, div=5.0, **kwargs):
        """Transfer learning (Ch. 1 and 5): train the new head with the body frozen, then train everything
        with discriminative learning rates (earlier layers get base_lr/lr_mult)."""
        self.freeze()
        self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
        base_lr /= 2
        self.unfreeze()
        self.fit_one_cycle(epochs, slice(base_lr / lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)

    def lr_find(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, fname="lr_find.png", show=True):
        """The learning rate finder (Ch. 5). Returns SuggestedLRs(lr_min, lr_steep) and plots loss vs lr."""
        state = copy.deepcopy(self.model.state_dict())
        opt_state = copy.deepcopy(self.opt.state_dict()) if self.opt is not None else None
        n_epoch = math.ceil(num_it / len(self.dls.train))
        silent, self.silent = self.silent, True
        self.fit(n_epoch, cbs=[LRFinder(start_lr, end_lr, num_it, stop_div)])
        self.silent = silent
        self.model.load_state_dict(state)
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        else:
            self.opt = None
        lrs, losses = np.array(self.recorder.lrs), np.array(self.recorder.losses)
        skip = min(10, len(lrs) // 4)
        lrs_, losses_ = lrs[skip:-3] if len(lrs) > skip + 5 else lrs, losses[skip:-3] if len(lrs) > skip + 5 else losses
        lr_min = lrs_[np.argmin(losses_)] / 10
        grads = np.gradient(losses_, np.log(lrs_))
        lr_steep = lrs_[np.argmin(grads)]
        if show:
            print(f"lr_find: minimum/10 = {lr_min:.2e}, steepest point = {lr_steep:.2e}")
            plt = try_import_matplotlib()
            if plt is not None and fname:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(lrs, losses)
                ax.set_xscale("log")
                ax.set_xlabel("learning rate (log scale)")
                ax.set_ylabel("loss")
                ax.plot(lr_steep, losses_[np.argmin(grads)], "ro", label="steepest")
                ax.legend()
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)
                print(f"saved {fname}")
        return SuggestedLRs(float(lr_min), float(lr_steep))

    # ---- inference ----
    def _activation(self):
        return getattr(self.loss_func, "activation", lambda x: x)

    def get_preds(self, ds_idx=1, dl=None, with_input=False, with_loss=False, act=True, reorder=True):
        """Predictions (after the loss's activation: softmax, sigmoid, ...) and targets for a whole set."""
        dl = dl or self.dls[ds_idx]
        self.model.eval()
        preds, targs, inps, losses = [], [], [], []
        with torch.no_grad():
            for b in dl:
                xb = b[:self.dls.n_inp]
                with torch.autocast(self.dls.device.type, enabled=self.use_amp):
                    p = self.model(*xb).float()
                if len(b) > self.dls.n_inp:
                    y = b[self.dls.n_inp]
                    targs.append(y)
                    if with_loss:
                        lf = copy.copy(self.loss_func)
                        inner = getattr(lf, "func", lf)
                        if hasattr(inner, "reduction"):
                            inner = copy.copy(inner)
                            inner.reduction = "none"
                            if hasattr(lf, "func"):
                                lf.func = inner
                            else:
                                lf = inner
                        loss = lf(p, y)
                        losses.append(loss.view(len(p), -1).mean(1))
                preds.append(self._activation()(p) if act else p)
                if with_input:
                    inps.append(xb[0] if self.dls.n_inp == 1 else xb)
        res = [torch.cat(preds).cpu(), torch.cat(targs).cpu() if targs else None]
        if with_input:
            res.insert(0, torch.cat(inps).cpu() if self.dls.n_inp == 1 else inps)
        if with_loss:
            res.append(torch.cat(losses).cpu())
        return tuple(res)

    def validate(self, dl=None):
        self._do_epoch_validate(dl)
        return self.valid_results

    def tta(self, ds_idx=1, n=4, beta=0.25):
        """Test-time augmentation (Ch. 7): average predictions over n augmented copies of each image,
        then mix with the plain prediction: beta * plain + (1 - beta) * augmented."""
        dl = self.dls[ds_idx]
        aug = torch.stack([self.get_preds(dl=dl.new(train=True))[0] for _ in range(n)]).mean(0)
        plain, targs = self.get_preds(dl=dl)
        return torch.lerp(aug, plain, beta), targs

    def predict(self, item):
        """Predict one item (a file name, a PIL image, a text, ...). Needs DataLoaders made by a DataBlock."""
        dl = self.dls.test_dl([item])
        inp, preds = self.get_preds(dl=dl, with_input=True)[0:2]
        probs = preds[0]
        decodes = getattr(self.loss_func, "decodes", None)
        if decodes is None:
            return probs
        dec = decodes(probs.unsqueeze(0))[0]
        vocab = self.dls.vocab
        if dec.dtype == torch.bool:                            # multi-label
            return [vocab[i] for i in torch.where(dec)[0]], dec, probs
        if dec.ndim == 0:
            return vocab[int(dec)], dec, probs
        return dec, dec, probs                                 # e.g. a segmentation mask

    # ---- saving ----
    def save(self, name, with_opt=False):
        fname = self.path / f"{name}.pth"
        fname.parent.mkdir(parents=True, exist_ok=True)
        state = {"model": self.model.state_dict()}
        if with_opt and self.opt is not None:
            state["opt"] = self.opt.state_dict()
        torch.save(state, fname)
        return fname

    def load(self, name):
        state = torch.load(self.path / f"{name}.pth", map_location=self.dls.device)
        self.model.load_state_dict(state["model"])
        if "opt" in state:
            if self.opt is None:
                self.create_opt()
            self.opt.load_state_dict(state["opt"])
        return self

    def export(self, fname="export.pkl"):
        """Save the model *and* the data pipeline needed for inference (Ch. 2)."""
        fname = self.path / fname
        fname.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.cpu(), "loss_func": self.loss_func,
                    "datablock": getattr(self.dls, "datablock", None), "vocab": self.dls.vocab}, fname)
        self.model.to(self.dls.device)
        print(f"exported to {fname}")
        return fname

    def summary(self):
        """Output shape and number of parameters of every layer, for one batch (Ch. 13)."""
        from .hooks import model_summary
        b = self.dls.one_batch()
        return model_summary(self.model, *b[:self.dls.n_inp])

    def show_results(self, max_n=6, fname="results.png"):
        b = self.dls.valid.one_batch()
        with torch.no_grad():
            self.model.eval()
            p = self.model(*b[:self.dls.n_inp])
        decodes = getattr(self.loss_func, "decodes", lambda x: x)
        self.dls.show_batch(b=b, max_n=max_n, fname=fname, preds=decodes(self._activation()(p.float())))


def load_learner(fname, device=None):
    """Load a Learner saved with `learn.export()`, ready for `predict` (Ch. 2)."""
    state = torch.load(fname, map_location="cpu", weights_only=False)
    device = device or torch.device("cpu")
    dblock = state["datablock"]
    dls = dblock.empty_dataloaders(device=device)
    return Learner(dls, state["model"], loss_func=state["loss_func"])


# ==================================================================================================
# Interpretation
# ==================================================================================================
class ClassificationInterpretation:
    """Which classes get confused, and which items the model is most wrong about (Ch. 2 and 5)."""

    def __init__(self, learn, inputs, preds, targs, losses):
        self.learn, self.inputs, self.preds, self.targs, self.losses = learn, inputs, preds, targs, losses
        self.decoded = preds.argmax(dim=-1)
        self.vocab = learn.dls.vocab

    @classmethod
    def from_learner(cls, learn, ds_idx=1):
        inputs, preds, targs, losses = learn.get_preds(ds_idx=ds_idx, with_input=True, with_loss=True)
        return cls(learn, inputs, preds, targs, losses)

    def confusion_matrix(self):
        n = len(self.vocab)
        cm = torch.zeros(n, n, dtype=torch.long)
        for t, p in zip(self.targs, self.decoded):
            cm[t, p] += 1
        return cm

    def print_confusion_matrix(self):
        cm = self.confusion_matrix()
        w = max(len(str(v)) for v in self.vocab)
        print(" " * (w + 8) + "predicted")
        print(f"{'actual':<{w}}  " + " ".join(f"{str(v)[:6]:>6}" for v in self.vocab))
        for v, row in zip(self.vocab, cm):
            print(f"{str(v):<{w}}  " + " ".join(f"{int(x):>6}" for x in row))

    def plot_confusion_matrix(self, fname="confusion_matrix.png", figsize=(6, 6)):
        plt = try_import_matplotlib()
        if plt is None:
            return
        cm = self.confusion_matrix().numpy()
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(self.vocab)), self.vocab, rotation=90)
        ax.set_yticks(range(len(self.vocab)), self.vocab)
        if len(self.vocab) <= 12:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fname}")

    def most_confused(self, min_val=1):
        cm = self.confusion_matrix()
        res = [(self.vocab[i], self.vocab[j], int(cm[i, j])) for i in range(len(cm)) for j in range(len(cm))
               if i != j and cm[i, j] >= min_val]
        return sorted(res, key=lambda r: -r[2])

    def top_losses(self, k=9):
        return self.losses.topk(min(k, len(self.losses)))

    def plot_top_losses(self, k=9, fname="top_losses.png"):
        from .utils import show_images
        vals, idxs = self.top_losses(k)
        imgs = self.learn.dls.decode_x(self.inputs[idxs])
        titles = [f"{self.vocab[self.decoded[i]]}/{self.vocab[self.targs[i]]}\n{float(self.losses[i]):.2f} / "
                  f"{float(self.preds[i, self.decoded[i]]):.2f}" for i in idxs]
        print("captions: prediction/actual, loss / probability of the prediction")
        show_images(imgs, titles, fname, ncols=min(k, 5))
