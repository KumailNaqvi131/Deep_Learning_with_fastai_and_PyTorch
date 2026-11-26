"""Shared by the Chapter 16 scripts: Imagenette DataLoaders, a learner factory, and the generic
Optimizer the chapter builds (Topic 2) together with its step callbacks (Topics 2-6).

    def get_learner(**kwargs):
        return cnn_learner(dls, resnet34, pretrained=False, metrics=accuracy, **kwargs).to_fp16()
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import (CategoryBlock, DataBlock, GrandparentSplitter, ImageBlock, Normalize, Resize, URLs,  # noqa: E402
                    accuracy, aug_transforms, cnn_learner, get_image_files, imagenet_stats, parent_label,
                    resnet34, untar_data)

path = untar_data(URLs.IMAGENETTE_160)
dls = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
                splitter=GrandparentSplitter(valid_name="val"), get_y=parent_label, item_tfms=Resize(160),
                batch_tfms=[*aug_transforms(min_scale=0.5, size=128), Normalize.from_stats(*imagenet_stats)]
                ).dataloaders(path, bs=64)


def get_learner(**kwargs):
    return cnn_learner(dls, resnet34, pretrained=False, metrics=accuracy, **kwargs).to_fp16()


# ==================================================================================================
# A generic optimizer (book section "A Generic Optimizer")
# ==================================================================================================
class Optimizer(torch.optim.Optimizer):
    """Each optimizer step runs a list of callbacks on every parameter. A callback gets the parameter,
    the hyper-parameters (lr, mom, wd, ...) and the parameter's state (e.g. its moving average of
    gradients), and may return new state to store. SGD, momentum, RMSProp, Adam, weight decay: all
    are just different callbacks. (Subclassing torch.optim.Optimizer lets it work with PyTorch's
    mixed-precision GradScaler.)"""

    def __init__(self, params, cbs, **defaults):
        super().__init__(params, defaults)
        self.cbs = cbs

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            hyper = {k: v for k, v in group.items() if k != "params"}
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                for cb in self.cbs:
                    res = cb(p, **{**state, **hyper})
                    if res:
                        state.update(res)


def sgd_cb(p, lr, **kwargs):
    p.data.add_(p.grad.data, alpha=-lr)


def weight_decay(p, lr, wd=0.0, **kwargs):
    """Decoupled weight decay: shrink the weights directly (Loshchilov & Hutter)."""
    p.data.mul_(1 - lr * wd)


def l2_reg(p, lr, wd=0.0, **kwargs):
    """L2 regularization: add wd * weights to the gradient."""
    p.grad.data.add_(p.data, alpha=wd)


def average_grad(p, mom, grad_avg=None, dampening=False, **kwargs):
    if grad_avg is None:
        grad_avg = torch.zeros_like(p.grad.data)
    damp = 1 - mom if dampening else 1.0
    return {"grad_avg": grad_avg.mul_(mom).add_(p.grad.data, alpha=damp)}


def momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(grad_avg, alpha=-lr)


def average_sqr_grad(p, sqr_mom, sqr_avg=None, **kwargs):
    if sqr_avg is None:
        sqr_avg = torch.zeros_like(p.grad.data)
    return {"sqr_avg": sqr_avg.mul_(sqr_mom).addcmul_(p.grad.data, p.grad.data, value=1 - sqr_mom)}


def rms_prop_step(p, lr, sqr_avg, eps, grad_avg=None, **kwargs):
    denom = sqr_avg.sqrt().add_(eps)
    p.data.addcdiv_(grad_avg if grad_avg is not None else p.grad, denom, value=-lr)


def step_stat(p, step=0, **kwargs):
    return {"step": step + 1}


def adam_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, **kwargs):
    debias1 = 1 - mom ** step                 # the averages start at 0: correct for that early on
    debias2 = 1 - sqr_mom ** step
    p.data.addcdiv_(grad_avg, (sqr_avg / debias2).sqrt() + eps, value=-lr / debias1)


def average_grad_dampened(p, mom, grad_avg=None, **kwargs):
    return average_grad(p, mom, grad_avg, dampening=True)
