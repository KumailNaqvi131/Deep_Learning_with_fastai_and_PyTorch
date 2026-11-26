"""
Chapter 7 — Training a State-of-the-Art Model
Topic 5: Mixup (book section "Mixup")

Mixup (Zhang et al., 2017) is a data augmentation for any kind of data. For each image:
  1. pick another random image from the batch,
  2. pick a weight lam from a Beta(0.4, 0.4) distribution (mostly near 0 or 1),
  3. mix the two images:  new_image = lam * image1 + (1-lam) * image2,
  4. mix the targets the same way: the loss is lam * loss(target1) + (1-lam) * loss(target2).
The model can never be 100% sure, which reduces over-confidence. It makes training harder: it
pays off when training for many epochs (the book: 80+). In fastai: cbs=MixUp().

Here MixUp is written as a callback: it changes the batch before the model sees it
(before_batch) and wraps the loss function.

Run:
    python 05_mixup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.distributions.beta import Beta  # noqa: E402

from common import Callback, CrossEntropyLossFlat, Learner, accuracy, outputs_dir, set_seed, show_images, xresnet50  # noqa: E402
from imagenette_data import get_dls  # noqa: E402


class MixUp(Callback):
    def __init__(self, alpha=0.4):
        self.distrib = Beta(torch.tensor(alpha), torch.tensor(alpha))

    def before_fit(self):
        self.old_loss_func = self.learn.loss_func
        self.learn.loss_func = self.mixup_loss

    def after_fit(self):
        self.learn.loss_func = self.old_loss_func

    def before_batch(self):
        if not self.learn.training:
            return
        x, y = self.learn.batch
        lam = self.distrib.sample((len(y),)).to(x.device)
        self.lam = torch.stack([lam, 1 - lam], 1).max(1)[0]          # the "main" image keeps weight >= 0.5
        shuffle = torch.randperm(len(y), device=x.device)
        self.y_other = y[shuffle]
        x = torch.lerp(x[shuffle], x, self.lam.view(-1, 1, 1, 1))   # lam*x + (1-lam)*x[shuffle]
        self.learn.batch = (x, y)

    def mixup_loss(self, pred, y):
        if not self.learn.training:
            return self.old_loss_func(pred, y)
        loss1 = F.cross_entropy(pred, y, reduction="none")
        loss2 = F.cross_entropy(pred, self.y_other, reduction="none")
        return torch.lerp(loss2, loss1, self.lam).mean()


set_seed(42)
dls = get_dls(64, 128)

# What mixed images look like
cb = MixUp()
cb.learn = type("L", (), {"training": True, "batch": dls.one_batch()})()
cb.before_batch()
x_mixed = cb.learn.batch[0][:6]
show_images(dls.decode_x(x_mixed).clamp(0, 1).cpu(),
            [f"{dls.vocab[int(a)][:9]} {l:.2f}\n+ {dls.vocab[int(b)][:9]}" for a, b, l in
             zip(cb.learn.batch[1][:6], cb.y_other[:6], cb.lam[:6])],
            outputs_dir(__file__) / "mixup_examples.png", ncols=3)

learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=[MixUp()])
learn.fit_one_cycle(5, 3e-3)
print("With only 5 epochs, mixup usually doesn't beat the baseline yet: it shines on long training runs.")
