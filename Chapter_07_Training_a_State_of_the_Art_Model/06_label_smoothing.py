"""
Chapter 7 — Training a State-of-the-Art Model
Topic 6: Label smoothing (book section "Label Smoothing")

With one-hot targets, cross-entropy pushes the right class's activation ever higher (softmax can
only reach 1 at infinity): the model becomes over-confident and mislabelled data hurts a lot.
Label smoothing replaces the 1 by 1-eps and the 0s by eps/N (N classes):

    loss = (1-eps) * cross_entropy(correct class) + eps/N * sum over all classes of -log(p)

    learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy)

Run:
    python 06_label_smoothing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import LabelSmoothingCrossEntropy, Learner, accuracy, set_seed, xresnet50  # noqa: E402
from imagenette_data import get_dls  # noqa: E402

# What the targets become with eps=0.1 and 10 classes
eps, n = 0.1, 10
target = torch.tensor(3)
smoothed = torch.full((n,), eps / n)
smoothed[target] += 1 - eps
print("one-hot target:  ", F.one_hot(target, n).tolist())
print("smoothed target: ", [round(v, 3) for v in smoothed.tolist()])

acts = torch.randn(4, n)
targs = torch.tensor([3, 1, 0, 7])
print("LabelSmoothingCrossEntropy(eps=0.1):", LabelSmoothingCrossEntropy(0.1)(acts, targs).item(),
      "| plain cross-entropy:", F.cross_entropy(acts, targs).item())
print("same as cross-entropy against the smoothed targets:",
      -(F.log_softmax(acts, -1) * (F.one_hot(targs, n) * (1 - eps) + eps / n)).sum(-1).mean().item())

set_seed(42)
dls = get_dls(64, 128)
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
preds, _ = learn.get_preds()
print(f"average confidence of the predictions: {preds.max(1)[0].mean():.3f} (less over-confident than usual)")
