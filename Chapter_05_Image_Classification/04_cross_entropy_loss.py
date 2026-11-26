"""
Chapter 5 — Image Classification
Topic 4: Cross-entropy loss: viewing activations, softmax, log likelihood, taking the log
(book section "Cross-Entropy Loss")

With 37 classes the model outputs 37 numbers (activations) per image. Cross-entropy loss:
  1. softmax: exp(x) / exp(x).sum()  -> positive numbers that add up to 1, and it exaggerates the
     largest activation: the model "picks" one class;
  2. pick the probability of the *correct* class for each image (that's what F.nll_loss does,
     with a minus sign);
  3. take the log: going from 0.99 to 0.999 is as big an improvement as 0.9 to 0.99.
log_softmax + nll_loss = cross-entropy = nn.CrossEntropyLoss = F.cross_entropy.

Run:
    python 04_cross_entropy_loss.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import cnn_learner, error_rate, resnet34, set_seed  # noqa: E402
from pets_data import get_dls  # noqa: E402

set_seed(42)

# ---- Viewing activations and labels on a real model -------------------------------------------------
dls = get_dls()
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
x, y = dls.one_batch()
print("labels of one batch:", y[:10].tolist(), "...")
preds, _ = learn.get_preds(dl=[(x, y)])
print("preds[0] (after softmax):", [round(v, 4) for v in preds[0][:6].tolist()], "...", len(preds[0]), "values")
print("sum of preds[0]:", round(preds[0].sum().item(), 4))

# ---- Softmax ---------------------------------------------------------------------------------------
torch.manual_seed(42)
acts = torch.randn((6, 2)) * 2              # 6 images, 2 classes (say 3s and 7s)
print("\nacts:\n", acts)
print("sigmoid of acts (doesn't add up to 1 across classes):\n", acts.sigmoid())
print("sigmoid(acts[:,0]-acts[:,1]) = softmax column 0:", (acts[:, 0] - acts[:, 1]).sigmoid().tolist())


def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)


sm_acts = softmax(acts)
print("softmax(acts):\n", sm_acts, "\nsame as torch.softmax:", torch.allclose(sm_acts, torch.softmax(acts, dim=1)))

# ---- Log likelihood --------------------------------------------------------------------------------
targ = torch.tensor([0, 1, 0, 1, 1, 0])
idx = range(6)
print("\nprobability of the correct class for each image:", sm_acts[idx, targ].tolist())
print("-F.nll_loss(reduction='none') gives the same:", (-F.nll_loss(sm_acts, targ, reduction="none")).tolist())

# ---- Taking the log ----------------------------------------------------------------------------------
print("\nlog of 0.5, 0.9, 0.99, 0.999:", [round(v, 4) for v in torch.log(torch.tensor([0.5, 0.9, 0.99, 0.999])).tolist()])
loss_func = nn.CrossEntropyLoss()
print("nn.CrossEntropyLoss()(acts, targ) =", loss_func(acts, targ).item())
print("F.cross_entropy(acts, targ)       =", F.cross_entropy(acts, targ).item())
print("-log(softmax)[correct].mean()     =", (-torch.log(sm_acts[idx, targ])).mean().item())
print("nll_loss(log_softmax(acts))       =", F.nll_loss(F.log_softmax(acts, dim=1), targ).item())
print("per item (reduction='none'):", nn.CrossEntropyLoss(reduction="none")(acts, targ).tolist())
