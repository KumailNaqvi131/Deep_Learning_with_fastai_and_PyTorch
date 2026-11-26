"""
Chapter 13 — Convolutional Neural Networks
Topic 7: Batch normalization (book section of the same name)

Batch normalization (Ioffe & Szegedy, 2015): after a layer, normalize the activations using the
mean and standard deviation of the current batch, then scale and shift them with two learnable
parameters (gamma, beta), so the network can still choose any mean and std. Training becomes
much smoother, higher learning rates work, and the randomness of batch statistics even acts as a
regularizer.

    def conv(ni, nf, ks=3, act=True):
        layers = [nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)]
        layers.append(nn.BatchNorm2d(nf))
        if act: layers.append(nn.ReLU())
        return nn.Sequential(*layers)

Run:
    python 07_batch_normalization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import ActivationStats, Learner, accuracy, outputs_dir, set_seed  # noqa: E402
from mnist_full import get_dls, simple_cnn  # noqa: E402

OUT = outputs_dir(__file__)

# What BatchNorm computes, by hand (training mode)
x = torch.randn(64, 8, 4, 4) * 3 + 5
bn = torch.nn.BatchNorm2d(8)
mean, var = x.mean(dim=(0, 2, 3), keepdim=True), x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
by_hand = (x - mean) / torch.sqrt(var + bn.eps) * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
print("BatchNorm2d == (x - batch mean) / batch std * gamma + beta:", torch.allclose(bn(x), by_hand, atol=1e-5))

set_seed(42)
dls = get_dls(512)
print("\nwith batchnorm, fit_one_cycle(1, 0.06):")
learn = Learner(dls, simple_cnn(bn=True), loss_func=F.cross_entropy, metrics=accuracy, cbs=[ActivationStats(with_hist=True)])
learn.fit_one_cycle(1, 0.06)
learn.activation_stats.color_dim(-4, fname=OUT / "color_dim_batchnorm.png")

print("\nmore epochs and a higher learning rate, fit_one_cycle(5, 0.1):")
learn = Learner(dls, simple_cnn(bn=True), loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(5, 0.1)
