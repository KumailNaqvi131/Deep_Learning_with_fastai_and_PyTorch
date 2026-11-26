"""
Chapter 19 — A fastai Learner from Scratch
Topic 2: Module and Parameter; a simple CNN (book sections "Module and Parameter", "Simple CNN")

What does nn.Module do? It keeps track of parameters and sub-modules (so .parameters() can find
them all, recursively), passes the training/eval flag down, and lets hooks run after forward.
Our own version:
  * Parameter: a tensor that requires grad (a tensor subclass, so a module can recognise it)
  * Module.__setattr__: when you assign a Parameter or a Module to an attribute, register it
  * ConvLayer (F.conv2d + ReLU, Kaiming init), Linear, Sequential, AdaptivePool
  * a `hook` attribute called with the output after forward: we use it to print activation stats

Run:
    python 02_module_and_parameter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402


class Parameter(torch.Tensor):
    def __new__(self, x):
        return torch.Tensor._make_subclass(Parameter, x, True)

    def __init__(self, *args, **kwargs):
        self.requires_grad_()


class Module:
    def __init__(self):
        self.hook, self.params, self.children, self._training = None, [], [], False

    def register_parameters(self, *ps):
        self.params += ps

    def register_modules(self, *ms):
        self.children += ms

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, v):
        self._training = v
        for m in self.children:
            m.training = v

    def parameters(self):
        return self.params + sum([m.parameters() for m in self.children], [])

    def __setattr__(self, k, v):
        super().__setattr__(k, v)
        if isinstance(v, Parameter):
            self.register_parameters(v)
        if isinstance(v, Module):
            self.register_modules(v)

    def __call__(self, *args, **kwargs):
        res = self.forward(*args, **kwargs)
        if self.hook is not None:
            self.hook(res, args)
        return res

    def to(self, device):
        for p in self.parameters():
            p.data = p.data.to(device)
        return self


class ConvLayer(Module):
    def __init__(self, ni, nf, stride=1, bias=True, act=True):
        super().__init__()
        self.w = Parameter(torch.zeros(nf, ni, 3, 3))
        self.b = Parameter(torch.zeros(nf)) if bias else None
        self.act, self.stride = act, stride
        init = nn.init.kaiming_normal_ if act else nn.init.xavier_normal_
        init(self.w)

    def forward(self, x):
        x = F.conv2d(x, self.w, self.b, stride=self.stride, padding=1)
        if self.act:
            x = F.relu(x)
        return x


class Linear(Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.w = Parameter(torch.zeros(nf, ni))
        self.b = Parameter(torch.zeros(nf))
        nn.init.xavier_normal_(self.w)

    def forward(self, x):
        return x @ self.w.t() + self.b


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        self.register_modules(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AdaptivePool(Module):
    def forward(self, x):
        return x.mean((2, 3))


def simple_cnn():
    return Sequential(
        ConvLayer(3, 16, stride=2),    # 32
        ConvLayer(16, 32, stride=2),   # 16
        ConvLayer(32, 64, stride=2),   # 8
        ConvLayer(64, 128, stride=2),  # 4
        AdaptivePool(),
        Linear(128, 10),
    )


if __name__ == "__main__":
    class T(Module):
        def __init__(self):
            super().__init__()
            self.c, self.l = ConvLayer(3, 4), Linear(4, 2)

    t = T()
    print("T has", len(t.parameters()), "parameters:", [tuple(p.shape) for p in t.parameters()])
    t.training = True
    print("setting t.training = True propagates to the children:", t.l.training)

    import importlib
    data = importlib.import_module("01_data_and_dataset")
    xb, yb = next(iter(data.train_dl))
    xb = data.tfm_x(xb)
    m = simple_cnn()
    print("simple_cnn has", len(m.parameters()), "parameter tensors")

    def print_stats(outp, inp):
        print(f"   layer output mean {outp.mean().item():.3f}, std {outp.std().item():.3f}")

    for i in range(4):
        m.layers[i].hook = print_stats
    print("forward pass of one batch, with a hook printing the stats of each conv layer:")
    r = m(xb)
    print("output:", tuple(r.shape))
