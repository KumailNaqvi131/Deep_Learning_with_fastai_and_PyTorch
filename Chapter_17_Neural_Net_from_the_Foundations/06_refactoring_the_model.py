"""
Chapter 17 — A Neural Net from the Foundations
Topic 6: Refactoring the model (book section of the same name)

Each function has a forward and a backward part: wrap both in a class that remembers its inputs
and output. A Model is then a list of layers + a loss: forward = call them in order, backward =
call their .backward() in reverse order. Then a base class, LayerFunction, factors out the common
code (store the arguments and the output; backward calls bwd(out, *args)).

Run:
    python 06_refactoring_the_model.py
"""

import math

import torch

torch.manual_seed(0)
x, y = torch.randn(200, 100), torch.randn(200)
w1, b1 = torch.randn(100, 50) * math.sqrt(2 / 100), torch.zeros(50)
w2, b2 = torch.randn(50, 1) * math.sqrt(2 / 50), torch.zeros(1)


class Relu:
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.0)
        return self.out

    def backward(self):
        self.inp.g = (self.inp > 0).float() * self.out.g


class Lin:
    def __init__(self, w, b):
        self.w, self.b = w, b

    def __call__(self, inp):
        self.inp = inp
        self.out = inp @ self.w + self.b
        return self.out

    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)


class Mse:
    def __call__(self, inp, targ):
        self.inp, self.targ = inp, targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out

    def backward(self):
        x = (self.inp.squeeze() - self.targ).unsqueeze(-1)
        self.inp.g = 2.0 * x / self.targ.shape[0]


class Model:
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1, b1), Relu(), Lin(w2, b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for layer in self.layers:
            x = layer(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()


model = Model(w1, b1, w2, b2)
loss = model(x, y)
model.backward()
g_first = w1.g.clone()
print(f"Model with Relu/Lin/Mse classes: loss {loss:.4f}, w1.g[0,:3] = {w1.g[0, :3].tolist()}")


# ---- A base class --------------------------------------------------------------------------------------------
class LayerFunction:
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):
        raise Exception("not implemented")

    def bwd(self):
        raise Exception("not implemented")

    def backward(self):
        self.bwd(self.out, *self.args)


class Relu2(LayerFunction):
    def forward(self, inp):
        return inp.clamp_min(0.0)

    def bwd(self, out, inp):
        inp.g = (inp > 0).float() * out.g


class Lin2(LayerFunction):
    def __init__(self, w, b):
        self.w, self.b = w, b

    def forward(self, inp):
        return inp @ self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g            # (the book's listing says self.inp here; it's the argument inp)
        self.b.g = out.g.sum(0)


class Mse2(LayerFunction):
    def forward(self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()

    def bwd(self, out, inp, targ):
        inp.g = 2 * (inp.squeeze() - targ).unsqueeze(-1) / targ.shape[0]


model.layers, model.loss = [Lin2(w1, b1), Relu2(), Lin2(w2, b2)], Mse2()
loss = model(x, y)
model.backward()
print(f"Model with LayerFunction subclasses: loss {loss:.4f}, same gradients: {torch.allclose(w1.g, g_first)}")
