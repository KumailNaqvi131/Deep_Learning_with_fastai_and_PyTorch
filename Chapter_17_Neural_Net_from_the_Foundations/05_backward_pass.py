"""
Chapter 17 — A Neural Net from the Foundations
Topic 5: Gradients and the backward pass (book section of the same name)

The chain rule, applied from the loss backwards: each function knows how to turn the gradient of
the loss w.r.t. its OUTPUT into the gradient w.r.t. its INPUTS (and weights). We store the
gradients in a `.g` attribute, like PyTorch's `.grad`:

    def mse_grad(inp, targ):  inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
    def relu_grad(inp, out):  inp.g = (inp>0).float() * out.g
    def lin_grad(inp, out, w, b):
        inp.g = out.g @ w.t()
        w.g = inp.t() @ out.g
        b.g = out.g.sum(0)

Then we check our gradients against PyTorch's autograd.

Run:
    python 05_backward_pass.py
"""

import math

import torch

torch.manual_seed(0)
x, y = torch.randn(200, 100), torch.randn(200)
w1, b1 = torch.randn(100, 50) * math.sqrt(2 / 100), torch.zeros(50)
w2, b2 = torch.randn(50, 1) * math.sqrt(2 / 50), torch.zeros(1)


def relu(x):
    return x.clamp_min(0.0)


def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()


def mse_grad(inp, targ):
    inp.g = 2.0 * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]


def relu_grad(inp, out):
    inp.g = (inp > 0).float() * out.g


def lin_grad(inp, out, w, b):
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(0)


def forward_and_backward(inp, targ):
    # forward pass
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    loss = mse(out, targ)          # (not needed for the backward pass)
    # backward pass, in reverse order
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
    return loss


loss = forward_and_backward(x, y)
print(f"loss {loss:.4f}; our gradients: w1.g {tuple(w1.g.shape)}, b1.g {tuple(b1.g.shape)}, "
      f"w2.g {tuple(w2.g.shape)}, b2.g {tuple(b2.g.shape)}")

# Check with autograd
ws = [t.clone().requires_grad_() for t in (w1, b1, w2, b2)]
xa = x.clone().requires_grad_()
loss2 = mse(relu(xa @ ws[0] + ws[1]) @ ws[2] + ws[3], y)
loss2.backward()
for name, ours, ref in zip(("w1", "b1", "w2", "b2", "x"), (w1.g, b1.g, w2.g, b2.g, x.g), (*[t.grad for t in ws], xa.grad)):
    print(f"   {name}: matches PyTorch autograd: {torch.allclose(ours, ref, atol=1e-5)}")

try:
    from sympy import diff, symbols
    sx = symbols("sx")
    print("\nsympy: d(sx**2)/d(sx) =", diff(sx ** 2, sx))
except ImportError:
    print("\n(sympy, for symbolic derivatives, is not installed: pip install sympy)")
