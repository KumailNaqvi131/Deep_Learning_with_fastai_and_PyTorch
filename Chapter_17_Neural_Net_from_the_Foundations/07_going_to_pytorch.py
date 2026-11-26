"""
Chapter 17 — A Neural Net from the Foundations
Topic 7: Going to PyTorch (book section of the same name)

PyTorch's version of our LayerFunction is torch.autograd.Function: forward computes the output and
saves what backward will need (ctx.save_for_backward); backward receives the gradient of the
output and returns the gradient of the input. PyTorch records the graph of Functions for us.

nn.Module registers parameters (nn.Parameter) automatically, so an optimizer can find them; its
forward defines the computation. Note PyTorch stores a linear layer's weight as [n_out, n_in].

    class LinearLayer(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_out, n_in) * sqrt(2/n_in))
            self.bias = nn.Parameter(torch.zeros(n_out))
        def forward(self, x): return x @ self.weight.t() + self.bias

Run:
    python 07_going_to_pytorch.py
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class MyRelu(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.clamp_min(0.0)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output * (i > 0).float()


x = torch.randn(10, requires_grad=True)
MyRelu.apply(x).sum().backward()
g_ours = x.grad.clone()
x.grad = None
F.relu(x).sum().backward()
print("MyRelu gradient == F.relu gradient:", torch.equal(g_ours, x.grad))


class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * math.sqrt(2 / n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


lin = LinearLayer(10, 2)
p1, p2 = lin.parameters()
print("LinearLayer(10,2) parameters:", tuple(p1.shape), tuple(p2.shape))


class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, nh), nn.ReLU(), nn.Linear(nh, n_out))
        self.loss = lambda x, y: F.mse_loss(x.squeeze(-1), y)

    def forward(self, x, targ):
        return self.loss(self.layers(x), targ)


torch.manual_seed(0)
x, y = torch.randn(200, 100), torch.randn(200)
model = Model(100, 50, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(5):
    loss = model(x, y)
    opt.zero_grad()
    loss.backward()                  # PyTorch does everything 05_backward_pass.py did by hand
    opt.step()
    print(f"step {step}: loss {loss.item():.4f}")
