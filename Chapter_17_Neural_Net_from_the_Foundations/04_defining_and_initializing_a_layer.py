"""
Chapter 17 — A Neural Net from the Foundations
Topic 4: The forward pass: defining and initializing a layer (book section of the same name)

    def lin(x, w, b): return x @ w + b

With weights from torch.randn (std 1), activations grow at every layer: after 50 layers they
are nan (too big for floats). Scaled too small, they vanish to 0. The right scale:
  * Xavier/Glorot init: std = 1/sqrt(n_in) keeps the std at 1 through linear layers;
  * with ReLU (which zeroes half the activations), Kaiming/He init: std = sqrt(2/n_in).

Run:
    python 04_defining_and_initializing_a_layer.py
"""

import math

import torch

torch.manual_seed(0)


def lin(x, w, b):
    return x @ w + b


def relu(x):
    return x.clamp_min(0.0)


x = torch.randn(200, 100)
y = torch.randn(200)

for name, scale in [("std 1 (torch.randn)", 1.0), ("std 0.01", 0.01), ("Xavier 1/sqrt(100)", 1 / math.sqrt(100))]:
    a = torch.randn(200, 100)
    for i in range(50):
        a = a @ (torch.randn(100, 100) * scale)
    print(f"50 linear layers, weights {name:22s}: activations mean {a.mean():.3g}, std {a.std():.3g}")

print()
for name, scale in [("Xavier 1/sqrt(100)", 1 / math.sqrt(100)), ("Kaiming sqrt(2/100)", math.sqrt(2 / 100))]:
    a = torch.randn(200, 100)
    for i in range(50):
        a = relu(a @ (torch.randn(100, 100) * scale))
    print(f"50 linear+ReLU layers, weights {name:20s}: activations mean {a.mean():.3g}, std {a.std():.3g}")

# The two-layer model of the chapter, with Kaiming init
w1 = torch.randn(100, 50) * math.sqrt(2 / 100)
b1 = torch.zeros(50)
w2 = torch.randn(50, 1) * math.sqrt(2 / 50)
b2 = torch.zeros(1)
l1 = lin(x, w1, b1)
l2 = relu(l1)
print(f"\nfirst layer: {tuple(l1.shape)}, std {l1.std():.3f}; after ReLU mean {l2.mean():.3f}, std {l2.std():.3f}")


def model(x):
    l1 = lin(x, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3


out = model(x)
print("model output:", tuple(out.shape))


def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()


print(f"mse loss: {mse(out, y):.4f}  (squeeze(-1): the output is [200,1], the target [200])")
